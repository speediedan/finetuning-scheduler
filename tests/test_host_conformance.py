# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the direct-API schedule keys and the Lightning training host (#51)."""
import pytest
import torch
from lightning.pytorch import Callback, seed_everything, Trainer
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from finetuning_scheduler import FinetuningScheduler
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.host import LRStepConfig, TrainingHost
from finetuning_scheduler.core.schedule import parse_phase_extensions, refuse_direct_api_keys
from finetuning_scheduler.lightning.host import LightningHost
from tests.test_finetuning_scheduler_callback import FinetuningSchedulerBoringModel


def test_parse_phase_extensions():
    schedule = {0: {"params": ["a"]}, 1: {"params": ["b"], "max_transition_step": 50, "phase_config": {"k": 1},
                                          "pg_overrides": {"weight_decay": 0.0}}}
    parse_phase_extensions(schedule)
    assert schedule[0]["max_transition_step"] == -1 and schedule[1]["max_transition_step"] == 50


@pytest.mark.parametrize(
    "phase, match",
    [
        ({"max_transition_step": -2}, "must be an integer"),
        ({"max_transition_step": True}, "must be an integer"),
        ({"phase_config": ["not", "a", "mapping"]}, "must be a mapping"),
        ({"pg_overrides": 0.1}, "must be a mapping of optimizer options"),
        ({"pg_overrides": {"lr": 0.1, "params": []}}, r"may not set \['lr', 'params'\]"),
    ],
)
def test_parse_phase_extensions_refuses(phase, match):
    with pytest.raises(FTSConfigurationError, match=match):
        parse_phase_extensions({0: {"params": ["a"], **phase}})


def test_refuse_direct_api_keys():
    """By default, the keys the Lightning callback does not interpret yet (``max_transition_step``, ``promote``);
    ``phase_config`` and ``pg_overrides`` are accepted under Lightning too."""
    from finetuning_scheduler.core.schedule import DIRECT_API_PHASE_KEYS

    refuse_direct_api_keys({0: {"params": ["a"], "max_transition_epoch": 2}})
    mixed = {0: {"params": ["a"]}, 1: {"max_transition_step": 3, "phase_config": {}}, 2: {"pg_overrides": {}},
             3: {"promote": {"role:instrument": "latent:x"}}}
    with pytest.raises(FTSConfigurationError, match=r"\(phase 1: max_transition_step; phase 3: promote\)"):
        refuse_direct_api_keys(mixed)
    with pytest.raises(FTSConfigurationError, match=r"phase 1: max_transition_step, phase_config; phase 2: pg_over"):
        refuse_direct_api_keys(mixed, keys=DIRECT_API_PHASE_KEYS)


def test_lightning_refuses_direct_api_keys(tmpdir):
    """The Lightning callback refuses, rather than silently ignores, the PyTorch API's phase keys."""
    seed_everything(42)
    schedule = {0: {"params": ["model.3.weight", "model.3.bias"]},
                1: {"params": ["model.2.weight", "model.2.bias"], "max_transition_step": 10}}
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[FinetuningScheduler(ft_schedule=schedule)], devices=1,
                      max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    with pytest.raises(MisconfigurationException, match="only the PyTorch API"):
        trainer.fit(FinetuningSchedulerBoringModel())


class _HostProbe(Callback):
    def on_train_start(self, trainer, pl_module):
        host = LightningHost(trainer, pl_module)
        opt = trainer.optimizers[0]
        replacement = torch.optim.SGD([p for p in pl_module.parameters() if p.requires_grad], lr=0.5)
        host.set_optimizer(0, replacement)
        sched = torch.optim.lr_scheduler.StepLR(replacement, step_size=1)
        host.set_lr_scheduler(0, sched, LRStepConfig(interval="step", frequency=2))
        self.observed = {
            "conforms": isinstance(host, TrainingHost),
            "module": host.module is pl_module,
            "optimizers": host.get_optimizers() == [replacement] and opt is not replacement,
            "schedulers": host.get_lr_schedulers() == [(sched, LRStepConfig(interval="step", frequency=2))],
            "counters": (host.current_epoch, host.global_step, host.max_epochs),
            "dist": (host.is_global_zero, host.world_size, host.reduce_max(3), host.reduce_any(True),
                     host.broadcast_object("x"), host.all_gather_object(1)),
            "log_dir": host.log_dir == trainer.log_dir,
            "rng_devices": host.rng_devices(),
        }
        host.set_max_epochs(1)
        host.barrier("probe")
        host.on_compile_reset()
        trainer.should_stop = True


def test_lightning_host_conformance(tmpdir):
    """``LightningHost`` satisfies the ``TrainingHost`` protocol against a real ``Trainer``."""
    seed_everything(42)
    probe = _HostProbe()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[FinetuningScheduler(), probe], accelerator="cpu",
                      devices=1, max_epochs=2, limit_train_batches=1, limit_val_batches=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(FinetuningSchedulerBoringModel())
    obs = probe.observed
    assert obs["conforms"] and obs["module"] and obs["optimizers"] and obs["schedulers"] and obs["log_dir"]
    assert obs["counters"] == (0, 0, 2)
    assert obs["dist"] == (True, 1, 3, True, "x", [1])
    assert obs["rng_devices"] == []
    assert trainer.fit_loop.max_epochs == 1


def test_lightning_extra_notice_once(monkeypatch):
    """The first ``FinetuningScheduler`` in a process announces the move of Lightning behind an extra, once."""
    import warnings

    from finetuning_scheduler.lightning import fts as lightning_fts

    monkeypatch.setattr(lightning_fts, "_lightning_extra_notice_issued", False)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        FinetuningScheduler()
        FinetuningScheduler()
    notices = [w for w in record if issubclass(w.category, FutureWarning) and "optional dependency" in str(w.message)]
    assert len(notices) == 1 and "finetuning-scheduler[lightning]" in str(notices[0].message)


def test_hosts_implement_the_counter_members():
    from finetuning_scheduler.core.host import TorchHost, TrainingHost
    from finetuning_scheduler.lightning.host import LightningHost

    for member in ("advance_epoch", "advance_step", "load_counters"):
        assert hasattr(TrainingHost, member)
        assert callable(getattr(TorchHost, member)) and callable(getattr(LightningHost, member))
