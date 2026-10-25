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
"""Lightning callback, interface, bridge and strategy adapter paths the rest of the suite does not reach (#49)."""
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from finetuning_scheduler import FinetuningScheduler
from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.views import CanonicalView
from finetuning_scheduler.lightning.tuning_scheduler_bridge import FTSCheckpointPolicy, LightningPhaseExecutor
from finetuning_scheduler.lightning.fts import _optimizer_class_name
from finetuning_scheduler.lightning.interface import LightningInterface
from finetuning_scheduler.strategies.plan import _reshards_after_forward
from finetuning_scheduler.strategy_adapters import (FSDPStrategyAdapter, ModelParallelStrategyAdapter,
                                                    StrategyAdapter)
from tests.helpers.legacy_naming_adapter import LogicalNamesBoringModel
from tests.test_finetuning_scheduler_callback import ScriptedLossModel

THREE_PHASES = {0: {"params": ["model.2.weight", "model.2.bias"], "max_transition_epoch": 1},
                1: {"params": ["model.1.weight", "model.1.bias"], "lr": 0.05, "max_transition_epoch": 2},
                2: {"params": ["model.0.weight", "model.0.bias"], "lr": 0.05, "max_transition_epoch": 4}}


class LegacyStepFTS(FinetuningScheduler):
    """The callback with its own transitions rather than its externally driven tuning scheduler's
    (``_use_tuning_scheduler``)."""

    _use_tuning_scheduler = False


def test_legacy_step_without_restoring_best(tmp_path):
    """The callback's own transitions with ``restore_best=False`` configure each phase without a restore."""
    seed_everything(42)
    fts = LegacyStepFTS(ft_schedule=THREE_PHASES, epoch_transitions_only=True, restore_best=False)
    model = ScriptedLossModel([1.0] * 4)
    trainer = Trainer(default_root_dir=tmp_path, callbacks=[fts], max_epochs=4, accelerator="cpu", devices=1,
                      limit_train_batches=2, limit_val_batches=1, enable_progress_bar=False,
                      enable_model_summary=False, logger=False)
    trainer.fit(model)
    assert fts._tuning_scheduler is None and fts.curr_depth == 2
    assert all(p.requires_grad for p in model.parameters())
    assert len(trainer.optimizers[0].param_groups) == 3


def test_phase_hooks_refused_without_the_tuning_scheduler():
    with pytest.raises(MisconfigurationException, match="need the callback's tuning scheduler"):
        LegacyStepFTS().add_phase_hook(lambda event: None)


@pytest.mark.parametrize("new_optimizer, expected", [
    ({"optimizer_init": {"class_path": "torch.optim.Adam"}}, "Adam"),
    ({"optimizer_init": {"class_path": "SGD"}}, "SGD"),
    ({"optimizer_init": None}, None),
    ({}, None),
])
def test_optimizer_class_name(new_optimizer, expected):
    assert _optimizer_class_name(new_optimizer) == expected


def test_grad_mismatch_feedback_delegates_to_core():
    summary = {"optim_params": ["a.weight"], "phase_0_params": ["b.weight"]}
    msg = FinetuningScheduler._grad_mismatch_feedback("The optimizer has", summary)
    assert msg.startswith("The optimizer has a differing set of trainable parameters")
    assert "'a.weight'" in msg and "'b.weight'" in msg


class _FakeFTS:
    """Records the phase execution calls the bridge makes on the callback."""

    def __init__(self, curr_depth=1, best_depth=0, checkpoint_callback=None):
        self.curr_depth = curr_depth
        self.calls = []
        self._fts_state = SimpleNamespace(_best_ckpt_depth=best_depth)
        self.trainer = SimpleNamespace(checkpoint_callback=checkpoint_callback)

    def thaw_to_depth(self, *args):
        self.calls.append(("thaw_to_depth", args))

    def restore_best_ckpt(self):
        self.calls.append(("restore_best_ckpt", ()))


def test_phase_executor_resumes_at_another_depth():
    """The callback's own depth resumes with ``thaw_to_depth()``; any other with ``thaw_to_depth(depth)``."""
    fts = _FakeFTS(curr_depth=1)
    executor = LightningPhaseExecutor(fts)
    executor.resume_at(1)
    executor.resume_at(2)
    assert fts.calls == [("thaw_to_depth", ()), ("thaw_to_depth", (2,))]


def test_checkpoint_policy_reports_the_lightning_checkpoint_callback():
    ckpt_cb = SimpleNamespace(monitor="val_loss", best_model_path="/ckpts/best.ckpt",
                              best_model_score=torch.tensor(0.5))
    fts = _FakeFTS(best_depth=1, checkpoint_callback=ckpt_cb)
    policy = FTSCheckpointPolicy(fts)
    assert policy.observe({"val_loss": 0.1}, 0) is False
    assert policy.best_path == "/ckpts/best.ckpt" and policy.best_depth == 1
    assert policy.best_score is not None and policy.best_score.item() == 0.5
    policy.pin_boundary(1, "/ckpts/best.ckpt")
    assert policy.boundary_path(1) is None
    assert policy.state_dict() == {}
    policy.load_state_dict({"anything": 1})
    assert policy.config() == {"monitor": "val_loss"}
    assert policy.restore() == 1 and fts.calls == [("restore_best_ckpt", ())]
    with pytest.raises(NotImplementedError, match="writes the checkpoints"):
        policy.path_for("best")
    # without a checkpoint callback (or before it saves) there is no best checkpoint
    empty = FTSCheckpointPolicy(_FakeFTS(checkpoint_callback=SimpleNamespace(best_model_path="")))
    assert empty.best_path is None and empty.best_score is None and empty.config() == {"monitor": None}


def _interface(world_size, pl_module=None):
    trainer = SimpleNamespace(world_size=world_size, strategy=SimpleNamespace(root_device=torch.device("cpu")))
    return LightningInterface(trainer, pl_module if pl_module is not None else nn.Linear(1, 1))


def test_interface_collectives_with_several_ranks(monkeypatch):
    """With more than one rank the interface reduces and gathers through ``torch.distributed`` (faked here: a CPU
    test has no multi-rank process group)."""
    calls = []

    def fake_all_reduce(t, op):
        calls.append(("all_reduce", op))
        t.fill_(7)

    def fake_all_gather_object(gathered, obj):
        calls.append(("all_gather_object", len(gathered)))
        gathered[:] = [obj, f"{obj}-peer"]

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    interface = _interface(world_size=2)
    assert interface.reduce_max(3) == 7 and interface.reduce_any(False) is True
    assert interface.all_gather_object("r0") == ["r0", "r0-peer"]
    assert calls == [("all_reduce", torch.distributed.ReduceOp.MAX)] * 2 + [("all_gather_object", 2)]
    single = _interface(world_size=1)
    assert single.reduce_max(3) == 3 and single.all_gather_object("r0") == ["r0"]


@pytest.mark.parametrize("compiler, resets", [("dynamo", 1), ("other", 0), (None, 0)])
def test_interface_compile_reset_follows_the_modules_compiler(monkeypatch, compiler, resets):
    calls = []
    monkeypatch.setattr(torch._dynamo, "reset", lambda: calls.append("reset"))
    module = nn.Linear(1, 1)
    if compiler is not None:
        module._compiler_ctx = {"compiler": compiler}
    _interface(world_size=1, pl_module=module).on_compile_reset()
    assert len(calls) == resets


class _FakeUnit:
    """An FSDP2 unit whose state is faked: ``_reshards_after_forward`` reads FSDP2 internals, not a public API."""

    def __init__(self, state):
        self._state = state

    def _get_fsdp_state(self):
        if isinstance(self._state, Exception):
            raise self._state
        return self._state


@pytest.mark.parametrize("state, expected", [
    # an FSDP2 state with one parameter group (`_fsdp_param_group`) rather than a list of them
    (SimpleNamespace(_fsdp_param_group=SimpleNamespace(post_forward_mesh_info=object())), True),
    (SimpleNamespace(_fsdp_param_group=SimpleNamespace(post_forward_mesh_info=None)), False),
    (SimpleNamespace(_fsdp_param_group=None), False),
    # state that cannot be read is assumed to reshard
    (AttributeError("no state"), True),
    (AssertionError("not initialized"), True),
    (SimpleNamespace(), True),
])
def test_reshards_after_forward_reads_single_group_and_unreadable_state(state, expected):
    assert _reshards_after_forward(_FakeUnit(state), is_outermost=False) is expected


def test_compiled_physical_for_names_that_are_not_a_whole_phase():
    """A name list matching no phase's ``params`` maps through the compiled schedule's logical names."""
    module = LogicalNamesBoringModel()
    schedule = {0: {"params": ["model.4.weight", "model.4.bias"]}, 1: {"params": ["model.0.weight", "model.0.bias"]}}
    adapter = StrategyAdapter()
    adapter.fts_handle = SimpleNamespace(pl_module=module, trainer=None, ft_schedule=schedule)
    adapter.compiled_schedule = compile_schedule(schedule, CanonicalView(module))
    assert adapter.fts_optim_transform(["model.4.weight", "model.4.bias"]) == ["model.4.weight", "model.4.bias"]
    assert adapter.fts_optim_transform(["model.0.bias"]) == ["model.0.bias"]


def test_ddp_reducer_not_rebuilt_when_trainable_parameters_are_unchanged():
    """The reducer is rebuilt only when the set of trainable parameters changed since it was built (a
    ``DistributedDataParallel`` instance is faked: building one needs a process group)."""
    module = nn.Linear(2, 2)
    ddp = DistributedDataParallel.__new__(DistributedDataParallel)
    object.__setattr__(ddp, "module", module)
    strategy = SimpleNamespace(model=ddp)
    adapter = StrategyAdapter()
    adapter.fts_handle = SimpleNamespace(pl_module=SimpleNamespace(_trainer=SimpleNamespace(strategy=strategy)))
    adapter._ddp_reducer_params = frozenset(id(p) for p in module.parameters())
    adapter.on_after_phase_transition()
    assert strategy.model is ddp


def test_fsdp_restore_optimizer_patch_teardown_is_idempotent():
    class PatchedStrategy:
        pass

    adapter = FSDPStrategyAdapter()
    adapter.on_fts_teardown()  # nothing patched yet
    adapter._disable_lightning_optimizer_restore(PatchedStrategy)
    assert PatchedStrategy.lightning_restore_optimizer is False
    adapter.on_fts_teardown()
    assert "lightning_restore_optimizer" not in vars(PatchedStrategy)
    adapter.on_fts_teardown()  # a second teardown leaves the strategy class alone
    assert "lightning_restore_optimizer" not in vars(PatchedStrategy)


def test_model_parallel_validation_error_refuses():
    """An ``error`` from the distributed adapter's validation (here the unsharded adapter's: a scheduled name the
    module does not have) refuses the configuration."""
    adapter = ModelParallelStrategyAdapter()
    adapter.fts_handle = SimpleNamespace(pl_module=nn.Sequential(nn.Linear(2, 2)),
                                         ft_schedule={0: {"params": ["0.weight", "missing.weight"]}})
    with pytest.raises(MisconfigurationException, match=r"does not have \(phase 0: \['missing.weight'\]\)"):
        adapter._validate_fsdp_fts_config()
