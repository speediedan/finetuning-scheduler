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
"""The ``pg_overrides`` and ``phase_config`` schedule keys under the Lightning callback, which runs its transitions
on a host-driven ``FTSController`` (#85)."""
import pytest
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint
from tests.test_finetuning_scheduler_callback import ScriptedLossModel, SnapshotAndStop

SCHEDULE = {0: {"params": ["model.2.weight", "model.2.bias"], "max_transition_epoch": 1,
                "phase_config": {"name": "head"}},
            1: {"params": ["model.1.weight", "model.1.bias"], "lr": 0.05, "max_transition_epoch": 2,
                "pg_overrides": {"momentum": 0.5, "weight_decay": 0.01}, "phase_config": {"name": "middle"}},
            2: {"params": ["model.0.weight", "model.0.bias"], "lr": 0.05, "max_transition_epoch": 3}}


def _fit(tmp_path, fts, max_epochs=3, extra=(), ckpt_path=None, losses=None):
    trainer = Trainer(default_root_dir=tmp_path, callbacks=[fts, *extra], max_epochs=max_epochs, accelerator="cpu",
                      devices=1, limit_train_batches=2, limit_val_batches=1, enable_progress_bar=False,
                      enable_model_summary=False, logger=False)
    trainer.fit(ScriptedLossModel(losses or [1.0] * max_epochs), ckpt_path=ckpt_path)
    return trainer


def test_pg_overrides_apply_to_a_later_phases_groups(tmp_path):
    seed_everything(42)
    fts = FinetuningScheduler(ft_schedule=SCHEDULE, epoch_transitions_only=True)
    trainer = _fit(tmp_path, fts)
    groups = trainer.optimizers[0].param_groups
    assert len(groups) == 3
    assert groups[1]["momentum"] == 0.5 and groups[1]["weight_decay"] == 0.01 and groups[1]["lr"] == 0.05
    assert groups[2]["momentum"] == 0 and groups[2]["weight_decay"] == 0  # the optimizer's defaults


def test_phase_zero_pg_overrides_are_refused(tmp_path):
    schedule = {**SCHEDULE, 0: {**SCHEDULE[0], "pg_overrides": {"momentum": 0.5}}}
    with pytest.raises(MisconfigurationException, match="Phase 0's `pg_overrides` is not supported"):
        _fit(tmp_path, FinetuningScheduler(ft_schedule=schedule, epoch_transitions_only=True))


def test_phase_hooks_receive_each_phases_config(tmp_path):
    """Start hooks run for phase 0 at fit start and for each later phase at its transition, end hooks when a phase
    ends (the final one at train end), each with the phase's ``phase_config``."""
    seed_everything(42)
    events = []
    fts = FinetuningScheduler(ft_schedule=SCHEDULE, epoch_transitions_only=True)
    fts.add_phase_hook(lambda e: events.append(("start", e.depth, e.epoch, e.phase_config)))
    fts.add_phase_hook(lambda e: events.append(("end", e.depth, e.phase_config)), when="end")
    _fit(tmp_path, fts)
    assert events == [("start", 0, 0, {"name": "head"}), ("end", 0, {"name": "head"}),
                      ("start", 1, 1, {"name": "middle"}), ("end", 1, {"name": "middle"}), ("start", 2, 2, None),
                      ("end", 2, None)]
    with pytest.raises(MisconfigurationException, match="`when` must be"):
        fts.add_phase_hook(lambda e: None, when="middle")


def test_a_resumed_fit_does_not_restart_phase_zero(tmp_path):
    """A fit resumed at phase 1 runs no start hooks for phases 0 or 1, and phase 1's group keeps its overrides
    (improving losses keep the best checkpoint at the resumed phase)."""
    seed_everything(42)
    losses, snapshot = [1.0, 0.9, 0.8], tmp_path / "snap.ckpt"
    first = FinetuningScheduler(ft_schedule=SCHEDULE, epoch_transitions_only=True)
    with pytest.raises(RuntimeError, match="interrupted"):
        _fit(tmp_path, first, losses=losses,
             extra=[FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "a", save_top_k=1),
                    SnapshotAndStop(snapshot=snapshot, snap_at=1, stop_after=1)])
    events = []
    resumed = FinetuningScheduler(ft_schedule=SCHEDULE, epoch_transitions_only=True)
    resumed.add_phase_hook(lambda e: events.append(("start", e.depth)))
    trainer = _fit(tmp_path, resumed, losses=losses, ckpt_path=snapshot,
                   extra=[FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "a", save_top_k=1)])
    assert events == [("start", 2)] and resumed.curr_depth == 2
    groups = trainer.optimizers[0].param_groups
    assert len(groups) == 3 and groups[1]["momentum"] == 0.5  # phase 1's group, with its overrides
