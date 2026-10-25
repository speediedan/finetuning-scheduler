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
on an externally driven ``FTSController`` (#85)."""
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


def test_end_hooks_see_the_restored_best_state(tmp_path):
    """As under the PyTorch API, the best checkpoint is restored before a phase's end hooks run."""
    import torch
    from lightning.pytorch.callbacks import Callback

    class BestWeights(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch == 0:  # the best epoch (losses 1.0, then 2.0)
                self.best = pl_module.model[2].weight.detach().clone()

    seed_everything(42)
    seen = {}
    schedule = {**SCHEDULE, 0: {**SCHEDULE[0], "max_transition_epoch": 2}}
    fts = FinetuningScheduler(ft_schedule=schedule, epoch_transitions_only=True)
    fts.add_phase_hook(lambda e: seen.setdefault(e.depth, e.controller.model.model[2].weight.detach().clone()),
                       when="end")
    best = BestWeights()
    _fit(tmp_path, fts, extra=[best, FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "c", save_top_k=1)],
         losses=[1.0, 2.0, 2.0])
    assert torch.equal(seen[0], best.best)


def test_hooks_survive_a_subclass_on_fit_start_without_super(tmp_path):
    """A subclass whose ``on_fit_start`` does not call ``super()`` gets its controller, with its phase hooks, at
    the first transition."""

    class NoSuperFitStart(FinetuningScheduler):
        def on_fit_start(self, trainer, pl_module):
            self.strategy_adapter.on_before_fts_fit_start()

    seed_everything(42)
    events = []
    fts = NoSuperFitStart(ft_schedule=SCHEDULE, epoch_transitions_only=True)
    fts.add_phase_hook(lambda e: events.append(e.depth))
    _fit(tmp_path, fts)
    assert 1 in events and 2 in events


def test_the_controller_follows_the_callbacks_depth():
    """The callback's depth is authoritative: before advancing, a controller behind it is synced to it."""
    from types import SimpleNamespace

    advanced = []
    controller = SimpleNamespace(curr_depth=0, _depth=0, request_advance=lambda: advanced.append(controller._depth))
    fts = FinetuningScheduler(ft_schedule=SCHEDULE)
    fts._fts_controller = controller
    fts._fts_state._curr_depth = 2  # set by a subclass, two phases on
    fts._controller_step()
    assert advanced == [1]


def test_teardown_reaches_the_strategy_adapter():
    """A fit that ends normally (``teardown``) or fails (``on_exception``) lets the adapter undo its changes."""
    from types import SimpleNamespace

    calls = []
    fts = FinetuningScheduler(ft_schedule=SCHEDULE)
    fts.strategy_adapter = SimpleNamespace(on_fts_teardown=lambda: calls.append(1))
    fts.teardown(None, None, "fit")
    fts.on_exception(None, None, RuntimeError("x"))
    assert calls == [1, 1]


def test_a_left_over_restore_patch_is_not_taken_for_the_original():
    """If an earlier fit's teardown never ran, the next fit's adapter does not record its leftover ``False`` as the
    strategy class's own value, so its teardown removes the patch."""
    from finetuning_scheduler.strategy_adapters import FSDPStrategyAdapter

    class SomeStrategy:
        pass

    SomeStrategy.lightning_restore_optimizer = False  # left by an earlier fit
    adapter = FSDPStrategyAdapter()
    adapter._disable_lightning_optimizer_restore(SomeStrategy)
    adapter.on_fts_teardown()
    assert "lightning_restore_optimizer" not in vars(SomeStrategy)


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
