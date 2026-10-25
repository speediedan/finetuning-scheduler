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
"""The order in which the Lightning callback dispatches to its strategy adapter, pinned against committed
snapshots.

Third-party and built-in strategy adapters customize fine-tuning by overriding adapter hooks, and tests subclass the
callback's own dispatch methods, so the order and arguments of those calls are part of the callback's contract. Each
scenario records them (see :mod:`tests.helpers.dispatch_trace`) and compares the trace with
``tests/helpers/dispatch_traces.json``. A change to the callback's internals must leave every trace unchanged, or
change the snapshot deliberately (``FTS_UPDATE_DISPATCH_TRACES=1``) with the reason in its review.
"""
import pytest
from lightning.pytorch import seed_everything, Trainer

from finetuning_scheduler import FTSCheckpoint, FTSEarlyStopping
from tests.helpers.dispatch_trace import check_snapshot, TracingFinetuningScheduler
from tests.helpers.legacy_naming_adapter import LogicalNamesBoringModel
from tests.test_finetuning_scheduler_callback import ScriptedLossModel, SnapshotAndStop
from tests.test_legacy_adapter_contract import LOGICAL_SCHEDULE

BASE = {"single_device": "tests.helpers.dispatch_trace:TracedStrategyAdapter"}
LEGACY = {"single_device": "tests.helpers.dispatch_trace:TracedLegacyNamingAdapter"}
THREE_PHASES = {0: {"params": ["model.2.weight", "model.2.bias"], "max_transition_epoch": 2},
                1: {"params": ["model.1.weight", "model.1.bias"], "lr": 0.05, "max_transition_epoch": 4},
                2: {"params": ["model.0.weight", "model.0.bias"], "lr": 0.05, "max_transition_epoch": 6}}
REINIT = {**THREE_PHASES, 1: {**THREE_PHASES[1],
                              "new_optimizer": {"optimizer_init": {"class_path": "torch.optim.Adam",
                                                                   "init_args": {"lr": 0.01}}},
                              "new_lr_scheduler": {"lr_scheduler_init": {
                                  "class_path": "torch.optim.lr_scheduler.StepLR",
                                  "init_args": {"step_size": 1, "gamma": 0.5}}}}}


class ScheduledLossModel(ScriptedLossModel):
    """With an lr scheduler, which the callback's lr scheduler reinitialization replaces."""

    def configure_optimizers(self):
        import torch

        optimizer = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=0.05)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)]


def _fit(tmp_path, fts, model, max_epochs, extra=(), ckpt_path=None, **kwargs):
    trainer = Trainer(default_root_dir=tmp_path, callbacks=[fts, *extra], max_epochs=max_epochs, accelerator="cpu",
                      devices=1, limit_train_batches=2, limit_val_batches=1, enable_progress_bar=False,
                      enable_model_summary=False, logger=False, **kwargs)
    trainer.fit(model, ckpt_path=ckpt_path)
    return trainer


def test_epoch_driven(tmp_path):
    seed_everything(42)
    fts = TracingFinetuningScheduler(ft_schedule=THREE_PHASES, epoch_transitions_only=True,
                                     custom_strategy_adapters=BASE)
    _fit(tmp_path, fts, ScriptedLossModel([1.0] * 6), 6)
    assert fts._fts_controller is not None and fts._fts_controller.curr_depth == fts.curr_depth == 2
    check_snapshot("epoch_driven", fts.dispatch_trace)


def test_legacy_naming_adapter(tmp_path):
    seed_everything(42)
    fts = TracingFinetuningScheduler(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True,
                                     custom_strategy_adapters=LEGACY)
    _fit(tmp_path, fts, LogicalNamesBoringModel(), 4)
    check_snapshot("legacy_naming_adapter", fts.dispatch_trace)


@pytest.mark.parametrize("scenario, losses", [
    ("early_stopping_driven", [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25]),
    # epochs 1 and 3 are worse than their phase's best, so each transition restores an earlier checkpoint
    ("restore_best", [1.0, 2.0, 0.5, 3.0, 0.25, 0.3, 0.3]),
])
def test_early_stopping_driven_with_best_restores(tmp_path, scenario, losses):
    seed_everything(42)
    schedule = {d: {k: v for k, v in p.items() if k != "max_transition_epoch"} for d, p in THREE_PHASES.items()}
    fts = TracingFinetuningScheduler(ft_schedule=schedule, custom_strategy_adapters=BASE)
    extra = [FTSEarlyStopping(monitor="val_loss", patience=1, min_delta=0.0),
             FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "ckpts", save_top_k=1)]
    _fit(tmp_path, fts, ScriptedLossModel(losses), len(losses), extra=extra)
    check_snapshot(scenario, fts.dispatch_trace)


def test_optimizer_and_lr_scheduler_reinitialization(tmp_path):
    seed_everything(42)
    fts = TracingFinetuningScheduler(ft_schedule=REINIT, epoch_transitions_only=True, custom_strategy_adapters=BASE)
    _fit(tmp_path, fts, ScheduledLossModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]), 6)
    check_snapshot("reinitialization", fts.dispatch_trace)


def test_restore_across_an_optimizer_class_change(tmp_path):
    """The best checkpoint stays phase 0's (SGD) while phase 1 reinitializes to Adam: at the transition to phase 2
    the callback restores the model state only (no optimizer restore hook) and rebuilds the optimizer (#108)."""
    seed_everything(42)
    fts = TracingFinetuningScheduler(ft_schedule=REINIT, epoch_transitions_only=True, custom_strategy_adapters=BASE)
    _fit(tmp_path, fts, ScheduledLossModel([1.0, 2.0, 2.0, 2.0, 2.0, 2.0]), 6,
         extra=[FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "ckpts", save_top_k=1)])
    check_snapshot("class_change_restore", fts.dispatch_trace)


def test_resume_at_a_later_depth(tmp_path):
    """A fit interrupted at depth 1 and resumed from its snapshot: the resumed fit's dispatch."""
    seed_everything(42)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    snapshot = tmp_path / "snap.ckpt"
    first = TracingFinetuningScheduler(ft_schedule=THREE_PHASES, epoch_transitions_only=True,
                                       custom_strategy_adapters=BASE)
    with pytest.raises(RuntimeError, match="interrupted"):
        _fit(tmp_path, first, ScriptedLossModel(losses), 6,
             extra=[FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "a", save_top_k=1),
                    SnapshotAndStop(snapshot=snapshot, snap_at=3, stop_after=3)])
    seed_everything(42)
    resumed = TracingFinetuningScheduler(ft_schedule=THREE_PHASES, epoch_transitions_only=True,
                                         custom_strategy_adapters=BASE)
    _fit(tmp_path, resumed, ScriptedLossModel(losses), 6,
         extra=[FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "b", save_top_k=1)], ckpt_path=snapshot)
    assert resumed._fts_controller is not None and resumed._fts_controller.curr_depth == resumed.curr_depth == 2
    check_snapshot("resume", resumed.dispatch_trace)


class LegacyStepFTS(TracingFinetuningScheduler):
    """The callback with its own transitions rather than its externally driven controller's
    (``_use_fts_controller``)."""

    _use_fts_controller = False


@pytest.mark.parametrize("scenario", ["epoch_driven", "reinitialization", "early_stopping_driven", "restore_best",
                                      "legacy_naming_adapter", "resume"])
def test_the_callbacks_own_transitions_dispatch_as_its_controller_does(tmp_path, scenario):
    """The callback sequences transitions through its externally driven controller (#85); with its own transitions
    instead (``_use_fts_controller = False``, kept for one release), every scenario dispatches identically."""
    seed_everything(42)
    if scenario == "resume":
        losses, snapshot = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], tmp_path / "snap.ckpt"
        first = LegacyStepFTS(ft_schedule=THREE_PHASES, epoch_transitions_only=True, custom_strategy_adapters=BASE)
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit(tmp_path, first, ScriptedLossModel(losses), 6,
                 extra=[FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "a", save_top_k=1),
                        SnapshotAndStop(snapshot=snapshot, snap_at=3, stop_after=3)])
        seed_everything(42)
        fts = LegacyStepFTS(ft_schedule=THREE_PHASES, epoch_transitions_only=True, custom_strategy_adapters=BASE)
        _fit(tmp_path, fts, ScriptedLossModel(losses), 6,
             extra=[FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "b", save_top_k=1)], ckpt_path=snapshot)
    elif scenario == "legacy_naming_adapter":
        fts = LegacyStepFTS(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True,
                                custom_strategy_adapters=LEGACY)
        _fit(tmp_path, fts, LogicalNamesBoringModel(), 4)
    elif scenario in ("epoch_driven", "reinitialization"):
        schedule, model = (THREE_PHASES, ScriptedLossModel([1.0] * 6)) if scenario == "epoch_driven" else (
            REINIT, ScheduledLossModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))
        fts = LegacyStepFTS(ft_schedule=schedule, epoch_transitions_only=True, custom_strategy_adapters=BASE)
        _fit(tmp_path, fts, model, 6)
    else:
        losses = {"early_stopping_driven": [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25],
                  "restore_best": [1.0, 2.0, 0.5, 3.0, 0.25, 0.3, 0.3]}[scenario]
        schedule = {d: {k: v for k, v in p.items() if k != "max_transition_epoch"} for d, p in THREE_PHASES.items()}
        fts = LegacyStepFTS(ft_schedule=schedule, custom_strategy_adapters=BASE)
        extra = [FTSEarlyStopping(monitor="val_loss", patience=1, min_delta=0.0),
                 FTSCheckpoint(monitor="val_loss", dirpath=tmp_path / "ckpts", save_top_k=1)]
        _fit(tmp_path, fts, ScriptedLossModel(losses), len(losses), extra=extra)
    assert fts._fts_controller is None  # never attached
    check_snapshot(scenario, fts.dispatch_trace)


def test_schedule_generation(tmp_path):
    fts = TracingFinetuningScheduler(gen_ft_sched_only=True, custom_strategy_adapters=BASE)
    with pytest.raises(SystemExit):
        _fit(tmp_path, fts, ScriptedLossModel([1.0] * 2), 2)
    check_snapshot("schedule_generation", fts.dispatch_trace)
