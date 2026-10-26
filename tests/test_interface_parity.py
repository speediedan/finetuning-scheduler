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
"""Interface parity: one scenario run by the Lightning callback and by ``TuningScheduler`` must follow the same
phase path (#51).

Transitions are driven by scripted validation metrics, so the path does not depend on numerics. Each run records, at
the first batch of every epoch, the depth, the number of trainable parameters and every optimizer group's learning
rate.
"""
import os
from functools import partial

import pytest
import torch
from lightning.pytorch import Callback, LightningModule, seed_everything, Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from finetuning_scheduler import (FinetuningScheduler, FTSCheckpoint, TuningScheduler, FTSEarlyStopping,
                                  PlateauTransition)
from finetuning_scheduler.core.checkpoint import BestCheckpoint


def _layers(module: nn.Module) -> None:
    module.l0, module.l1, module.l2 = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 1)


def _forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return module.l2(torch.relu(module.l1(torch.relu(module.l0(x)))))


def _data() -> TensorDataset:
    x = torch.randn(16, 4, generator=torch.Generator().manual_seed(0))
    return TensorDataset(x, x.sum(1, keepdim=True))


class RawNet(nn.Module):
    def __init__(self):
        super().__init__()
        _layers(self)

    def forward(self, x):
        return _forward(self, x)


class ParityModule(LightningModule):
    def __init__(self, scripted: list[float], all_params: bool = False, rlrop: bool = False):
        super().__init__()
        _layers(self)
        self.scripted = scripted
        self.all_params, self.rlrop = all_params, rlrop

    def forward(self, x):
        return _forward(self, x)

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, _):
        self.log("val_loss", torch.tensor(self.scripted[self.current_epoch]))

    def configure_optimizers(self):
        params = self.parameters() if self.all_params else [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.1)
        if self.rlrop:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, factor=0.5)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val_loss"}}
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5),
                                                   "interval": "epoch"}}

    def train_dataloader(self):
        return DataLoader(_data(), batch_size=8)

    def val_dataloader(self):
        return DataLoader(_data(), batch_size=16)


def _record(depth, module, optimizer):
    return (depth, sum(p.requires_grad for p in module.parameters()),
            tuple(round(g["lr"], 10) for g in optimizer.param_groups), type(optimizer).__name__)


class PathRecorder(Callback):
    def __init__(self):
        self.path = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0:
            fts = next(c for c in trainer.callbacks if isinstance(c, FinetuningScheduler))
            self.path.append(_record(fts.curr_depth, pl_module, trainer.optimizers[0]))


THREE_PHASES = {
    0: {"params": ["l2.weight", "l2.bias"]},
    1: {"params": ["l1.weight", "l1.bias"], "lr": 0.05},
    2: {"params": ["l0.weight", "l0.bias"]},
}

# Lightning's epoch_transitions_only also requires the final phase's key (then overridden by max_epochs)
EPOCH_DRIVEN = {0: {**THREE_PHASES[0], "max_transition_epoch": 1}, 1: {**THREE_PHASES[1], "max_transition_epoch": 3},
                2: {**THREE_PHASES[2], "max_transition_epoch": 5}}

SCENARIOS = {
    # epoch-driven transitions only
    "epoch_driven": dict(
        schedule=EPOCH_DRIVEN,
        scripted=[1.0] * 5, max_epochs=5, plateau=False, max_depth=-1,
    ),
    # plateau-driven transitions (patience 1), ending when the final phase plateaus
    "plateau": dict(
        schedule=THREE_PHASES, scripted=[1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25], max_epochs=7, plateau=True,
        max_depth=-1,
    ),
    # plateau-driven, capped at depth 1
    "plateau_max_depth1": dict(
        schedule=THREE_PHASES, scripted=[1.0, 1.0, 0.5, 0.5, 0.5], max_epochs=5, plateau=True, max_depth=1,
    ),
    # an optimizer configured over every parameter, reconfigured for phase 0
    "enforce_phase0": dict(
        # max_epochs=4 is extended to the schedule's highest max_transition_epoch (5) on both interfaces
        schedule=EPOCH_DRIVEN, scripted=[1.0] * 5, max_epochs=4, plateau=False, max_depth=-1, all_params=True,
    ),
    # an explicit lr scheduler reinitialization at phase 1
    "reinit_lr": dict(
        schedule={**EPOCH_DRIVEN, 1: {**EPOCH_DRIVEN[1], "new_lr_scheduler": {
            "lr_scheduler_init": {"class_path": "torch.optim.lr_scheduler.ExponentialLR",
                                  "init_args": {"gamma": 0.9}}}}},
        scripted=[1.0] * 5, max_epochs=5, plateau=False, max_depth=-1,
    ),
    # an explicit optimizer (and lr scheduler) reinitialization at phase 1
    "reinit_optimizer": dict(
        schedule={**EPOCH_DRIVEN, 1: {**EPOCH_DRIVEN[1],
                                      "new_optimizer": {"optimizer_init": {"class_path": "torch.optim.Adam",
                                                                           "init_args": {"lr": 0.01}}},
                                      "new_lr_scheduler": {"lr_scheduler_init": {
                                          "class_path": "torch.optim.lr_scheduler.StepLR",
                                          "init_args": {"step_size": 1, "gamma": 0.5}}}}},
        scripted=[1.0] * 5, max_epochs=5, plateau=False, max_depth=-1,
    ),
    # plateau-driven with the best checkpoint restored at each boundary (epochs 1 and 3 are worse than their best)
    "restore_best": dict(
        schedule=THREE_PHASES, scripted=[1.0, 2.0, 0.5, 3.0, 0.25, 0.3, 0.3], max_epochs=7, plateau=True,
        max_depth=-1, restore_best=True,
    ),
    # a metric-driven lr scheduler
    "rlrop": dict(
        schedule=EPOCH_DRIVEN, scripted=[1.0, 0.5, 0.5, 0.25, 0.25], max_epochs=5, plateau=False, max_depth=-1,
        rlrop=True,
    ),
}


def _lightning_path(tmpdir, schedule, scripted, max_epochs, plateau, max_depth, all_params=False, rlrop=False,
                    restore_best=False):
    seed_everything(7)
    recorder = PathRecorder()
    callbacks = [FinetuningScheduler(ft_schedule=schedule, restore_best=restore_best,
                                     epoch_transitions_only=not plateau, max_depth=max_depth), recorder]
    if plateau:
        callbacks.append(FTSEarlyStopping(monitor="val_loss", patience=1))
    if restore_best:
        callbacks.append(FTSCheckpoint(monitor="val_loss", save_top_k=1, dirpath=tmpdir))
    trainer = Trainer(default_root_dir=tmpdir, accelerator="cpu", devices=1, max_epochs=max_epochs,
                      num_sanity_val_steps=0, callbacks=callbacks, logger=False, enable_progress_bar=False,
                      enable_model_summary=False)
    trainer.fit(ParityModule(scripted, all_params=all_params, rlrop=rlrop))
    return recorder.path


def _torch_path(schedule, scripted, max_epochs, plateau, max_depth, all_params=False, rlrop=False,
                restore_best=False, ckpt_dir=None):
    seed_everything(7)
    model = RawNet()
    if all_params:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    else:
        optimizer = partial(torch.optim.SGD, lr=0.1)
        lr_scheduler = (partial(torch.optim.lr_scheduler.ReduceLROnPlateau, patience=0, factor=0.5) if rlrop else
                        partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5))
    fts = TuningScheduler(model, schedule, optimizer=optimizer, lr_scheduler=lr_scheduler,
                          lr_scheduler_cfg={"interval": "epoch", "monitor": "val_loss" if rlrop else None},
                          max_depth=max_depth,
                          transition=PlateauTransition("val_loss", patience=1) if plateau else None,
                          checkpoint=BestCheckpoint(ckpt_dir, "val_loss") if restore_best else None)
    fts.setup()
    path = []
    for epoch in fts.epochs(max_epochs):
        for i, (x, y) in enumerate(DataLoader(_data(), batch_size=8)):
            if i == 0:
                path.append(_record(fts.curr_depth, model, fts.optimizer))
            torch.nn.functional.mse_loss(model(x), y).backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
        metrics = {"val_loss": scripted[epoch]}
        fts.epoch_end(metrics)
        fts.step_lr_schedulers("epoch", metrics)
    return path


# (depth, trainable parameter count, per-group learning rates, optimizer type) at the first batch of each epoch
_E = (1, 4, (0.05, 0.05), "SGD"), (1, 4, (0.025, 0.025), "SGD")
EXPECTED_PATHS = {
    "epoch_driven": [(0, 2, (0.1,), "SGD"), *_E, (2, 6, (0.0125, 0.0125, 1e-05), "SGD"),
                     (2, 6, (0.00625, 0.00625, 5e-06), "SGD")],
    "plateau": [(0, 2, (0.1,), "SGD"), (0, 2, (0.05,), "SGD"), (1, 4, (0.025, 0.05), "SGD"),
                (1, 4, (0.0125, 0.025), "SGD"), (2, 6, (0.00625, 0.0125, 1e-05), "SGD"),
                (2, 6, (0.003125, 0.00625, 5e-06), "SGD")],
    "plateau_max_depth1": [(0, 2, (0.1,), "SGD"), (0, 2, (0.05,), "SGD"), (1, 4, (0.025, 0.05), "SGD"),
                           (1, 4, (0.0125, 0.025), "SGD")],
    "enforce_phase0": [(0, 2, (0.1,), "SGD"), *_E, (2, 6, (0.0125, 0.0125, 1e-05), "SGD"),
                       (2, 6, (0.00625, 0.00625, 5e-06), "SGD")],
    # the reinitialized scheduler restarts the existing groups at their initial lr
    "reinit_lr": [(0, 2, (0.1,), "SGD"), (1, 4, (0.1, 0.05), "SGD"), (1, 4, (0.09, 0.045), "SGD"),
                  (2, 6, (0.081, 0.0405, 1e-05), "SGD"), (2, 6, (0.0729, 0.03645, 9e-06), "SGD")],
    "reinit_optimizer": [(0, 2, (0.1,), "SGD"), (1, 4, (0.01, 0.05), "Adam"), (1, 4, (0.005, 0.025), "Adam"),
                         (2, 6, (0.0025, 0.0125, 1e-05), "Adam"), (2, 6, (0.00125, 0.00625, 5e-06), "Adam")],
    # restoring the best at each boundary keeps the lr progression (the latest lrs are re-applied after the restore)
    "restore_best": [(0, 2, (0.1,), "SGD"), (0, 2, (0.05,), "SGD"), (1, 4, (0.025, 0.05), "SGD"),
                     (1, 4, (0.0125, 0.025), "SGD"), (2, 6, (0.00625, 0.0125, 1e-05), "SGD"),
                     (2, 6, (0.003125, 0.00625, 5e-06), "SGD")],
    "rlrop": [(0, 2, (0.1,), "SGD"), (1, 4, (0.1, 0.05), "SGD"), (1, 4, (0.1, 0.05), "SGD"),
              (2, 6, (0.05, 0.025, 1e-05), "SGD"), (2, 6, (0.05, 0.025, 1e-05), "SGD")],
}


@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=list(SCENARIOS))
def test_interface_parity(tmpdir, scenario):
    """Both interfaces follow the scenario's expected path."""
    cfg = SCENARIOS[scenario]
    lightning_path = _lightning_path(os.path.join(tmpdir, "lightning"), **cfg)
    torch_path = _torch_path(**cfg, ckpt_dir=os.path.join(tmpdir, "torch"))
    assert torch_path == lightning_path
    assert lightning_path == EXPECTED_PATHS[scenario]
