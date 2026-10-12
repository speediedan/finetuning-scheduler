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
"""Host parity: one scenario run by the Lightning callback and by the direct PyTorch controller must follow the same
phase path (#51).

Transitions are driven by scripted validation metrics, so the path does not depend on numerics. Each run records, at
the first batch of every epoch, the depth, the number of trainable parameters and every optimizer group's learning
rate.
"""
from functools import partial

import pytest
import torch
from lightning.pytorch import Callback, LightningModule, seed_everything, Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from finetuning_scheduler import FinetuningScheduler, FTSController, FTSEarlyStopping, PlateauTransition


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
    def __init__(self, scripted: list[float]):
        super().__init__()
        _layers(self)
        self.scripted = scripted

    def forward(self, x):
        return _forward(self, x)

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, _):
        self.log("val_loss", torch.tensor(self.scripted[self.current_epoch]))

    def configure_optimizers(self):
        opt = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5),
                                                   "interval": "epoch"}}

    def train_dataloader(self):
        return DataLoader(_data(), batch_size=8)

    def val_dataloader(self):
        return DataLoader(_data(), batch_size=16)


def _record(depth, module, optimizer):
    return (depth, sum(p.requires_grad for p in module.parameters()),
            tuple(round(g["lr"], 10) for g in optimizer.param_groups))


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

SCENARIOS = {
    # epoch-driven transitions only
    "epoch_driven": dict(
        # Lightning's epoch_transitions_only also requires the final phase's key (then overridden by max_epochs)
        schedule={0: {**THREE_PHASES[0], "max_transition_epoch": 1}, 1: {**THREE_PHASES[1], "max_transition_epoch": 3},
                  2: {**THREE_PHASES[2], "max_transition_epoch": 5}},
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
}


def _lightning_path(tmpdir, schedule, scripted, max_epochs, plateau, max_depth):
    seed_everything(7)
    recorder = PathRecorder()
    callbacks = [FinetuningScheduler(ft_schedule=schedule, restore_best=False, epoch_transitions_only=not plateau,
                                     max_depth=max_depth), recorder]
    if plateau:
        callbacks.append(FTSEarlyStopping(monitor="val_loss", patience=1))
    trainer = Trainer(default_root_dir=tmpdir, accelerator="cpu", devices=1, max_epochs=max_epochs,
                      num_sanity_val_steps=0, callbacks=callbacks, logger=False, enable_progress_bar=False,
                      enable_model_summary=False)
    trainer.fit(ParityModule(scripted))
    return recorder.path


def _torch_path(schedule, scripted, max_epochs, plateau, max_depth):
    seed_everything(7)
    model = RawNet()
    fts = FTSController(model, schedule, optimizer=partial(torch.optim.SGD, lr=0.1),
                        lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5),
                        lr_scheduler_cfg={"interval": "epoch"}, max_depth=max_depth,
                        transition=PlateauTransition("val_loss", patience=1) if plateau else None)
    fts.setup()
    path = []
    for epoch in fts.epochs(max_epochs):
        for i, (x, y) in enumerate(DataLoader(_data(), batch_size=8)):
            if i == 0:
                path.append(_record(fts.curr_depth, model, fts.optimizer))
            torch.nn.functional.mse_loss(model(x), y).backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
        fts.epoch_end({"val_loss": scripted[epoch]})
        fts.step_lr_schedulers("epoch")
    return path


# (depth, trainable parameter count, per-group learning rates) at the first batch of each epoch
EXPECTED_PATHS = {
    "epoch_driven": [
        (0, 2, (0.1,)), (1, 4, (0.05, 0.05)), (1, 4, (0.025, 0.025)), (2, 6, (0.0125, 0.0125, 1e-05)),
        (2, 6, (0.00625, 0.00625, 5e-06)),
    ],
    "plateau": [
        (0, 2, (0.1,)), (0, 2, (0.05,)), (1, 4, (0.025, 0.05)), (1, 4, (0.0125, 0.025)),
        (2, 6, (0.00625, 0.0125, 1e-05)), (2, 6, (0.003125, 0.00625, 5e-06)),
    ],
    "plateau_max_depth1": [(0, 2, (0.1,)), (0, 2, (0.05,)), (1, 4, (0.025, 0.05)), (1, 4, (0.0125, 0.025))],
}


@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=list(SCENARIOS))
def test_host_parity(tmpdir, scenario):
    """Both hosts follow the scenario's expected path."""
    cfg = SCENARIOS[scenario]
    assert _lightning_path(tmpdir, **cfg) == EXPECTED_PATHS[scenario]
    assert _torch_path(**cfg) == EXPECTED_PATHS[scenario]
