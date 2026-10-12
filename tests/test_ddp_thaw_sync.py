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
"""Regression tests for gradient synchronization of parameters thawed after ``DistributedDataParallel`` wraps the
module (#72)."""
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from lightning.pytorch import Callback, LightningModule, seed_everything, Trainer
from torch.utils.data import DataLoader, TensorDataset

from finetuning_scheduler import FinetuningScheduler
from finetuning_scheduler.strategy_adapters import StrategyAdapter
from tests.helpers.runif import RunIf


class TwoLayerModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(8, 8)
        self.layer2 = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, _):
        x, y = batch
        self.log("val_loss", torch.nn.functional.mse_loss(self(x), y), sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=0.1)

    def train_dataloader(self):
        x = torch.randn(128, 8, generator=torch.Generator().manual_seed(0))
        return DataLoader(TensorDataset(x, x.sum(1, keepdim=True) * 0.3), batch_size=16)

    def val_dataloader(self):
        return self.train_dataloader()


class RankParity(Callback):
    """At the end of training, rank 0 records each parameter's largest difference across ranks."""

    def __init__(self, out_file: str):
        self.out_file = out_file

    def on_train_end(self, trainer, pl_module):
        diffs = {}
        for name, p in pl_module.named_parameters():
            gathered = [torch.empty_like(p) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, p.detach().contiguous())
            diffs[name] = max((g - gathered[0]).abs().max().item() for g in gathered)
        if trainer.is_global_zero:
            with open(self.out_file, "w") as f:
                json.dump(diffs, f)


class NoReducerSyncAdapter(StrategyAdapter):
    """The behavior before the fix: the reducer is never rebuilt after a phase transition."""

    def on_after_phase_transition(self) -> None:
        pass


# layer2 trains from phase 0 (trainable when DDP wraps the module); layer1 is thawed in phase 1 at epoch 1
SCHEDULE = {
    0: {"params": ["layer2.weight", "layer2.bias"], "max_transition_epoch": 1},
    1: {"params": ["layer1.weight", "layer1.bias"], "max_transition_epoch": 3, "lr": 0.1},
}


def _fit_and_measure(tmpdir, strategy: str, accelerator: str, **fts_kwargs) -> dict:
    seed_everything(1)
    out_file = os.path.join(tmpdir, "rank_parity.json")
    trainer = Trainer(
        default_root_dir=tmpdir, accelerator=accelerator, devices=2, strategy=strategy, max_epochs=3,
        callbacks=[FinetuningScheduler(ft_schedule=SCHEDULE, epoch_transitions_only=True, **fts_kwargs),
                   RankParity(out_file)],
        logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(TwoLayerModule())
    with open(out_file) as f:
        return json.load(f)


@RunIf(skip_windows=True)
def test_ddp_syncs_parameters_thawed_after_wrapping(tmpdir):
    """Every parameter, including those thawed after ``DistributedDataParallel`` wrapped the module, stays
    identical across ranks (CPU, gloo)."""
    diffs = _fit_and_measure(tmpdir, "ddp_spawn", "cpu")
    assert diffs == {name: 0.0 for name in diffs}, diffs


@RunIf(skip_windows=True)
def test_ddp_reducer_not_rebuilt_diverges(tmpdir):
    """Negative control: without the reducer rebuild, the layer thawed after the wrap diverges across ranks while the
    phase-0 layer does not, so the test above can detect the bug."""
    diffs = _fit_and_measure(tmpdir, "ddp_spawn", "cpu",
                             custom_strategy_adapters={"ddp_spawn": f"{__name__}.NoReducerSyncAdapter"})
    assert diffs["layer2.weight"] == 0.0
    assert diffs["layer1.weight"] > 1e-3, diffs


@RunIf(standalone=True, min_cuda_gpus=2)
def test_ddp_syncs_parameters_thawed_after_wrapping_cuda(tmpdir):
    """The same check with NCCL on two GPUs."""
    diffs = _fit_and_measure(tmpdir, "ddp", "gpu")
    assert diffs == {name: 0.0 for name in diffs}, diffs


def test_sync_ddp_reducer_is_noop_without_ddp():
    """Strategies that do not wrap the module in ``DistributedDataParallel`` are left alone."""

    class _Probe(StrategyAdapter):
        @property
        def pls_handle(self):
            return SimpleNamespace(model=torch.nn.Linear(2, 2))

    adapter = _Probe()
    adapter.on_after_phase_transition()
    assert adapter._ddp_reducer_params is None
