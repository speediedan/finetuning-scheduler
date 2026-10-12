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
"""The Lightning callback thaws the parameters FSDP2 trains, although the validation loop's forward-only pass
leaves the root FSDP2 module's unsharded parameters registered at the next phase transition (#92)."""
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, seed_everything, Trainer
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, TensorDataset

from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint
from finetuning_scheduler.core import phase as core_phase
from tests.dist_cases._mp_models import FTSToyTransformer, TestModelArgs
from tests.helpers.runif import RunIf

SEQ = 16
# the Lightning model-parallel tests' schedule, with an ``lr`` for each later phase so their training is measurable
SCHEDULE = {
    0: {"params": [r"model.output.weight", r"model.norm.*"], "max_transition_epoch": 1},
    1: {"params": [r"model.layers.1.(feed_forward|ffn_norm|attention.w.*|attention_norm).*"],
        "max_transition_epoch": 2, "lr": 0.1},
    2: {"params": [r"model.layers.0.(feed_forward|ffn_norm|attention.w.*|attention_norm).*",
                   r"model.(pos_embeddings|tok_embeddings).weight"], "max_transition_epoch": 3, "lr": 0.1},
}


class ShardedToyTransformer(LightningModule):
    """FSDP2 on each block and then on ``model``, the root FSDP module, which owns the embeddings thawed last."""

    def __init__(self):
        super().__init__()
        self.model = FTSToyTransformer(TestModelArgs(vocab_size=64, max_seq_len=SEQ, dim=16, n_heads=2))

    def configure_model(self):
        mesh = self.device_mesh["data_parallel"]
        for layer in self.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(self.model, mesh=mesh)

    def forward(self, tokens):
        return self.model(tokens)

    def _loss(self, batch):
        (tokens,) = batch
        logits = self(tokens[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))

    def training_step(self, batch, _):
        return self._loss(batch)

    def validation_step(self, batch, _):
        self.log("val_loss", self._loss(batch))

    def configure_optimizers(self):
        return torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=0.1)

    def train_dataloader(self):
        tokens = torch.randint(0, 64, (16, SEQ + 1), generator=torch.Generator().manual_seed(0))
        return DataLoader(TensorDataset(tokens), batch_size=4)

    def val_dataloader(self):
        return self.train_dataloader()


class Movement(Callback):
    """Each parameter's largest change over the fit."""

    @staticmethod
    def _values(pl_module) -> dict:
        for module in pl_module.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        return {n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
                for n, p in pl_module.named_parameters()}

    def on_train_start(self, trainer, pl_module):
        self.initial = self._values(pl_module)

    def on_train_end(self, trainer, pl_module):
        final = self._values(pl_module)
        self.moved = {n: (final[n] - self.initial[n]).abs().max().item() for n in final}


def _fit(tmp_path) -> dict:
    seed_everything(1)
    movement = Movement()
    trainer = Trainer(
        default_root_dir=tmp_path, accelerator="cpu", devices=1, max_epochs=3,
        strategy=ModelParallelStrategy(data_parallel_size=1, tensor_parallel_size=1),
        # with ``restore_best`` or a checkpoint written at the last validation, FSDP2's state-dict hooks reshard first
        callbacks=[FinetuningScheduler(ft_schedule=SCHEDULE, epoch_transitions_only=True, restore_best=False),
                   FTSCheckpoint(monitor="val_loss", save_top_k=0), movement],
        logger=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(ShardedToyTransformer())
    return movement.moved


@RunIf(skip_windows=True, skip_mac_os=True)
def test_params_thawed_after_a_validation_pass_train(tmp_path):
    moved = _fit(tmp_path)
    assert all(moved[n] > 1e-4 for n in ("model.tok_embeddings.weight", "model.pos_embeddings.weight")), moved


@RunIf(skip_windows=True, skip_mac_os=True)
def test_without_the_reshard_root_params_never_train(tmp_path, monkeypatch):
    """Negative control: without the reshard, the root's embeddings are thawed as unsharded copies and never move."""
    monkeypatch.setattr(core_phase, "reshard_fsdp_modules", lambda module: None)
    moved = _fit(tmp_path)
    assert moved["model.tok_embeddings.weight"] == moved["model.pos_embeddings.weight"] == 0.0
    assert moved["model.layers.0.feed_forward.w2.weight"] > 1e-4  # thawed in the same phase, by a non-root module


def test_reshard_is_a_noop_without_fsdp():
    module = torch.nn.Linear(2, 2)
    core_phase.reshard_fsdp_modules(module)
    assert isinstance(module.weight, torch.nn.Parameter)
