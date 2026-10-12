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
"""The direct PyTorch controller on FSDP2 (pool cases, gloo or NCCL).

With equal per-rank shards of each batch, FSDP2's gradient averaging makes a sharded run mathematically identical to a
single-process run on the full batch, so the phase path must match exactly and the parameters to float tolerance.
"""
import os
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import fully_shard

from finetuning_scheduler import FTSController, GateResult, PlateauTransition, TorchHost
from finetuning_scheduler.core.checkpoint import BestCheckpoint

SCHEDULE = {0: {"params": ["4.weight", "4.bias"]}, 1: {"params": ["2.weight", "2.bias"], "lr": 0.05},
            2: {"params": ["0.weight", "0.bias"]}}
VAL = [1.0, 1.2, 0.8, 0.9, 0.7, 0.75, 0.75, 0.75]  # plateaus at epochs 1, 3 and 5


def _model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _data(epochs: int, steps: int, batch: int, seed: int):
    gen = torch.Generator().manual_seed(seed + 1)
    return torch.randn(epochs, steps, batch, 8, generator=gen), torch.randn(epochs, steps, batch, 4, generator=gen)


def _controller(model, ckpt_dir, restore_best, gate=None):
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9),
                        transition=PlateauTransition("val_loss", patience=1), host=TorchHost(model),
                        checkpoint=BestCheckpoint(ckpt_dir, "val_loss") if ckpt_dir else None,
                        restore_best=restore_best if ckpt_dir else False)
    if gate is not None:
        fts.add_gate(gate)
    fts.setup()
    return fts


def _train(fts, model, x, y, shard, device, epochs, start=0, path=None):
    path = [] if path is None else path
    for epoch in fts.epochs(epochs):
        path.append((epoch, fts.curr_depth, len(fts.optimizer.param_groups)))
        for step in range(x.shape[1]):
            F.mse_loss(model(x[epoch, step, shard].to(device)), y[epoch, step, shard].to(device)).backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
        fts.epoch_end({"val_loss": VAL[epoch]})
    return path


def _params(model) -> dict:
    full = (lambda p: p.full_tensor()) if any(type(p).__name__ == "DTensor" for p in model.parameters()) else (
        lambda p: p)
    return {n: full(p).detach().flatten().tolist() for n, p in model.named_parameters()}


def _sharded_model(ctx, seed):
    model = _model(seed).to(ctx.device)
    for layer in (model[0], model[2], model[4]):
        fully_shard(layer, mesh=ctx.mesh())
    fully_shard(model, mesh=ctx.mesh())
    return model


def phases(ctx, ckpt_dir: str, restore_best: bool = True, epochs: int = 7, steps: int = 2, batch: int = 8,
           seed: int = 7) -> dict:
    model = _sharded_model(ctx, seed)
    fts = _controller(model, ckpt_dir, restore_best)
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    path = _train(fts, model, x, y, shard, ctx.device, epochs)
    return {"path": path, "params": _params(model), "restored": fts.restored_count, "stop": fts.stop_reason,
            "sharded_best": os.path.isdir(os.path.join(ckpt_dir, "best.pt"))}


def reference(ckpt_dir: str, restore_best: bool = True, epochs: int = 7, steps: int = 2, batch: int = 8,
              seed: int = 7) -> dict:
    model = _model(seed)
    fts = _controller(model, ckpt_dir, restore_best)
    x, y = _data(epochs, steps, batch, seed)
    path = _train(fts, model, x, y, slice(None), torch.device("cpu"), epochs)
    return {"path": path, "params": _params(model), "restored": fts.restored_count, "stop": fts.stop_reason}


def resume(ctx, ckpt_dir: str, epochs: int = 7, split: int = 3, steps: int = 2, batch: int = 8, seed: int = 7) -> dict:
    """Train ``split`` epochs, save, then load into a fresh model and controller and finish the run."""
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    first_model = _sharded_model(ctx, seed)
    first = _controller(first_model, os.path.join(ckpt_dir, "first"), True)
    path = _train(first, first_model, x, y, shard, ctx.device, split)
    saved = first.save(os.path.join(ckpt_dir, "resume"))
    model = _sharded_model(ctx, seed + 100)  # different initial weights: the load must overwrite them
    fts = _controller(model, os.path.join(ckpt_dir, "first"), True)
    fts.load(saved)
    path = _train(fts, model, x, y, shard, ctx.device, epochs, path=path)
    return {"path": path, "params": _params(model), "saved_is_dir": os.path.isdir(saved)}


def veto_on_rank(ctx, ckpt_dir: str, veto_rank: int = 1, epochs: int = 7, steps: int = 2, batch: int = 8,
                 seed: int = 7) -> dict:
    """A gate that vetoes phase 1 on one rank only stops every rank and restores phase 1's boundary state."""
    model = _sharded_model(ctx, seed)
    gate = (lambda e: GateResult(accept=not (e.depth == 1 and ctx.rank == veto_rank), reason="probe regressed"))
    fts = _controller(model, ckpt_dir, True, gate=gate)
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    path = _train(fts, model, x, y, shard, ctx.device, epochs)
    return {"path": path, "depth": fts.curr_depth, "stop": fts.stop_reason, "params": _params(model)}
