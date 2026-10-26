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
"""``TuningScheduler`` on FSDP2 (pool cases, gloo or NCCL).

With equal per-rank shards of each batch, FSDP2's gradient averaging makes a sharded run mathematically identical to a
single-process run on the full batch, so the phase path must match exactly and the parameters to float tolerance.
"""
import os
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import fully_shard

from finetuning_scheduler import TuningScheduler, GateResult, PlateauTransition, TorchInterface
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


def _tuning_scheduler(model, ckpt_dir, restore_best, gate=None):
    fts = TuningScheduler(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9),
                          transition=PlateauTransition("val_loss", patience=1), interface=TorchInterface(model),
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
    fts = _tuning_scheduler(model, ckpt_dir, restore_best)
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    path = _train(fts, model, x, y, shard, ctx.device, epochs)
    return {"path": path, "params": _params(model), "restored": fts.restored_count, "stop": fts.stop_reason,
            "sharded_best": os.path.isdir(fts._checkpoint.best_path)}


def reference(ckpt_dir: str, restore_best: bool = True, epochs: int = 7, steps: int = 2, batch: int = 8,
              seed: int = 7) -> dict:
    model = _model(seed)
    fts = _tuning_scheduler(model, ckpt_dir, restore_best)
    x, y = _data(epochs, steps, batch, seed)
    path = _train(fts, model, x, y, slice(None), torch.device("cpu"), epochs)
    return {"path": path, "params": _params(model), "restored": fts.restored_count, "stop": fts.stop_reason}


def resume(ctx, ckpt_dir: str, epochs: int = 7, split: int = 3, steps: int = 2, batch: int = 8, seed: int = 7) -> dict:
    """Train ``split`` epochs, save, then load into a fresh model and tuning scheduler and finish the run."""
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    first_model = _sharded_model(ctx, seed)
    first = _tuning_scheduler(first_model, os.path.join(ckpt_dir, "first"), True)
    path = _train(first, first_model, x, y, shard, ctx.device, split)
    saved = first.save(os.path.join(ckpt_dir, "resume"))
    model = _sharded_model(ctx, seed + 100)  # different initial weights: the load must overwrite them
    fts = _tuning_scheduler(model, os.path.join(ckpt_dir, "first"), True)
    fts.load(saved)
    path = _train(fts, model, x, y, shard, ctx.device, epochs, path=path)
    return {"path": path, "params": _params(model), "saved_is_dir": os.path.isdir(saved)}


def veto_on_rank(ctx, ckpt_dir: str, veto_rank: int = 1, epochs: int = 7, steps: int = 2, batch: int = 8,
                 seed: int = 7) -> dict:
    """A gate that vetoes phase 1 on one rank only stops every rank and restores phase 1's boundary state."""
    model = _sharded_model(ctx, seed)
    gate = (lambda e: GateResult(accept=not (e.depth == 1 and ctx.rank == veto_rank), reason="probe regressed"))
    fts = _tuning_scheduler(model, ckpt_dir, True, gate=gate)
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    path = _train(fts, model, x, y, shard, ctx.device, epochs)
    return {"path": path, "depth": fts.curr_depth, "stop": fts.stop_reason, "params": _params(model)}


def unsynced_metrics(ctx, ckpt_dir: str, epochs: int = 5, steps: int = 2, batch: int = 8, seed: int = 7) -> dict:
    """Ranks report different metrics: the best-checkpoint decision (which gates collective saves) must still be
    agreed, so the run neither deadlocks nor diverges."""
    model = _model(seed).to(ctx.device)
    fts = _tuning_scheduler(model, ckpt_dir, True)
    x, y = _data(epochs, steps, batch, seed)
    val = [1.0, 0.8, 0.7, 0.6, 0.5] if ctx.rank == 0 else [1.0, 1.1, 1.2, 1.3, 1.4]
    path = []
    for epoch in fts.epochs(epochs):
        path.append((epoch, fts.curr_depth))
        for step in range(steps):
            F.mse_loss(model(x[epoch, step].to(ctx.device)), y[epoch, step].to(ctx.device)).backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
        fts.epoch_end({"val_loss": val[epoch]})
    return {"path": path, "best_depth": fts._checkpoint.best_depth, "restored": fts.restored_count}


def optimizer_state_edges(ctx, ckpt_dir: str, epochs: int = 7, steps: int = 2, batch: int = 8, seed: int = 7) -> dict:
    """Sharded checkpoint edge cases around DSD's optimizer-state initialization: saving before any step must not
    change the optimizer (Adam, ``save_initial``); restoring while gradients are present must still load the state
    (``zero_grad(set_to_none=False)``)."""
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    out = {}
    for save_initial in (False, True):
        model = _sharded_model(ctx, seed)
        fts = TuningScheduler(model, SCHEDULE, optimizer=partial(torch.optim.Adam, lr=0.01),
                              transition=PlateauTransition("val_loss", patience=1), interface=TorchInterface(model),
                              checkpoint=BestCheckpoint(os.path.join(ckpt_dir, f"adam_{save_initial}"), "val_loss",
                                                      save_initial=save_initial))
        fts.setup()
        _train(fts, model, x, y, shard, ctx.device, 2)
        out[f"adam_save_initial_{save_initial}"] = _params(model)
    for set_to_none in (True, False):
        model = _sharded_model(ctx, seed)
        fts = _tuning_scheduler(model, os.path.join(ckpt_dir, f"stn_{set_to_none}"), True)
        for epoch in fts.epochs(epochs):
            for step in range(steps):
                F.mse_loss(model(x[epoch, step, shard].to(ctx.device)),
                           y[epoch, step, shard].to(ctx.device)).backward()
                fts.optimizer.step()
                fts.optimizer.zero_grad(set_to_none=set_to_none)
            fts.epoch_end({"val_loss": VAL[epoch]})
        out[f"set_to_none_{set_to_none}"] = _params(model)
    return out


def save_at_boundary(ctx, ckpt_dir: str, epochs: int = 7, steps: int = 2, batch: int = 8, seed: int = 7) -> dict:
    """A sharded save right after a phase boundary, before the new phase's parameters have optimizer state, loads
    and resumes exactly."""
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    first_model = _sharded_model(ctx, seed)
    first = _tuning_scheduler(first_model, os.path.join(ckpt_dir, "first"), True)
    path, saved = [], None
    for epoch in first.epochs(epochs):
        path.append((epoch, first.curr_depth, len(first.optimizer.param_groups)))
        if first.curr_depth == 1:
            saved = first.save(os.path.join(ckpt_dir, "boundary_resume"))
            break
        for step in range(steps):
            F.mse_loss(first_model(x[epoch, step, shard].to(ctx.device)),
                       y[epoch, step, shard].to(ctx.device)).backward()
            first.optimizer.step()
            first.optimizer.zero_grad()
        first.epoch_end({"val_loss": VAL[epoch]})
    model = _sharded_model(ctx, seed + 100)
    fts = _tuning_scheduler(model, os.path.join(ckpt_dir, "first"), True)
    fts.load(saved)
    path.pop()  # the boundary epoch is run again by the resumed tuning scheduler
    path = _train(fts, model, x, y, shard, ctx.device, epochs, path=path)
    return {"path": path, "params": _params(model)}


class _Position:
    def __init__(self, n):
        self.n = n

    def state_dict(self):
        return {"n": self.n}

    def load_state_dict(self, state):
        self.n = state["n"]


def per_rank_state(ctx, ckpt_dir: str, sharded: bool, seed: int = 7) -> dict:
    """Registered state and random streams are per rank, and each rank gets its own back on load."""
    model = _sharded_model(ctx, seed) if sharded else _model(seed).to(ctx.device)
    fts = _tuning_scheduler(model, os.path.join(ckpt_dir, "first"), True)
    position = _Position(100 + ctx.rank)
    fts.register_stateful("loader", position)
    torch.manual_seed(1000 + ctx.rank)
    saved = fts.save(os.path.join(ckpt_dir, "state" if sharded else "state.pt"))
    expected = torch.rand(3).tolist()
    model2 = _sharded_model(ctx, seed) if sharded else _model(seed).to(ctx.device)
    fts2 = _tuning_scheduler(model2, os.path.join(ckpt_dir, "first"), True)
    position2 = _Position(-1)
    fts2.register_stateful("loader", position2)
    fts2.load(saved)
    return {"rank": ctx.rank, "position": position2.n, "rng_matches": torch.rand(3).tolist() == expected}


def resave_and_partial_metrics(ctx, ckpt_dir: str, epochs: int = 4, steps: int = 2, batch: int = 8,
                               seed: int = 7) -> dict:
    """Saving a sharded checkpoint again under the same directory spelled with a trailing slash replaces it; and a
    rank that reports no metrics for an epoch still joins the best-checkpoint agreement."""
    model = _sharded_model(ctx, seed)
    # a lenient plateau policy, so the rank without metrics reaches the checkpoint agreement this case is about
    fts = TuningScheduler(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9),
                          transition=PlateauTransition("val_loss", patience=1, strict=False),
                          interface=TorchInterface(model),
                          checkpoint=BestCheckpoint(os.path.join(ckpt_dir, "best"), "val_loss"))
    fts.setup()
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    for epoch in fts.epochs(epochs):
        for step in range(steps):
            F.mse_loss(model(x[epoch, step, shard].to(ctx.device)), y[epoch, step, shard].to(ctx.device)).backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
        fts.epoch_end({"val_loss": VAL[epoch]} if ctx.rank == 0 else None)
    target = os.path.join(ckpt_dir, "resave")
    fts.save(target)
    fts.save(target + os.sep)
    leftovers = sorted(n for n in os.listdir(ckpt_dir) if n.startswith("resave."))
    return {"best_depth": fts._checkpoint.best_depth, "resaved": os.path.isfile(os.path.join(target, "fts_meta.pt")),
            "leftovers": leftovers}
