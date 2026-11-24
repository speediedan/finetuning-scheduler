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
"""Promotion on a sharded model (pool cases, gloo or NCCL): FSDP2 units, FSDP2's root unit alone, and tensor
parallelism.

Each layout must match one process: the same tensors thawed in each phase, the same losses and parameters, and the same
values at promotion wherever they are read back (a rollback, a resume, an export). The hidden size is odd, so the
promoted tensors' shards are uneven.
"""
import os
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import fully_shard

from finetuning_scheduler import CanonicalView, GateResult, TorchInterface, TuningScheduler
from finetuning_scheduler.core.checkpoint import BestCheckpoint, load_payload
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.promotion import full_value

ROLES = {"4.weight": "adapter:head", "4.bias": "adapter:head", "0.weight": "artifact", "0.bias": "artifact"}
PROBE = ["0.bias", "0.weight"]
SCHEDULE = {0: {"params": ["role:adapter"], "max_transition_epoch": 2},
            1: {"params": ["role:latent:probe"], "promote": {"role:artifact": "latent:probe"}, "lr": 0.05,
                "max_transition_epoch": 4},
            2: {"params": ["2.weight"], "lr": 0.05}}
EPOCHS, STEPS, BATCH, SEED = 6, 2, 8, 7
VAL = [1.0, 0.9, 1.2, 1.1, 0.8, 0.7]  # phase 1 never beats phase 0, so its restores put the promoted values back


def _model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 15), nn.ReLU(), nn.Linear(15, 15), nn.ReLU(), nn.Linear(15, 4))


def _placed(ctx, layout: str, seed: int = SEED, tie: bool = False) -> nn.Module:
    model = _model(seed).to(ctx.device)
    if tie:  # in one unit: FSDP2 unties parameters shared between units
        model[2].bias = model[0].bias
    if layout == "tp":
        from finetuning_scheduler.strategies.plan import parallelize_model

        mesh = ctx.mesh((ctx.world_size,), ("tensor_parallel",))["tensor_parallel"]
        return parallelize_model(model, mesh, {i: {"style": "colwise", "output_layouts": "replicate"}
                                               for i in ("0", "2", "4")})
    if layout == "units":
        for layer in (model[0], model[2], model[4]):
            fully_shard(layer, mesh=ctx.mesh())
    fully_shard(model, mesh=ctx.mesh())
    return model


def _rows(ctx, layout: str) -> slice:
    """This rank's share of each batch: every row under tensor parallelism, an equal shard under FSDP2."""
    if layout == "tp":
        return slice(None)
    return slice(ctx.rank * BATCH // ctx.world_size, (ctx.rank + 1) * BATCH // ctx.world_size)


def _tuning_scheduler(model, ckpt_dir, view=None, gate=None, **kwargs) -> TuningScheduler:
    fts = TuningScheduler(model, {k: dict(v) for k, v in SCHEDULE.items()},
                          optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9), interface=TorchInterface(model),
                          checkpoint=BestCheckpoint(ckpt_dir, "val_loss"),
                          view=view or CanonicalView(model, roles=ROLES), **kwargs)
    fts.add_phase_hook(partial(_shift_probe, model))
    if gate is not None:
        fts.add_gate(gate)
    fts.setup()
    return fts


def _shift_probe(model, event) -> None:
    """Change the probe when phase 1 starts, so its values at promotion are neither the built nor the trained
    ones."""
    if event.depth == 1:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in PROBE:
                    (p.to_local() if type(p).__name__ == "DTensor" else p).mul_(0.5)


def _veto_phase_1(event) -> GateResult:
    return GateResult(accept=event.depth != 1, reason="the probe phase hurt")


def _full(t) -> list:
    return (t.full_tensor() if type(t).__name__ == "DTensor" else t).detach().flatten().tolist()


def _params(model) -> dict:
    return {n: _full(p) for n, p in model.named_parameters()}


def _at_promotion(fts) -> dict:
    return {k: full_value(t, v).flatten().tolist() for k, (t, v) in fts._promoted_tensors().items()}


def _train(fts, model, rows, device, epochs=EPOCHS, mean=lambda loss: loss, stop_after: int | None = None) -> dict:
    """Train to ``epochs``: the tensors thawed and the depth at each epoch, and each step's loss (``mean`` averages
    it over the ranks' shares).

    ``stop_after`` leaves the loop after that epoch, before the run ends.
    """
    gen = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(EPOCHS, STEPS, BATCH, 8, generator=gen)
    y = torch.randn(EPOCHS, STEPS, BATCH, 4, generator=gen)
    thawed, losses = [], []
    for epoch in fts.epochs(epochs):
        thawed.append((fts.curr_depth, sorted(n for n, p in model.named_parameters() if p.requires_grad)))
        for step in range(STEPS):
            loss = F.mse_loss(model(x[epoch, step, rows].to(device)), y[epoch, step, rows].to(device))
            loss.backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
            losses.append(mean(loss.detach()).item())
        fts.epoch_end({"val_loss": VAL[epoch]})
        if epoch == stop_after:
            break
    return {"thawed": thawed, "losses": losses}


def _mean(ctx, layout: str):
    if layout == "tp":
        return lambda loss: loss

    def mean(loss):
        total = loss.clone()
        torch.distributed.all_reduce(total)
        return total / ctx.world_size
    return mean


def _run(model, ckpt_dir, rows, device, mean=lambda loss: loss, gate=None, restore_best=True) -> dict:
    fts = _tuning_scheduler(model, ckpt_dir, gate=gate, restore_best=restore_best)
    out = _train(fts, model, rows, device, mean=mean)
    return {**out, "params": _params(model), "at_promotion": _at_promotion(fts), "stop": fts.stop_reason,
            "role": fts.roles["0.weight"], "adapter": fts._distributed.name, "promotions": fts._promotions}


def promote(ctx, ckpt_dir: str, layout: str, veto: bool = False, restore_best: bool = True) -> dict:
    """Train through the promotion; with ``veto``, phase 1 is rolled back to a checkpoint saved before it."""
    return _run(_placed(ctx, layout), ckpt_dir, _rows(ctx, layout), ctx.device, _mean(ctx, layout),
                _veto_phase_1 if veto else None, restore_best)


def reference(ckpt_dir: str, veto: bool = False, restore_best: bool = True) -> dict:
    return _run(_model(SEED), ckpt_dir, slice(None), torch.device("cpu"), gate=_veto_phase_1 if veto else None,
                restore_best=restore_best)


def resume(ctx, ckpt_dir: str, layout: str, veto: bool = False) -> dict:
    """Save in the promoting phase, load into a fresh model with other weights and finish; with ``veto``, the
    resumed run rolls phase 1 back to the values at promotion the save holds."""
    gate = _veto_phase_1 if veto else None
    first_model = _placed(ctx, layout)
    first = _tuning_scheduler(first_model, os.path.join(ckpt_dir, "best"), gate=gate)
    _train(first, first_model, _rows(ctx, layout), ctx.device, stop_after=2)
    saved = first.save(os.path.join(ckpt_dir, "resume"))
    model = _placed(ctx, layout, SEED + 100)
    fts = _tuning_scheduler(model, os.path.join(ckpt_dir, "best"), gate=gate)
    fts.load(saved)
    out = _train(fts, model, _rows(ctx, layout), ctx.device, mean=_mean(ctx, layout))
    return {**out, "params": _params(model), "at_promotion": _at_promotion(fts), "saved": saved,
            "at_promotion_saved": _at_promotion(first), "stop": fts.stop_reason, "role": fts.roles["0.weight"]}


def resume_state(ctx, ckpt_dir: str, layout: str) -> dict:
    """A loop that checkpoints its own model: each rank's resume state carries its own part of the values at
    promotion, and a state saved under another layout is refused."""
    first_model = _placed(ctx, layout)
    first = _tuning_scheduler(first_model, os.path.join(ckpt_dir, str(ctx.rank)))
    _train(first, first_model, _rows(ctx, layout), ctx.device, stop_after=2)
    state = first.resume_state(rng=False)
    model = _placed(ctx, layout, SEED + 100)
    fts = _tuning_scheduler(model, os.path.join(ckpt_dir, str(ctx.rank)))
    fts.load_resume_state(state)
    plain = _model(SEED)
    unsharded = _tuning_scheduler(plain, os.path.join(ckpt_dir, f"plain{ctx.rank}"))
    try:
        unsharded.load_resume_state(state)
        refused = ""
    except FTSConfigurationError as err:
        refused = str(err)
    return {"at_promotion": _at_promotion(fts), "expected": _at_promotion(first), "refused": refused}


def export(ctx, ckpt_dir: str, layout: str) -> dict:
    """Export the promoted tensors from a best checkpoint saved before their promotion."""
    model = _placed(ctx, layout)
    fts = _tuning_scheduler(model, ckpt_dir, restore_best=False)
    _train(fts, model, _rows(ctx, layout), ctx.device, 4)
    return {"best_depth": fts._checkpoint.best_depth, "expected": _at_promotion(fts), "rank": ctx.rank,
            "live": _params(model), "exported": _export(fts)}


def _export(fts) -> dict | None:
    path = fts.export_best(roles=("latent", "adapter"))
    if not fts._interface.is_global_zero:
        return None
    return {n: t.flatten().tolist() for n, t in load_payload(path)["tensors"].items()}


def load_unsharded(saved: str, ckpt_dir: str) -> dict:
    """Load a sharded save made in the promoting phase into a single-process model."""
    model = _model(SEED + 100)
    fts = _tuning_scheduler(model, ckpt_dir)
    fts.load(saved)
    return {"params": _params(model), "at_promotion": _at_promotion(fts)}


class _Replaces(CanonicalView):
    def promote(self, physical, role):
        for n in physical:
            owner, _, attr = n.rpartition(".")
            module = self.root.get_submodule(owner)
            setattr(module, attr, nn.Parameter(getattr(module, attr).detach().clone(), requires_grad=False))
        return {n: n for n in physical}


class _Keeps(CanonicalView):
    def promote(self, physical, role):
        return {n: n for n in physical}


def own_promote(ctx, ckpt_dir: str, layout: str) -> dict:
    """A view that owns promotion keeps the parameters and trains, or replaces them and is refused."""
    out = {}
    for name, cls in (("keeps", _Keeps), ("replaces", _Replaces)):
        model = _placed(ctx, layout)
        fts = _tuning_scheduler(model, os.path.join(ckpt_dir, name), view=cls(model, roles=ROLES))
        try:
            out[name] = _train(fts, model, _rows(ctx, layout), ctx.device, mean=_mean(ctx, layout))
        except FTSConfigurationError as err:
            out[name] = str(err)
    return out


def tied_refused(ctx, ckpt_dir: str, layout: str) -> dict:
    """A promoted parameter another module shares is refused at setup."""
    try:
        _tuning_scheduler(_placed(ctx, layout, tie=True), ckpt_dir)
    except FTSConfigurationError as err:
        return {"refused": str(err)}
    return {"refused": ""}
