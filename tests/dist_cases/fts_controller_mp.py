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
"""``FTSController`` on a model-parallel toy transformer (pool cases, gloo or NCCL).

These mirror the Lightning callback's model-parallel tests (``tests/test_model_parallel.py``) on the same toy
transformer and phase schedule, with the schedule's names unprefixed (the controller is given the transformer itself).
Each rank trains on an equal shard of every batch, so FSDP2's gradient averaging makes a data-parallel run match a
single-process run on the full batch; under tensor parallelism every rank sees the whole batch.
"""
import math
import os
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel
from torch.nn.attention import SDPBackend

from finetuning_scheduler import FTSController, TorchHost
from finetuning_scheduler.core.checkpoint import BestCheckpoint
from tests.dist_cases._mp_models import FTSToyTransformer, gen_apply_transformer_tp_plan, TestModelArgs

# the Lightning tests' schedule (``model_parallel_ft_schedule``), with the ``model.`` prefix dropped and an ``lr`` for
# each later phase, so every phase's training moves its parameters well beyond float tolerance
SCHEDULE = {
    0: {"params": [r"output.weight", r"norm.*"], "max_transition_epoch": 1},
    1: {"params": [r"layers.1.(feed_forward|ffn_norm|attention.w.*|attention_norm).*"], "max_transition_epoch": 2,
        "lr": 0.1},
    2: {"params": [r"layers.0.(feed_forward|ffn_norm|attention.w.*|attention_norm).*",
                   r"(pos_embeddings|tok_embeddings).weight"], "lr": 0.1},
}
# the parameters whose state ``tests/model_parallel_expected_paths.py`` records each epoch
SAMPLED = ["layers.0.feed_forward.w2.weight", "layers.0.feed_forward.w2.bias", "layers.1.feed_forward.w2.weight",
           "layers.1.feed_forward.w2.bias", "norm.weight", "norm.bias", "output.weight"]
VOCAB, SEQ, DIM = 64, 16, 16


def _model(seed: int, device: torch.device, checkpoint_activations: bool = False,
           math_sdpa: bool = False) -> FTSToyTransformer:
    torch.manual_seed(seed)
    # DTensor has no sharding rule for the fused CPU attention kernels, so TP runs use the math backend (as the
    # Lightning tests' ``sdp_math_impl_tt`` does)
    sdpa = {"avail_sdp_backends": [SDPBackend.MATH], "sdp_use_implicit_replication": True} if math_sdpa else {}
    args = TestModelArgs(vocab_size=VOCAB, max_seq_len=SEQ, dim=DIM, n_heads=2,
                         checkpoint_activations=checkpoint_activations, **sdpa)
    return FTSToyTransformer(args).to(device)


def _data(epochs: int, steps: int, batch: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + 1)
    return torch.randint(0, VOCAB, (epochs, steps, batch, SEQ + 1), generator=gen)


def _loss(model: nn.Module, tokens: torch.Tensor, tp_loss_parallel: bool = False) -> torch.Tensor:
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    logits = model(inputs)
    if tp_loss_parallel:  # the logits stay sharded over the vocabulary
        with loss_parallel():
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
        return loss.detach()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
    loss.backward()
    return loss.detach()


def _controller(model, ckpt_dir=None, restore_best=False, callback=None):
    """``callback`` is ``(where, fn)``: a gate (``"gate"``) or a phase hook (``"start"`` or ``"end"``)."""
    checkpoint = BestCheckpoint(ckpt_dir, "val_loss") if ckpt_dir else None
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1), transition=None,
                        host=TorchHost(model), checkpoint=checkpoint, restore_best=restore_best and bool(ckpt_dir))
    if callback is not None:
        where, fn = callback
        if where == "gate":
            fts.add_gate(fn)
        else:
            fts.add_phase_hook(fn, when=where)
    fts.setup()
    return fts


def _state(model: nn.Module) -> dict:
    params = dict(model.named_parameters())
    return {"p_states": {n: {"requires_grad": params[n].requires_grad, "is_DTensor": isinstance(params[n], DTensor)}
                         for n in SAMPLED},
            "thawed": sum(p.requires_grad for p in model.parameters())}


def _fsdp_layout(model: nn.Module, planned: list[str], world_size: int) -> dict:
    """The planned modules are FSDP modules, and every sharded parameter holds its share of dimension 0."""
    modules = dict(model.named_modules())
    uneven = [n for n, p in model.named_parameters() if isinstance(p, DTensor)
              and p.to_local().shape[0] != math.ceil(p.shape[0] / world_size)]
    return {"fsdp_modules": sorted(n for n in planned if isinstance(modules[n], FSDPModule)), "uneven": uneven}


def _params(model: nn.Module) -> dict:
    return {n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().float().cpu().flatten().tolist()
            for n, p in model.named_parameters()}


def _unmoved(model: nn.Module, initial: dict, tol: float = 1e-4) -> list[str]:
    """The parameters training left within ``tol`` of their initial values (every one is thawed by the last
    phase)."""
    final = _params(model)
    return [n for n in final if max(abs(u - v) for u, v in zip(final[n], initial[n])) <= tol]


def _train(fts, model, data, shard, device, epochs, val=None, tp_loss_parallel=False, start_path=None,
           evaluate=False):
    path = list(start_path or [])
    for epoch in fts.epochs(epochs):
        path.append((epoch, fts.curr_depth, _state(model)))
        for step in range(data.shape[1]):
            _loss(model, data[epoch, step, shard].to(device), tp_loss_parallel)
            fts.optimizer.step()
            fts.optimizer.zero_grad()
        if evaluate:  # a forward-only pass before ``epoch_end``, as a loop computing its validation metric runs
            with torch.no_grad():
                model(data[epoch, 0, shard, :-1].to(device))
        fts.epoch_end({"val_loss": val[epoch]} if val else None)
    return path


def _fsdp_plan(model: nn.Module, mesh, shard_root: bool = True, mp_policy=None, root_only: bool = False) -> list[str]:
    """``fsdp_cm_only``'s plan (``output``, ``layers.1``, ``norm``) plus ``layers.0``, then the root (or the root
    alone)."""
    kwargs = {"mesh": mesh} | ({"mp_policy": mp_policy} if mp_policy is not None else {})
    planned = [] if root_only else ["layers.0", "layers.1", "norm", "output"]
    for name in planned:
        fully_shard(model.get_submodule(name), **kwargs)
    if shard_root:
        fully_shard(model, **kwargs)
        planned.append("")
    return planned


def _evaluate(model: nn.Module, tokens: torch.Tensor, device: torch.device) -> None:
    """A forward-only pass, as a validation metric, a gate's probe or a phase hook's baseline runs."""
    with torch.no_grad():
        model(tokens[:, :-1].to(device))


def _shard(ctx, batch):
    return slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)


def fsdp(ctx, epochs: int = 3, steps: int = 2, batch: int = 4, seed: int = 7, checkpoint_activations: bool = False,
         ckpt_dir: str | None = None, val: list | None = None, fp16: bool = False, evaluate: bool = False,
         callback_eval: str | None = None, root_only: bool = False) -> dict:
    """FSDP2 with a manual plan sharding the root last (T1), optionally with activation checkpointing (T2), a best-
    checkpoint policy restoring at boundaries (T3) or fp16 mixed precision (T4); with a forward-only pass before
    each ``epoch_end`` (``evaluate``) or inside a gate or a phase's ``start`` or ``end`` hook
    (``callback_eval``)."""
    model = _model(seed, ctx.device, checkpoint_activations)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32) if fp16 else None
    planned = _fsdp_plan(model, ctx.mesh(), mp_policy=mp_policy, root_only=root_only)
    initial = _params(model)
    unsharded_dtypes = set()
    if fp16:  # record the dtype each FSDP module computes in, after FSDP2's all-gather
        model.layers[0].register_forward_pre_hook(
            lambda mod, args: unsharded_dtypes.add(str(mod.attention.wq.weight.dtype)))
    data, shard = _data(epochs, steps, batch, seed), _shard(ctx, batch)
    callback = None if callback_eval is None else (
        callback_eval, lambda event: _evaluate(model, data[0, 0, shard], ctx.device) or True)
    fts = _controller(model, ckpt_dir, restore_best=ckpt_dir is not None, callback=callback)
    path = _train(fts, model, data, shard, ctx.device, epochs, val=val, evaluate=evaluate)
    return {"path": path, "layout": _fsdp_layout(model, planned, ctx.world_size), "params": _params(model),
            "unmoved": _unmoved(model, initial),
            "restored": fts.restored_count, "unsharded_dtypes": sorted(unsharded_dtypes),
            "sharded_best": bool(ckpt_dir) and os.path.isdir(os.path.join(ckpt_dir, "best.pt"))}


def reference(epochs: int = 3, steps: int = 2, batch: int = 4, seed: int = 7, ckpt_dir: str | None = None,
              val: list | None = None) -> dict:
    """The single-process, full-batch run the parallel cases are compared against."""
    model = _model(seed, torch.device("cpu"))
    initial = _params(model)
    fts = _controller(model, ckpt_dir, restore_best=ckpt_dir is not None)
    path = _train(fts, model, _data(epochs, steps, batch, seed), slice(None), torch.device("cpu"), epochs, val=val)
    return {"path": path, "params": _params(model), "restored": fts.restored_count, "unmoved": _unmoved(model, initial)}


def fsdp_resume(ctx, ckpt_dir: str, epochs: int = 3, split: int = 2, steps: int = 2, batch: int = 4, seed: int = 7,
                val: list | None = None, hook_eval: bool = False) -> dict:
    """Train ``split`` epochs, save (a DCP directory), load into a fresh sharded model and controller, finish.

    With
    ``hook_eval``, the fresh controller's phase-0 start hook runs a forward-only pass before the load.
    """
    data, shard = _data(epochs, steps, batch, seed), _shard(ctx, batch)
    first_model = _model(seed, ctx.device)
    _fsdp_plan(first_model, ctx.mesh())
    first = _controller(first_model, os.path.join(ckpt_dir, "best"), restore_best=True)
    path = _train(first, first_model, data, shard, ctx.device, split, val=val)
    saved = first.save(os.path.join(ckpt_dir, "resume"))
    model = _model(seed + 100, ctx.device)  # different initial weights: the load must overwrite them
    _fsdp_plan(model, ctx.mesh())
    hook = ("start", lambda event: _evaluate(model, data[0, 0, shard], ctx.device)) if hook_eval else None
    fts = _controller(model, os.path.join(ckpt_dir, "best"), restore_best=True, callback=hook)
    fts.load(saved)
    path = _train(fts, model, data, shard, ctx.device, epochs, val=val, start_path=path)
    return {"path": path, "params": _params(model), "saved_is_dir": os.path.isdir(saved)}


def tp(ctx, epochs: int = 3, steps: int = 2, batch: int = 4, seed: int = 7, tp_loss_parallel: bool = True,
       ckpt_dir: str | None = None, val: list | None = None) -> dict:
    """Tensor parallelism over every rank (T5): each rank sees the whole batch."""
    mesh = ctx.mesh((ctx.world_size,), ("tensor_parallel",))
    model = gen_apply_transformer_tp_plan(_model(seed, ctx.device, math_sdpa=True), mesh, tp_loss_parallel)
    initial = _params(model)
    fts = _controller(model, ckpt_dir, restore_best=ckpt_dir is not None)
    path = _train(fts, model, _data(epochs, steps, batch, seed), slice(None), ctx.device, epochs, val=val,
                  tp_loss_parallel=tp_loss_parallel)
    return {"path": path, "params": _params(model), "restored": fts.restored_count, "unmoved": _unmoved(model, initial)}


def partial_plan(ctx, epochs: int = 3, steps: int = 2, batch: int = 4, seed: int = 7) -> dict:
    """A plan that leaves the root unsharded (as ``fsdp_cm_only`` does).

    ``layers.1`` is then a root FSDP module whose parameters are all frozen in phase 0, so FSDP2 keeps its unsharded
    parameters registered after its forward; the controller must reshard before thawing them in phase 1. The
    embeddings stay plain tensors, whose gradients nothing synchronizes: the controller warns.
    """
    import warnings

    model = _model(seed, ctx.device)
    _fsdp_plan(model, ctx.mesh(), shard_root=False)
    initial = _params(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fts = _controller(model)
    path = _train(fts, model, _data(epochs, steps, batch, seed), _shard(ctx, batch), ctx.device, epochs)
    plain = {n: p.detach().float().cpu().flatten().tolist() for n, p in model.named_parameters()
             if not isinstance(p, DTensor)}
    return {"path": path, "plain": plain, "unmoved": _unmoved(model, initial),
            "warned": [str(w.message) for w in caught if "not synchronized" in str(w.message)]}


def refused_wrappers(ctx) -> dict:
    """``DistributedDataParallel``, on the root or a submodule, and the composable ``replicate()`` are refused."""
    from torch.distributed._composable.replicate import replicate

    from finetuning_scheduler.core.exceptions import FTSConfigurationError

    def attempt(model):
        try:
            _controller(model)
        except FTSConfigurationError as err:
            return str(err).split(",")[0]
        return None

    ddp_root = nn.parallel.DistributedDataParallel(_model(1, ctx.device))
    inner = _model(1, ctx.device)
    inner.layers[1] = nn.parallel.DistributedDataParallel(inner.layers[1])
    return {"ddp_root": attempt(ddp_root), "ddp_submodule": attempt(inner),
            "replicate": attempt(replicate(_model(1, ctx.device)))}


def warn_2d(ctx, seed: int = 7) -> dict:
    """A (data-parallel, tensor-parallel) mesh with the TP plan over the whole model and FSDP2 on the blocks only:

    the root's parameters are tensor-parallel DTensors that nothing averages across data-parallel ranks.
    """
    import warnings

    mesh = ctx.mesh((ctx.world_size, 1), ("data_parallel", "tensor_parallel"))
    model = gen_apply_transformer_tp_plan(_model(seed, ctx.device, math_sdpa=True), mesh, False)
    for name in ("layers.0", "layers.1", "norm", "output"):
        fully_shard(model.get_submodule(name), mesh=mesh["data_parallel"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _controller(model)
    return {"warned": [str(w.message) for w in caught if "not synchronized" in str(w.message)],
            "tok_embeddings_is_dtensor": isinstance(model.tok_embeddings.weight, DTensor)}
