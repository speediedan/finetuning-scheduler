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
"""A self-contained distributed smoke check for vetting a multi-GPU host, ROCm in particular, before relying on it.

It needs only PyTorch: no Lightning, no finetuning-scheduler install and no test suite, so it can run on a rented
instance. It exercises what the direct-PyTorch distributed path depends on:

- ``collectives``: all-reduce, all-gather, reduce-scatter and broadcast at 1 KB to 64 MB, checked exactly, plus the
  all-reduce bus bandwidth;
- ``soak``: repeated checked all-reduces cycling small and large messages, which catch intermittent hangs and
  corruption (small messages matter: RCCL picks a different protocol for them);
- ``fsdp2_thaw``: a two-phase schedule on FSDP2 (train the head, then thaw a middle layer and add it to the optimizer),
  compared with the same schedule run in one process;
- ``fsdp2_bf16``: the same schedule under a bf16 ``MixedPrecisionPolicy``;
- ``dcp_restore_best``: save a sharded checkpoint with DCP, train on, restore it, and check parameters and optimizer
  state come back exactly;
- ``compile_phase``: compile the model at the phase transition and compare with eager.

Each variant runs every check in fresh processes. By default the GPU backend runs twice, once as configured and once
with ``NCCL_PROTO=Simple``, the workaround for RCCL deadlocking in small collectives on RDNA4 GPUs, where it selects
its LL protocol automatically (ROCm/rocm-systems#5480).

Usage::

    python scripts/rocm_smoke.py                              # every check, both variants
    python scripts/rocm_smoke.py --checks soak --soak-iters 2000 --variants default
    python scripts/rocm_smoke.py --backend gloo               # CPU only: checks the script, not the GPUs

The exit code is 0 only if every check passes in every variant. ``--report`` also writes the results as JSON.
"""
import argparse
import json
import multiprocessing
import os
import platform
import queue
import sys
import tempfile
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

SIZES = (256, 256 * 1024, 16 * 1024 * 1024)  # elements: 1 KB, 1 MB and 64 MB at float32
VARIANTS = {"default": {}, "proto_simple": {"NCCL_PROTO": "Simple"}}
SEED = 7


class Ctx:
    def __init__(self, rank: int, world_size: int, device: torch.device, workdir: str, args: argparse.Namespace):
        self.rank, self.world_size, self.device, self.workdir, self.args = rank, world_size, device, workdir, args

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _model() -> nn.Sequential:
    torch.manual_seed(SEED)
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))


def _data(steps: int, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(steps, batch, 32, generator=gen), torch.randn(steps, batch, 8, generator=gen)


def _sharded(ctx: Ctx, mp_policy=None) -> nn.Sequential:
    from torch.distributed.fsdp import fully_shard

    model = _model().to(ctx.device)
    kwargs = {"mp_policy": mp_policy} if mp_policy is not None else {}
    for layer in (model[0], model[2], model[4]):
        fully_shard(layer, **kwargs)
    return fully_shard(model, **kwargs)


def _schedule(model, x, y, steps, thaw_at, lr=0.1, momentum=0.0, compile_at=None, reduce=None):
    """Train the head, then thaw ``model[2]`` at ``thaw_at`` (optionally compiling the layers at
    ``compile_at``)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model[4].requires_grad_(True)
    opt = torch.optim.SGD(model[4].parameters(), lr=lr, momentum=momentum)
    losses = []
    for step in range(steps):
        if step == thaw_at:
            model[2].requires_grad_(True)
            opt.add_param_group({"params": list(model[2].parameters())})
        if step == compile_at:
            # Compile each sharded layer's forward, not the module: FSDP2's hooks are excluded from compilation, and
            # a hook inside a compiled region sends the whole frame back to eager, so compiling the root module or
            # `layer.compile()` captures nothing. Compiling `forward` alone keeps the hooks outside compiled code.
            for layer in (model[0], model[2], model[4]):
                layer.forward = torch.compile(layer.forward)
        loss = F.mse_loss(model(x[step]), y[step])
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(reduce(loss) if reduce else loss.item())
    return losses, opt


def _shard_batch(ctx: Ctx, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    per_rank = x.shape[1] // ctx.world_size
    rows = slice(ctx.rank * per_rank, (ctx.rank + 1) * per_rank)
    return x[:, rows].to(ctx.device), y[:, rows].to(ctx.device)


def _global_loss(ctx: Ctx):
    def reduce(loss: torch.Tensor) -> float:
        total = loss.detach().clone()
        dist.all_reduce(total)
        return total.item() / ctx.world_size

    return reduce


def _full_params(model) -> dict:
    return {name: p.full_tensor().detach().cpu().clone() for name, p in model.named_parameters()}


def check_collectives(ctx: Ctx) -> dict:
    ws, dev = ctx.world_size, ctx.device
    dtypes = (torch.float32, torch.bfloat16) if dev.type == "cuda" else (torch.float32,)
    for dtype in dtypes:
        for n in SIZES:
            t = torch.full((n,), float(ctx.rank + 1), dtype=dtype, device=dev)
            dist.all_reduce(t)
            assert torch.all(t == ws * (ws + 1) / 2), f"all_reduce wrong at {n} x {dtype}"
            gathered = torch.empty(n * ws, dtype=dtype, device=dev)
            dist.all_gather_into_tensor(gathered, torch.full((n,), float(ctx.rank), dtype=dtype, device=dev))
            want = torch.arange(ws, dtype=dtype, device=dev).repeat_interleave(n)
            assert torch.equal(gathered, want), f"all_gather wrong at {n} x {dtype}"
            scattered = torch.empty(n, dtype=dtype, device=dev)
            dist.reduce_scatter_tensor(scattered, torch.ones(n * ws, dtype=dtype, device=dev))
            assert torch.all(scattered == ws), f"reduce_scatter wrong at {n} x {dtype}"
            b = torch.full((n,), float(ctx.rank + 1), dtype=dtype, device=dev)
            dist.broadcast(b, src=0)
            assert torch.all(b == 1), f"broadcast wrong at {n} x {dtype}"
    t, iters = torch.ones(SIZES[-1], device=dev), 10
    for _ in range(3):  # warm up
        dist.all_reduce(t)
    ctx.sync()
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t)
    ctx.sync()
    seconds = (time.perf_counter() - start) / iters
    bus_gbps = 2 * (ws - 1) / ws * t.numel() * t.element_size() / seconds / 1e9
    return {"all_reduce_64MB_ms": round(seconds * 1e3, 2), "bus_bandwidth_GBps": round(bus_gbps, 2)}


def check_soak(ctx: Ctx) -> dict:
    ws, start = ctx.world_size, time.perf_counter()
    sizes = (256, 16 * 1024, 8 * 1024 * 1024)  # 1 KB and 64 KB (small-message protocols) and 32 MB, in turn
    for i in range(ctx.args.soak_iters):
        t = torch.full((sizes[i % len(sizes)],), float(i % 7 + ctx.rank), device=ctx.device)
        dist.all_reduce(t)
        assert torch.all(t == ws * (i % 7) + ws * (ws - 1) / 2), f"all_reduce wrong at iteration {i} ({t.numel()})"
    return {"iterations": ctx.args.soak_iters, "seconds": round(time.perf_counter() - start, 2)}


def check_fsdp2_thaw(ctx: Ctx) -> dict:
    steps, thaw_at, batch = 6, 3, 8 * ctx.world_size
    x, y = _data(steps, batch)
    ref = _model()
    ref_losses, _ = _schedule(ref, x, y, steps, thaw_at)
    model = _sharded(ctx)
    losses, _ = _schedule(model, *_shard_batch(ctx, x, y), steps, thaw_at, reduce=_global_loss(ctx))
    got, want = _full_params(model), {n: p.detach() for n, p in ref.named_parameters()}
    diff = max((got[n] - want[n]).abs().max().item() for n in want)
    assert diff < 1e-4, f"FSDP2 parameters differ from the single-process run by {diff:.3g}"
    assert torch.equal(got["0.weight"], _model()[0].weight.detach()), "a never-thawed layer changed"
    return {"max_param_diff": diff, "max_loss_diff": max(abs(a - b) for a, b in zip(losses, ref_losses))}


def check_fsdp2_bf16(ctx: Ctx) -> dict:
    from torch.distributed.fsdp import MixedPrecisionPolicy

    steps, batch = 20, 8 * ctx.world_size
    x, y = _data(1, batch)
    x, y = x.expand(steps, -1, -1), y.expand(steps, -1, -1)  # one batch repeated, so the loss must fall
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    model = _sharded(ctx, policy)
    losses, _ = _schedule(model, *_shard_batch(ctx, x, y), steps, thaw_at=10, lr=0.05, reduce=_global_loss(ctx))
    assert all(map(lambda v: v == v and abs(v) != float("inf"), losses)), f"non-finite loss: {losses}"
    assert losses[-1] < losses[0], f"bf16 training did not reduce the loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
    return {"first_loss": round(losses[0], 4), "last_loss": round(losses[-1], 4)}


def check_dcp_restore_best(ctx: Ctx) -> dict:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    steps, batch = 7, 8 * ctx.world_size
    x, y = _shard_batch(ctx, *_data(steps, batch))
    model = _sharded(ctx)
    _, opt = _schedule(model, x[:4], y[:4], 4, thaw_at=2, momentum=0.9)
    path = os.path.join(ctx.workdir, "best")
    model_sd, optim_sd = get_state_dict(model, opt)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=path)
    saved = _full_params(model)
    buffers = {i: s["momentum_buffer"].to_local().clone() for i, s in enumerate(opt.state.values())}
    for step in range(4, steps):  # train on past the saved state
        F.mse_loss(model(x[step]), y[step]).backward()
        opt.step()
        opt.zero_grad()
    assert any(not torch.equal(saved[n], p) for n, p in _full_params(model).items()), "training did not move"
    model_sd, optim_sd = get_state_dict(model, opt)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    restored = _full_params(model)
    assert all(torch.equal(saved[n], restored[n]) for n in saved), "restored parameters differ from the checkpoint"
    for i, s in enumerate(opt.state.values()):
        assert torch.equal(s["momentum_buffer"].to_local(), buffers[i]), "restored optimizer state differs"
    return {"params": len(saved), "momentum_buffers": len(buffers)}


def check_compile_phase(ctx: Ctx) -> dict:
    from torch._dynamo.utils import counters

    counters.clear()
    steps, thaw_at, batch = 6, 3, 8 * ctx.world_size
    x, y = _shard_batch(ctx, *_data(steps, batch))
    eager, compiled = _sharded(ctx), _sharded(ctx)
    eager_losses, _ = _schedule(eager, x, y, steps, thaw_at, reduce=_global_loss(ctx))
    compiled_losses, _ = _schedule(compiled, x, y, steps, thaw_at, compile_at=thaw_at, reduce=_global_loss(ctx))
    got, want = _full_params(compiled), _full_params(eager)
    diff = max((got[n] - want[n]).abs().max().item() for n in want)
    assert diff < 1e-4, f"compiled parameters differ from eager by {diff:.3g}"
    graphs = counters["stats"]["unique_graphs"]
    assert graphs > 0, f"nothing was compiled; graph breaks: {dict(counters['graph_break'])}"
    max_loss_diff = max(abs(a - b) for a, b in zip(compiled_losses, eager_losses))
    return {"graphs": graphs, "max_param_diff": diff, "max_loss_diff": max_loss_diff}


CHECKS = {
    "collectives": check_collectives,
    "soak": check_soak,
    "fsdp2_thaw": check_fsdp2_thaw,
    "fsdp2_bf16": check_fsdp2_bf16,
    "dcp_restore_best": check_dcp_restore_best,
    "compile_phase": check_compile_phase,
}


def _worker(rank: int, world_size: int, args: argparse.Namespace, workdir: str, results) -> None:
    device = torch.device("cpu")
    if args.backend != "gloo":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    dist.init_process_group(
        args.backend,
        init_method=f"file://{os.path.join(workdir, 'rendezvous')}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.timeout),
        device_id=device if device.type == "cuda" else None,
    )
    ctx = Ctx(rank, world_size, device, workdir, args)
    for name in args.checks:
        dist.barrier()
        start = time.perf_counter()
        try:
            ok, detail = True, CHECKS[name](ctx)
        except Exception:
            ok, detail = False, traceback.format_exc(limit=6)
        results.put((rank, name, ok, round(time.perf_counter() - start, 2), detail))
        if not ok:
            break  # the other ranks may be blocked in a collective; the parent's deadline ends them
    dist.destroy_process_group()


def run_variant(name: str, env: dict, args: argparse.Namespace) -> dict:
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)  # spawned workers inherit it, so it applies before their process group exists
    try:
        mp = multiprocessing.get_context("spawn")
        results, workdir = mp.Queue(), tempfile.mkdtemp(prefix=f"rocm_smoke_{name}_")
        procs = [mp.Process(target=_worker, args=(r, args.world_size, args, workdir, results)) for r in
                 range(args.world_size)]
        for p in procs:
            p.start()
        deadline, replies = time.monotonic() + args.variant_timeout, {}
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            try:
                rank, check, ok, seconds, detail = results.get(timeout=1)
                replies[(rank, check)] = (ok, seconds, detail)
            except queue.Empty:
                pass
        for p in procs:
            if p.is_alive():
                p.kill()
            p.join(5)
        while True:
            try:
                rank, check, ok, seconds, detail = results.get_nowait()
                replies[(rank, check)] = (ok, seconds, detail)
            except queue.Empty:
                break
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    report = {}
    for check in args.checks:
        ranks = [replies.get((r, check)) for r in range(args.world_size)]
        missing = [r for r, reply in enumerate(ranks) if reply is None]
        failed = {r: reply[2] for r, reply in enumerate(ranks) if reply is not None and not reply[0]}
        report[check] = {
            "ok": not missing and not failed,
            "seconds": max((reply[1] for reply in ranks if reply), default=None),
            "detail": failed or (f"no result from ranks {missing} (hung, crashed or not reached)" if missing else
                                 ranks[0][2]),
        }
    return report


def topology() -> dict:
    info = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
            "python": platform.python_version()}
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        props = [torch.cuda.get_device_properties(i) for i in range(n)]
        info["devices"] = [f"{p.name} ({p.total_memory / 2**30:.0f} GB)" for p in props]
        info["peer_access"] = {f"{i}->{j}": torch.cuda.can_device_access_peer(i, j)
                               for i in range(n) for j in range(n) if i != j}
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    prefixes = ("NCCL_", "RCCL_", "HSA_", "HIP_", "ROCR_", "CUDA_VISIBLE")
    info["env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}
    return info


def main() -> int:
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", default="nccl" if gpus >= 2 else "gloo", help="nccl (RCCL on ROCm) or gloo")
    parser.add_argument("--world-size", type=int, default=max(gpus, 2))
    parser.add_argument("--checks", default=",".join(CHECKS), help=f"comma-separated subset of {', '.join(CHECKS)}")
    parser.add_argument("--variants", default=None, help=f"comma-separated subset of {', '.join(VARIANTS)}")
    parser.add_argument("--soak-iters", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=300, help="process group timeout in seconds")
    parser.add_argument("--variant-timeout", type=int, default=1200, help="wall-clock budget per variant in seconds")
    parser.add_argument("--report", help="also write the results to this JSON file")
    args = parser.parse_args()
    args.checks = args.checks.split(",")
    if unknown := [c for c in args.checks if c not in CHECKS]:
        parser.error(f"unknown checks: {unknown}")
    variants = (args.variants or ("default,proto_simple" if args.backend != "gloo" else "default")).split(",")
    if args.backend != "gloo" and gpus < args.world_size:
        parser.error(f"--world-size {args.world_size} needs that many GPUs; {gpus} visible")

    results = {"topology": topology(), "backend": args.backend, "world_size": args.world_size, "variants": {}}
    print(json.dumps(results["topology"], indent=1))
    for variant in variants:
        results["variants"][variant] = report = run_variant(variant, VARIANTS[variant], args)
        for check, r in report.items():
            seconds = "-" if r["seconds"] is None else f"{r['seconds']}s"
            print(f"[{variant:>12}] {'PASS' if r['ok'] else 'FAIL'} {check:<17} {seconds:>8}  "
                  f"{r['detail'] if r['ok'] else ''}")
            if not r["ok"]:
                print(r["detail"] if isinstance(r["detail"], str) else "\n".join(map(str, r["detail"].values())))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=1, default=str)
    return 0 if all(r["ok"] for v in results["variants"].values() for r in v.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
