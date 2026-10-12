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
"""Worker side of the persistent distributed test pool.

Each worker initializes one process group, then serves cases until told to stop. Around every case it enforces
the pool's validity rules, so a case can neither inherit state from the one before it nor leave state for the next:

- a barrier, then the same seed on every rank;
- ``os.environ`` and the global torch flags snapshotted before the case and restored after it;
- ``torch._dynamo.reset()``, ``gc.collect()`` and (on CUDA) ``empty_cache()`` after it, then device memory must be
  back at its baseline;
- no process group may outlive the case, except those a mesh from :meth:`CaseContext.mesh` creates once for the
  pool's lifetime;
- ``init_process_group`` and ``destroy_process_group`` raise, which proves the code under test never owns the group.

A violation is reported like a failure, and the parent respawns the pool rather than trusting its state.
"""
import ctypes
import gc
import importlib
import multiprocessing
import os
import queue
import random
import signal
import sys
import time
import traceback
from datetime import timedelta
from typing import Any
from collections.abc import Callable

import torch
import torch.distributed as dist

_REAL: dict[str, Callable] = {}  # the real init and destroy functions, which only the pool itself may call
_CASES: dict[str, Callable] = {}


def _num_groups() -> int:
    return len(dist.distributed_c10d._world.pg_map)


class CaseContext:
    """What a case receives: its rank and device, and meshes cached for the pool's lifetime."""

    def __init__(self, rank: int, world_size: int, backend: str, device: torch.device) -> None:
        self.rank, self.world_size, self.backend, self.device = rank, world_size, backend, device
        self._meshes: dict[tuple, Any] = {}
        self.pg_baseline = _num_groups()
        self.mem_baseline = 0

    def mesh(self, shape: tuple[int, ...] | None = None, names: tuple[str, ...] | None = None) -> Any:
        """A ``DeviceMesh`` for ``shape`` (default: one dimension over the world), created once per pool."""
        key = (tuple(shape or (self.world_size,)), tuple(names) if names else None)
        if key not in self._meshes:
            from torch.distributed.device_mesh import init_device_mesh

            self._meshes[key] = init_device_mesh(self.device.type, key[0], mesh_dim_names=key[1])
            self.pg_baseline = _num_groups()  # groups a cached mesh creates belong to the pool, not the case
        return self._meshes[key]


def _forbidden(name: str) -> Callable:
    def _raise(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"code under test may not call {name}: the test pool owns the process group")

    return _raise


def _flags() -> tuple:
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.get_float32_matmul_precision(),
        torch.get_default_dtype(),
        torch.is_grad_enabled(),
    )


def _restore_flags(flags: tuple) -> None:
    torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[2], flags[3]
    torch.set_float32_matmul_precision(flags[4])
    torch.set_default_dtype(flags[5])
    torch.set_grad_enabled(flags[6])


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass


def _resolve(ref: str) -> Callable:
    if ref not in _CASES:
        module, qualname = ref.split(":")
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        _CASES[ref] = obj
    return _CASES[ref]


def _die_with_parent() -> None:
    """Have the kernel kill this worker if the parent dies, so an orphan can never keep a device (Linux only)."""
    if sys.platform.startswith("linux"):
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # 1 == PR_SET_PDEATHSIG


def _start_coverage() -> None:
    """Measure this worker when ``COVERAGE_PROCESS_START`` names a config, which must set ``parallel = true``.

    A no-op when coverage's own ``.pth`` hook already started it. Data is saved at exit, which a spawned worker
    reaches through ``sys.exit``, so ``coverage combine`` then merges every worker's file.
    """
    if os.environ.get("COVERAGE_PROCESS_START"):
        import coverage

        coverage.process_startup()


def _init(rank: int, world_size: int, backend: str, init_file: str, pg_timeout: float) -> CaseContext:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    device = torch.device("cpu")
    if backend == "nccl":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=pg_timeout),
        device_id=device if device.type == "cuda" else None,
    )
    warm = torch.ones(1, device=device)
    dist.all_reduce(warm)  # create the communicators now, outside every case's timing
    dist.barrier()
    del warm
    importlib.import_module("torch._dynamo")  # warm the import outside every case's timing

    _REAL.update(init=dist.init_process_group, destroy=dist.destroy_process_group)
    for module in (dist, dist.distributed_c10d):
        module.init_process_group = _forbidden("init_process_group")  # type: ignore[assignment]
        module.destroy_process_group = _forbidden("destroy_process_group")  # type: ignore[assignment]
    ctx = CaseContext(rank, world_size, backend, device)
    gc.collect()
    gc.freeze()  # see _after_case
    if device.type == "cuda":
        torch.cuda.empty_cache()
        ctx.mem_baseline = torch.cuda.memory_allocated(device)
    return ctx


def _after_case(ctx: CaseContext) -> list[str]:
    torch._dynamo.reset()
    gc.collect()
    problems = []
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)
        torch.cuda.empty_cache()
        if (leaked := torch.cuda.memory_allocated(ctx.device) - ctx.mem_baseline) > 0:
            problems.append(f"{leaked} bytes of device memory still allocated after the case")
    if (grown := _num_groups() - ctx.pg_baseline) > 0:
        problems.append(f"{grown} process group(s) created by the case and not released")
    # With torch loaded a full collection scans about 330k objects (about 110 ms, nearly all of a case's overhead).
    # Freezing what survived it limits the next collection to what the next case creates. Everything frozen is live,
    # so the checks above have already seen it.
    gc.freeze()
    return problems


def _run_case(ctx: CaseContext, ref: str, kwargs: dict, seed: int) -> tuple[str, Any, dict]:
    env, flags = dict(os.environ), _flags()
    start = time.perf_counter()
    dist.barrier()
    _seed(seed)
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(ctx.device)
    case_start = time.perf_counter()
    try:
        outcome, payload = "ok", _resolve(ref)(ctx, **kwargs)
    except BaseException as err:  # a case's failure is reported, never allowed to end the worker
        skipped = type(err).__name__ == "Skipped"  # pytest.skip(), detected without importing pytest here
        outcome, payload = ("skip", str(err)) if skipped else ("error", traceback.format_exc())
    case_end = time.perf_counter()
    peak = torch.cuda.max_memory_allocated(ctx.device) if ctx.device.type == "cuda" else None
    os.environ.clear()
    os.environ.update(env)
    _restore_flags(flags)
    if (problems := _after_case(ctx)) and outcome != "error":
        outcome, payload = "violation", "; ".join(problems)
    overhead = (case_start - start) + (time.perf_counter() - case_end)
    return outcome, payload, {"case_s": case_end - case_start, "overhead_s": overhead, "peak_bytes": peak}


def worker_main(
    rank: int,
    world_size: int,
    backend: str,
    init_file: str,
    requests: Any,
    responses: Any,
    pg_timeout: float,
) -> None:
    """Entry point of one pool worker (a spawned process)."""
    _die_with_parent()
    _start_coverage()
    started = time.perf_counter()
    try:
        ctx = _init(rank, world_size, backend, init_file, pg_timeout)
    except BaseException:
        responses.put((rank, "init_error", traceback.format_exc(), {}))
        return
    responses.put((rank, "ready", None, {"init_s": time.perf_counter() - started}))
    parent = multiprocessing.parent_process()
    while True:
        try:
            msg = requests.get(timeout=5)
        except queue.Empty:  # the death signal covers Linux; this covers the other platforms
            if parent is not None and not parent.is_alive():
                return
            continue
        if msg is None:
            break
        responses.put((rank, *_run_case(ctx, *msg)))
    gc.unfreeze()  # let shutdown collect what the cases froze; bindings report frozen objects as leaks otherwise
    _REAL["destroy"]()
