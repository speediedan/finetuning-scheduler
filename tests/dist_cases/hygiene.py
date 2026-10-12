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
"""Cases that exercise the pool's own validity rules, including deliberate violations used as negative controls."""
import os
import sys
import time

import pytest
import torch
import torch.distributed as dist

_STASH: list = []  # module state survives between cases in a worker, which is what the leak cases need


def identity(ctx) -> dict:
    lightning = [m for m in sys.modules if m.split(".")[0] in ("lightning", "pytorch_lightning")]
    return {"rank": ctx.rank, "device": ctx.device.type, "pid": os.getpid(), "lightning_modules": len(lightning)}


def draw(ctx, n: int = 3) -> list:
    return torch.rand(n).tolist()


def set_env(ctx, key: str, value: str) -> None:
    os.environ[key] = value


def read_env(ctx, key: str) -> str | None:
    return os.environ.get(key)


def set_flags(ctx) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)
    torch.set_float32_matmul_precision("medium")
    torch.set_grad_enabled(False)


def read_flags(ctx) -> list:
    return [
        torch.are_deterministic_algorithms_enabled(),
        str(torch.get_default_dtype()),
        torch.get_float32_matmul_precision(),
        torch.is_grad_enabled(),
    ]


def _square(x):
    return x * x


def _compiled_entries() -> int:
    return len(torch._C._dynamo.eval_frame._debug_get_cache_entry_list(_square.__code__))


def compile_square(ctx) -> int:
    torch.compile(_square, backend="eager")(torch.ones(2))
    return _compiled_entries()


def compiled_entries(ctx) -> int:
    return _compiled_entries()


def mesh_id(ctx, shape: tuple | None = None) -> int:
    return id(ctx.mesh(shape))


def call_init(ctx) -> None:
    dist.init_process_group("gloo")


def call_destroy(ctx) -> None:
    dist.destroy_process_group()


def leak_group(ctx) -> None:
    _STASH.append(dist.new_group(list(range(ctx.world_size))))


def leak_device_memory(ctx) -> None:
    _STASH.append(torch.ones(1024, 1024, device=ctx.device))


def raise_on(ctx, rank: int) -> None:
    if ctx.rank == rank:
        raise AssertionError(f"deliberate failure on rank {rank}")
    dist.barrier()  # the other ranks wait here for a rank that will never arrive


def skip_on(ctx, rank: int, fail_rank: int | None = None) -> None:
    if ctx.rank == rank:
        pytest.skip(f"deliberately skipped on rank {rank}")
    if ctx.rank == fail_rank:
        raise AssertionError(f"deliberate failure on rank {fail_rank}")


def skip_all(ctx) -> None:
    pytest.skip("deliberately skipped on every rank")


def sleep_on(ctx, rank: int, seconds: float) -> None:
    if ctx.rank == rank:
        time.sleep(seconds)


def noop(ctx) -> None:
    return None
