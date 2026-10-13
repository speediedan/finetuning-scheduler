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
"""Run a distributed pool case in this process, once per simulated rank, on PyTorch's fake process group.

The fake group's collectives communicate nothing, so a case's numerics are meaningless here: what it checks is the
logic that does not depend on them (which parameters each phase thaws, which modules are sharded and how, which
configurations are refused or warned about). That runs on every hosted OS, where the gloo pool (Linux only) cannot.

Only ``torch.testing._internal.distributed.fake_pg`` is imported from PyTorch's private testing package: unlike the
rest of it, that module imports nothing beyond ``torch.distributed`` and changes no global state other than
registering the ``fake`` backend. Like the cases, this module imports nothing from Lightning or ``tests.helpers``, so
the ``core-no-lightning`` CI job can run the tests that use it.
"""
import os
import random
from typing import Any
from collections.abc import Callable

import torch
import torch.distributed as dist

from tests.dist_cases._worker import CaseContext


def _fake_store() -> Any:
    from torch.testing._internal.distributed.fake_pg import FakeStore  # registers the ``fake`` backend

    return FakeStore()


def run_fake(case: Callable, world_size: int = 2, ranks: list[int] | None = None, seed: int = 1234,
             **kwargs: Any) -> list:
    """Run ``case(ctx, **kwargs)`` once for each simulated rank (every rank by default), each in its own fake
    process group, and return the results in rank order.

    As in the pool, the case may not create or destroy the default process group, and the environment it leaves is
    restored.
    """
    if dist.is_initialized():
        raise RuntimeError("a process group is already initialized in this process")
    results = []
    for rank in range(world_size) if ranks is None else ranks:
        env = dict(os.environ)
        dist.init_process_group("fake", store=_fake_store(), rank=rank, world_size=world_size)
        init, destroy = dist.init_process_group, dist.destroy_process_group
        try:
            random.seed(seed)
            torch.manual_seed(seed)

            def _forbidden(*args: Any, **kw: Any) -> None:
                raise AssertionError("the code under test must not create or destroy the default process group")

            dist.init_process_group = dist.destroy_process_group = _forbidden  # type: ignore[assignment]
            results.append(case(CaseContext(rank, world_size, "fake", torch.device("cpu")), **kwargs))
        finally:
            dist.init_process_group, dist.destroy_process_group = init, destroy  # type: ignore[assignment]
            destroy()
            os.environ.clear()
            os.environ.update(env)
    return results


def assert_ranks_agree(values: list) -> Any:
    """Assert every simulated rank returned the same value, and return it (as the pool's helper of that name does,
    without importing ``tests.helpers``)."""
    for rank, value in enumerate(values[1:], start=1):
        assert value == values[0], f"rank {rank} disagrees with rank 0: {value!r} != {values[0]!r}"
    return values[0]
