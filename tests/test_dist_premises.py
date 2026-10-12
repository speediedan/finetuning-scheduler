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
"""Upstream distributed behavior that FTS's scheduling depends on or works around, checked on real collectives in
the persistent pool (gloo on Linux, NCCL with ``PL_RUN_CUDA_TESTS=1``)."""
import pytest

from tests.dist_cases import ddp
from tests.helpers.dist_pool import assert_ranks_agree, dist_pool_fixture
from tests.helpers.runif import RunIf

gloo_pool = dist_pool_fixture(world_size=2, backend="gloo")
nccl_pool = dist_pool_fixture(world_size=2, backend="nccl")


@pytest.fixture(
    params=[
        pytest.param("gloo", marks=RunIf(skip_windows=True, skip_mac_os=True)),
        pytest.param("nccl", marks=RunIf(min_cuda_gpus=2)),
    ]
)
def pool(request):
    return request.getfixturevalue(f"{request.param}_pool")


def test_ddp_ignores_parameters_thawed_after_the_wrap(pool):
    stale = pool.run(ddp.late_thaw, rebuild=False)  # per rank; the ranks diverge, so they disagree on "moved"
    rebuilt = assert_ranks_agree(pool.run(ddp.late_thaw, rebuild=True))
    assert all(r["moved"] > 0 for r in stale) and rebuilt["moved"] > 0, "the thawed layer did not train"
    stale = assert_ranks_agree([r["rank_spread"] for r in stale])  # gathered, so every rank reports the same spread
    # if this starts failing, DDP now tracks late requires_grad changes and rebuilding the wrapper may be unnecessary
    assert stale > 1e-3, "ranks agree on a layer DDP was expected not to synchronize"
    assert rebuilt["rank_spread"] == 0.0, "a rebuilt wrapper left the thawed layer unsynchronized"
