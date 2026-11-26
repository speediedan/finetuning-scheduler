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
"""The profiling core's FSDP2 memory tracker on steps that leave a sharded unit unused (issue 246)."""
from tests.dist_cases import fsdp2_mem_tracker
from tests.helpers.dist_pool import assert_ranks_agree, dist_pool_fixture
from tests.helpers.runif import RunIf

gloo_pool = dist_pool_fixture(world_size=2, backend="gloo")
GLOO = RunIf(skip_windows=True, skip_mac_os=True)  # multi-process gloo runs on Linux only, as upstream's tests do


@GLOO
def test_a_step_that_leaves_a_unit_unused_is_tracked(gloo_pool):
    got = assert_ranks_agree(gloo_pool.run(fsdp2_mem_tracker.unused_unit_step))
    assert got["error"] is None
    # every step's used units are tracked through post-backward; the unused one, never run, is not tracked at all
    assert got["post_backward_snapshots"] == [{"used": 1, "unused": None, "root": 1}] * 2


@GLOO
def test_upstream_tracker_fails_on_the_same_step(gloo_pool):
    """The control: upstream's tracker reads ``_unsharded_param`` of the unused unit's parameters.

    When a torch release fixes that, this fails, and the profiling core's tracker can be retired.
    """
    got = assert_ranks_agree(gloo_pool.run(fsdp2_mem_tracker.unused_unit_step, upstream=True))
    assert got["error"] is not None and "_unsharded_param" in got["error"]
