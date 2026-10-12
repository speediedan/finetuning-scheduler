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
"""The direct PyTorch controller on FSDP2 across two processes (gloo on CPU; NCCL on two GPUs) (#51)."""
import pytest

from tests.dist_cases import fts_controller
from tests.helpers.dist_pool import assert_ranks_agree, dist_pool_fixture
from tests.helpers.runif import RunIf

gloo_pool = dist_pool_fixture(world_size=2, backend="gloo")
nccl_pool = dist_pool_fixture(world_size=2, backend="nccl")
GLOO = RunIf(skip_windows=True, skip_mac_os=True)


def _close(a: dict, b: dict, tol: float = 1e-5) -> bool:
    return a.keys() == b.keys() and all(
        max(abs(u - v) for u, v in zip(a[k], b[k])) <= tol * (1 + max(abs(v) for v in b[k])) for k in a)


@GLOO
@pytest.mark.parametrize("restore_best", [False, True], ids=["no_restore", "restore_best"])
def test_fsdp2_matches_single_process(gloo_pool, tmp_path, restore_best):
    """Same phase path (and best-checkpoint restores, through DCP) as a single-process full-batch run; parameters
    equal to float tolerance."""
    results = gloo_pool.run(fts_controller.phases, ckpt_dir=str(tmp_path / "sharded"), restore_best=restore_best)
    sharded = assert_ranks_agree([{k: v for k, v in r.items() if k != "params"} for r in results])
    ref = fts_controller.reference(ckpt_dir=str(tmp_path / "ref"), restore_best=restore_best)
    assert sharded["path"] == ref["path"] and sharded["restored"] == ref["restored"]
    assert sharded["path"][-1][1] == 2  # every phase was reached
    assert sharded["sharded_best"]
    assert _close(results[0]["params"], ref["params"])


@GLOO
def test_fsdp2_save_load_resumes(gloo_pool, tmp_path):
    """A sharded save (a DCP directory) loaded into a fresh model and controller continues exactly."""
    resumed = gloo_pool.run(fts_controller.resume, ckpt_dir=str(tmp_path / "resume_run"))
    straight = gloo_pool.run(fts_controller.phases, ckpt_dir=str(tmp_path / "straight"))
    assert all(r["saved_is_dir"] for r in resumed)
    assert resumed[0]["path"] == straight[0]["path"]
    assert _close(resumed[0]["params"], straight[0]["params"], tol=1e-6)


@GLOO
def test_fsdp2_gate_veto_on_one_rank(gloo_pool, tmp_path):
    results = gloo_pool.run(fts_controller.veto_on_rank, ckpt_dir=str(tmp_path / "veto"))
    agreed = assert_ranks_agree([{k: v for k, v in r.items() if k != "params"} for r in results])
    assert agreed["depth"] == 1 and agreed["stop"].startswith("veto: phase 1")
    assert _close(results[0]["params"], results[1]["params"], tol=0.0)


@RunIf(min_cuda_gpus=2)
def test_fsdp2_matches_single_process_nccl(nccl_pool, tmp_path):
    results = nccl_pool.run(fts_controller.phases, ckpt_dir=str(tmp_path / "sharded"), restore_best=True)
    sharded = assert_ranks_agree([{k: v for k, v in r.items() if k != "params"} for r in results])
    ref = fts_controller.reference(ckpt_dir=str(tmp_path / "ref"), restore_best=True)
    assert sharded["path"] == ref["path"]
    assert _close(results[0]["params"], ref["params"], tol=1e-4)
