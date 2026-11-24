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
"""Promotion (``promote``) on a sharded model (#240), on two processes (gloo on CPU; NCCL on two GPUs): FSDP2
units, FSDP2's root unit alone and tensor parallelism, each against one process."""
import pytest

from tests.dist_cases import fts_promotion
from tests.helpers.dist_pool import assert_ranks_agree, dist_pool_fixture
from tests.helpers.runif import RunIf
from tests.test_tuning_scheduler_distributed import _close

gloo_pool = dist_pool_fixture(world_size=2, backend="gloo")
nccl_pool = dist_pool_fixture(world_size=2, backend="nccl")
GLOO = RunIf(skip_windows=True, skip_mac_os=True)
LAYOUTS = pytest.mark.parametrize("layout", ["units", "root", "tp"])
PROBE = fts_promotion.PROBE


def _matches(result: dict, ref: dict, tol: float = 1e-5) -> None:
    assert result["thawed"] == ref["thawed"]
    assert result["losses"] == pytest.approx(ref["losses"], abs=tol)
    assert _close(result["params"], ref["params"], tol=tol)
    assert _close(result["at_promotion"], ref["at_promotion"], tol=tol)
    assert result["role"] == ref["role"] == "latent:probe"


def _promotes_as_one_process(pool, tmp_path, layout, tol=1e-5, **kwargs) -> tuple[dict, dict]:
    results = pool.run(fts_promotion.promote, ckpt_dir=str(tmp_path / "sharded"), layout=layout, **kwargs)
    ref = fts_promotion.reference(ckpt_dir=str(tmp_path / "ref"), **kwargs)
    result = assert_ranks_agree(results)
    _matches(result, ref, tol)
    return result, ref


@GLOO
@LAYOUTS
@pytest.mark.parametrize("restore_best", [True, False], ids=["restore_best", "no_restore"])
def test_promotion_on_a_sharded_model_matches_one_process(gloo_pool, tmp_path, layout, restore_best):
    """The probe is thawed from its phase on, trains to the same values and losses, and (``restore_best``) goes
    back to its values at promotion when phase 2 restores a best saved before it."""
    result, ref = _promotes_as_one_process(gloo_pool, tmp_path, layout, restore_best=restore_best)
    assert result["adapter"] == ("dtensor" if layout == "tp" else "fsdp2") and ref["adapter"] == "unsharded"
    assert [PROBE[0] in names for _, names in result["thawed"]] == [False, False, True, True, True, True]
    assert result["promotions"] == [{"depth": 1, "names": PROBE, "from": "artifact", "to": "latent:probe",
                                     "materialized": []}]
    assert all(result["params"][n] != result["at_promotion"][n] for n in PROBE)


@RunIf(min_cuda_gpus=2)
@LAYOUTS
def test_promotion_on_a_sharded_model_matches_one_process_nccl(nccl_pool, tmp_path, layout):
    _promotes_as_one_process(nccl_pool, tmp_path, layout, tol=1e-4)


@GLOO
@LAYOUTS
def test_a_veto_rolls_back_to_the_values_at_promotion(gloo_pool, tmp_path, layout):
    """The checkpoint saved before the promotion loads, and the promoted tensors take their values at promotion."""
    result, _ = _promotes_as_one_process(gloo_pool, tmp_path, layout, veto=True)
    assert result["stop"].startswith("veto: phase 1")
    assert all(result["params"][n] == result["at_promotion"][n] for n in PROBE)


@GLOO
@LAYOUTS
@pytest.mark.parametrize("veto", [False, True], ids=["continues", "rolls_back"])
def test_a_resume_in_the_promoting_phase_continues_the_run(gloo_pool, tmp_path, layout, veto):
    """The values at promotion are saved by every rank and read back, so a resumed run (from other weights) ends as
    the uninterrupted one does, rollback included."""
    results = gloo_pool.run(fts_promotion.resume, ckpt_dir=str(tmp_path / "sharded"), layout=layout, veto=veto)
    result = assert_ranks_agree(results)
    ref = fts_promotion.reference(ckpt_dir=str(tmp_path / "ref"), veto=veto)
    assert result["at_promotion"] == result["at_promotion_saved"]
    assert _close(result["params"], ref["params"]) and _close(result["at_promotion"], ref["at_promotion"])
    assert result["role"] == "latent:probe"
    # a rollback at the end of phase 1 ends the run; otherwise the resumed epochs are the uninterrupted run's last ones
    assert result["thawed"] == ref["thawed"][3:] and result["stop"] == ref["stop"]
    if veto:
        assert all(result["params"][n] == result["at_promotion"][n] for n in PROBE)


@GLOO
@LAYOUTS
def test_a_sharded_save_loads_into_one_process_with_the_values_at_promotion(gloo_pool, tmp_path, layout):
    results = gloo_pool.run(fts_promotion.resume, ckpt_dir=str(tmp_path / "sharded"), layout=layout)
    loaded = fts_promotion.load_unsharded(results[0]["saved"], ckpt_dir=str(tmp_path / "plain"))
    assert loaded["at_promotion"] == results[0]["at_promotion_saved"]
    assert any(loaded["params"][n] != loaded["at_promotion"][n] for n in PROBE)  # saved after training on


@GLOO
@LAYOUTS
def test_resume_state_carries_each_ranks_part_of_the_values_at_promotion(gloo_pool, tmp_path, layout):
    for r in gloo_pool.run(fts_promotion.resume_state, ckpt_dir=str(tmp_path), layout=layout):
        assert r["at_promotion"] == r["expected"]
        assert "saved under another layout" in r["refused"]


@GLOO
@LAYOUTS
def test_exporting_a_best_saved_before_the_promotion_takes_the_values_at_promotion(gloo_pool, tmp_path, layout):
    results = gloo_pool.run(fts_promotion.export, ckpt_dir=str(tmp_path), layout=layout)
    zero = next(r for r in results if r["rank"] == 0)
    assert zero["best_depth"] == 0 and sorted(zero["exported"]) == ["0.bias", "0.weight", "4.bias", "4.weight"]
    assert {n: zero["exported"][n] for n in PROBE} == zero["expected"]
    assert all(zero["exported"][n] != zero["live"][n] for n in PROBE)
    assert all(r["exported"] is None for r in results if r["rank"] != 0)


@GLOO
@LAYOUTS
def test_a_view_that_owns_promotion_must_keep_the_parameters(gloo_pool, tmp_path, layout):
    result = assert_ranks_agree(gloo_pool.run(fts_promotion.own_promote, ckpt_dir=str(tmp_path), layout=layout))
    ref = fts_promotion.reference(ckpt_dir=str(tmp_path / "ref"))
    assert result["keeps"]["thawed"] == ref["thawed"]
    assert result["keeps"]["losses"] == pytest.approx(ref["losses"], abs=1e-5)
    assert "replaced the parameters ['0.bias', '0.weight'] of a sharded model" in result["replaces"]


@GLOO
def test_promoting_a_tied_parameter_of_a_sharded_model_is_refused(gloo_pool, tmp_path):
    for r in gloo_pool.run(fts_promotion.tied_refused, ckpt_dir=str(tmp_path), layout="root"):
        assert "tied parameters cannot be promoted" in r["refused"] and "'0.bias': ['2.bias']" in r["refused"]
