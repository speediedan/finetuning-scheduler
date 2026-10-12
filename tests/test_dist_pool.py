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
"""The persistent distributed test pool: its numerics on real collectives, and a negative control for each of its
validity rules (each deliberately breaks a rule and asserts the pool reports it)."""
import time

import pytest
import torch

from tests.dist_cases import collectives, fsdp2, hygiene
from tests.helpers.dist_pool import DistPool, assert_ranks_agree, case_ref, dist_pool_fixture
from tests.helpers.runif import RunIf

gloo_pool = dist_pool_fixture(world_size=2, backend="gloo")
nccl_pool = dist_pool_fixture(world_size=2, backend="nccl")

_GLOO = RunIf(skip_windows=True, skip_mac_os=True)  # multi-process gloo runs on Linux only, as upstream's tests do
_NCCL = RunIf(min_cuda_gpus=2)


@pytest.fixture(params=[pytest.param("gloo", marks=_GLOO), pytest.param("nccl", marks=_NCCL)])
def pool(request):
    return request.getfixturevalue(f"{request.param}_pool")


def test_workers_are_reused_and_lightning_free(pool):
    first, second = pool.run(hygiene.identity), pool.run(hygiene.identity)
    assert [r["rank"] for r in first] == [0, 1]
    assert [r["pid"] for r in first] == [r["pid"] for r in second], "the pool respawned between clean cases"
    assert {r["device"] for r in first} == {"cuda" if pool.backend == "nccl" else "cpu"}
    assert all(r["lightning_modules"] == 0 for r in first)


def test_collectives(pool):
    assert pool.run(collectives.all_reduce_sum, n=4) == [[0, 3, 6, 9]] * 2
    assert pool.run(collectives.all_gather_ranks) == [[0, 1]] * 2


def test_every_case_starts_from_the_same_seed(pool):
    first, second = pool.run(hygiene.draw), pool.run(hygiene.draw)
    torch.manual_seed(1234)
    assert first == second == [torch.rand(3).tolist()] * 2


def test_mesh_is_created_once(pool):
    assert_ranks_agree([len(set(ids)) for ids in zip(pool.run(hygiene.mesh_id), pool.run(hygiene.mesh_id))]) == 1


def test_fsdp2_thaw_schedule_matches_single_process(pool):
    got = assert_ranks_agree(pool.run(fsdp2.thaw_schedule))
    want = fsdp2.reference()
    assert got["losses"] == pytest.approx(want["losses"], rel=1e-5, abs=1e-6)
    assert got["params"].keys() == want["params"].keys()
    for name, values in want["params"].items():
        assert got["params"][name] == pytest.approx(values, rel=1e-5, abs=1e-6), name
    # the thawed layer trained only after the transition, and the never-thawed one not at all
    assert fsdp2.reference(steps=3)["params"]["2.weight"] == fsdp2.reference(steps=0)["params"]["2.weight"]
    assert got["params"]["0.weight"] == fsdp2.reference(steps=0)["params"]["0.weight"]
    assert got["params"]["2.weight"] != fsdp2.reference(steps=3)["params"]["2.weight"]


@_NCCL
def test_leaked_device_memory_is_a_violation(nccl_pool):
    with pytest.raises(pytest.fail.Exception, match="bytes of device memory still allocated"):
        nccl_pool.run(hygiene.leak_device_memory)
    assert nccl_pool.run(hygiene.noop) == [None, None]


# Negative controls for the backend-independent rules, on gloo only: each one that fails a case respawns the pool.


@_GLOO
def test_env_and_flags_are_restored(gloo_pool):
    gloo_pool.run(hygiene.set_env, key="FTS_DIST_POOL_LEAK", value="1")
    assert gloo_pool.run(hygiene.read_env, key="FTS_DIST_POOL_LEAK") == [None, None]
    defaults = gloo_pool.run(hygiene.read_flags)
    gloo_pool.run(hygiene.set_flags)
    assert gloo_pool.run(hygiene.read_flags) == defaults


@_GLOO
def test_dynamo_is_reset_between_cases(gloo_pool):
    assert gloo_pool.run(hygiene.compile_square) == [1, 1]
    assert gloo_pool.run(hygiene.compiled_entries) == [0, 0]


@_GLOO
@pytest.mark.parametrize("case", [hygiene.call_init, hygiene.call_destroy], ids=lambda c: c.__name__)
def test_owning_the_process_group_is_forbidden(gloo_pool, case):
    gloo_pool.run(hygiene.noop)  # make sure the workers exist, so the respawn below is observable
    before = gloo_pool.pids()
    with pytest.raises(pytest.fail.Exception, match="the test pool owns the process group"):
        gloo_pool.run(case)
    gloo_pool.run(hygiene.noop)
    assert set(gloo_pool.pids()).isdisjoint(before), "a failed case must respawn the pool"


@_GLOO
def test_a_leaked_process_group_is_a_violation(gloo_pool):
    with pytest.raises(pytest.fail.Exception, match=r"process group\(s\) created by the case"):
        gloo_pool.run(hygiene.leak_group)


@_GLOO
def test_a_failure_on_one_rank_beats_a_skip_on_another(gloo_pool):
    with pytest.raises(pytest.fail.Exception, match="deliberate failure on rank 1"):
        gloo_pool.run(hygiene.skip_on, rank=0, fail_rank=1)


@_GLOO
def test_ranks_that_disagree_about_skipping_fail(gloo_pool):
    with pytest.raises(pytest.fail.Exception, match="disagreed about skipping"):
        gloo_pool.run(hygiene.skip_on, rank=0)


@_GLOO
def test_a_skip_on_every_rank_skips(gloo_pool):
    with pytest.raises(pytest.skip.Exception, match="deliberately skipped on every rank"):
        gloo_pool.run(hygiene.skip_all)


@_GLOO
def test_an_error_on_one_rank_does_not_wait_for_the_timeout(gloo_pool):
    start = time.monotonic()
    with pytest.raises(pytest.fail.Exception, match="deliberate failure on rank 1"):
        gloo_pool.run(hygiene.raise_on, rank=1, timeout=60)
    assert time.monotonic() - start < 30, "the pool waited for the case timeout instead of its grace period"


@_GLOO
def test_a_hung_rank_times_out_and_the_pool_respawns(gloo_pool):
    with pytest.raises(pytest.fail.Exception, match="no reply before the deadline"):
        gloo_pool.run(hygiene.sleep_on, rank=1, seconds=60, timeout=3)
    assert gloo_pool.run(collectives.all_gather_ranks) == [[0, 1]] * 2


def _measured_pool_run(tmp_path, *cases) -> list:
    """Run ``(case, kwargs)`` pairs in a fresh pool under a nested coverage of ``collectives``, close the pool, and
    return the lines of ``collectives`` this process's coverage holds afterwards."""
    coverage = pytest.importorskip("coverage")
    # config_file=False: a run's own config (`parallel`, `source`) would otherwise change what this coverage measures
    cov = coverage.Coverage(
        data_file=str(tmp_path / "parent.coverage"), include=[collectives.__file__], config_file=False
    )
    cov.start()
    try:
        pool = DistPool(2, "gloo", tmp_path)
        try:
            for case, kwargs in cases:
                try:
                    pool.run(case, **kwargs)
                except pytest.fail.Exception:
                    pass
        finally:
            pool.close()
    finally:
        cov.stop()
    assert not list((tmp_path / "coverage").iterdir()), "the workers' data files were not merged and removed"
    return cov.get_data().lines(collectives.__file__) or []


@_GLOO
def test_workers_are_measured_when_this_process_is(tmp_path, monkeypatch):
    """Under coverage, the lines only the workers execute (every case's body) reach this process's data."""
    monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)  # its own route would send the data elsewhere
    body = collectives.all_reduce_sum.__code__.co_firstlineno + 1
    lines = _measured_pool_run(tmp_path, (collectives.all_reduce_sum, {"n": 4}))
    assert body in lines, "the case body, run only by the workers, was not measured"


@_GLOO
def test_a_failed_case_keeps_the_coverage_of_earlier_ones(tmp_path, monkeypatch):
    """A case that fails on a rank while every rank still replies retires the workers cleanly, so what they
    measured for earlier cases is kept rather than lost with a killed process."""
    monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)
    body = collectives.all_reduce_sum.__code__.co_firstlineno + 1
    lines = _measured_pool_run(
        tmp_path, (collectives.all_reduce_sum, {"n": 4}), (hygiene.skip_on, {"rank": 0, "fail_rank": 1})
    )
    assert body in lines, "the earlier case's coverage was lost when the failed case retired the workers"


def test_cases_must_live_in_the_case_package():
    with pytest.raises(ValueError, match="pool cases must live in tests.dist_cases"):
        case_ref(test_collectives)
