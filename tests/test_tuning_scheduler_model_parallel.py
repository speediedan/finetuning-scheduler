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
"""``TuningScheduler`` on a model-parallel toy transformer, mirroring the Lightning callback's model-parallel tests
(#87): the same phase path and parameter states, ranks in agreement, and parameters equal to a single-process
run."""
import pytest

from tests.dist_cases import fts_tuning_scheduler_mp as mp
from tests.helpers.dist_pool import assert_ranks_agree, dist_pool_fixture
from tests.helpers.runif import RunIf
from tests.model_parallel_expected_paths import path_fsdp_autocm, path_tp

gloo_pool = dist_pool_fixture(world_size=2, backend="gloo")
nccl_pool = dist_pool_fixture(world_size=2, backend="nccl")
GLOO = RunIf(skip_windows=True, skip_mac_os=True)
VAL = [1.0, 0.8, 0.9]  # the epoch-1 boundary restores epoch 0's best, the epoch-2 boundary epoch 1's


def _expected(path: dict) -> list:
    """An expected path's per-epoch parameter states, with the Lightning module's ``model.`` prefix dropped."""
    return [({n.removeprefix("model."): s for n, s in path[e][0]["p_states"].items()}, path[e][1])
            for e in sorted(path)]


def _states(path: list) -> list:
    return [(state["p_states"], state["thawed"]) for _, _, state in path]


def _phases(path: list) -> list:
    return [(epoch, depth, state["thawed"]) for epoch, depth, state in path]


_close = mp.params_close


def _agreed(results: list) -> dict:
    return assert_ranks_agree([{k: v for k, v in r.items() if k != "params"} for r in results])


@GLOO
@pytest.mark.parametrize("checkpoint_activations, evaluate", [(False, False), (True, False), (False, True)],
                         ids=["fsdp", "fsdp_act", "fsdp_eval"])
def test_fsdp_plan_follows_the_lightning_path(gloo_pool, checkpoint_activations, evaluate):
    """A manual FSDP2 plan (the root sharded last) follows the Lightning test's path, every planned module is an
    FSDP module holding its share of each parameter, and the parameters match a single-process run, with or without
    activation checkpointing and with a forward-only evaluation before each ``epoch_end``."""
    results = gloo_pool.run(mp.fsdp, checkpoint_activations=checkpoint_activations, evaluate=evaluate)
    agreed = _agreed(results)
    assert _states(agreed["path"]) == _expected(path_fsdp_autocm)
    assert agreed["layout"] == {"fsdp_modules": ["", "layers.0", "layers.1", "norm", "output"], "uneven": []}
    assert agreed["unmoved"] == []
    ref = mp.reference()
    assert _phases(agreed["path"]) == _phases(ref["path"]) and ref["unmoved"] == []
    assert _close(results[0]["params"], ref["params"])


@GLOO
def test_fsdp_restores_the_best_and_resumes(gloo_pool, tmp_path):
    """Best-checkpoint restores at each boundary (through DCP) match a single-process run, and a save loaded into a
    fresh sharded model finishes exactly as an uninterrupted run."""
    results = gloo_pool.run(mp.fsdp, ckpt_dir=str(tmp_path / "sharded"), val=VAL)
    agreed = _agreed(results)
    ref = mp.reference(ckpt_dir=str(tmp_path / "ref"), val=VAL)
    assert agreed["sharded_best"] and agreed["restored"] == ref["restored"] == 2
    assert _close(results[0]["params"], ref["params"])
    resumed = gloo_pool.run(mp.fsdp_resume, ckpt_dir=str(tmp_path / "resume"), val=VAL)
    assert all(r["saved_is_dir"] for r in resumed)
    assert _phases(resumed[0]["path"]) == _phases(agreed["path"])
    assert _close(resumed[0]["params"], results[0]["params"], tol=1e-6)


@GLOO
@pytest.mark.parametrize("where", ["gate", "end", "start"])
def test_forward_in_a_gate_or_phase_hook_before_the_thaw(gloo_pool, where):
    """A gate or a phase hook may evaluate the model between the tuning scheduler's restore and its thaw; the thaw
    must still reach the parameters FSDP2 trains (#90)."""
    results = gloo_pool.run(mp.fsdp, callback_eval=where)
    agreed = _agreed(results)
    assert _states(agreed["path"]) == _expected(path_fsdp_autocm) and agreed["unmoved"] == []
    assert _close(results[0]["params"], mp.reference()["params"])


@GLOO
def test_load_after_a_forward_resumes_exactly(gloo_pool, tmp_path):
    """A phase-0 start hook evaluates the fresh model before ``load()`` replays the thaw up to the saved depth
    (where the root's parameters are thawed); the resumed run still matches an uninterrupted one (#90)."""
    val = [1.0, 0.8, 0.9, 0.7]
    resumed = gloo_pool.run(mp.fsdp_resume, ckpt_dir=str(tmp_path / "resume"), epochs=4, split=3, val=val,
                            hook_eval=True)
    straight = gloo_pool.run(mp.fsdp, epochs=4, ckpt_dir=str(tmp_path / "straight"), val=val)
    assert straight[0]["unmoved"] == []
    assert _phases(resumed[0]["path"]) == _phases(straight[0]["path"])
    assert _close(resumed[0]["params"], straight[0]["params"], tol=1e-6)


@GLOO
def test_root_only_plan_saves_sharded_after_a_forward(gloo_pool, tmp_path):
    """With FSDP2 on the root alone, an evaluation forward leaves every parameter unsharded; best checkpoints are
    still written and restored through DCP, and the run matches a single-process one."""
    results = gloo_pool.run(mp.fsdp, root_only=True, evaluate=True, ckpt_dir=str(tmp_path / "root"), val=VAL)
    agreed = _agreed(results)
    assert agreed["layout"]["fsdp_modules"] == [""] and agreed["sharded_best"] and agreed["unmoved"] == []
    ref = mp.reference(ckpt_dir=str(tmp_path / "ref"), val=VAL)
    assert agreed["restored"] == ref["restored"] == 2
    assert _close(results[0]["params"], ref["params"])


@GLOO
def test_warning_covers_tensor_parallel_params_outside_fsdp(gloo_pool):
    """On a (data-parallel, tensor-parallel) mesh, a tensor-parallel DTensor no FSDP module manages is not averaged
    across data-parallel ranks, so it is warned about like a plain tensor."""
    results = gloo_pool.run(mp.warn_2d)
    assert results[0]["tok_embeddings_is_dtensor"]
    assert len(results[0]["warned"]) == 1 and "tok_embeddings.weight" in results[0]["warned"][0]


@GLOO
@pytest.mark.parametrize("tp_loss_parallel", [True, False], ids=["loss_parallel", "replicated_loss"])
def test_tp_follows_the_lightning_path(gloo_pool, tmp_path, tp_loss_parallel):
    """Tensor parallelism over both ranks follows the Lightning test's TP path, restores the best through DCP, and
    matches a single-process run."""
    results = gloo_pool.run(mp.tp, tp_loss_parallel=tp_loss_parallel, ckpt_dir=str(tmp_path / "tp"), val=VAL)
    agreed = _agreed(results)
    assert _states(agreed["path"]) == _expected(path_tp)
    assert agreed["unmoved"] == []
    ref = mp.reference(ckpt_dir=str(tmp_path / "ref"), val=VAL)
    assert agreed["restored"] == ref["restored"] == 2
    assert _close(results[0]["params"], ref["params"], tol=1e-4)


@GLOO
def test_partial_plan_reshards_before_thawing_and_warns(gloo_pool):
    """With the root unsharded, ``layers.1`` is a root FSDP module frozen through phase 0, which FSDP2 leaves
    unsharded after its forward: the tuning scheduler reshards before thawing, so phase 1 trains it. The embeddings,
    which no FSDP module manages, are not synchronized across ranks, and the tuning scheduler warns about them."""
    results = gloo_pool.run(mp.partial_plan)
    path = assert_ranks_agree([r["path"] for r in results])
    assert [state["thawed"] for _, _, state in path] == [3, 15, 29]
    assert all(r["unmoved"] == [] for r in results)
    assert len(results[0]["warned"]) == 1 and "tok_embeddings.weight" in results[0]["warned"][0]
    a, b = (r["plain"] for r in results)
    assert sorted(a) == ["pos_embeddings.weight", "tok_embeddings.weight"]
    assert all(max(abs(u - v) for u, v in zip(a[k], b[k])) > 1e-4 for k in a)  # what the warning is about


@GLOO
def test_wrappers_that_miss_late_thaws_are_refused(gloo_pool):
    """``DistributedDataParallel`` (on the root or a submodule) and the composable ``replicate()`` would never
    synchronize parameters thawed after phase 0 (#89)."""
    refused = assert_ranks_agree(gloo_pool.run(mp.refused_wrappers))
    assert refused == {"ddp_root": "the model is wrapped in `DistributedDataParallel`",
                       "ddp_submodule": "`layers.1` is wrapped in `DistributedDataParallel`",
                       "replicate": "the model is under the composable `replicate()`"}


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize("case", ["fsdp", "fsdp_fp16", "fsdp_offload", "tp"])
def test_model_parallel_nccl(nccl_pool, tmp_path, case):
    """The FSDP2 (with an evaluation forward before each ``epoch_end``) and TP cases on NCCL, plus FSDP2 fp16 mixed
    precision (one of the CI GPUs has no bf16) and FSDP2 with a CPU offload policy (#55), whose shards stay on the
    CPU while thawing, best-checkpoint restores and training follow the same path."""
    if case == "tp":
        results = nccl_pool.run(mp.tp, ckpt_dir=str(tmp_path / "tp"), val=VAL)
        expected, tol = path_tp, 1e-3
    else:
        results = nccl_pool.run(mp.fsdp, fp16=case == "fsdp_fp16", offload=case == "fsdp_offload",
                                ckpt_dir=str(tmp_path / "fsdp"), val=VAL, evaluate=True)
        expected, tol = path_fsdp_autocm, (2e-2 if case == "fsdp_fp16" else 1e-3)
    agreed = _agreed(results)
    assert _states(agreed["path"]) == _expected(expected)
    assert agreed["unmoved"] == []
    if case == "fsdp_fp16":
        assert agreed["unsharded_dtypes"] == ["torch.float16"]
    if case.startswith("fsdp"):
        assert agreed["shard_devices"] == (["cpu"] if case == "fsdp_offload" else ["cuda"])
    ref = mp.reference(ckpt_dir=str(tmp_path / "ref"), val=VAL)
    assert agreed["restored"] == ref["restored"] == 2
    assert _close(results[0]["params"], ref["params"], tol=tol)
