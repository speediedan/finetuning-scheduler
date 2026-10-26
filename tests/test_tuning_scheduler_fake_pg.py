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
"""``TuningScheduler``'s FSDP2 and tensor-parallel scheduling in one process, on the fake process group.

These run the model-parallel pool cases (``tests/dist_cases/fts_tuning_scheduler_mp.py``) once per simulated rank, so
the scheduling logic is checked on every hosted OS, not only where the gloo pool runs. The fake group's collectives move
no data, so those runs compare no numerics; the tensor-parallel case also runs once under ``LocalTensorMode``
(``run_local``), whose numerics are real, and is compared with a single-process run. DCP checkpointing (whose global
plan needs every rank's real chunks) stays with the gloo tests in ``tests/test_tuning_scheduler_model_parallel.py``.

The ``core-no-lightning`` CI job runs this module with ``--noconftest``, so it must not import Lightning or
``tests.helpers``.
"""
import sys

import pytest
import torch
import torch.distributed as dist

from tests.dist_cases import fts_tuning_scheduler_mp as mp
from tests.dist_cases.fts_tuning_scheduler_mp import params_close
from tests.dist_cases._fake_pg import assert_ranks_agree, run_fake, run_local
from tests.model_parallel_expected_paths import path_fsdp_autocm, path_tp

pytestmark = pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")


def _expected(path: dict) -> list:
    """An expected path's per-epoch parameter states, with the Lightning module's ``model.`` prefix dropped."""
    return [({n.removeprefix("model."): s for n, s in path[e][0]["p_states"].items()}, path[e][1])
            for e in sorted(path)]


def _states(path: list) -> list:
    return [(state["p_states"], state["thawed"]) for _, _, state in path]


def _thawed_plain(path: list) -> list:
    """Per epoch, the thawed parameters that are plain tensors rather than DTensors: an unsharded copy thawed in
    place of its sharded parameter (#90), which the sampled states and the thawed count do not show."""
    return [state["thawed_plain"] for _, _, state in path]


def _scheduling(results: list, *keys: str) -> dict:
    """What every simulated rank must agree on: the phase path, and any of ``keys`` the case returns."""
    return assert_ranks_agree([{k: r[k] for k in ("path", *keys) if k in r} for r in results])


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("kwargs", [{}, {"checkpoint_activations": True}, {"evaluate": True},
                                    {"callback_eval": "gate"}, {"callback_eval": "start"}, {"callback_eval": "end"}],
                         ids=["fsdp", "fsdp_act", "fsdp_eval", "gate_eval", "start_eval", "end_eval"])
def test_fsdp_plan_follows_the_lightning_path(world_size, kwargs):
    """A manual FSDP2 plan (the root sharded last) follows the Lightning test's path on every rank, and every
    planned module is an FSDP module holding its even share of each parameter, with or without activation
    checkpointing and a forward-only evaluation before ``epoch_end`` or inside a gate or a phase hook."""
    agreed = _scheduling(run_fake(mp.fsdp, world_size=world_size, **kwargs), "layout")
    assert _states(agreed["path"]) == _expected(path_fsdp_autocm)
    assert _thawed_plain(agreed["path"]) == [[], [], []]
    assert agreed["layout"] == {"fsdp_modules": ["", "layers.0", "layers.1", "norm", "output"], "uneven": []}


def test_root_only_plan_follows_the_lightning_path():
    """With FSDP2 on the root alone and an evaluation forward before each ``epoch_end``, the path is unchanged."""
    agreed = _scheduling(run_fake(mp.fsdp, root_only=True, evaluate=True), "layout")
    assert _states(agreed["path"]) == _expected(path_fsdp_autocm)
    assert _thawed_plain(agreed["path"]) == [[], [], []]
    assert agreed["layout"]["fsdp_modules"] == [""]


# torch 2.12's FSDP2 assigns the fp16 gradient to the fp32 parameter on CPU, on gloo as on the fake group; 2.14 is
# the oldest version verified to reduce it in fp32 as the policy asks
@pytest.mark.skipif(torch.__version__ < "2.14", reason="FSDP2 fp16 mixed precision on CPU needs torch 2.14 or later")
def test_fsdp_fp16_computes_in_half_precision():
    """Under an fp16 ``MixedPrecisionPolicy`` the path is unchanged and each FSDP module computes in fp16."""
    agreed = _scheduling(run_fake(mp.fsdp, fp16=True), "unsharded_dtypes")
    assert _states(agreed["path"]) == _expected(path_fsdp_autocm)
    assert _thawed_plain(agreed["path"]) == [[], [], []]
    assert agreed["unsharded_dtypes"] == ["torch.float16"]


@pytest.mark.parametrize("tp_loss_parallel", [True, False], ids=["loss_parallel", "replicated_loss"])
def test_tp_follows_the_lightning_path(tp_loss_parallel):
    """Tensor parallelism over both ranks follows the Lightning test's TP path on every rank."""
    agreed = _scheduling(run_fake(mp.tp, tp_loss_parallel=tp_loss_parallel))
    assert _states(agreed["path"]) == _expected(path_tp)
    assert _thawed_plain(agreed["path"]) == [[], [], []]


@pytest.mark.parametrize("tp_loss_parallel", [True, False], ids=["loss_parallel", "replicated_loss"])
def test_tp_matches_a_single_process_run(tp_loss_parallel):
    """Tensor parallelism over two ranks simulated with real numerics (``LocalTensorMode``) follows the Lightning
    test's TP path and ends with a single-process run's parameters (#57): the gloo pool checks the same on Linux,
    with a best-checkpoint policy, and this runs on every hosted OS."""
    out = run_local(mp.tp, tp_loss_parallel=tp_loss_parallel)
    assert _states(out["path"]) == _expected(path_tp) and out["unmoved"] == []
    assert params_close(out["params"], mp.reference()["params"])


def test_partial_plan_thaws_on_schedule_and_warns_once():
    """With the root unsharded, ``layers.1`` is a root FSDP module frozen through phase 0: phase 1 still thaws it.

    The tuning scheduler warns, on rank 0 only, about the embeddings no FSDP module manages.
    """
    results = run_fake(mp.partial_plan)
    path = _scheduling(results)["path"]
    assert [state["thawed"] for _, _, state in path] == [3, 15, 29]
    # the embeddings, which no FSDP module manages, are plain by design; every FSDP-managed parameter is sharded
    assert _thawed_plain(path) == [[], [], ["pos_embeddings.weight", "tok_embeddings.weight"]]
    assert len(results[0]["warned"]) == 1 and "tok_embeddings.weight" in results[0]["warned"][0]
    assert results[1]["warned"] == []


def test_warning_covers_tensor_parallel_params_outside_fsdp():
    """On a (data-parallel, tensor-parallel) mesh, a tensor-parallel DTensor no FSDP module manages is warned about
    like a plain tensor."""
    results = run_fake(mp.warn_2d)
    assert results[0]["tok_embeddings_is_dtensor"]
    assert len(results[0]["warned"]) == 1 and "tok_embeddings.weight" in results[0]["warned"][0]


def test_wrappers_that_miss_late_thaws_are_refused():
    """``DistributedDataParallel`` (on the root or a submodule) and the composable ``replicate()`` are refused on
    every rank."""
    refused = assert_ranks_agree(run_fake(mp.refused_wrappers))
    assert refused == {"ddp_root": "the model is wrapped in `DistributedDataParallel`",
                       "ddp_submodule": "`layers.1` is wrapped in `DistributedDataParallel`",
                       "replicate": "the model is under the composable `replicate()`"}


def test_run_fake_leaves_no_process_group():
    """The harness itself: no group outlives a run, a case may neither create or destroy the default group nor
    broadcast an object, and the hooks and RNG states ``init_process_group`` and the seeding change are restored."""

    def owns_the_group(ctx):
        dist.destroy_process_group()

    def broadcasts(ctx):
        dist.broadcast_object_list([None], src=0)

    excepthook, rng = sys.excepthook, torch.get_rng_state()
    with pytest.raises(AssertionError, match="must not create or destroy"):
        run_fake(owns_the_group)
    with pytest.raises(AssertionError, match="cannot broadcast objects"):
        run_fake(broadcasts)
    assert not dist.is_initialized()
    assert run_fake(lambda ctx: (ctx.rank, dist.get_world_size()), world_size=3) == [(0, 3), (1, 3), (2, 3)]
    assert not dist.is_initialized()
    assert sys.excepthook is excepthook and torch.equal(torch.get_rng_state(), rng)


def test_run_local_leaves_no_process_group():
    run_local(lambda ctx: dist.get_world_size())
    assert not dist.is_initialized()
