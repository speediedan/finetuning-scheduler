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
"""``shard_model`` (#55): an FSDP2 plan applied to a plain PyTorch model, the root last, and the FSDP2 adapter's
feedback on FSDP units whose parameters thaw in different phases. Run once per simulated rank on the fake process
group, so on every hosted OS; the gloo and NCCL tuning scheduler tests cover training through sharded models.

The ``core-no-lightning`` CI job runs this module with ``--noconftest``, so it must not import Lightning or
``tests.helpers``.
"""
import pytest
import torch.distributed as dist

from tests.dist_cases import fts_shard_plan as cases
from tests.dist_cases._fake_pg import assert_ranks_agree, run_fake

pytestmark = pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")

BLOCKS = ["layers.0", "layers.1"]


def _agreed(case=cases.shard, **kwargs) -> dict:
    return assert_ranks_agree(run_fake(case, **kwargs))


def test_plan_shards_selected_modules_then_the_root():
    """A ``class:`` key, a name and a regex select the modules; the root is sharded last, so every parameter is a
    DTensor, and with each FSDP unit's parameters thawing together there is no phase feedback."""
    out = _agreed(plan={"class:TransformerBlock": {}, "norm": {}, r"^output$": {}})
    assert out["fsdp_modules"] == ["", *BLOCKS, "norm", "output"]
    assert out["plain"] == [] and out["feedback"] == []


RESHARD = {"reshard_after_forward": True}


def test_a_unit_spanning_phases_is_reported():
    """Without ``norm`` in the plan, the root unit holds it (phase 0) beside the embeddings (phase 2).

    Resharded after forward, it all-gathers the embeddings again in every backward pass from phase 0 on, which the
    feedback says, naming the phases.
    """
    out = _agreed(plan={"class:TransformerBlock": {}, "output": {}}, default_kwargs=RESHARD)
    assert out["fsdp_modules"] == ["", *BLOCKS, "output"]
    (level, message), = out["feedback"]
    assert level == "warning" and "'<root>'" in message and "phase 0, phase 2" in message


def test_a_root_kept_unsharded_after_forward_is_not_reported():
    """By default ``fully_shard`` keeps the root unsharded after forward, so its backward all-gathers nothing and a
    root spanning phases costs no communication: no feedback."""
    out = _agreed(plan={"class:TransformerBlock": {}, "output": {}})
    assert out["feedback"] == []


def test_ignored_params_are_not_counted_in_their_unit():
    """Parameters a unit ignores are not managed by it: with ``norm`` (phase 0) ignored by the root, the root manages
    only the embeddings (phase 2), so no feedback."""
    out = _agreed(plan={"class:TransformerBlock": {}, "output": {}, "": {"ignored_params": r"norm\..*"}},
                  default_kwargs=RESHARD)
    assert out["plain"] == ["norm.bias", "norm.weight"]
    assert [level for level, _ in out["feedback"]] == ["info"]  # only the mixed DTensor/plain phase 0 notice


def test_ignored_params_by_name_stay_plain():
    """``ignored_params`` named in the plan are left unsharded (and unsynchronized: a schedule should not thaw them
    without synchronizing them some other way)."""
    out = _agreed(plan={"output": {"ignored_params": "output.weight"}, "class:TransformerBlock": {}, "norm": {}})
    assert out["plain"] == ["output.weight"]


def test_a_module_the_caller_sharded_is_left_alone():
    out = _agreed(plan={"class:TransformerBlock": {}, "norm": {}, "output": {}}, pre_shard="layers.1")
    assert out["fsdp_modules"] == ["", *BLOCKS, "norm", "output"] and out["plain"] == []


@pytest.mark.parametrize("plan, default_kwargs, match", [
    ({"class:NoSuchBlock": {}}, None, "The class 'class:NoSuchBlock' specified in `fsdp_plan`"),
    ({"output": {"ignored_params": "norm.weight"}}, None, "'norm.weight' in the `ignored_params`"),
    ({"output": {}}, {"act_ckpt": ["wrapped"]}, "`act_ckpt` mode 'wrapped' is not supported"),
    ({"output": {"act_ckpt": {"mode": "composable"}}}, None, "must be a mode or a [mode, kwargs] list"),
    ({"layers.0": {}, "layers.0.attention": {}}, None, "its ancestor 'layers.0' is already sharded"),
], ids=["unmatched_class", "ignored_outside_module", "non_composable_act_ckpt", "act_ckpt_mapping",
        "parent_before_child"])
def test_refusals(plan, default_kwargs, match):
    out = _agreed(plan=plan, default_kwargs=default_kwargs)
    assert match in out["error"]


def test_composable_activation_checkpointing():
    """``act_ckpt: composable`` checkpoints a module before sharding it."""
    out = _agreed(plan={"class:TransformerBlock": {"act_ckpt": ["composable"]}, "norm": {}, "output": {}})
    assert out["fsdp_modules"] == ["", *BLOCKS, "norm", "output"] and out["plain"] == []


def test_cpu_offload_policy_alias(monkeypatch):
    """The ``cpu_offload_policy`` alias becomes an ``offload_policy`` on every module, the root included (CPU
    offload works on the PyTorch API, unlike under Lightning's ``ModelParallelStrategy``).

    Checked with ``fully_shard``
    stubbed, since an offload policy pins host memory, which needs CUDA.
    """
    from torch import nn
    from torch.distributed.fsdp import CPUOffloadPolicy

    from finetuning_scheduler.strategies import plan as plan_mod

    calls = []
    monkeypatch.setattr(plan_mod, "fully_shard", lambda module, mesh, **kw: calls.append(kw))
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    plan_mod.shard_model(model, None, {"0": {}}, default_kwargs={"cpu_offload_policy": {"pin_memory": False}})
    assert len(calls) == 2
    assert all(isinstance(kw["offload_policy"], CPUOffloadPolicy) and not kw["offload_policy"].pin_memory
               and "cpu_offload_policy" not in kw for kw in calls)


@pytest.mark.parametrize("config", ["torch_fsdp_plan.yaml", "fts_fsdp_auto_plan.yaml", "fts_fsdp_profiling.yaml"])
def test_the_example_plans_need_no_feedback(monkeypatch, config):
    """The model-parallel examples' FSDP2 plans give ``model.norm`` its own unit wherever the unit holding it would
    reshard after forward, so no unit is reported and their runs' warning checks stay clean; the examples
    themselves run only on GPUs."""
    monkeypatch.setenv("TORCH_CPP_LOG_LEVEL", "ERROR")  # the example script sets it on import; undone after the test
    pytest.importorskip("fts_examples.model_parallel.torch_mp_example")
    out = _agreed(case=cases.example_plan, config=config)
    assert out["plain"] == [] and out["feedback"] == []
    assert {"model.output", "model.layers.0"} <= set(out["fsdp_modules"])
