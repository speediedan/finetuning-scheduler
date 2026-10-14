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
group, so on every hosted OS; the gloo and NCCL controller tests cover training through sharded models.

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


def test_a_unit_spanning_phases_is_reported():
    """Without ``norm`` in the plan the root unit holds it (phase 0) beside the embeddings (phase 2): it
    communicates the embeddings from phase 0 on, which the feedback says, naming the phases."""
    out = _agreed(plan={"class:TransformerBlock": {}, "output": {}})
    assert out["fsdp_modules"] == ["", *BLOCKS, "output"]
    (level, message), = out["feedback"]
    assert level == "warning" and "'<root>'" in message and "phase 0, phase 2" in message


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
], ids=["unmatched_class", "ignored_outside_module", "non_composable_act_ckpt"])
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


def test_the_plain_pytorch_example_plan_needs_no_feedback(monkeypatch):
    """The plain PyTorch FSDP2 example gives ``model.norm`` its own unit, so no unit spans phases and its run's
    warning check stays clean; the example itself runs only on GPUs."""
    monkeypatch.setenv("TORCH_CPP_LOG_LEVEL", "ERROR")  # the example script sets it on import; undone after the test
    pytest.importorskip("fts_examples.model_parallel.torch_mp_example")
    out = _agreed(case=cases.example_plan)
    assert out["plain"] == [] and out["feedback"] == []
    assert {"", "model.norm", "model.output", "model.layers.0"} <= set(out["fsdp_modules"])
