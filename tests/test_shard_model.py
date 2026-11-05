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
import re

import pytest
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.strategies.plan import resolve_tp_plan
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
    ({"output": {"act_ckpt": ["selective"]}}, None, "`act_ckpt` mode 'selective' for 'output' is not supported"),
    ({"": {"act_ckpt": ["wrapped"]}}, None, "cannot wrap the root module itself"),
    ({"output": {"act_ckpt": {"mode": "composable"}}}, None, "must be a mode or a [mode, kwargs] list"),
    ({"output": {"mp_policy": {"param_dtype": "float99"}}}, None, "names no torch dtype"),
], ids=["unmatched_class", "ignored_outside_module", "unknown_act_ckpt_mode", "wrapped_root", "act_ckpt_mapping",
        "mp_policy_bad_dtype"])
def test_refusals(plan, default_kwargs, match):
    out = _agreed(plan=plan, default_kwargs=default_kwargs)
    assert match in out["error"]


def test_composable_activation_checkpointing():
    """``act_ckpt: composable`` checkpoints a module before sharding it."""
    out = _agreed(plan={"class:TransformerBlock": {"act_ckpt": ["composable"]}, "norm": {}, "output": {}})
    assert out["fsdp_modules"] == ["", *BLOCKS, "norm", "output"] and out["plain"] == []


@pytest.mark.parametrize("mode", ["wrapped", "wrapped_offload"])
def test_wrapped_activation_checkpointing_thaws_under_wrapper_names(mode):
    """``act_ckpt``'s wrapped modes (#149) replace each block with a checkpoint wrapper before sharding: its
    parameters carry the wrapper's prefix, and a phased schedule naming them by those names thaws exactly phase 0's
    (block 0 under the wrapper, the norm outside it)."""
    out = _agreed(cases.shard_wrapped_schedule, mode=mode)
    assert out["wrapped"] == BLOCKS and out["plain"] == []
    assert out["wrapped_params"] and all(n.startswith(tuple(f"{b}." for b in BLOCKS)) for n in out["wrapped_params"])
    assert out["thawed"] == sorted(
        [n for n in out["wrapped_params"] if n.startswith("layers.0.")] + ["norm.bias", "norm.weight"])


def test_plan_entries_apply_deepest_first():
    """A plan may list a module before its children: the children are sharded first whatever the plan order."""
    out = _agreed(plan={"layers.0": {}, "layers.0.attention": {}, "class:TransformerBlock": {}, "norm": {},
                        "output": {}})
    assert "layers.0.attention" in out["fsdp_modules"] and "layers.0" in out["fsdp_modules"]
    assert out["plain"] == [] and "error" not in out


def test_mp_policy_alias_by_dtype_name(monkeypatch):
    """An ``mp_policy`` mapping with dtype names becomes a ``MixedPrecisionPolicy`` (so a YAML plan can carry one),
    checked with ``fully_shard`` stubbed, as the cpu-offload alias test does."""
    import torch
    from torch import nn
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from finetuning_scheduler import shard_model as root_shard_model
    from finetuning_scheduler.strategies import plan as plan_mod

    assert root_shard_model is plan_mod.shard_model  # the package root exports it (#149)
    calls = []
    monkeypatch.setattr(plan_mod, "fully_shard", lambda module, mesh, **kw: calls.append(kw))
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    plan_mod.shard_model(model, None, {"0": {"mp_policy": {"param_dtype": "bfloat16",
                                                           "reduce_dtype": torch.float32}}})
    (first, _root), = [calls]
    assert isinstance(first["mp_policy"], MixedPrecisionPolicy)
    assert first["mp_policy"].param_dtype is torch.bfloat16 and first["mp_policy"].reduce_dtype is torch.float32


def test_plan_helpers_importable_from_package_root():
    import finetuning_scheduler as fts
    from finetuning_scheduler.strategies import plan as plan_mod

    assert fts.shard_model is plan_mod.shard_model
    assert fts.resolve_fsdp_plan is plan_mod.resolve_fsdp_plan
    assert fts.phase_feedback is plan_mod.phase_feedback


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


def test_tp_plan_keeps_names_and_schedules_through_the_dtensor_adapter():
    """A configured tensor-parallel plan (#150) leaves parameter names alone and no parameter plain, so the
    unchanged schedule validates under the adapter and thaws phase 0."""
    out = _agreed(cases.parallelized)
    assert out["names_unchanged"] and out["plain"] == []
    assert out["adapter"] == "DTensorAdapter"
    assert out["thawed"] == ["norm.bias", "norm.weight", "output.weight"]


def test_tp_plan_schedule_naming_a_missing_parameter_is_refused():
    assert "no_such_parameter" in _agreed(cases.parallelized_bad_schedule)


def test_tp_plan_refuses_a_mesh_with_more_than_one_dimension():
    assert "mesh['tensor_parallel']" in _agreed(cases.parallelize_refuses_a_2d_mesh)


def _toy():
    with torch.device("meta"):
        return nn.ModuleDict({"blocks": nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)]), "head": nn.Linear(2, 2),
                              "embed": nn.Embedding(2, 2)})


def test_resolve_tp_plan_builds_a_new_style_per_module_from_each_key_form():
    """Names, regexes and ``class:`` keys select modules, and a key's modules do not share a style."""
    resolved = resolve_tp_plan(_toy(), {"head": {"style": "rowwise"}, r"blocks\.\d+$": {"style": "colwise"},
                                        "class:Embedding": {"style": "rowwise"}})
    assert list(resolved) == ["head", "blocks.0", "blocks.1", "embed"]
    assert isinstance(resolved["blocks.0"], ColwiseParallel) and isinstance(resolved["head"], RowwiseParallel)
    assert resolved["blocks.0"] is not resolved["blocks.1"]


def test_resolve_tp_plan_reads_layouts_and_the_loss_parallel_overlay():
    plan = {"head": {"style": "colwise", "input_layouts": {"shard": 1}, "output_layouts": "replicate",
                     "loss_parallel": {"output_layouts": {"shard": -1}, "use_local_output": False}},
            "blocks.0": {"style": "prepare_input", "input_layouts": [{"shard": 1}, None],
                         "desired_input_layouts": ["replicate", None],
                         "input_kwarg_layouts": {"mask": {"shard": 0}},
                         "desired_input_kwarg_layouts": {"mask": "replicate"}}}
    plain = resolve_tp_plan(_toy(), plan)
    assert plain["head"].input_layouts == (Shard(1),)
    assert plain["head"].output_layouts == (Replicate(),) and plain["head"].use_local_output
    assert plain["blocks.0"].input_layouts == (Shard(1), None)
    assert plain["blocks.0"].input_kwarg_layouts == {"mask": Shard(0)}
    overlaid = resolve_tp_plan(_toy(), plan, loss_parallel=True)["head"]
    assert overlaid.output_layouts == (Shard(-1),) and not overlaid.use_local_output
    assert yaml.safe_load(yaml.safe_dump(plan)) == plan  # the plan is plain data


def test_resolve_tp_plan_prunes_containers_with_a_warning():
    warned = []
    assert list(resolve_tp_plan(_toy(), {"blocks": {"style": "colwise"}, "head": {"style": "rowwise"}},
                                warn=warned.append)) == ["head"]
    assert "blocks" in warned[0]


@pytest.mark.parametrize("plan, kwargs, message", [
    ({"nothing": {"style": "colwise"}}, {}, "did not match"),
    ({"class:Nothing": {"style": "colwise"}}, {}, "class 'class:Nothing'"),
    ({"head": {"style": "diagonal"}}, {}, "must be a mapping whose `style`"),
    ({"head": None}, {}, "must be a mapping whose `style`"),
    ({"head": {"style": "colwise", "input_layouts": "sharded"}}, {}, "is not 'replicate'"),
    ({"head": {"style": "colwise", "input_layouts": {"shard": "1"}}}, {}, "is not 'replicate'"),
    ({"head": {"style": "sequence", "output_layouts": "replicate"}}, {}, "does not take"),
    ({"head": {"style": "colwise"}}, {"loss_parallel": True}, "needs a `tp_plan` entry"),
])
def test_resolve_tp_plan_refuses_a_malformed_plan(plan, kwargs, message):
    with pytest.raises(FTSConfigurationError, match=re.escape(message)):
        resolve_tp_plan(_toy(), plan, **kwargs)
