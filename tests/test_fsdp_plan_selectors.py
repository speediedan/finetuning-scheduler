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
"""How the ``ModelParallelStrategyAdapter`` resolves an ``fsdp_plan`` (#55): module names, regexes and ``class:``
selectors, an ``ignored_params`` given by name, and the refusal of CPU offload, which Lightning's
``ModelParallelStrategy`` cannot run.

Resolution needs no process group, so these run on CPU; the standalone
model-parallel tests apply the resolved plans with ``fully_shard``, and the PyTorch API's NCCL tests run CPU offload.
"""
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy

from finetuning_scheduler.strategy_adapters import ModelParallelStrategyAdapter
from tests.dist_cases._mp_models import FTSToyTransformer, TestModelArgs, TransformerBlock


class _Wrapper(torch.nn.Module):
    """The ``LightningModule`` stand-in: the transformer under ``model``, as the model-parallel tests hold it."""

    def __init__(self):
        super().__init__()
        self.model = FTSToyTransformer(TestModelArgs())


class _SpecialBlock(TransformerBlock):
    pass


@pytest.fixture
def named_modules():
    return dict(_Wrapper().named_modules())


@pytest.mark.parametrize("key, expected", [
    ("model.norm", ["model.norm"]),
    (r"^model\.layers\.\d+$", ["model.layers.0", "model.layers.1"]),
    ("class:TransformerBlock", ["model.layers.0", "model.layers.1"]),
    (f"class:{TransformerBlock.__module__}.{TransformerBlock.__qualname__}", ["model.layers.0", "model.layers.1"]),
    ("class: TransformerBlock", ["model.layers.0", "model.layers.1"]),
    ("class:FTSToyTransformer", ["model"]),
    ("class:NoSuchBlock", []),
    ("class:other_pkg.TransformerBlock", []),
], ids=["name", "regex", "class", "qualified_class", "class_with_space", "class_of_root_child", "unknown_class",
        "wrong_module"])
def test_plan_keys_select_modules(named_modules, key, expected):
    assert ModelParallelStrategyAdapter._plan_key_modules(key, named_modules) == expected


def test_class_selector_matches_subclasses():
    """As ``ModuleWrapPolicy`` does, a class selects its subclasses' instances too."""
    wrapper = _Wrapper()
    wrapper.model.layers[1] = _SpecialBlock(TestModelArgs())
    named = dict(wrapper.named_modules())
    assert ModelParallelStrategyAdapter._plan_key_modules("class:TransformerBlock", named) == ["model.layers.0",
                                                                                               "model.layers.1"]
    assert ModelParallelStrategyAdapter._plan_key_modules("class:_SpecialBlock", named) == ["model.layers.1"]


class _Adapter(ModelParallelStrategyAdapter):
    """The adapter with only a plan and a stand-in ``LightningModule``: plan resolution needs none of the Lightning
    wiring."""

    def __init__(self, plan: dict, default_kwargs: dict | None = None):
        self.fsdp_plan, self.fsdp_default_kwargs, self._module = plan, default_kwargs or {}, _Wrapper()

    @property
    def pl_module(self):
        return self._module

    @property
    def pls_handle(self):
        return self._pls


def _validate(plan: dict, default_kwargs: dict | None = None) -> dict:
    adapter = _Adapter(plan, default_kwargs)
    adapter._validate_fsdp_plan()
    return adapter.fsdp_plan


def test_validate_expands_class_selectors_in_plan_order():
    plan = _validate({"class:TransformerBlock": {"reshard_after_forward": False}, "model.norm": {}})
    assert list(plan) == ["model.layers.0", "model.layers.1", "model.norm"]
    assert plan["model.layers.0"] == {"reshard_after_forward": False}


def test_validate_refuses_an_unmatched_class():
    with pytest.raises(MisconfigurationException, match="The class 'class:NoSuchBlock' specified in `fsdp_plan`"):
        _validate({"class:NoSuchBlock": {}})


def _shard_kwargs(plan_entry: dict, n: str = "model") -> tuple[dict, dict]:
    adapter = _Adapter({n: plan_entry})
    return adapter._shard_kwargs(n), dict(adapter.pl_module.named_parameters())


@pytest.mark.parametrize("names, expected", [
    ("model.output.weight", ["model.output.weight"]),
    (["model.norm.weight", r"model\.output\."], ["model.norm.weight", "model.output.weight"]),
    (r"^model\.layers\.1\.ffn_norm", ["model.layers.1.ffn_norm.weight", "model.layers.1.ffn_norm.bias"]),
], ids=["name", "list_with_regex", "regex"])
def test_ignored_params_by_name_resolve_to_parameters(names, expected):
    kwargs, params = _shard_kwargs({"ignored_params": names, "reshard_after_forward": True})
    expected_params = {params[n] for n in expected}
    assert kwargs["ignored_params"] == expected_params and kwargs["reshard_after_forward"] is True


def test_ignored_params_leaves_the_plan_unchanged_and_passes_sets_through():
    adapter = _Adapter({"model.output": {"ignored_params": "model.output.weight"}})
    adapter._shard_kwargs("model.output")
    assert adapter.fsdp_plan["model.output"]["ignored_params"] == "model.output.weight"  # resolved per application
    given = {next(adapter.pl_module.parameters())}
    adapter.fsdp_plan["model.output"]["ignored_params"] = given
    assert adapter._shard_kwargs("model.output")["ignored_params"] == given


def test_ignored_params_refuses_an_unmatched_name():
    with pytest.raises(MisconfigurationException, match="'model.nosuch' in the `ignored_params` of the `fsdp_plan`"):
        _shard_kwargs({"ignored_params": ["model.output.weight", "model.nosuch"]})


def test_ignored_params_resolve_only_among_the_entrys_own_parameters():
    """``fully_shard`` ignores only the module's own parameters, so a name elsewhere in the model is refused rather
    than silently left sharded."""
    kwargs, params = _shard_kwargs({"ignored_params": r"model\."}, n="model.output")
    assert kwargs["ignored_params"] == {params["model.output.weight"]}
    with pytest.raises(MisconfigurationException, match="did not match any parameter of that module"):
        _shard_kwargs({"ignored_params": "model.tok_embeddings.weight"}, n="model.output")


@pytest.mark.parametrize("container", [frozenset, list, tuple, iter], ids=["frozenset", "list", "tuple", "iterator"])
def test_ignored_params_collections_of_parameters_pass_through_as_a_set(container):
    adapter = _Adapter({"model.output": {}})
    expected = set(adapter.pl_module.model.output.parameters())
    adapter.fsdp_plan["model.output"]["ignored_params"] = container(adapter.pl_module.model.output.parameters())
    assert adapter._shard_kwargs("model.output")["ignored_params"] == expected and expected


def test_ignored_params_name_every_name_of_a_tied_parameter():
    """A tied parameter can be named by any of its names, not only the first one registered."""
    adapter = _Adapter({"model.output": {"ignored_params": "model.output.weight"}})
    adapter.pl_module.model.output.weight = adapter.pl_module.model.tok_embeddings.weight
    assert adapter._shard_kwargs("model.output")["ignored_params"] == {adapter.pl_module.model.output.weight}


def test_ignored_params_refuses_names_mixed_with_parameters():
    adapter = _Adapter({"model.output": {}})
    adapter.fsdp_plan["model.output"]["ignored_params"] = ["model.output.weight",
                                                           adapter.pl_module.model.output.weight]
    with pytest.raises(MisconfigurationException, match="mixes parameter names and parameters"):
        adapter._shard_kwargs("model.output")


def test_ignored_params_resolve_before_a_checkpoint_wrapper_renames_them(monkeypatch):
    """With ``act_ckpt: wrapped`` the module's parameters are renamed (``_checkpoint_wrapped_module``) before
    ``fully_shard`` runs, so names are resolved first."""
    from finetuning_scheduler.strategy_adapters import model_parallel

    captured = {}
    monkeypatch.setattr(model_parallel, "fully_shard", lambda module, **kw: captured.update(kw))
    n = "model.layers.0"
    adapter = _Adapter({n: model_parallel.ModelParallelStrategyAdapter._resolve_cfg_aliases(
        {"act_ckpt": ("wrapped",), "ignored_params": f"{n}.attention_norm.weight"})})
    adapter._pls = SimpleNamespace(device_mesh={"data_parallel": None})
    norm_weight = adapter.pl_module.get_submodule(n).attention_norm.weight
    adapter._compose_or_warn(n)
    assert any("_checkpoint_wrapped_module" in pn for pn, _ in adapter.pl_module.named_parameters())
    assert captured["ignored_params"] == {norm_weight}


def test_restored_scores_move_to_the_cpu():
    """A distributed checkpoint's restored scores arrive on rank zero's device on every rank (#112)."""
    from finetuning_scheduler.lightning.fts_supporters import _scores_to_cpu

    scores = {"a.ckpt": torch.tensor(0.5), "b.ckpt": torch.tensor(0.7)}
    moved = _scores_to_cpu(scores)
    assert all(v.device.type == "cpu" for v in moved.values()) and moved["a.ckpt"].item() == 0.5
    assert _scores_to_cpu(torch.tensor(1.0)).device.type == "cpu" and _scores_to_cpu(None) is None


@pytest.mark.parametrize("plan, defaults, where", [
    ({"model.norm": {}}, {"cpu_offload_policy": {}}, "`fsdp_default_kwargs`"),
    ({"model.norm": {"cpu_offload_policy": {"pin_memory": False}}}, {}, "the `fsdp_plan` entry for 'model.norm'"),
    ({"class:TransformerBlock": {"offload_policy": CPUOffloadPolicy()}}, {},
     "the `fsdp_plan` entry for 'class:TransformerBlock'"),
], ids=["default_alias", "entry_alias", "entry_policy"])
def test_cpu_offload_is_refused_under_lightning(plan, defaults, where):
    """Lightning's ``ModelParallelStrategy`` moves the model to the GPU after ``configure_model``, which CPU-
    offloaded FSDP2 shards cannot follow, so a plan asking for CPU offload is refused before any module is
    sharded."""
    with pytest.raises(MisconfigurationException, match=f"{where} asks for CPU offload"):
        _validate(plan, defaults)


def test_an_offload_policy_that_keeps_shards_on_the_gpu_is_accepted():
    """``OffloadPolicy()`` is FSDP2's default (no offload), so a plan may still name it."""
    plan = _validate({"model.norm": {"offload_policy": OffloadPolicy()}})
    assert isinstance(plan["model.norm"]["offload_policy"], OffloadPolicy)
