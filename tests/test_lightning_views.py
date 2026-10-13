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
"""The Lightning callback scheduling a module through an attached tunable view (``fts_tunable_view``), and its
precedence rules against a legacy parameter-naming strategy adapter (#53, #54)."""
import re
import warnings
from types import SimpleNamespace

import pytest
import torch
import yaml
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from finetuning_scheduler import CanonicalView, FinetuningScheduler, TunableSupport
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.strategy_adapters import FSDPStrategyAdapter, StrategyAdapter
from finetuning_scheduler.strategy_adapters import base as adapter_base
from finetuning_scheduler.strategy_adapters.base import NAMING_METHODS
from tests.helpers.expected_warns import NAMING_OVERRIDES_DEPRECATION_WARN, NAMING_OVERRIDES_PRECEDENCE_WARN
from tests.helpers.legacy_naming_adapter import LegacyNamingAdapter, LogicalNamesBoringModel
from tests.helpers.runif import RunIf
from tests.test_legacy_adapter_contract import (_fts, _trainer, EquivalentView, EXPECTED_PATH, LOGICAL_SCHEDULE,
                                                PathRecorder)
from tests.test_phase_characterization import _rank_zero_debug_records


def _viewed_model(view_cls=EquivalentView):
    module = LogicalNamesBoringModel()
    module.fts_tunable_view = view_cls(module)
    return module


def _initial(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def test_attached_view_reproduces_the_legacy_adapter_path(tmp_path):
    """The default adapter, scheduling through an attached view equivalent to the legacy adapter, follows the
    legacy adapter's pinned path (thawed tensors, the scheduler's record and parameter groups), and every thawed
    tensor trains."""
    seed_everything(42)
    module = _viewed_model()
    initial = _initial(module)
    fts, recorder = FinetuningScheduler(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True), PathRecorder()
    with _rank_zero_debug_records() as messages:
        _trainer(tmp_path, [fts, recorder]).fit(module)
    assert type(fts.strategy_adapter) is StrategyAdapter
    assert recorder.rows == EXPECTED_PATH
    assert fts.ft_schedule[1]["params"] == ["blocks.1.W"]  # the schedule keeps its logical names
    assert all(not torch.equal(initial[n], p) for n, p in module.named_parameters() if p.requires_grad)
    # the scheduler reports logical names
    assert any("blocks.1.W" in m for m in messages if m.startswith("Thawed the following"))


def test_role_selectors_under_lightning(tmp_path):
    """``role:`` selectors resolve through an attached view's roles, and tensors with frozen roles never train."""
    seed_everything(42)
    module = LogicalNamesBoringModel()
    roles = {"model.4.weight": "adapter:head", "model.4.bias": "adapter:head", "model.0.weight": "fitted"}
    module.fts_tunable_view = CanonicalView(module, roles=roles)
    schedule = {0: {"params": ["role:adapter:head"], "max_transition_epoch": 1},
                1: {"params": [r"model\.[23]\..*"], "max_transition_epoch": 2},
                2: {"params": [r"model\.1\..*"], "max_transition_epoch": 3}}
    initial = _initial(module)
    fts, recorder = FinetuningScheduler(ft_schedule=schedule, epoch_transitions_only=True), PathRecorder()
    _trainer(tmp_path, [fts, recorder], max_epochs=3).fit(module)
    assert recorder.rows[0][2] == ["model.4.bias", "model.4.weight"]
    assert not torch.equal(initial["model.4.weight"], module.model[4].weight)


def test_role_selectors_without_a_view(tmp_path):
    """Without an attached view, selectors resolve through the module's own names (every parameter is ``base``)."""
    seed_everything(42)
    module = LogicalNamesBoringModel()
    initial = _initial(module)
    fts = FinetuningScheduler(ft_schedule={0: {"params": ["role:base"], "max_transition_epoch": 1}},
                              epoch_transitions_only=True)
    recorder = PathRecorder()
    _trainer(tmp_path, [fts, recorder], max_epochs=1).fit(module)
    assert recorder.rows[0][2] == sorted(initial)
    assert all(not torch.equal(initial[n], p) for n, p in module.named_parameters())


def test_a_view_not_rooted_at_the_lightning_module_refuses(tmp_path):
    module = LogicalNamesBoringModel()
    module.fts_tunable_view = CanonicalView(module.model)
    fts = FinetuningScheduler(ft_schedule={0: {"params": [r"4\..*"], "max_transition_epoch": 1}},
                              epoch_transitions_only=True)
    with pytest.raises(MisconfigurationException, match="must be the LightningModule itself"):
        _trainer(tmp_path, [fts], max_epochs=1).fit(module)


def test_generated_schedules_leave_out_frozen_roles(tmp_path):
    module = LogicalNamesBoringModel()
    module.fts_tunable_view = CanonicalView(module, roles={"model.0.weight": "fitted", "model.0.bias": "fitted"})
    with pytest.raises(SystemExit):
        _trainer(tmp_path, [FinetuningScheduler(gen_ft_sched_only=True)], logger=True).fit(module)
    sched_file = tmp_path / "lightning_logs" / "version_0" / "LogicalNamesBoringModel_ft_schedule.yaml"
    scheduled = {n for phase in yaml.safe_load(sched_file.read_text()).values() for n in phase["params"]}
    assert scheduled and not scheduled & {"model.0.weight", "model.0.bias"}


def test_untrainable_attached_view_refuses_with_its_reason(tmp_path):
    class Untrainable(EquivalentView):
        def support(self):
            return TunableSupport(trainable=False, reason="remote execution only")

    fts = FinetuningScheduler(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True)
    with pytest.raises(FTSConfigurationError, match="remote execution only"):
        _trainer(tmp_path, [fts]).fit(_viewed_model(Untrainable))


def test_a_legacy_naming_adapter_takes_precedence_over_an_attached_view(tmp_path, monkeypatch):
    """A configured adapter with parameter-naming overrides keeps naming parameters (its pinned path is unchanged),
    and says once that it ignores the attached view."""
    monkeypatch.setattr(adapter_base, "_PRECEDENCE_NOTICED", set())
    seed_everything(42)
    fts, recorder = _fts(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True), PathRecorder()
    with pytest.warns(FutureWarning, match=NAMING_OVERRIDES_PRECEDENCE_WARN):
        _trainer(tmp_path, [fts, recorder]).fit(_viewed_model())
    assert type(fts.strategy_adapter) is LegacyNamingAdapter and fts.strategy_adapter.compiled_schedule is None
    assert recorder.rows == EXPECTED_PATH
    with warnings.catch_warnings(record=True) as again:  # once per adapter class
        warnings.simplefilter("always")
        _trainer(tmp_path / "again", [_fts(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True)]).fit(
            _viewed_model())
    assert not [w for w in again if "fts_tunable_view" in str(w.message)]


def test_attached_view_generates_its_schedule_over_logical_names(tmp_path):
    fts = FinetuningScheduler(gen_ft_sched_only=True)
    with pytest.raises(SystemExit):
        _trainer(tmp_path, [fts], logger=True).fit(_viewed_model())
    sched_file = tmp_path / "lightning_logs" / "version_0" / "LogicalNamesBoringModel_ft_schedule.yaml"
    generated = yaml.safe_load(sched_file.read_text())
    assert {n for phase in generated.values() for n in phase["params"]} == {"head.W", "blocks.1.W", "blocks.0.W"}


def test_naming_overrides_classification():
    """Naming overrides are measured against the nearest built-in adapter: FSDP's flat-parameter transform is not
    one."""

    class SubclassesFSDP(FSDPStrategyAdapter):
        def logical_param_translation(self, param_names):
            return param_names

    assert StrategyAdapter.naming_overrides() == frozenset()
    assert FSDPStrategyAdapter.naming_overrides() == frozenset()
    assert SubclassesFSDP.naming_overrides() == {"logical_param_translation"}
    assert LegacyNamingAdapter.naming_overrides() <= set(NAMING_METHODS)
    assert {"fts_optim_transform", "logical_param_translation"} <= LegacyNamingAdapter.naming_overrides()


def test_fsdp1_refuses_an_attached_view():
    adapter = FSDPStrategyAdapter()
    module = _viewed_model()
    adapter.fts_handle = SimpleNamespace(pl_module=module, trainer=None)
    with pytest.raises(MisconfigurationException, match="FSDP1.*does not support"):
        adapter._refuse_attached_view()
    module.fts_tunable_view = CanonicalView(module)  # the identity view changes nothing, so it is accepted
    adapter._refuse_attached_view()


def test_the_promote_key_is_refused_under_lightning(tmp_path):
    schedule = {**LOGICAL_SCHEDULE, 1: {**LOGICAL_SCHEDULE[1], "promote": {"role:instrument": "latent:x"}}}
    fts = FinetuningScheduler(ft_schedule=schedule, epoch_transitions_only=True)
    with pytest.raises(FTSConfigurationError, match="phase 1: promote"):
        _trainer(tmp_path, [fts]).fit(_viewed_model())


def test_naming_overrides_are_deprecated_once_per_class(tmp_path, monkeypatch):
    """An adapter class overriding the naming methods hears it is deprecated once, at resolution; built-in adapters
    never do."""
    monkeypatch.setattr(adapter_base, "_DEPRECATION_NOTICED", set())
    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        _trainer(tmp_path, [_fts(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True)]).fit(
            LogicalNamesBoringModel())
        _trainer(tmp_path / "again", [_fts(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True)]).fit(
            LogicalNamesBoringModel())
    notices = [w for w in first if re.search(NAMING_OVERRIDES_DEPRECATION_WARN, str(w.message))]
    assert len(notices) == 1 and notices[0].category is FutureWarning
    assert "`LegacyNamingAdapter`" in str(notices[0].message)
    for builtin in (StrategyAdapter, FSDPStrategyAdapter):
        with warnings.catch_warnings(record=True) as quiet:
            warnings.simplefilter("always")
            builtin.notice_naming_overrides()
        assert not quiet


class ShardedLogicalModel(LogicalNamesBoringModel):
    """The logical-names model, each layer and then the model sharded with FSDP2 over the data-parallel mesh."""

    def configure_model(self):
        from torch.distributed.fsdp import fully_shard

        dp_mesh = self.device_mesh["data_parallel"]
        for layer in self.model:
            fully_shard(layer, mesh=dp_mesh)
        fully_shard(self.model, mesh=dp_mesh)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_attached_view_under_model_parallel_strategy(tmp_path):
    """Under ``ModelParallelStrategy`` (FSDP2 on two GPUs), an attached view schedules the module as the legacy
    adapter's pinned path does, validated through the FSDP2 distributed adapter, and every thawed tensor trains."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import ModelParallelStrategy

    seed_everything(42)
    module = ShardedLogicalModel()
    module.fts_tunable_view = EquivalentView(module)
    initial = _initial(module)
    fts, recorder = FinetuningScheduler(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True), PathRecorder()
    trainer = Trainer(default_root_dir=tmp_path, callbacks=[fts, recorder], max_epochs=4, accelerator="gpu", devices=2,
                      strategy=ModelParallelStrategy(data_parallel_size=2, tensor_parallel_size=1),
                      limit_train_batches=2, limit_val_batches=2, enable_progress_bar=False,
                      enable_model_summary=False, logger=False)
    trainer.fit(module)
    assert fts.strategy_adapter.distributed_adapter.name == "fsdp2"
    assert recorder.rows == EXPECTED_PATH
    assert fts.ft_schedule[1]["params"] == ["blocks.1.W"]
    trained = {n: p.full_tensor().detach().cpu() for n, p in module.named_parameters() if p.requires_grad}
    assert trained and all(not torch.equal(initial[n], t) for n, t in trained.items())


def test_scheduled_physical_translates_through_naming_overrides():
    """An adapter with naming overrides (and no compiled schedule) reports a phase's physical parameters through
    its own transform, so layout validation sees the real tensors."""
    adapter = LegacyNamingAdapter()
    module = LogicalNamesBoringModel()
    adapter.fts_handle = SimpleNamespace(pl_module=module, trainer=None, ft_schedule=LOGICAL_SCHEDULE)
    adapter.on_before_init_fts()
    assert adapter.compiled_schedule is None
    assert adapter.scheduled_physical(1) == ["model.2.weight", "model.2.bias", "model.3.weight", "model.3.bias"]
    plain = StrategyAdapter()
    plain.fts_handle = SimpleNamespace(pl_module=module, trainer=None, ft_schedule={0: {"params": ["model.4.weight"]}})
    assert plain.scheduled_physical(0) == ["model.4.weight"]
