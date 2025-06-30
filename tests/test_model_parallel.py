import os
from typing import Any, Callable, Dict, Optional, Tuple, KeysView, Union
from copy import deepcopy
from logging import DEBUG
from pathlib import Path
from functools import partialmethod
from dataclasses import dataclass, field
from itertools import chain

import pytest
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.experimental import implicit_replication
from torch.nn.attention import SDPBackend
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel,
                                                SequenceParallel, parallelize_module, loss_parallel)

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT

from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping
from finetuning_scheduler.strategy_adapters.model_parallel import ActCkptEnum
from tests.helpers.boring_models import FTSToyTransformer, TestModelArgs, FTSWikiText2
from tests.helpers.common import (ExpectedResults, fts_check_warns, pytest_param_factory, get_fts,
                                  default_fts_sanity_chk, DeviceMeshSummary)
from tests.model_parallel_expected_paths import (path_fsdp_autocm, path_tp_fsdp_autocm,
                                                 path_fsdp, path_tp)
from tests.helpers.runif import RunIf
from tests.helpers.expected_warns import MODEL_PARALLEL_BASE_WARNS, MODEL_PARALLEL_DYNAMO_EXPECTED_WARNS
from tests.test_finetuning_scheduler_callback import (
    FinetuningSchedulerBoringModel,
    TestFinetuningScheduler,
    get_sched_fixture_tmpdir,
)

FTS_GLOBAL_STATE_LOG_MODE = os.environ.get("FTS_GLOBAL_STATE_LOG_MODE", "0") == "1"


class FSDPStateInspectMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_fsdp2_modules = None

    def on_train_start(self, trainer, pl_module) -> None:
        if self._uses_fsdp2:
            self.expected_fsdp2_modules = set(
                chain(getattr(self.pl_module, 'cm_fsdp_plan', {}).keys(),
                      getattr(self.strategy_adapter, "fsdp_plan", {}).keys()))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._uses_fsdp2:
            self._assert_fsdp_state()

    def on_val_epoch_end(self, trainer, pl_module) -> None:
        if self._uses_fsdp2:
            self._assert_fsdp_state()

    @property
    def _uses_fsdp_name_based_plan(self) -> bool:
        return any(self.strategy_adapter_cfg.get(k) for k in ("fsdp_plan", "fsdp_default_kwargs"))

    @property
    def _uses_fsdp2(self) -> bool:
        return any((self.pl_module.cm_fsdp_plan, self._uses_fsdp_name_based_plan))

    def _assert_fsdp_state(self) -> None:
        precision = None
        if self.pl_module.precision_key == "auto_16":
            assert isinstance(self.trainer.strategy.precision_plugin, FSDPPrecision)
            precision = torch.float16 if self.trainer.precision == "16-true" else torch.bfloat16
        for n, m in self.pl_module.named_modules():
            if n in self.expected_fsdp2_modules:
                self._inspect_composable_fsdp_state(m, precision)
            else:
                assert not issubclass(type(m), FSDPModule)  # orig module should not be composed with FSDPModule

    def _inspect_composable_fsdp_state(self, m: nn.Module, precision: Optional[torch.dtype] = None) -> None:
            assert issubclass(type(m), FSDPModule)  # orig module should be composed with FSDPModule
            mod_fsdp_state = m._get_fsdp_state()
            mixed_prec_state = mod_fsdp_state._mp_policy
            if self.pl_module.precision_key:
                assert mixed_prec_state.param_dtype == precision
                assert mixed_prec_state.reduce_dtype == precision
                assert mixed_prec_state.output_dtype == precision
                # assert mixed_prec_state.cast_forward_inputs == True  # not currently inspected
            for fsdp_p in mod_fsdp_state._fsdp_param_group.fsdp_params:
                # test currently assumes 1D sharding on dim 0
                dp_dim0 = self.trainer.strategy.device_mesh['data_parallel'].shape[0]
                assert fsdp_p.sharded_size[0] == fsdp_p._orig_size[0] // dp_dim0

    def _collect_fsdp_mod_states(self, fsdp_keys: KeysView) -> Dict[Any, Dict]:
        fsdp_mod_states = {}
        if len(fsdp_keys) > 0:
            for n, m in self.pl_module.named_modules():
                if n in fsdp_keys:
                    fsdp_mod_states.setdefault(n, {})
                    fsdp_mod_states[n]['is_fsdp_managed'] = getattr(m, '_is_fsdp_managed_module', False)
                    fsdp_mod_states[n]['is_fsdp_composed'] = issubclass(type(m), FSDPModule)
                    if fsdp_mod_states[n]['is_fsdp_composed']:
                        mod_fsdp_state = m._get_fsdp_state()
                        fsdp_mod_states[n]['prec_policy_summ'] = (
                            mod_fsdp_state._mp_policy.param_dtype, mod_fsdp_state._mp_policy.reduce_dtype,
                            mod_fsdp_state._mp_policy.output_dtype, mod_fsdp_state._mp_policy.cast_forward_inputs)
                        fsdp_mod_states[n]['param_group_summ'] = (
                            [(fsdp_p._param_fqn, fsdp_p._orig_size, fsdp_p.sharded_size) for fsdp_p in \
                             mod_fsdp_state._fsdp_param_group.fsdp_params])
        return fsdp_mod_states


################################################################################
# Model Parallel Test Models
################################################################################

class FTSBaseModelParallel(FinetuningSchedulerBoringModel):
    def __init__(self, cm_fsdp_plan: Dict, cm_tp_plan: Union[Dict, Callable],
                 module_cls: nn.Module = FTSToyTransformer, loss_parallel: bool = True,
                 tt_cfg: Optional[TestModelArgs] = None,
                 precision_key: Optional[str] = None,
                 monitor_sync_dist: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cm_fsdp_plan = cm_fsdp_plan or {}
        self.cm_tp_plan = cm_tp_plan or {}
        self.precision_key = precision_key
        self.tt_cfg = tt_cfg
        self.loss_parallel = loss_parallel
        self.monitor_sync_dist = monitor_sync_dist
        self.module_cls = module_cls
        self.model = self.module_cls(self.tt_cfg)

    def loss_fn(self, output: Tensor, target: Tensor) -> Tensor:
        if self.loss_parallel:
            with loss_parallel():
                loss = F.cross_entropy(output.reshape(-1, output.size(-1)), target.reshape(-1))
        else:
            loss = F.cross_entropy(output.reshape(-1, output.size(-1)), target.reshape(-1))
        return loss

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.model(inputs, mask=mask)

    def backward(self, *args, **kwargs):
        if self.loss_parallel:
            with loss_parallel():
                super().backward(*args, **kwargs)
        else:
            super().backward(*args, **kwargs)

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)
        self.training_step_outputs.append(loss)
        return {"loss": loss}

    def validation_step(self, batch: Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        inputs, target = batch
        output = self(inputs)
        # TODO: for now, not using diverge_on_epoch for simplicity
        loss = self.loss_fn(output, target)
        self.validation_step_outputs.append(loss)
        self.log(self.monitor_metric, loss.item(), prog_bar=False, sync_dist=self.monitor_sync_dist)
        return {"x": loss}

    def prepare_data(self) -> None:
        FTSWikiText2(download=True)

    def setup(self, stage):
        self.dataset = FTSWikiText2()

    def get_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset, *args, **kwargs)

    train_dataloader = partialmethod(get_dataloader, batch_size=2)
    val_dataloader = partialmethod(get_dataloader, batch_size=2)
    test_dataloader = partialmethod(get_dataloader, batch_size=2)
    predict_dataloader = partialmethod(get_dataloader, batch_size=2)

class FTSCmModelParallel(FTSBaseModelParallel):
    def __init__(self, leave_auto_composable: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._leave_composable = leave_auto_composable

    def configure_model(self) -> None:
        if self.cm_tp_plan:
            tp_mesh = self.device_mesh["tensor_parallel"]
            if callable(self.cm_tp_plan):
                self.model = self.cm_tp_plan(self.model, tp_mesh, self.loss_parallel)
            else:
                self.model = parallelize_module(self.model, tp_mesh, self.cm_tp_plan)

        if self.cm_fsdp_plan:
            fts_cb = [cb for cb in self.trainer.callbacks if isinstance(cb, (FinetuningScheduler))][0]
            dp_mesh = self.device_mesh["data_parallel"]
            assert dp_mesh.ndim == 1  # Hybrid-sharding not supported

            cm_plan_keys = self.cm_fsdp_plan.keys()
            named_modules = dict(self.named_modules())
            for n in cm_plan_keys:
                if n in named_modules:
                    # we use our auto FSDP plan helpers to configure our manual configure_model plan
                    self.cm_fsdp_plan[n] = fts_cb.strategy_adapter._resolve_cfg_aliases(self.cm_fsdp_plan[n])
                    if act_ckpt := self.cm_fsdp_plan[n].pop('act_ckpt', None):
                        if act_ckpt.mode == ActCkptEnum.COMPOSABLE:
                            act_ckpt._func(self.get_submodule(n), **act_ckpt.cfg)
                        elif act_ckpt.mode in (ActCkptEnum.WRAPPED, ActCkptEnum.WRAPPED_OFFLOAD):
                            module_prefix, _, module_name = n.rpartition('.')
                            outer_mod = self.model if not module_prefix else self.get_submodule(module_prefix)
                            setattr(outer_mod, module_name, act_ckpt._func(self.get_submodule(n), **act_ckpt.cfg))
                    fully_shard(self.get_submodule(n), mesh=dp_mesh, **self.cm_fsdp_plan[n])

################################################################################
# Model Parallel Specially Instrumented FTS Versions
################################################################################

class ModelParallelTestFTS(FSDPStateInspectMixin, TestFinetuningScheduler):

    def __init__(self, ext_tensor_details: Optional[bool] = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ext_tensor_details = ext_tensor_details

    def state_dict(self) -> Dict[str, Any]:
        return super(TestFinetuningScheduler, self).state_dict()

    def restore_best_ckpt(self) -> None:
        super(TestFinetuningScheduler, self).restore_best_ckpt()
        self.restored_best_cnt += 1

    def on_train_epoch_start(self, trainer, pl_module):
        super(TestFinetuningScheduler, self).on_train_epoch_start(trainer, pl_module)
        model_parallel_sample = {}
        state_key = trainer.current_epoch
        if expected_epoch_state := self.expected_state.get(state_key):
            if target_p_keys := expected_epoch_state[0].get('p_states', {}).keys():
                model_parallel_sample['p_states'] = self._collect_p_states(target_p_keys)
            if target_mod_keys := expected_epoch_state[0].get('fsdp_mod_states', {}).keys():
                model_parallel_sample['fsdp_mod_states'] = self._collect_fsdp_mod_states(target_mod_keys)
            if target_mod_keys or target_p_keys:
                current_state = (
                    model_parallel_sample,
                    len(self._fts_state._curr_thawed_params),
                )
                lrs_state = None
                self.inspect_or_assert(current_state, lrs_state, state_key)

    def _collect_p_states(self, tp_keys: KeysView) -> Dict[Any, Dict]:
        p_states = {}
        if len(tp_keys) > 0:
            for n, p in self.pl_module.named_parameters():
                if n in tp_keys:
                    p_states.setdefault(n, {})
                    p_states[n]['requires_grad'] = p.requires_grad
                    p_states[n]['is_DTensor'] = isinstance(p, DTensor)
                    if self.ext_tensor_details:
                        p_states = self._extended_tensor_details(p_states, n, p)
        return p_states

    def _extended_tensor_details(self, p_states: Dict, n: str, p: Tensor) -> None:
        p_states[n]['dtype'] = p.dtype
        p_states[n]['orig_shape'] = p.shape
        if p_states[n]['is_DTensor']:
            p_states[n]['local_shape'] = getattr(p._local_tensor, 'shape', None)
            placement_summ = []
            for pl in p.placements:
                if pl.is_replicate():
                    placement_summ.append('replica')
                elif pl.is_shard():
                    placement_summ.append('shard(dim=' + str(pl.dim) + ')')
                else:
                    placement_summ.append(None)
            p_states[n]['device_mesh'] = DeviceMeshSummary(p.ndim, p.device_mesh.ndim, p.device_mesh.shape,
                                                        p.device_mesh.mesh_dim_names, placement_summ)
        return p_states


################################################################################
# Model Parallel FTS Test Fixtures
################################################################################

@pytest.fixture(scope="module")
def model_parallel_ft_schedule(tmpdir_factory) -> Tuple[Path, Dict]:
    """Generates a default fine-tuning schedule for 'implicit' testing, a modified one for 'explicit' mode and an
    epoch-driven transitions only one for epoch_transitions_only testing."""
    seed_everything(42)
    callbacks = [FinetuningScheduler(gen_ft_sched_only=True)]
    model = FinetuningSchedulerBoringModel()
    tmpdir, rank = get_sched_fixture_tmpdir(tmpdir_factory)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks, devices=1)
    unmod_schedule_file = tmpdir / "lightning_logs" / "version_0" / f"{model.__class__.__name__}_ft_schedule.yaml"
    # N.B. Though we run this fixture for each rank to avoid adding special logic to each distributed client test, we
    # only generate a schedule on rank 0, linking to it on the other ranks.
    if rank == 0:
        with pytest.raises(SystemExit):
            trainer.fit(model)
    mp_tp_sched_dict = get_fts(trainer).load_yaml_schedule(unmod_schedule_file)
    mp_tp_sched_dict[0]["params"] = [r"model.output.weight", r"model.norm.*"]
    mp_tp_sched_dict[0]["max_transition_epoch"] = 1
    mp_tp_sched_dict[1]["params"] = [r"model.layers.1.(feed_forward|ffn_norm|attention.w.*|attention_norm).*"]
    mp_tp_sched_dict[1]["max_transition_epoch"] = 2
    mp_tp_sched_dict[2]["params"] = [r"model.layers.0.(feed_forward|ffn_norm|attention.w.*|attention_norm).*",
                                     r"model.(pos_embeddings|tok_embeddings).weight"]
    mp_tp_sched_dict.pop(3)
    # TODO: remove this dbg schedule before initial release
    mp_tp_dbg_req_grad = deepcopy(mp_tp_sched_dict)
    mp_tp_dbg_req_grad[0]["params"] = [r"model.layers.1.attention_norm.*", r"model.layers.1.feed_forward.w2.*",
    ]
    mp_tp_dbg_req_grad[1] = {"params": [
        r"model.output.weight",
        r"model.norm.*",
        r"model.layers.[0-1].(ffn_norm|attention.w.*).*",
        r"model.layers.0.attention_norm.*",
        r"model.layers.[0-1].feed_forward.w1.*",
        r"model.layers.0.feed_forward.w2.*",
    ]}
    mp_tp_dbg_req_grad[2] = {"params": [r"model.pos_embeddings.weight",]}

    return (
        unmod_schedule_file,
        mp_tp_sched_dict,
        mp_tp_dbg_req_grad,
    )

# modified version of https://bit.ly/torchtitan_transformer_tp_plan
def gen_apply_transformer_tp_plan(model: nn.Module, device_mesh: DeviceMesh, loss_parallel: bool) -> nn.Module:
    """Apply tensor parallelism."""

    # we're only applying tensor parallelism, composable fsdp is applied subsequently elsewhere if requested
    tp_mesh = device_mesh["tensor_parallel"]

    # 1. Parallelize the embedding and shard its outputs
    # 2. Parallelize the root norm layer over the sequence dim
    non_transformerblock_tp_plan = {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1),),
            "pos_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(0)),
            "norm": SequenceParallel(),
        }

    model = parallelize_module(model, tp_mesh, non_transformerblock_tp_plan)

    # N.B. We're not supporting Float8 parallelism in this test initially (e.g. Float8RowwiseParallel)

    # Apply tensor + sequence parallelism to every transformer block
    for transformer_block in model.layers:
        layer_plan = {
            "attention": PrepareModuleInput(
                input_layouts=Shard(1),
                desired_input_layouts=Replicate(),
            ),
            "attention_norm": SequenceParallel(),
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward.w1": ColwiseParallel(input_layouts=Shard(1)),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    output_parallelize_plan = (
        ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        )
    )
    parallelize_module(model.output, tp_mesh, output_parallelize_plan)

    # Manually set output.weight so that parameters and gradients are shared.
    if model.init_args.weight_tying:
        model.output.weight = model.tok_embeddings.weight

    return model


################################################################################
# Model Parallel Test Configuration Aliases
################################################################################

## Model Aliases
fsdp_auto_mod_parallel = FTSBaseModelParallel  # model w/ automatic `configure_model` method (for FSDP)
cm_mod_parallel = FTSCmModelParallel  # model w/ manual `configure_model` method

## toy transformer cfgs
basic_tt = TestModelArgs()
sdp_math_impl_tt = TestModelArgs(avail_sdp_backends=[SDPBackend.MATH], sdp_use_implicit_replication=True)

## DTensor Placement Plan Aliases
cm_tt_tp_plan = gen_apply_transformer_tp_plan

## Auto and Manual FSDP Plan Aliases
fsdp_basic = {'model.output': {}, 'model.layers.1': {}, 'model.norm': {}}
fsdp_cm_shard_tt_basic_act = {'model.output': {'act_ckpt': ('composable', {})},
                              'model.layers.1': {'act_ckpt': ('composable',)}, 'model.norm': {}}
fsdp_auto_tt_basic_act = {'model.output': {'act_ckpt': ('wrapped_offload', {})},
                          'model.layers.1': {'act_ckpt': ('composableX',)},
                          'model.norm': {'act_ckpt': ('wrapped', {})}}
fsdp_autocm_shard_tt = {'^.*model.layers.0$': {}}
fsdp_tp_autocm_shard_tt = {**fsdp_autocm_shard_tt, 'model': {}}
# should warn the second spec cannot be applied and the third spec is a disallowed type
fsdp_autocm_shard_tt_warn = {**fsdp_autocm_shard_tt, 'model.layers.1.ffn_norm': {}, 'model.layers': {},
                            '^.*model.laye.*s$': {}}
# generate an unmatched auto plan spec misconfiguration exception
fsdp_autocm_shard_tt_err = {**fsdp_autocm_shard_tt,'^.*model.nomatch.*s$': {}}

## Model Parallel Model Configuration Aliases
fsdp_cm_only = {"cm_fsdp_plan": fsdp_basic, "cm_tp_plan": None, "module_cls": FTSToyTransformer, "tt_cfg": basic_tt}
fsdp_cm_act = {**fsdp_cm_only, "cm_fsdp_plan": fsdp_cm_shard_tt_basic_act}
tp_only = {"cm_fsdp_plan": None, "cm_tp_plan": cm_tt_tp_plan, "module_cls": FTSToyTransformer, "tt_cfg": basic_tt}
tp_no_lp = {**tp_only, "loss_parallel": False}
tp_lp_math_sdp_impl = {**tp_only, "tt_cfg": sdp_math_impl_tt}
fsdp_tp = {**tp_only, "cm_fsdp_plan": fsdp_basic}
fsdp_auto = {"cm_fsdp_plan": None, "cm_tp_plan": None, "module_cls": FTSToyTransformer, "tt_cfg": basic_tt}

## Model Parallel Strategy Adpater Configuration Aliases
fsdp_auto_only = {"fsdp_plan": fsdp_basic, "fsdp_default_kwargs":{}}
fsdp_auto_cpuoffld = {"fsdp_plan": fsdp_basic, "fsdp_default_kwargs":{'cpu_offload_policy': {}}}
fsdp_auto_act = {"fsdp_plan": fsdp_auto_tt_basic_act, "fsdp_default_kwargs":{}}
fsdp_autocm = {"fsdp_plan": fsdp_autocm_shard_tt_warn, "fsdp_default_kwargs":{}}
fsdp_autocm_err = {"fsdp_plan": fsdp_autocm_shard_tt_err, "fsdp_default_kwargs":{}}
fsdp_autocm_tp = {"fsdp_plan": fsdp_tp_autocm_shard_tt, "fsdp_default_kwargs":{}}

## Model Parallel Strategy Aliases
dp1_tp2 = {"data_parallel_size": 1, "tensor_parallel_size": 2}
dp2_tp1 = {"data_parallel_size": 2, "tensor_parallel_size": 1}

## Lightning Trainer Configuration Aliases
trainer_defaults = {"accelerator": "gpu", "devices": 2, 'limit_train_batches': 2, 'limit_val_batches': 2,
                    'num_sanity_val_steps': 0}

## Precision Configuration Aliases
fp16 = {"precision": "16-true"}
bf16 = {"precision": "bf16-true"}

## Model Parallel Test Configuration Dataclass

# TODO: update this to use dataclass inheritance once python 3.10 is the minimum supported version of python
@dataclass
class ModParallelTestCfg:
    model_cfg_key: str
    model_cls: Callable
    model_cfg: Dict = field(default_factory=dict)
    trainer_cfg: Dict = field(default_factory=lambda: {'max_epochs': 3})
    strategy_cfg: Dict = field(default_factory=lambda: dp2_tp1)
    strategy_adapter_cfg: Dict = field(default_factory=dict)
    precision_opts: Dict = field(default_factory=lambda: {'precision': '32-true'})
    use_dynamo: bool = False
    use_implicit_replication: bool = False
    fts_cls: Callable = ModelParallelTestFTS
    fts_cfg: Dict = field(default_factory=dict)
    ft_sched_idx: int = 1
    es_cls: Callable = FTSEarlyStopping
    es_cfg: Dict = field(default_factory=lambda: {"patience": 1})
    ckpt_cfg: Dict = field(default_factory=lambda: {"save_top_k": 3})
    ckpt_cls: Callable = FTSCheckpoint
    expected_results: ExpectedResults = ExpectedResults()
    runif_alias: Optional[str] = None

    def __post_init__(self):
        self.default_fts_cfg = {
            "logging_level": DEBUG,
            "strategy_adapter_cfg": self.strategy_adapter_cfg,
        }
        default_dep_cfg = {"monitor": "val_loss"}
        self.es_cfg = {**self.es_cfg, **default_dep_cfg}
        self.ckpt_cfg = {**self.ckpt_cfg, **default_dep_cfg}


## Model Parallel Test Definitions
FTS_MODEL_PARALLEL_PATH_TESTS = (
    # FSDP2 tests
    ModParallelTestCfg(model_cfg_key="fsdp_cm_only", model_cls=cm_mod_parallel, model_cfg=fsdp_cm_only,
                       # use_dynamo=True,  # TODO: test with dynamo once Lightning supports it with ModelParallel
                       runif_alias="alone", expected_results=ExpectedResults(expected_state=path_fsdp)),
    ModParallelTestCfg(model_cfg_key="fsdp_cm_act", model_cls=cm_mod_parallel, model_cfg=fsdp_cm_act,
                       expected_results=ExpectedResults(expected_state=path_fsdp), runif_alias="alone"),
    ModParallelTestCfg(model_cfg_key="fsdp_cm_fp16", model_cls=cm_mod_parallel, precision_opts=fp16,
                       model_cfg=fsdp_cm_only, runif_alias="alone"),
    # FSDP2 auto plan tests
    ModParallelTestCfg(model_cfg_key="fsdp_auto", model_cls=fsdp_auto_mod_parallel, model_cfg=fsdp_auto,
                       strategy_adapter_cfg=fsdp_auto_cpuoffld, runif_alias="alone",
                       expected_results=ExpectedResults(expected_state=path_fsdp)),
    # TODO: upstream bug currently preventing use of `fsdp_auto_cpuoffld` here w/ fp16
    ModParallelTestCfg(model_cfg_key="fsdp_auto_fp16", model_cls=fsdp_auto_mod_parallel, model_cfg=fsdp_auto,
                       precision_opts=fp16, strategy_adapter_cfg=fsdp_auto_only, runif_alias="alone",
                       expected_results=ExpectedResults(expected_state=path_fsdp)),
    ModParallelTestCfg(model_cfg_key="fsdp_auto_act", model_cls=fsdp_auto_mod_parallel,
                       strategy_adapter_cfg=fsdp_auto_act, model_cfg=fsdp_auto, runif_alias="alone",
                       expected_results=ExpectedResults(
                           expected_state=path_fsdp, warns_expected=("Unknown AC mode", "to filter out AC"))),
    ModParallelTestCfg(model_cfg_key="fsdp_autocm", model_cls=cm_mod_parallel, strategy_adapter_cfg=fsdp_autocm,
                       model_cfg=fsdp_cm_only, use_implicit_replication=True, runif_alias="alone",
                       expected_results=ExpectedResults(
                           expected_state=path_fsdp_autocm,
                           warns_expected=("parents has already registered", "has the unsupported type"))),
    ModParallelTestCfg(model_cfg_key="fsdp_autocm_unmatched", model_cls=cm_mod_parallel, model_cfg=fsdp_cm_only,
                       strategy_adapter_cfg=fsdp_autocm_err, runif_alias="alone", use_implicit_replication=True,
                       expected_results=ExpectedResults(expected_state=path_fsdp_autocm,
                                                        exceptions_expected=("did not match any named modules",))),
    # TP tests
    ModParallelTestCfg(model_cfg_key="tp_only", model_cls=cm_mod_parallel, model_cfg=tp_no_lp, strategy_cfg=dp1_tp2,
                       runif_alias="alone", expected_results=ExpectedResults(expected_state=path_tp)),
    ModParallelTestCfg(model_cfg_key="tp_lp_fp16", model_cls=cm_mod_parallel, precision_opts=fp16, model_cfg=tp_only,
                       strategy_cfg=dp1_tp2, runif_alias="alone",
                       expected_results=ExpectedResults(expected_state=path_tp)),
    # TODO: given that SDPBackend.MATH with lp causes (at least without special accommodation)
    #       https://github.com/pytorch/pytorch/pull/149764 as of PT 2.8, we are disabling this test until our
    #       infrastructure is upgraded and we can use SDPBackend.FLASH with bf16 on all our cards
    # ModParallelTestCfg(model_cfg_key="tp_lp_bf16", model_cls=cm_mod_parallel, precision_opts=bf16,
    #                    model_cfg=tp_lp_math_sdp_impl, strategy_cfg=dp1_tp2, runif_alias="bf16_alone"),
    # FSDP2 + TP (trivial submesh) tests
    # temporarily disabling this test as it triggers a hang in `fsdp_autocm_tp` about 10% of the time and
    # `fsdp_autocm_tp` and the marginal utility of `fsdp_tp` is minimal
    # ModParallelTestCfg(model_cfg_key="fsdp_tp", model_cls=cm_mod_parallel, model_cfg=fsdp_tp,
    # runif_alias="einsum_exp", expected_results=ExpectedResults(expected_state=path_tp_fsdp)),
    ModParallelTestCfg(model_cfg_key="fsdp_autocm_tp", model_cls=cm_mod_parallel, strategy_adapter_cfg=fsdp_autocm_tp,
                       model_cfg=fsdp_tp, runif_alias="einsum_exp",
                       expected_results=ExpectedResults(expected_state=path_tp_fsdp_autocm)),
)
@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize("test_cfg", pytest_param_factory(FTS_MODEL_PARALLEL_PATH_TESTS))
def test_fts_model_parallel_integration(tmpdir, recwarn, model_parallel_ft_schedule, test_cfg):
    """Validate :class:`~finetuning_scheduler.FinetuningScheduler` functions properly in a supported 'ddp'
    distributed context."""
    seed_everything(42)
    # one can manually set this to True for a local test override
    state_log_dir = tmpdir if FTS_GLOBAL_STATE_LOG_MODE else None
    ft_sched = model_parallel_ft_schedule[test_cfg.ft_sched_idx]
    callbacks = callbacks_cfg(ft_sched, state_log_dir, test_cfg)
    strategy = ModelParallelStrategy(**test_cfg.strategy_cfg)
    trainer_config = {"strategy": strategy, "callbacks": callbacks, "default_root_dir": tmpdir, **trainer_defaults,
                      **test_cfg.trainer_cfg, **test_cfg.precision_opts}
    trainer = Trainer(**trainer_config)
    with trainer.init_module(empty_init=True):
        model = test_cfg.model_cls(**test_cfg.model_cfg)
    configured_model = torch.compile(model) if test_cfg.use_dynamo else model
    if exc_expect := test_cfg.expected_results.exceptions_expected:
        gen_exceptions(trainer, configured_model, exc_expect)
    else:
        if test_cfg.use_implicit_replication:
            with implicit_replication():
                trainer.fit(configured_model)
        else:
            trainer.fit(configured_model)
        default_fts_sanity_chk(trainer)
    if trainer.is_global_zero:
        fts_check_warns(recwarn, expected_warns=MODEL_PARALLEL_BASE_WARNS,
                        warns_expected=test_cfg.expected_results.warns_expected,
                        expected_warns_dynamo=MODEL_PARALLEL_DYNAMO_EXPECTED_WARNS, use_dynamo=test_cfg.use_dynamo)

def gen_exceptions(trainer, model, exception_expected):
    if isinstance(exception_expected, tuple):
        exception_expected = "|".join(exception_expected)
    with pytest.raises(MisconfigurationException, match=exception_expected):
            trainer.fit(model)


def callbacks_cfg(ft_sched, state_log_dir, test_cfg):
    active_fts_cfg = {"ft_schedule": ft_sched, "strategy_adapter_cfg": test_cfg.strategy_adapter_cfg,
                      "expected_state": test_cfg.expected_results.expected_state, "state_log_dir": state_log_dir,
                      **test_cfg.fts_cfg, }
    callbacks = [test_cfg.fts_cls(**active_fts_cfg)]
    for tcls, tcfg, subc in zip((test_cfg.es_cls, test_cfg.ckpt_cls), (test_cfg.es_cfg, test_cfg.ckpt_cfg),
                                (FTSEarlyStopping, FTSCheckpoint)):
        if issubclass(tcls, subc):
            callbacks.append(tcls(**tcfg))
    return callbacks
