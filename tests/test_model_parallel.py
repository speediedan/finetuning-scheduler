import os
from copy import deepcopy, copy
from logging import DEBUG
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, KeysView
from unittest import mock
from functools import partialmethod
from dataclasses import dataclass, field

import pytest
import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.distributed.tensor.parallel import loss_parallel
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping
from finetuning_scheduler.strategy_adapters import ModelParallelStrategyAdapter
from tests.helpers.boring_models import FTSToyTransformer, TestModelArgs, FTSWikiText2
from tests.helpers.common import (ExpectedResults, fts_check_warns, pytest_param_factory, get_fts,
                                  default_fts_sanity_chk, DeviceMeshSummary)

from tests.model_parallel_expected_paths import (path_tt_fsdp_tp,
                                                 path_tt_fsdp_no_tp, path_tt_tp_no_fsdp)

from tests.helpers.runif import RunIf

from tests.test_finetuning_scheduler_callback import (
    EXPECTED_WARNS,
    FinetuningSchedulerBoringModel,
    TestFinetuningScheduler,
    get_sched_fixture_tmpdir,
)

FTS_GLOBAL_STATE_LOG_MODE = os.environ.get("FTS_GLOBAL_STATE_LOG_MODE", "0") == "1"
MODEL_PARALLEL_BASE_WARNS = copy(EXPECTED_WARNS)
additional_model_parallel_warns = [
    "model contains an instance of `UninitializedParameter`",
    "The number of training batches",  # minimizing cost of training for these tests
    #"Please use torch.distributed.all_gather_into_tensor",  # still required for PyTorch/Lightning <=2.1
    #"Please use torch.distributed.reduce_scatter_tensor",  # still required for PyTorch/Lightning <=2.1
    "when logging on epoch level in distributed",  # validating FTS handling in this scenario
    "torch.cpu.amp.autocast",  # required as of PT 2.4
    "FSDP.state_dict_type", # temporarily required until Lightning uses new FSDP state dict API with PT 2.4
    "Final phase max_transition_epoch",  # required for some experimental dtensor tests with PT 2.4
    "interactive_bk attribute",  # TODO: remove, only for temporary debugging with torch from source
]
MODEL_PARALLEL_BASE_WARNS.extend(additional_model_parallel_warns)
MODEL_PARALLEL_DYNAMO_EXPECTED_WARNS = [
    "Final phase max_transition_epoch",  # still required for PyTorch/Lightning <=2.4
]

################################################################################
# Model Parallel Test Models
################################################################################

class FTSBaseModelParallel(FinetuningSchedulerBoringModel):
    def __init__(self, fsdp_plan: Dict, tp_plan: Dict | Callable,
                 module_cls: nn.Module = FTSToyTransformer, loss_parallel: bool = True,
                 tt_cfg: Optional[TestModelArgs] = None,
                 precision_key: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fsdp_plan = fsdp_plan or {}
        self.tp_plan = tp_plan or {}
        self.precision_key = precision_key
        self.tt_cfg = tt_cfg
        self.loss_parallel = loss_parallel
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

    def on_train_epoch_end(self) -> None:
        self._assert_fsdp_state()

    def on_val_epoch_end(self) -> None:
        self._assert_fsdp_state()

    def validation_step(self, batch: Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        inputs, target = batch
        output = self(inputs)
        # TODO: for now, not using diverge_on_epoch for simplicity
        loss = self.loss_fn(output, target)
        self.validation_step_outputs.append(loss)
        # we would normally use sync_dist for epoch-only logging in a distributed context but leaving it `False` here
        # to test FTS transition behavior when the test model is used in a distributed context
        self.log(self.monitor_metric, loss, prog_bar=False)
        return {"x": loss}

    def configure_model(self) -> None:
        if self.tp_plan:
            tp_mesh = self.device_mesh["tensor_parallel"]
            if callable(self.tp_plan):
                self.model = self.tp_plan(self.model, tp_mesh, self.loss_parallel)
            else:
                self.model = parallelize_module(self.model, tp_mesh, self.tp_plan)
        if self.fsdp_plan:
            from torch.distributed._composable.fsdp.fully_shard import fully_shard
            dp_mesh = self.device_mesh["data_parallel"]
            assert dp_mesh.ndim == 1  # Hybrid-sharding not supported

            for n, m in self.named_modules():
                if n in self.fsdp_plan["sharded_mods"]:
                    fully_shard(m, mesh=dp_mesh)
            fully_shard(self.model, mesh=dp_mesh)

    def _assert_fsdp_state(self) -> None:
        precision = None
        if self.precision_key == "auto_16":
            assert isinstance(self.trainer.strategy.precision_plugin, FSDPPrecision)  # TODO: Or maybe just Precision
            precision = torch.float16 if self.trainer.precision == "16-true" else torch.bfloat16
        if self.fsdp_plan:
            for n, m in self.named_modules():
                if n:  # if m is not self, the lightning module
                    # we currently shard the outer user model so all submodules should be fsdp managed if fsdp is used
                    assert getattr(m, '_is_fsdp_managed_module', False)
                if n in self.fsdp_plan["sharded_mods"] or n == 'model':
                    self._inspect_composable_fsdp_state(m, precision)
                else:
                    assert not issubclass(type(m), FSDPModule)  # orig module should not be composed with FSDPModule

    def _inspect_composable_fsdp_state(self, m: nn.Module, precision: Optional[torch.dtype] = None) -> None:
            assert issubclass(type(m), FSDPModule)  # orig module should be composed with FSDPModule
            mod_fsdp_state = m._get_fsdp_state()
            mixed_prec_state = mod_fsdp_state._mp_policy
            if self.precision_key:
                assert mixed_prec_state.param_dtype == precision
                assert mixed_prec_state.reduce_dtype == precision
                assert mixed_prec_state.output_dtype == precision
                # assert mixed_prec_state.cast_forward_inputs == True  # not currently inspected
            for fsdp_p in mod_fsdp_state._fsdp_param_group.fsdp_params:
                # test currently assumes 1D sharding on dim 0
                dp_dim0 = self.trainer.strategy.device_mesh['data_parallel'].shape[0]
                assert fsdp_p.sharded_size[0] == fsdp_p._orig_size[0] // dp_dim0

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


################################################################################
# Model Parallel Specially Instrumented FTS Versions
################################################################################

class ModelParallelTestFTS(TestFinetuningScheduler):

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
    # 3. Parallelize the final linear output layer
    non_transformerblock_tp_plan = {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1),
            ),
            "pos_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(0)),
            "norm": SequenceParallel(),
        }

    model = parallelize_module(model, tp_mesh, non_transformerblock_tp_plan)

    # N.B. We're not supporting Float8 parallelism in this test initially (e.g. Float8RowwiseParallel)

    # TODO: transform this block into a separate transformer_block_plan function that returns updated
    # transformerblock and transformerblock_tp_plan
    # TODO: adjust the layer plan to use new "tpattentionheadparallel" and transformerblockparallel functions so we can
    #       define a single tp_plan dict for the entire transformer
    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    #for layer_id, transformer_block in model.layers.items():
    # support
    # we currently support `ModuleList` and `ModuleDict` transformer_block containers
    # if isinstance(model.layers, nn.ModuleList):
    #     module_iterable = model.layers
    # elif isinstance(model.layers, nn.ModuleDict):
    #     module_iterable = model.layers.values()
    # else:
    #     raise "Unsupported transformer_block container, expected `ModuleList` or `ModuleDict` model.layers"
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

        # # Adjust attention module to use the local number of heads
        # attn_layer = transformer_block.attention
        # attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        # attn_layer.n_kv_heads = getattr(attn_layer, 'n_kv_heads', attn_layer.n_heads) // tp_mesh.size()

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
tt_mod_parallel = FTSBaseModelParallel

## DTensor Placement Plan Aliases
# TODO: set tp_plan and model loss_parallel from same config
tt_tp_plan = gen_apply_transformer_tp_plan

## FSDP2 Model Configuration Aliases
shard_tt_basic = {"sharded_mods": ['model.layers.1', 'model.norm', 'model.output']}

## toy transformer cfgs
basic_tt = TestModelArgs()

## Model Parallel Model Configuration Aliases
tt_fsdp_tp = {"fsdp_plan": shard_tt_basic, "tp_plan": tt_tp_plan, "module_cls": FTSToyTransformer, "tt_cfg": basic_tt}
tt_fsdp_no_tp = {"fsdp_plan": shard_tt_basic, "tp_plan": None, "module_cls": FTSToyTransformer, "tt_cfg": basic_tt}
tt_tp_no_fsdp = {"fsdp_plan": None, "tp_plan": tt_tp_plan, "module_cls": FTSToyTransformer, "tt_cfg": basic_tt}
tt_tp_no_fsdp_lp = {**tt_tp_no_fsdp, "loss_parallel": True}
tt_tp_no_fsdp_no_lp = {**tt_tp_no_fsdp, "loss_parallel": False}

## Model Parallel Strategy Aliases
dp1_tp2 = {"data_parallel_size": 1, "tensor_parallel_size": 2}
dp2_tp1 = {"data_parallel_size": 2, "tensor_parallel_size": 1}

## Lightning Trainer Configuration Aliases
trainer_defaults = {"accelerator": "gpu", "devices": 2, 'limit_train_batches': 2, 'limit_val_batches': 2,
                    'num_sanity_val_steps': 0}
# no_sanity_val = {"num_sanity_val_steps": 0}
# max_epoch_4 = {"max_epochs": 4}

## Precision Configuration Aliases
fp16 = {"precision": "16-true"}
bf16 = {"precision": "bf16-true"}

## cust ckpt cfg
no_ckpt_save = {"save_top_k": 0}

## cust FTS configuration aliases
max_depth_0 = {"max_depth": 0}
no_restore_best = {"restore_best": False}


## Model Parallel Test Configuration Dataclass

# TODO: update this to use dataclass inheritance once python 3.10 is the minimum supported version of python
@dataclass
class ModelParallelTestConfig:
    model_cfg_key: str
    model_cls: Callable
    model_cfg: Dict = field(default_factory=dict)
    trainer_cfg: Dict = field(default_factory=lambda: {'max_epochs': 3})
    strategy_cfg: Dict = field(default_factory=dict)
    strategy_adapter_cfg: Dict = field(default_factory=dict)
    precision_opts: Dict = field(default_factory=lambda: {'precision': '32-true'})
    auto_wrap_policy: Optional[Callable] = None
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

@mock.patch("finetuning_scheduler.strategy_adapters.model_parallel._TORCH_GREATER_EQUAL_2_5", False)
def test_torch_greater_equal_2_5():
    with pytest.raises(MisconfigurationException, match="requires PyTorch 2.5 or higher"):
        ModelParallelStrategyAdapter()

## Model Parallel Test Definitions
FTS_MODEL_PARALLEL_PATH_TESTS = (
    ModelParallelTestConfig(model_cfg_key="path_tt_fsdp_tp", model_cls=tt_mod_parallel,
                            model_cfg=tt_fsdp_tp, fts_cfg=no_restore_best, ckpt_cfg=no_ckpt_save,
                            strategy_cfg=dp2_tp1, runif_alias="einsum_exp",
                            expected_results=ExpectedResults(expected_state=path_tt_fsdp_tp)),
    ModelParallelTestConfig(model_cfg_key="path_tt_fsdp_no_tp", model_cls=tt_mod_parallel,
                            model_cfg=tt_fsdp_no_tp, strategy_cfg=dp2_tp1, runif_alias="alone",
                            expected_results=ExpectedResults(expected_state=path_tt_fsdp_no_tp)),
    ModelParallelTestConfig(model_cfg_key="path_tt_tp_no_fsdp_lp", model_cls=tt_mod_parallel,
                            model_cfg=tt_tp_no_fsdp_lp, strategy_cfg=dp1_tp2, runif_alias="alone",
                            expected_results=ExpectedResults(expected_state=path_tt_tp_no_fsdp)),
    ModelParallelTestConfig(model_cfg_key="path_tt_tp_no_fsdp_no_lp", model_cls=tt_mod_parallel,
                            model_cfg=tt_tp_no_fsdp_no_lp, strategy_cfg=dp1_tp2, runif_alias="alone",
                            expected_results=ExpectedResults(expected_state=path_tt_tp_no_fsdp)),
    ModelParallelTestConfig(model_cfg_key="tt_fsdp_no_tp_fp16", model_cls=tt_mod_parallel,
                            fts_cfg=no_restore_best,
                            precision_opts=fp16,
                            model_cfg=tt_fsdp_no_tp, strategy_cfg=dp2_tp1, runif_alias="alone"),
    # ModelParallelTestConfig(model_cfg_key="tt_tp_no_fsdp_bf16", model_cls=tt_mod_parallel, precision_opts=bf16,
    #                         model_cfg=tt_tp_no_fsdp_lp, strategy_cfg=dp1_tp2, runif_alias="bf16_alone"),
    # ModelParallelTestConfig(model_cfg_key="tt_tp_no_fsdp_fp16", model_cls=tt_mod_parallel, precision_opts=fp16,
    #                         model_cfg=tt_tp_no_fsdp_no_lp, strategy_cfg=dp1_tp2, runif_alias="alone")
)
@RunIf(min_cuda_gpus=2, min_torch="2.5.0")
@pytest.mark.parametrize("test_cfg", pytest_param_factory(FTS_MODEL_PARALLEL_PATH_TESTS))
def test_fts_model_parallel_integration(tmpdir, recwarn, model_parallel_ft_schedule, test_cfg):
    """Validate :class:`~finetuning_scheduler.FinetuningScheduler` functions properly in a supported 'ddp'
    distributed context."""
    seed_everything(42)
    # one can manually set this to True for a local test override
    state_log_dir = tmpdir if FTS_GLOBAL_STATE_LOG_MODE else None
    use_dynamo = True if test_cfg.model_cfg.pop("use_dynamo", None) else False
    ft_sched = model_parallel_ft_schedule[test_cfg.ft_sched_idx]
    callbacks = callbacks_cfg(ft_sched, state_log_dir, test_cfg)
    strategy = ModelParallelStrategy(**test_cfg.strategy_cfg)
    trainer_config = {"strategy": strategy, "callbacks": callbacks, "default_root_dir": tmpdir, **trainer_defaults,
                      **test_cfg.trainer_cfg, **test_cfg.precision_opts}
    trainer = Trainer(**trainer_config)
    with trainer.init_module(empty_init=True):
        model = test_cfg.model_cls(**test_cfg.model_cfg)  # TODO: verify updated tt_cfg is applied here
    configured_model = torch.compile(model) if use_dynamo else model
    if exc_expect := test_cfg.expected_results.exceptions_expected:
        gen_exceptions(trainer, configured_model, test_cfg.model_cfg_key, exc_expect)
    else:
        trainer.fit(configured_model)
        default_fts_sanity_chk(trainer)
    if trainer.is_global_zero:
        fts_check_warns(recwarn, expected_warns=MODEL_PARALLEL_BASE_WARNS,
                        warns_expected=test_cfg.expected_results.warns_expected,
                        expected_warns_dynamo=MODEL_PARALLEL_DYNAMO_EXPECTED_WARNS, use_dynamo=use_dynamo)

def gen_exceptions(trainer, model, exception_expected):
    with pytest.raises(MisconfigurationException, match=exception_expected):
            trainer.fit(model)


def callbacks_cfg(ft_sched, state_log_dir, test_cfg):
    active_fts_cfg = {"ft_schedule": ft_sched, "expected_state": test_cfg.expected_results.expected_state,
                      'state_log_dir': state_log_dir, **test_cfg.fts_cfg}
    callbacks = [test_cfg.fts_cls(**active_fts_cfg)]
    for tcls, tcfg, subc in zip((test_cfg.es_cls, test_cfg.ckpt_cls), (test_cfg.es_cfg, test_cfg.ckpt_cfg),
                                (FTSEarlyStopping, FTSCheckpoint)):
        if issubclass(tcls, subc):
            callbacks.append(tcls(**tcfg))
    return callbacks