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
from torch.distributed._tensor.debug import CommDebugMode
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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping
from finetuning_scheduler.strategy_adapters import FSDPStrategyAdapter
from tests.helpers.boring_models import FTSToyTransformer, TestModelArgs, FTSWikiText2
from tests.helpers.common import (ExpectedResults, fts_check_warns, pytest_param_factory, get_fts,
                                  default_fts_sanity_chk, DeviceMeshSummary)

from tests.model_parallel_expected_paths import (path_ff_tp_no_fsdp, path_ff_fsdp_no_tp, path_ff_fsdp_tp,
                                                 path_tt_fsdp_no_tp, path_tt_tp_no_fsdp)

from tests.helpers.runif import RunIf

from tests.test_finetuning_scheduler_callback import (
    EXPECTED_WARNS,
    FinetuningSchedulerBoringModel,
    TestFinetuningScheduler,
    get_sched_fixture_tmpdir,
)

FTS_GLOBAL_STATE_LOG_MODE = os.environ.get("FTS_GLOBAL_STATE_LOG_MODE", "0") == "1"


if torch.distributed.is_available():
    from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel, MixedPrecision
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, wrap
else:
    FullyShardedDataParallel = None  # type: ignore[misc,assignment]
    MixedPrecision = None  # type: ignore[misc,assignment]
    BackwardPrefetch = None  # type: ignore[misc,assignment]
    CPUOffload = None  # type: ignore[misc,assignment]
    size_based_auto_wrap_policy = object
    wrap = object


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
]
MODEL_PARALLEL_BASE_WARNS.extend(additional_model_parallel_warns)
MODEL_PARALLEL_DYNAMO_EXPECTED_WARNS = [
    "Final phase max_transition_epoch",  # still required for PyTorch/Lightning <=2.4
]


# def _parallelize_base_model_parallel_tp(model, device_mesh):
#     from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

#     tp_mesh = device_mesh["tensor_parallel"]
#     tp_plan = {
#         "w1": ColwiseParallel(),
#         "w2": ColwiseParallel(),
#         "w3": RowwiseParallel(),
#     }
#     parallelize_module(model, tp_mesh, tp_plan)
#     return model

# def _parallelize_feed_forward_tp(model, device_mesh):
#     from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

#     tp_mesh = device_mesh["tensor_parallel"]
#     tp_plan = {
#         "w1": ColwiseParallel(),
#         "w2": ColwiseParallel(),
#         "w3": RowwiseParallel(),
#     }
#     parallelize_module(model, tp_mesh, tp_plan)
#     return model


# def _parallelize_base_model_parallel_fsdp2(model, device_mesh):
#     from torch.distributed._composable.fsdp.fully_shard import fully_shard

#     dp_mesh = device_mesh["data_parallel"]
#     assert dp_mesh.ndim == 1  # Hybrid-sharding not supported

#     # Fully-shard each layer
#     fully_shard(model.w1, mesh=dp_mesh)
#     fully_shard(model.w2, mesh=dp_mesh)
#     fully_shard(model.w3, mesh=dp_mesh)

#     # TODO: Re-enable activation checkpointing
#     # Currently, state dict keys get prefixed with '_checkpoint_wrapper' in the keys
#     # which leads to mismatches when loading weights into a checkpoint-wrapped module.
#     # PyTorch should handle this automatically.

#     # model = checkpoint_wrapper(model)

#     return model


# def _parallelize_base_model_parallel_fsdp2_tp(model, device_mesh):
#     model = _parallelize_base_model_parallel_tp(model, device_mesh)
#     model = _parallelize_base_model_parallel_fsdp2(model, device_mesh)
#     return model

# class DeviceMeshSummary(NamedTuple):
#     tensor_ndim: int
#     mesh_ndim: int
#     mesh_shape: Tuple
#     mesh_dim_names: Tuple
#     placement_summary: List[Optional[str | int]]

################################################################################
# Model Parallel Test Models
################################################################################

class FeedForward(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.w1 = nn.Linear(32, 64)
        self.w2 = nn.Linear(32, 64)
        self.w3 = nn.Linear(64, 2)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

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
        #output = self(inputs, target)
        with CommDebugMode() as comm_mode:
            output = self(inputs)
        #print(comm_mode.advanced_module_tracker.sharding_dict)
        #with CommDebugMode() as comm_mode:
            loss = self.loss_fn(output, target)
        #comm_mode.advanced_module_tracker.sharding_dict
        #loss = F.cross_entropy(output.reshape(-1, output.size(-1)), target.reshape(-1))
        self.training_step_outputs.append(loss)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        self._assert_fsdp_state()

    def on_val_epoch_end(self) -> None:
        self._assert_fsdp_state()

    def validation_step(self, batch: Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        inputs, target = batch
        #output = self(inputs, target)
        output = self(inputs)
        #loss = self.val_loss(batch, output)
        # TODO: for now, not using diverge_on_epoch for simplicity
        loss = self.loss_fn(output, target)
        #loss = F.cross_entropy(output.reshape(-1, output.size(-1)), target.reshape(-1))
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
        target_p_keys = self.expected_state[state_key][0]['p_states'].keys()
        model_parallel_sample['p_states'] = self._collect_p_states(target_p_keys)
        if target_mod_keys := self.expected_state[state_key][0].get('fsdp_mod_states', {}).keys():
            model_parallel_sample['fsdp_mod_states'] = self._collect_fsdp_mod_states(target_mod_keys)
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
    # TODO: NEXT: continue setting up ftstransformer schedule fixture!
    mp_tp_sched_dict = get_fts(trainer).load_yaml_schedule(unmod_schedule_file)
    mp_tp_sched_dict[0]["params"] = [r"model.output.weight", r"model.norm.*",
                                     #r"model.tok_embeddings.weight",
                                     #r"model.(pos_embeddings|tok_embeddings).weight",
                                     ]
    mp_tp_sched_dict[0]["max_transition_epoch"] = 1
    mp_tp_sched_dict[1]["params"] = [#r"model.pos_embeddings.weight",
                                     r"model.layers.[0-1].(feed_forward|ffn_norm|attention|attention_norm).*"]
    mp_tp_sched_dict[1]["max_transition_epoch"] = 2
    mp_tp_sched_dict[2]["params"] = [#r"model.layers.0.(feed_forward|ffn_norm|attention|attention_norm).*",
                                     r"model.(pos_embeddings|tok_embeddings).weight"]
    mp_tp_sched_dict.pop(3)
    # mp_tp_all_req_grad = deepcopy(mp_tp_sched_dict)
    # mp_tp_all_req_grad[0]["params"] = [r"model.output.weight", r"model.norm.*",
    #                                    r"model.layers.[0-1].(feed_forward|ffn_norm|attention|attention_norm).*"]
    # mp_tp_all_req_grad[1]["params"] = [r"model.(pos_embeddings|tok_embeddings).weight"]
    # mp_tp_all_req_grad.pop(2)
    mp_tp_all_req_grad = deepcopy(mp_tp_sched_dict)
    mp_tp_all_req_grad[0]["params"] = [r"model.output.weight", r"model.norm.*",
                                       r"model.layers.[0-1].(feed_forward|ffn_norm|attention|attention_norm).*",
                                       r"model.(pos_embeddings|tok_embeddings).weight"]
    #mp_tp_all_req_grad[1]["params"] = [r"model.(pos_embeddings|tok_embeddings).weight"]
    mp_tp_all_req_grad.pop(1)
    mp_tp_all_req_grad.pop(2)
    mp_tp_ln_no_grad = deepcopy(mp_tp_sched_dict)
    mp_tp_ln_no_grad[0]["params"] = [r"model.output.weight", r"model.norm.*"]
    mp_tp_ln_no_grad[1]["params"] = [r"model.layers.[0-1].(feed_forward|ffn_norm|attention|attention_norm).*"]
    mp_tp_ln_no_grad[2]["params"] = [r"model.(pos_embeddings|tok_embeddings).weight"]
    # mp_tp_sched_dict = {0: {'params': ['model.w3.bias', 'model.w3.weight']},
    #                  1: {'params': ['model.w2.bias', 'model.w2.weight']},
    #                  2: {'params': ['model.w1.bias', 'model.w1.weight']}}
    # mp_tp_sched_dict[0]["max_transition_epoch"] = 1
    # mp_tp_sched_dict[1]["max_transition_epoch"] = 2
    return (
        unmod_schedule_file,
        mp_tp_sched_dict,
        mp_tp_all_req_grad,
        mp_tp_ln_no_grad,
        #mp_tp_sched_dict
    )

# class FSDP2Model(FTSBaseModelParallel):

#     def configure_model(self):
#         _parallelize_base_model_parallel_fsdp2(self.model, device_mesh=self.device_mesh)

# class TensorParallelModel(FTSBaseModelParallel):
#     def configure_model(self):
#         _parallelize_base_model_parallel_tp(self.model, device_mesh=self.device_mesh)


# class FSDP2TensorParallelModel(FTSBaseModelParallel):
#     def configure_model(self):
#         _parallelize_base_model_parallel_fsdp2_tp(self.model, device_mesh=self.device_mesh)

# modified version of https://bit.ly/torchtitan_transformer_tp_plan
def gen_apply_transformer_tp_plan(model: nn.Module, device_mesh: DeviceMesh, loss_parallel: bool) -> nn.Module:
    """Apply tensor parallelism."""

    # we're only applying tensor parallelism, composable fsdp is applied subsequently elsewhere if requested
    tp_mesh = device_mesh["tensor_parallel"]
    #loss_parallel = enable_loss_parallel

    # 1. Parallelize the embedding and shard its outputs
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    non_transformerblock_tp_plan = {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1),
            ),
            "pos_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(0)),
            "norm": SequenceParallel(),
            # TODO: prob not necessary, inspect opportunities for refactoring/optimization
            # "layers.0": PrepareModuleInput(
            #     input_layouts=(Replicate(), None),
            #     desired_input_layouts=(Shard(1), None),
            #     use_local_output=True,
            # ),
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
    if isinstance(model.layers, nn.ModuleList):
        module_iterable = model.layers
    elif isinstance(model.layers, nn.ModuleDict):
        module_iterable = model.layers.values()
    else:
        raise "Unsupported transformer_block container, expected `ModuleList` or `ModuleDict` model.layers"
    for transformer_block in model.layers:
        layer_plan = {
            "attention": PrepareModuleInput(
                input_layouts=Shard(1),
                desired_input_layouts=Replicate(),
            ),
            "attention_norm": SequenceParallel(),
            # "attention": PrepareModuleInput(
            #     input_layouts=(Shard(1), None),
            #     desired_input_layouts=(Replicate(), None),
            # ),

            "attention.wq": ColwiseParallel(use_local_output=False), # try use_local_output=False ?
            "attention.wk": ColwiseParallel(use_local_output=False), # try use_local_output=False ?
            "attention.wv": ColwiseParallel(use_local_output=False),  # try use_local_output=False ?
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            # "feed_forward": PrepareModuleInput(
            #     input_layouts=(Shard(1),),
            #     desired_input_layouts=(Replicate(),),
            # ),
            #"feed_forward.w1": ColwiseParallel(),
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

        # "output": ColwiseParallel(
        #         input_layouts=Shard(1),
        #         output_layouts=Shard(-1) if loss_parallel else Replicate(),
        #         use_local_output=not loss_parallel,
        #     ),

    # Manually set output.weight so that parameters and gradients are shared.
    if model.init_args.weight_tying:
        model.output.weight = model.tok_embeddings.weight

    return model


################################################################################
# Model Parallel Test Configuration Aliases
################################################################################

## Model Aliases
ff_mod_parallel = FTSBaseModelParallel

tt_mod_parallel = FTSBaseModelParallel

## DTensor Placement Plan Aliases

basic_tp_plan = {
        "w1": ColwiseParallel(),
        "w2": ColwiseParallel(),
        "w3": RowwiseParallel(),
    }

# TODO: set tp_plan and model loss_parallel from same config
tt_tp_plan = gen_apply_transformer_tp_plan
#tt_tp_plan = partial(gen_apply_transformer_tp_plan, enable_loss_parallel=True)
# tt_tp_plan_no_loss_parallel = partial(gen_apply_transformer_tp_plan, enable_loss_parallel=False)

## FSDP2 Model Configuration Aliases

shard_all = {"sharded_mods": ['model.w1', 'model.w2', 'model.w3'], "unsharded_mods": []}
shard_tt_basic = {"sharded_mods": ['model.layers.1', 'model.norm', 'model.output'], "unsharded_mods": []}

## toy transformer cfgs

basic_tt = TestModelArgs()

## Model Parallel Model Configuration Aliases
tp_no_fsdp = {"fsdp_plan": None, "tp_plan": basic_tp_plan}
fsdp_no_tp = {"fsdp_plan": shard_all, "tp_plan": None}
fsdp_tp = {"fsdp_plan": shard_all, "tp_plan": basic_tp_plan}

tt_fsdp_no_tp = {"fsdp_plan": shard_tt_basic, "tp_plan": None, "module_cls": FTSToyTransformer, "tt_cfg": basic_tt}
tt_tp_no_fsdp_lp = {"fsdp_plan": None, "tp_plan": tt_tp_plan, "module_cls": FTSToyTransformer, "tt_cfg": basic_tt,
                 "loss_parallel": True}
tt_tp_no_fsdp_no_lp = {"fsdp_plan": None, "tp_plan": tt_tp_plan, "module_cls": FTSToyTransformer, "tt_cfg": basic_tt,
                 "loss_parallel": False}

## Model Parallel Strategy Aliases
dp1_tp2 = {"data_parallel_size": 1, "tensor_parallel_size": 2}
dp2_tp1 = {"data_parallel_size": 2, "tensor_parallel_size": 1}

## Lightning Trainer Configuration Aliases
trainer_defaults = {"accelerator": "gpu", "devices": 2, 'limit_train_batches': 2, 'limit_val_batches': 2,
                    'num_sanity_val_steps': 0}
no_sanity_val = {"num_sanity_val_steps": 0}
max_epoch_4 = {"max_epochs": 4}

## cust ckpt cfg
no_ckpt_save = {"save_top_k": 0}

## cust FTS configuration aliases
max_depth_0 = {"max_depth": 0}
no_restore_best = {"restore_best": False}

# with mock.patch.object(ModelCheckpoint, "_save_checkpoint"):

## Model Parallel Test Configuration Dataclass

# TODO: update this to use dataclass inheritance once python 3.10 is the minimum supported version of python
@dataclass
class ModelParallelTestConfig:
    model_cfg_key: str
    expected_results: ExpectedResults
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
FTS_MODEL_PARALLEL_TESTS = (
    ModelParallelTestConfig(model_cfg_key="ff_tp_no_fsdp", model_cls=ff_mod_parallel, model_cfg=tp_no_fsdp,
                            strategy_cfg=dp1_tp2, runif_alias="min2_4",
                            expected_results=ExpectedResults(expected_state=path_ff_tp_no_fsdp)),
    ModelParallelTestConfig(model_cfg_key="ff_fsdp_no_tp", model_cls=ff_mod_parallel,
                            model_cfg=fsdp_no_tp,
                            strategy_cfg=dp2_tp1, runif_alias="min2_4",
                            expected_results=ExpectedResults(expected_state=path_ff_fsdp_no_tp)),
    ModelParallelTestConfig(model_cfg_key="ff_fsdp_tp", model_cls=ff_mod_parallel,
                            model_cfg=fsdp_tp, fts_cfg=no_restore_best, ckpt_cfg=no_ckpt_save,
                            strategy_cfg=dp2_tp1, runif_alias="min2_4",
                            expected_results=ExpectedResults(expected_state=path_ff_fsdp_tp)),
    ModelParallelTestConfig(model_cfg_key="tt_fsdp_no_tp", model_cls=tt_mod_parallel,
                            model_cfg=tt_fsdp_no_tp, strategy_cfg=dp2_tp1, runif_alias="min2_4",
                            expected_results=ExpectedResults(expected_state=path_tt_fsdp_no_tp)),
    ModelParallelTestConfig(model_cfg_key="tt_tp_no_fsdp", model_cls=tt_mod_parallel,
                            model_cfg=tt_tp_no_fsdp_no_lp, strategy_cfg=dp1_tp2, runif_alias="min2_4",
                            expected_results=ExpectedResults(expected_state=path_tt_tp_no_fsdp)),
    # ModelParallelTestConfig(model_cfg_key="tt_tp_no_fsdp_no_lp", model_cls=tt_mod_parallel,
    #                         model_cfg=tt_tp_no_fsdp_no_lp, strategy_cfg=dp1_tp2, ft_sched_idx=1,
    #                         runif_alias="min2_4",
    #                         expected_results=ExpectedResults(expected_state=path_tt_tp_no_fsdp)),
    ModelParallelTestConfig(model_cfg_key="tt_tp_no_fsdp_no_lp_no_lnreq", model_cls=tt_mod_parallel,
                            model_cfg=tt_tp_no_fsdp_lp, strategy_cfg=dp1_tp2, ft_sched_idx=3,
                            runif_alias="min2_4",
                            expected_results=ExpectedResults(expected_state=path_tt_tp_no_fsdp)),
    ModelParallelTestConfig(model_cfg_key="tt_tp_no_fsdp_lp_all_req", model_cls=tt_mod_parallel,
                            ft_sched_idx=2, fts_cfg=max_depth_0,
                            model_cfg=tt_tp_no_fsdp_lp, strategy_cfg=dp1_tp2, runif_alias="min2_4",
                            expected_results=ExpectedResults(expected_state=path_tt_tp_no_fsdp)),
    ModelParallelTestConfig(model_cfg_key="tt_tp_no_fsdp_no_lp_all_req", model_cls=tt_mod_parallel,
                            model_cfg=tt_tp_no_fsdp_no_lp, strategy_cfg=dp1_tp2, ft_sched_idx=2, fts_cfg=max_depth_0,
                            runif_alias="min2_4",
                            expected_results=ExpectedResults(expected_state=path_tt_tp_no_fsdp)),

)


@RunIf(standalone=False, min_cuda_gpus=2)
@pytest.mark.parametrize("test_cfg", pytest_param_factory(FTS_MODEL_PARALLEL_TESTS))
def test_fts_model_parallel(tmpdir, recwarn, model_parallel_ft_schedule, test_cfg):
    """Validate :class:`~finetuning_scheduler.FinetuningScheduler` functions properly in a supported 'ddp'
    distributed context."""
    # some experimental tests may require version/os-env gated dependency patches to be applied, they may be loaded here
    if test_cfg.model_cfg_key in ("ff_fsdp_tp"):
        pass
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
    if test_cfg.expected_results.exceptions_expected:
        gen_exceptions(trainer, configured_model, test_cfg.model_cfg_key, test_cfg.expected_results.exceptions_expected)
    else:
        trainer.fit(configured_model)
        default_fts_sanity_chk(trainer)
    if trainer.is_global_zero:
        fts_check_warns(recwarn, expected_warns=MODEL_PARALLEL_BASE_WARNS,
                        warns_expected=test_cfg.expected_results.warns_expected,
                        expected_warns_dynamo=MODEL_PARALLEL_DYNAMO_EXPECTED_WARNS, use_dynamo=use_dynamo)


def gen_exceptions(trainer, model, model_cfg_key, exception_expected):
    if model_cfg_key == "no_fsdp_params_p0":
        with mock.patch.object(FSDPStrategyAdapter, "_rank_zero_logger", 42):
            with pytest.raises(MisconfigurationException, match=exception_expected):
                trainer.fit(model)
    else:
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




# @RunIf(min_torch="2.3", standalone=False, min_cuda_gpus=2)
# def test_fsdp2_trivial_tp():
#     from torch.distributed._tensor import DTensor

#     class Model(FSDP2TensorParallelModel):
#         def on_train_start(self):
#             optimizer = self.optimizers()
#             #assert all(isinstance(weight, DTensor) for weight in self.model.parameters())
#             #assert all(isinstance(tensor, DTensor) for tensor in optimizer.param_groups[0]["params"])
#             assert self.model.w1.weight.device_mesh.ndim == 2
#             assert self.model.w1.weight.device_mesh.size(0) == 2
#             assert self.model.w1.weight.device_mesh.size(1) == 1
#             #assert all(weight.device.type != "meta" for weight in self.model.parameters())
#             #assert all(tensor.device_mesh.ndim == 2 for tensor in optimizer.param_groups[0]["params"])
#             #assert all(tensor.device.type != "meta" for tensor in optimizer.param_groups[0]["params"])

#             # No data sharding across TP dimension, sharding across data-parallel dimension only
#             device_mesh = self.device_mesh
#             dp_mesh = device_mesh["data_parallel"]
#             dataloader = self.trainer.train_dataloader
#             assert len(dataloader) == 8 // dataloader.batch_size // dp_mesh.size()
#             assert isinstance(dataloader.sampler, DistributedSampler)

#         def training_step(self, batch):
#             batches = self.all_gather(batch)
#             dp_mesh = self.device_mesh["data_parallel"]
#             tp_mesh = self.device_mesh["tensor_parallel"]

#             # Batches across the TP dimension must be identical
#             batches_tp = batches[tp_mesh.mesh]
#             assert all(torch.equal(batches_tp[0], batches_tp[i]) for i in range(1, len(batches_tp)))
#             # Batches across the DP dimension must be different
#             batches_dp = batches[dp_mesh.mesh]
#             assert all(not torch.equal(batches_dp[0], batches_dp[i]) for i in range(1, len(batches_dp)))

#             return super().training_step(batch)

#     strategy = ModelParallelStrategy(
#         data_parallel_size=2,
#         tensor_parallel_size=1,
#     )
#     trainer = Trainer(
#         accelerator="auto",
#         devices=2,
#         strategy=strategy,
#         max_steps=2,
#         enable_checkpointing=False,
#         logger=False,
#     )

#     seed_everything(0)
#     with trainer.init_module(empty_init=True):
#         model = Model()

#     trainer.fit(model)


# @RunIf(min_torch="2.3", standalone=False, min_cuda_gpus=2)
# def test_fsdp2_notp():
#     #from torch.distributed._tensor import DTensor

#     class Model(FSDP2Model):
#         def on_train_start(self):
#             optimizer = self.optimizers()
#             #assert all(isinstance(weight, DTensor) for weight in self.model.parameters())
#             #assert all(isinstance(tensor, DTensor) for tensor in optimizer.param_groups[0]["params"])
#             #assert self.model.w1.weight.device_mesh.ndim == 2
#             #assert self.model.w1.weight.device_mesh.size(0) == 2
#             #assert self.model.w1.weight.device_mesh.size(1) == 2
#             #assert all(weight.device.type != "meta" for weight in self.model.parameters())
#             #assert all(tensor.device_mesh.ndim == 2 for tensor in optimizer.param_groups[0]["params"])
#             #assert all(tensor.device.type != "meta" for tensor in optimizer.param_groups[0]["params"])

#             # No data sharding across TP dimension, sharding across data-parallel dimension only
#             device_mesh = self.device_mesh
#             dp_mesh = device_mesh["data_parallel"]
#             dataloader = self.trainer.train_dataloader
#             #assert len(dataloader) == 8 // dataloader.batch_size // dp_mesh.size()
#             #assert isinstance(dataloader.sampler, DistributedSampler)

#         def training_step(self, batch):
#             batches = self.all_gather(batch)
#             dp_mesh = self.device_mesh["data_parallel"]
#             #tp_mesh = self.device_mesh["tensor_parallel"]

#             # Batches across the TP dimension must be identical
#             #batches_tp = batches[tp_mesh.mesh]
#             #assert all(torch.equal(batches_tp[0], batches_tp[i]) for i in range(1, len(batches_tp)))
#             # Batches across the DP dimension must be different
#             #batches_dp = batches[dp_mesh.mesh]
#             #assert all(not torch.equal(batches_dp[0], batches_dp[i]) for i in range(1, len(batches_dp)))

#             return super().training_step(batch)
