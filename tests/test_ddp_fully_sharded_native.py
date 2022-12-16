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
from copy import deepcopy
from logging import DEBUG
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from torch.utils.data import DataLoader

from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping
from tests.helpers.boring_model import RandomDataset, unexpected_warns, unmatched_warns
from tests.helpers.runif import RunIf
from tests.test_finetuning_scheduler_callback import (
    EXPECTED_WARNS,
    FinetuningSchedulerBoringModel,
    FitStartOnlyFTS,
    get_fts,
    TestFinetuningScheduler,
)

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import wrap

additional_fsdp_warns = [
    "The number of training batches",  # minimizing cost of training for these tests
    "is still running",  # TODO: explicitly cleanup subprocess
    # "Deallocating Tensor that still",  # TODO: can be triggered by policy tracing, suppress or potentially open PR
    "Please use torch.distributed.all_gather_into_tensor",  # can be removed once PyTorch stops using internally,
    "Please use torch.distributed.reduce_scatter_tensor",  # can be removed once PyTorch stops using internally,
    "when logging on epoch level in distributed",  # validating FTS handling in this scenario
]
EXPECTED_WARNS.extend(additional_fsdp_warns)
FSDP_BASE_WARNS = EXPECTED_WARNS


@pytest.fixture(scope="function")
def fsdp_ft_schedules(tmpdir_factory) -> Tuple[Path, Dict]:
    """Generates a default fine-tuning schedule for 'implicit' testing, a modified one for 'explicit' mode and an
    epoch-driven transitions only one for epoch_transitions_only testing."""
    seed_everything(42)
    callbacks = [FinetuningScheduler(gen_ft_sched_only=True), FTSCheckpoint(monitor="val_loss")]
    model = FinetuningSchedulerBoringModel()
    tmpdir = tmpdir_factory.getbasetemp()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks)
    unmod_schedule_file = tmpdir / "lightning_logs" / "version_0" / f"{model.__class__.__name__}_ft_schedule.yaml"
    with pytest.raises(SystemExit):
        trainer.fit(model)
    mod_sched_dict = get_fts(trainer).load_yaml_schedule(unmod_schedule_file)
    mod_sched_dict[0]["params"].extend(mod_sched_dict.pop(1)["params"])
    mod_sched_dict[0]["max_transition_epoch"] = 3
    mod_sched_dict[1] = mod_sched_dict.pop(2)
    mod_sched_dict[1]["lr"] = 1e-06
    mod_sched_dict[2] = mod_sched_dict.pop(3)
    mod_sched_dict[2]["params"] = ["layer.0.*"]
    fsdp_sched_dict = deepcopy(mod_sched_dict)
    fsdp_sched_dict[0]["params"] = ["layer.(4|2).*"]
    fsdp_gen_sched_dict = deepcopy(fsdp_sched_dict)
    fsdp_gen_sched_dict[0]["params"] = ["layer.(7|5).*"]
    fsdp_gen_sched_dict[0]["max_transition_epoch"] = 1
    fsdp_gen_sched_dict[1]["params"] = ["layer.[1-4].*"]
    fsdp_gen_sched_dict[1]["max_transition_epoch"] = 2
    fsdp_bn_gen_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_bn_gen_sched_dict[0]["params"] = ["layer.(8|[4-6]).*"]
    fsdp_bn_gen_sched_dict[1]["params"] = ["layer.[1-3].*"]
    fsdp_shared_param_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_shared_param_sched_dict[0]["params"] = ["layer.(7|4).*", "layer.5.weight", "layer.5.bias"]
    fsdp_shared_param_sched_dict[1]["params"] = ["layer.2.*", "layer.3.weight", "layer.3.bias"]
    fsdp_shared_param_sched_dict[2]["params"] = ["layer.[0-1].*"]
    fsdp_nondis_mod_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_nondis_mod_sched_dict[1]["params"] = ["layer.[1-4].*", "layer.0.bias"]
    fsdp_nondis_mod_sched_dict[2]["params"] = ["layer.0.weight"]
    return (fsdp_gen_sched_dict, fsdp_nondis_mod_sched_dict, fsdp_bn_gen_sched_dict, fsdp_shared_param_sched_dict)


class FTSBaseFSDPModel(FinetuningSchedulerBoringModel):
    def __init__(
        self, fsdp_mask: Dict, outer_is_wrapped: bool = True, precision_key: Optional[str] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.fsdp_mask = fsdp_mask
        self.outer_is_wrapped = outer_is_wrapped
        self.precision_key = precision_key

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    # reference working version for inspection
    def dev_configure_sharded_model(self) -> None:
        # the model is already wrapped with FSDP: no need to wrap again!
        if isinstance(self.layer, FullyShardedDataParallel):
            return
        module_map = self.trainer.callbacks[0]._gen_ft_sched_module_map()
        with self.trainer.callbacks[0]._enable_explicit_wrap():
            for _, mod_paths in module_map.items():
                # trace_state_dicts[phase] = self.state_dict()
                for modp in mod_paths:
                    parent_name, _, child_name = modp.rpartition(".")
                    if modp != "layer.7":
                        setattr(self.get_submodule(parent_name), child_name, wrap(self.get_submodule(modp)))
            for n, m in self.named_children():
                setattr(self, n, wrap(m))

    def tmp_configure_sharded_model(self) -> None:
        for m in self.modules():
            # if the model is already wrapped with FSDP, tracing with auto-policy would fail
            if isinstance(m, FullyShardedDataParallel):
                raise MisconfigurationException(
                    "The provided model is already wrapped by FSDP. Cannot apply an FSDP auto-wrapping policy along"
                    " fine-tuning schedule phase boundaries if the model is already wrapped."
                )
        # # link phase params to modules, until PT adds fix for param-level specificaiton
        # module_map = self.trainer.callbacks[0]._gen_ft_sched_module_map()  # TODO: move to fts setup?
        # for _, mod_paths in module_map.items():
        #     should_wrap = self.trainer.callbacks[0]._inspect_policy_trace(mod_paths)
        #     self.trainer.callbacks[0]._apply_phase_wrap(should_wrap, mod_paths)
        # with self.trainer.callbacks[0]._enable_explicit_wrap():
        #     for n, m in self.named_children():
        #         setattr(self, n, wrap(m))
        self.trainer.callbacks[0]._phase_constrained_auto_wrap()
        # with self.trainer.callbacks[0]._enable_explicit_wrap():
        #     for n, m in self.named_children():
        #         setattr(self, n, wrap(m))
        assert self

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        return torch.optim.SGD(parameters, lr=0.1)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self._assert_layer_fsdp_instance()

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        self._assert_layer_fsdp_instance()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        self._assert_layer_fsdp_instance()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        self._assert_layer_fsdp_instance()

    def _assert_layer_fsdp_instance(self) -> None:
        if self.outer_is_wrapped:
            assert isinstance(self.layer, FullyShardedDataParallel)
        else:
            assert isinstance(self.layer, torch.nn.Sequential)
        if self.precision_key == "auto_16":
            assert isinstance(self.trainer.strategy.precision_plugin, FullyShardedNativeNativeMixedPrecisionPlugin)
            precision = torch.float16 if self.precision == 16 else torch.bfloat16
        # ensure our ignored module is not wrapped
        for i in self.fsdp_mask["unwrapped_mods"]:
            assert not isinstance(self.layer[i], FullyShardedDataParallel)
        # but that all other modules are wrapped
        for i in self.fsdp_mask["wrapped_mods"]:
            assert isinstance(self.layer[i], FullyShardedDataParallel)
            if self.precision_key:
                if isinstance(self.layer[i].module, torch.nn.modules.batchnorm._BatchNorm):
                    assert self.layer[i].mixed_precision.param_dtype is None
                    assert self.layer[i].mixed_precision.reduce_dtype is None
                    assert self.layer[i].mixed_precision.buffer_dtype is None
                else:
                    assert self.layer[i].mixed_precision.param_dtype == precision
                    assert self.layer[i].mixed_precision.reduce_dtype == precision
                    assert self.layer[i].mixed_precision.buffer_dtype == precision


class FTSCsmFSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_sharded_model(self) -> None:
        # the model is already wrapped with FSDP: no need to wrap again!
        if isinstance(self.layer, FullyShardedDataParallel):
            return
        for i, layer in enumerate(self.layer):
            if i in self.fsdp_mask["wrapped_mods"]:
                self.layer[i] = wrap(layer)
        self.layer = wrap(self.layer)


class FTSBatchNormFSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def train_dataloader(self):
        # when testing BatchNorm layers, we need to ensure there are more than 1 samples per batch
        return DataLoader(RandomDataset(32, 64), batch_size=2)


class FTSSharedParamFSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )
        self.layer[4].weight = self.layer[5].weight
        self.layer[4].bias = self.layer[5].bias
        self.layer[3].weight = self.layer[1].weight
        self.layer[3].bias = self.layer[1].bias


class FSDPTestFinetuningScheduler(TestFinetuningScheduler):
    def state_dict(self) -> Dict[str, Any]:
        return super(TestFinetuningScheduler, self).state_dict()

    def restore_best_ckpt(self) -> None:
        super(TestFinetuningScheduler, self).restore_best_ckpt()

    def on_train_epoch_start(self, trainer, pl_module):
        super(TestFinetuningScheduler, self).on_train_epoch_start(trainer, pl_module)
        state_key = trainer.current_epoch
        current_state = (
            len(self._fts_state._curr_thawed_params),
            len(self.strategy_adapter.logical_param_translation(self._fts_state._curr_thawed_params)),
        )
        if self.expected_state:
            assert current_state == self.expected_state[state_key]


def custom_auto_wrap_policy(
    module,
    recurse,
    unwrapped_params: int,
    min_num_params: int = int(1e8),
) -> bool:
    return unwrapped_params >= 67


def warn_custom_auto_wrap_policy(
    module,
    recurse,
    unwrapped_params: int,
    min_num_params: int = int(1e8),
) -> bool:
    return unwrapped_params >= 1100


EXPECTED_FSDP_FTS_RESULTS = {
    "default_awp_noprec": (
        {
            0: (2, 4),
            1: (6, 12),
            2: (7, 14),
        },
        "Because `fts_flex_awp`",
        None,
    ),
    "override_csm_noprec": (
        {
            0: (2, 4),
            1: (6, 12),
            2: (7, 14),
        },
        None,
        None,
    ),
    "non_disjoint_phase_fsdp_params": ({}, None, "do not have disjoint FSDP-flattened parameter"),
    "non_disjoint_phase_mods": ({}, None, "not have disjoint"),
    "no_fsdp_params_p0": ({}, None, "one or more FSDP"),
    "cust_awp_prec": (
        {
            0: (2, 4),
            1: (6, 12),
            2: (7, 14),
        },
        None,
        None,
    ),
    "batch_norm_auto_prec": (
        {
            0: (4, 8),
            1: (7, 14),
            2: (8, 16),
        },
        "Both mixed precision",
        None,
    ),
    "shared_params_auto_prec": (
        {
            0: (2, 4),
            1: (3, 6),
            2: (5, 10),
        },
        "Pruning explicitly specified",
        None,
    ),
}


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize(
    "model_cfg_key, model, auto_wrap_policy, use_precision, ft_sched_idx, fit_start_only, strategy_adapter_cfg,\
         test_model_cfg, test_fts_cfg",
    [
        (
            "default_awp_noprec",
            FTSBaseFSDPModel,
            custom_auto_wrap_policy,
            False,
            0,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": list(range(6)) + [7], "unwrapped_mods": []}},
            None,
        ),
        (
            "override_csm_noprec",
            FTSCsmFSDPModel,
            None,
            False,
            0,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}},
            None,
        ),
        (
            "non_disjoint_phase_fsdp_params",
            FTSBaseFSDPModel,
            warn_custom_auto_wrap_policy,
            False,
            0,
            False,
            {"fts_flex_awp": False},
            {"fsdp_mask": {"wrapped_mods": [5], "unwrapped_mods": [i for i in list(range(8)) if i != 5]}},
            None,
        ),
        (
            "non_disjoint_phase_mods",
            FTSCsmFSDPModel,
            None,
            False,
            1,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}},
            None,
        ),
        (
            "no_fsdp_params_p0",
            FTSCsmFSDPModel,
            None,
            False,
            0,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": list(range(5)), "unwrapped_mods": [5, 7]}},
            None,
        ),
        (
            "cust_awp_prec",
            FTSBaseFSDPModel,
            custom_auto_wrap_policy,
            True,
            0,
            False,
            {"fts_flex_awp": False},
            {
                "fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]},
                "precision_key": "auto_16",
            },
            None,
        ),
        (
            "batch_norm_auto_prec",
            FTSBatchNormFSDPModel,
            custom_auto_wrap_policy,
            True,
            2,
            False,
            {"fts_flex_awp": False},
            {"fsdp_mask": {"wrapped_mods": list(range(7)), "unwrapped_mods": [8]}, "precision_key": "auto_16"},
            None,
        ),
        (
            "shared_params_auto_prec",
            FTSSharedParamFSDPModel,
            custom_auto_wrap_policy,
            True,
            3,
            False,
            {"fts_flex_awp": False},
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}, "precision_key": "auto_16"},
            None,
        ),
    ],
    ids=[
        "default_awp_noprec",
        "override_csm_noprec",
        "non_disjoint_phase_fsdp_params",
        "non_disjoint_phase_mods",
        "no_fsdp_params_p0",
        "cust_awp_prec",
        "batch_norm_auto_prec",
        "shared_params_auto_prec",
    ],
)
def test_fsdp_native_multi_gpus(
    tmpdir,
    recwarn,
    fsdp_ft_schedules,
    model_cfg_key,
    model,
    auto_wrap_policy,
    use_precision,
    ft_sched_idx,
    fit_start_only,
    strategy_adapter_cfg,
    test_model_cfg,
    test_fts_cfg,
):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run."""
    fsdp_warns = FSDP_BASE_WARNS
    model_cfg = test_model_cfg or {}
    additional_fts_cfg = test_fts_cfg or {}
    expected_state = EXPECTED_FSDP_FTS_RESULTS[model_cfg_key]
    warns_expected = expected_state[1]
    exception_expected = expected_state[2]
    seed_everything(42)
    model = model(**model_cfg)
    fts_cls = FitStartOnlyFTS if fit_start_only else FSDPTestFinetuningScheduler
    callbacks = [
        fts_cls(
            ft_schedule=fsdp_ft_schedules[ft_sched_idx],
            logging_level=DEBUG,
            strategy_adapter_cfg=strategy_adapter_cfg,
            expected_state=expected_state[0],
            **additional_fts_cfg,
        ),
        FTSEarlyStopping(monitor="val_loss", patience=1),
        FTSCheckpoint(monitor="val_loss", save_last=True, verbose=True),
    ]
    strategy = DDPFullyShardedNativeStrategy(
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
    )
    precision_opts = {"precision": 16} if use_precision else {}
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs=3,
        **precision_opts,
    )

    if exception_expected:
        with pytest.raises(MisconfigurationException, match=exception_expected):
            trainer.fit(model)
    elif fit_start_only:
        with pytest.raises(SystemExit):
            trainer.fit(model)
    else:
        trainer.fit(model)
        finetuningscheduler_callback = get_fts(trainer)
        assert finetuningscheduler_callback.depth_remaining == 0
        assert finetuningscheduler_callback.curr_depth == 2
        assert finetuningscheduler_callback.curr_depth == finetuningscheduler_callback.max_depth

    if trainer.is_global_zero:
        if warns_expected:
            unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=warns_expected)
            assert not unmatched
            fsdp_warns.extend(warns_expected)
        # ensure no unexpected warnings detected
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=fsdp_warns)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
