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
import os
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from logging import DEBUG
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from lightning_lite.utilities.cloud_io import get_filesystem
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy, StrategyRegistry
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from torch import nn
from torch.distributed.utils import _replace_by_prefix
from torch.multiprocessing import ProcessRaisedException
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.data import DataLoader, Dataset

from finetuning_scheduler import CallbackResolverMixin, FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping
from tests.helpers import BoringModel
from tests.helpers.boring_model import CustomLRScheduler, unexpected_warns, unmatched_warns
from tests.helpers.runif import RunIf
from tests.test_finetuning_scheduler_callback import (
    EXPECTED_WARNS,
    FinetuningSchedulerBoringModel,
    FitStartOnlyFTS,
    get_fts,
)

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel, MixedPrecision
    from torch.distributed.fsdp.wrap import wrap

additional_fsdp_warns = [
    "The number of training batches",  # minimizing cost of training for these tests
    "is still running",  # TODO: explicitly cleanup subprocess
    'Deallocating Tensor that still',  # TODO: can be triggered by policy tracing, suppress or potentially open PR
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
    reinitlr_sched_dict = deepcopy(mod_sched_dict)
    lambdalr_sched_dict = deepcopy(mod_sched_dict)
    rlrop_sched_dict = deepcopy(mod_sched_dict)
    mod_sched_dict[0]["params"].extend(mod_sched_dict.pop(1)["params"])
    mod_sched_dict[0]["max_transition_epoch"] = 3
    mod_sched_dict[1] = mod_sched_dict.pop(2)
    mod_sched_dict[1]["lr"] = 1e-06
    mod_sched_dict[2] = mod_sched_dict.pop(3)
    mod_sched_dict[2]["params"] = ["layer.0.*"]
    epoch_only_sched = deepcopy(mod_sched_dict)
    epoch_only_sched[1]["max_transition_epoch"] = 2
    epoch_only_sched[2]["max_transition_epoch"] = 2
    reinitlr_sched_dict[1]["new_lr_scheduler"] = {
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.StepLR",
            "init_args": {"step_size": 1, "gamma": 0.7, "verbose": True},
        },
        "pl_lrs_cfg": {"interval": "epoch", "frequency": 1, "name": "Custom_Reinit_LR"},
    }
    reinitlr_sched_dict[2]["lr"] = 3.0e-06
    reinitlr_sched_dict[2]["new_lr_scheduler"] = {
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
            "init_args": {"T_0": 1, "T_mult": 2, "eta_min": 1.0e-07},
        },
        "pl_lrs_cfg": {"interval": "epoch", "frequency": 1, "name": "Custom_Reinit_LR"},
        "init_pg_lrs": [1.0e-06, 2.0e-06],
    }
    lambdalr_sched_dict[1]["new_lr_scheduler"] = {
        "lr_scheduler_init": {
            "class_path": "tests.helpers.boring_model.LinearWarmupLR",
            "init_args": {"num_warmup_steps": 100, "num_training_steps": 1000},
        },
        "pl_lrs_cfg": {"interval": "step", "frequency": 1, "name": "Custom_Reinit_LR"},
    }
    lambdalr_sched_dict[2]["lr"] = 3.0e-06
    lambdalr_sched_dict[2]["new_lr_scheduler"] = {
        "lr_scheduler_init": {
            "class_path": "tests.helpers.boring_model.LinearWarmupLR",
            "init_args": {"num_warmup_steps": 100, "num_training_steps": 1000},
        },
        "pl_lrs_cfg": {"interval": "step", "frequency": 1, "name": "Custom_Reinit_LR"},
        "init_pg_lrs": [1.0e-06, 2.0e-06],
    }
    rlrop_sched_dict[0]["max_transition_epoch"] = 4
    rlrop_sched_dict[1]["max_transition_epoch"] = 8
    rlrop_sched_dict[1]["new_lr_scheduler"] = {
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "init_args": {"patience": 1, "min_lr": [2.0e-07, 1.0e-07]},
        },
        "pl_lrs_cfg": {"interval": "epoch", "frequency": 1, "monitor": "val_loss", "name": "Custom_Reinit_LR"},
        "init_pg_lrs": [1.5e-06],
    }
    rlrop_sched_dict[2]["lr"] = 3.0e-06
    rlrop_sched_dict[2]["max_transition_epoch"] = 10
    rlrop_sched_dict[2]["new_lr_scheduler"] = {
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.StepLR",
            "init_args": {"step_size": 1, "gamma": 0.7, "verbose": True},
        },
        "pl_lrs_cfg": {"interval": "epoch", "frequency": 1, "name": "Custom_Reinit_LR"},
        "init_pg_lrs": [2.0e-06, 3.0e-06],
    }
    fsdp_sched_dict = deepcopy(mod_sched_dict)
    fsdp_sched_dict[0]["params"] = ["layer.(4|2).*"]
    fsdp_single_trans_sched_dict = deepcopy(fsdp_sched_dict)
    fsdp_single_trans_sched_dict[0]["max_transition_epoch"] = 1
    fsdp_single_trans_sched_dict[1]["params"] = ["layer.(1|0).*"]
    del fsdp_single_trans_sched_dict[2]
    fsdp_twotrans_sched_dict = deepcopy(fsdp_sched_dict)
    fsdp_twotrans_sched_dict[0]["max_transition_epoch"] = 1
    fsdp_twotrans_sched_dict[1]["params"] = ["layer.1.bias", "layer.1.weight"]
    fsdp_twotrans_sched_dict[2]["params"] = ["layer.0.*"]
    fsdp_gen_sched_dict = deepcopy(fsdp_sched_dict)
    fsdp_gen_sched_dict[0]["params"] = ["layer.(7|5).*"]
    fsdp_gen_sched_dict[0]["max_transition_epoch"] = 1
    fsdp_gen_sched_dict[1]["params"] = ["layer.[1-4].*"]
    fsdp_gen_sched_dict[1]["max_transition_epoch"] = 2
    fsdp_all_sched_dict = deepcopy(fsdp_sched_dict)
    fsdp_all_sched_dict[0]["params"] = ["layer.*"]
    fsdp_all_sched_dict[0]["max_transition_epoch"] = 1
    for i in range(1, 3):
        del fsdp_all_sched_dict[i]
    return (
        unmod_schedule_file,
        mod_sched_dict,
        epoch_only_sched,
        reinitlr_sched_dict,
        lambdalr_sched_dict,
        rlrop_sched_dict,
        fsdp_sched_dict,
        fsdp_all_sched_dict,
        fsdp_single_trans_sched_dict,
        fsdp_twotrans_sched_dict,
        fsdp_gen_sched_dict,
    )


class FTSBaseFSDPModel(FinetuningSchedulerBoringModel):
    def __init__(self, fsdp_mask: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fsdp_mask = fsdp_mask

        # self.layer: Optional[torch.nn.Module] = None
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
        # assert isinstance(self.layer, torch.nn.Sequential)
        assert isinstance(self.layer, FullyShardedDataParallel)
        # assert isinstance(self.trainer.strategy.precision_plugin, FullyShardedNativeNativeMixedPrecisionPlugin)
        # precision = torch.float16 if self.precision == 16 else torch.bfloat16
        # ensure our ignored module is not wrapped
        for i in self.fsdp_mask["unwrapped_mods"]:
            assert not isinstance(self.layer[i], FullyShardedDataParallel)
        # but that all other modules are wrapped
        for i in self.fsdp_mask["wrapped_mods"]:
            # assert self.layer[layer_num].mixed_precision.param_dtype == precision
            # assert self.layer[layer_num].mixed_precision.reduce_dtype == precision
            # assert self.layer[layer_num].mixed_precision.buffer_dtype == precision
            assert isinstance(self.layer[i], FullyShardedDataParallel)


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


@RunIf(min_torch="1.12", min_cuda_gpus=1)
@pytest.mark.parametrize("precision, expected", [(16, torch.float16), ("bf16", torch.bfloat16)])
def test_precision_plugin_config(precision, expected):
    plugin = FullyShardedNativeNativeMixedPrecisionPlugin(precision=precision, device="cuda")
    config = plugin.mixed_precision_config
    assert config.param_dtype == expected
    assert config.buffer_dtype == expected
    assert config.reduce_dtype == expected


@RunIf(min_torch="1.12")
def test_fsdp_custom_mixed_precision(tmpdir):
    """Test to ensure that passing a custom mixed precision config works."""
    config = MixedPrecision()
    strategy = DDPFullyShardedNativeStrategy(mixed_precision=config)
    assert strategy.mixed_precision_config == config


EXPECTED_FSDP_FTS_RESULTS = {
    ("csm_overridden", False, 10): ({"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}, None),
    ("custom_auto_wrap_policy", False, 10): ({"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}, None),
    ("warn_custom_auto_wrap_policy", False, 10): (
        {"wrapped_mods": [5], "unwrapped_mods": [i for i in list(range(8)) if i != 5]},
        ("Training an FSDP wrapped model requires",),
    ),
}


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize(
    "model, auto_wrap_policy, use_precision, ft_sched_idx, fit_start_only",
    [
        (FTSBaseFSDPModel, custom_auto_wrap_policy, False, 10, False),
        (FTSCsmFSDPModel, None, False, 10, False),
        (FTSBaseFSDPModel, warn_custom_auto_wrap_policy, False, 10, True),
        # TODO: test with provided size based policy
        # TODO: test modified schedule if phase 0 doesn't have at least 1 FSDP param (and warning gen)
        # TODO: add fts option and test to bypass phase-aligned application of auto policy
        # TODO: enable precision tests
        # TODO: test misconfiguration error from validation func generated when auto policy bypass or user csm override
        #  doesn't yield a valid schedule (non disjoint phase mods, no fsdp in the first phase)
        # TODO: test with CPUOffload = False?
        # TODO: test precision with BatchNorm
    ],
    ids=[
        "fts_cust_auto_noprec",
        "fts_no_auto_noprec",
        "fts_warn_auto_noprec",
        # "fts_size_auto_no_prec"
        # "fts_user_auto_only_no_prec",
        # "fts_cust_auto_prec"
    ],
)
def test_fsdp_native_multi_gpus(
    tmpdir, recwarn, fsdp_ft_schedules, model, auto_wrap_policy, use_precision, ft_sched_idx, fit_start_only
):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run."""
    fsdp_warns = FSDP_BASE_WARNS
    auto_wrap_key = getattr(auto_wrap_policy, "__name__", None) or "csm_overridden"
    expected_state = EXPECTED_FSDP_FTS_RESULTS[
        (
            auto_wrap_key,
            use_precision,
            ft_sched_idx,
        )
    ]
    warns_expected = expected_state[1]
    seed_everything(42)
    model = model(fsdp_mask=expected_state[0])
    test_ignored = False
    ignored_modules = [model.layer[1]] if test_ignored else None
    fts_cls = FitStartOnlyFTS if fit_start_only else FinetuningScheduler
    callbacks = [
        fts_cls(ft_schedule=fsdp_ft_schedules[ft_sched_idx], logging_level=DEBUG),  # max_depth=0
        FTSEarlyStopping(monitor="val_loss", patience=1),
        FTSCheckpoint(monitor="val_loss", save_last=True, verbose=True),
    ]

    # precision_config = MixedPrecision(reduce_dtype=torch.float32, param_dtype=torch.float32, buffer_dtype=torch.float32)
    strategy = DDPFullyShardedNativeStrategy(
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
        ignored_modules=ignored_modules,
        # mixed_precision=precision_config if use_precision else None
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

    if fit_start_only:
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


@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True, min_torch="1.12")
def test_invalid_parameters_in_optimizer(tmpdir):
    trainer = Trainer(strategy="fsdp_native", accelerator="cuda", devices=1)

    class EmptyParametersModel(BoringModel):
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-2)

    model = EmptyParametersModel()
    with pytest.raises(ValueError, match="The optimizer does not seem to reference any FSDP parameters"):
        trainer.fit(model)

    class NoFlatParametersModel(BoringModel):
        def configure_optimizers(self):
            layer = torch.nn.Linear(4, 5)
            return torch.optim.Adam(layer.parameters(), lr=1e-2)

    model = NoFlatParametersModel()
    with pytest.raises(ValueError, match="The optimizer does not seem to reference any FSDP parameters"):
        trainer.fit(model)