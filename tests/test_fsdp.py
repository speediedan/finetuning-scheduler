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
from functools import partial
from logging import DEBUG
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from unittest import mock

import pytest
import torch
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_1_13,
    _TORCH_GREATER_EQUAL_2_0,
    _TORCH_GREATER_EQUAL_2_1,
)
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.plugins.precision.fsdp import FSDPPrecisionPlugin
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping
from finetuning_scheduler.strategy_adapters import FSDPStrategyAdapter
from tests.helpers.boring_model import RandomDataset, unexpected_warns, unmatched_warns
from tests.helpers.runif import RunIf
from tests.test_finetuning_scheduler_callback import (
    EXPECTED_WARNS,
    ExplicitLossFTSCheckpoint,
    FinetuningSchedulerBoringModel,
    get_fts,
    nones,
    TestFinetuningScheduler,
    get_sched_fixture_tmpdir,
)

_distributed_available = torch.distributed.is_available()
_min_fsdp_available = _TORCH_GREATER_EQUAL_1_13 and _distributed_available

if _min_fsdp_available:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel, MixedPrecision
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, wrap
else:
    FullyShardedDataParallel = None  # type: ignore[misc,assignment]
    MixedPrecision = None  # type: ignore[misc,assignment]
    BackwardPrefetch = None  # type: ignore[misc,assignment]
    CPUOffload = None  # type: ignore[misc,assignment]
    size_based_auto_wrap_policy = object
    wrap = object

if _TORCH_GREATER_EQUAL_2_1:
    from torch.distributed.fsdp.wrap import CustomPolicy

    DISABLE_USE_ORIG = {"use_orig_params": False}
    _FSDPPolicy = object
elif _TORCH_GREATER_EQUAL_2_0:
    from torch.distributed.fsdp.wrap import _FSDPPolicy

    DISABLE_USE_ORIG = {"use_orig_params": False}
else:
    CustomPolicy = object
    _FSDPPolicy = object
    DISABLE_USE_ORIG = {}


additional_fsdp_warns = [
    "The number of training batches",  # minimizing cost of training for these tests
    "does not support loading the optimizer",  # with PyTorch 1.x Lightning lacks OSD restoration support
    "Please use torch.distributed.all_gather_into_tensor",  # can be removed once PyTorch stops using internally,
    "Please use torch.distributed.reduce_scatter_tensor",  # can be removed once PyTorch stops using internally,
    "when logging on epoch level in distributed",  # validating FTS handling in this scenario
    "Deallocating Tensor that still has live",  # still encounter this in some edge cases
]
EXPECTED_WARNS.extend(additional_fsdp_warns)
FSDP_BASE_WARNS = EXPECTED_WARNS
FSDP_DYNAMO_EXPECTED_WARNS = [
    "Final phase max_transition_epoch",
]

##########################
# FTS FSDP Test Fixtures #
##########################


@pytest.fixture(scope="module")
def fsdp_ft_schedules(tmpdir_factory) -> Tuple[Path, Dict]:
    """Generates a default fine-tuning schedule for 'implicit' testing, a modified one for 'explicit' mode and an
    epoch-driven transitions only one for epoch_transitions_only testing."""
    seed_everything(42)
    callbacks = [FinetuningScheduler(gen_ft_sched_only=True), FTSCheckpoint(monitor="val_loss")]
    model = FinetuningSchedulerBoringModel()
    tmpdir, rank = get_sched_fixture_tmpdir(tmpdir_factory)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks, devices=1)
    unmod_schedule_file = tmpdir / "lightning_logs" / "version_0" / f"{model.__class__.__name__}_ft_schedule.yaml"
    if rank == 0:
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
    fsdp_reinit_optim_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_reinitlr_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_reinit_optim_sched_dict[1]["new_optimizer"] = {
        "optimizer_init": {
            "class_path": "torch.optim.Adam",
            "init_args": {"lr": 2.1e-04},
        },
    }
    fsdp_reinit_optim_sched_dict[2]["new_optimizer"] = {
        "optimizer_init": {
            "class_path": "torch.optim.SGD",
            "init_args": {"lr": 2.0e-03, "momentum": 0.9, "weight_decay": 2.0e-06},
        }
    }
    fsdp_reinitlr_sched_dict[1]["new_lr_scheduler"] = {
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.StepLR",
            "init_args": {"step_size": 1, "gamma": 0.7, "verbose": True},
        },
        "pl_lrs_cfg": {"interval": "epoch", "frequency": 1, "name": "Custom_Reinit_LR"},
    }
    fsdp_reinitlr_sched_dict[2]["lr"] = 3.0e-06
    fsdp_reinitlr_sched_dict[2]["new_lr_scheduler"] = {
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
            "init_args": {"T_0": 1, "T_mult": 2, "eta_min": 1.0e-07},
        },
        "pl_lrs_cfg": {"interval": "epoch", "frequency": 1, "name": "Custom_Reinit_LR"},
        "init_pg_lrs": [1.0e-06, 2.0e-06],
    }
    fsdp_reinitlr_optim_sched_dict = deepcopy(fsdp_reinitlr_sched_dict)
    fsdp_reinitlr_optim_sched_dict[1]["new_optimizer"] = deepcopy(fsdp_reinit_optim_sched_dict[1]["new_optimizer"])
    fsdp_reinitlr_optim_sched_dict[2]["new_optimizer"] = deepcopy(fsdp_reinit_optim_sched_dict[2]["new_optimizer"])
    fsdp_reinitlr_optim_sched_dict[2]["new_lr_scheduler"] = {
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.StepLR",
            "init_args": {"step_size": 1, "gamma": 0.2, "verbose": True},
        },
        "pl_lrs_cfg": {"interval": "epoch", "frequency": 1, "name": "Custom_Reinit_LR"},
    }
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
    fsdp_nondis_mod_ex_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_nondis_mod_ex_sched_dict[1]["params"] = ["layer.[2-4].*"]
    del fsdp_nondis_mod_ex_sched_dict[1]["max_transition_epoch"]
    del fsdp_nondis_mod_ex_sched_dict[2]
    fsdp_adam_gen_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_adam_gen_sched_dict[0]["max_transition_epoch"] = 2
    fsdp_adam_gen_sched_dict[1]["max_transition_epoch"] = 4
    fsdp_ext_gen_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_ext_gen_sched_dict[0]["params"] = ["layer.(5|[7-8]).*"]
    fsdp_epoch_only_sched = deepcopy(fsdp_gen_sched_dict)
    fsdp_epoch_only_sched[1]["max_transition_epoch"] = 2
    fsdp_epoch_only_sched[2]["max_transition_epoch"] = 3
    return (
        fsdp_gen_sched_dict,
        fsdp_nondis_mod_sched_dict,
        fsdp_bn_gen_sched_dict,
        fsdp_shared_param_sched_dict,
        fsdp_adam_gen_sched_dict,
        fsdp_nondis_mod_ex_sched_dict,
        fsdp_ext_gen_sched_dict,
        fsdp_epoch_only_sched,
        fsdp_reinitlr_optim_sched_dict,
    )


@pytest.fixture(scope="module", params=[True, False], ids=["use_orig", "no_use_orig"])
def fsdp_ckpt(tmpdir_factory, fsdp_ft_schedules, request) -> Tuple[Dict, bool]:
    """A fixture that generates a checkpoint with a sharded model."""
    seed_everything(42)
    test_model_cfg = {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}}
    use_orig_kwargs = DISABLE_USE_ORIG if not request.param else {}
    strategy = FSDPStrategy(
        auto_wrap_policy=custom_auto_wrap_policy,
        **use_orig_kwargs,
        cpu_offload=CPUOffload(offload_params=True),
    )
    callbacks = [
        FinetuningScheduler(ft_schedule=fsdp_ft_schedules[0]),
        FTSEarlyStopping(monitor="val_loss", patience=1),
        FTSCheckpoint(monitor="val_loss", save_last=True, verbose=True),
    ]
    model = FTSAdamFSDPModel(**test_model_cfg)
    trainer = Trainer(
        default_root_dir=tmpdir_factory.getbasetemp(),
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs=1,
    )
    trainer.fit(model)
    return trainer.checkpoint_callback.best_model_path, request.param


########################
# FTS FSDP Test Models #
########################


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

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [lr_scheduler]

    def val_loss(self, batch, prediction):
        # Make arbitrary val_loss the inverse of train_loss so val_loss diverges when desired
        val_func = (
            torch.full_like(prediction, 100)
            if self.current_epoch >= self.diverge_on_epoch
            else torch.zeros_like(prediction)
        )
        return torch.nn.functional.mse_loss(prediction, val_func)

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.val_loss(batch, output)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)
        return {"x": loss}

    def on_train_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def on_test_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def on_validation_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def on_predict_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def _assert_layer_fsdp_instance(self) -> None:
        if self.outer_is_wrapped:
            assert isinstance(self.layer, FullyShardedDataParallel)
        else:
            assert isinstance(self.layer, torch.nn.Sequential)
        if self.precision_key == "auto_16":
            assert isinstance(self.trainer.strategy.precision_plugin, FSDPPrecisionPlugin)
            precision = torch.float16 if self.trainer.precision == "16-true" else torch.bfloat16
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


class NonDynamicLossAdamFSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def val_loss(self, batch, prediction):
        # Make val_loss non-dynamic for dynamo testing
        val_func = torch.ones_like(prediction)
        return torch.nn.functional.mse_loss(prediction, val_func)

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.val_loss(batch, output)
        self.validation_step_outputs.append(loss)
        # we would normally use sync_dist for epoch-only logging in a distributed context but leaving it `False` here
        # to test FTS transition behavior when the test model is used in a distributed context
        # temporarily disable logging loss until resolution upstream compile issue
        # self.log("val_loss", loss, prog_bar=False)
        return {"x": loss}

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(parameters, weight_decay=1.0e-05, eps=1.0e-07, lr=1.0e-05)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [lr_scheduler]


class FTSCmFSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_model(self) -> None:
        # the model is already wrapped with FSDP: no need to wrap again!
        if isinstance(self.layer, FullyShardedDataParallel):
            return
        for i, layer in enumerate(self.layer):
            if i in self.fsdp_mask["wrapped_mods"]:
                self.layer[i] = wrap(layer)
        self.layer = wrap(self.layer)

        # verify activation checkpointing can be manually applied
        check_fn = lambda submodule: isinstance(submodule, tuple([torch.nn.Linear]))  # noqa E731
        wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(self.layer, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)


class FTSNoDecayFSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_decay = ["bias"]


class AlreadyWrappedFSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage: str):
        with self._trainer.strategy.model_sharded_context():
            self.layer[0] = wrap(self.layer[0])
            assert isinstance(self.layer[0], FullyShardedDataParallel)


class FTSEnforceP0FSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [lr_scheduler]


class FTSCmAdamFSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(parameters, weight_decay=1.0e-05, eps=1.0e-07, lr=1.0e-05)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [lr_scheduler]

    def configure_model(self) -> None:
        # the model is already wrapped with FSDP: no need to wrap again!
        if isinstance(self.layer, FullyShardedDataParallel):
            return
        for i, layer in enumerate(self.layer):
            if i in self.fsdp_mask["wrapped_mods"]:
                self.layer[i] = wrap(layer)
        self.layer = wrap(self.layer)


class FTSAdamFSDPModel(FTSBaseFSDPModel):
    # use of this non-SGD optimizer config is required for tests that restore optimizer state with PyTorch 2.0.x
    # due to https://github.com/pytorch/pytorch/issues/99079
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(parameters, weight_decay=1.0e-05, eps=1.0e-07, lr=1.0e-05)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [lr_scheduler]


class FTSExtFSDPModel(FTSBaseFSDPModel):
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
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 2),
        )


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
        self.restored_best_cnt += 1

    def on_train_epoch_start(self, trainer, pl_module):
        super(TestFinetuningScheduler, self).on_train_epoch_start(trainer, pl_module)
        state_key = trainer.current_epoch
        current_state = (
            len(self._fts_state._curr_thawed_params),
            len(self.strategy_adapter.logical_param_translation(self._fts_state._curr_thawed_params)),
        )
        lrs_state = tuple(round(pg["lr"], 9) for pg in trainer.optimizers[0].param_groups)
        self.inspect_or_assert(current_state, lrs_state, state_key)


class FSDPOptInspectFTS(FSDPTestFinetuningScheduler):
    def on_train_epoch_start(self, trainer, pl_module):
        super(TestFinetuningScheduler, self).on_train_epoch_start(trainer, pl_module)
        state_key = trainer.current_epoch
        current_state = (
            len(self._fts_state._curr_thawed_params),
            len(self.strategy_adapter.logical_param_translation(self._fts_state._curr_thawed_params)),
            trainer.optimizers[0].__class__.__name__,
            trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer_steps,
            trainer.optimizers[0].defaults["lr"],
        )
        lrs_state = tuple(round(pg["lr"], 9) for pg in trainer.optimizers[0].param_groups)
        self.inspect_or_assert(current_state, lrs_state, state_key)


# model aliases
base_model = FTSBaseFSDPModel
nond_loss_adam_model = NonDynamicLossAdamFSDPModel
cust_model = FTSCmFSDPModel
wrapped_model = AlreadyWrappedFSDPModel
nodecay_model = FTSNoDecayFSDPModel
BN_model = FTSBatchNormFSDPModel
shared_model = FTSSharedParamFSDPModel
cm_adam_model = FTSCmAdamFSDPModel
adam_model = FTSAdamFSDPModel
ext_model = FTSExtFSDPModel
enforceP0_model = FTSEnforceP0FSDPModel

# model configuration aliases
fp16_cfg = {"precision_key": "auto_16"}
unwrap_7 = {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}}
unwrap_7_dyn = {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}, "use_dynamo": True}
unwrap_4_7 = {"fsdp_mask": {"wrapped_mods": [0, 1, 2, 3, 5], "unwrapped_mods": [4, 7]}}
unwrap_5_7 = {"fsdp_mask": {"wrapped_mods": list(range(5)), "unwrapped_mods": [5, 7]}}
unwrap_0_1_7 = {"fsdp_mask": {"wrapped_mods": [2, 3, 4, 5], "unwrapped_mods": [0, 1, 7]}}
wrap_5_7 = {"fsdp_mask": {"wrapped_mods": [5, 7], "unwrapped_mods": [i for i in list(range(8)) if i not in [5, 7]]}}
wrap_5 = {"fsdp_mask": {"wrapped_mods": [5], "unwrapped_mods": [i for i in list(range(8)) if i != 5]}}
outer_wrap_only = {"fsdp_mask": {"wrapped_mods": [], "unwrapped_mods": list(range(8))}}
unwrap_7_diverge = {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}, "diverge_on_epoch": 1}
unwrap_7_mp = {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}, **fp16_cfg}
unwrap_8_mp = {"fsdp_mask": {"wrapped_mods": list(range(7)), "unwrapped_mods": [8]}, **fp16_cfg}
wrap_all_mp = {"fsdp_mask": {"wrapped_mods": list(range(6)) + [7], "unwrapped_mods": []}, **fp16_cfg}
wrap_ext_mp = {"fsdp_mask": {"wrapped_mods": list(range(6)) + [7, 8], "unwrapped_mods": []}, **fp16_cfg}

##########################
# FTS FSDP Test Policies #
##########################


def custom_auto_wrap_policy(
    module,
    recurse,
    nonwrapped_numel: int,
) -> bool:
    return nonwrapped_numel >= 67


def custom_auto_wrap_ext_policy(
    module,
    recurse,
    nonwrapped_numel: int,
) -> bool:
    return nonwrapped_numel >= 529


def warn_custom_auto_wrap_policy(
    module,
    recurse,
    nonwrapped_numel: int,
) -> bool:
    return nonwrapped_numel >= 1100


class CustomWrapPolicy(_FSDPPolicy):
    """This is a wrapper around :func:`_module_wrap_policy`."""

    def __init__(self, min_num_params: int):
        self._policy: Callable = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)

    @property
    def policy(self):
        return self._policy


# RunIf aliases
runif_map = {
    "min2_0": {"min_torch": "2.0.0"},
    "only2_0": {"min_torch": "2.0.0", "max_torch": "2.0.1"},
    "min2_1": {"min_torch": "2.1.0"},
    "max1_13": {"max_torch": "1.13.1"},
    "min2_0_max3_11": {"min_torch": "2.0.0", "max_python": "3.11"},
}

# auto-wrap policy aliases
cust_awp = custom_auto_wrap_policy
cust_ext_awp = custom_auto_wrap_ext_policy
warn_cust_awp = warn_custom_auto_wrap_policy
if _TORCH_GREATER_EQUAL_2_1:
    numel_constant = 67
    awp_mwp_2_1_parity = CustomPolicy(lambda_fn=lambda m: sum(p.numel() for p in m.parameters()) >= numel_constant)
    awp_mwp_2_0_parity = None
elif _TORCH_GREATER_EQUAL_2_0:
    awp_mwp_2_0_parity = CustomWrapPolicy(min_num_params=67)
    awp_mwp_2_1_parity = None
else:
    awp_mwp_2_0_parity, awp_mwp_2_1_parity = None, None

# awp_overrides configuration aliases
awp_5_9 = {"awp_overrides": ["layer.9", "layer.5"]}
awp_1 = {"awp_overrides": ["l.*yer.1"]}
awp_7 = {"awp_overrides": ["layer.7"]}
awp_7_8 = {"awp_overrides": ["l.*yer.8", "layer.7"]}

# FSDP strategy configuration aliases
act_ckpt_cfg = {"activation_checkpointing_policy": {torch.nn.Linear}, **DISABLE_USE_ORIG}
ignore_mod_cfg = {"test_ignored_modules_names": ["layer.4"], "cpu_offload": False, **DISABLE_USE_ORIG}
ignore_states_cfg = {
    "test_ignored_parameters_names": ["layer.4.weight", "layer.4.bias"],
    "cpu_offload": False,
    "use_orig_params": True,
}

if not _TORCH_GREATER_EQUAL_2_0:
    cust_mp_args = {"param_dtype": torch.float16, "reduce_dtype": torch.float16, "buffer_dtype": torch.float16}
cust_mp_args = {"param_dtype": torch.float16, "reduce_dtype": torch.float16, "buffer_dtype": torch.float16}
cust_fp16_mp = {"mixed_precision": MixedPrecision(**cust_mp_args), **DISABLE_USE_ORIG} if _min_fsdp_available else {}

# trainer configuration alias
max_epoch_5 = {"max_epochs": 5}
max_epoch_4 = {"max_epochs": 4}

# fts config aliases
epoch_t_only = {
    "epoch_transitions_only": True,
    "test_es": "disable",
    "test_ckpt": ExplicitLossFTSCheckpoint(monitor="val_loss", verbose=True),
}
opt_inspect = {"test_fts": FSDPOptInspectFTS}

# expected training path aliases
path_default = {0: (2, 4), 1: (6, 12), 2: (7, 14)}
path_default_orig = {0: (4, 4), 1: (12, 12), 2: (14, 14)}
path_default_orig_eo_dyn = {0: (4, 4), 1: (12, 12), 2: (14, 14), 3: (14, 14)}
path_ignore_p_uo = {0: (4, 4), 1: (12, 12), 2: (14, 14)}
path_8_14 = {0: (2, 4), 1: (7, 12), 2: (8, 14)}
path_8_16 = {0: (4, 8), 1: (7, 14), 2: (8, 16)}
path_5_10 = {0: (2, 4), 1: (3, 6), 2: (5, 10)}
path_ext_7_14 = {0: (2, 4), 1: (2, 4), 2: (6, 12), 3: (6, 12), 4: (7, 14)}
path_ext_8_16 = {0: (3, 6), 1: (7, 14), 2: (8, 16)}
path_optimlr_reinit = {0: (2, 4, "SGD", 0, 0.1), 1: (6, 12, "Adam", 32, 0.00021), 2: (7, 14, "SGD", 64, 0.002)}
lrs_path_default = {0: (0.1,), 1: (0.07, 1e-06), 2: (0.049, 7e-07, 1e-05)}
lrs_path_optimlr_reinit = {0: (0.1,), 1: (0.00021, 1e-06), 2: (0.002, 1e-06, 3e-06)}

# consolidate all core FTS FSDP test configuration into this dictionary to dedup config
FTS_FSDP_TESTS = {
    "cust_awp_noprec_no_use_orig": (
        (adam_model, cust_awp, False, 0, unwrap_7, *nones(3), act_ckpt_cfg),
        None,
        (path_default, *nones(3)),
    ),
    "cust_awp_noprec": (
        (adam_model, cust_awp, False, 0, unwrap_7, *nones(4)),
        "min2_0",
        (path_default_orig, *nones(3)),
    ),
    "noawp_noprec": (
        (base_model, None, False, 0, outer_wrap_only, *nones(4)),
        "min2_1",
        (path_default_orig, "degenerate and unintended", *nones(2)),
    ),
    "noawp_prec": (
        (base_model, None, True, 0, outer_wrap_only, *nones(4)),
        "min2_1",
        (path_default_orig, "degenerate and unintended", *nones(2)),
    ),
    "cust_awp_noprec_dynamo": (
        (nond_loss_adam_model, cust_awp, False, 7, unwrap_7_dyn, None, epoch_t_only, max_epoch_4, None),
        "min2_0_max3_11",
        (path_default_orig_eo_dyn, *nones(3)),
    ),
    "cust_awp_mwp_2_1_reinitlr_optim_no_use_orig": (
        (base_model, awp_mwp_2_1_parity, True, 8, unwrap_7_mp, None, opt_inspect, None, DISABLE_USE_ORIG),
        "min2_1",
        (
            path_optimlr_reinit,
            ("Incompatible check",),
            None,
            lrs_path_optimlr_reinit,
        ),
    ),
    "cust_awp_mwp_2_0_parity_no_use_orig": (
        (adam_model, awp_mwp_2_0_parity, True, 0, unwrap_7_mp, *nones(3), DISABLE_USE_ORIG),
        "only2_0",
        (path_default, *nones(3)),
    ),
    "cust_awp_mwp_2_1_parity_no_use_orig": (
        (base_model, awp_mwp_2_1_parity, True, 0, unwrap_7_mp, *nones(3), DISABLE_USE_ORIG),
        "min2_1",
        (path_default, *nones(3)),
    ),
    "override_cm_noprec_no_use_orig": (
        (cust_model, None, False, 0, unwrap_7, *nones(3), DISABLE_USE_ORIG),
        "min2_1",
        (path_default, *nones(3)),
    ),
    "cust_awp_nop_ignore_m_no_ofld_no_use_orig": (
        (base_model, cust_awp, False, 0, unwrap_4_7, *nones(3), ignore_mod_cfg),
        "min2_1",
        (path_8_14, *nones(3)),
    ),  # TODO: once PyTorch deprecates ``ignored_modules``, check for the warning with this test
    "cust_awp_nop_ignore_p_no_ofld": (
        (base_model, cust_awp, False, 0, unwrap_4_7, *nones(3), ignore_states_cfg),
        "min2_1",
        (path_ignore_p_uo, *nones(3)),
    ),
    "unsupp_torch_version": (
        (base_model, cust_awp, False, 0, unwrap_7, *nones(4)),
        None,
        ({}, None, "is supported from PyTorch", None),
    ),
    "non_disjoint_params_allowed": (
        (base_model, warn_cust_awp, False, 0, outer_wrap_only, *nones(4)),
        "min2_1",
        (path_default_orig, *nones(3)),
    ),
    "non_disjoint_phase_fsdp_params_no_use_orig": (
        (base_model, warn_cust_awp, False, 0, wrap_5, *nones(3), DISABLE_USE_ORIG),
        None,
        ({}, None, "do not have disjoint FSDP-flattened parameter", None),
    ),
    "non_disjoint_phase_mods_no_use_orig": (
        (cust_model, None, False, 1, unwrap_7, *nones(3), DISABLE_USE_ORIG),
        None,
        ({}, None, "not have disjoint", None),
    ),
    "non_disjoint_excluded_ft_params_no_use_orig": (
        (cust_model, None, False, 5, unwrap_0_1_7, *nones(3), DISABLE_USE_ORIG),
        None,
        ({}, None, "parameters not included in", None),
    ),
    "already_fsdp_wrapped": (
        (wrapped_model, cust_awp, False, 0, unwrap_7, *nones(4)),
        "min2_1",
        ({}, None, "already wrapped by FSDP", None),
    ),
    "no_fsdp_params_p0": (
        (cust_model, None, False, 0, unwrap_5_7, *nones(3), DISABLE_USE_ORIG),
        None,
        ({}, None, "one or more FSDP", None),
    ),
    "no_nonzero_local_shards_p0": (
        (base_model, warn_cust_awp, True, 0, outer_wrap_only, *nones(4)),
        "min2_1",
        (path_default_orig, *nones(3)),
    ),  # exercise shard allocation DEBUG diagnostics
    "warn_unsupp_nodecay": (
        (nodecay_model, cust_awp, False, 0, unwrap_7, *nones(4)),
        "min2_1",
        ({}, "will now be unset", *nones(2)),
    ),
    "unmatched_awp_overrides": (
        (base_model, warn_cust_awp, True, 0, wrap_5_7, awp_5_9, *nones(3)),
        None,
        ({}, None, "did not match any named modules", None),
    ),
    "cust_awp_prec_no_use_orig": (
        (base_model, cust_awp, True, 0, unwrap_7_mp, *nones(3), DISABLE_USE_ORIG),
        "min2_1",
        (path_default, *nones(3)),
    ),
    "cust_awp_prec_pt1x": (
        (base_model, cust_awp, True, 0, unwrap_7_mp, *nones(3), cust_fp16_mp),
        "max1_13",
        (path_default, *nones(3)),
    ),
    "enforceP0_cust_awp_prec_no_use_orig": (
        (enforceP0_model, cust_awp, True, 0, unwrap_7_mp, *nones(3), DISABLE_USE_ORIG),
        "min2_1",
        (path_default, *nones(3)),
    ),
    "batch_norm_auto_prec_no_use_orig": (
        (BN_model, cust_awp, True, 2, unwrap_8_mp, *nones(3), DISABLE_USE_ORIG),
        "min2_1",
        (path_8_16, ("Both mixed precision",), *nones(2)),
    ),
    "shared_params_auto_prec_no_use_orig": (
        (shared_model, cust_awp, True, 3, unwrap_7_mp, awp_1, *nones(2), DISABLE_USE_ORIG),
        "min2_1",
        (
            path_5_10,
            ("Pruning explicitly specified",),
            *nones(2),
        ),
    ),
    "override_cm_adam_noprec_no_use_orig": (
        (cm_adam_model, None, False, 4, unwrap_7_diverge, *nones(2), max_epoch_5, DISABLE_USE_ORIG),
        None,
        (path_ext_7_14, *nones(3)),
    ),
    "cust_awp_overrides_prec_no_use_orig": (
        (adam_model, cust_awp, True, 0, wrap_all_mp, awp_7, *nones(2), cust_fp16_mp),
        "min2_0",
        (path_default, *nones(3)),
    ),
    "cust_awp_overrides_mwp_prec_no_use_orig": (
        (base_model, awp_mwp_2_1_parity, True, 0, wrap_all_mp, awp_7, *nones(2), cust_fp16_mp),
        "min2_1",
        (path_default, *nones(3)),
    ),
    "cust_awp_overrides_mwp_prec_ext_no_use_orig": (
        (ext_model, cust_ext_awp, True, 6, wrap_ext_mp, awp_7_8, *nones(2), cust_fp16_mp),
        "min2_1",
        (path_ext_8_16, *nones(3)),
    ),
    "warn_ignore_awp_override": (
        (cust_model, None, False, 0, unwrap_7, awp_7, *nones(3)),
        "min2_1",
        ({}, "will be unset and not applied", *nones(2)),
    ),
}


EXPECTED_FSDP_FTS_RESULTS = {k: test_cfg[2] for k, test_cfg in FTS_FSDP_TESTS.items()}


FSDP_TEST_CFGS = [
    pytest.param(
        k,
        *test_cfg[0],
        id=k,
        marks=RunIf(**runif_map[test_cfg[1]]) if runif_map.get(test_cfg[1], None) else tuple(),
    )
    for k, test_cfg in FTS_FSDP_TESTS.items()
]


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.13")
@pytest.mark.parametrize(
    "model_cfg_key, model_cls, auto_wrap_policy, use_precision, ft_sched_idx, model_cfg, strategy_adapter_cfg, fts_cfg,\
          trainer_cfg, strategy_cfg",
    FSDP_TEST_CFGS,
)
def test_fsdp_multi_gpus(
    tmpdir,
    recwarn,
    fsdp_ft_schedules,
    model_cfg_key,
    model_cls,
    auto_wrap_policy,
    use_precision,
    ft_sched_idx,
    model_cfg,
    strategy_adapter_cfg,
    fts_cfg,
    trainer_cfg,
    strategy_cfg,
):
    """Conservative (end-to-end) set of tests for FTS support of FSDP."""
    cfg = init_test_cfg(model_cfg_key, model_cfg, fts_cfg, trainer_cfg, strategy_cfg, use_precision)
    (fts_state, lrs_state, w_expected, e_expected, model_cfg, fts_cfg, trainer_cfg, strategy_cfg, precision_opts) = cfg
    seed_everything(42)
    use_dynamo = True if model_cfg.pop("use_dynamo", None) else False
    model = model_cls(**model_cfg)
    strategy_cfg = load_ignore_directives(strategy_cfg, model)
    ft_sched = fsdp_ft_schedules[ft_sched_idx]
    test_cfg, fts_cls = init_fts_cfg(fts_state, lrs_state, strategy_adapter_cfg, fts_cfg, tmpdir)
    callbacks = callbacks_cfg(fts_cls, ft_sched, test_cfg, {"patience": 2}, {"save_top_k": 3})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, **strategy_cfg)
    trainer = configure_trainer(tmpdir, strategy, callbacks, {**trainer_cfg, **precision_opts})
    configured_model = torch.compile(model) if use_dynamo else model
    if e_expected:
        gen_exceptions(trainer, configured_model, model_cfg_key, e_expected)
    else:
        trainer.fit(configured_model)
        default_fts_sanity_chk(trainer)
    if trainer.is_global_zero and recwarn:
        check_fts_fsdp_warns(w_expected, recwarn, use_dynamo)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.13")
@pytest.mark.parametrize(
    "model_cls, awp, ft_sched_idx, model_cfg",
    [pytest.param(adam_model, cust_awp, 0, unwrap_7, id="cust_noprec_resume")],
)
def test_fsdp_multi_gpus_resume(tmpdir, recwarn, fsdp_ft_schedules, fsdp_ckpt, model_cls, awp, ft_sched_idx, model_cfg):
    """Conservative (end-to-end) test for FTS training resumption with FSDP."""
    model_cfg = model_cfg or {}
    expected_state = (path_default, *nones(3))
    warns_expected = expected_state[1]
    seed_everything(42)
    model = model_cls(**model_cfg)
    ft_sched = fsdp_ft_schedules[ft_sched_idx]
    callbacks = callbacks_cfg(FinetuningScheduler, ft_sched, {}, {"patience": 1}, {"save_last": True})
    ckpt_path, use_orig_params = fsdp_ckpt
    use_orig_kwargs = DISABLE_USE_ORIG if not use_orig_params else {}
    strategy = FSDPStrategy(auto_wrap_policy=awp, **use_orig_kwargs, cpu_offload=CPUOffload(offload_params=True))
    trainer = configure_trainer(tmpdir, strategy, callbacks, {"max_epochs": 3})
    trainer.ckpt_path = ckpt_path
    trainer.fit(model)
    default_fts_sanity_chk(trainer)
    if trainer.is_global_zero:
        check_fts_fsdp_warns(warns_expected, recwarn)


def gen_exceptions(trainer, model, model_cfg_key, exception_expected):
    if model_cfg_key == "no_fsdp_params_p0":
        with mock.patch.object(FSDPStrategyAdapter, "_rank_zero_logger", 42):
            with pytest.raises(MisconfigurationException, match=exception_expected):
                trainer.fit(model)
    elif model_cfg_key == "unsupp_torch_version":
        with mock.patch("finetuning_scheduler.strategy_adapters.fsdp._TORCH_GREATER_EQUAL_1_13", False):
            with pytest.raises(MisconfigurationException, match=exception_expected):
                trainer.fit(model)
    else:
        with pytest.raises(MisconfigurationException, match=exception_expected):
            trainer.fit(model)


def init_fts_cfg(fts_state, lrs_state, strategy_adapter_cfg, fts_cfg, tmpdir):
    def_fts_cfg = {
        "logging_level": DEBUG,
        "expected_state": fts_state,
        "lrs_state": lrs_state,
        "strategy_adapter_cfg": strategy_adapter_cfg,
        # "state_log_dir": tmpdir
    }
    fts_cls = fts_cfg.pop("test_fts") if fts_cfg and fts_cfg.get("test_fts") else FSDPTestFinetuningScheduler
    test_cfg = {**fts_cfg, **def_fts_cfg}
    return test_cfg, fts_cls


def init_test_cfg(model_cfg_key, model_cfg, fts_cfg, trainer_cfg, strategy_cfg, use_precision):
    expected_state = EXPECTED_FSDP_FTS_RESULTS[model_cfg_key]
    fts_state, warns_expected, exception_expected, lrs_state = expected_state
    init_cfg = model_cfg, fts_cfg, trainer_cfg, strategy_cfg, use_precision
    model_cfg, fts_cfg, trainer_cfg, strategy_cfg, precision_opts = map_component_cfgs(*init_cfg)
    return (
        fts_state,
        lrs_state,
        warns_expected,
        exception_expected,
        model_cfg,
        fts_cfg,
        trainer_cfg,
        strategy_cfg,
        precision_opts,
    )


def map_component_cfgs(model_cfg, fts_cfg, trainer_cfg, strategy_cfg, use_precision):
    trainer_cfg = trainer_cfg or {"max_epochs": 3}
    model_cfg = model_cfg or {}
    fts_cfg = fts_cfg or {}
    strategy_cfg = strategy_cfg or {"cpu_offload": True}
    precision_opts = {"precision": "16-true"} if use_precision else {}
    return model_cfg, fts_cfg, trainer_cfg, strategy_cfg, precision_opts


def load_ignore_directives(strategy_cfg, model):
    if strategy_cfg.get("test_ignored_parameters_names", None):
        strategy_cfg["ignored_states"] = [
            model.get_parameter(n) for n in strategy_cfg.pop("test_ignored_parameters_names")
        ]
    elif strategy_cfg.get("test_ignored_modules_names", None):
        strategy_cfg["ignored_modules"] = [
            model.get_submodule(n) for n in strategy_cfg.pop("test_ignored_modules_names")
        ]
    return strategy_cfg


def callbacks_cfg(fts_cls, ft_sched, non_def_fts_cfg, fts_es_cfg, fts_ckpt_cfg):
    default_dep_cfg = {"monitor": "val_loss", "verbose": True}
    fts_es_cfg = {**fts_es_cfg, **default_dep_cfg}
    fts_ckpt_cfg = {**fts_ckpt_cfg, **default_dep_cfg}
    override_fts_cb = {
        "test_es": {"override_inst": None, "default_cls": FTSEarlyStopping, "default_cfg": fts_es_cfg},
        "test_ckpt": {"override_inst": None, "default_cls": FTSCheckpoint, "default_cfg": fts_ckpt_cfg},
    }
    for test_k in ["test_es", "test_ckpt"]:
        if non_def_fts_cfg.get(test_k):
            override_fts_cb[test_k]["override_inst"] = non_def_fts_cfg.pop(test_k)
    default_fts_cfg = {"ft_schedule": ft_sched}
    fts_cfg = {**non_def_fts_cfg, **default_fts_cfg}
    callbacks = [fts_cls(**fts_cfg)]
    for test_k in ["test_es", "test_ckpt"]:
        if override_fts_cb[test_k]["override_inst"]:
            if isinstance(override_fts_cb[test_k]["override_inst"], (FTSEarlyStopping, FTSCheckpoint)):
                callbacks.append(override_fts_cb[test_k]["override_inst"])
        else:
            callbacks.append(override_fts_cb[test_k]["default_cls"](**override_fts_cb[test_k]["default_cfg"]))
    return callbacks


def configure_trainer(tmpdir, strategy, callbacks, extra_trainer_cfg):
    defaults = {"accelerator": "gpu", "devices": 2, "default_root_dir": tmpdir}
    base_config = {"strategy": strategy, "callbacks": callbacks, **defaults}
    trainer = Trainer(**base_config, **extra_trainer_cfg)
    return trainer


def check_fts_fsdp_warns(warns_expected, recwarn, use_dynamo=False):
    fsdp_warns = FSDP_BASE_WARNS
    if use_dynamo:
        fsdp_warns.extend(FSDP_DYNAMO_EXPECTED_WARNS)
    if warns_expected:
        unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=warns_expected)
        assert not unmatched
        fsdp_warns.extend(warns_expected)
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=fsdp_warns)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


def default_fts_sanity_chk(trainer):
    finetuningscheduler_callback = get_fts(trainer)
    assert finetuningscheduler_callback.depth_remaining == 0
    assert finetuningscheduler_callback.curr_depth == 2
    assert finetuningscheduler_callback.curr_depth == finetuningscheduler_callback.max_depth
