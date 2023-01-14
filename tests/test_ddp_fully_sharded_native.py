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
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
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
    "The distutils package is deprecated",  # for tensorboard import as of PT 1.13.1
    "is still running",  # subprocess is implicitly cleaned up
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
    fsdp_nondis_mod_ex_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_nondis_mod_ex_sched_dict[1]["params"] = ["layer.[2-4].*"]
    del fsdp_nondis_mod_ex_sched_dict[1]["max_transition_epoch"]
    del fsdp_nondis_mod_ex_sched_dict[2]
    fsdp_adam_gen_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_adam_gen_sched_dict[0]["max_transition_epoch"] = 2
    fsdp_adam_gen_sched_dict[1]["max_transition_epoch"] = 4
    fsdp_ext_gen_sched_dict = deepcopy(fsdp_gen_sched_dict)
    fsdp_ext_gen_sched_dict[0]["params"] = ["layer.(5|[7-8]).*"]
    return (
        fsdp_gen_sched_dict,
        fsdp_nondis_mod_sched_dict,
        fsdp_bn_gen_sched_dict,
        fsdp_shared_param_sched_dict,
        fsdp_adam_gen_sched_dict,
        fsdp_nondis_mod_ex_sched_dict,
        fsdp_ext_gen_sched_dict,
    )


@pytest.fixture(scope="function")
def fsdp_ckpt(tmpdir_factory, fsdp_ft_schedules) -> Dict:
    """A fixture that generates a checkpoint with a sharded model."""
    seed_everything(42)
    test_model_cfg = {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}}
    strategy = DDPFullyShardedNativeStrategy(
        auto_wrap_policy=custom_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
    )
    callbacks = [
        FinetuningScheduler(ft_schedule=fsdp_ft_schedules[0]),
        FTSEarlyStopping(monitor="val_loss", patience=1),
        FTSCheckpoint(monitor="val_loss", save_last=True, verbose=True),
    ]
    model = FTSBaseFSDPModel(**test_model_cfg)
    trainer = Trainer(
        default_root_dir=tmpdir_factory.getbasetemp(),
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs=1,
    )
    trainer.fit(model)
    return trainer.checkpoint_callback.best_model_path


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
        # return torch.optim.SGD(parameters, lr=0.1)
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


class FTSCsmAdamFSDPModel(FTSBaseFSDPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(parameters, weight_decay=1.0e-05, eps=1.0e-07, lr=1.0e-05)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [lr_scheduler]

    def configure_sharded_model(self) -> None:
        # the model is already wrapped with FSDP: no need to wrap again!
        if isinstance(self.layer, FullyShardedDataParallel):
            return
        for i, layer in enumerate(self.layer):
            if i in self.fsdp_mask["wrapped_mods"]:
                self.layer[i] = wrap(layer)
        self.layer = wrap(self.layer)


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


def custom_auto_wrap_ext_policy(
    module,
    recurse,
    unwrapped_params: int,
    min_num_params: int = int(1e8),
) -> bool:
    return unwrapped_params >= 529


def warn_custom_auto_wrap_policy(
    module,
    recurse,
    unwrapped_params: int,
    min_num_params: int = int(1e8),
) -> bool:
    return unwrapped_params >= 1100


EXPECTED_FSDP_FTS_RESULTS = {
    "cust_awp_noprec": (
        {
            0: (2, 4),
            1: (6, 12),
            2: (7, 14),
        },
        None,
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
    "cust_awp_noprec_ignore_no_offload": (
        {
            0: (2, 4),
            1: (7, 12),
            2: (8, 14),
        },
        None,
        None,
    ),
    "non_disjoint_phase_fsdp_params": ({}, None, "do not have disjoint FSDP-flattened parameter"),
    "non_disjoint_phase_mods": ({}, None, "not have disjoint"),
    "non_disjoint_excluded_ft_params": ({}, None, "parameters not included in"),
    "already_fsdp_wrapped": ({}, None, "already wrapped by FSDP"),
    "no_fsdp_params_p0": ({}, None, "one or more FSDP"),
    "warn_unsupp_nodecay": ({}, "will now be unset", None),
    "unmatched_awp_overrides": ({}, None, "did not match any named modules"),
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
        ("Pruning explicitly specified",),
        None,
    ),
    "override_csm_adam_noprec": (
        {
            0: (2, 4),
            1: (2, 4),
            2: (6, 12),
            3: (6, 12),
            4: (7, 14),
        },
        None,
        None,
    ),
    "cust_awp_override_prec": (
        {
            0: (2, 4),
            1: (6, 12),
            2: (7, 14),
        },
        None,
        None,
    ),
    "cust_awp_override_prec_ext": (
        {
            0: (3, 6),
            1: (7, 14),
            2: (8, 16),
        },
        None,
        None,
    ),
    "warn_ignore_awp_override": ({}, "will be unset and not applied", None),
    "cust_noprec_resume": (
        {
            0: (2, 4),
            1: (6, 12),
            2: (7, 14),
        },
        None,
        None,
    ),
}


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize(
    "model_cfg_key, model_cls, auto_wrap_policy, use_precision, ft_sched_idx, fit_start_only, strategy_adapter_cfg,\
         model_cfg, fts_cfg, trainer_cfg, strategy_cfg",
    [
        (
            "cust_awp_noprec",
            FTSBaseFSDPModel,
            custom_auto_wrap_policy,
            False,
            0,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}},
            None,
            None,
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
            None,
            None,
        ),
        (
            "cust_awp_noprec_ignore_no_offload",
            FTSBaseFSDPModel,
            custom_auto_wrap_policy,
            False,
            0,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": [0, 1, 2, 3, 5], "unwrapped_mods": [4, 7]}},
            None,
            None,
            {"test_ignored_modules_names": ["layer.4"], "cpu_offload": False},
        ),
        (
            "non_disjoint_phase_fsdp_params",
            FTSBaseFSDPModel,  # FTSCsmFSDPModel,
            warn_custom_auto_wrap_policy,  # None,
            False,
            0,
            False,
            None,
            {
                "fsdp_mask": {"wrapped_mods": [5], "unwrapped_mods": [i for i in list(range(8)) if i != 5]}
            },  # {"fsdp_mask": {"wrapped_mods": [0,1,2,3,5], "unwrapped_mods": [4, 7]}}
            None,
            None,
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
            None,
            None,
        ),
        (
            "non_disjoint_excluded_ft_params",
            FTSCsmFSDPModel,
            None,
            False,
            5,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": [2, 3, 4, 5], "unwrapped_mods": [0, 1, 7]}},
            None,
            None,
            None,
        ),
        (
            "already_fsdp_wrapped",
            AlreadyWrappedFSDPModel,
            custom_auto_wrap_policy,
            False,
            0,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}},
            None,
            None,
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
            None,
            None,
        ),
        (
            "warn_unsupp_nodecay",
            FTSNoDecayFSDPModel,
            custom_auto_wrap_policy,
            False,
            0,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}},
            None,
            None,
            None,
        ),
        (
            "unmatched_awp_overrides",
            FTSBaseFSDPModel,
            warn_custom_auto_wrap_policy,
            True,
            0,
            False,
            {"awp_overrides": ["layer.9", "layer.5"]},
            {"fsdp_mask": {"wrapped_mods": [5, 7], "unwrapped_mods": [i for i in list(range(8)) if i not in [5, 7]]}},
            None,
            None,
            None,
        ),
        (
            "cust_awp_prec",
            FTSBaseFSDPModel,
            custom_auto_wrap_policy,
            True,
            0,
            False,
            None,
            {
                "fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]},
                "precision_key": "auto_16",
            },
            None,
            None,
            None,
        ),
        (
            "batch_norm_auto_prec",
            FTSBatchNormFSDPModel,
            custom_auto_wrap_policy,
            True,
            2,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": list(range(7)), "unwrapped_mods": [8]}, "precision_key": "auto_16"},
            None,
            None,
            None,
        ),
        (
            "shared_params_auto_prec",
            FTSSharedParamFSDPModel,
            custom_auto_wrap_policy,
            True,
            3,
            False,
            {"awp_overrides": ["l.*yer.1"]},
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}, "precision_key": "auto_16"},
            None,
            None,
            None,
        ),
        (
            "override_csm_adam_noprec",
            FTSCsmAdamFSDPModel,
            None,
            False,
            4,
            False,
            None,
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}, "diverge_on_epoch": 1},
            None,
            {"max_epochs": 5},
            None,
        ),
        (
            "cust_awp_override_prec",
            FTSBaseFSDPModel,
            custom_auto_wrap_policy,
            True,
            0,
            False,
            {"awp_overrides": ["l.*yer.7"]},
            {
                "fsdp_mask": {"wrapped_mods": list(range(6)) + [7], "unwrapped_mods": []},
                "precision_key": "auto_16",
            },
            None,
            None,
            None,
        ),
        (
            "cust_awp_override_prec_ext",
            FTSExtFSDPModel,
            custom_auto_wrap_ext_policy,
            True,
            6,
            False,
            {"awp_overrides": ["l.*yer.8", "layer.7"]},
            {
                "fsdp_mask": {"wrapped_mods": list(range(6)) + [7, 8], "unwrapped_mods": []},
                "precision_key": "auto_16",
            },
            None,
            None,
            None,
        ),
        (
            "warn_ignore_awp_override",
            FTSCsmFSDPModel,
            None,
            False,
            0,
            False,
            {"awp_overrides": ["layer.7"]},
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}},
            None,
            None,
            None,
        ),
    ],
    ids=[
        "cust_awp_noprec",
        "override_csm_noprec",
        "cust_awp_noprec_ignore_no_offload",
        "non_disjoint_phase_fsdp_params",
        "non_disjoint_phase_mods",
        "non_disjoint_excluded_ft_params",
        "already_fsdp_wrapped",
        "no_fsdp_params_p0",
        "warn_unsupp_nodecay",
        "unmatched_awp_overrides",
        "cust_awp_prec",
        "batch_norm_auto_prec",
        "shared_params_auto_prec",
        "override_csm_adam_noprec",
        "cust_awp_override_prec",
        "cust_awp_override_prec_ext",
        "warn_ignore_awp_override",
    ],
)
def test_fsdp_native_multi_gpus(
    tmpdir,
    recwarn,
    fsdp_ft_schedules,
    model_cfg_key,
    model_cls,
    auto_wrap_policy,
    use_precision,
    ft_sched_idx,
    fit_start_only,
    strategy_adapter_cfg,
    model_cfg,
    fts_cfg,
    trainer_cfg,
    strategy_cfg,
):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run."""
    model_cfg = model_cfg or {}
    expected_state = EXPECTED_FSDP_FTS_RESULTS[model_cfg_key]
    warns_expected = expected_state[1]
    fts_cfg = fts_cfg or {}
    trainer_cfg = trainer_cfg or {"max_epochs": 3}
    strategy_cfg = strategy_cfg or {"cpu_offload": True, "ignored_modules": None}
    exception_expected = expected_state[2]
    seed_everything(42)
    model = model_cls(**model_cfg)
    fts_cls = FitStartOnlyFTS if fit_start_only else FSDPTestFinetuningScheduler
    callbacks = [
        fts_cls(
            ft_schedule=fsdp_ft_schedules[ft_sched_idx],
            logging_level=DEBUG,
            strategy_adapter_cfg=strategy_adapter_cfg,
            expected_state=expected_state[0],
            **fts_cfg,
        ),
        FTSEarlyStopping(monitor="val_loss", patience=2),
        FTSCheckpoint(monitor="val_loss", save_top_k=3, verbose=True),
    ]
    if strategy_cfg.get("test_ignored_modules_names", None):
        strategy_cfg["ignored_modules"] = [
            model.get_submodule(n) for n in strategy_cfg.pop("test_ignored_modules_names")
        ]
    strategy = DDPFullyShardedNativeStrategy(auto_wrap_policy=auto_wrap_policy, **strategy_cfg)
    precision_opts = {"precision": 16} if use_precision else {}
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        callbacks=callbacks,
        **trainer_cfg,
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
        check_fts_fsdp_warns(warns_expected, recwarn)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize(
    "model_cfg_key, model_cls, auto_wrap_policy, ft_sched_idx, model_cfg",
    [
        (
            "cust_noprec_resume",
            FTSBaseFSDPModel,
            custom_auto_wrap_policy,
            0,
            {"fsdp_mask": {"wrapped_mods": list(range(6)), "unwrapped_mods": [7]}},
        ),
    ],
    ids=["cust_noprec_resume"],
)
def test_fsdp_native_multi_gpus_resume(
    tmpdir,
    recwarn,
    fsdp_ft_schedules,
    fsdp_ckpt,
    model_cfg_key,
    model_cls,
    auto_wrap_policy,
    ft_sched_idx,
    model_cfg,
):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run."""
    model_cfg = model_cfg or {}
    expected_state = EXPECTED_FSDP_FTS_RESULTS[model_cfg_key]
    warns_expected = expected_state[1]
    seed_everything(42)
    model = model_cls(**model_cfg)
    callbacks = [
        FinetuningScheduler(ft_schedule=fsdp_ft_schedules[ft_sched_idx]),
        FTSEarlyStopping(monitor="val_loss", patience=1),
        FTSCheckpoint(monitor="val_loss", save_last=True, verbose=True),
    ]
    strategy = DDPFullyShardedNativeStrategy(
        auto_wrap_policy=auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True)
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs=3,
    )
    trainer.fit(model, ckpt_path=fsdp_ckpt)
    finetuningscheduler_callback = get_fts(trainer)
    assert finetuningscheduler_callback.depth_remaining == 0
    assert finetuningscheduler_callback.curr_depth == 2
    assert finetuningscheduler_callback.curr_depth == finetuningscheduler_callback.max_depth

    if trainer.is_global_zero:
        check_fts_fsdp_warns(warns_expected, recwarn)


def check_fts_fsdp_warns(warns_expected, recwarn):
    fsdp_warns = FSDP_BASE_WARNS
    if warns_expected:
        unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=warns_expected)
        assert not unmatched
        fsdp_warns.extend(warns_expected)
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=fsdp_warns)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
