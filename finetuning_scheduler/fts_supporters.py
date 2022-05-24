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
r"""
Finetuning Scheduler Supporters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classes composed to support scheduled finetuning

"""
import itertools
import logging
import os
import pathlib
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import KeysView
from dataclasses import dataclass, field, fields
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.core.optimizer import _MockOptimizer
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_10, rank_zero_info, rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
from pytorch_lightning.utilities.types import _LRScheduler, LRSchedulerConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer

log = logging.getLogger(__name__)

CALLBACK_DEP_PARENTS = {"ModelCheckpoint": ModelCheckpoint, "EarlyStopping": EarlyStopping}
CALLBACK_ATTRS = ("ft_schedule", "max_depth")
TARGET_CALLBACK_REF = "FinetuningScheduler"


@dataclass
class FTSState:
    """Dataclass to encapsulate the
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` internal state."""

    _resume_fit_from_ckpt: bool = False
    _ft_epoch: int = 0
    _ft_global_steps: int = 0
    _curr_depth: int = 0
    _best_ckpt_depth: int = 0
    _ft_sync_props: Tuple = (
        ("epoch_progress.current.completed", "_ft_epoch"),
        ("epoch_loop.global_step", "_ft_global_steps"),
    )
    _ft_sync_objects: Optional[Tuple] = None
    _curr_thawed_params: List = field(default_factory=list)
    _fts_ckpt_metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._fts_ckpt_metadata = {
            "current_ckpt_depth": self._curr_depth,
            "best_ckpt_depth": self._best_ckpt_depth,
            "best_ckpt_pgs": {},
        }


class CallbackResolverMixin(ABC):
    """Give user-provided callbacks with the ability to connect to another user-provided callback.

    This resolution logic is provided in order to avoid callback-dependent trainer attributes (e.g.
    trainer.finetuningscheduler_callback)
    """

    def __init__(
        self,
        callback_attrs: Tuple = CALLBACK_ATTRS,
        callback_parents: Dict = CALLBACK_DEP_PARENTS,
        target_callback_ref: str = TARGET_CALLBACK_REF,
        support_multiple: bool = False,
    ) -> None:
        """Initialize the user-provided callback depedency resolver in accordance with the user-provided module
        configuration.

        Args:
            callback_attrs (Tuple, optional): Attribute signature of user-provided callback to be structurally detected
                and connected. Defaults to CALLBACK_ATTRS defined in the user-provided module.
            callback_parents (Dict, optional): The parent classes of all user-provided callbacks in the module that
                should be connected to the target user-provided callback. Defaults to CALLBACK_DEP_PARENTS in the
                user-provided module.
            target_callback_ref (str, optional): The name of the target user-provided callback to connect to. For each
                subclass of CALLBACK_DEP_PARENTS, an attribute named ``(target_callback_ref.lower())_callback`` will be
                added. Defaults to TARGET_CALLBACK_REF in the user-provided module.
            support_multiple (bool, optional): Whether multiple instances of the target user-provided callback (only the
                first of which will be connected to) are allowed. Defaults to False.
        """
        super().__init__()
        self.callback_attrs = callback_attrs
        self.callback_parents = callback_parents
        self.target_callback_ref = target_callback_ref
        self.callback_handle = f"{self.target_callback_ref.lower()}_callback"
        self.support_multiple = support_multiple
        setattr(self, self.callback_handle, None)

    def connect_callback(self, trainer: "pl.Trainer", reconnect: bool = False) -> None:
        """Connect each user-provided callback dependency that needs to be connected to the target user-provided
        callback.

        Args:
            trainer (pl.Trainer): The :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object.
            reconnect (bool, optional): Whether to check for an updated target callback object even if one is already
                resolved. Predominantly useful in the context of testing. Defaults to False.

        Raises:
            MisconfigurationException: If no target callback is detected
            MisconfigurationException: if :attr:`support_multiple` is ``False`` and multiple target callbacks are
                detected.
        """
        if self.__dict__[self.callback_handle] and not reconnect:
            return
        resolved_callbacks = [c for c in trainer.callbacks if all([hasattr(c, a) for a in self.callback_attrs])]
        if not resolved_callbacks:
            raise MisconfigurationException(
                f"{self.__class__.__name__} is intended for use with a {self.target_callback_ref}. If not using a"
                f"{self.target_callback_ref} callback, please use the standard "
                f"{[k for k,v in self.callback_parents.items() if isinstance(self,v)][0]} callback."
            )
        elif not self.support_multiple and len(resolved_callbacks) > 1:
            raise MisconfigurationException(
                f"Use of multiple {resolved_callbacks[0].__class__.__name__} callbacks is"
                "not currently supported. Please provide a maximum of one."
            )
        else:
            setattr(self, self.callback_handle, resolved_callbacks[0])


class FTSEarlyStopping(EarlyStopping, CallbackResolverMixin):
    r"""
    Extends/specializes :external+pl:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` to facilitate
    multi-phase scheduled finetuning.

    Adds :attr:`es_phase_complete`, :attr:`final_phase` and :attr:`finetuningscheduler_callback` attributes and modifies
    ``EarlyStopping._evaluate_stopping_criteria`` to enable multi-phase behavior. Usage of
    :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` is identical to
    :external+pl:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` except the former will evaluate the
    specified early stopping criteria at every scheduled phase.
    :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` will automatically be
    used if a :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback is detected
    and :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.epoch_transitions_only` is ``False``

    .. warning::

       :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` is in beta and subject to change. For detailed
       usage information, see :external+pl:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Attributes:
            es_phase_complete (bool):
                Used to determine if the current phase's early stopping criteria have been met.
            final_phase (bool):
                Used to indicate whether the current phase is the final scheduled phase.
            finetuningscheduler_callback (pytorch_lightning.callbacks.Callback):
                Reference to the :class:`~finetuning_scheduler.fts.FinetuningScheduler`
                callback being used.
            check_on_train_epoch_end (bool): Whether to run early stopping check at the end of the training epoch. If
                this is ``False``, then the check runs at the end of the validation. Defaults to ``None`` similar to
                :external+pl:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` but is set to
                ``False`` during setup unless overridden.
        """
        super().__init__(*args, **kwargs)
        self.es_phase_complete = True
        self.final_phase = True
        self.finetuningscheduler_callback = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Ensure a :class:`~finetuning_scheduler.fts.FinetuningScheduler` is provided before beginning
        training."""
        self.connect_callback(trainer)
        if self._check_on_train_epoch_end is None:
            # post-validation saving/evaluation is the most common fts usage pattern
            self._check_on_train_epoch_end = False
        super().setup(trainer, pl_module, stage)

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """Evaluate whether and why to stop the current training session.

        Args:
            current (torch.Tensor): The current monitored value to be evaluated

        Returns:
            Tuple[bool, Optional[str]]: Whether the training session should stop and if so, the reason why.
        """
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                if self.final_phase:
                    should_stop = True
                    reason = (
                        f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                        f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                    )
                else:
                    self.es_phase_complete = True
                    self.wait_count = 0
        return should_stop, reason


class FTSCheckpoint(ModelCheckpoint, CallbackResolverMixin):
    r"""
    Extends/specializes :external+pl:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` to facilitate
    multi-phase scheduled finetuning. Overrides the
    ``state_dict`` and ``load_state_dict`` hooks to maintain additional state (:attr:`current_ckpt_depth`,
    :attr:`best_ckpt_depth`, :attr:`finetuningscheduler_callback`). Usage of
    :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` is identical to
    :external+pl:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` and
    :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` will automatically be used if a
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback is detected.

    .. warning::
        :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` is in beta and subject to change. For detailed usage
        information, see :external+pl:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Attributes:
            current_ckpt_depth (int):
                Used to track the depth of most recently saved checkpoint
            best_ckpt_depth (int):
                Used to track the depth of the best known checkpoint (it may be from a different training depth)
            finetuningscheduler_callback (pytorch_lightning.callbacks.Callback):
                Reference to the :class:`~finetuning_scheduler.fts.FinetuningScheduler`
                callback being used.
            save_on_train_epoch_end (bool): Whether to run checkpointing at the end of the training epoch. If this is
                ``False``, then the check runs at the end of the validation. Defaults to ``None`` similar to
                :external+pl:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` but is set to
                ``False`` during setup unless overridden.
        """
        super().__init__(*args, **kwargs)
        self.current_ckpt_depth = 0
        self.best_ckpt_depth = 0
        self.finetuningscheduler_callback = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Verify a valid callback configuration is present before beginning training.

        Args:
            trainer: The :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object

        Raises:
            MisconfigurationException:
                If a :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback is not
                found on initialization (``finetuningscheduler_callback`` is ``None``)
            MisconfigurationException:
                If :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.restore_best` is
                ``True`` and ``ModelCheckpoint.save_top_k`` is either ``None`` or ``0``
            MisconfigurationException:
                If :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.restore_best` is
                ``True`` and ``monitor`` is ``None``
        """
        # note if only saving best ckpt rather than top k > 1, current_ckpt_depth == best_ckpt_depth
        self.connect_callback(trainer)
        if self.finetuningscheduler_callback.restore_best:  # type: ignore[attr-defined]
            if not self.save_top_k or self.save_top_k == 0:
                raise MisconfigurationException(
                    f"{type(self.finetuningscheduler_callback)} was directed to restore checkpoints"
                    f"(restore_best=True) but {self.__class__.__name__} is configured to save no intermediate"
                    "checkpoints (save_top_k is 0 or None). Please set save_top_k to a non-zero value or set"
                    "restore_best=False"
                )
            elif not self.monitor:
                raise MisconfigurationException(
                    f"{type(self.finetuningscheduler_callback)} was directed to restore checkpoints"
                    f"(restore_best=True) but {self.__class__.__name__} but has no quantity to monitor (monitor=None)."
                    "Please provide a value to monitor or set restore_best=False."
                )
        if self._save_on_train_epoch_end is None:
            # post-validation saving/evaluation is the most common fts usage pattern
            self._save_on_train_epoch_end = False
        super().setup(trainer, pl_module, stage)

    def state_dict(self) -> Dict[str, Any]:
        """Overrides. :external+pl:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`'s
        ``state_dict`` method to maintain multi-phase training depth state.

        Returns:
            Dict[str, Any]: the callback state dictionary that will be saved.
        """
        self.current_ckpt_depth = self.finetuningscheduler_callback.curr_depth  # type: ignore[attr-defined]
        # note, if current score is precisely the best score but a previous depth had the same score the
        # best ckpt depth will be set to the latest (deepest) depth with that score.
        # a future enhancement of per-depth best score mapping could allow more fine-grained control of this behavior
        if self.current_score == self.best_model_score:
            self.best_ckpt_depth = self.current_ckpt_depth
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath,
            "best_k_models": self.best_k_models,
            "kth_best_model_path": self.kth_best_model_path,
            "kth_value": self.kth_value,
            "last_model_path": self.last_model_path,
            "current_ckpt_depth": self.current_ckpt_depth,
            "best_ckpt_depth": self.best_ckpt_depth,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Overrides :external+pl:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`'s
        ``load_state_dict`` method to load multi-phase training depth state.

        Args:
            state_dict: the callback state dict of :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint`.
        """
        assert self.finetuningscheduler_callback is not None
        assert isinstance(self.finetuningscheduler_callback.pl_module.trainer.early_stopping_callback, FTSEarlyStopping)
        # if we're starting a new level from another checkpoint depth, wait_count could be > 0 contingent on the
        # min_delta
        if self.finetuningscheduler_callback.curr_depth > self.best_ckpt_depth:
            if not self.finetuningscheduler_callback.epoch_transitions_only:
                self.finetuningscheduler_callback.pl_module.trainer.early_stopping_callback.wait_count = 0
        if self.finetuningscheduler_callback._fts_state._resume_fit_from_ckpt:
            dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)
            if self.dirpath == dirpath_from_ckpt:
                self.best_k_models = state_dict.get("best_k_models", self.best_k_models)
                self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
                self.kth_value = state_dict.get("kth_value", self.kth_value)
                self.current_ckpt_depth = state_dict["current_ckpt_depth"]
                self.best_ckpt_depth = state_dict["best_ckpt_depth"]
            else:
                warnings.warn(
                    f"The dirpath has changed from {dirpath_from_ckpt!r} to {self.dirpath!r}, therefore"
                    " `best_model_score`, `kth_best_model_path`, `kth_value` and `best_k_models` won't be reloaded."
                    " Only `last_model_path`, `best_model_path` and `current_ckpt_depth` will be reloaded."
                )
                self.current_ckpt_depth = state_dict["current_ckpt_depth"]
                self.best_ckpt_depth = self.current_ckpt_depth
            self.last_model_path = state_dict.get("last_model_path", self.last_model_path)
            self.best_model_path = state_dict["best_model_path"]


class UniqueKeyLoader(yaml.SafeLoader):
    """Alters SafeLoader to enable duplicate key detection by the SafeConstructor."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> Dict:
        """Overrides the construct_mapping method of the SafeConstructor to raise a ValueError if duplicate keys
        are found.

        Inspired by and adapated from https://stackoverflow.com/a/63215043
        """
        mapping = []
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key not in mapping:
                mapping.append(key)
            else:
                raise ValueError(key)
        return super().construct_mapping(node, deep)


class ScheduleParsingMixin(ABC):
    """Functionality for parsing and validating finetuning schedules."""

    # proper initialization of these variables should be done in the child class
    pl_module: pl.LightningModule
    ft_schedule: Optional[Union[str, dict]]
    reinit_lr_cfg: Optional[Dict]

    def _validate_ft_sched(self) -> Tuple[int, int]:
        """Ensure the explicitly specified finetuning schedule has a valid configuration.

        Returns:
            Tuple[int, int]: A tuple of ints specifying:
                1. The depth of the final scheduled phase
                2. The maximum epoch watermark explicitly specified in the schedule
        """
        max_epoch_wm = -1
        max_phase = 0
        self._validate_schedule_keys()
        self._validate_lr_scheduler_cfg()
        named_params = dict(self.pl_module.named_parameters()).keys()
        for depth in self.ft_schedule.keys():  # type: ignore[union-attr]
            max_phase = max(max_phase, depth)
            self._parse_phase(depth, named_params)
            if depth > 0:
                assert isinstance(self.ft_schedule, Dict)
                curr_max_epoch = self.ft_schedule[depth]["max_transition_epoch"]
                if 0 <= curr_max_epoch <= max_epoch_wm:
                    es_addendum = " depending upon EarlyStopping criteria."
                    rank_zero_info(
                        f"Specified max_transition_epoch of depth {depth}"
                        f"({self.ft_schedule[depth]['max_transition_epoch']}) is less than or equal to a "
                        f"previous max_transition_epoch ({max_epoch_wm}), depth may execute only a single "
                        f"epoch{'.' if self.epoch_transitions_only else es_addendum}"  # type: ignore[attr-defined]
                    )
                max_epoch_wm = max(max_epoch_wm, curr_max_epoch)
        self._validate_phases_disjoint()
        if self.epoch_transitions_only:  # type: ignore[attr-defined]
            self._validate_epoch_transitions()
        return max_phase, max_epoch_wm

    def _prune_pl_lrs(self, pl_lrs_cfg: Dict) -> Dict:
        """Prune keys not part of a valid PyTorch Lightning lr scheduler configuration (if automatic optimization
        used)

        Args:
            pl_lrs_cfg (Dict): User-provided PyTorch Lightning lr scheduler configuration

        Returns:
            Dict: PyTorch Lightning lr scheduler configuration without extra keys
        """
        if self.pl_module.automatic_optimization:
            supported_keys = {field.name for field in fields(LRSchedulerConfig)}
            extra_keys = pl_lrs_cfg.keys() - supported_keys
            if extra_keys:
                rank_zero_warn(
                    f"Found unsupported keys in the lr scheduler dict: {extra_keys}.",
                    category=RuntimeWarning,
                )
        return {k: v for k, v in pl_lrs_cfg.items() if k in supported_keys}

    def _pl_lrs_validation(self, pl_lrs_cfg: Dict) -> None:
        """Check basic pl lrs config (we aren't instantiating the new scheduler yet so can't validate everything)
        replicating basic PL lr schedule config validation here, originally based on https://bit.ly/3NldbaG.

        Args:
            pl_lrs_cfg (Dict): The PyTorch Lightning learning rate scheduler configuration option dictionary

        Raises:
            MisconfigurationException: If `pl_lrs_cfg['interval']` is not either `step` or `epoch`. Warnings raised for
                unsupported keys that will be ignored.
        """
        if self.pl_module.automatic_optimization:
            if "interval" in pl_lrs_cfg and pl_lrs_cfg["interval"] not in ("step", "epoch"):
                raise MisconfigurationException(
                    'The "interval" key in lr scheduler dict must be "step" or "epoch"'
                    f' but is "{pl_lrs_cfg["interval"]}"'
                )

    def _lr_scheduler_reinit_key_validation(self, target_sched: Dict, depth: Optional[int] = None) -> None:
        """Validate the keys in a given lr reinitialization configuration.

        Args:
            target_sched (Dict): The provided lr scheduler reinitialization configuration for either an implicit mode
                finetuning schedule (passed via `reinit_lr_cfg`) or for a given explicity mode finetuning phase (passed
                via `new_lr_scheduler` for a given phase)
            depth (Optional[int], optional): If parsing an explicit schedule, the current phase. Defaults to None.

        Raises:
            MisconfigurationException: If an `init_pg_lrs` key is provided in implicit mode training
                (via `reinit_lr_cfg`).
            MisconfigurationException: If an `lr_scheduler_init` key is missing in the lr scheduler reinitialization
                configuration.
            MisconfigurationException: If the configuration provided in `lr_scheduler_init` does not specify a
                `class_path` for the lr scheduler to be instantiated.
        """
        if "init_pg_lrs" in target_sched.keys() and (self.reinit_lr_cfg or not depth):
            raise MisconfigurationException(
                "Specifying a `init_pg_lrs` key in the lr scheduler configuration passed via `reinit_lr_cfg` (i.e. "
                "implicit mode training) is not a valid configuration since the same lr scheduler configuration "
                "is intended to be reinitialized at every finetuning phase with implicit mode finetuning."
            )
        # validate lr_scheduler_init config
        if "lr_scheduler_init" not in target_sched.keys():
            phase_specific_msg = "" if not depth else f"for phase {depth}"
            raise MisconfigurationException(
                "Specifying a lr scheduler configuration to reinitialize with requires a valid lr scheduler "
                "configuration dictionary be provided via a `lr_scheduler_init` key but no such key was found "
                + phase_specific_msg
                + "."
            )
        # if we're passing pl lr scheduler configuration, validate the keys
        if "pl_lrs_cfg" in target_sched.keys():
            self._pl_lrs_validation(pl_lrs_cfg=target_sched["pl_lrs_cfg"])
        if not target_sched["lr_scheduler_init"].get("class_path"):
            phase_specific_msg = "the specified lr schedule config." if not depth else f"the specified phase ({depth})."
            raise MisconfigurationException(
                "Specifying an `lr_scheduler_init` requires at least a  `class_path` to be specified "
                "but this is not the case for " + phase_specific_msg
            )
        if "init_pg_lrs" in target_sched.keys():
            warn_msg = (
                "Found an `init_pg_lrs` key in the specified lr scheduler reinitialization config. Remember to "
                "ensure the number of specified parameter groups matches the number of parameter groups created in "
                "in previous phases. This aspect of the optimization path is not currently fully simulated on "
                "`FinetuningScheduler` initialization so is left to the user to validate."
            )
            assert depth
            ScheduleParsingMixin._parse_reint_pg_lrs(depth=depth, init_pg_lrs=target_sched["init_pg_lrs"])
            rank_zero_warn(warn_msg)
        lr_scheduler_init = target_sched.get("lr_scheduler_init")
        assert lr_scheduler_init
        ScheduleParsingMixin._lr_scheduler_sanity_chk(lr_scheduler_init)

    def _lr_scheduler_init_validation(self, lr_reinit_phases: Dict) -> None:
        """Trigger lr scheduler reinitialization configuration validation for all provided configurations. This
        will be a single configuration for implicit mode finetuning or n configurations for explicit mode.

        Args:
            lr_reinit_phases (Dict): Dictionary of lr scheduler reinitialization configurations to parse/validate
        """
        if self.reinit_lr_cfg:
            self._lr_scheduler_reinit_key_validation(lr_reinit_phases)
        else:
            for k, lr_cfg in lr_reinit_phases.items():
                self._lr_scheduler_reinit_key_validation(lr_cfg, k)

    def _validate_lr_scheduler_cfg(self) -> None:
        """Orchestrate lr scheduler reinitialization configuration validation.

        Raises:
            MisconfigurationException: If a `new_lr_scheduler` configuration is passed to the initial training phase.
        """
        assert isinstance(self.ft_schedule, Dict)
        lr_reinit_phases = self.reinit_lr_cfg or {
            k: self.ft_schedule[k].get("new_lr_scheduler")
            for k in self.ft_schedule.keys()
            if self.ft_schedule[k].get("new_lr_scheduler")
        }
        if not lr_reinit_phases:
            return  # no further validation needed since there is no lr scheduler reinitialization configuration
        assert self.pl_module.trainer is not None
        assert self.pl_module.trainer.log_dir is not None
        if 0 in lr_reinit_phases.keys():
            raise MisconfigurationException(
                "You have specified a `new_lr_scheduler` for the initial training phase which is an invalid "
                "configuration. The initial lr_scheduler configuration should be passed to your LightningModule."
            )
        self._lr_scheduler_init_validation(lr_reinit_phases)

    def _validate_schedule_keys(self) -> None:
        """Ensures schedule keys are integers, zero-based and contiguous. If the schedule does not meet these
        requirements, attempts to transform the passed schedule to meet them and writes the candidate schedule out
        for subsequent user validation.

        Raises:
            MisconfigurationException: Raised if the schedule contains non-integer keys and/or non-zero-based and
                contiguous keys.
        """
        assert self.pl_module.trainer is not None
        assert self.pl_module.trainer.log_dir is not None
        assert isinstance(self.ft_schedule, Dict)
        all_ints = all([isinstance(k, int) for k in self.ft_schedule.keys()])
        contiguous = len(self.ft_schedule.keys()) == (max(self.ft_schedule.keys()) + 1)
        rewrite_dest = None
        if not (all_ints and contiguous):
            for i, k in enumerate(sorted(self.ft_schedule.keys())):
                self.ft_schedule[i] = self.ft_schedule.pop(k)
            # write the reconfigured schedule to our log directory to allow user validation
            rewrite_dest = ScheduleImplMixin.save_schedule(
                f"{self.pl_module.__class__.__name__}_ft_schedule_valid.yaml",
                self.ft_schedule,
                self.pl_module.trainer.log_dir,
            )
            err_msg = "The supplied schedule was found to"
            reason_msg = " use non-integer keys " if not all_ints else " have non-contiguous or non-zero-indexed keys "
            raise MisconfigurationException(
                err_msg + reason_msg + "and has thus been reconfigured and saved to "
                f"'{rewrite_dest}'. Please validate the reconfigured schedule and restart "
                "training with a valid schedule."
            )

    def _validate_epoch_transitions(self) -> None:
        """If not composing :external+pl:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` and epoch-driven
        stopping criteria (the default behavior) but instead specifying exclusively epoch-driven transitions (
        :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.epoch_transitions_only` is
        ``True``), ensure the specified schedule specifies transitions for every phase.

        Raises:
            MisconfigurationException: If the specified schedule does not include epoch-driven transitions for all
                phases.
        """
        assert isinstance(self.ft_schedule, Dict)
        missing_transitions = [d for d in self.ft_schedule.keys() if self.ft_schedule[d]["max_transition_epoch"] < 0]
        if missing_transitions:
            raise MisconfigurationException(
                f"epoch_transitions_only specified but some phases "
                f"({', '.join(str(d) for d in missing_transitions)}) are missing a "
                "max_transition_epoch. Please unset epoch_transitions_only or "
                "specify a max_transition_epoch for each phase."
            )

    def _parse_phase_lr(self, depth: int) -> None:
        """Parse/Define per-phase base learning rates.

        Args:
            depth (int): Schedule depth/phase to parse
        Raises:
            MisconfigurationException: If the specified per-phase learning rate is not convertable to a float.
        """
        assert isinstance(self.ft_schedule, Dict)
        if depth > 0:
            self.ft_schedule[depth].setdefault("lr", self.base_max_lr)  # type: ignore[attr-defined]
            try:
                float(self.ft_schedule[depth]["lr"])
            except ValueError:
                raise MisconfigurationException(
                    f"The lr '{self.ft_schedule[depth]['lr']}' in phase {depth} of the provided explicit schedule"
                    "could not be cast to a float. Specified learning rates must be convertable to a float."
                )
        else:
            if self.ft_schedule[depth].get("lr", None):
                rank_zero_warn(
                    f"A lr for finetuning phase 0 has been specified ({self.ft_schedule[0]['lr']}). This"
                    " lr will be overridden by the lr specified via the initial optimizer configuration"
                    " (typically in `configure_optimizers()`)."
                )
                del self.ft_schedule[depth]["lr"]

    def _parse_phase(self, depth: int, named_params: KeysView) -> None:
        """Expand any regex expressions specified in an ft_schedule phase to fully qualified parameter names.

        Args:
            depth (int): Schedule depth/phase to parse
            named_params (KeysView): The named parameters of the model

        Raises:
            MisconfigurationException: If a specified parameter or regex does not resolve to at least one parameter.
        """
        assert isinstance(self.ft_schedule, Dict)
        self.ft_schedule[depth].setdefault("max_transition_epoch", -1)
        self._parse_phase_lr(depth)
        orig_params = self.ft_schedule[depth].get("params", [])
        resolved_params = []
        for p in orig_params:
            regex_params = []
            explicit_params = False
            if p in named_params:
                explicit_params = True
                resolved_params.append(p)
            else:
                ppat = re.compile(p)
                regex_params = [n for n in named_params if ppat.match(n)]
                resolved_params.extend(regex_params)
            if not (regex_params or explicit_params):
                raise MisconfigurationException(
                    f"The parameter or regex '{p}' specified in phase {depth} of the "
                    "provided explicit schedule did not match any named parameter in the "
                    "model."
                )
        self.ft_schedule[depth]["params"] = resolved_params

    def _validate_phases_disjoint(self) -> None:
        """Validate that the defined schedule does not specify any parameter in multiple phases.

        Raises:
            MisconfigurationException: Provides a list of the parameters specified in more than one phase.
        """
        assert isinstance(self.ft_schedule, Dict)
        phase_lists = [self.ft_schedule[d]["params"] for d in self.ft_schedule.keys()]
        params = Counter(list(itertools.chain(*phase_lists)))
        unique_params = Counter(list(set().union(*phase_lists)))
        params.subtract(unique_params)
        dup_params = list(params.elements())
        if dup_params:
            raise MisconfigurationException(
                f"Phases are not disjoint. The following parameters are specified in "
                f"multiple phases: {', '.join(dup_params)}"
            )

    def reinit_lr_scheduler(self, new_lr_scheduler: Dict, trainer: pl.Trainer, optimizer: Optimizer) -> None:
        """Reinitialize the learning rate scheduler, using a validated learning rate scheduler configuration and
        wrapping the existing optimizer.

        Args:
            new_lr_scheduler (Dict): A dictionary defining the new lr scheduler configuration to be initialized.
            trainer (pl.Trainer): The :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object.
            optimizer (class:`~torch.optim.Optimizer`): The :class:`~torch.optim.Optimizer` around which the new lr
                scheduler will be wrapped.
        """
        lr_scheduler_init = new_lr_scheduler["lr_scheduler_init"]
        lrs_class = ScheduleParsingMixin._import_lr_scheduler(lr_scheduler_init)
        # unless overridden by user directive, reset optimizer pg lrs to initial before wrapping in new scheduler
        curr_optimizer_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        reset_init_pg_lrs = True if new_lr_scheduler.get("init_pg_lrs", None) else False
        initial_optimizer_lrs = new_lr_scheduler.get("init_pg_lrs", curr_optimizer_lrs)
        for _, data in enumerate(zip(optimizer.param_groups, initial_optimizer_lrs)):
            param_group, lr = data
            param_group["lr"] = lr
            if reset_init_pg_lrs:
                param_group["initial_lr"] = lr
        if "pl_lrs_cfg" in new_lr_scheduler.keys():
            new_lr_scheduler["pl_lrs_cfg"] = self._prune_pl_lrs(new_lr_scheduler["pl_lrs_cfg"])
        new_lrs_config = LRSchedulerConfig(
            opt_idx=0,
            scheduler=lrs_class(optimizer=optimizer, **lr_scheduler_init.get("init_args", {})),
            **new_lr_scheduler.get("pl_lrs_cfg", {}),
        )
        trainer.strategy.lr_scheduler_configs = [new_lrs_config]

    @staticmethod
    def _parse_reint_pg_lrs(depth: int, init_pg_lrs: List) -> None:
        """Parse/Define per-phase base-learning rate overrides for an lr scheduler reinitialization.

        Args:
            depth (int): the current schedule depth being evaluated
            init_pg_lrs (List): the list of new lrs to set as initial for the new lr scheduler.
        Raises:
            MisconfigurationException: If any of the specified per-phase learning rates are not convertable to a float.
        """
        for lr in init_pg_lrs:
            try:
                float(lr)
            except ValueError:
                raise MisconfigurationException(
                    f"Not all of the lrs specified in `init_pg_lrs`: ({init_pg_lrs}) associated with phase {depth} of "
                    "the provided explicit schedule could be cast to a float. Specified learning rates must be "
                    "convertable to a float."
                )

    @staticmethod
    def _is_supported_reinit_lr(lr_class: Type[_LRScheduler]) -> None:
        """Evaulate whether the provided lr scheduler is currently supported in a lr scheduler reinitialization
        context.

        .. note::
        This may be changed from a nominal subtype approach to a protocol/structural subtype design once Python >=
            3.8 is required
        """
        if _TORCH_GREATER_EQUAL_1_10:
            from torch.optim.lr_scheduler import ChainedScheduler, SequentialLR

            unsupported_schedulers = (ChainedScheduler, SequentialLR)
            if issubclass(lr_class, unsupported_schedulers):
                error_msg = (
                    f"The provided lr scheduler type ({lr_class}) is not currently supported in the context of lr "
                    "scheduler reinitialization. The following lr scheduler types are currently unsupported in lr "
                    f"reinitialization configurations: { unsupported_schedulers } "
                )
                rank_zero_warn(error_msg)
                raise MisconfigurationException(error_msg)

    @staticmethod
    def _import_lr_scheduler(lr_scheduler_init: Dict) -> Type[_LRScheduler]:
        """Import the lr scheduler specified in the provided `lr_scheduler_init` configuration.

        Args:
            lr_scheduler_init (Dict): The user-provided lr scheduler reinitialization configuration.

        Raises:
            MisconfigurationException: If the specified LR scheduler cannot be imported successfully.

        Returns:
            Type[_LRScheduler]: The lr scheduler class to be instantiated.
        """
        try:
            class_module, class_name = lr_scheduler_init["class_path"].rsplit(".", 1)
            module = __import__(class_module, fromlist=[class_name])
            lrs_class = getattr(module, class_name)
            if _TORCH_GREATER_EQUAL_1_10:
                ScheduleParsingMixin._is_supported_reinit_lr(lrs_class)
        except (ImportError, AttributeError) as err:
            error_msg = (
                f"Could not import specified LR scheduler class using class_path ({lr_scheduler_init['class_path']}) "
                f"Recieved the following error while importing: {err}. Please validate specified `class_path` before "
                "resubmitting."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        return lrs_class

    @staticmethod
    def _lr_scheduler_sanity_chk(lr_scheduler_init: Dict) -> None:
        """Before beginning execution of defined finetuning schedule, perform a sanity check of the specified lr
        scheduler reinitialization configuration. To the extent reasonable (i.e. without simulating the entire
        training path), if the provided lr scheduler reinitialization configuration is expected to fail, it is
        user-friendly to provide this feedback to the user before training begins.

        Args:
            lr_scheduler_init (Dict): The user-provided lr scheduler reinitialization configuration.

        Raises:
            MisconfigurationException: If a valid and supported scheduler cannot be instantiated with the specified
                init args.
        """
        lrs_class = ScheduleParsingMixin._import_lr_scheduler(lr_scheduler_init)
        if lr_scheduler_init.get("init_args") and "optimizer" in lr_scheduler_init.get("init_args", {}).keys():
            warn_msg = (
                f"Found an `optimizer` key in the provided `lr_scheduler_init`: {lr_scheduler_init['init_args']} "
                f"Note that the existing optimizer and all associated parameter groups will be used when "
                "reinitializing the lr schedule using the specified scheduler so the provided `optimizer` key will "
                "have no effect."
            )
            rank_zero_warn(warn_msg)
            del lr_scheduler_init["init_args"]["optimizer"]
        try:
            testlr = lrs_class(optimizer=_MockOptimizer(), **lr_scheduler_init.get("init_args", {}))
        except Exception as err:
            error_msg = (
                "Could not configure the specified LR scheduler class using the `init_args` "
                f"({lr_scheduler_init['init_args']}). Recieved the following error while sanity checking schedule "
                f"phases: {err}. Please validate specified `init_args` before resubmitting."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        assert isinstance(testlr, torch.optim.lr_scheduler._LRScheduler)


class ScheduleImplMixin(ABC):
    """Functionality for generating and executing finetuning schedules."""

    # proper initialization of these variables should be done in the child class
    pl_module: pl.LightningModule
    ft_schedule: Optional[Union[str, dict]]
    reinit_lr_cfg: Optional[Dict]
    max_depth: int
    _fts_state: FTSState

    @property
    @abstractmethod
    def curr_depth(self) -> int:
        pass

    def init_fts(self) -> None:
        """Initializes the finetuning schedule and prepares the first scheduled level
        1. Generate the default finetuning schedule and/or load it into
        :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.ft_schedule`.
        2. Prepare the first scheduled finetuning level, unfreezing the relevant parameters."""
        self.init_ft_sched()
        assert isinstance(self.ft_schedule, Dict)
        _, self._fts_state._curr_thawed_params = self.exec_ft_phase(
            self.pl_module, thaw_pl=self.ft_schedule[0]["params"], init_thaw=True
        )

    def gen_or_load_sched(self) -> None:
        """Load an explicitly specified finetuning schedule if one provided, otherwise generate a default one."""
        assert self.pl_module.trainer is not None
        if not self.ft_schedule and self.max_depth == -1:
            rank_zero_info("No finetuning schedule provided, max_depth set to -1 so iteratively thawing entire model")
        assert self.pl_module.trainer.log_dir is not None
        if self.ft_schedule and self.reinit_lr_cfg:
            error_msg = (
                "Specifying both `ft_schedule` and `reinit_lr_cfg` is an invalid configuration. `reinit_lr_cfg` "
                "specifies an lr scheduler configuration to reinitialize with at every new phase of an implicitly "
                "defined finetuning shedule whereas `ft_schedule` is an explicity defined schedule. To reinitialize "
                "a given lr scheduler configuration with an explicit finetuning schedule, please add the desired "
                "lr scheduler configurations to your explicit schedule using the `new_lr_scheduler` key of the "
                "relevant phases."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        if self.ft_schedule:  # thaw according to an explicit schedule
            self.ft_schedule = (
                self.load_yaml_schedule(pathlib.Path(self.ft_schedule))
                if not isinstance(self.ft_schedule, Dict)
                else self.ft_schedule
            )
            # save the parsed schedule to our log directory to ensure reproducability
            ScheduleImplMixin.save_schedule(
                f"{self.pl_module.__class__.__name__}_ft_schedule.yaml",
                self.ft_schedule,
                self.pl_module.trainer.log_dir,
            )
        else:
            self.gen_implicit_schedule(self.pl_module.trainer.log_dir)
            self.ft_schedule = self.pl_module.trainer.strategy.broadcast(self.ft_schedule)

    def init_ft_sched(self) -> None:
        """Generate the default finetuning schedule and/or load it into
        :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.ft_schedule`. Broadcast the
        schedule to ensure it is available for use in a distributed context."""
        self.gen_or_load_sched()
        assert isinstance(self.ft_schedule, Dict)
        if self.max_depth == -1:
            self.max_depth = len(self.ft_schedule) - 1
        else:
            self.max_depth = min(self.max_depth, len(self.ft_schedule) - 1)
        max_phase, max_epoch_wm = self._validate_ft_sched()  # type: ignore[attr-defined]
        # if the final phase is not using EarlyStopping, apply the maximum phase-specified epoch to global max_epochs
        if self.ft_schedule[max_phase]["max_transition_epoch"] >= 0:
            assert self.pl_module.trainer is not None
            rank_zero_warn(
                "Final phase max_transition_epoch"
                f" ({self.ft_schedule[max_phase]['max_transition_epoch']})"
                f" will be overidden by the greater of max_epochs ({self.pl_module.trainer.max_epochs}) and"
                f" the maximum phase-specified epoch ({max_epoch_wm})."
            )
            self.pl_module.trainer.fit_loop.max_epochs = max(max_epoch_wm, self.pl_module.trainer.max_epochs)

    @rank_zero_only
    def gen_implicit_schedule(self, sched_dir: os.PathLike) -> None:
        """Generate the default schedule, save it to ``sched_dir`` and load it into
        :attr:`~finetuning_scheduler.fts.FinetuningScheduler.ft_schedule`

        Args:
            sched_dir: directory to which the generated schedule should be written. By default will be
                ``Trainer.log_dir``.
        """
        default_ft_schedule = self.gen_ft_schedule(self.pl_module, sched_dir)
        rank_zero_info(f"Generated default finetuning schedule '{default_ft_schedule}' for iterative finetuning")
        self.ft_schedule = self.load_yaml_schedule(default_ft_schedule)

    @staticmethod
    def save_schedule(schedule_name: str, layer_config: Dict, dump_loc: Union[str, os.PathLike]) -> os.PathLike:
        """Save loaded or generated schedule to a directory to ensure reproducability.

        Args:
            schedule_name (str): The name of the schedule.
            layer_config (Dict): The saved schedule dictionary.
            dump_loc (os.PathLike): The directory to which the generated schedule (.yaml) should be written

        Returns:
            os.PathLike: The path to the generated schedule, by default ``Trainer.log_dir`` and named after the
            :external+pl:class:`~pytorch_lightning.core.lightning.LightningModule` subclass in use with the suffix
            ``_ft_schedule.yaml``)
        """
        dump_path = pathlib.Path(dump_loc)
        dump_path.mkdir(exist_ok=True, parents=True)
        ft_schedule_yaml = dump_path / schedule_name
        fs = get_filesystem(ft_schedule_yaml)
        with fs.open(ft_schedule_yaml, "w", newline="") as fp:
            yaml.dump(layer_config, fp)
        assert os.access(ft_schedule_yaml, os.F_OK)
        rank_zero_info(f"Finetuning schedule dumped to {ft_schedule_yaml}.")
        return ft_schedule_yaml

    @staticmethod
    def gen_ft_schedule(module: Module, dump_loc: Union[str, os.PathLike]) -> os.PathLike:
        """Generate the default finetuning schedule using a naive, 2-parameters per-level heuristic.

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` for which a finetuning schedule will be
                generated
            dump_loc: The directory to which the generated schedule (.yaml) should be written
        Returns:
            os.PathLike: The path to the generated schedule, by default ``Trainer.log_dir`` and named after the
            :external+pl:class:`~pytorch_lightning.core.lightning.LightningModule` subclass in use with the suffix
            ``_ft_schedule.yaml``)
        """
        # Note: This initial default finetuning schedule generation approach is intentionally simple/naive but is
        # effective for a suprising fraction of models. Future versions of this callback may use module introspection to
        # generate default schedules that better accommodate more complex structures and specific architectures if the
        # callback proves sufficiently useful.
        log.info(f"Proceeding with dumping default finetuning schedule for {module.__class__.__name__}")
        param_lists: List = []
        cur_group: List = []
        model_params = list(module.named_parameters())[::-1]
        for i, (n, _) in enumerate(model_params):
            if i % 2 == 0:
                cur_group = []
                cur_group.append(n)
            else:
                cur_group.append(n)
                param_lists.append(cur_group)
        if len(model_params) % 2 == 1:
            param_lists.append([model_params[-1][0]])
        layer_config = {}
        for i, l in enumerate(param_lists):
            layer_config[i] = {"params": l}
        schedule_name = f"{module.__class__.__name__}_ft_schedule.yaml"
        assert dump_loc is not None
        return ScheduleImplMixin.save_schedule(schedule_name, layer_config, dump_loc)

    @staticmethod
    def load_yaml_schedule(schedule_yaml_file: os.PathLike) -> Dict:
        """Load a schedule defined in a .yaml file and transform it into a dictionary.

        Args:
            schedule_yaml_file (str): The .yaml finetuning schedule file

        Raises:
            MisconfigurationException: If the specified schedule file is not found

        Returns:
            Dict: the Dict representation of the finetuning schedule
        """
        try:
            with open(schedule_yaml_file) as df:
                schedule_dict = yaml.load(df, Loader=UniqueKeyLoader)
        except FileNotFoundError as fnf:
            error_msg = (
                f"Could not find specified finetuning scheduling file '{schedule_yaml_file}': {fnf}."
                f"Please reconfigure and try again."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        except ValueError as dup_key:
            error_msg = (
                f"Duplicate key ({dup_key.args[0]}) found in supplied schedule: {schedule_yaml_file}'. Please validate "
                "schedule before resubmitting."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        return schedule_dict

    def thaw_to_depth(self, depth: int = None) -> None:
        """Thaw/unfreeze the current
        :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.pl_module` to the specified
        finetuning depth (aka level)

        Args:
            depth: The depth/level to which the
                :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.pl_module` will be
                thawed. If no depth is is specified,
                :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.curr_depth` will be
                used. Defaults to ``None``.
        """
        # configure optimizer parameter groups for next fts level, adding parameter groups beyond
        # restored optimizer state up to current depth
        depth = depth or self.curr_depth
        for i, next_tl in self.ft_schedule.items():  # type: ignore[union-attr]
            if i <= depth:
                _, self._fts_state._curr_thawed_params = self.exec_ft_phase(self.pl_module, thaw_pl=next_tl["params"])

    @staticmethod
    def add_optimizer_groups(
        module: Module,
        optimizer: Optimizer,
        thawed_pl: List,
        no_decay: Optional[list] = None,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10.0,
    ) -> None:
        """Add optimizer parameter groups associated with the next scheduled finetuning depth/level and extend the
        relevent :paramref:`~pytorch_lighting.trainer.trainer.Trainer.lr_scheduler_configs`.

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` from which the target optimizer parameters
                will be read.
            optimizer (:class:`~torch.optim.Optimizer`): The :class:`~torch.optim.Optimizer` to which parameter groups
                will be configured and added.
            thawed_pl: The list of thawed/unfrozen parameters that should be added to the new parameter group(s)
            no_decay: A list of parameters that should always have weight_decay set to 0. e.g.:
                ["bias", "LayerNorm.weight"]. Defaults to ``None``.
            lr: The initial learning rate for the new parameter group(s). If not specified,
                the ``lr`` of the first scheduled finetuning depth will be used. Defaults to ``None``.
            initial_denom_lr: The scaling factor by which to scale the initial learning rate for new
                parameter groups when no initial learning rate is specified. Defaults to 10.0.
        """
        if len(thawed_pl) == 0:
            rank_zero_warn("No thawed parameters passed so no new optimizer groups will be added.")
        else:
            params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
            denom_lr = initial_denom_lr if lr is None else 1.0
            lr_factor = params_lr / denom_lr
            added_pgs = 0
            if no_decay:
                optimizer.add_param_group(
                    {
                        "params": [
                            p
                            for n, p in module.named_parameters()
                            if not any(nd in n for nd in no_decay) and n in thawed_pl and p.requires_grad
                        ],
                        "lr": lr_factor,
                        "initial_lr": lr_factor,
                    }
                )
                optimizer.add_param_group(
                    {
                        "params": [
                            p
                            for n, p in module.named_parameters()
                            if any(nd in n for nd in no_decay) and n in thawed_pl and p.requires_grad
                        ],
                        "weight_decay": 0.0,
                        "lr": lr_factor,
                        "initial_lr": lr_factor,
                    }
                )
                added_pgs = 2
            else:
                optimizer.add_param_group(
                    {
                        "params": [p for n, p in module.named_parameters() if n in thawed_pl and p.requires_grad],
                        "lr": lr_factor,
                        "initial_lr": lr_factor,
                    }
                )
                added_pgs = 1
            # extend base_lrs for added groups
            for config in module.trainer.lr_scheduler_configs:  # type: ignore[union-attr]
                config.scheduler.base_lrs.extend([lr_factor] * added_pgs)

    @staticmethod
    def sync(objs: Tuple, asets: Tuple, agg_func: Callable = max) -> None:
        """Synchronize sets of object attributes using a given aggregation function.

        Args:
            objs: The target objects to synchronize
            asets: The attribute sets to synchronize
            agg_func: The aggregation function use to synchronize the target object attribute sets. Defaults to max.
        """
        for attrs in asets:
            agg = reduce(agg_func, [reduce(getattr, a.split(sep="."), o) for o, a in zip(objs, attrs)])
            for o, a in zip(objs, attrs):
                setattr(o, a, agg)

    @staticmethod
    def exec_ft_phase(module: Module, thaw_pl: List, init_thaw: bool = False) -> Tuple[List, List]:
        """Thaw/unfreeze the provided list of parameters in the provided :class:`~torch.nn.Module`

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` that will have parameters selectively
                unfrozen/thawed.
            thaw_pl: The list of parameters that should be thawed/unfrozen in the :class:`~torch.nn.Module`
            init_thaw: If ``True``, modifies message to user accordingly. Defaults to ``False``.

        Returns:
            Tuple[List, List]: A Tuple of two lists.
                1. The list of newly thawed/unfrozen parameters thawed by this function
                2. A list of all currently thawed/unfrozen parameters in the target :class:`~torch.nn.Module`
        """
        thawed_p_names = []
        curr_thawed = []
        for n, p in module.named_parameters():
            if not p.requires_grad and n in thaw_pl:
                p.requires_grad = True
                thawed_p_names.append(n)
            elif p.requires_grad:
                curr_thawed.append(n)
        if thawed_p_names:
            rank_zero_debug(
                f"{'Initializing with' if init_thaw else 'Thawed'} the following module parameters: "
                f"{[n for n in thawed_p_names]}"
            )
        curr_thawed.extend(thawed_p_names)
        rank_zero_debug(f"The following module parameters are currently thawed: {[n for n in curr_thawed]}")
        return thawed_p_names, curr_thawed


class CallbackDepMixin(ABC):
    """Functionality for validating/managing callback dependencies."""

    def _inspect_callback_deps(self, trainer: "pl.Trainer") -> List[bool]:
        """Inspect the trainer :paramref:`~pytorch_lighting.trainer.trainer.Trainer.callbacks` for earlystopping
        and scheduled finetuning capabilities.

        Args:
            trainer (pl.Trainer):  The :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object to
                inspect the callbacks of

        Returns:
            Tuple[bool]: The ascertained :paramref:`~pytorch_lighting.trainer.trainer.Trainer.callbacks` capabilities
        """
        callbacks_inspected = [FTSCheckpoint, ModelCheckpoint, FTSEarlyStopping, EarlyStopping, LearningRateMonitor]
        callback_inspection = []
        for ci in callbacks_inspected:
            callback_inspection.append(any([isinstance(c, ci) for c in trainer.callbacks]))
        return callback_inspection

    @staticmethod
    def _reorder_callback_by_type(callbacks: List[Callback], target_callback: type) -> List[Callback]:
        """Moves all ModelCheckpoint callbacks to the end of the list. The sequential order within the group of
        checkpoint callbacks is preserved, as well as the order of all other callbacks.

        Args:
            callbacks: A list of callbacks.

        Return:
            A new list in which the last elements are ModelCheckpoints if there were any present in the
            input.
        """
        target_callbacks = [c for c in callbacks if isinstance(c, target_callback)]
        other_callbacks = [c for c in callbacks if not isinstance(c, target_callback)]
        return other_callbacks + target_callbacks

    def _configure_callback_deps(self, trainer: "pl.Trainer") -> Tuple[List[Callback], bool, bool]:
        """Ensures FTSCheckpoint and :external+pl:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
        callbacks are present and configured, removing any.

        :external+pl:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`s if present.

        Args:
            trainer: The :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object that may have its
                callbacks list altered.

        Returns:
            List[Callback]: A new callback list that includes at least one FTSCheckpoint and EarlyStopping class,
                ensuring the FTSCheckpoint is at the end of list.
            Bool: Whether a :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` callback was added
            Bool: Whether a :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` callback was added
        """
        has_ckpt_fts, has_ckpt_base, has_es_fts, has_es_base, has_lr_monitor = self._inspect_callback_deps(trainer)
        added_ckpt_fts, added_es_fts = False, False
        if not any([has_es_fts, self.epoch_transitions_only, self.gen_ft_sched_only]):  # type: ignore[attr-defined]
            if has_es_base:
                rank_zero_warn(
                    f"{self.__class__.__name__} currently depends upon a finetuning schedule "
                    "capable EarlyStopping callback such as FTSEarlyStopping. Substituting current "
                    "EarlyStopping for FTSCheckpoint"
                )
                trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, EarlyStopping)]
            else:
                rank_zero_warn(
                    f"{self.__class__.__name__} currently depends upon an FTSEarlyStopping callback unless configured "
                    "in epoch_transitions_only mode. Adding an FTSEarlyStopping callback with default configuration."
                )
            trainer.callbacks.append(FTSEarlyStopping(monitor="val_loss"))
            added_es_fts = True
        if (has_es_fts or has_es_base) and self.epoch_transitions_only:  # type: ignore[attr-defined]
            rank_zero_warn(
                "You have specified an EarlyStopping callback along with epoch_transitions_only. Pruning the "
                "extraneous EarlyStopping callback"
            )
            trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, EarlyStopping)]
        if not has_ckpt_fts:
            if has_ckpt_base:
                rank_zero_warn(
                    f"{self.__class__.__name__} currently depends upon a finetuning schedule "
                    "capable ModelCheckpoint callback such as FTSCheckpoint. Substituting current "
                    "ModelCheckpoint for FTSCheckpoint"
                )
                trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, ModelCheckpoint)]
            trainer.callbacks.append(FTSCheckpoint(monitor="val_loss", verbose=True))
            added_ckpt_fts = True
        for uc in [c for c in trainer.callbacks if any([isinstance(c, d) for d in CALLBACK_DEP_PARENTS.values()])]:
            uc.connect_callback(trainer)  # type: ignore[attr-defined]
        if has_lr_monitor:
            trainer.callbacks = CallbackDepMixin._reorder_callback_by_type(trainer.callbacks, LearningRateMonitor)
        # ensure existing callback_connector logic is adhered to. Adding an FTS configuration method to
        # CallbackConnector or forcing users to manually add default EarlyStopping and FTSCheckpoint classes
        # would avoid this callback_connector call
        return trainer._callback_connector._reorder_callbacks(trainer.callbacks), added_es_fts, added_ckpt_fts
