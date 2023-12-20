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
Fine-Tuning Scheduler Supporters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classes composed to support scheduled fine-tuning

"""
import itertools
import logging
import os
import pathlib
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import KeysView
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields
from functools import reduce
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import lightning.pytorch as pl
import torch
import yaml
from lightning.fabric.utilities import rank_zero_info, rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.core.optimizer import _MockOptimizer
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities import find_shared_parameters
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_debug
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import Module

from finetuning_scheduler.strategy_adapters.fsdp import FSDPStrategyAdapter, StrategyAdapter
from finetuning_scheduler.types import FTSLRSchedulerType, FTSLRSchedulerTypeTuple, ParamGroupAddable

log = logging.getLogger(__name__)

CALLBACK_DEP_PARENTS = {"ModelCheckpoint": ModelCheckpoint, "EarlyStopping": EarlyStopping}
CALLBACK_ATTRS = ("ft_schedule", "max_depth")
TARGET_CALLBACK_REF = "FinetuningScheduler"
STRATEGY_ADAPTERS = {"fsdp": FSDPStrategyAdapter}


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
    _ft_init_epoch: Optional[int] = None
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
        support_multiple_targets: bool = False,
    ) -> None:
        """Arguments used to initialize the user-provided callback depedency resolver in accordance with the user-
        provided module configuration:

        Args:
            callback_attrs (Tuple, optional): Attribute signature of user-provided callback to be structurally detected
                and connected. Defaults to CALLBACK_ATTRS defined in the user-provided module.
            callback_parents (Dict, optional): The parent classes of all user-provided callbacks in the module that
                should be connected to the target user-provided callback. Defaults to CALLBACK_DEP_PARENTS in the
                user-provided module.
            target_callback_ref (str, optional): The name of the target user-provided callback to connect to. For each
                subclass of CALLBACK_DEP_PARENTS, an attribute named ``(target_callback_ref.lower())_callback`` will be
                added. Defaults to TARGET_CALLBACK_REF in the user-provided module.
            support_multiple_targets (bool, optional): Whether multiple instances of the target user-provided callback
                (only the first of which will be connected to) are allowed. Defaults to False.
        """
        super().__init__()
        self.callback_attrs = callback_attrs
        self.callback_parents = callback_parents
        self.target_callback_ref = target_callback_ref
        self.callback_handle = f"{self.target_callback_ref.lower()}_callback"
        self.support_multiple_targets = support_multiple_targets
        setattr(self, self.callback_handle, None)

    def connect_callback(self, trainer: "pl.Trainer", reconnect: bool = False) -> None:
        """Connect each user-provided callback dependency that needs to be connected to the target user-provided
        callback.

        Args:
            trainer (pl.Trainer): The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object.
            reconnect (bool, optional): Whether to check for an updated target callback object even if one is already
                resolved. Predominantly useful in the context of testing. Defaults to False.

        Raises:
            MisconfigurationException: If no target callback is detected
            MisconfigurationException: if :attr:`support_multiple_targets` is ``False`` and multiple target callbacks
                are detected.
        """
        if self.__dict__[self.callback_handle] and not reconnect:
            return
        resolved_target_callbacks = [c for c in trainer.callbacks if all([hasattr(c, a) for a in self.callback_attrs])]
        if not resolved_target_callbacks:
            raise MisconfigurationException(
                f"{self.__class__.__name__} is intended for use with a {self.target_callback_ref}. If not using a"
                f"{self.target_callback_ref} callback, please use the standard "
                f"{[k for k,v in self.callback_parents.items() if isinstance(self,v)][0]} callback."
            )
        elif not self.support_multiple_targets and len(resolved_target_callbacks) > 1:
            raise MisconfigurationException(
                f"Use of multiple {resolved_target_callbacks[0].__class__.__name__} callbacks is"
                "not currently supported. Please provide a maximum of one."
            )
        else:
            setattr(self, self.callback_handle, resolved_target_callbacks[0])


class FTSEarlyStopping(EarlyStopping, CallbackResolverMixin):
    r"""
    Extends/specializes :external+pl:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` to facilitate
    multi-phase scheduled fine-tuning.

    Adds :attr:`es_phase_complete`, :attr:`final_phase` and :attr:`finetuningscheduler_callback` attributes and modifies
    ``EarlyStopping._evaluate_stopping_criteria`` to enable multi-phase behavior. Usage of
    :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` is identical to
    :external+pl:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` except the former will evaluate the
    specified early stopping criteria at every scheduled phase.
    :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` will automatically be
    used if a :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback is detected
    and :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.epoch_transitions_only` is ``False``

    .. note::

       For detailed usage information,
       see :external+pl:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`.

    .. note::

       Currently, :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports the use of one
       :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` callback instance at a time.

    """
    _check_on_train_epoch_end: Optional[bool]
    best_score: Tensor
    wait_count: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Attributes:
            es_phase_complete (bool):
                Used to determine if the current phase's early stopping criteria have been met.
            final_phase (bool):
                Used to indicate whether the current phase is the final scheduled phase.
            finetuningscheduler_callback (lightning.pytorch.callbacks.Callback):
                Reference to the :class:`~finetuning_scheduler.fts.FinetuningScheduler`
                callback being used.
            reduce_transition_decisions (bool):
                Used to indicate whether the callback is operating in a distributed context without the monitored metric
                being synchronized (via ``sync_dist`` being set to ``True`` when logging).
            check_on_train_epoch_end (bool): Whether to run early stopping check at the end of the training epoch. If
                this is ``False``, then the check runs at the end of the validation. Defaults to ``None`` similar to
                :external+pl:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` but is set to
                ``False`` during setup unless overridden.
        """
        super().__init__(*args, **kwargs)
        self.es_phase_complete = True
        self.final_phase = True
        self.finetuningscheduler_callback = None
        self.reduce_transition_decisions = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Ensure a :class:`~finetuning_scheduler.fts.FinetuningScheduler` is provided before beginning
        training."""
        self.connect_callback(trainer)
        if self._check_on_train_epoch_end is None:
            # post-validation saving/evaluation is the most common fts usage pattern
            self._check_on_train_epoch_end = False
        super().setup(trainer, pl_module, stage)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Ascertain whether the execution context of this callback requires that we reduce transition decisions
        over all distributed training processes.

        Args:
            trainer: The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object
            pl_module  (:external+pl:class:`~lightning.pytorch.core.module.LightningModule`): The
                :external+pl:class:`~lightning.pytorch.core.module.LightningModule` object
        """
        if trainer.state.fn == TrainerFn.FITTING:
            self.reduce_transition_decisions = self._check_sync_dist(trainer)
        super().on_validation_end(trainer, pl_module)

    def _check_sync_dist(self, trainer: "pl.Trainer") -> bool:
        """Inspect the monitored metric and execution context to determine whether transition decisions for this
        callback need to be reduced over all distributed training processes.

        Args:
            trainer: The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object

        Returns:
            bool: Whether to reduce transition decisions for this callback over all training processes
        """
        assert self.finetuningscheduler_callback is not None
        monitor_metric = [
            m
            for m in self.finetuningscheduler_callback.pl_module.trainer._results.result_metrics
            if m.meta.name == self.monitor
        ]
        assert monitor_metric[0] is not None
        no_sync = (torch.distributed.is_available() and monitor_metric[0].is_tensor) and not monitor_metric[
            0
        ].meta.sync.should
        return no_sync

    def _transition_es_phase(self) -> None:
        """Encapsulates updating the :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` internal state
        while transitioning to the next scheduled fine-tuning phase."""
        assert self.finetuningscheduler_callback is not None
        self.es_phase_complete = True
        self.wait_count = 0
        rank_zero_debug(
            "Preparing the FTSEarlyStopping callback for transition to the next scheduled fine-tuning phase (phase"
            f" {self.finetuningscheduler_callback.curr_depth + 1})"
        )

    def _reset_es_phase(self) -> None:
        """Encapsulates resetting of :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` internal state
        for the next scheduled fine-tuning phase."""
        assert self.finetuningscheduler_callback is not None
        self.es_phase_complete = False
        self.wait_count = 0
        rank_zero_debug(
            "Reset the FTSEarlyStopping callback for the next scheduled fine-tuning phase (phase"
            f" {self.finetuningscheduler_callback.curr_depth})"
        )

    def _evaluate_stopping_criteria(self, current: Tensor) -> Tuple[bool, Optional[str]]:
        """Evaluate whether and why to stop the current training session.

        Args:
            current (Tensor): The current monitored value to be evaluated

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
                    self._transition_es_phase()
        return should_stop, reason


class FTSCheckpoint(ModelCheckpoint, CallbackResolverMixin):
    r"""
    Extends/specializes :external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` to facilitate
    multi-phase scheduled fine-tuning. Overrides the
    ``state_dict`` and ``load_state_dict`` hooks to maintain additional state (:attr:`current_ckpt_depth`,
    :attr:`best_ckpt_depth`, :attr:`finetuningscheduler_callback`). Usage of
    :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` is identical to
    :external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` and
    :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` will automatically be used if a
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback is detected.

    .. note::
        For detailed usage information, see
        :external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`.

    .. note::

       Currently, :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports the use of one
       :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` callback instance at a time.
    """
    _save_on_train_epoch_end: Optional[bool]
    best_model_path: str
    kth_best_model_path: str
    last_model_path: str
    best_k_models: Dict
    kth_value: Tensor

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Attributes:
            current_ckpt_depth (int):
                Used to track the depth of most recently saved checkpoint
            best_ckpt_depth (int):
                Used to track the depth of the best known checkpoint (it may be from a different training depth)
            finetuningscheduler_callback (lightning.pytorch.callbacks.Callback):
                Reference to the :class:`~finetuning_scheduler.fts.FinetuningScheduler`
                callback being used.
            save_on_train_epoch_end (Optional[bool]): Whether to run checkpointing at the end of the training epoch.
                If this is ``False``, then the check runs at the end of the validation. Defaults to ``None`` similar to
                :external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` but is set to
                ``False`` during setup unless overridden.
        """
        super().__init__(*args, **kwargs)
        self.current_ckpt_depth = 0
        self.best_ckpt_depth = 0
        self.finetuningscheduler_callback = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Verify a valid callback configuration is present before beginning training.

        Args:
            trainer: The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object

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
        """Overrides. :external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`'s
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
        """Overrides :external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`'s
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
            if self.dirpath == dirpath_from_ckpt or self.dirpath == pathlib.Path(dirpath_from_ckpt):
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
    """Functionality for parsing and validating fine-tuning schedules."""

    SANITY_CHK_ITERABLE = (torch.nn.Parameter(torch.empty(1)),)
    VALID_REINIT_ATTR = ("reinit_lr_cfg", "reinit_optim_cfg")
    VALID_REINIT_KEYS = ("new_lr_scheduler", "new_optimizer")
    # proper initialization of these variables should be done in the child class
    pl_module: pl.LightningModule
    ft_schedule: Optional[Union[str, dict]]
    reinit_optim_cfg: Optional[Dict]
    reinit_lr_cfg: Optional[Dict]

    def _validate_ft_sched(self) -> Tuple[int, int]:
        """Ensure the explicitly specified fine-tuning schedule has a valid configuration.

        Returns:
            Tuple[int, int]: A tuple of ints specifying:
                1. The depth of the final scheduled phase
                2. The maximum epoch watermark explicitly specified in the schedule
        """
        max_epoch_wm = -1
        max_phase = 0
        self._validate_schedule_keys()
        self._validate_reinit_cfg()
        named_params = dict(self.pl_module.named_parameters()).keys()
        model_shared_params = find_shared_parameters(self.pl_module)
        msp_ref = tuple((model_shared_params, set(itertools.chain(*model_shared_params))))
        for depth in self.ft_schedule.keys():  # type: ignore[union-attr]
            max_phase = max(max_phase, depth)
            self._parse_phase(depth, named_params, msp_ref)
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

    def _update_pl_lrs(self, pl_lrs_cfg: Dict, lrs_class: FTSLRSchedulerType) -> Dict:
        """Prune keys not part of a valid PyTorch Lightning lr scheduler configuration (if automatic optimization
        used) and update configuration if :external+torch:class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` is
        used.

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
            pl_lrs_cfg["reduce_on_plateau"] = pl_lrs_cfg.get(
                "reduce_on_plateau", issubclass(lrs_class, torch.optim.lr_scheduler.ReduceLROnPlateau)
            )
            if pl_lrs_cfg["reduce_on_plateau"] and pl_lrs_cfg.get("monitor", None) is None:
                raise MisconfigurationException(
                    "The lr scheduler dict must include a monitor when a `ReduceLROnPlateau` scheduler is used."
                    ' For example: {"optimizer": optimizer, "lr_scheduler":'
                    ' {"scheduler": scheduler, "monitor": "your_loss"}}'
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

    def _common_reinit_key_validation(self, target_sched: Dict, target_key: str, depth: Optional[int] = None) -> None:
        """Key validation common to all reinitialzation configuration dictionaries.

        Args:
            target_sched (Dict): The provided reinitialization configuration for either an implicit mode fine-tuning
                schedule or for a given explicity mode fine-tuning phase.
            target_key (str): The expected reinitialization key for the current parsing context.
            depth (Optional[int], optional): If parsing an explicit schedule, the current phase. Defaults to None.

        Raises:
            MisconfigurationException: If a valid reinitialization key is missing in the reinitialization configuration.
            MisconfigurationException: If the configuration provided in valid reinitialization key but did not specify
                a `class_path` for the class to be instantiated.
        """
        if target_key not in target_sched.keys():
            phase_specific_msg = "" if not depth else f"for phase {depth}"
            key_specific_msg = "a lr scheduler" if target_key == "lr_scheduler_init" else "an optimizer"
            raise MisconfigurationException(
                f"Specifying {key_specific_msg} configuration to reinitialize with requires a valid configuration "
                f"dictionary be provided via a `{target_key}` key but no such key was found " + phase_specific_msg + "."
            )
        if not target_sched[target_key].get("class_path"):
            phase_specific_msg = "the specified config." if not depth else f"the specified phase ({depth})."
            raise MisconfigurationException(
                f"Specifying `{target_key}` requires at least a  `class_path` to be specified but this is not the case "
                "for " + phase_specific_msg
            )

    def _optimizer_reinit_key_validation(self, target_sched: Dict, depth: Optional[int] = None) -> None:
        """Validate the keys in a given lr scheduler reinitialization configuration.

        Args:
            target_sched (Dict): The provided optimizer reinitialization configuration for either an implicit mode
                fine-tuning schedule (passed via `reinit_optim_cfg`) or for a given explicity mode fine-tuning phase
                (passed via `new_optimizer` for a given phase)
            depth (Optional[int], optional): If parsing an explicit schedule, the current phase. Defaults to None.
        """
        self._common_reinit_key_validation(target_sched, "optimizer_init", depth)
        optimizer_init = target_sched.get("optimizer_init")
        assert optimizer_init
        self._optimizer_sanity_chk(optimizer_init)

    def _lr_scheduler_reinit_key_validation(self, target_sched: Dict, depth: Optional[int] = None) -> None:
        """Validate the keys in a given lr scheduler reinitialization configuration.

        Args:
            target_sched (Dict): The provided lr scheduler reinitialization configuration for either an implicit mode
                fine-tuning schedule (passed via `reinit_lr_cfg`) or for a given explicity mode fine-tuning phase
                (passed via `new_lr_scheduler` for a given phase)
            depth (Optional[int], optional): If parsing an explicit schedule, the current phase. Defaults to None.

        Raises:
            MisconfigurationException: If an `init_pg_lrs` key is provided in implicit mode training
                (via `reinit_lr_cfg`).
        """
        self._common_reinit_key_validation(target_sched, "lr_scheduler_init", depth)
        implicit_chk = bool(self.reinit_lr_cfg)
        if "init_pg_lrs" in target_sched.keys() and implicit_chk:
            raise MisconfigurationException(
                "Specifying a `init_pg_lrs` key in the lr scheduler configuration passed via `reinit_lr_cfg` (i.e. "
                "implicit mode training) is not a valid configuration since the same lr scheduler configuration "
                "is intended to be reinitialized at every fine-tuning phase with implicit mode fine-tuning."
            )
        # if we're passing pl lr scheduler configuration, validate the keys
        if "pl_lrs_cfg" in target_sched.keys():
            self._pl_lrs_validation(pl_lrs_cfg=target_sched["pl_lrs_cfg"])
        if (
            "use_current_optimizer_pg_lrs" in target_sched.keys()
            and target_sched["use_current_optimizer_pg_lrs"]
            and "init_pg_lrs" not in target_sched.keys()
        ):
            info_msg = (
                "Since `use_current_optimizer_pg_lrs` has been set to `True`, lr scheduler reinitializations "
                f"associated with phase {depth} will use the current optimizer `lr`s rather than defaulting "
                "to the existing optimizer's `initial_lr` configuration for existing parameter groups."
            )
            rank_zero_info(info_msg)
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
        self._lr_scheduler_sanity_chk(lr_scheduler_init, implicit_chk)

    def _reinit_validation(self, reinit_cfg: Dict) -> None:
        """Trigger reinitialization configuration validation for all provided configurations. This will be a single
        configuration for implicit mode fine-tuning or n configurations for explicit mode.

        Args:
            reinit_cfg (Dict): An lr scheduler and/or optimizer reinitialization configuration to parse/validate
        """
        reinit_validation_funcs = (self._lr_scheduler_reinit_key_validation, self._optimizer_reinit_key_validation)
        for (rk, rp), rattr, rfunc in zip(
            reinit_cfg.items(), ScheduleParsingMixin.VALID_REINIT_ATTR, reinit_validation_funcs
        ):
            if getattr(self, rattr):
                rfunc(reinit_cfg[rk])
            else:
                for k, r_cfg in rp.items():
                    rfunc(r_cfg, k)

    def _validate_reinit_cfg(self) -> None:
        """Orchestrate optimizer and lr scheduler reinitialization configuration validation.

        Raises:
            MisconfigurationException: If a `new_optimizer` or `new_lr_scheduler` configuration is passed to the initial
                training phase.
        """
        assert isinstance(self.ft_schedule, Dict)
        reinit_cfg = {}
        for reinit_k, attr in zip(ScheduleParsingMixin.VALID_REINIT_KEYS, ScheduleParsingMixin.VALID_REINIT_ATTR):
            reinit_cfg[reinit_k] = getattr(self, attr) or {
                k: self.ft_schedule[k].get(reinit_k)
                for k in self.ft_schedule.keys()
                if self.ft_schedule[k].get(reinit_k)
            }
        if not any(reinit_cfg.values()):
            return  # no further validation needed since there is no reinitialization configuration
        self._has_reinit_schedule = True  # schedules that reinitialize require special handling in some contexts
        assert self.pl_module.trainer is not None
        assert self.pl_module.trainer.log_dir is not None
        for rk, rp in reinit_cfg.items():
            if isinstance(rp, dict) and 0 in rp.keys():
                raise MisconfigurationException(
                    f"You have specified a `{rk}` reinitialization directive for the initial training phase which is an"
                    "invalid configuration. The initial configuration should be passed to your LightningModule."
                )
        self._reinit_validation(reinit_cfg)

    def _convert_phase_keys(self) -> None:
        """Ensures phase keys are integers, converting them to integers if possible and raising an error otherwise.

        Raises:
            MisconfigurationException: If the phase keys provided in the schedule are not convertible to integers.
        """
        assert isinstance(self.ft_schedule, Dict)
        try:
            orig_keys = set(self.ft_schedule.keys())
            self.ft_schedule = {int(k): v for k, v in self.ft_schedule.items()}
            key_diff = set(self.ft_schedule.keys()) ^ orig_keys
            if key_diff:
                rank_zero_warn(
                    "Note, the specified fine-tuning schedule had non-integer keys implicitly converted to "
                    f"integers. Key diff: {key_diff}"
                )
                self._rewrite_schedule()
        except ValueError as value_err:
            raise MisconfigurationException(
                "The supplied schedule was found to use one or more keys that were not convertible to integers. "
                f"The encountered error was: {value_err}"
            )

    def _rewrite_schedule(self, err_msg: Optional[str] = None) -> None:
        """Saves a reconfigured schedule to ``Trainer.log_dir`` and optionally raises an error message if
        specified.

        Args:
            err_msg (Optional[str], optional): The error message that should be raised after saving the transformed
            schedule. Defaults to None.

        Raises:
            MisconfigurationException: Provides the specified error message if the caller specifies one. e.g. if the
                schedule contains (non-convertible) non-integer keys and/or non-zero-based and contiguous keys.
        """
        assert self.pl_module.trainer is not None and self.pl_module.trainer.log_dir is not None
        assert isinstance(self.ft_schedule, Dict)
        rewrite_dest = None
        # write the reconfigured schedule to our log directory to allow user validation
        rewrite_dest = ScheduleImplMixin.save_schedule(
            f"{self.pl_module.__class__.__name__}_ft_schedule_valid.yaml",
            self.ft_schedule,
            self.pl_module.trainer.log_dir,
        )
        if err_msg:
            raise MisconfigurationException(
                err_msg + f"and has thus been reconfigured and saved to '{rewrite_dest}'. Please validate the "
                "reconfigured schedule and restart training with a valid schedule."
            )

    def _validate_schedule_keys(self) -> None:
        """Ensures schedule keys are integers, zero-based and contiguous.

        If the schedule does not meet these requirements, attempts to transform the passed schedule to meet them and
        writes the candidate schedule out for subsequent user validation.
        """
        assert self.pl_module.trainer is not None and self.pl_module.trainer.log_dir is not None
        assert isinstance(self.ft_schedule, Dict)
        self._convert_phase_keys()
        contiguous = len(self.ft_schedule.keys()) == (max(self.ft_schedule.keys()) + 1)
        if not contiguous:
            for i, k in enumerate(sorted(self.ft_schedule.keys())):
                self.ft_schedule[i] = self.ft_schedule.pop(k)
            err_msg = "The supplied schedule was found to have non-contiguous or non-zero-indexed keys "
            self._rewrite_schedule(err_msg=err_msg)

    def _validate_epoch_transitions(self) -> None:
        """If not composing :external+pl:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` and
        epoch-driven stopping criteria (the default behavior) but instead specifying exclusively epoch-driven
        transitions ( :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.epoch_transitions_only` is
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
                    f"A lr for fine-tuning phase 0 has been specified ({self.ft_schedule[0]['lr']}). This"
                    " lr will be overridden by the lr specified via the initial optimizer configuration"
                    " (typically in `configure_optimizers()`)."
                )
                del self.ft_schedule[depth]["lr"]

    def _parse_phase(self, depth: int, named_params: KeysView, shared_params: Tuple) -> None:
        """Expand any regex expressions specified in an ft_schedule phase to fully qualified parameter names. If
        any shared parameter copies are explicitly specified in the schedule, the copies will be pruned from the
        schedule with a warning.

        Args:
            depth (int): Schedule depth/phase to parse
            named_params (KeysView): The named parameters of the model
            shared_params (Tuple): A tuple containing the shared parameter names of the current model in both
                associative list and set forms.

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
                if p in shared_params[1]:
                    pruning_param_msg = (
                        f"Pruning explicitly specified shared parameter {p} from provided schedule (it will be thawed "
                        f" when its registered source parameter {[pl[0] for pl in shared_params[0] if p in pl][0]} is"
                        " thawed."
                    )
                    rank_zero_warn(pruning_param_msg)
                else:
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

    def _reinit_phase0_pgs(self, thawed_pl: List) -> List:
        """Reconstruct the parameter groups associated with phase 0 of the schedule.

        Args:
            thawed_pl (List): A list of parameter names from which to construct the initial parameter groups.

        Returns:
            List: A list of one or two new parameter groups (contingent on the module's use of ``no_decay``)
        """
        no_decay = getattr(self.pl_module, "no_decay", None)
        if no_decay:
            pgs = [
                {
                    "params": [
                        p
                        for n, p in self.pl_module.named_parameters()
                        if not any(nd in n for nd in no_decay) and n in thawed_pl and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in self.pl_module.named_parameters()
                        if any(nd in n for nd in no_decay) and n in thawed_pl and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            pgs = [{"params": [p for n, p in self.pl_module.named_parameters() if n in thawed_pl and p.requires_grad]}]
        return pgs

    def _save_pre_reinit_lr_state(self, trainer: pl.Trainer) -> Tuple[Dict, List]:
        """Capture the existing lr state for all parameter groups associated with previous depths to enable
        restoration during the next phase transition.

        Args:
            trainer (pl.Trainer): The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object.

        Returns:
            Tuple[Dict, List]: The lr state to restore from the current lr scheduler and the most recent `lr`s for
                parameter groups associated with the current phases's optimizer.
        """
        curr_lr_state: Dict = {}
        if trainer.lr_scheduler_configs:
            curr_lr_state = deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict())
        prev_optimizer_lrs = copy([group["lr"] for group in trainer.strategy.optimizers[0].param_groups])
        return curr_lr_state, prev_optimizer_lrs

    def reinit_optimizer(self, new_optimizer: Dict, trainer: pl.Trainer, init_params: List) -> ParamGroupAddable:
        """Reinitialize the optimizer, using a validated optimizer reinitialization configuration.

        Args:
            new_optimizer (Dict): A dictionary defining the new optimizer configuration to be initialized.
            trainer (pl.Trainer): The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object.
            init_params (List): The list of parameter names with which to initialize the new optimizer.

        Returns:
            ParamGroupAddable: A handle for the newly reinitialized optimizer.
        """
        optimizer_init = new_optimizer["optimizer_init"]
        prev_optim_repr = repr(trainer.strategy.optimizers[0])
        optimizer_class = self._import_reinit_class(optimizer_init, reinit_target="optimizer")
        reinit_pgs = self._reinit_phase0_pgs(thawed_pl=init_params)
        new_optimizer_handle = optimizer_class(
            reinit_pgs, **optimizer_init.get("init_args", {})  # type: ignore[operator, arg-type]
        )
        # If the user or optimizer doesn't set `initial_lr` keys, add them based on the initial lr values.
        # The latest LR state will still be set in subsequent phases, but this allows subsequent lr scheduler
        # reinitializations to access an `initial_lr` for the existing optimizer if desired (important for consistency
        # with lr scheduler-only reinitializations).
        for group in new_optimizer_handle.param_groups:  # type: ignore[union-attr]
            group["initial_lr"] = group.get("initial_lr", group["lr"])
        trainer.strategy.optimizers = [new_optimizer_handle]  # type: ignore[list-item]
        if trainer.lr_scheduler_configs:
            trainer.lr_scheduler_configs[0].scheduler.optimizer = new_optimizer_handle
        self._maybe_trace_reinit("optimizer", prev_optim_repr, repr(trainer.strategy.optimizers[0]))
        return new_optimizer_handle  # type:ignore[return-value]

    def reinit_lr_scheduler(self, new_lr_scheduler: Dict, trainer: pl.Trainer, optimizer: ParamGroupAddable) -> None:
        """Reinitialize the learning rate scheduler, using a validated learning rate scheduler configuration and
        wrapping the existing optimizer.

        Args:
            new_lr_scheduler (Dict): A dictionary defining the new lr scheduler configuration to be initialized.
            trainer (pl.Trainer): The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object.
            optimizer (:class:`~finetuning_scheduler.types.ParamGroupAddable`): A supported optimizer instance around
                which the new lr scheduler will be wrapped.
        """
        ################################################################################################################
        # The following precedence governs the configuration of existing parameter group `lr`s when reinitializing an LR
        # scheduler:
        #   1. User-provided `lr`s from the `init_pg_lrs` directive if it exists
        #   2. Existing optimizer `lr`s if ``use_current_optimizer_pg_lrs`` is set to ``True``
        #   3. The ``initial_lr`` of the current optimizer parameter groups by default
        #   4. The existing optimizer `lr`s if ``use_current_optimizer_pg_lrs`` is not set to ``True`` but the relevant
        #      parameter group does not have an ``initial_lr`` key
        ################################################################################################################
        lr_scheduler_init = new_lr_scheduler["lr_scheduler_init"]
        prev_lrs_repr = repr(trainer.lr_scheduler_configs[0])
        lrs_class = self._import_reinit_class(lr_scheduler_init, reinit_target="lr_scheduler")
        existing_lr_key = "initial_lr" if not new_lr_scheduler.get("use_current_optimizer_pg_lrs", None) else "lr"
        curr_optimizer_lrs = [group.get(existing_lr_key, group["lr"]) for group in optimizer.param_groups]
        reset_init_pg_lrs = True if new_lr_scheduler.get("init_pg_lrs", None) else False
        initial_optimizer_lrs = new_lr_scheduler.get("init_pg_lrs", curr_optimizer_lrs)
        for _, data in enumerate(zip(optimizer.param_groups, initial_optimizer_lrs)):
            param_group, lr = data
            param_group["lr"] = lr
            if reset_init_pg_lrs:
                param_group["initial_lr"] = lr
        if "pl_lrs_cfg" in new_lr_scheduler.keys():
            new_lr_scheduler["pl_lrs_cfg"] = self._update_pl_lrs(
                new_lr_scheduler["pl_lrs_cfg"], lrs_class=lrs_class  # type:ignore[arg-type]
            )
        assert callable(lrs_class)
        new_lrs_config = LRSchedulerConfig(
            scheduler=lrs_class(
                optimizer=optimizer, **lr_scheduler_init.get("init_args", {})  # type: ignore[arg-type]
            ),
            **new_lr_scheduler.get("pl_lrs_cfg", {}),
        )
        trainer.strategy.lr_scheduler_configs = [new_lrs_config]
        self._maybe_trace_reinit("lr scheduler", prev_lrs_repr, repr(trainer.lr_scheduler_configs[0]))

    def _maybe_trace_reinit(self, target_type: str, prev_repr: str, new_repr: str) -> None:
        """Trace valid optimizer and lr scheduler transitions (including intermediate restorations).

        Args:
            target_type (str): The type of object being reinitialized.
            prev_repr (str): A representation of the state of the target object before reinitialization.
            new_repr (str): A representation of the state of the target object after reinitialization.
        """
        reinit_msg = f"Fine-Tuning Scheduler has reinitialized the {target_type} as directed:{os.linesep}"
        rank_zero_debug(
            reinit_msg + f"Previous {target_type} state:`{prev_repr}`{os.linesep}"
            f"New {target_type} state: `{new_repr}`{os.linesep}"
        )

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

    def _is_supported_lr(self, lr_class: FTSLRSchedulerType) -> None:
        """Evaulate whether the provided lr scheduler is currently supported.

        Args:
            lr_class (FTSLRSchedulerType): The lr scheduler class to be inspected for support.

        Raises:
            MisconfigurationException: If :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.allow_untested` is
                ``False`` and the provided lr scheduler class is not a subclass allowed by ``FTSLRSchedulerTypeTuple``.
        """
        if not issubclass(lr_class, FTSLRSchedulerTypeTuple):
            if not self.allow_untested:  # type: ignore[attr-defined]
                error_msg = (
                    f"The provided lr scheduler ({lr_class}) is not currently supported by"
                    " FinetuningScheduler. Please use a currently supported torch scheduler (or subclass thereof)"
                    f" ({([i.__name__ for i in FTSLRSchedulerTypeTuple])}) or if you would like to attempt to use the"
                    " currently specified scheduler, pass ``allow_untested=True`` to the FinetuningScheduler callback"
                    " when adding it."
                )
                rank_zero_warn(error_msg)
                raise MisconfigurationException(error_msg)
            else:
                warn_msg = (
                    "Allowing untested scheduler"
                    f" '{type(lr_class)}' because ``allow_untested`` is ``True``."  # type: ignore[attr-defined]
                )
                rank_zero_warn(warn_msg)

    def _is_supported_reinit_optimizer(self, optim_class: Union[Any, ParamGroupAddable]) -> None:
        """Evaulate whether the provided optimizer is currently supported in the context of optimizer
        reinitialization.

        Args:
            optim_class (ParamGroupAddable): The optimizer class to be inspected for support.

        Raises:
            MisconfigurationException: If the provided optimizer class is known to be currently unsupported in the
                context of optimizer reinitialization.
        """
        if issubclass(optim_class, ZeroRedundancyOptimizer):  # type: ignore[arg-type]
            error_msg = (
                f"The provided optimizer ({optim_class}) is not currently supported by FinetuningScheduler in the"
                " context of optimizer reinitialization. Please use a currently supported torch optimizer (or subclass"
                " thereof) from those provided in `torch.optim`: https://pytorch.org/docs/stable/optim.html#algorithms."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)

    def _import_reinit_class(
        self, reinit_cfg: Dict, reinit_target: str
    ) -> Union[FTSLRSchedulerType, ParamGroupAddable]:
        """Import the reinitialization class (lr scheduler or optimizer) specified in the provided reinitialization
        configuration.

        Args:
            reinit_cfg (Dict): The user-provided reinitialization configuration.
            reinit_target (str): The reinitialization target, currently "optimizer" or "lr_scheduler".

        Raises:
            MisconfigurationException: If the specified class cannot be imported successfully.

        Returns:
            Union[FTSLRSchedulerType, ParamGroupAddable]: The class to reinitialize.
        """
        # TODO: refactor this function to enable type narrowing while continuing to share relevant code paths
        try:
            class_module, class_name = reinit_cfg["class_path"].rsplit(".", 1)
            module = __import__(class_module, fromlist=[class_name])
            reinit_class = getattr(module, class_name)
            if reinit_target == "lr_scheduler":
                self._is_supported_lr(reinit_class)
        except (ImportError, AttributeError) as err:
            error_msg = (
                "Could not import specified reinitialization configuration class using class_path "
                f"({reinit_cfg['class_path']}). Recieved the following error while importing: {err}. Please validate "
                "specified `class_path` before resubmitting."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        return reinit_class

    @staticmethod
    def _import_strategy_adapter(strategy_key: str, adapter_map: Dict[str, str]) -> Type[StrategyAdapter]:
        """Import the custom strategy adapter specified in the ``custom_strategy_adapter`` configuration.

        Args:
            qualname (Dict): The user-provided custom strategy adapter fully qualified class name.

        Raises:
            MisconfigurationException: If the specified custom strategy adapter cannot be imported successfully.
            MisconfigurationException: If the specified `strategy_key` does not match the current strategy.

        Returns:
            StrategyAdapter: The custom strategy adapter class to be instantiated.
        """
        try:
            qualname = adapter_map.get(strategy_key, None)
            if not qualname:
                raise MisconfigurationException(
                    f"Current strategy name ({strategy_key}) does not map to a custom strategy adapter in the"
                    f" provided `custom_strategy_adapter` mapping ({adapter_map})."
                )
            class_module, class_name = qualname.rsplit(".", 1)
            module = __import__(class_module, fromlist=[class_name])
            custom_strategy_adapter_cls = getattr(module, class_name)
            issubclass(custom_strategy_adapter_cls, StrategyAdapter)
        except (ImportError, AttributeError) as err:
            error_msg = (
                "Could not import the specified custom strategy adapter class using the provided fully qualified class"
                f" name ({qualname}). Recieved the following error while importing: {err}. Please validate specified"
                " path."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        return custom_strategy_adapter_cls

    def _optimizer_sanity_chk(self, optimizer_init: Dict) -> None:
        """Before beginning execution of defined fine-tuning schedule, perform a sanity check of the specified
        optimizer reinitialization configuration. To the extent reasonable (i.e. without simulating the entire
        training path), if the provided optimizer reinitialization configuration is expected to fail, it is user-
        friendly to provide this feedback to the user before training begins.

        Args:
            optimizer_init (Dict): The user-provided optimizer reinitialization configuration.

        Raises:
            MisconfigurationException: If a valid and supported scheduler cannot be instantiated with the specified
                init args.
        """
        optimizer_class = self._import_reinit_class(optimizer_init, reinit_target="optimizer")
        self._is_supported_reinit_optimizer(optimizer_class)
        test_optimizer_init = copy(optimizer_init.get("init_args", {}))
        try:
            assert callable(optimizer_class)
            test_optimizer = optimizer_class(
                ScheduleParsingMixin.SANITY_CHK_ITERABLE, **test_optimizer_init  # type: ignore[arg-type]
            )
        except Exception as err:
            error_msg = (
                "Could not configure the specified optimizer class using the `init_args` "
                f"({optimizer_init['init_args']}). Recieved the following error while sanity checking schedule "
                f"phases: {err}. Please validate specified `init_args` before resubmitting."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        assert isinstance(test_optimizer, ParamGroupAddable)

    def _lr_scheduler_sanity_chk(self, lr_scheduler_init: Dict, is_implicit_mode: bool = False) -> None:
        """Before beginning execution of defined fine-tuning schedule, perform a sanity check of the specified lr
        scheduler reinitialization configuration. To the extent reasonable (i.e. without simulating the entire
        training path), if the provided lr scheduler reinitialization configuration is expected to fail, it is
        user-friendly to provide this feedback to the user before training begins.

        Args:
            lr_scheduler_init (Dict): The user-provided lr scheduler reinitialization configuration.

        Raises:
            MisconfigurationException: If a valid and supported scheduler cannot be instantiated with the specified
                init args.
        """
        lrs_class = self._import_reinit_class(lr_scheduler_init, reinit_target="lr_scheduler")
        if lr_scheduler_init.get("init_args") and "optimizer" in lr_scheduler_init.get("init_args", {}).keys():
            warn_msg = (
                f"Found an `optimizer` key in the provided `lr_scheduler_init`: {lr_scheduler_init['init_args']} "
                f"Note that the existing optimizer and all associated parameter groups will be used when "
                "reinitializing the lr schedule using the specified scheduler so the provided `optimizer` key will "
                "have no effect."
            )
            rank_zero_warn(warn_msg)
            del lr_scheduler_init["init_args"]["optimizer"]
        min_lr_param = lr_scheduler_init["init_args"].get("min_lr")
        invalid_min_lr = (
            True if min_lr_param and (isinstance(min_lr_param, list) or isinstance(min_lr_param, tuple)) else False
        )
        reinit_rlrop = is_implicit_mode and issubclass(
            lrs_class, torch.optim.lr_scheduler.ReduceLROnPlateau  # type: ignore[arg-type]
        )
        if reinit_rlrop and invalid_min_lr:
            raise MisconfigurationException(
                "In the lr scheduler configuration passed via `reinit_lr_cfg` (i.e. implicit mode training)"
                " `min_lr` cannot be a list or tuple since the same lr scheduler configuration is intended to be"
                " reinitialized at every fine-tuning phase with implicit mode fine-tuning."
            )
        test_lr_init = copy(lr_scheduler_init.get("init_args", {}))
        if min_lr_param:
            del test_lr_init["min_lr"]  # our mock optimizer will not have any param groups
        try:
            assert callable(lrs_class)
            testlr = lrs_class(optimizer=_MockOptimizer(), **test_lr_init)
        except Exception as err:
            error_msg = (
                "Could not configure the specified LR scheduler class using the `init_args` "
                f"({lr_scheduler_init['init_args']}). Recieved the following error while sanity checking schedule "
                f"phases: {err}. Please validate specified `init_args` before resubmitting."
            )
            rank_zero_warn(error_msg)
            raise MisconfigurationException(error_msg)
        assert issubclass(type(testlr), FTSLRSchedulerTypeTuple)


class ScheduleImplMixin(ABC):
    """Functionality for generating and executing fine-tuning schedules."""

    # proper initialization of these variables should be done in the child class
    pl_module: pl.LightningModule
    ft_schedule: Optional[Union[str, dict]]
    reinit_optim_cfg: Optional[Dict]
    reinit_lr_cfg: Optional[Dict]
    max_depth: int
    _fts_state: FTSState
    PHASE_0_DIVERGENCE_MSG = (
        "After executing the provided `configure_optimizers` method, the optimizer state differs from the configuration"
        " FinetuningScheduler expected at the beginning of scheduled fine-tuning (phase 0).\n"
    )

    @property
    @abstractmethod
    def curr_depth(self) -> int:
        pass

    def init_fts(self) -> None:
        """Initializes the fine-tuning schedule and prepares the first scheduled level.

        Calls the relevant
        :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` hooks before and after fine-tuning schedule
        initialization.
        1. Generate the default fine-tuning schedule and/or load it into
        :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.ft_schedule`.
        2. Prepare the first scheduled fine-tuning level, unfreezing the relevant parameters.
        """
        self.strategy_adapter.on_before_init_fts()
        if not self._fts_state._ft_init_epoch:
            self._fts_state._ft_init_epoch = max(self.pl_module.trainer.current_epoch, 0)
        self.init_ft_sched()
        self.strategy_adapter.on_after_init_fts()

    def gen_or_load_sched(self) -> None:
        """Load an explicitly specified fine-tuning schedule if one provided, otherwise generate a default one."""
        assert self.pl_module.trainer is not None
        if not self.ft_schedule and self.max_depth == -1:
            rank_zero_info("No fine-tuning schedule provided, max_depth set to -1 so iteratively thawing entire model")
        assert self.pl_module.trainer.log_dir is not None
        if self.ft_schedule and self.reinit_lr_cfg:
            error_msg = (
                "Specifying both `ft_schedule` and `reinit_lr_cfg` is an invalid configuration. `reinit_lr_cfg` "
                "specifies an lr scheduler configuration to reinitialize with at every new phase of an implicitly "
                "defined fine-tuning shedule whereas `ft_schedule` is an explicity defined schedule. To reinitialize "
                "a given lr scheduler configuration with an explicit fine-tuning schedule, please add the desired "
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
        """Generate the default fine-tuning schedule and/or load it into
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
    def gen_implicit_schedule(self, sched_dir: Union[str, os.PathLike]) -> None:
        """Generate the default schedule, save it to ``sched_dir`` and load it into
        :attr:`~finetuning_scheduler.fts.FinetuningScheduler.ft_schedule`

        Args:
            sched_dir: directory to which the generated schedule should be written. By default will be
                ``Trainer.log_dir``.
        """
        default_ft_schedule = ScheduleImplMixin.gen_ft_schedule(self.pl_module, sched_dir)
        assert default_ft_schedule is not None
        rank_zero_info(f"Generated default fine-tuning schedule '{default_ft_schedule}' for iterative fine-tuning")
        self.ft_schedule = self.load_yaml_schedule(default_ft_schedule)

    @staticmethod
    @rank_zero_only
    def save_schedule(schedule_name: str, layer_config: Dict, dump_loc: Union[str, os.PathLike]) -> os.PathLike:
        """Save loaded or generated schedule to a directory to ensure reproducability.

        Args:
            schedule_name (str): The name of the schedule.
            layer_config (Dict): The saved schedule dictionary.
            dump_loc (os.PathLike): The directory to which the generated schedule (.yaml) should be written

        Returns:
            os.PathLike: The path to the generated schedule, by default ``Trainer.log_dir`` and named after the
            :external+pl:class:`~lightning.pytorch.core.module.LightningModule` subclass in use with the suffix
            ``_ft_schedule.yaml``)
        """
        dump_path = pathlib.Path(dump_loc)
        dump_path.mkdir(exist_ok=True, parents=True)
        ft_schedule_yaml = dump_path / schedule_name
        fs = get_filesystem(ft_schedule_yaml)
        with fs.open(ft_schedule_yaml, "w", newline="") as fp:
            yaml.dump(layer_config, fp)
        assert os.access(ft_schedule_yaml, os.F_OK)
        rank_zero_info(f"fine-tuning schedule dumped to {ft_schedule_yaml}.")
        return ft_schedule_yaml

    @staticmethod
    @rank_zero_only
    def gen_ft_schedule(module: Module, dump_loc: Union[str, os.PathLike]) -> Optional[os.PathLike]:
        """Generate the default fine-tuning schedule using a naive, 2-parameters per-level heuristic.

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` for which a fine-tuning schedule will be
                generated
            dump_loc: The directory to which the generated schedule (.yaml) should be written
        Returns:
            os.PathLike: The path to the generated schedule, by default ``Trainer.log_dir`` and named after the
            :external+pl:class:`~lightning.pytorch.core.module.LightningModule` subclass in use with the suffix
            ``_ft_schedule.yaml``)
        """
        # Note: This initial default fine-tuning schedule generation approach is intentionally simple/naive but is
        # effective for a suprising fraction of models. Future versions of this callback may use module introspection to
        # generate default schedules that better accommodate more complex structures and specific architectures if the
        # callback proves sufficiently useful.
        log.info(f"Proceeding with dumping default fine-tuning schedule for {module.__class__.__name__}")
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
        for i, param_l in enumerate(param_lists):
            layer_config[i] = {"params": param_l}
        schedule_name = f"{module.__class__.__name__}_ft_schedule.yaml"
        assert dump_loc is not None
        return ScheduleImplMixin.save_schedule(schedule_name, layer_config, dump_loc)

    @staticmethod
    def load_yaml_schedule(schedule_yaml_file: os.PathLike) -> Dict:
        """Load a schedule defined in a .yaml file and transform it into a dictionary.

        Args:
            schedule_yaml_file (str): The .yaml fine-tuning schedule file

        Raises:
            MisconfigurationException: If the specified schedule file is not found

        Returns:
            Dict: the Dict representation of the fine-tuning schedule
        """
        try:
            with open(schedule_yaml_file) as df:
                schedule_dict = yaml.load(df, Loader=UniqueKeyLoader)
        except FileNotFoundError as fnf:
            error_msg = (
                f"Could not find specified fine-tuning scheduling file '{schedule_yaml_file}': {fnf}."
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

    def thaw_to_depth(self, depth: Optional[int] = None) -> None:
        """Thaw/unfreeze the current
        :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.pl_module` to the specified
        fine-tuning depth (aka level)

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
                _, self._fts_state._curr_thawed_params = self.strategy_adapter.exec_ft_phase(
                    self.pl_module, thaw_pl=self.strategy_adapter.fts_optim_transform(next_tl["params"])
                )

    @staticmethod
    def _repartition_sharded_optim(optimizer: ParamGroupAddable) -> None:
        """Repartition and reset a sharded optimizer state.

        Args:
            optimizer (ParamGroupAddable): The target optimizer to repartition.
        """
        # For optimizers that shard their states like (e.g. ZeroRedundancyOptimizer), one use case for this method is
        # to clear local optimizer partition caches and repartition to support restoring across multiple depths
        partition_params = (
            optimizer._partition_parameters
            if callable(optimizer._partition_parameters)
            else optimizer.partition_parameters
        )
        optimizer._clear_cache()
        optimizer.optim.param_groups = partition_params()[optimizer.rank]
        optimizer._sync_param_groups(optimizer.optim.param_groups, optimizer.param_groups)

    def _restore_latest_lr_state(self, curr_lr_state: Dict, prev_optimizer_lrs: List) -> None:
        """Adapt the existing lr state for all parameter groups associated with previous depths (new groups for the
        current phase should use the schedule or new optimizer defaults).

        Args:
            curr_lr_state (Dict): The lr state to restore from the current lr scheduler (captured prior to mutation
            associated with adding groups to the new optimizer)
            prev_optimizer_lrs (List): The most recent `lr`s for parameter groups associated with the previous optimizer
        """
        trainer = self.pl_module.trainer
        if trainer.lr_scheduler_configs:  # type: ignore[union-attr]
            for lrs_cfg in trainer.lr_scheduler_configs:  # type: ignore[union-attr]
                lrs_cfg.scheduler.load_state_dict(curr_lr_state)
        for _, data in enumerate(zip(trainer.strategy.optimizers[0].param_groups, prev_optimizer_lrs)):
            param_group, lr = data
            param_group["lr"] = lr
        rank_zero_debug("Current LR state restored for previous depth parameter groups.")

    @staticmethod
    def add_optimizer_groups(
        module: Module,
        optimizer: ParamGroupAddable,
        thawed_pl: List,
        no_decay: Optional[list] = None,
        lr: Optional[float] = None,
        apply_lambdas: bool = False,
    ) -> None:
        """Add optimizer parameter groups associated with the next scheduled fine-tuning depth/level and extend the
        relevent :paramref:`~pytorch_lighting.trainer.trainer.Trainer.lr_scheduler_configs`.

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` from which the target optimizer parameters
                will be read.
            optimizer (:class:`~finetuning_scheduler.types.ParamGroupAddable`): The supported optimizer instance to
                which parameter groups will be configured and added.
            thawed_pl: The list of thawed/unfrozen parameters that should be added to the new parameter group(s)
            no_decay: A list of parameters that should always have weight_decay set to 0. e.g.:
                ["bias", "LayerNorm.weight"]. Defaults to ``None``.
            lr: The initial learning rate for the new parameter group(s). If not specified,
                the ``lr`` of the first scheduled fine-tuning depth will be used. Defaults to ``None``.
            apply_lambdas: Whether to apply lr lambdas to newly added groups. Defaults to False.

        .. note::

            If one relies upon the default FTS schedule, the lr provided to this method will be
            :attr:`~finetuning_scheduler.fts.FinetuningScheduler.base_max_lr` which defaults to ``1e-05``.
        """
        if len(thawed_pl) == 0:
            rank_zero_warn("No thawed parameters passed so no new optimizer groups will be added.")
        else:
            phase_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
            orig_lr_factor = phase_lr
            if module.trainer.lr_scheduler_configs:  # type: ignore[union-attr]
                for config in module.trainer.lr_scheduler_configs:  # type: ignore[union-attr]
                    scheduler = config.scheduler
                    if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas and apply_lambdas:
                        phase_lr = phase_lr * scheduler.lr_lambdas[-1](scheduler.last_epoch)
                    added_pgs = 0
                    added_pgs = ScheduleImplMixin._add_groups(no_decay, optimizer, module, thawed_pl, phase_lr)
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.min_lrs.extend([scheduler.min_lrs[0]] * added_pgs)  # type: ignore[attr-defined]
                    else:
                        scheduler.base_lrs.extend([orig_lr_factor] * added_pgs)
                        if hasattr(scheduler, "lr_lambdas"):
                            scheduler.lr_lambdas.extend([scheduler.lr_lambdas[-1]] * added_pgs)
            else:
                _ = ScheduleImplMixin._add_groups(no_decay, optimizer, module, thawed_pl, phase_lr)

    @staticmethod
    def _add_groups(
        no_decay: Optional[list], optimizer: ParamGroupAddable, module: Module, thawed_pl: List, phase_lr: float
    ) -> int:
        """The actual addition of optimizer groups is done here, separated from ``add_optimizer_groups`` to
        accommodate corner cases where FTS is being used without an lr scheduler configuration.

        Args:
            no_decay: A list of parameters that should always have weight_decay set to 0. e.g.:
                ["bias", "LayerNorm.weight"]. Defaults to ``None``.
            optimizer (:class:`~finetuning_scheduler.types.ParamGroupAddable`): The supported optimizer instance to
                which parameter groups will be configured and added.
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` from which the target optimizer parameters
                will be read.
            thawed_pl: The list of thawed/unfrozen parameters that should be added to the new parameter group(s)
            phase_lr (float): The initial learning rate for the new parameter group(s).

        Returns:
            int: The number of optimizer parameter groups that were added.
        """
        if no_decay:
            optimizer.add_param_group(
                {
                    "params": [
                        p
                        for n, p in module.named_parameters()
                        if not any(nd in n for nd in no_decay) and n in thawed_pl and p.requires_grad
                    ],
                    "lr": phase_lr,
                    "initial_lr": phase_lr,
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
                    "lr": phase_lr,
                    "initial_lr": phase_lr,
                }
            )
            added_pgs = 2
        else:
            optimizer.add_param_group(
                {
                    "params": [p for n, p in module.named_parameters() if n in thawed_pl and p.requires_grad],
                    "lr": phase_lr,
                    "initial_lr": phase_lr,
                }
            )
            added_pgs = 1
        return added_pgs

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

    def _maybe_sync_loops(self) -> None:
        """Synchronize total and current progress loops for the restart of a multi-phase training session."""
        assert self.pl_module._trainer is not None
        fit_loop = self.pl_module._trainer.fit_loop
        if fit_loop.epoch_loop.restarting:  # if ``True``, we haven't completed resetting state
            # since we're restoring from a checkpoint saved prior to processed and completed incrementing
            fit_loop.epoch_progress.increment_processed()
            fit_loop.epoch_progress.increment_completed()
            # ensure current and total are synchronized for the continuation of our multi-phase fine-tuning session
            fit_loop.epoch_progress.current = copy(fit_loop.epoch_progress.total)
            # restarting outside of epoch end is not supported so the assumption here is to start with a fresh epoch
            fit_loop.epoch_loop.restarting = False
            fit_loop.epoch_loop.val_loop._restarting = False

    def _inspect_fts_opt_state(self) -> Tuple:
        """Distills relevant initialized optimizer state for validation prior to fit start.

        Returns:
            Tuple: Distilled optimizer state to be validated.
        """
        assert isinstance(self.ft_schedule, Dict)
        opt = self.pl_module.trainer.optimizers[0]
        sched = self.ft_schedule
        no_grad_cnt = len([p for pg in opt.param_groups for p in pg["params"] if not p.requires_grad])
        init_ft_cnt = len(self.strategy_adapter.fts_optim_inspect(sched[0]["params"]))
        total_ft_cnt = len(
            [p for phase in sched for p in self.strategy_adapter.fts_optim_inspect(sched[phase]["params"])]
        )
        optim_grad_param_set = {p for pg in opt.param_groups for p in pg["params"] if p.requires_grad}
        sched_grad_param_set = {
            p
            for n, p in self.pl_module.named_parameters()
            if n in self.strategy_adapter.fts_optim_inspect(sched[0]["params"])
        }
        expected_params_sym_diff = optim_grad_param_set ^ sched_grad_param_set
        p_diff_summary = defaultdict(list)
        if expected_params_sym_diff:
            for n, p in self.pl_module.named_parameters():
                if p in optim_grad_param_set:
                    p_diff_summary["optim_params"].append(n)
                if p in sched_grad_param_set:
                    p_diff_summary["phase_0_params"].append(n)
        p_diff_summary = {k: self.strategy_adapter.logical_param_translation(v) for k, v in p_diff_summary.items()}
        return no_grad_cnt, init_ft_cnt, total_ft_cnt, p_diff_summary

    @staticmethod
    def _grad_mismatch_feedback(w_msg: str, param_diff_summary: Dict) -> str:
        """Assemble feedback for the user regarding the current optimizer state's divergence from the state
        expected in scheduled fine-tuning phase 0 (with respect to thawed parameters).

        Args:
            w_msg (str): Initial warning message context.
            param_diff_summary (Dict): A summary of the current optimizer state's divergence from the state expected in
                scheduled fine-tuning phase 0 (with respect to thawed parameters).

        Returns:
            str: The user feedback warning with appropriate context.
        """
        w_msg += (
            " a differing set of trainable parameters. Please find below a summary of the differences between"
            " the currently thawed parameters in the optimizer and those scheduled to be optimized during fine-tuning"
            " phase 0: \n"
            "Currently thawed parameters included in the optimizer:\n"
            f"{pformat(param_diff_summary['optim_params'])}{os.linesep}"
            "Parameters expected to be thawed and optimized in phase 0:\n"
            f"{pformat(param_diff_summary['phase_0_params'])}{os.linesep}"
        )
        return w_msg

    def _validate_opt_init(self) -> None:
        """Validate the user-initialized optimizer state (necessary for fine-tuning phase 0) and warn user if
        appropriate.

        Args:
            optimizer (ParamGroupAddable): The optimizer initialized.
            ft_schedule (Dict): The fine-tuning schedule to be inspected vis-a-vis the optimizer state.
        """
        no_grad_cnt, init_ft_cnt, total_ft_cnt, param_diff_summary = self._inspect_fts_opt_state()
        if param_diff_summary or no_grad_cnt > 0:
            if self.enforce_phase0_params:
                # implemented in `StrategyAdapter` since override behavior may be strategy-dependent in the future
                self.strategy_adapter.phase0_optimizer_override()
            else:
                w_msg = ScheduleImplMixin.PHASE_0_DIVERGENCE_MSG + (
                    " Since `enforce_phase0_params` is currently set to `False`, FinetuningScheduler will not override"
                    " the user-configured optimizer configuration to enforce the expected phase 0 configuration of"
                    " thawed parameters."
                    "\n\n"
                    "HINT: Leaving `enforce_phase0_params` to its default (`True`) will avoid discrepancies like this"
                    " in the majority of use cases. If that solution is not desired or sufficient, please find more"
                    " detailed information about the configuration divergence below. \n\n"
                    f"In this case, FinetuningScheduler configured the provided model to have {init_ft_cnt} trainable"
                    " parameters in phase 0 (the initial training phase) but the optimizer has subsequently been"
                    " initialized with"
                )
                if param_diff_summary:
                    w_msg = ScheduleImplMixin._grad_mismatch_feedback(w_msg, param_diff_summary)
                if no_grad_cnt > 0:
                    w_msg += (
                        f"Also note that there are {no_grad_cnt} parameters in the optimizer that do not require a"
                        " gradient. If non-intentional, this state is commonly caused by failing to filter out"
                        " parameters that do not require a gradient when initializing the optimizer (e.g.,"
                        " `parameters = list(filter(lambda x: x.requires_grad, self.parameters()))`. If you intended to"
                        " initialize the optimizer with parameters that do not require a gradient you may want to"
                        f" ensure they are not included in the {total_ft_cnt} parameters that the FinetuningScheduler"
                        " is currently configured to thaw (sum of all phases) to avoid triggering a parameter collision"
                        " and training failure in pytorch during a future fine-tuning phase."
                    )
                rank_zero_warn(w_msg)


class CallbackDepMixin(ABC):
    """Functionality for validating/managing callback dependencies."""

    def __init__(self, callback_dep_parents: Dict = CALLBACK_DEP_PARENTS) -> None:
        """Arguments used to initialize the user-provided callback dependency validation in accordance with the
        user-provided module configuration:

        Args:
            callback_dep_parents (Dict, optional): The parent classes of all user-provided callbacks in the module that
                should be connected to the target user-provided callback. Defaults to CALLBACK_DEP_PARENTS in the
                user-provided module.
        """
        super().__init__()
        self.callback_dep_parents = callback_dep_parents

    def _inspect_callback_deps(self, trainer: "pl.Trainer") -> List[bool]:
        """Inspect the trainer :paramref:`~pytorch_lighting.trainer.trainer.Trainer.callbacks` for earlystopping
        and scheduled fine-tuning capabilities.

        Args:
            trainer (pl.Trainer):  The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object to
                inspect the callbacks of

        Returns:
            Tuple[bool]: The ascertained :paramref:`~pytorch_lighting.trainer.trainer.Trainer.callbacks` capabilities
        """
        callbacks_inspected = [FTSCheckpoint, ModelCheckpoint, FTSEarlyStopping, EarlyStopping, LearningRateMonitor]
        callback_inspection = []
        self._validate_dep_callbacks(trainer)
        for ci in callbacks_inspected:
            callback_inspection.append(any([isinstance(c, ci) for c in trainer.callbacks]))
        return callback_inspection

    def _validate_dep_callbacks(self, trainer: "pl.Trainer") -> None:
        """Validate multiple instances of a given user-provided callback dependency parent are not present.

        Args:
            trainer (pl.Trainer): The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object to
                inspect the callbacks of

        Raises:
            MisconfigurationException: If multiple instances of a callback dependency parent are found
        """
        dep_callback_cnts: Dict = {}
        dep_callback_errs = []
        err_suffix = "callbacks is not currently supported. Please provide a maximum of one."
        for k, v in self.callback_dep_parents.items():
            dep_callback_cnts.setdefault(k, 0)
            for c in trainer.callbacks:
                if isinstance(c, v):
                    dep_callback_cnts[k] += 1
                if dep_callback_cnts[k] > 1:
                    break
        for k, v in dep_callback_cnts.items():
            # this block is only required for any non-stateful callback dependencies as multiple stateful callback
            # dependencies will be prevented in on_train_init
            if v > 1:
                dep_callback_errs.append(f"Use of multiple {k} {err_suffix}")
        if dep_callback_errs:
            raise MisconfigurationException(dep_callback_errs)

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

    def _callback_dep_setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Ensures all :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback dependencies are met, adding
        and configuring them if necessary.

        Args:
            trainer (:external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer`): The
                :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object
            pl_module (:external+pl:class:`~lightning.pytorch.core.module.LightningModule`): The
                :external+pl:class:`~lightning.pytorch.core.module.LightningModule` object
            stage: The ``RunningStage.{SANITY_CHECKING,TRAINING,VALIDATING}``. Defaults to None.
        """
        trainer.callbacks, added_es_fts, added_ckpt_fts = self._configure_callback_deps(trainer)
        # if we added callbacks for the user after the setup hooks loop was initiated from trainer, we'll need to
        # explicitly call the setup hooks for those added callbacks
        if added_ckpt_fts:
            trainer.checkpoint_callback.setup(trainer, pl_module, stage)  # type: ignore[union-attr]
        if added_es_fts:
            trainer.early_stopping_callback.setup(trainer, pl_module, stage)  # type: ignore[union-attr]

    def _configure_callback_deps(self, trainer: "pl.Trainer") -> Tuple[List[Callback], bool, bool]:
        """Ensures FTSCheckpoint and :external+pl:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`
        callbacks are present and configured, removing any.

        :external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`s if present.

        Args:
            trainer: The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object that may have its
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
                    f"{self.__class__.__name__} currently depends upon a fine-tuning schedule "
                    "capable EarlyStopping callback such as FTSEarlyStopping. Substituting current "
                    "EarlyStopping for FTSEarlyStopping"
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
                    f"{self.__class__.__name__} currently depends upon a fine-tuning schedule "
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
