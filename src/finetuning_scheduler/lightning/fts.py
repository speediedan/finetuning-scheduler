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
r"""Fine-Tuning Scheduler.

Used to implement flexible fine-tuning training schedules
"""
import logging
import os
from copy import deepcopy
from pprint import pformat
from typing import Any  # Dict is used for runtime isinstance() checks
from collections.abc import Callable, Sequence
from typing_extensions import override

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities import rank_zero_info, rank_zero_warn
from lightning.fabric.utilities.distributed import ReduceOp
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.strategies.strategy import Strategy
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_debug
from lightning.pytorch.trainer.connectors.logger_connector.result import _ResultMetric

from finetuning_scheduler.core import optim as core_optim
from finetuning_scheduler.core import phase as core_phase
from finetuning_scheduler.core import transitions as core_transitions
from finetuning_scheduler.core.views import canonical_fqn
from finetuning_scheduler.lightning.fts_supporters import (
    CallbackDepMixin,
    FTSEarlyStopping,
    FTSState,
    ScheduleImplMixin,
    ScheduleParsingMixin,
    STRATEGY_ADAPTERS,
    _discover_strategy_adapters
)
from finetuning_scheduler.strategy_adapters.base import StrategyAdapter
from finetuning_scheduler.types import ParamGroupAddable

# the logger keeps its established name, so existing logging configuration still applies
log = logging.getLogger("finetuning_scheduler.fts")

LIGHTNING_EXTRA_NOTICE = (
    "Lightning will become an optional dependency of finetuning-scheduler, no earlier than 2.17. Nothing changes before"
    " then: Lightning remains a base dependency and the Lightning integration works as it does today. Depending on"
    " `finetuning-scheduler[lightning]` (or `[pytorch-lightning]` for the standalone package) now means the change"
    " needs no edit to your requirements; the extra is already accepted."
)
_lightning_extra_notice_issued = False


def _notify_lightning_extra() -> None:
    """Issue the Lightning-extra notice once per process."""
    global _lightning_extra_notice_issued
    if not _lightning_extra_notice_issued:
        _lightning_extra_notice_issued = True
        rank_zero_warn(LIGHTNING_EXTRA_NOTICE, category=FutureWarning)


def _optimizer_class_name(new_optimizer: dict) -> str | None:
    """The class name an explicit ``new_optimizer`` configuration builds (``torch.optim.Adam`` and ``Adam`` alike
    give ``Adam``)."""
    class_path = (new_optimizer.get("optimizer_init") or {}).get("class_path")
    return class_path.rsplit(".", 1)[-1] if isinstance(class_path, str) else None


class FinetuningScheduler(ScheduleImplMixin, ScheduleParsingMixin, CallbackDepMixin, BaseFinetuning):
    r"""This callback enables flexible, multi-phase, scheduled fine-tuning of foundation models. Gradual
    unfreezing/thawing can help maximize foundation model knowledge retention while allowing (typically upper
    layers of) the model to optimally adapt to new tasks during transfer learning.
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` orchestrates the gradual unfreezing of models via a
    fine-tuning schedule that is either implicitly generated (the default) or explicitly provided by the user (more
    computationally efficient).

    Fine-tuning phase transitions are driven by
    :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria (a multi-phase
    extension of :py:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`), user-specified epoch
    transitions or a composition of the two (the default mode). A
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` training session completes
    when the final phase of the schedule has its stopping criteria met. See
    :ref:`Early Stopping<common/early_stopping:Early stopping>` for more details on that callback's configuration.

    Schedule definition is facilitated via
    :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.gen_ft_schedule` which dumps
    a default fine-tuning schedule (by default using a naive, 2-parameters per level heuristic) which can be adjusted as
    desired by the user and subsuquently passed to the callback. Implicit fine-tuning mode generates the default
    schedule and proceeds to fine-tune according to the generated schedule. Implicit fine-tuning will often be less
    computationally efficient than explicit fine-tuning but can often serve as a good baseline for subsquent explicit
    schedule refinement and can marginally outperform many explicit schedules.

    Example::

        import lightning as L
        from lightning.pytorch.callbacks import FinetuningScheduler
        trainer = L.Trainer(callbacks=[FinetuningScheduler()])

    .. note::

        Currently, :class:`~finetuning_scheduler.fts.FinetuningScheduler` does not support the use of multiple
        :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` or
        :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` callback instances.

    .. note::

       While :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports the use of
       :py:class:`~torch.distributed.optim.ZeroRedundancyOptimizer`, setting ``overlap_with_ddp`` to
       ``True`` is not supported because that optimizer mode only supports a single parameter group.

    .. note::

       While :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports the use of
       :py:class:`~torch.distributed.optim.ZeroRedundancyOptimizer`, setting ``overlap_with_ddp`` to
       ``True`` is not supported because that optimizer mode only supports a single parameter group.
    """
    pl_module: pl.LightningModule
    trainer: pl.Trainer
    # set while restoring a best checkpoint whose optimizer state predates the optimizer in use (model state only)
    _restore_model_only: bool = False
    # the class name of the optimizer `configure_optimizers` built (set at fit start)
    _initial_optimizer_cls: str | None = None
    # transitions are sequenced by an externally driven `FTSController` (set to `False` to use the callback's own,
    # for one release); the controller is attached at the fit's first transition or resume
    _use_fts_controller: bool = True
    _fts_controller: Any = None
    strategy_adapter: StrategyAdapter

    def __init__(
        self,
        ft_schedule: str | dict | None = None,
        max_depth: int = -1,
        base_max_lr: float = 1e-5,
        restore_best: bool = True,
        gen_ft_sched_only: bool = False,
        epoch_transitions_only: bool = False,
        reinit_optim_cfg: dict | None = None,
        reinit_lr_cfg: dict | None = None,
        strategy_adapter_cfg: dict | None = None,
        custom_strategy_adapters: dict[str, str] | None = None,
        allow_untested: bool = False,
        apply_lambdas_new_pgs: bool = False,
        logging_level: int = logging.INFO,
        enforce_phase0_params: bool = True,
        frozen_bn_track_running_stats: bool = True,
        log_dir: str | os.PathLike | None = None,
    ):
        r"""
        Arguments used to define and configure a scheduled fine-tuning training session:

        Args:
            ft_schedule: The fine-tuning schedule to be executed. Usually will be a .yaml file path but can also be a
                properly structured Dict. See
                :ref:`Specifying a Fine-Tuning Schedule<index:Specifying a fine-tuning schedule>`
                for the basic schedule format. See
                :ref:`LR Scheduler Reinitialization<explicit-lr-reinitialization-schedule>` for more complex
                schedule configurations (including per-phase LR scheduler reinitialization). If a schedule is not
                provided, will generate and execute a default fine-tuning schedule using the provided
                :py:class:`~lightning.pytorch.core.module.LightningModule`. See
                :ref:`the default schedule<index:The Default Fine-Tuning Schedule>`. Defaults to ``None``.
            max_depth: Maximum schedule depth to which the defined fine-tuning schedule should be executed. Specifying
                -1 or an integer > (number of defined schedule layers) will result in the entire fine-tuning schedule
                being executed. Defaults to -1.
            base_max_lr: The default maximum learning rate to use for the parameter groups associated with each
                scheduled fine-tuning depth if not explicitly specified in the fine-tuning schedule. If overridden to
                ``None``, will be set to the ``lr`` of the first scheduled fine-tuning depth. Defaults to 1e-5.
            restore_best: If ``True``, restore the best available (defined by the
                :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint`) checkpoint
                before fine-tuning depth transitions. Defaults to ``True``.
            gen_ft_sched_only: If ``True``, generate the default fine-tuning schedule to
                :attr:`~finetuning_scheduler.fts.FinetuningScheduler.log_dir` (it will be named after your
                :py:class:`~lightning.pytorch.core.module.LightningModule` subclass with
                the suffix ``_ft_schedule.yaml``) and exit without training. Typically used to generate a default
                schedule that will be adjusted by the user before training. Defaults to ``False``.
            epoch_transitions_only: If ``True``, use epoch-driven stopping criteria exclusively (rather than composing
                :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` and
                epoch-driven criteria which is the default). If using this mode, an epoch-driven transition
                (``max_transition_epoch`` >= 0) must be specified for each phase. If unspecified,
                ``max_transition_epoch`` defaults to -1 for each phase which signals the application of
                :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria only.
                epoch_transitions_only defaults to ``False``.
            reinit_optim_cfg: An optimizer reinitialization configuration dictionary consisting of at minimum a nested
                ``optimizer_init`` dictionary with a ``class_path`` key specifying the class of the optimizer to be
                instantiated. Optionally, an ``init_args`` dictionary of arguments with which to initialize the
                optimizer may be included. A ``reinit_lr_cfg`` configuration can also be specified concurrently. By way
                of example, one could configure this dictionary via the
                :py:class:`~lightning.pytorch.cli.LightningCLI` with the following:

                .. code-block:: yaml

                    reinit_optim_cfg:
                        optimizer_init:
                            class_path: torch.optim.SGD
                            init_args:
                                lr: 1.0e-05
                                momentum: 0.9
                                weight_decay: 1.0e-06

            reinit_lr_cfg: A lr scheduler reinitialization configuration dictionary consisting of at minimum a nested
                ``lr_scheduler_init`` dictionary with a ``class_path`` key specifying the class of the lr scheduler
                to be instantiated. Optionally, an ``init_args`` dictionary of arguments to initialize the lr scheduler
                with may be included. Additionally, one may optionally include arguments to pass to PyTorch Lightning's
                lr scheduler configuration :class:`~lightning.pytorch.utilities.types.LRSchedulerConfig` in the
                ``pl_lrs_cfg`` dictionary. A ``reinit_optim_cfg`` configuration can also be specified concurrently. By
                way of example, one could configure this dictionary via the
                :py:class:`~lightning.pytorch.cli.LightningCLI` with the following:

                .. code-block:: yaml

                    reinit_lr_cfg:
                        lr_scheduler_init:
                            class_path: torch.optim.lr_scheduler.StepLR
                            init_args:
                                step_size: 1
                                gamma: 0.7
                            pl_lrs_cfg:
                                interval: epoch
                                frequency: 1
                                name: Implicit_Reinit_LR_Scheduler
                            use_current_optimizer_pg_lrs: true

            allow_untested: If ``True``, allows the use of custom or unsupported training strategies and lr schedulers
                (e.g. ``single_tpu``, ``MyCustomStrategy``, ``MyCustomLRScheduler``) . Defaults to ``False``.

                .. note:: Custom or officially unsupported strategies and lr schedulers can be used by setting
                    :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.allow_untested` to ``True``.

                    Some officially unsupported strategies may work unaltered and are only unsupported due to
                    the ``Fine-Tuning Scheduler`` project's lack of CI/testing resources for that strategy (e.g.
                    ``single_tpu``).

                    Most unsupported strategies and schedulers, however, are currently unsupported because they require
                    varying degrees of modification to be compatible.

                    For instance, with respect to strategies, ``deepspeed`` will require a
                    :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` similar to the one written for
                    ``FSDP`` (:class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter`) to be written before
                    support can be added (PRs welcome!), while ``tpu_spawn`` would require an override of the current
                    broadcast method to include python objects.

                    Regarding lr schedulers, :py:class:`~torch.optim.lr_scheduler.ChainedScheduler` and
                    :py:class:`~torch.optim.lr_scheduler.SequentialLR` are examples of schedulers not
                    currently supported due to the configuration complexity and semantic conflicts supporting them would
                    introduce. If a supported torch lr scheduler does not meet your requirements, one can always
                    subclass a supported lr scheduler and modify it as required
                    (e.g. :py:class:`~torch.optim.lr_scheduler.LambdaLR` is especially useful for this).
            strategy_adapter_cfg: A configuration dictionary that will be applied to the
                :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` associated with the current training
                :py:class:`~lightning.pytorch.strategies.Strategy`. See the relevant
                :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` documentation for strategy-specific
                configuration options. Defaults to None.
            custom_strategy_adapters: A dictionary mapping PyTorch Lightning strategy flags (canonical strategy
                names like ``"single_device"``, ``"auto"``, ``"ddp"``, etc.) to strategy adapter references. Multiple
                ``strategy_flag`` keys can be associated with the same adapter. The adapter reference can be: (1) an
                entry point name registered under ``finetuning_scheduler.strategy_adapters`` (see
                :ref:`strategy_adapter_entry_points`) (2) a fully
                qualified :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` subclass path in the
                format ``"module.path:ClassName"`` or (3) a fully qualified dot path in the format
                ``"module.path.ClassName"``. This is an experimental feature that is subject to change.
                Defaults to None.
            apply_lambdas_new_pgs: If ``True``, applies most recent lambda in ``lr_lambdas`` list to newly added
                optimizer groups for lr schedulers that have a ``lr_lambdas`` attribute. Note this option only applies
                to phases without reinitialized lr schedulers. Phases with defined lr scheduler reinitialization configs
                will always apply the specified lambdas. Defaults to ``False``.
            logging_level: Sets the logging level for :class:`~finetuning_scheduler.fts.FinetuningScheduler`. Defaults
                to ``INFO``.
            enforce_phase0_params: Whether :class:`~finetuning_scheduler.fts.FinetuningScheduler` will reconfigure the
                user-configured optimizer (configured via `configure_optimizers`) to optimize the parameters (and only
                those parameters) scheduled to be optimized in phase 0 of the current fine-tuning schedule.
                Reconfiguration will only take place if FTS discovers the set of parameters to be initially thawed
                and present in the optimizer differs from the parameters specified in phase 0. Only the parameters
                included in the optimizer are affected; the choice of optimizer, lr_scheduler etc. remains unaltered.
                Defaults to ``True``.
            frozen_bn_track_running_stats: When freezing ``torch.nn.modules.batchnorm._BatchNorm`` layers, whether
                :class:`~finetuning_scheduler.fts.FinetuningScheduler` should set ``BatchNorm`` ``track_running_stats``
                to ``True``. Setting this to ``True`` overrides the default Lightning behavior that sets
                ``BatchNorm`` ``track_running_stats`` to ``False`` when freezing ``BatchNorm`` layers. Defaults to
                ``True``.
            log_dir: Directory to use for :class:`~finetuning_scheduler.fts.FinetuningScheduler` artifacts. Defaults to
                ``trainer.log_dir`` if ``None`` or ``trainer.default_root_dir`` if ``trainer.log_dir`` is also ``None``.

        Attributes:
            _fts_state: The internal :class:`~finetuning_scheduler.fts.FinetuningScheduler` state.
            strategy_adapter_cfg: A configuration dictionary that will be applied to the
                :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` associated with the current training
                :py:class:`~lightning.pytorch.strategies.Strategy`.
            epoch_transitions_only: Whether to use epoch-driven stopping criteria exclusively.
            base_max_lr: The default maximum learning rate to use for the parameter groups associated with each
                scheduled fine-tuning depth if not explicitly specified in the fine-tuning schedule. If overridden to
                ``None``, will be set to the ``lr`` of the first scheduled fine-tuning depth. Defaults to 1e-5.
        """
        super().__init__()
        _notify_lightning_extra()
        self._fts_state = FTSState()
        self.max_depth = max_depth
        self.restore_best = restore_best
        self.ft_schedule = ft_schedule
        self.base_max_lr = base_max_lr
        self.gen_ft_sched_only = gen_ft_sched_only
        self.epoch_transitions_only = epoch_transitions_only
        self.reinit_optim_cfg = reinit_optim_cfg
        self.reinit_lr_cfg = reinit_lr_cfg
        self.strategy_adapter_cfg = strategy_adapter_cfg or {}
        self.custom_strategy_adapters = custom_strategy_adapters
        self._phase_hooks: list[tuple] = []  # (fn, when, isolate_rng), registered on each fit's controller
        self.allow_untested = allow_untested
        self.apply_lambdas_new_pgs = apply_lambdas_new_pgs
        self.enforce_phase0_params = enforce_phase0_params
        self.frozen_bn_track_running_stats = frozen_bn_track_running_stats
        self._has_reinit_schedule = False
        self._log_dir = log_dir
        self._msg_cache = set()
        rz_logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")
        rz_logger.setLevel(logging_level)

    @property
    def log_dir(self) -> str | os.PathLike | None:
        """Directory to used for :class:`~finetuning_scheduler.fts.FinetuningScheduler` artifacts.

        Returns:
            str | os.PathLike | None: The directory to use, falling back to ``trainer.log_dir`` if
            ``FinetuningScheduler._log_dir`` is not set, and ``trainer.default_root_dir`` if ``trainer.log_dir`` is also
            ``None``.
        """
        if self._log_dir is not None:  # short-circuit for explicit log_dir
            return self._log_dir

        log_dir, trainer = None, getattr(self, 'trainer', None)
        if trainer:
            # Use trainer.log_dir if available, otherwise fall back to trainer.default_root_dir
            log_dir = trainer.log_dir if getattr(trainer, 'log_dir', None) is not None else trainer.default_root_dir
        return log_dir

    @property
    def curr_depth(self) -> int:
        """Index of the fine-tuning schedule depth currently being trained.

        Returns:
            int: The index of the current fine-tuning training depth
        """
        return self._fts_state._curr_depth

    @property
    def depth_remaining(self) -> int:
        """Remaining number of fine-tuning training levels in the schedule.

        Returns:
            int: The number of remaining fine-tuning training levels
        """
        return core_phase.depth_remaining(self.max_depth, self._fts_state._curr_depth)

    @staticmethod
    def _supported_strategy_flags() -> Sequence[str]:
        return (
            "ddp",
            "ddp_find_unused_parameters_false",
            "ddp_find_unused_parameters_true",
            "ddp_spawn",
            "ddp_fork",
            "ddp_notebook",
            "single_device",
            "fsdp",
            "fsdp_cpu_offload",
            'modelparallelstrategy',
            # "deepspeed",  # relevant FTS strategy adapter not yet available, PRs welcome!
        )

    @override
    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        """Freezes all model parameters so that parameter subsets can be subsequently thawed according to the fine-
        tuning schedule.

        Args:
            pl_module (:py:class:`~lightning.pytorch.core.module.LightningModule`): The target
                :py:class:`~lightning.pytorch.core.module.LightningModule` to freeze parameters of
        """
        # We avoid overriding `BaseFinetuning`'s `freeze` and `freeze_module` methods at the small marginal cost
        # of conditionally revisiting `BatchNorm` layers to set `track_running_stats` to `True` when we are in
        # `frozen_bn_track_running_stats` mode.
        BaseFinetuning.freeze(modules=pl_module, train_bn=False)
        self.strategy_adapter._module_specific_freezing(modules=pl_module)

    def step(self) -> None:
        """Prepare and execute the next scheduled fine-tuning level
        1. Restore the current best model checkpoint if appropriate
        2. Thaw model parameters according the the defined schedule
        3. Synchronize the states of ``FitLoop`` and :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state`

        .. note::

            The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback initially
            only supports single-schedule/optimizer fine-tuning configurations
        """
        assert self.trainer is not None
        # the validation loop's forward-only pass leaves an FSDP2 root module's unsharded parameters registered
        core_phase.reshard_fsdp_modules(self.pl_module)
        if self._use_fts_controller:
            self._controller_step()
        else:
            self._legacy_step()
        if self.depth_remaining == 0 and not self.epoch_transitions_only:
            assert self.trainer.early_stopping_callback is not None
            self.trainer.early_stopping_callback.final_phase = True  # type: ignore[attr-defined]
        assert self._fts_state._ft_sync_objects is not None
        FinetuningScheduler.sync(self._fts_state._ft_sync_objects, self._fts_state._ft_sync_props)
        if self.pl_module._compiler_ctx and self.pl_module._compiler_ctx.get("compiler", None) == "dynamo":
            # reset currently required as `AOTAutograd`` is getting confused by `requires_grad` alteration
            torch._dynamo.reset()
        rank_zero_info(f"Multi-phase fine-tuned training continuing at level {self.curr_depth}.")
        if self.depth_remaining == 0:
            max_epochs_msg = f"`max_epochs` ({self.trainer.fit_loop.max_epochs}) is reached."
            composition_msg = "the early stopping conditions are met or " + max_epochs_msg
            rank_zero_info(
                f"Given the current configuration of `max_depth` ({self.max_depth}), this training session"
                f" will now end when {max_epochs_msg if self.epoch_transitions_only else composition_msg}"
            )

    def _controller_step(self) -> None:
        """Hand the transition (or the resume) the callback decided on to its externally driven
        :class:`~finetuning_scheduler.core.controller.FTSController`, which sequences it and carries it out through
        this callback's ``restore_best_ckpt``, ``step_pg`` and ``thaw_to_depth`` and the strategy adapter's
        ``on_after_phase_transition``, as :meth:`_legacy_step` calls them."""
        from finetuning_scheduler.lightning.controller_bridge import attach_controller

        if self._fts_controller is None:  # a subclass's `on_fit_start` did not attach it
            self._fts_controller = attach_controller(self, self._phase_hooks,
                                                     run_start_hooks=not self._fts_state._resume_fit_from_ckpt)
        controller = self._fts_controller
        if self._fts_state._resume_fit_from_ckpt:
            controller.resume_at(self.curr_depth)
            return
        if controller.curr_depth != self.curr_depth - 1:
            # the callback's depth is authoritative (it is checkpointed, and subclasses may set it)
            rank_zero_debug(f"Syncing the controller's phase ({controller.curr_depth}) to the callback's"
                            f" ({self.curr_depth - 1}) before advancing.")
            controller._depth = self.curr_depth - 1
        controller.request_advance()

    def _legacy_step(self) -> None:
        """The callback's own transition (or resume), for one release beside the controller's (see
        ``_use_fts_controller``)."""
        assert self.trainer is not None
        pre_reinit_state = self._save_pre_reinit_lr_state(self.trainer)
        if not self._fts_state._resume_fit_from_ckpt:
            if self.restore_best:
                self.restore_best_ckpt()
                self.step_pg(
                    depth=self.curr_depth,
                    optimizer=self.trainer.optimizers[0],  # type: ignore[arg-type]
                    pre_reinit_state=pre_reinit_state,
                )
            else:
                self.step_pg(
                    depth=self.curr_depth,
                    optimizer=self.trainer.optimizers[0],  # type: ignore[arg-type]
                    depth_sync=False,
                    pre_reinit_state=pre_reinit_state,
                )
        else:
            self.thaw_to_depth()
        self.strategy_adapter.on_after_phase_transition()

    def step_pg(
        self,
        optimizer: ParamGroupAddable,
        depth: int,
        depth_sync: bool = True,
        pre_reinit_state: tuple | None = None,
    ) -> None:
        """Configure optimizer parameter groups for the next scheduled fine-tuning level, adding parameter groups
        beyond the restored optimizer state up to
        :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.current_depth` and reinitializing the optimizer
        and/or learning rate scheduler as configured.

        Args:
            optimizer (:class:`~finetuning_scheduler.types.ParamGroupAddable`): The supported optimizer instance to
                which parameter groups will be configured and added.
            depth: The maximum index of the fine-tuning schedule for which to configure the optimizer parameter
                groups.
            depth_sync: If ``True``, configure optimizer parameter groups for all depth indices greater
                than the restored checkpoint. If ``False``, configure groups only for the specified depth. Defaults to
                ``True``.
        """
        next_tl: dict = {}
        assert isinstance(self.ft_schedule, dict)
        assert isinstance(self.pl_module, pl.LightningModule)
        assert isinstance(self.trainer, pl.Trainer)
        core_phase.reshard_fsdp_modules(self.pl_module)  # thaw and optimize the parameters FSDP2 trains
        # if the target depth is 0, implicit optimizer reinitialization should not be executed. A restored best
        # checkpoint that predates an explicit optimizer reinitialization had its optimizer state skipped, so the
        # optimizer is rebuilt as that reinitialization configured it
        stale_cfg = self._optimizer_reinit_since_best(depth) if depth_sync else None
        new_optimizer_cfg, thaw_layers = core_phase.thaw_plan(
            self.ft_schedule, depth, self._fts_state._best_ckpt_depth, depth_sync, self.reinit_optim_cfg or stale_cfg
        )
        for i, orig_next_tl in thaw_layers:
            self.strategy_adapter._maybe_set_bn_track_running_stats(i)
            next_tl = deepcopy(orig_next_tl)
            if i <= depth:
                next_tl["params"] = self.strategy_adapter.fts_optim_transform(next_tl["params"])
                _, self._fts_state._curr_thawed_params = self.strategy_adapter.exec_ft_phase(
                    self.pl_module, thaw_pl=next_tl["params"]
                )
                if new_optimizer_cfg and i == 0:
                    # If reinitializing the optimizer, we need to re-add the initial parameter groups (phase 0)
                    optimizer = self.reinit_optimizer(
                        new_optimizer=new_optimizer_cfg, trainer=self.trainer, init_params=next_tl["params"]
                    )
                else:
                    # Add pgs and configure the learning rate scheduler using the current optimizer/schedule
                    self._add_pgs_config_lrs(optimizer, next_tl, depth, depth == i, pre_reinit_state)

    def _add_pgs_config_lrs(
        self,
        optimizer: ParamGroupAddable,
        next_tl: dict,
        depth: int,
        is_target_depth: bool,
        pre_reinit_state: tuple | None,
    ) -> None:
        """Add optimizer parameter groups and potentially reinitialize/reconfigure the learning rate scheduler
        according to a given schedule phase configuration.

        Args:
            optimizer (:class:`~finetuning_scheduler.types.ParamGroupAddable`): The supported optimizer instance to
                which parameter groups will be configured and added.
            next_tl (Dict): A dictionary containing the schedule configuration associated with the current phase
                context.
            depth (int): The current target depth.
            is_target_depth (bool): Whether this restoration stage is the current target depth.
            pre_reinit_state (Tuple): Contingent on restoration context, lr scheduler and optimizer lr state to restore.
        """
        # if the target depth is 0, lr scheduler reinitialization should not be executed
        new_scheduler_cfg = core_phase.lr_reinit_cfg_for(next_tl, depth, self.reinit_lr_cfg)
        if is_target_depth and depth > 0:
            # NB: The latest optimizer and lr scheduler lr state will be re-applied to all existing param groups before
            # implementing the next phase.
            assert pre_reinit_state
            self._restore_latest_lr_state(*pre_reinit_state)
        FinetuningScheduler.add_optimizer_groups(
            module=self.pl_module,
            optimizer=optimizer,
            thawed_pl=next_tl["params"],
            lr=next_tl.get("lr", optimizer.defaults["lr"]),
            no_decay=getattr(self.pl_module, "no_decay", None),
            apply_lambdas=self.apply_lambdas_new_pgs,
            **({"pg_overrides": next_tl["pg_overrides"]} if next_tl.get("pg_overrides") else {}),
        )
        if new_scheduler_cfg:
            self.reinit_lr_scheduler(
                new_lr_scheduler=new_scheduler_cfg, trainer=self.trainer, optimizer=optimizer
            )
        else:
            self._maybe_warn_lr_lambdas()

    def _maybe_warn_lr_lambdas(self) -> None:
        """If appropriate, warn the user that `lr_lambdas` will not be applied given the current configuration."""
        core_optim.warn_unapplied_lr_lambdas(
            [config.scheduler for config in self.trainer.lr_scheduler_configs], self.apply_lambdas_new_pgs
        )

    def restore_best_ckpt(self) -> None:
        """Restore the current best model checkpoint, according to
        :paramref:`~finetuning_scheduler.fts_supporters.FTSCheckpoint.best_model_path`"""
        assert self.trainer is not None
        # wait for all processes to be ready to restore ckpt before restoring
        self.trainer.strategy.barrier("setup_next_level")
        stale = self._optimizer_reinit_since_best(self.curr_depth) is not None
        self._restore_model_only = stale
        if stale:
            rank_zero_info(
                f"The best checkpoint (phase {self._fts_state._best_ckpt_depth}) predates the optimizer"
                " reinitialization configured by a later phase; restoring its model state only and rebuilding the"
                " optimizer as that phase configures it."
            )
        # if restarting across multiple depths, need to ensure we're restoring optimizer state appropriately
        # by resetting optimizer groups and allowing state dict to be reset commensurate w/ ckpt state
        # the saved groups name parameters as the saving run's wrappers did (FSDP1's `_fsdp_wrapped_module.`, for a
        # checkpoint resumed under FSDP2), so both sides are matched by wrapper-free name
        named_params = {canonical_fqn(n): p for n, p in self.pl_module.named_parameters()}
        for opt_idx, optimizer in enumerate(self.trainer.optimizers if not stale else []):
            saved_pgs = [{**g, "params": [canonical_fqn(n) for n in g["params"]]}
                         for g in self._fts_state._fts_ckpt_metadata["best_ckpt_pgs"][opt_idx]]
            optimizer.param_groups = BaseFinetuning._apply_mapping_to_param_groups(saved_pgs, named_params)
            if self.strategy_adapter.using_sharded_optimizer:
                ScheduleImplMixin._repartition_sharded_optim(optimizer)  # type: ignore[arg-type]
        # we're restoring everything but callbacks and loops, otherwise, checkpoint_connector.restore() could be used
        assert self.trainer.checkpoint_callback is not None
        checkpoint_path = self.trainer.checkpoint_callback.best_model_path  # type: ignore[attr-defined]
        if (adapter := getattr(self, "strategy_adapter", None)) is not None:
            checkpoint_path = adapter.best_ckpt_path(checkpoint_path)
        try:
            self.trainer._checkpoint_connector.resume_start(checkpoint_path=checkpoint_path)
        except KeyError as ke:  # we may want to allow training to progress conditioned on context of restoration
            self._maybe_allow_incompatible_reinit_ckpt(ke)
        self.trainer._checkpoint_connector.restore_datamodule()

        # Inspect state before strategy adapter transformation
        loaded_ckpt = self.trainer._checkpoint_connector._loaded_checkpoint

        # Allow strategy-specific adapters to transform the state dict before model restoration
        if self.strategy_adapter is not None:
            try:
                loaded_ckpt = self.strategy_adapter.before_restore_model(loaded_ckpt)
                # assign back in case adapter replaced or modified the checkpoint
                self.trainer._checkpoint_connector._loaded_checkpoint = loaded_ckpt
            except Exception as err:
                rank_zero_warn(f"Strategy adapter before_restore_model hook raised: {err}")

        self.trainer._checkpoint_connector.restore_model()
        # we need to override checkpoint_connector.restore_training_state() to bypass loop restoration
        # if additional customizations are required, may make sense to subclass _CheckpointConnector at some point
        try:
            self._restore_training_state()
        finally:
            self._restore_model_only = False
        self.trainer._checkpoint_connector.resume_end()

    def _restore_training_state(self) -> None:
        """Restore training state without restoring loops from the pre-loaded checkpoint.

        This includes the precision settings, optimizer states and learning rate scheduler states.
        """
        assert self.pl_module is not None and self.trainer is not None
        checkpoint_connector = self.trainer._checkpoint_connector

        # restore precision plugin (scaler etc.)
        checkpoint_connector.restore_precision_plugin_state()

        # checkpoint_connector.restore_training_state() would restore loops here
        # checkpoint_connector.restore_loops()

        assert self.trainer.state.fn is not None
        if self.trainer.state.fn == TrainerFn.FITTING and not self._restore_model_only:
            try:
                # enable strategy adapters to restore optimizer if `Strategy.lightning_restore_optimizer` is overridden
                self.strategy_adapter.on_before_restore_optimizers_and_lrs()
                # restore optimizers and schedulers state
                checkpoint_connector.restore_optimizers_and_schedulers()
            except KeyError as ke:
                self._maybe_allow_incompatible_reinit_ckpt(ke)

    def _optimizer_reinit_since_best(self, depth: int) -> dict | None:
        """The configuration of the latest explicit optimizer reinitialization between the best checkpoint's phase
        and ``depth`` (exclusive), when it changed the optimizer's class: the best checkpoint's optimizer state
        then cannot be restored into the optimizer in use (as ``FTSController`` decides, by class).

        ``None`` when nothing is restored (``restore_best=False``), when the optimizer is reinitialized in every phase
        (implicit mode) or by phase ``depth`` itself (it is rebuilt from phase 0 either way), when no such
        reinitialization exists, or when the best checkpoint's optimizer has the class of the one in use (its state is
        restored, as before).
        """
        schedule = self.ft_schedule
        if not isinstance(schedule, dict) or not self.restore_best or self.reinit_optim_cfg or \
                schedule[depth].get("new_optimizer"):
            return None
        best = self._fts_state._best_ckpt_depth
        later = [d for d in schedule if best < d < depth and schedule[d].get("new_optimizer")]
        if not later:
            return None
        earlier = [d for d in schedule if d <= best and schedule[d].get("new_optimizer")]
        best_cls = _optimizer_class_name(schedule[max(earlier)]["new_optimizer"]) if earlier else \
            self._initial_optimizer_cls
        if best_cls is None or best_cls == type(self.trainer.optimizers[0]).__name__:
            return None
        return schedule[max(later)]["new_optimizer"]

    def _maybe_allow_incompatible_reinit_ckpt(self, key_error: KeyError) -> None:
        """Inspect context for a given ``KeyError`` and permit continued training if using lr scheduler or
        optimizer reinitialization.

        Args:
            key_error (KeyError): The current key error encountered during checkpoint restoration.

        Raises:
            key_error: If not training in the context of lr scheduler or optimizer reinitialization, the provided
                ``KeyError`` will be raised.
        """
        if self._has_reinit_schedule:
            rank_zero_warn(
                "Incompatible checkpoint detected when attempting to restore the optimizer and/or lr "
                "scheduler from a previous phase. Attempting to proceed with next phase of training since this "
                "schedule reinitializes the optimizer and/or lr scheduler.\n"
                "HINT: If subsequent errors are encountered, you can either set ``restore_best`` to ``False`` "
                "or alter your reinitialization schedule for the relevant training components (i.e. optimizer, "
                "lr scheduler)."
            )
        else:
            raise key_error

    def _check_sync_dist(self, monitor: str) -> bool:
        """Inspect the monitored metric and execution context to determine whether the current monitored metric is
        a tensor in a distributed context that isn't being synced.

        Returns:
            bool: Whether whether the monitor metric is a tensor in a distributed context that isn't being synced.
        """
        if self.trainer._results is None:
            return False
        monitor_metric = [m for m in self.trainer._results.result_metrics if m.meta.name == monitor]
        if len(monitor_metric) > 0:
            assert isinstance(monitor_metric[0], _ResultMetric)
            monitor_metric_syncing = getattr(monitor_metric[0].meta.sync, "should", False)
            no_sync = all((torch.distributed.is_available(), torch.distributed.is_initialized(),
                        monitor_metric[0].is_tensor)) and not monitor_metric_syncing
            return no_sync
        return False  # couldn't find the monitor metric in the current context, so by definition ``False``

    def _reduce_transition(self, strategy: Strategy, decision: bool) -> bool:
        """Reduce a transition decision across all world processes (effectively a global `any` collective)

        Args:
            strategy (Strategy): The PL :py:class:`~lightning.pytorch.strategies.Strategy` context to use.
            decision (bool): The local process decision.

        Returns:
            bool: The reduced decision across all world processes.
        """
        decision_tensor = torch.tensor(int(decision), device=strategy.root_device)
        reduced_decision = bool(strategy.reduce(decision_tensor, reduce_op=ReduceOp.SUM))  # type:ignore[arg-type]
        return reduced_decision

    def _sync_es_state(self, trainer: "pl.Trainer") -> None:
        """Synchronize the :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` callback transition state
        across distributed processes to avoid rare transition divergences when the user does not set ``sync_dist``
        to ``True`` in logging the monitored metric.

        Args:
            trainer (:py:class:`~lightning.pytorch.trainer.trainer.Trainer`): The
                :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object
        """
        early_stopping_callback = trainer.early_stopping_callback
        assert early_stopping_callback is not None
        # reset the FTSEarlyStopping state in every distributed process if any world process is ready to transition to
        # the next fine-tuning phase
        should_reset_es = early_stopping_callback.es_phase_complete
        should_reset_es = self._reduce_transition(trainer.strategy, should_reset_es)
        if should_reset_es != early_stopping_callback.es_phase_complete:
            warn_msg = (
                "The FTSEarlyStopping quantity you are monitoring is not being synchronized across processes and FTS"
                " has detected that two or more world processes are disagreeing on whether to continue the current"
                " training phase. Training is continuing by transitioning all training processes to the next"
                " fine-tuning phase when the early stopping conditions of any training process are met. To avoid this"
                " behavior in the future, you may want to either log the monitored metric with ``sync_dist`` set to"
                " ``True`` or increase the configured FTSEarlyStopping ``min_delta``."
            )
            rank_zero_warn(warn_msg)
            early_stopping_callback._transition_es_phase()

    def _strategy_setup(self, trainer: "pl.Trainer") -> None:
        """Validate a compatible :py:class:`~lightning.pytorch.strategies.Strategy` strategy is being used and
        connects the relevant :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`.

        Args:
            trainer (:py:class:`~lightning.pytorch.trainer.trainer.Trainer`): The
                :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object

        Raises:
            MisconfigurationException: If the
                :py:class:`~lightning.pytorch.strategies.Strategy` strategy being used is not currently
                compatible with the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback.
        """
        strategy = trainer.strategy
        connect_flg = getattr(trainer._accelerator_connector, "_strategy_flag", "")
        strategy_flag = getattr(connect_flg, "strategy_name", connect_flg.__class__.__name__.lower()) if \
            isinstance(connect_flg, Strategy) else connect_flg
        _discover_strategy_adapters()  # Discover strategy adapter plugins lazily
        supported = [t.lower() for t in self._supported_strategy_flags()]
        if strategy_flag and strategy_flag not in supported:  # type: ignore[attr-defined]
            if not self.allow_untested:
                raise MisconfigurationException(
                    "FTS is has not yet been adapted for or rigorously tested using the specified distributed strategy."
                    f" Please select from currently compatible distributed strategies ({supported}) or if you would"
                    " like to attempt to use the currently specified strategy, pass ``allow_untested=True`` to the"
                    " FinetuningScheduler callback when adding it."
                )
            else:
                warn_msg = (
                    "Allowing untested strategy"
                    f" '{strategy}' because ``allow_untested`` is ``True``."  # type: ignore[attr-defined]
                )
                rank_zero_warn(warn_msg)
        if self.custom_strategy_adapters:
            strategy_cls = self._resolve_strategy_adapter(strategy_flag, self.custom_strategy_adapters)
            rank_zero_info(
                f"Imported custom strategy adapter class type `{strategy_cls}` associated with the current strategy"
                f" `{strategy_flag}`."
            )
        else:
            strategy_cls = STRATEGY_ADAPTERS.get(strategy_flag, StrategyAdapter)
        self.strategy_adapter = strategy_cls(**self.strategy_adapter_cfg)
        self.strategy_adapter.connect(self)
        strategy_cls.notice_naming_overrides()

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Validate a compatible :py:class:`~lightning.pytorch.strategies.Strategy` strategy is being used and
        ensure all :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback dependencies are met. If a valid
        configuration is present, then either dump the default fine-tuning schedule OR
        1. configure the :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping`
        callback (if relevant)
        2. initialize the :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state`
        3. freeze the target :py:class:`~lightning.pytorch.core.module.LightningModule` parameters
        Finally, initialize the :class:`~finetuning_scheduler.fts.FinetuningScheduler`
        training session in the training environment.

        Args:
            trainer (:py:class:`~lightning.pytorch.trainer.trainer.Trainer`): The
                :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object
            pl_module (:py:class:`~lightning.pytorch.core.module.LightningModule`): The
                :py:class:`~lightning.pytorch.core.module.LightningModule` object
            stage: The ``RunningStage.{SANITY_CHECKING,TRAINING,VALIDATING}``. Defaults to None.

        Raises:
            SystemExit: Gracefully exit before training if only generating and not executing a fine-tuning schedule.
        """
        # TODO: might not be necessary if we move introspection back to IT where it belongs
        # Save pl_module/trainer refs early to allow strategy adapter selection to introspect the module
        self.pl_module, self.trainer = pl_module, trainer  # save pl_module/trainer refs for downstream convenience
        self._fts_controller = None  # attached afresh for each fit
        self._callback_dep_setup(trainer, pl_module, stage)
        self._strategy_setup(trainer)
        if self.gen_ft_sched_only:
            if trainer.is_global_zero:
                assert self.log_dir is not None, "log_dir must be set to generate a fine-tuning schedule"
                _ = self.strategy_adapter.gen_ft_schedule(self.log_dir)
                log.info("Bypassing training, generating fine-tuning schedule for review and subsequent fine-tuning")
            raise SystemExit(0)
        if not self.epoch_transitions_only:
            assert isinstance(trainer.early_stopping_callback, FTSEarlyStopping)
            trainer.early_stopping_callback.final_phase = False
            trainer.early_stopping_callback.es_phase_complete = False
        self._fts_state._ft_sync_objects = self.trainer.fit_loop, self._fts_state
        self.freeze_before_training(self.pl_module)
        self.init_fts()

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           checkpoint: dict[str, Any]) -> None:
        """Mark a fit as resuming from a checkpoint when Lightning restores the callbacks from one.

        Lightning calls this before any callback's ``load_state_dict``, and only once it is restoring a checkpoint, so a
        resume is detected however its path was given (``trainer.fit(model, ckpt_path=...)`` or ``trainer.ckpt_path``):
        ``setup`` runs too early to see a path passed to ``fit``.

        Args:
            trainer (:py:class:`~lightning.pytorch.trainer.trainer.Trainer`): The
                :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object
            pl_module (:py:class:`~lightning.pytorch.core.module.LightningModule`): The
                :py:class:`~lightning.pytorch.core.module.LightningModule` object
            checkpoint: The checkpoint being restored.
        """
        if trainer.state.fn == TrainerFn.FITTING:
            self._fts_state._resume_fit_from_ckpt = True

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Before beginning training, ensure an optimizer configuration supported by
        :class:`~finetuning_scheduler.fts.FinetuningScheduler` is present.

        Args:
            trainer (:py:class:`~lightning.pytorch.trainer.trainer.Trainer`): The
                :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object
            pl_module (:py:class:`~lightning.pytorch.core.module.LightningModule`): The
                :py:class:`~lightning.pytorch.core.module.LightningModule` object

        Raises:
            MisconfigurationException: If more than 1 optimizers are configured indicates a configuration error
        """
        self.strategy_adapter.on_before_fts_fit_start()
        if len(trainer.optimizers) > 1:
            raise MisconfigurationException("FTS currently only supports a single-optimizer configuration")
        # the class `configure_optimizers` built, which checkpoints before any optimizer reinitialization hold
        self._initial_optimizer_cls = type(trainer.optimizers[0]).__name__
        if getattr(trainer.optimizers[0], "_overlap_with_ddp", False):
            raise MisconfigurationException(
                "Configuring an optimizer using `overlap_with_ddp=True` is not supported"
                " with FTS since that optimizer mode only supports a single parameter"
                " group."
            )
        if trainer.lr_scheduler_configs:
            self._is_supported_lr(type(trainer.lr_scheduler_configs[0].scheduler))
        if self.curr_depth == 0:
            assert isinstance(self.ft_schedule, dict)
            self._validate_opt_init()
        super().on_fit_start(trainer, pl_module)
        if self._use_fts_controller:
            from finetuning_scheduler.lightning.controller_bridge import attach_controller

            # phase 0's start hooks run now, unless the fit resumes a run (the controller resumes at its phase)
            self._fts_controller = attach_controller(self, self._phase_hooks,
                                                     run_start_hooks=not self._fts_state._resume_fit_from_ckpt)

    def add_phase_hook(self, fn: Callable, when: str = "start", *, isolate_rng: bool | None = None) -> None:
        """Call ``fn(event)`` whenever a phase starts (phase 0 at fit start) or ends (the final phase at train
        end), as :meth:`~finetuning_scheduler.core.controller.FTSController.add_phase_hook` does: ``event`` is a
        :class:`~finetuning_scheduler.core.controller.PhaseEvent`, whose ``phase_config`` is the phase's
        ``phase_config`` schedule key. Hooks run in registration order, for every later fit with this callback
        (experimental).

        Args:
            fn: The hook.
            when: ``"start"`` or ``"end"``.
            isolate_rng: Whether to isolate the training random streams while ``fn`` runs (the default is to).

        Raises:
            MisconfigurationException: If ``when`` is neither ``"start"`` nor ``"end"``, or the callback's own
                transitions are in use (``_use_fts_controller``).
        """
        if when not in ("start", "end"):
            raise MisconfigurationException(f"`when` must be 'start' or 'end', got {when!r}.")
        if not self._use_fts_controller:
            raise MisconfigurationException("Phase hooks need the callback's controller (`_use_fts_controller`).")
        self._phase_hooks.append((fn, when, isolate_rng))

    def state_dict(self) -> dict[str, Any]:
        """Before saving a checkpoint, add the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback
        state to be saved.

        Returns:
            dict[str, Any]: The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback state dictionary
                that will be added to the checkpoint
        """
        # callback state such as `_ft_init_epoch` does not currently need to be persisted
        assert self.pl_module is not None and self.trainer is not None
        trainer = self.trainer
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback._should_update_depth_meta:  # type: ignore [union-attr]
            self._fts_state._best_ckpt_depth = self._fts_state._curr_depth
            for opt_idx, _ in enumerate(trainer.optimizers):
                self._fts_state._fts_ckpt_metadata["best_ckpt_pgs"][opt_idx] = deepcopy(
                    self._internal_optimizer_metadata[opt_idx]
                )
        self._fts_state._fts_ckpt_metadata["current_ckpt_depth"] = self._fts_state._curr_depth
        self._fts_state._fts_ckpt_metadata["best_ckpt_depth"] = self._fts_state._best_ckpt_depth
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "fts_metadata": self._fts_state._fts_ckpt_metadata,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """After loading a checkpoint, load the saved :class:`~finetuning_scheduler.fts.FinetuningScheduler`
        callback state and update the current callback state accordingly.

        Args:
            state_dict: The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback state dictionary that will
                be loaded from the checkpoint
        """
        self._restarting = True
        self._internal_optimizer_metadata = state_dict["internal_optimizer_metadata"]
        self._fts_state._fts_ckpt_metadata = state_dict["fts_metadata"]
        if self._fts_state._resume_fit_from_ckpt:  # if resuming training, on_fit_start will already be called
            # if resuming from a checkpoint, we need to update current fts depth from the used ckpt
            self._fts_state._curr_depth = self._fts_state._fts_ckpt_metadata["current_ckpt_depth"]
            # the checkpoint callback reloads the run's best checkpoint, so the phases to replay onto its restored
            # optimizer state at the next transition start after the depth it was saved at (`best_ckpt_pgs`' depth)
            self._fts_state._best_ckpt_depth = self._fts_state._fts_ckpt_metadata["best_ckpt_depth"]

    def should_transition(self, trainer: "pl.Trainer") -> bool:
        """Phase transition logic is contingent on whether we are composing
        :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria with epoch-driven transition
        constraints or exclusively using epoch-driven transition scheduling. (i.e.,
        :attr:`~finetuning_scheduler.fts.FinetuningScheduler.epoch_transitions_only` is ``True``)

        Args:
            trainer (:py:class:`~lightning.pytorch.trainer.trainer.Trainer`): The
                :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object
        """
        assert self.pl_module is not None
        assert isinstance(self.ft_schedule, dict)
        early_stopping_callback = trainer.early_stopping_callback
        curr_max_epoch = core_transitions.current_max_epoch(
            self.ft_schedule, self.curr_depth, self.depth_remaining, trainer.fit_loop.max_epochs  # type: ignore[arg-type]
        )
        assert isinstance(curr_max_epoch, (int, float)), (
            f"Expected max_transition_epoch/max_epochs to be an int or float, but got {type(curr_max_epoch)}"
        )
        epoch_due = core_transitions.epoch_transition_due(curr_max_epoch, trainer.current_epoch)
        if not self.epoch_transitions_only:  # if we're considering FTSEarlyStopping criteria
            assert early_stopping_callback is not None
            # in the edge case where transition decisions diverge among distributed processes because the user is
            # running in a distributed context without ``sync_dist`` set to ``True`` and ``min_delta`` is
            # sufficiently low, we should reduce the transition decision over all training processes to avoid deadlocks
            if early_stopping_callback.reduce_transition_decisions:
                self._sync_es_state(trainer)
            phase_transition = core_transitions.compose_transition(
                early_stopping_callback.es_phase_complete,  # type: ignore[attr-defined]
                early_stopping_callback.final_phase,  # type: ignore[attr-defined]
                epoch_due,
            )
        else:  # we're only considering epoch-driven transition constraints
            phase_transition = epoch_due
        return phase_transition

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Before beginning a training epoch, configure the internal
        :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state`, prepare the next scheduled fine-tuning
        level and store the updated optimizer configuration before continuing training.

        Args:
            trainer (:py:class:`~lightning.pytorch.trainer.trainer.Trainer`): The
                :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object
            pl_module (:py:class:`~lightning.pytorch.core.module.LightningModule`): The
                :py:class:`~lightning.pytorch.core.module.LightningModule` object
        """
        # if resuming from a ckpt, we need to sync fts_state
        if self._fts_state._resume_fit_from_ckpt:
            # refuse a best checkpoint this strategy cannot restore now, rather than at the next phase transition (a
            # fit that never restores it, without `restore_best` or resumed at its last phase, is left to proceed)
            best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
            if best_model_path and self.restore_best and self.depth_remaining > 0:
                self.strategy_adapter.best_ckpt_path(best_model_path)
            self.step()
            self._fts_state._resume_fit_from_ckpt = False
        # increment ft_epoch on each train epoch
        assert isinstance(self._fts_state._ft_init_epoch, int)
        if trainer.current_epoch > self._fts_state._ft_init_epoch:
            assert self._fts_state._ft_sync_objects is not None
            self.sync(self._fts_state._ft_sync_objects, self._fts_state._ft_sync_props)
        if self.should_transition(trainer):
            self._fts_state._curr_depth += 1  # increment depth
            self.step()
            rank_zero_debug(
                f"Current depth is {self.curr_depth}."
                "\nCurrent logical parameters thawed by Fine-Tuning Scheduler:\n "
                f"{pformat(self.strategy_adapter.logical_param_translation(self._fts_state._curr_thawed_params))}."
                "\nCurrent actual parameters thawed by Fine-Tuning Scheduler:\n"
                f"{pformat(self._fts_state._curr_thawed_params)}. "
            )
            if not self.epoch_transitions_only:
                assert isinstance(trainer.early_stopping_callback, FTSEarlyStopping)
                trainer.early_stopping_callback._reset_es_phase()
        if self.depth_remaining == 0:
            if not self.epoch_transitions_only:
                assert isinstance(trainer.early_stopping_callback, FTSEarlyStopping)
                trainer.early_stopping_callback.final_phase = True
        # capture optimizer config for all optimizers (though initially we'll only support a single optimizer)
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            num_saved_groups = (
                len(self._internal_optimizer_metadata[opt_idx]) if opt_idx in self._internal_optimizer_metadata else 0
            )
            current_param_groups = optimizer.param_groups
            self._store(pl_module, opt_idx, num_saved_groups, current_param_groups)

    def on_before_zero_grad(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: ParamGroupAddable,  # type: ignore[override]
    ) -> None:
        """Afer the latest optimizer step, update the
        :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state`, incrementing the global fine-tuning steps
        taken.

        Args:
            trainer (:py:class:`~lightning.pytorch.trainer.trainer.Trainer`): The
                :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object
            pl_module  (:py:class:`~lightning.pytorch.core.module.LightningModule`): The
                :py:class:`~lightning.pytorch.core.module.LightningModule` object
            optimizer (:class:`~finetuning_scheduler.types.ParamGroupAddable`): The supported optimizer instance to
                which parameter groups will be configured and added.
        """
        self._fts_state._ft_global_steps += 1

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Synchronize internal :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state` on end of training
        to ensure final training state is consistent with epoch semantics.

        Args:
            trainer: The :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object
            pl_module  (:py:class:`~lightning.pytorch.core.module.LightningModule`): The
                :py:class:`~lightning.pytorch.core.module.LightningModule` object
        """
        assert self._fts_state._ft_sync_objects is not None
        self.sync(self._fts_state._ft_sync_objects, self._fts_state._ft_sync_props)
        if self._fts_controller is not None:
            self._fts_controller.train_end()  # the final phase's end hooks

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Let the strategy adapter undo what it changed beyond this fit (see
        :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.on_fts_teardown`)."""
        super().teardown(trainer, pl_module, stage)
        self._fts_adapter_teardown()

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """As :meth:`teardown`, for a fit that raised."""
        super().on_exception(trainer, pl_module, exception)
        self._fts_adapter_teardown()

    def _fts_adapter_teardown(self) -> None:
        adapter = getattr(self, "strategy_adapter", None)
        if adapter is not None:
            adapter.on_fts_teardown()
