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
Fine-Tuning Scheduler
^^^^^^^^^^^^^^^^^^^^^

Used to implement flexible fine-tuning training schedules

"""
import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import rank_zero_info
from lightning_fabric.utilities.distributed import ReduceOp
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_warn
from torch.optim.optimizer import Optimizer

from finetuning_scheduler.fts_supporters import (
    CallbackDepMixin,
    FTSEarlyStopping,
    FTSState,
    ScheduleImplMixin,
    ScheduleParsingMixin,
    STRATEGY_ADAPTERS,
)
from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

log = logging.getLogger(__name__)


class FinetuningScheduler(ScheduleImplMixin, ScheduleParsingMixin, CallbackDepMixin, BaseFinetuning):
    r"""
    This callback enables flexible, multi-phase, scheduled fine-tuning of foundation models. Gradual
    unfreezing/thawing can help maximize foundation model knowledge retention while allowing (typically upper layers
    of) the model to optimally adapt to new tasks during transfer learning.
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` orchestrates the gradual unfreezing of models via a
    fine-tuning schedule that is either implicitly generated (the default) or explicitly provided by the user (more
    computationally efficient).

    Fine-tuning phase transitions are driven by
    :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria (a multi-phase
    extension of :external+pl:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`), user-specified epoch
    transitions or a composition of the two (the default mode). A
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` training session completes
    when the final phase of the schedule has its stopping criteria met. See
    :ref:`Early Stopping<common/early_stopping:Early stopping>` for more details on that callback's configuration.

    Schedule definition is facilitated via
    :meth:`~finetuning_scheduler.fts_supporters.ScheduleImplMixin.gen_ft_schedule` which dumps
    a default fine-tuning schedule (by default using a naive, 2-parameters per level heuristic) which can be adjusted as
    desired by the user and subsuquently passed to the callback. Implicit fine-tuning mode generates the default
    schedule and proceeds to fine-tune according to the generated schedule. Implicit fine-tuning will often be less
    computationally efficient than explicit fine-tuning but can often serve as a good baseline for subsquent explicit
    schedule refinement and can marginally outperform many explicit schedules.

    Example::

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import FinetuningScheduler
        trainer = Trainer(callbacks=[FinetuningScheduler()])

    .. note::

        Currently, :class:`~finetuning_scheduler.fts.FinetuningScheduler` does not support the use of multiple
        :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` or
        :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` callback instances.

    .. note::

       While :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports the use of
       :external+torch:class:`~torch.distributed.optim.ZeroRedundancyOptimizer`, setting ``overlap_with_ddp`` to
       ``True`` is not supported because that optimizer mode only supports a single parameter group.
    """
    pl_module: pl.LightningModule
    strategy_adapter: StrategyAdapter

    def __init__(
        self,
        ft_schedule: Optional[Union[str, dict]] = None,
        max_depth: int = -1,
        base_max_lr: float = 1e-5,
        restore_best: bool = True,
        gen_ft_sched_only: bool = False,
        epoch_transitions_only: bool = False,
        reinit_lr_cfg: Optional[Dict] = None,
        strategy_adapter_cfg: Optional[Dict] = None,
        custom_strategy_adapter: Optional[Dict[str, str]] = None,
        allow_untested: bool = False,
        apply_lambdas_new_pgs: bool = False,
        logging_level: int = logging.INFO,
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
                :external+pl:class:`~pytorch_lightning.core.module.LightningModule`. See
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
            gen_ft_sched_only: If ``True``, generate the default fine-tuning schedule to ``Trainer.log_dir`` (it will be
                named after your :external+pl:class:`~pytorch_lightning.core.module.LightningModule` subclass with
                the suffix ``_ft_schedule.yaml``) and exit without training. Typically used to generate a default
                schedule that will be adjusted by the user before training. Defaults to ``False``.
            epoch_transitions_only: If ``True``, use epoch-driven stopping criteria exclusively (rather than composing
                :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` and
                epoch-driven criteria which is the default). If using this mode, an epoch-driven transition
                (``max_transition_epoch`` >= 0) must be specified for each phase. If unspecified,
                ``max_transition_epoch`` defaults to -1 for each phase which signals the application of
                :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria only.
                epoch_transitions_only defaults to ``False``.
            reinit_lr_cfg: A lr scheduler reinitialization configuration dictionary consisting of at minimum a nested
                ``lr_scheduler_init`` dictionary with a ``class_path`` key specifying the class of the lr scheduler
                to be instantiated. Optionally, an ``init_args`` dictionary of arguments to initialize the lr scheduler
                with may be included. Additionally, one may optionally include arguments to pass to PyTorch Lightning's
                lr scheduler configuration :class:`~pytorch_lightning.utilities.types.LRSchedulerConfig` in the
                ``pl_lrs_cfg`` dictionary. By way of example, one could configure this dictionary via the
                :external+pl:class:`~pytorch_lightning.cli.LightningCLI` with the following:

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

                    Regarding lr schedulers, :external+torch:class:`~torch.optim.lr_scheduler.ChainedScheduler` and
                    :external+torch:class:`~torch.optim.lr_scheduler.SequentialLR` are examples of schedulers not
                    currently supported due to the configuration complexity and semantic conflicts supporting them would
                    introduce. If a supported torch lr scheduler does not meet your requirements, one can always
                    subclass a supported lr scheduler and modify it as required
                    (e.g. :external+torch:class:`~torch.optim.lr_scheduler.LambdaLR` is especially useful for this).
            strategy_adapter_cfg: A configuration dictionary that will be applied to the
                :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` associated with the current training
                :external+pl:class:`~pytorch_lightning.strategies.Strategy`. See the relevant
                :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` documentation for strategy-specific
                configuration options. Defaults to None.
            custom_strategy_adapter: A dictionary associating the canonical ``strategy_name`` of a
                :external+pl:class:`~pytorch_lightning.strategies.Strategy` (potentially a custom user-registered one)
                to the fully qualified path of a
                :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` subclass. This is an experimental
                feature that is subject to change. Requires ``allow_untested`` to be set to ``True``. Defaults to None.
            apply_lambdas_new_pgs: If ``True``, applies most recent lambda in ``lr_lambdas`` list to newly added
                optimizer groups for lr schedulers that have a ``lr_lambdas`` attribute. Note this option only applies
                to phases without reinitialized lr schedulers. Phases with defined lr scheduler reinitialization configs
                will always apply the specified lambdas. Defaults to ``False``.
            logging_level: Sets the logging level for :class:`~finetuning_scheduler.fts.FinetuningScheduler`. Defaults
                to ``INFO``.

        Attributes:
            _fts_state: The internal :class:`~finetuning_scheduler.fts.FinetuningScheduler` state.
            strategy_adapter_cfg: A configuration dictionary that will be applied to the
                :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` associated with the current training
                :external+pl:class:`~pytorch_lightning.strategies.Strategy`.
            epoch_transitions_only: Whether to use epoch-driven stopping criteria exclusively.
            base_max_lr: The default maximum learning rate to use for the parameter groups associated with each
                scheduled fine-tuning depth if not explicitly specified in the fine-tuning schedule. If overridden to
                ``None``, will be set to the ``lr`` of the first scheduled fine-tuning depth. Defaults to 1e-5.
        """
        super().__init__()
        self._fts_state = FTSState()
        self.max_depth = max_depth
        self.restore_best = restore_best
        self.ft_schedule = ft_schedule
        self.base_max_lr = base_max_lr
        self.gen_ft_sched_only = gen_ft_sched_only
        self.epoch_transitions_only = epoch_transitions_only
        self.reinit_lr_cfg = reinit_lr_cfg
        self.strategy_adapter_cfg = strategy_adapter_cfg or {}
        self.custom_strategy_adapter = custom_strategy_adapter
        self.allow_untested = allow_untested
        self.apply_lambdas_new_pgs = apply_lambdas_new_pgs
        rz_logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
        rz_logger.setLevel(logging_level)

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
        return max(self.max_depth - self._fts_state._curr_depth, 0)

    @staticmethod
    def _supported_strategy_types() -> Sequence[str]:
        return (
            "dp",
            "ddp",
            "ddp_find_unused_parameters_false",
            "ddp_spawn",
            "ddp_fork",
            "ddp_notebook",
            "single_device",
            # TODO: `native` suffix will be removed from `fsdp` strategies in 2.0
            "fsdp_native",
            "fsdp_native_full_shard_offload",
            "ddp_sharded",  # TODO: remove in 2.0
            "ddp_sharded_spawn"  # TODO: remove in 2.0,
            # "deepspeed",  # relevant FTS strategy adapter not yet available, PRs welcome!
        )

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        """Freezes all model parameters so that parameter subsets can be subsequently thawed according to the fine-
        tuning schedule.

        Args:
            pl_module (:external+pl:class:`~pytorch_lightning.core.module.LightningModule`): The target
                :external+pl:class:`~pytorch_lightning.core.module.LightningModule` to freeze parameters of
        """
        self.freeze(modules=pl_module, train_bn=False)

    def step(self) -> None:
        """Prepare and execute the next scheduled fine-tuning level
        1. Restore the current best model checkpoint if appropriate
        2. Thaw model parameters according the the defined schedule
        3. Synchronize the states of ``FitLoop`` and :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state`

        .. note::

            The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback initially
            only supports single-schedule/optimizer fine-tuning configurations
        """
        assert self.pl_module.trainer is not None
        if not self._fts_state._resume_fit_from_ckpt:
            if self.restore_best:
                self.restore_best_ckpt()
                self.step_pg(depth=self.curr_depth, optimizer=self.pl_module.trainer.optimizers[0])
            else:
                self.step_pg(depth=self.curr_depth, optimizer=self.pl_module.trainer.optimizers[0], depth_sync=False)
        else:
            self.thaw_to_depth()
        if self.depth_remaining == 0 and not self.epoch_transitions_only:
            assert self.pl_module.trainer.early_stopping_callback is not None
            self.pl_module.trainer.early_stopping_callback.final_phase = True  # type: ignore[attr-defined]
        assert self._fts_state._ft_sync_objects is not None
        FinetuningScheduler.sync(self._fts_state._ft_sync_objects, self._fts_state._ft_sync_props)
        rank_zero_info(f"Multi-phase fine-tuned training continuing at level {self.curr_depth}.")
        if self.depth_remaining == 0:
            max_epochs_msg = f"`max_epochs` ({self.pl_module.trainer.fit_loop.max_epochs}) is reached."
            composition_msg = "the early stopping conditions are met or " + max_epochs_msg
            rank_zero_info(
                f"Given the current configuration of `max_depth` ({self.max_depth}), this training session"
                f" will now end when {max_epochs_msg if self.epoch_transitions_only else composition_msg}"
            )

    def step_pg(self, optimizer: Optimizer, depth: int, depth_sync: bool = True) -> None:
        """Configure optimizer parameter groups for the next scheduled fine-tuning level, adding parameter groups
        beyond the restored optimizer state up to
        :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.current_depth`

        Args:
            optimizer (:external+torch:class:`~torch.optim.Optimizer`): The
                :external+torch:class:`~torch.optim.Optimizer` to which parameter groups will be configured and added.
            depth: The maximum index of the fine-tuning schedule for which to configure the optimizer parameter
                groups.
            depth_sync: If ``True``, configure optimizer parameter groups for all depth indices greater
                than the restored checkpoint. If ``False``, configure groups only for the specified depth. Defaults to
                ``True``.
        """
        next_tl: Dict = {}
        assert isinstance(self.ft_schedule, dict)
        assert isinstance(self.pl_module, pl.LightningModule)
        assert isinstance(self.pl_module.trainer, pl.Trainer)
        if depth_sync:
            thaw_layers = {d: l for d, l in self.ft_schedule.items() if d > self._fts_state._best_ckpt_depth}.items()
        else:
            thaw_layers = {depth: self.ft_schedule[depth]}.items()
        for i, orig_next_tl in thaw_layers:
            next_tl = deepcopy(orig_next_tl)
            next_tl["params"] = self.strategy_adapter.fts_optim_view(next_tl["params"])
            if i <= depth:
                _, self._fts_state._curr_thawed_params = self.strategy_adapter.exec_ft_phase(
                    self.pl_module, thaw_pl=next_tl["params"]
                )
                FinetuningScheduler.add_optimizer_groups(
                    module=self.pl_module,
                    optimizer=optimizer,
                    thawed_pl=next_tl["params"],
                    lr=next_tl["lr"],
                    no_decay=getattr(self.pl_module, "no_decay", None),
                    apply_lambdas=self.apply_lambdas_new_pgs,
                )
                new_scheduler_cfg = self.reinit_lr_cfg or next_tl.get("new_lr_scheduler", None)
                if new_scheduler_cfg:
                    self.reinit_lr_scheduler(
                        new_lr_scheduler=new_scheduler_cfg, trainer=self.pl_module.trainer, optimizer=optimizer
                    )
                else:
                    if self.pl_module.trainer.lr_scheduler_configs:
                        for config in self.pl_module.trainer.lr_scheduler_configs:
                            show_warn_lambdas = (
                                hasattr(config.scheduler, "lr_lambdas")
                                and config.scheduler.lr_lambdas[-1] is not None  # type: ignore[union-attr]
                                and not self.apply_lambdas_new_pgs
                            )
                            if show_warn_lambdas:
                                rank_zero_warn(
                                    "The lr scheduler used in this phase has lr_lambdas but will use a "
                                    "configured lr for new parameter groups because `apply_lambdas_new_pgs` is "
                                    "set to the default of `False`. If you would like new groups to have lr "
                                    "lambdas applied, set `apply_lambdas_new_pgs` to `True`."
                                )

    def restore_best_ckpt(self) -> None:
        """Restore the current best model checkpoint, according to
        :paramref:`~finetuning_scheduler.fts_supporters.FTSCheckpoint.best_model_path`"""
        assert self.pl_module.trainer is not None
        # wait for all processes to be ready to restore ckpt before restoring
        self.pl_module.trainer.strategy.barrier("setup_next_level")
        # if restarting across multiple depths, need to ensure we're restoring optimizer state appropriately
        # by resetting optimizer groups and allowing state dict to be reset commensurate w/ ckpt state
        for opt_idx, optimizer in enumerate(self.pl_module.trainer.optimizers):
            optimizer.param_groups = BaseFinetuning._apply_mapping_to_param_groups(
                self._fts_state._fts_ckpt_metadata["best_ckpt_pgs"][opt_idx], dict(self.pl_module.named_parameters())
            )
            if hasattr(optimizer, "consolidate_state_dict"):
                # for optimizers that shard their states like (e.g. ZeroRedundancyOptimizer, OSS), we need to clear
                # local optimizer partition caches and repartition to support restoring across multiple depths
                partition_params = (
                    optimizer._partition_parameters
                    if callable(optimizer._partition_parameters)
                    else optimizer.partition_parameters
                )
                optimizer._clear_cache()
                optimizer.optim.param_groups = partition_params()[optimizer.rank]
                optimizer._sync_param_groups(optimizer.optim.param_groups, optimizer.param_groups)
        # we're restoring everything but callbacks and loops, otherwise, checkpoint_connector.restore() could be used
        assert self.pl_module.trainer.checkpoint_callback is not None
        checkpoint_path = self.pl_module.trainer.checkpoint_callback.best_model_path  # type: ignore[attr-defined]
        self.pl_module.trainer._checkpoint_connector.resume_start(checkpoint_path=checkpoint_path)
        self.pl_module.trainer._checkpoint_connector.restore_datamodule()
        self.pl_module.trainer._checkpoint_connector.restore_model()
        # we need to override checkpoint_connector.restore_training_state() to bypass loop restoration
        # if additional customizations are required, may make sense to subclass CheckpointConnector at some point
        self._restore_training_state()
        self.pl_module.trainer._checkpoint_connector.resume_end()

    def _restore_training_state(self) -> None:
        """Restore training state without restoring loops from the pre-loaded checkpoint.

        This includes the precision settings, optimizer states and learning rate scheduler states.
        """
        assert self.pl_module is not None and self.pl_module.trainer is not None
        checkpoint_connector = self.pl_module.trainer._checkpoint_connector

        # restore precision plugin (scaler etc.)
        checkpoint_connector.restore_precision_plugin_state()

        # checkpoint_connector.restore_training_state() would restore loops here
        # self.restore_loops()

        assert self.pl_module.trainer.state.fn is not None
        if self.pl_module.trainer.state.fn == TrainerFn.FITTING:
            try:
                # enable strategy adapters to restore optimizer if `Strategy.lightning_restore_optimizer` is overridden
                self.strategy_adapter.on_before_restore_optimizers_and_lrs()
                # restore optimizers and schedulers state
                checkpoint_connector.restore_optimizers_and_schedulers()
            except KeyError:
                assert isinstance(self.ft_schedule, dict)
                if self.ft_schedule[self.curr_depth].get("new_lr_scheduler", None):
                    rank_zero_warn(
                        "incompatible checkpoint detected but attempting to proceed with next phase of training since "
                        "we're reinitializing the lr scheduler."
                    )

    def _reduce_transition(self, strategy: Strategy, decision: bool) -> bool:
        """Reduce a transition decision across all world processes (effectively a global `any` collective)

        Args:
            strategy (Strategy): The PL :external+pl:class:`~pytorch_lightning.strategies.Strategy` context to use.
            decision (bool): The local process decision.

        Returns:
            bool: The reduced decision across all world processes.
        """
        decision = torch.tensor(int(decision), device=strategy.root_device)
        decision = bool(strategy.reduce(decision, reduce_op=ReduceOp.SUM))
        return decision

    def _sync_es_state(self, trainer: "pl.Trainer") -> None:
        """Synchronize the :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` callback transition state
        across distributed processes to avoid rare transition divergences when the user does not set ``sync_dist``
        to ``True`` in logging the monitored metric.

        Args:
            trainer (:external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer`): The
                :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object
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
        """Validate a compatible :external+pl:class:`~pytorch_lightning.strategies.Strategy` strategy is being used
        and connects the relevant :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`.

        Args:
            trainer (:external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer`): The
                :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object

        Raises:
            MisconfigurationException: If the
                :external+pl:class:`~pytorch_lightning.strategies.Strategy` strategy being used is not currently
                compatible with the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback.
        """
        strategy = trainer.strategy
        supported = [t.lower() for t in self._supported_strategy_types()]
        if strategy.strategy_name and strategy.strategy_name not in supported:  # type: ignore[attr-defined]
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
                    f" '{strategy.strategy_name}' because ``allow_untested`` is ``True``."  # type: ignore[attr-defined]
                )
                rank_zero_warn(warn_msg)
        if self.custom_strategy_adapter:
            strategy_cls = self._import_strategy_adapter(strategy.strategy_name, self.custom_strategy_adapter)
            rank_zero_info(
                f"Imported custom strategy adapter class type `{strategy_cls}` associated with the current strategy"
                f" `{strategy.strategy_name}`."
            )
        else:
            strategy_cls = STRATEGY_ADAPTERS.get(strategy.strategy_name, StrategyAdapter)
        self.strategy_adapter = strategy_cls(**self.strategy_adapter_cfg)
        self.strategy_adapter.connect(self)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Validate a compatible :external+pl:class:`~pytorch_lightning.strategies.Strategy` strategy is being used and
        ensure all :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback dependencies are met. If a valid
        configuration is present, then either dump the default fine-tuning schedule OR
        1. configure the :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping`
        callback (if relevant)
        2. initialize the :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state`
        3. freeze the target :external+pl:class:`~pytorch_lightning.core.module.LightningModule` parameters
        Finally, initialize the :class:`~finetuning_scheduler.fts.FinetuningScheduler`
        training session in the training environment.

        Args:
            trainer (:external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer`): The
                :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object
            pl_module (:external+pl:class:`~pytorch_lightning.core.module.LightningModule`): The
                :external+pl:class:`~pytorch_lightning.core.module.LightningModule` object
            stage: The ``RunningStage.{SANITY_CHECKING,TRAINING,VALIDATING}``. Defaults to None.

        Raises:
            SystemExit: Gracefully exit before training if only generating and not executing a fine-tuning schedule.
        """
        self._callback_dep_setup(trainer, pl_module, stage)
        self._strategy_setup(trainer)
        if self.gen_ft_sched_only:
            if trainer.is_global_zero:
                assert trainer.log_dir is not None
                _ = self.gen_ft_schedule(pl_module, trainer.log_dir)
                log.info("Bypassing training, generating fine-tuning schedule for review and subsequent fine-tuning")
            raise SystemExit()
        if not self.epoch_transitions_only:
            assert isinstance(trainer.early_stopping_callback, FTSEarlyStopping)
            trainer.early_stopping_callback.final_phase = False
            trainer.early_stopping_callback.es_phase_complete = False
        self._fts_state._ft_sync_objects = pl_module.trainer.fit_loop, self._fts_state
        if trainer.ckpt_path:
            self._fts_state._resume_fit_from_ckpt = True
        self.freeze_before_training(pl_module)
        self.pl_module = pl_module  # save pl_module ref for downstream configuration convenience
        self.init_fts()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Before beginning training, ensure an optimizer configuration supported by
        :class:`~finetuning_scheduler.fts.FinetuningScheduler` is present.

        Args:
            trainer (:external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer`): The
                :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object
            pl_module (:external+pl:class:`~pytorch_lightning.core.module.LightningModule`): The
                :external+pl:class:`~pytorch_lightning.core.module.LightningModule` object

        Raises:
            MisconfigurationException: If more than 1 optimizers are configured indicates a configuration error
        """
        self.strategy_adapter.on_before_fts_fit_start()
        if len(trainer.optimizers) > 1:
            raise MisconfigurationException("FTS currently only supports a single-optimizer configuration")
        if getattr(trainer.optimizers[0], "_overlap_with_ddp", False):
            raise MisconfigurationException(
                "Configuring an optimizer using `overlap_with_ddp=True` is not supported"
                " with FTS since that optimizer mode only supports a single parameter"
                " group."
            )
        if trainer.lr_scheduler_configs:
            self._is_supported_lr(type(trainer.lr_scheduler_configs[0].scheduler))
        if self.curr_depth == 0:
            assert isinstance(self.ft_schedule, Dict)
            self._validate_opt_init()
        super().on_fit_start(trainer, pl_module)

    def state_dict(self) -> Dict[str, Any]:
        """Before saving a checkpoint, add the
        :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback state to be saved.

        Returns:
            Dict[str, Any]: The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback state dictionary
                that will be added to the checkpoint
        """
        assert self.pl_module is not None and self.pl_module.trainer is not None
        trainer = self.pl_module.trainer
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback.current_score == checkpoint_callback.best_model_score:  # type: ignore[union-attr]
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

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """After loading a checkpoint, load the saved
        :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback state and update the
        current callback state accordingly.

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
            # if we're restoring from a non-best ckpt depth, ensure it is the new training incarnation's initial best
            self._fts_state._best_ckpt_depth = self._fts_state._fts_ckpt_metadata["current_ckpt_depth"]

    def should_transition(self, trainer: "pl.Trainer") -> bool:
        """Phase transition logic is contingent on whether we are composing
        :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria with
        epoch-driven transition constraints or exclusively using epoch-driven transition scheduling. (i.e.,
        :attr:`~finetuning_scheduler.fts.FinetuningScheduler.epoch_transitions_only` is
        ``True``)

        Args:
            trainer (:external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer`): The
                :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object
        """
        assert self.pl_module is not None
        assert isinstance(self.ft_schedule, Dict)
        early_stopping_callback = trainer.early_stopping_callback
        curr_max_epoch = (
            self.ft_schedule[self.curr_depth]["max_transition_epoch"]
            if self.depth_remaining > 0
            else trainer.fit_loop.max_epochs
        )
        if not self.epoch_transitions_only:  # if we're considering FTSEarlyStopping criteria
            assert early_stopping_callback is not None
            # in the edge case where transition decisions diverge among distributed processes because the user is
            # running in a distributed context without ``sync_dist`` set to ``True`` and ``min_delta`` is
            # sufficiently low, we should reduce the transition decision over all training processes to avoid deadlocks
            if early_stopping_callback.reduce_transition_decisions:
                self._sync_es_state(trainer)
            is_final_phase = early_stopping_callback.final_phase  # type: ignore[attr-defined]
            epoch_driven_transition = (
                True if not is_final_phase and (0 <= curr_max_epoch <= trainer.current_epoch) else False
            )
            if early_stopping_callback.es_phase_complete or epoch_driven_transition:  # type: ignore[attr-defined]
                phase_transition = True
            else:
                phase_transition = False
        else:  # we're only considering epoch-driven transition constraints
            phase_transition = True if 0 <= curr_max_epoch <= trainer.current_epoch else False
        return phase_transition

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Before beginning a training epoch, configure the internal
        :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state`, prepare the next
        scheduled fine-tuning level and store the updated optimizer configuration before continuing training

        Args:
            trainer (:external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer`): The
                :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object
            pl_module (:external+pl:class:`~pytorch_lightning.core.module.LightningModule`): The
                :external+pl:class:`~pytorch_lightning.core.module.LightningModule` object
        """
        # if resuming from a ckpt, we need to sync fts_state
        if self._fts_state._resume_fit_from_ckpt:
            self.step()
            self._fts_state._resume_fit_from_ckpt = False
        # increment ft_epoch on each train epoch
        if trainer.current_epoch > 0:
            assert self._fts_state._ft_sync_objects is not None
            self.sync(self._fts_state._ft_sync_objects, self._fts_state._ft_sync_props)
        if self.should_transition(trainer):
            self._fts_state._curr_depth += 1  # increment depth
            self.step()
            rank_zero_debug(
                f"Current logical parameters thawed by Fine-Tuning Scheduler: "
                f"{self.strategy_adapter.logical_param_translation(self._fts_state._curr_thawed_params)}."
                f"Current actual parameters thawed by Fine-Tuning Scheduler: {self._fts_state._curr_thawed_params}. "
                f"Current depth is {self.curr_depth}."
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

    def on_before_zero_grad(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer) -> None:
        """Afer the latest optimizer step, update the
        :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state`, incrementing the
        global fine-tuning steps taken

        Args:
            trainer (:external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer`): The
                :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object
            pl_module  (:external+pl:class:`~pytorch_lightning.core.module.LightningModule`): The
                :external+pl:class:`~pytorch_lightning.core.module.LightningModule` object
            optimizer (:external+torch:class:`~torch.optim.Optimizer`): The
                :external+torch:class:`~torch.optim.Optimizer` to which parameter groups will be configured and added.
        """
        self._fts_state._ft_global_steps += 1

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Synchronize internal :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state` on end of training
        to ensure final training state is consistent with epoch semantics.

        Args:
            trainer: The :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer` object
            pl_module  (:external+pl:class:`~pytorch_lightning.core.module.LightningModule`): The
                :external+pl:class:`~pytorch_lightning.core.module.LightningModule` object
        """
        assert self._fts_state._ft_sync_objects is not None
        self.sync(self._fts_state._ft_sync_objects, self._fts_state._ft_sync_props)
