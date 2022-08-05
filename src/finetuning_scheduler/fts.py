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
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _StrategyType, rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_warn
from torch.optim.optimizer import Optimizer

from finetuning_scheduler.fts_supporters import (
    CallbackDepMixin,
    FTSEarlyStopping,
    FTSState,
    ScheduleImplMixin,
    ScheduleParsingMixin,
)

log = logging.getLogger(__name__)


class FinetuningScheduler(BaseFinetuning, ScheduleImplMixin, ScheduleParsingMixin, CallbackDepMixin):
    r"""
    This callback enables flexible, multi-phase, scheduled fine-tuning of foundational models. Gradual
    unfreezing/thawing can help maximize foundational model knowledge retention while allowing (typically upper layers
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
    """

    def __init__(
        self,
        ft_schedule: Optional[Union[str, dict]] = None,
        max_depth: int = -1,
        base_max_lr: float = 1e-5,
        restore_best: bool = True,
        gen_ft_sched_only: bool = False,
        epoch_transitions_only: bool = False,
        reinit_lr_cfg: Optional[Dict] = None,
        allow_untested: bool = False,
        apply_lambdas_new_pgs: bool = False,
    ):
        r"""
        Define and configure a scheduled fine-tuning training session.

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
                ``None``, will be set to the ``lr`` of the first scheduled fine-tuning depth scaled by 1e-1. Defaults to
                1e-5.
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
                ``lr_scheduler_init`` dictionary with a ``class_path`` key specifying the class of the
                :class:`~torch.optim.lr_scheduler._LRScheduler` to be instantiated. Optionally, an ``init_args``
                dictionary of arguments to initialize the lr scheduler with may be included. Additionally, one may
                optionally include arguments to pass to PyTorch Lightning's lr scheduler configuration
                :class:`~pytorch_lightning.utilities.types.LRSchedulerConfig` in the ``pl_lrs_cfg`` dictionary. By way
                of example, one could configure this dictionary via the
                :external+pl:class:`~pytorch_lightning.utilities.cli.LightningCLI` with the following:

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

            allow_untested: If ``True``, allows the use of custom or unsupported training strategies (e.g.
                ``single_tpu``, ``MyCustomStrategy``). Defaults to ``False``.

                .. note:: Custom or officially unsupported strategies can be used by setting
                    :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.allow_untested` to ``True``.

                    Some officially unsupported strategies may work unaltered and are only unsupported due to
                    the ``Fine-Tuning Scheduler`` project's lack of CI/testing resources for that strategy (e.g.
                    ``single_tpu``).

                    Most unsupported strategies, however, are currently unsupported because they require varying degrees
                    of modification to be compatible (e.g. ``deepspeed`` requires an ``add_param_group`` method,
                    ``tpu_spawn`` an override of the current broadcast method to include python objects).
            apply_lambdas_new_pgs: If ``True``, applies most recent lambda in ``lr_lambdas`` list to newly added
                optimizer groups for lr schedulers that have a ``lr_lambdas`` attribute. Note this option only applies
                to phases without reinitialized lr schedulers. Phases with defined lr scheduler reinitialization configs
                will always apply the specified lambdas. Defaults to ``False``.

        Attributes:
            _fts_state: The internal :class:`~finetuning_scheduler.fts.FinetuningScheduler` state.
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
        self.allow_untested = allow_untested
        self.apply_lambdas_new_pgs = apply_lambdas_new_pgs
        self.pl_module: pl.LightningModule

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
    def _supported_strategy_types() -> Sequence[Union[_StrategyType, str]]:
        return (
            _StrategyType.DP,
            _StrategyType.DDP,
            _StrategyType.DDP_SPAWN,
            # _StrategyType.DEEPSPEED,  # support to be re-evaluated if add optimizer pg functionality added to DS API
            _StrategyType.DDP_SHARDED,
            _StrategyType.DDP_SHARDED_SPAWN,
            "single_device",
        )

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        """Freezes all model parameters so that parameter subsets can be subsequently thawed according to the fine-
        tuning schedule.

        Args:
            pl_module (:external+pl:class:`~pytorch_lightning.core.module.LightningModule`): The target
                :external+pl:class:`~pytorch_lightning.core.module.LightningModule` to freeze parameters of
        """
        self.freeze(modules=pl_module)

    def step(self) -> None:
        """Prepare and execute the next scheduled fine-tuning level
        1. Restore the current best model checkpoint if appropriate
        2. Thaw model parameters according the the defined schedule
        3. Synchronize the states of :external+pl:class:`~pytorch_lightning.loops.FitLoop` and
        :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state`

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

    def step_pg(self, optimizer: Optimizer, depth: int, depth_sync: bool = True) -> None:
        """Configure optimizer parameter groups for the next scheduled fine-tuning level, adding parameter groups
        beyond the restored optimizer state up to
        :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.current_depth`

        Args:
            optimizer (:class:`~torch.optim.Optimizer`): The :class:`~torch.optim.Optimizer` to which parameter groups
                will be configured and added.
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
        for i, next_tl in thaw_layers:
            if i <= depth:
                _, self._fts_state._curr_thawed_params = FinetuningScheduler.exec_ft_phase(
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
                # restore optimizers and schedulers state
                checkpoint_connector.restore_optimizers_and_schedulers()
            except KeyError:
                assert isinstance(self.ft_schedule, dict)
                if self.ft_schedule[self.curr_depth].get("new_lr_scheduler", None):
                    rank_zero_warn(
                        "incompatible checkpoint detected but attempting to proceed with next phase of training since "
                        "we're reinitializing the lr scheduler."
                    )

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
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
            MisconfigurationException: If the
                :external+pl:class:`~pytorch_lightning.strategies.Strategy` strategy being used is not currently
                compatible with the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback.
        """
        trainer.callbacks, added_es_fts, added_ckpt_fts = self._configure_callback_deps(trainer)
        strategy = trainer.strategy
        # if we added callbacks for the user after the setup hooks loop was initiated from trainer, we'll need to
        # explicitly call the setup hooks for those added callbacks
        if added_ckpt_fts:
            trainer.checkpoint_callback.setup(trainer, pl_module, stage)  # type: ignore[union-attr]
        if added_es_fts:
            trainer.early_stopping_callback.setup(trainer, pl_module, stage)  # type: ignore[union-attr]
        assert pl_module is not None and pl_module.trainer is not None
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
        if self.gen_ft_sched_only:
            if trainer.is_global_zero:
                assert trainer.log_dir is not None
                _ = self.gen_ft_schedule(pl_module, trainer.log_dir)
                log.info("Bypassing training, generating fine-tuning schedule for review and subsequent fine-tuning")
            raise SystemExit()
        else:
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
        if len(trainer.optimizers) > 1:
            raise MisconfigurationException("fts currently only supports a single-optimizer configuration")
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
                f"Current parameters thawed by the Fine-Tuning Scheduler: {self._fts_state._curr_thawed_params}. "
                f"Current depth is {self.curr_depth}."
            )
            if not self.epoch_transitions_only:
                assert isinstance(trainer.early_stopping_callback, FTSEarlyStopping)
                trainer.early_stopping_callback.es_phase_complete = False
                trainer.early_stopping_callback.wait_count = 0
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
            optimizer (:class:`~torch.optim.Optimizer`): The :class:`~torch.optim.Optimizer` to which parameter groups
                will be configured and added.
        """
        self._fts_state._ft_global_steps += 1

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Synchronize internal :attr:`~finetuning_scheduler.fts.FinetuningScheduler._fts_state` on end of training
        to ensure final training state is consistent with epoch semantics.

        Args:
            trainer (:external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer`): _description_
            pl_module (:external+pl:class:`~pytorch_lightning.core.module.LightningModule`): _description_
        """
        assert self._fts_state._ft_sync_objects is not None
        self.sync(self._fts_state._ft_sync_objects, self._fts_state._ft_sync_props)
