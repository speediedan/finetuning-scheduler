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
Fine-Tuning Scheduler Strategy Adapters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Base adapter class to extend Fine-Tuning Scheduler support of complex or custom training strategies.

"""
from functools import partialmethod
from pprint import pformat as pfmt
from typing import Callable, List, Optional, Tuple, Dict

import torch

from lightning.fabric.utilities import rank_zero_info
from lightning.fabric.utilities.types import ReduceLROnPlateau
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.strategies.strategy import Strategy
from lightning.pytorch.utilities.rank_zero import rank_zero_debug


class StrategyAdapter:
    r"""Base class for all strategy adapters. Implements the default
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` hooks. Can be subclassed to extend
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` support for a complex or custom
    :external+pl:class:`~lightning.pytorch.strategies.Strategy` via an associated
    :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`.

    .. warning::

        :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` is in BETA and subject to change. The interface
        can bring breaking changes and new features with the next release of FTS.

    .. tip::

        If you want to extend FTS to use a custom, currently unsupported strategy or override current FTS behavior in
        the context of a given training strategy, subclassing
        :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` is a way to do so. See
        :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` for an example implementation.

    """

    fts_handle: Callback
    _ft_schedule_module_map: Dict
    _unscheduled_params: List

    def __init__(self) -> None:
        """The default fine-tuning phase execution function is set on
        :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` initialization. This can be overridden by
        :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` subclasses to adapt fine-tuning phase
        execution to meet strategy-specific requirements.
        """
        self.exec_ft_phase = StrategyAdapter.base_ft_phase

    def connect(self, fts_parent: Callback) -> None:
        """Create a handle for the associated :class:`~finetuning_scheduler.fts.FinetuningScheduler` instance.

        Args:
            fts_parent (Callback): The associated :class:`~finetuning_scheduler.fts.FinetuningScheduler` instance
        """
        self.fts_handle = fts_parent

    @property
    def pl_module(self) -> LightningModule:
        """Convenient access to the :external+pl:class:`~lightning.pytorch.core.module.LightningModule` being fine-
        tuned.

        Returns:
            LightningModule: The user's :external+pl:class:`~lightning.pytorch.core.module.LightningModule`
        """
        return self.fts_handle.pl_module

    @property
    def pls_handle(self) -> Strategy:
        """Convenient access to the current :external+pl:class:`~lightning.pytorch.strategies.Strategy` in use.

        Returns:
            Strategy: The :external+pl:class:`~lightning.pytorch.strategies.Strategy` in use.
        """
        assert self.pl_module._trainer is not None
        return self.pl_module._trainer.strategy

    @property
    def using_sharded_optimizer(self) -> bool:
        """Whether the currently used optimizer is a supported sharded optimizer.

        Returns:
            bool: Returns ``True`` if the current optimizer is a supported sharded optimizer.
        """
        return hasattr(self.pl_module.trainer.optimizers[0], "consolidate_state_dict")

    def on_before_init_fts(self) -> None:
        """Hook executed in :class:`~finetuning_scheduler.fts.FinetuningScheduler` setup immediately before
        :meth:`~finetuning_scheduler.fts_supporters.ScheduleImplMixin.init_fts`
        """

    def on_after_init_fts(self) -> None:
        """Hook executed in :class:`~finetuning_scheduler.fts.FinetuningScheduler` setup immediately after
        :meth:`~finetuning_scheduler.fts_supporters.ScheduleImplMixin.init_fts`.
        """
        self._gen_ft_sched_module_map()
        self.scheduled_mod_lists = [list(self._ft_schedule_module_map[d]) for d in self._ft_schedule_module_map.keys()]
        self._maybe_set_bn_track_running_stats(0)
        _, self.fts_handle._fts_state._curr_thawed_params = self.exec_ft_phase(
            self.pl_module,
            thaw_pl=self.fts_optim_transform(self.fts_handle.ft_schedule[0]["params"]),
            init_thaw=True,
        )

    def on_before_fts_fit_start(self) -> None:
        """Hook executed immediately before the :class:`~finetuning_scheduler.fts.FinetuningScheduler`
        :meth:`~finetuning_scheduler.fts.on_fit_start` hook begins.
        """

    def on_before_restore_optimizers_and_lrs(self) -> None:
        """Hook executed immediately before :class:`~finetuning_scheduler.fts.FinetuningScheduler` restores
        optimizers and schedulers."""

    def fts_optim_transform(self, orig_pl: List, inspect_only: bool = False) -> List:
        """A method that can be overridden by a :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` if
        a :external+pl:class:`~lightning.pytorch.strategies.Strategy` performs parameter transformations that cause
        the current optimizer's view of parameter names to diverge from the original parameter names. By default,
        no transformation of schedule parameter names is required for optimizer operations.

        Args:
            orig_pl (List): The original parameter name list before a given
                :external+pl:class:`~lightning.pytorch.strategies.Strategy`'s transformation of them.
            inspect_only (bool): Whether to use the specified transform in read-only (i.e. ``inspect_only``) mode,
                avoiding any persistent state transformation that may accompany normal usage. Typically useful for state
                inspection and validation contexts.

        Returns:
            List: A transformed parameter name list that matches the current optimizer's view of them after a given
                :external+pl:class:`~lightning.pytorch.strategies.Strategy`'s transformation of the original parameter
                names.
        """
        return orig_pl

    def logical_param_translation(self, param_names: List) -> List:
        """Effectively the reverse transformation of
        :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.fts_optim_transform`. Can be overridden by a
        :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` if a
        :external+pl:class:`~lightning.pytorch.strategies.Strategy` performs parameter transformations that cause the
        original user view of parameter names to diverge from the current optimizer's view. By default, no
        transformation of optimizer parameter names is required.

        Args:
            param_names (List): A parameter name list from the current optimizer's view of them after a
                :external+pl:class:`~lightning.pytorch.strategies.Strategy`'s transformation of the original parameter
                names.

        Returns:
            List: The original parameter name list before a given
                :external+pl:class:`~lightning.pytorch.strategies.Strategy`'s transformation.
        """
        return param_names

    def _gen_ft_sched_module_map(self) -> None:
        """Generate a module-level mapping of the modules associated with each fine-tuning phase, including modules
        not present in the fine-tuning schedule grouped together into a single unscheduled phase to facilitate the
        relevant disjointness check."""
        assert isinstance(self.fts_handle.ft_schedule, Dict)
        module_map: Dict = {}
        for depth in self.fts_handle.ft_schedule.keys():  # type: ignore[union-attr]
            phase_params = self.fts_handle.ft_schedule[depth].get("params", [])  # type: ignore[union-attr]
            module_map[depth] = set()
            for p in phase_params:
                module_map[depth].add(p.rpartition(".")[0])
        self._ft_schedule_module_map = module_map
        scheduled_mods = list(set().union(*module_map.values()))
        unscheduled_mods = tuple(
            n for n, m in self.pl_module.named_modules() if n not in scheduled_mods and m._parameters
        )
        self._unscheduled_params = [
            f"{m}.{n}" for m in unscheduled_mods for n, _ in self.pl_module.get_submodule(m).named_parameters()
        ]

    @staticmethod
    def _clean_optim_lr_pgs(trainer: Trainer) -> List:
        """Delete existing param groups from an optimizer that was found to be misaligned with respect to phase 0
        of the specified fine-tuning schedule.

        Args:
            trainer (Trainer): The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object.

        Returns:
            List: A list of the number of parameter groups pruned for each optimizer (since only a single optimizer is
                currently supported by FTS, this list will have only a single element in this verison.)
        """
        orig_num_pgs = []
        for optimizer in trainer.optimizers:
            orig_num_pgs.append(len(optimizer.param_groups))
            optimizer.param_groups = []
        for lrs_cfg in trainer.lr_scheduler_configs:
            lrs_cfg.scheduler.last_epoch = -1  # type: ignore[union-attr]
            if not isinstance(lrs_cfg.scheduler, ReduceLROnPlateau):
                lrs_cfg.scheduler.base_lrs = []
        return orig_num_pgs

    def _reconfigure_optimizer_for_phase0(self, trainer: Trainer) -> None:
        """Reconfigure optimizer state to comport with the scheduled phase 0.

        Args:
            trainer (Trainer): The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object.
        """
        if self.using_sharded_optimizer:
            # update the parent (sharded) optimizer defaults with the wrapped optimizer defaults to ensure they are
            # included in our newly configured param groups
            trainer.optimizers[0].defaults.update(trainer.optimizers[0].optim.defaults)
        # thaw only params scheduled in phase 0
        self.fts_handle.step_pg(depth=self.fts_handle.curr_depth, optimizer=trainer.optimizers[0], depth_sync=False)
        if self.using_sharded_optimizer:
            # update internal sharded optimizer state with the new set of parameters
            trainer.optimizers[0]._verify_and_init_params([p for p in self.pl_module.parameters() if p.requires_grad])
            # repartition the sharded optimizer state
            self.fts_handle._repartition_sharded_optim(trainer.optimizers[0])

    def _reconfigure_lrs_for_phase0(self, trainer: Trainer, orig_num_pgs: List) -> None:
        """Reconfigure lr scheduler state to comport with the scheduled phase 0.

        Args:
            trainer (Trainer): The :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer` object.
            orig_num_pgs (List): A list of the number of parameter groups pruned for each optimizer (since only a single
                optimizer is currently supported by FTS, this list will have only a single element in this verison.)
        """
        # since we may have added parameter groups (e.g. implementing ``no_decay`` for user), we need to reinitialize
        # certain lr_scheduler variables (including type-dependent ones like ``min_lrs`` and ``lr_lambdas``)
        if trainer.lr_scheduler_configs:
            for lrs_cfg in trainer.lr_scheduler_configs:
                if not isinstance(lrs_cfg.scheduler, ReduceLROnPlateau):
                    lrs_cfg.scheduler._initial_step()
                lrs_cfg.scheduler._last_lr = [  # type: ignore[union-attr]
                    group["lr"] for group in lrs_cfg.scheduler.optimizer.param_groups
                ]
                if isinstance(lrs_cfg.scheduler, ReduceLROnPlateau):
                    lrs_cfg.scheduler.min_lrs = lrs_cfg.scheduler.min_lrs[orig_num_pgs[0] :]
                elif hasattr(lrs_cfg.scheduler, "lr_lambdas"):
                    lrs_cfg.scheduler.lr_lambdas = lrs_cfg.scheduler.lr_lambdas[orig_num_pgs[0] :]

    def phase0_optimizer_override(self) -> None:
        """Reconfigure the user-configured optimizer (configured via `configure_optimizers`) to optimize the
        parameters (and only those parameters) scheduled to be optimized in phase 0 of the current fine-tuning
        schedule.

        Reconfiguration only takes place here if FTS discovers the set of parameters to be initially thawed and present
        in the optimizer differs from the parameters specified in phase 0. Only the parameters included in the optimizer
        are affected; the choice of optimizer, lr_scheduler etc. remains unaltered.
        """
        trainer = self.pl_module.trainer
        orig_num_pgs = StrategyAdapter._clean_optim_lr_pgs(trainer)
        # refreeze in case user has thawed parameters not present in phase 0
        self.fts_handle.freeze_before_training(self.pl_module)
        self._reconfigure_optimizer_for_phase0(trainer)
        self._reconfigure_lrs_for_phase0(trainer, orig_num_pgs)
        p0_override_msg = self.fts_handle.PHASE_0_DIVERGENCE_MSG + (
            "Since `enforce_phase0_params` is currently set to `True` (the default), FinetuningScheduler has"
            " reconfigured the optimizer to optimize the parameters (and only those parameters) scheduled to be"
            " optimized in phase 0 of the current fine-tuning schedule.\n\n"
        )
        rank_zero_info(p0_override_msg)

    @staticmethod
    def base_ft_phase(
        module: torch.nn.Module, thaw_pl: List, translation_func: Optional[Callable] = None, init_thaw: bool = False) \
            -> Tuple[List, List]:
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
                f"{pfmt(translation_func(thawed_p_names)) if translation_func else pfmt([n for n in thawed_p_names])}"
            )
        curr_thawed.extend(thawed_p_names)
        rank_zero_debug(
            f"The following module parameters are currently thawed: "
            f"{pfmt(translation_func(curr_thawed)) if translation_func else pfmt([n for n in curr_thawed])}"
        )
        return thawed_p_names, curr_thawed

    ####################################################################################################################
    # BatchNorm module-specific handling
    # (if additional modules require special handling, these will be refactored to accommodate a more generic
    # dispatching pattern for module-specific handling)
    ####################################################################################################################

    def _module_specific_freezing(self, modules: torch.nn.Module) -> None:
        """Orchestrates module-specific freezing behavior. Currently only.

        :external+torch:class:`~torch.nn.modules.batchnorm._BatchNorm` layers require special handling. Running
        statistics tracking for frozen `BatchNorm` layers is conditionally re-enabled here based on the
        `frozen_bn_track_running_stats` flag.

        Args:
            modules (torch.nn.Module): The modules for which the `BatchNorm` layer running statistics should be enabled.
        Returns:
            None
        """
        if self.fts_handle.frozen_bn_track_running_stats:
            rank_zero_debug("Since `frozen_bn_track_running_stats` is currently set to `True`, FinetuningScheduler"
                            " will set `track_running_stats` to `True` for all `BatchNorm` layers.")
            modules = BaseFinetuning.flatten_modules(modules)  # type: ignore[assignment]
            for mod in modules:
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.track_running_stats = True

    def _maybe_set_bn_track_running_stats(self, schedule_phase: int) -> None:
        """Enable `track_running_stats` for :external+torch:class:`~torch.nn.modules.batchnorm._BatchNorm` modules
        that may require it based on `frozen_bn_track_running_stats` and a given schedule phase.

        Args:
            schedule_phase (int): The phase of the schedule to evaluate.

        Returns:
            None
        """
        if not self.fts_handle.frozen_bn_track_running_stats:
            target_bn_modules = self._get_target_bn_modules(schedule_phase)
            for _, m in target_bn_modules:
                m.track_running_stats = True

    def _get_target_bn_modules(self, schedule_phase: int) -> List:
        """Enumerate the :external+torch:class:`~torch.nn.modules.batchnorm._BatchNorm` modules for a given
        schedule phase.

        Args:
            schedule_phase (int): The phase of the schedule to evaluate.

        Returns:
            List[Tuple[str, torch.nn.modules.batchnorm._BatchNorm]]: A list of tuples containing the names and instances
              of `BatchNorm` modules associated with a given schedule phase.
        """
        return [(n, m) for n, m in self.pl_module.named_modules() if
                n in self.scheduled_mod_lists[schedule_phase] and
                isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]

    fts_optim_inspect = partialmethod(fts_optim_transform, inspect_only=True)
