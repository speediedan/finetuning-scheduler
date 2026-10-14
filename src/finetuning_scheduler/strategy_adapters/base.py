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
from typing import Any  # Dict used for runtime isinstance() checks
from collections.abc import Callable, Iterable
import inspect
import logging
import os
import warnings

import torch
from torch.nn.parallel import DistributedDataParallel
from lightning.fabric.utilities import rank_zero_info, rank_zero_warn
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.strategies.strategy import Strategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_only

import finetuning_scheduler.lightning  # noqa: F401  (registers the Lightning backends the core logs through)
from finetuning_scheduler.core import optim as core_optim
from finetuning_scheduler.core import phase as core_phase
from finetuning_scheduler.core.compile import CompiledSchedule
from finetuning_scheduler.core.roles import is_frozen_role
from finetuning_scheduler.core.views import BoundView, resolve_view, VIEW_ATTR

log = logging.getLogger(__name__)

# the parameter-naming methods a third-party adapter may override, which a `TunableView` replaces
NAMING_METHODS = ("fts_optim_transform", "logical_param_translation", "get_named_params_for_schedule_validation",
                  "validate_ft_sched", "gen_ft_schedule")
# the methods whose override always renames parameters: an adapter overriding only `fts_optim_transform` (a sharding
# transform, for example) or `validate_ft_sched` (to add checks) does not name parameters itself
RENAMING_METHODS = ("logical_param_translation", "get_named_params_for_schedule_validation", "gen_ft_schedule")
# adapter classes already told that their naming overrides take precedence over an attached view
_PRECEDENCE_NOTICED: set[type] = set()
# adapter classes already told that overriding the naming methods is deprecated
_DEPRECATION_NOTICED: set[type] = set()


@rank_zero_only
def _warn_from_adapter_class(cls: type, message: str) -> None:
    """Issue a ``FutureWarning`` attributed to the definition of adapter class ``cls``, the code a user would
    change, rather than to Fine-Tuning Scheduler's own call site."""
    try:
        filename, lineno = inspect.getsourcefile(cls) or "<unknown>", inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        filename, lineno = "<unknown>", 0
    warnings.warn_explicit(message, FutureWarning, filename, lineno, module=cls.__module__)


class StrategyAdapter:
    r"""Base class for all strategy adapters. Implements the default
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` hooks. Can be subclassed to extend
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` support for a complex or custom
    :py:class:`~lightning.pytorch.strategies.Strategy` via an associated
    :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`.

    .. tip::

        If you want to extend FTS to use a custom, currently unsupported strategy or override current FTS behavior in
        the context of a given training strategy, subclassing
        :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` is a way to do so. See
        :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` for an example implementation.
    """

    fts_handle: Callback
    _ft_schedule_module_map: dict
    _unscheduled_params: list
    # the trainable parameters the DistributedDataParallel reducer covers (class default: subclasses need not call init)
    _ddp_reducer_params: frozenset | None = None
    # built-in adapters declare the naming methods they override as naming overrides: none (see `naming_overrides`)
    _fts_naming_overrides: frozenset[str] = frozenset()
    # the schedule compiled through an attached view (or for `role:` selectors), set by schedule validation
    compiled_schedule: CompiledSchedule | None = None
    _bound_view: tuple[int, BoundView] | None = None

    def __init__(self) -> None:
        """The default fine-tuning phase execution function is set on
        :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` initialization.

        This can be overridden by
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
        """Convenient access to the :py:class:`~lightning.pytorch.core.module.LightningModule` being fine- tuned.

        Returns:
            LightningModule: The user's :py:class:`~lightning.pytorch.core.module.LightningModule`
        """
        return self.fts_handle.pl_module

    @property
    def trainer(self) -> Trainer:
        """Convenient access to the :py:class:`~lightning.pytorch.trainer.trainer.Trainer` instance.

        Returns:
            Trainer: The :py:class:`~lightning.pytorch.trainer.trainer.Trainer` instance
        """
        return self.fts_handle.trainer

    @classmethod
    def naming_overrides(cls) -> frozenset[str]:
        """The parameter-naming methods (:data:`NAMING_METHODS`) this adapter class overrides relative to its
        nearest built-in adapter (built-ins declare ``_fts_naming_overrides``), when it names parameters itself.

        An adapter names parameters itself when it overrides one of the methods that only rename
        (:data:`RENAMING_METHODS`). Overriding ``fts_optim_transform`` alone (a sharding transform, such as FSDP's) or
        ``validate_ft_sched`` alone (to add checks) is not a naming override.
        """
        builtin = next(k for k in cls.__mro__ if "_fts_naming_overrides" in vars(k))
        if builtin is cls:
            return frozenset(vars(cls)["_fts_naming_overrides"])
        overridden = frozenset(m for m in NAMING_METHODS if getattr(cls, m) is not getattr(builtin, m))
        return overridden if overridden.intersection(RENAMING_METHODS) else frozenset()

    @classmethod
    def notice_naming_overrides(cls) -> None:
        """Warn, once per adapter class, that overriding the parameter-naming methods is deprecated in favor of a
        :class:`~finetuning_scheduler.core.views.TunableView`."""
        overrides = cls.naming_overrides()
        if not overrides or cls in _DEPRECATION_NOTICED:
            return
        _DEPRECATION_NOTICED.add(cls)
        _warn_from_adapter_class(
            cls,
            f"`{cls.__qualname__}` overrides StrategyAdapter's parameter-naming methods"
            f" ({', '.join(sorted(overrides))}). Overriding them to name parameters is deprecated since"
            " finetuning-scheduler 2.15 and will be unsupported no earlier than 2.17: implement a `TunableView`"
            " (`finetuning_scheduler.core.views`) and attach it to your module as `fts_tunable_view` instead. Sharding"
            " customization through `StrategyAdapter` subclasses (including a sharding `fts_optim_transform`) is"
            " unaffected.",
        )

    @property
    def tunable_view(self) -> BoundView | None:
        """The view schedules are resolved through: the view attached to the module as ``fts_tunable_view``, or the
        canonical view. ``None`` for an adapter with parameter-naming overrides, whose overrides name parameters
        instead; an attached view is then ignored, with a notice. Resolved lazily, on first use."""
        if self.naming_overrides():
            self._notice_ignored_view()
            return None
        module = self.pl_module
        if self._bound_view is None or self._bound_view[0] != id(module):
            view = resolve_view(module)
            if view.root is not module:
                raise MisconfigurationException(
                    f"The `fts_tunable_view` attached to {type(module).__name__} is rooted at"
                    f" {type(view.root).__name__}; its `root` must be the LightningModule itself, whose parameters the"
                    " optimizer and checkpoints see."
                )
            self._bound_view = (id(module), view)
        return self._bound_view[1]

    @property
    def attached_view(self) -> bool:
        """Whether the module carries a view (``fts_tunable_view``) that this adapter schedules through."""
        return getattr(self.pl_module, VIEW_ATTR, None) is not None and self.tunable_view is not None

    def _notice_ignored_view(self) -> None:
        cls = type(self)
        attached = getattr(self.pl_module, VIEW_ATTR, None)
        if attached is None or cls in _PRECEDENCE_NOTICED:
            return
        _PRECEDENCE_NOTICED.add(cls)
        _warn_from_adapter_class(
            cls,
            f"{cls.__qualname__} overrides StrategyAdapter's parameter-naming methods"
            f" ({', '.join(sorted(self.naming_overrides()))}), so it names the parameters of"
            f" {type(self.pl_module).__name__} and the `fts_tunable_view` attached to it"
            f" ({type(attached).__qualname__}) is not used.",
        )

    def scheduled_physical(self, depth: int) -> list:
        """The physical parameters phase ``depth`` thaws: its compiled parameters (coupled ones included) under an
        attached view or ``role:`` selectors, the adapter's own translation when it overrides the naming methods,
        else the schedule's own names."""
        if self.compiled_schedule is not None:
            return list(self.compiled_schedule.physical_for(depth))
        names = list(self.fts_handle.ft_schedule[depth]["params"])  # type: ignore[index]
        if self.naming_overrides():
            return list(self.fts_optim_transform(names, inspect_only=True))
        return names

    def _compiled_physical(self, names: list) -> list:
        """The physical parameters a phase's ``params`` (or other compiled names) stand for, coupled ones included
        for a whole phase."""
        assert self.compiled_schedule is not None
        for depth, phase in self.fts_handle.ft_schedule.items():  # type: ignore[union-attr]
            if list(phase.get("params", [])) == list(names):
                return list(self.compiled_schedule.physical_for(depth))
        return self.compiled_schedule.physical_of(names)

    @property
    def pls_handle(self) -> Strategy:
        """Convenient access to the current :py:class:`~lightning.pytorch.strategies.Strategy` in use.

        Returns:
            Strategy: The :py:class:`~lightning.pytorch.strategies.Strategy` in use.
        """
        assert self.pl_module._trainer is not None
        return self.pl_module._trainer.strategy

    @property
    def using_sharded_optimizer(self) -> bool:
        """Whether the currently used optimizer is a supported sharded optimizer.

        Returns:
            bool: Returns ``True`` if the current optimizer is a supported sharded optimizer.
        """
        return hasattr(self.trainer.optimizers[0], "consolidate_state_dict")

    def on_before_init_fts(self) -> None:
        """Hook executed in :class:`~finetuning_scheduler.fts.FinetuningScheduler` setup immediately before
        :meth:`~finetuning_scheduler.fts_supporters.ScheduleImplMixin.init_fts`"""

    def on_after_init_fts(self) -> None:
        """Hook executed in :class:`~finetuning_scheduler.fts.FinetuningScheduler` setup immediately after
        :meth:`~finetuning_scheduler.fts_supporters.ScheduleImplMixin.init_fts`."""
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
        :meth:`~finetuning_scheduler.fts.on_fit_start` hook begins."""

    def on_after_phase_transition(self) -> None:
        """Hook executed after :class:`~finetuning_scheduler.fts.FinetuningScheduler` thaws the parameters of a new
        phase (at each phase transition, and when resuming training at a depth beyond phase 0) and configures the
        optimizer for them.

        By default, keeps ``DistributedDataParallel``'s gradient synchronization covering every trainable parameter
        (see :meth:`_sync_ddp_reducer`). Adapters that override this hook should call ``super()``.
        """
        self._sync_ddp_reducer()

    def _sync_ddp_reducer(self) -> None:
        """Rebuild the ``DistributedDataParallel`` wrapper when the set of trainable parameters has changed since
        it was built.

        ``DistributedDataParallel`` registers with its gradient reducer only the parameters that require grad when it
        wraps the module, so parameters thawed in later phases would otherwise never have their gradients all-reduced
        and each process would train its replica on its own data alone. Rebuilding detaches the previous reducer's
        autograd hooks, wraps the same module again (which also re-broadcasts the module state from rank 0) and
        re-registers any configured communication hooks. A no-op for strategies that do not wrap the module in
        ``DistributedDataParallel``.
        """
        strategy = self.pls_handle
        ddp_model = getattr(strategy, "model", None)
        if not isinstance(ddp_model, DistributedDataParallel):
            return
        trainable = frozenset(id(p) for p in ddp_model.module.parameters() if p.requires_grad)
        if trainable == self._ddp_reducer_params:
            return
        ddp_model._remove_autograd_hooks()
        strategy.model = strategy._setup_model(ddp_model.module)  # type: ignore[attr-defined]
        strategy._register_ddp_hooks()  # type: ignore[attr-defined]
        self._ddp_reducer_params = trainable
        rank_zero_debug(
            "Rebuilt the DistributedDataParallel wrapper so its gradient reducer covers the"
            f" {len(trainable)} currently trainable parameters."
        )

    def on_validate_monitor_metric(self, monitor: str) -> None:
        should_warn = self.fts_handle._check_sync_dist(monitor)
        if should_warn:
            warn_msg = (
                f"The FTSCheckpoint quantity you are monitoring (`{monitor}`) is not being synchronized across"
                " processes. In this context, `best` checkpoint metadata may diverge among processes making checkpoint"
                " restoration errors possible. For this reason, logging the monitored metric with ``sync_dist`` set to"
                " ``True`` is recommended."
            )
            rank_zero_warn(warn_msg)

    def on_before_restore_optimizers_and_lrs(self) -> None:
        """Hook executed immediately before :class:`~finetuning_scheduler.fts.FinetuningScheduler` restores
        optimizers and schedulers."""

    def fts_optim_transform(self, orig_pl: list, inspect_only: bool = False) -> list:
        """A method that can be overridden by a :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` if
        a :py:class:`~lightning.pytorch.strategies.Strategy` performs parameter transformations that cause
        the current optimizer's view of parameter names to diverge from the original parameter names. By default,
        no transformation of schedule parameter names is required for optimizer operations.

        .. note::
            Overriding this method for a sharding transform is supported: call ``super().fts_optim_transform(orig_pl,
            inspect_only)`` first, which resolves the names of a view attached to the module, and transform its
            result. Overriding it to rename parameters (together with
            :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.logical_param_translation`) is deprecated
            since 2.15 in favor of a :class:`~finetuning_scheduler.core.views.TunableView`.

        Args:
            orig_pl (List): The original parameter name list before a given
                :py:class:`~lightning.pytorch.strategies.Strategy`'s transformation of them.
            inspect_only (bool): Whether to use the specified transform in read-only (i.e. ``inspect_only``) mode,
                avoiding any persistent state transformation that may accompany normal usage. Typically useful for state
                inspection and validation contexts.

        Returns:
            List: A transformed parameter name list that matches the current optimizer's view of them after a given
                :py:class:`~lightning.pytorch.strategies.Strategy`'s transformation of the original parameter
                names. Under an attached view (or a schedule with ``role:`` selectors), the physical parameters the
                schedule compiled to.
        """
        return orig_pl if self.compiled_schedule is None else self._compiled_physical(orig_pl)

    def best_ckpt_path(self, path: str) -> str:
        """The checkpoint to restore for the checkpoint callback's ``best_model_path`` (at a phase transition, or
        on resume): ``path`` itself unless the strategy cannot load it and an adapter substitutes a converted
        counterpart (or refuses with an actionable error).

        Args:
            path (str): The checkpoint callback's ``best_model_path``.

        Returns:
            str: The checkpoint path to restore.
        """
        return path

    def before_restore_model(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Adapter hook executed before model restore.

        Strategy adapters can override this to modify or translate the checkpoint contents (e.g. for state-dict
        translations) before the model's load path is executed.

        Args:
            checkpoint (dict[str, Any]): The full checkpoint dict loaded by the Trainer.

        Returns:
            dict[str, Any]: The checkpoint dictionary to be used for restore.
        """
        return checkpoint

    def logical_param_translation(self, param_names: list) -> list:
        """Effectively the reverse transformation of
        :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.fts_optim_transform`. Can be overridden by a
        :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` if a
        :py:class:`~lightning.pytorch.strategies.Strategy` performs parameter transformations that cause the
        original user view of parameter names to diverge from the current optimizer's view. By default, no
        transformation of optimizer parameter names is required.

        Args:
            param_names (List): A parameter name list from the current optimizer's view of them after a
                :py:class:`~lightning.pytorch.strategies.Strategy`'s transformation of the original parameter
                names.

        Returns:
            List: The original parameter name list before a given
                :py:class:`~lightning.pytorch.strategies.Strategy`'s transformation. Under an attached view with
                logical names, the view's logical names.
        """
        if self.compiled_schedule is not None and self.compiled_schedule.naming_scheme != "physical":
            assert self.tunable_view is not None
            return self.tunable_view.to_logical(param_names)
        return param_names

    def _gen_ft_sched_module_map(self) -> None:
        """Generate a module-level mapping of the modules associated with each fine-tuning phase, including modules
        not present in the fine-tuning schedule grouped together into a single unscheduled phase to facilitate the
        relevant disjointness check."""
        assert isinstance(self.fts_handle.ft_schedule, dict)
        module_map: dict = {}
        if self.compiled_schedule is not None:  # the modules owning each phase's physical parameters
            module_map = self.compiled_schedule.module_map()
        else:
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
    def _clean_optim_lr_pgs(trainer: Trainer) -> list:
        """Delete existing param groups from an optimizer that was found to be misaligned with respect to phase 0
        of the specified fine-tuning schedule.

        Args:
            trainer (Trainer): The :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object.

        Returns:
            List: A list of the number of parameter groups pruned for each optimizer (since only a single optimizer is
                currently supported by FTS, this list will have only a single element in this verison.)
        """
        return core_optim.clean_optim_lr_pgs(
            trainer.optimizers, [cfg.scheduler for cfg in trainer.lr_scheduler_configs]  # type: ignore[arg-type]
        )

    def _reconfigure_optimizer_for_phase0(self, trainer: Trainer) -> None:
        """Reconfigure optimizer state to comport with the scheduled phase 0.

        Args:
            trainer (Trainer): The :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object.
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

    def _reconfigure_lrs_for_phase0(self, trainer: Trainer, orig_num_pgs: list) -> None:
        """Reconfigure lr scheduler state to comport with the scheduled phase 0.

        Args:
            trainer (Trainer): The :py:class:`~lightning.pytorch.trainer.trainer.Trainer` object.
            orig_num_pgs (List): A list of the number of parameter groups pruned for each optimizer (since only a single
                optimizer is currently supported by FTS, this list will have only a single element in this version.)
        """
        # since we may have added parameter groups (e.g. implementing ``no_decay`` for user), we need to reinitialize
        # certain lr_scheduler variables (including type-dependent ones like ``min_lrs`` and ``lr_lambdas``)
        core_optim.reconfigure_lrs_for_phase0([cfg.scheduler for cfg in trainer.lr_scheduler_configs], orig_num_pgs)

    def phase0_optimizer_override(self) -> None:
        """Reconfigure the user-configured optimizer (configured via `configure_optimizers`) to optimize the
        parameters (and only those parameters) scheduled to be optimized in phase 0 of the current fine-tuning
        schedule.

        Reconfiguration only takes place here if FTS discovers the set of parameters to be initially thawed and present
        in the optimizer differs from the parameters specified in phase 0. Only the parameters included in the optimizer
        are affected; the choice of optimizer, lr_scheduler etc. remains unaltered.
        """
        trainer = self.trainer
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
        module: torch.nn.Module, thaw_pl: list, translation_func: Callable | None = None, init_thaw: bool = False) \
            -> tuple[list, list]:
        """Thaw/unfreeze the provided list of parameters in the provided :class:`~torch.nn.Module`

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` that will have parameters selectively
                unfrozen/thawed.
            thaw_pl: The list of parameters that should be thawed/unfrozen in the :class:`~torch.nn.Module`
            init_thaw: If ``True``, modifies message to user accordingly. Defaults to ``False``.

        Returns:
            tuple[List, List]: A Tuple of two lists.
                1. The list of newly thawed/unfrozen parameters thawed by this function
                2. A list of all currently thawed/unfrozen parameters in the target :class:`~torch.nn.Module`
        """
        return core_phase.thaw_params(module, thaw_pl, translation_func=translation_func, init_thaw=init_thaw)

    ####################################################################################################################
    # BatchNorm module-specific handling
    # (if additional modules require special handling, these will be refactored to accommodate a more generic
    # dispatching pattern for module-specific handling)
    ####################################################################################################################

    def _module_specific_freezing(self, modules: torch.nn.Module | Iterable[torch.nn.Module]) -> None:
        """Orchestrates module-specific freezing behavior. Currently only
        :py:class:`~torch.nn.modules.batchnorm._BatchNorm` layers require special handling. Running statistics
        tracking for frozen `BatchNorm` layers is conditionally re-enabled here based on the
        `frozen_bn_track_running_stats` flag.

        Args:
            modules: The modules for which the `BatchNorm` layer running statistics should be enabled.
                Can be a single Module or an iterable of Modules.
        Returns:
            None
        """
        if self.fts_handle.frozen_bn_track_running_stats:
            rank_zero_debug("Since `frozen_bn_track_running_stats` is currently set to `True`, FinetuningScheduler"
                            " will set `track_running_stats` to `True` for all `BatchNorm` layers.")
            modules = BaseFinetuning.flatten_modules(modules)
            for mod in modules:
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.track_running_stats = True

    def _maybe_set_bn_track_running_stats(self, schedule_phase: int) -> None:
        """Enable `track_running_stats` for :py:class:`~torch.nn.modules.batchnorm._BatchNorm` modules that may
        require it based on `frozen_bn_track_running_stats` and a given schedule phase.

        Args:
            schedule_phase (int): The phase of the schedule to evaluate.

        Returns:
            None
        """
        if not self.fts_handle.frozen_bn_track_running_stats:
            target_bn_modules = self._get_target_bn_modules(schedule_phase)
            for _, m in target_bn_modules:
                m.track_running_stats = True

    def _get_target_bn_modules(self, schedule_phase: int) -> list:
        """Enumerate the :py:class:`~torch.nn.modules.batchnorm._BatchNorm` modules for a given schedule phase.

        Args:
            schedule_phase (int): The phase of the schedule to evaluate.

        Returns:
            list[tuple[str, torch.nn.modules.batchnorm._BatchNorm]]: A list of tuples containing the names and instances
              of `BatchNorm` modules associated with a given schedule phase.
        """
        return [(n, m) for n, m in self.pl_module.named_modules() if
                n in self.scheduled_mod_lists[schedule_phase] and
                isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]

    fts_optim_inspect = partialmethod(fts_optim_transform, inspect_only=True)

    def get_named_params_for_schedule_validation(self) -> dict[str, torch.nn.Parameter]:
        """Get named parameters for schedule validation.

        This method can be overridden by :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`
        subclasses to customize parameter iteration for schedule validation (e.g., returning TL-style
        parameter names instead of canonical names).

        .. note::
            Strategy adapters can override validation behavior at two levels of abstraction:

            1. **Parameter naming only** (simpler): Override this method to provide custom parameter names
               while using the default validation logic from
               :meth:`~finetuning_scheduler.fts_supporters.ScheduleParsingMixin._validate_ft_sched`.

            2. **Full validation logic** (more control): Override
               :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.validate_ft_sched` to
               completely customize the validation process.

            Choose the approach that best suits your use case. Most adapters only need to override this
            method to provide custom parameter names.

        Returns:
            dict[str, torch.nn.Parameter]: A dictionary mapping parameter names to parameter tensors.
                By default, returns the standard ``named_parameters()`` dict, or the logical names of a view attached
                to the module.
        """
        if self.attached_view:
            assert self.tunable_view is not None
            # a view's logical names map to tensors (parameters, or views of them)
            return dict(self.tunable_view.logical_named_parameters())  # type: ignore[arg-type]
        return dict(self.pl_module.named_parameters())

    def validate_ft_sched(self) -> tuple[int, int]:
        """Validate the fine-tuning schedule configuration.

        This method can be overridden by :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`
        subclasses to customize schedule validation for specific strategies (e.g., strategies that
        require substantially different validation logic beyond just custom parameter naming).

        .. note::
            Strategy adapters can override validation behavior at two levels of abstraction:

            1. **Parameter naming only** (simpler): Override
               :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.get_named_params_for_schedule_validation`
               to provide custom parameter names while using the default validation logic from
               :meth:`~finetuning_scheduler.fts_supporters.ScheduleParsingMixin._validate_ft_sched`.

            2. **Full validation logic** (more control): Override this method to completely customize
               the validation process.

            Choose the approach that best suits your use case. Most adapters only need to override
            :meth:`get_named_params_for_schedule_validation` to provide custom parameter names.

        .. deprecated:: 2.15
            Customizing parameter naming through either level is deprecated in favor of a
            :class:`~finetuning_scheduler.core.views.TunableView` attached to the module, and will be unsupported no
            earlier than 2.17. Overriding this method to add strategy-specific checks (calling ``super()``) is
            unaffected.

        Returns:
            tuple[int, int]: A tuple of ints specifying:
                1. The depth of the final scheduled phase
                2. The maximum epoch watermark explicitly specified in the schedule
        """
        # Import here to avoid circular dependency
        from finetuning_scheduler.lightning.fts_supporters import ScheduleParsingMixin
        from typing import cast

        rank_zero_debug(
            f"[base StrategyAdapter.validate_ft_sched] Validating schedule for "
            f"{self.pl_module.__class__.__name__}"
        )
        # Delegate to the mixin's implementation by default.
        return ScheduleParsingMixin._validate_ft_sched(cast(ScheduleParsingMixin, self.fts_handle))

    def gen_ft_schedule(self, dump_loc: str | os.PathLike) -> os.PathLike | None:
        """Generate the default fine-tuning schedule using a naive, 2-parameters per-level heuristic.

        This method can be overridden by :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`
        subclasses to customize schedule generation for specific strategies (e.g., using strategy-specific
        parameter naming conventions).

        Args:
            dump_loc: The directory to which the generated schedule (.yaml) should be written

        Returns:
            os.PathLike: The path to the generated schedule, by default ``Trainer.log_dir`` and named after the
            :py:class:`~lightning.pytorch.core.module.LightningModule` subclass in use with the suffix
            ``_ft_schedule.yaml``)
        """
        # Import here to avoid circular dependency
        from finetuning_scheduler.lightning.fts_supporters import ScheduleImplMixin

        rank_zero_debug(
            f"[base StrategyAdapter.gen_ft_schedule] Generating schedule for {self.pl_module.__class__.__name__}"
        )
        view = self.tunable_view if self.attached_view else None
        if view is not None and (view.support().naming_scheme != "physical"
                                 or any(is_frozen_role(r) for r in view.roles().values())):
            # a view with logical names generates its schedule over them, and one with frozen roles leaves them out
            from finetuning_scheduler.core.compile import view_default_schedule

            return ScheduleImplMixin.save_schedule(f"{self.pl_module.__class__.__name__}_ft_schedule.yaml",
                                                   view_default_schedule(view), dump_loc)
        return ScheduleImplMixin._gen_ft_schedule_impl(self.pl_module, dump_loc)
