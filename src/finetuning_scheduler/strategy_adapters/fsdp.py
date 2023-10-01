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
Fine-Tuning Scheduler Fully Sharded Data Parallel (FSDP) Strategy Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` that extends Fine-Tuning Scheduler to support
Fully Sharded Data Parallel training.

"""
import itertools
import logging
import os
import re
import warnings
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from functools import partial, partialmethod, wraps
from pprint import pformat
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

import torch
from lightning.fabric.strategies.fsdp import _get_full_state_dict_context, _setup_activation_checkpointing
from lightning.fabric.utilities import rank_zero_info, rank_zero_warn
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_1_13,
    _TORCH_GREATER_EQUAL_2_0,
    _TORCH_GREATER_EQUAL_2_1,
)
from lightning.pytorch.strategies.strategy import Strategy
from lightning.pytorch.trainer.connectors.checkpoint_connector import _CheckpointConnector
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_debug
from torch import Tensor
from torch.distributed import all_gather_object
from torch.optim import Optimizer

from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

_distributed_available = torch.distributed.is_available()
_min_fsdp_available = _TORCH_GREATER_EQUAL_1_13 and _distributed_available

if _distributed_available:
    if _TORCH_GREATER_EQUAL_2_1:
        from torch.distributed.fsdp._common_utils import _get_param_to_fqns, _is_fsdp_flattened
        from torch.distributed.fsdp.wrap import _Policy

        from finetuning_scheduler.strategy_adapters._wrap_utils import NameDrivenPolicy
    elif _TORCH_GREATER_EQUAL_2_0:
        from torch.distributed.fsdp._common_utils import _get_param_to_fqns, _is_fsdp_flattened
        from torch.distributed.fsdp.wrap import _FSDPPolicy as _Policy  # type: ignore[no-redef]

        from finetuning_scheduler.strategy_adapters._wrap_utils import NameDrivenPolicy
    elif _TORCH_GREATER_EQUAL_1_13:
        _Policy = object  # type: ignore[assignment,misc]
        NameDrivenPolicy = object  # type: ignore[assignment,misc]
        from torch.distributed.fsdp._utils import _is_fsdp_flattened  # type: ignore[no-redef]
        from torch.distributed.fsdp.fully_sharded_data_parallel import _get_param_to_unflat_param_names

if _min_fsdp_available:
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FLAT_PARAM,
        FullyShardedDataParallel,
        OptimStateKeyType,
    )
    from torch.distributed.fsdp.wrap import _ConfigAutoWrap, _or_policy, lambda_auto_wrap_policy, wrap

    _get_params_to_fqns = _get_param_to_fqns if _TORCH_GREATER_EQUAL_2_0 else _get_param_to_unflat_param_names


class FSDPStrategyAdapter(StrategyAdapter):
    r"""
    A :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` that extends
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` (FTS) to support flexible, multi-phase, scheduled fine-tuning
    with the Fully Sharded Data Parallel (FSDP) strategy
    (:external+pl:class:`~lightning.pytorch.strategies.fsdp.FSDPStrategy`).

    As with standard FSDP usage, FSDP wrapping of a :external+pl:class:`~lightning.pytorch.core.module.LightningModule`
    can be performed either by providing an ``auto_wrap_policy`` or (for maximal control) by overriding the
    ``configure_model`` method of :external+pl:class:`~lightning.pytorch.core.module.LightningModule` and
    manually wrapping the module.

    In order to support multi-phase scheduled fine-tuning with FSDP, FTS's key precondition is that the defined
    fine-tuning schedule phases have disjoint sets of FSDP-flattened parameters (i.e. ``FlatParameter`` s, which are
    created when wrapping a set of modules in a FSDP instance/unit). This constraint is derived from the fact that the
    ``requires_grad`` attribute currently must be the same for all parameters flattened into the same ``FlatParameter``
    (for PyTorch < ``2.1.0`` or if in ``use_orig_params=False`` mode).

    In order to support multi-phase scheduled fine-tuning with FSDP in ``use_orig_params=False`` mode, FTS's key
    precondition is that the defined fine-tuning schedule phases have disjoint sets of FSDP-flattened parameters (i.e.
    ``FlatParameter`` s, which are created when wrapping a set of modules in a FSDP instance/unit). This constraint is
    derived from the fact that (for PyTorch < ``2.1.0`` or ``use_orig_params=False`` mode) the ``requires_grad``
    attribute must be the same for all parameters flattened into the same ``FlatParameter``.

    To facilitate module wrapping in alignment with fine-tuning schedule phases, FTS provides the
    :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` feature which allows users to
    provide module name-based complements to a given ``auto_wrap_policy``. See the :ref:`fsdp-fine-tuning-example`
    tutorial for a concrete example and additional guidance.

    FTS will attempt to validate that the module is wrapped in a manner that aligns with the defined fine-tuning
    schedule phases prior to the start of training and provided detailed feedback for the user if a misalignment is
    discovered.

    .. warning::

        :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` is in BETA and subject to change. The
        interface can bring breaking changes and new features with the next release of PyTorch.

    .. note::

       The ``no_decay`` attribute that FTS supports on
       :external+pl:class:`~lightning.pytorch.core.module.LightningModule` with the base
       :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` is not currently supported in the context of
       FSDP fine-tuning.

    .. tip::

       Because of inter-module dependencies (among other reasons), wrapping every submodule in its own separate FSDP
       instance is often not a viable approach to ensuring fine-tuning schedule/module wrapping alignment. Starting
       with a provided ``auto_wrap_policy`` (e.g. ``transformer_auto_wrap_policy``) and providing module name-based
       complements as needed using
       :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` is often the most expedient
       approach to auto-wrapping in alignment with a fine-tuning schedule. As always, if needed, one can override
       ``configure_model`` and manually wrap a given
       :external+pl:class:`~lightning.pytorch.core.module.LightningModule` to align with a desired fine-tuning schedule.

    .. deprecated:: v2.1.0

        :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` now uses the ``configure_model`` hook
        rather than the deprecated ``configure_sharded_model`` hook to apply the relevant model wrapping. See `this PR
        <https://github.com/Lightning-AI/lightning/pull/18004>`_ for more context regarding
        ``configure_sharded_model`` deprecation.
    """

    _fsdp_flat_to_unflat_mapping: Dict
    _fsdp_unflat_to_flat_mapping: Dict
    _ft_schedule_module_map: Dict
    _unscheduled_params: List
    _use_orig_params: bool
    _allow_mixed_req_grad: bool
    _rank_zero_logger: logging.Logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")

    def __init__(self, awp_overrides: Optional[List] = None, *args: Any, **kwargs: Any) -> None:
        """The only user-facing configuration for
        :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` is
        :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides`, an optional list of
        module names that should be wrapped in separate FSDP instances, complementing the modules that would be
        individually wrapped by ``auto_wrap_policy`` provided in the
        :external+pl:class:`~lightning.pytorch.strategies.fsdp.FSDPStrategy` strategy
        configuration.

        Args:
            awp_overrides (Optional[List]): A list of module names to wrap in separate FSDP instances (i.e.,
                ``auto_wrap_policy`` overrides). Only applicable when complementing/overriding an ``auto_wrap_policy``
                provided in the
                :external+pl:class:`~lightning.pytorch.strategies.fsdp.FSDPStrategy`
                strategy configuration. Override lists will be ignored when manually wrapping modules via a
                ``configure_model`` method. If the named modules cannot be found, an exception will be thrown.
                Defaults to None.

        Attributes:
            awp_overrides: A list of module names to wrap in separate FSDP instances.
        """
        if not _TORCH_GREATER_EQUAL_1_13:  # because `lambda_auto_wrap_policy` is used to adapt FSDP for FTS
            raise MisconfigurationException(
                "Use of Fine-Tuning Scheduler with FSDP (via the `FSDPStrategyAdapter`) is supported from PyTorch"
                " v1.13.0 onwards."
            )
        super().__init__(*args, **kwargs)
        self.awp_overrides = awp_overrides or []
        self._min_wrap_validated: bool = False
        self._suppress_cm_warns()
        self.exec_ft_phase = partial(StrategyAdapter.base_ft_phase, translation_func=self.logical_param_translation)

    @property
    def lightning_restore_optimizer(self) -> bool:
        """Disable Lightning's restoration of the optimizer to allow FTS to implement special handling.

        Returns:
            bool: Returns ``False`` to allow FTS control over optimizer restoration.
        """
        return False

    def on_before_init_fts(self) -> None:
        """In this hook executed immediately before
        :meth:`~finetuning_scheduler.fts_supporters.ScheduleImplMixin.init_fts`, to accommodate FSDP we:

        1. Disable Lightning's restoration of the optimizer to allow us to implement special handling
        2. Prune ``no_decay`` specification since it is not currently supported in the context of FSDP fine-tuning
        3. Validate the :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` configuration
        4. Configure FTS wrapping of the provided :external+pl:class:`~lightning.pytorch.core.module.LightningModule`
           to either use the provided ``LightningModule.configure_model`` method (if present) or a provided
           ``auto_wrap_policy``.
        """
        # hack to avoid subclassing FSDP strategy for this adapter
        setattr(Strategy, "lightning_restore_optimizer", self.lightning_restore_optimizer)
        setattr(self.pls_handle, "optimizer_state", self.optimizer_state)
        self._use_orig_params = self.pls_handle.kwargs.get("use_orig_params", False)
        # for PyTorch >= `2.1.0` w/ `use_orig_params`, schedule/wrapping alignment constraints can be relaxed
        self._allow_mixed_req_grad = self._use_orig_params and _TORCH_GREATER_EQUAL_2_1
        self._prune_nodecay()
        self._validate_awp_overrides()
        if is_overridden("configure_model", self.pl_module):
            rank_zero_info(
                "You have overridden the `LightningModule.configure_model` hook. Fine-Tuning Scheduler"
                " will attempt to validate that you have wrapped the provided model in a manner that aligns with the"
                " defined fine-tuning schedule phases. If you would like to have Fine-Tuning Scheduler"
                " automatically wrap your model according to a given auto wrap policy, avoid overriding"
                " `configure_model` in your module and provide the desired auto wrap policy."
            )
            csm_func = self._wrapped_configure_model(self.pl_module.configure_model)
            setattr(self.pl_module, "configure_model", csm_func)
        else:
            setattr(self.pl_module, "configure_model", self._fts_auto_configure_model)

    def on_after_init_fts(self) -> None:
        """To accommodate FSDP, we defer executing the first fine-tuning phase that would otherwise be executed in
        this hook, which fires in :class:`~finetuning_scheduler.fts.FinetuningScheduler` setup immediately after
        :meth:`~finetuning_scheduler.fts_supporters.ScheduleImplMixin.init_fts`"""

    def on_before_fts_fit_start(self) -> None:
        """In this hook executed immediately before the :class:`~finetuning_scheduler.fts.FinetuningScheduler`
        :meth:`~finetuning_scheduler.fts.FinetuningScheduler.on_fit_start` hook begins, we ensure the provided
        fine-tuning schedule and FSDP wrapped :external+pl:class:`~lightning.pytorch.core.module.LightningModule` are
        appropriately aligned and valid. If the fine-tuning schedule and wrapped module are detected to be incompatible,
        detailed feedback is provided to the user (which is why multiple checks are aggregated before returning any
        alignment exceptions).

        Raises:
            MisconfigurationException: If any FTS FSDP fine-tuning schedule/module wrapping alignment exceptions are
                thrown. The provided exceptions provide detailed feedback for the user to address the misalignment.
        """
        world_feedback_set: Set = set()
        world_feedback = [[None] for _ in range(self.pls_handle.world_size)]
        all_gather_object(world_feedback, self._validate_fsdp_fts_config())
        for feedback in world_feedback:
            world_feedback_set.update(feedback)  # feedback could be rank-specific
        if world_feedback_set:
            exceptions, debug_msgs = [], []
            for msg in world_feedback_set:
                if msg[0] == "ERROR":
                    exceptions.append(MisconfigurationException(msg[1]))
                else:
                    debug_msgs.append(msg[1])  # currently, diagnostics are for DEBUG level only
            if debug_msgs:
                for debug_msg in debug_msgs:
                    rank_zero_debug(debug_msg)
            if exceptions:
                raise MisconfigurationException(*exceptions)

    def on_before_restore_optimizers_and_lrs(self) -> None:
        """Allow the :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` to override the default
        ``load_optimizer_state_dict`` method.

        This is necessary so we can allow FSDP to manage the movement of restored optimizer states to the relevant
        devices.
        """
        checkpoint_connector = self.pl_module.trainer._checkpoint_connector

        # Restore the optimizer states from the pre-loaded checkpoint.
        self.load_optimizer_state_dict(checkpoint_connector)

    def load_optimizer_state_dict(self, checkpoint_connector: _CheckpointConnector) -> None:
        """Override the default ``load_optimizer_state_dict`` method so that we can allow FSDP to manage the
        movement of restored optimizer states to the relevant devices.

        Args:
            checkpoint_connector (_CheckpointConnector): The ``_CheckpointConnector`` associated with the current
                training session.
        """
        if not _TORCH_GREATER_EQUAL_2_0:
            rank_zero_debug(
                "Note that saving/restoring optimizer state using the FSDP strategy with PyTorch < 2.0 is not"
                " supported by Lightning. Bypassing restoration of optimizer state."
            )
            return
        optimizer_states = checkpoint_connector._loaded_checkpoint["optimizer_states"]

        assert self.pls_handle.model is not None

        # rank0_only should be false to enable loading of the optimizer state on all ranks
        # irrespective of `use_orig_params` mode, we start with a full, unflattened, unsharded, consolidated osd
        # we then ensure the local osd is properly keyed and transformed for loading into each rank's local optimizer
        with _get_full_state_dict_context(
            self.pls_handle.model, world_size=self.pls_handle.world_size, rank0_only=False
        ):
            for optimizer, opt_state in zip(self.pls_handle.optimizers, optimizer_states):

                # usually, this will basically be a noop since FTS should be restoring osd saved with param fqn keys
                opt_state = FullyShardedDataParallel.rekey_optim_state_dict(
                    opt_state, OptimStateKeyType.PARAM_NAME, self.pls_handle.model
                )

                opt_state = FullyShardedDataParallel.optim_state_dict_to_load(
                    optim_state_dict=opt_state,
                    model=self.pls_handle.model,
                    optim=optimizer,
                )

                optimizer.load_state_dict(opt_state)

    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
        """Override the default ``optimizer_state`` method so that we can unify `use_orig_params` code-paths and
        save a full, consolidated optimizer state dict to be restored via ``load_optimizer_state_dict``.

        Args:
            optimizer (Optimizer): The optimizer instance for which a full optimizer state dict will be captured.

        Returns:
            Dict[str, Tensor]: The consolidated full optimizer state dict (if on rank 0, otherwise an empty dict).
        """
        if not _TORCH_GREATER_EQUAL_2_0:
            rank_zero_debug(
                "Note that saving/restoring optimizer states using the FSDP strategy with PyTorch < 2.0 is not"
                " supported by Lightning. Bypassing saving of optimizer state."
            )
            return {}
        assert self.pls_handle.model is not None

        # irrespective of `use_orig_params` mode, we need the full, unflattened, unsharded, consolidated osd
        with _get_full_state_dict_context(self.pl_module, world_size=self.pls_handle.world_size, rank0_only=True):
            state_dict = FullyShardedDataParallel.optim_state_dict(self.pl_module, optimizer)

        return state_dict

    def fts_optim_transform(self, orig_pl: List, inspect_only: bool = False) -> List:
        """Because FSDP performs parameter transformations that cause the current optimizer's view of parameter
        names to diverge from the original parameter names, this parameter transformation is required for optimizer
        operations.

        Args:
            orig_pl (List): The original parameter name list before FSDP's transformation of them.
            inspect_only (bool): Whether to use the specified transform in read-only (i.e. ``inspect_only``) mode,
                avoiding any persistent state transformation that may accompany normal usage. Typically useful for state
                inspection and validation contexts.

        Returns:
            List: A transformed parameter name list that matches the current optimizer's view of them after FSDP's
                transformation of the original parameter names.
        """
        return self.fsdp_param_transform(orig_pl, inspect_only)

    def fsdp_param_transform(self, orig_thaw_pl: List, inspect_only: bool) -> List:
        """The parameter transformation function currently used by
        :meth:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.fts_optim_transform` to transform original
        parameter lists for optimizer operations.

        Args:
            orig_thaw_pl (List): The original parameter name list before FSDP's transformation of them.
            inspect_only (bool): Whether to use the specified transform in read-only (i.e. ``inspect_only``) mode,
                avoiding any persistent state transformation that may accompany normal usage. Typically useful for state
                inspection and validation contexts.

        Returns:
            List: A transformed parameter name list that matches the current optimizer's view of them after FSDP's
                transformation of the original parameter names.
        """
        flat_next_tl = {self._fsdp_unflat_to_flat_mapping[p] for p in orig_thaw_pl}
        if self._use_orig_params and not inspect_only:
            self._flat_param_thaw(flat_next_tl)
        return [n for n, p in self.pl_module.named_parameters() if p in flat_next_tl]

    def _flat_param_thaw(self, flat_next_tl: Set) -> None:
        """For FSDP modules that have been configured with ``_use_orig_params`` set to ``True``, this method
        ensures that the ``FlatParameter`` objects containing the logically original ``Parameter`` objects require
        grad when one or more of those contained original parameters are transformed for optimizer operations.

        Args:
            flat_next_tl (Set): The set of original ``Parameter`` s to transform for optimizer operations. These should
            be ``Parameter`` objects rather than ``FlatParameter`` objects because ``_use_orig_params`` is ``True`` in
            this context.
        """
        use_orig_flat_params_mods = set()
        for m in self.pl_module.modules():
            is_fsdp_managed = getattr(m, "_is_fsdp_managed_module", False)
            if is_fsdp_managed and m._fsdp_use_orig_params and getattr(m, FLAT_PARAM, None) is not None:
                use_orig_flat_params_mods.add(m)
        flat_params_to_thaw = set()
        for m in use_orig_flat_params_mods:
            for p in flat_next_tl:
                if any([p is ofp for ofp in m._flat_param._params]):  # type: ignore[union-attr]
                    flat_params_to_thaw.add((m, getattr(m, FLAT_PARAM)))
        thawed_fp_mods = set()
        for fpm, fp in flat_params_to_thaw:
            fp.requires_grad = True
            thawed_fp_mods.add(fpm)
        thawed_fp_fqns = [n + "." + FLAT_PARAM for n, m in self.pl_module.named_modules() if m in thawed_fp_mods]
        rank_zero_debug(
            "Since FSDP has been configured with `use_orig_params` set to `True`, the following `FlatParameter`s"
            " have been thawed because they contain the original parameters you specified be thawed."
            f" `FlatParameters` thawed: {os.linesep}{pformat(thawed_fp_fqns)}"
        )

    def logical_param_translation(self, param_names: List) -> List:
        """Effectively the reverse transformation of
        :meth:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.fts_optim_transform`.

        Args:
            param_names (List): A parameter name list from the current optimizer's view of them after FSDP's
                transformation of the original parameter names.

        Returns:
            List: The original parameter name list before a given FSDP's transformation.
        """
        logical_param_names = []
        for n, p in self.pl_module.named_parameters():
            if n in param_names:
                logical_param_names.extend(self._fsdp_flat_to_unflat_mapping[p])
        return logical_param_names

    def _prune_nodecay(self) -> None:
        """If the ``no_decay`` attribute is present on the provided.

        :external+pl:class:`~lightning.pytorch.core.module.LightningModule` s remove it (with a warning) because it is
        not currently supported in the context of FSDP fine-tuning.
        """
        if hasattr(self.pl_module, "no_decay") and self.pl_module.no_decay is not None:
            rank_zero_warn(
                "Specifying a `no_decay` lightning module attribute is not currently supported by the Fine-Tuning"
                f" Scheduler FSDP strategy adapter. The `no_decay` attribute currently set ({self.pl_module.no_decay})"
                " will now be unset by the adapter to allow training to proceed."
            )
            setattr(self.pl_module, "no_decay", None)

    def _suppress_cm_warns(self) -> None:
        """Because Fine-Tuning Scheduler internally leverages the ``configure_model`` method to implement FSDP
        auto-wrapping enhancements, we suppress superfluous warnings about ``configure_model`` overrides."""
        try:
            warns_to_suppress = (".*model is already wrapped.*", ".*model already contains checkpointed layers.*")
            for w in warns_to_suppress:
                warnings.filterwarnings("ignore", w, category=UserWarning)
        except Exception:
            # suppressing this message is largely cosmetic so if we cannot suppress this message for any reason at all
            # (e.g. logger rename) continue anyway
            pass

    def _validate_fsdp_fts_config(self) -> List:
        """Execute fine-tuning schedule/module wrapping misalignment checks, generating and aggregating detailed
        feedback to facilitate the user's remediation of the issue.

        Returns:
            List: Any FTS FSDP fine-tuning schedule/module wrapping misalignment feedback messages generated by the
            validation functions.
        """
        # collect all configuration feedback before returning it to the user to facilitate faster remediation
        return [cf for cf in [self._validate_min_wrap_condition(), *self._validate_fsdp_phases_disjoint()] if cf]

    def _validate_awp_overrides(self) -> None:
        """Expand any regex expressions specified in
        :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides`.

        Raises:
            MisconfigurationException: If a specified module name or regex does not resolve to at least one named
                module.
        """
        if not self.awp_overrides:
            return
        if is_overridden("configure_model", self.pl_module):
            rank_zero_warn(
                "You have overridden the `LightningModule.configure_model` hook but also provided"
                " an `awp_overrides` configuration. Since `awp_overrides` only applies to configurations that use"
                f" policy-based FSDP wrapping, this configuration ({self.awp_overrides}) will be unset and not applied."
            )
            self.awp_overrides = []
            return
        named_modules = dict(self.pl_module.named_modules()).keys()
        resolved_modules = []
        for m in self.awp_overrides:
            regex_modules = []
            explicit_mods = False
            if m in named_modules:
                explicit_mods = True
                resolved_modules.append(m)
            else:
                mpat = re.compile(m)
                regex_modules = [m for m in named_modules if mpat.match(m)]
                resolved_modules.extend(regex_modules)
            if not (regex_modules or explicit_mods):
                raise MisconfigurationException(
                    f"The module or regex '{m}' specified in `awp_overrides` did not match any named modules in the"
                    " provided model."
                )
        self.awp_overrides = resolved_modules

    @staticmethod
    def _phasewise_intersection(phase_lists: List[List]) -> Set:
        """Calculates a phase-wise intersection of elements (whether modules or parameters)

        Args:
            phase_lists (List[List]): Element lists (modules or parameters) for each fine-tuning schedule phase.

        Returns:
            Set: The set of elements present in more than one phase.
        """
        elems = Counter(list(itertools.chain(*phase_lists)))
        unique_elems = Counter(list(set().union(*phase_lists)))
        elems.subtract(unique_elems)
        dup_elems = set(elems.elements())
        return dup_elems

    def _log_nonzero_local_shards(self) -> Optional[str]:
        """If debugging diagnostics are requested, evaluate whether there are any ranks with no (non-zero sized)
        parameter shards and if so, provide parameter shard allocation debugging info for the user.

        Returns:
            Optional[str]: Per-rank debugging info distilling relevant parameter shard allocation.
        """
        curr_optimizer_params = [p for pg in self.pls_handle._optimizers[0].param_groups for p in pg["params"]]
        if not any(p.shape[0] for p in curr_optimizer_params if p.requires_grad):
            params_w_ls = set()
            for curr_optim_p in curr_optimizer_params:
                for fsdp_mod in FullyShardedDataParallel.fsdp_modules(self.pl_module):
                    fp = fsdp_mod._flat_param
                    assert fp is not None
                    assert isinstance(fp._params, Iterable)
                    assert isinstance(fp._shard_param_infos, Iterable)
                    for fp_p, fp_shard_info in zip(fp._params, fp._shard_param_infos):
                        if fp_p is curr_optim_p and not fp_shard_info[0]:
                            w_local = [p for p, spi in zip(fp._params, fp._shard_param_infos) if spi[0]]
                            params_w_ls.update(w_local)
            params_w_ls_names = [self._fsdp_flat_to_unflat_mapping[lsp][0] for lsp in params_w_ls]
            params_w_ls_names.sort()
            rank_specific_advice = (
                "Below are parameters in the same FSDP module as those currently specified in phase 0 but that DO have "
                f"local shards for rank {self.pl_module.global_rank}: "
                f"{os.linesep}{pformat(params_w_ls_names)}{os.linesep}"
            )
            local_shard_advice = (
                f"The global rank {self.pl_module.global_rank} optimizer has no (non-zero sized) local shards of the "
                "phase 0 fine-tuning schedule parameters. \n"
                f"Additional rank-specific details for **RANK {self.pl_module.global_rank}**: {os.linesep}"
                f"{rank_specific_advice}"
            )
            return local_shard_advice

    def _validate_fsdp_phases_disjoint(self) -> Tuple:
        """Validate that the defined schedule does not specify any wrapped module or parameter in multiple phases.

        Returns:
            Tuple: Any fine-tuning schedule/wrapped module misalignment feedback messages to be provided to the user.
        """
        feedback_errors: List[str] = []
        feedback_nonerrors: List[str] = []
        if self._allow_mixed_req_grad:
            rank_zero_debug(
                "Bypassing FSDP-specific phase disjointness validation because `use_orig_params` is "
                "``True`` and PyTorch is >= `2.1.0`"
            )
            assert self.pl_module._trainer is not None
            # check only required for mixed-precision training with DEBUG level logging requested
            if (
                self.pl_module._trainer.precision in ("16-mixed", "bf16-mixed", "16-true", "bf16-true")
                and self._rank_zero_logger.level <= 10
            ):
                has_no_local_shards = self._log_nonzero_local_shards()
                if has_no_local_shards:
                    feedback_nonerrors.append(has_no_local_shards)
        fsdp_dup_params = set()
        unsched_dup_params = set()
        scheduled_mod_lists = [list(self._ft_schedule_module_map[d]) for d in self._ft_schedule_module_map.keys()]
        ft_sched_dup_mods = FSDPStrategyAdapter._phasewise_intersection(scheduled_mod_lists)
        fsdp_dup_params = self._phase_unaligned_fsdp_params()
        if not fsdp_dup_params:  # unsched_dup_params will be a superset of fsdp_dup_params
            unsched_dup_params = self._phase_unaligned_fsdp_params(check_unsched=True)
        if ft_sched_dup_mods:
            feedback_errors.append(self._module_overlap_feedback(ft_sched_dup_mods))
        if unsched_dup_params:  # conditionally emphasize parameters not included in the fine-tuning schedule
            feedback_errors.append(self._fsdp_param_phase_overlap_feedback(unsched_dup_params, unsched_msg=True))
        elif fsdp_dup_params:
            feedback_errors.append(self._fsdp_param_phase_overlap_feedback(fsdp_dup_params))
        feedback_msgs = [("ERROR", fe) for fe in feedback_errors]
        for fw in feedback_nonerrors:
            feedback_msgs.append(("DEBUG", fw))
        return tuple(feedback_msgs)

    @staticmethod
    def _module_has_fp(submodule: torch.nn.Module) -> bool:
        """Evaluate whether a given module has any FSDP-flattened params.

        Args:
            submodule (torch.nn.Module): The module to inspect for FSDP-flattened params.

        Returns:
            bool: ``True`` if the specified module contains any FSDP-flattened params, otherwise ``False``.
        """
        return any(_is_fsdp_flattened(param) for param in submodule.parameters())

    def _validate_min_wrap_condition(self) -> Optional[Tuple]:
        """Validate (prior to optimizer validation via Lightning that occurs after a potential FTS phase 0
        override) that at least scheduled phase 0 contains FSDP flattened parameters with ``requires_grad`` set to
        ``True``.

        Returns:
            Optional[str]: Error message for the user if the first fine-tuning phase does not include one or more FSDP
                flattened parameters.
        """
        has_flattened = False
        # function configuration to inspect at a module level:
        mod_cfg = (self._ft_schedule_module_map[0], FSDPStrategyAdapter._module_has_fp, self.pl_module.get_submodule)
        # function configuration to inspect at a parameter level:
        param_cfg = (self.fts_handle.ft_schedule[0]["params"], _is_fsdp_flattened, self.pl_module.get_parameter)

        def inspect_flattened(iter_inspect: Iterable, inspect_func: Callable, inspect_prepare: Callable) -> bool:
            return any(inspect_func(inspect_prepare(i)) for i in iter_inspect)

        has_flattened = inspect_flattened(*mod_cfg) if not self._allow_mixed_req_grad else inspect_flattened(*param_cfg)
        if not has_flattened:
            fts_p0_err = (
                "Training an FSDP wrapped model requires one or more FSDP parameters to be included in the optimizer."
                " The `configure_model method or auto_wrap_policy` you have specified did not wrap any of the"
                " layers specified in fine-tuning phase 0. Ensure your overridden `configure_model` method or"
                " auto_wrap_policy wraps at least one module included in phase `0`."
            )
            return ("ERROR", fts_p0_err)

    def _phase_unaligned_fsdp_params(self, check_unsched: bool = False) -> Set:
        """Inspect the fine-tuning schedule and FSDP-wrapped module for parameters that are unaligned with the FSDP
        wrapped module.

        Args:
            check_unsched (bool, optional): Whether to include parameters not in the fine-tuning schedule in the
                disjointness check. The unscheduled parameter disjointness check will only be executed if the
                scheduled parameter phase disjointness check passes (since the unscheduled check is a superset of the
                scheduled one). Defaults to False.

        Returns:
            Set: The set of module parameters in more than one fine-tuning phase.
        """
        fsdp_param_sets: dict = {}
        inspection_map = deepcopy(self.fts_handle.ft_schedule)
        if check_unsched:
            inspection_map[-1] = {"params": self._unscheduled_params}
        for d, pl in inspection_map.items():
            fsdp_param_sets[d] = set()
            for lp in pl["params"]:
                fsdp_param_sets[d].update(self._fsdp_flat_to_unflat_mapping[self._fsdp_unflat_to_flat_mapping[lp]])
        fsdp_phase_lists = [list(fsdp_param_sets[d]) for d in fsdp_param_sets.keys()]
        return FSDPStrategyAdapter._phasewise_intersection(fsdp_phase_lists)

    def _fsdp_param_phase_overlap_feedback(self, dup_params: Set, unsched_msg: bool = False) -> str:
        """Generate parameter-level phase overlap feedback for the user, identifying owning FSDP instances
        associated with parameters that span more than one fine-tuning phase.

        Args:
            dup_params (Set): The set of module parameters in more than one fine-tuning phase.
            unsched_msg (bool, optional): Whether to indicate the misaligned parameters were not included in the
                fine-tuning schedule. Defaults to False.

        Returns:
            str: User feedback detailing the FSDP instances associated with any parameters spanning more than one
            fine-tuning phase.
        """

        def get_fsdp_owner(lp: str) -> str:
            owner = "no owner found"
            for fsdp_mod in FullyShardedDataParallel.fsdp_modules(self.pl_module):
                for p in fsdp_mod.params:
                    if self._fsdp_unflat_to_flat_mapping[lp] is p:
                        owner = fsdp_mod.module._get_name()
            return owner

        dup_params_fsdp_mapping = {lp: get_fsdp_owner(lp) for lp in dup_params}
        unsched_param_msg = (
            "In this particular case, there are parameters not included in your fine-tuning schedule that span more"
            " than one fine-tuning phase.\nHINT: parameters associated with unwrapped modules will be included in the"
            " top-level (aka 'root') FSDP instance so ensuring all modules associated with fine-tuning scheduled"
            " parameters are wrapped separately from the top-level FSDP instance may avoid triggering this exception."
        )
        warn_msg = (
            "\n\nFine-tuning schedule phases do not have disjoint FSDP-flattened parameter sets. Because the"
            " `requires_grad` attribute of FSDP-flattened parameters currently must be the same for all flattened"
            " parameters (for PyTorch < ``2.1.0`` or if in ``use_orig_params=False`` mode), fine-tuning schedules must"
            " avoid thawing parameters in the same FSDP-flattened parameter in different phases. Please ensure"
            " parameters associated with each phase are wrapped in separate phase-aligned FSDP instances.\n\n"
            f"""{unsched_param_msg if unsched_msg else ''}\n\n"""
            "The following logical parameters are associated with an FSDP-flattened parameter that spans more than one"
            " fine-tuning phase. The mapping of each logical parameter with the module name wrapped by its associated"
            " FSDP instance is provided below:\n"
            f"{pformat(dup_params_fsdp_mapping)}{os.linesep}"
        )
        return warn_msg

    def _module_overlap_feedback(self, dup_mods: Set) -> str:
        """Generate module-level phase overlap feedback for the user, identifying owning FSDP instances associated
        with modules that span more than one fine-tuning phase.

        Args:
            dup_mods (Set): The set of module parameters in more than one fine-tuning phase.

        Returns:
            str: User feedback detailing the FSDP instances associated with any modules spanning more than one
            fine-tuning phase.
        """
        ft_sched = self.fts_handle.ft_schedule
        dup_mod_dict = {
            m: list(
                itertools.chain(
                    *[self._fsdp_flat_to_unflat_mapping[p] for p in self.pl_module.get_submodule(m).parameters()]
                )
            )
            for m in dup_mods
        }
        phase_mod_intersect: Dict = {}
        for m, plist in dup_mod_dict.items():
            phase_mod_intersect[m] = {}
            for phase in ft_sched.keys():
                if set(plist).intersection(set(ft_sched[phase]["params"])):
                    phase_mod_intersect[m][phase] = set(plist).intersection(set(ft_sched[phase]["params"]))
        warn_msg = (
            "Fine-tuning schedule phases do not have disjoint module sets. FSDP currently wraps at a module level"
            " which requires fine-tuning schedules avoid thawing parameters of the same module in different phases."
            " The following modules span fine-tuning phases (with associated parameters by phase):"
            f" {os.linesep}{phase_mod_intersect}"
        )
        return warn_msg

    def _fts_auto_configure_model(self) -> None:
        """Apply the ``auto_wrap_policy`` provided by the user and generate the relevant module and parameter-level
        internal mappings that allow the FTS FSDP adapter to translate and orchestrate a fine-tuning schedule.

        1. Generate a mapping of fine-tuning schedule phases to associated modules
        2. Apply the provided ``auto_wrap_policy`` (composed w/ any ``awp_overrides``) to the user's ``LightningModule``
        3. After module wrapping, generate parameter-level bi-directional translations between unflat (original) and
            flat (FSDP-flattened) parameters.

        Raises:
            MisconfigurationException: If the module is already FSDP-wrapped before applying the ``auto_wrap_policy``.
        """
        for m in self.pl_module.modules():
            # if the model is already wrapped with FSDP
            if isinstance(m, FullyShardedDataParallel):
                raise MisconfigurationException(
                    "The provided model is already wrapped by FSDP. Cannot apply an FSDP auto-wrapping policy along"
                    " fine-tuning schedule phase boundaries if the model is already wrapped."
                )
        self._gen_ft_sched_module_map()
        self._fts_auto_wrap()
        self._after_configure_model()

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

    def _fts_auto_wrap(self) -> None:
        """Apply the provided ``auto_wrap_policy`` within a context-manager that composes any ``awp_overrides``
        directives with the policy.

        Subsequently, apply activation checkpointing wrappers if requested
        """
        if self.pls_handle.kwargs.get("auto_wrap_policy", None):
            with self._enable_name_based_overrides():
                for n, m in self.pl_module.named_children():
                    setattr(self.pl_module, n, wrap(m))
        else:
            rank_zero_warn(
                "Wrapping the provided model in an outer FSDP module since neither an ``auto_wrap_policy`` "
                "nor a manual ``configure_model`` method for wrapping have been provided. This "
                "configuration is often (but not always) degenerate and unintended by the user."
            )
            for n, m in self.pl_module.named_children():
                setattr(self.pl_module, n, wrap(m))

        # apply wrappers to enable activation checkpointing if requested
        if self.pls_handle._activation_checkpointing_kwargs:
            _setup_activation_checkpointing(
                module=self.pl_module, activation_checkpointing_kwargs=self.pls_handle._activation_checkpointing_kwargs
            )

    def _after_configure_model(self) -> None:
        """Generate the parameter-level bi-directional translations the FTS FSDP adapter requires and then execute
        the previously deferred first fine-tuning phase."""
        assert isinstance(self.fts_handle.ft_schedule, Dict)  # TODO: move/consolidate ft_schedule assertions
        self._init_fsdp_param_map()
        _, self.fts_handle._fts_state._curr_thawed_params = self.exec_ft_phase(
            self.pl_module,
            thaw_pl=self.fts_optim_transform(self.fts_handle.ft_schedule[0]["params"]),
            init_thaw=True,
        )

    def _init_fsdp_param_map(self) -> None:
        """Generate parameter-level bi-directional translations between unflat (original) and flat (FSDP-flattened)
        parameters."""
        self._fsdp_flat_to_unflat_mapping = _get_params_to_fqns(self.pl_module)
        self._fsdp_unflat_to_flat_mapping = {
            up: fpn for fpn, upl in self._fsdp_flat_to_unflat_mapping.items() for up in upl
        }

    def _wrapped_configure_model(self, csm_func: Callable) -> Callable:
        """If the user has overridden ``configure_model`` in their ``LightningModule``, wrap the user's
        explicit wrapping method with the required
        :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` methods.

        Args:
            csm_func (Callable): The user's overridden ``LightningModule.configure_model`` method

        Returns:
            Callable: The user's overridden ``LightningModule.configure_model`` method wrapped with this
            adapter's internal implementation methods.
        """

        @wraps(csm_func)
        def wrapped_func() -> None:
            self._gen_ft_sched_module_map()
            csm_func()
            self._after_configure_model()

        return wrapped_func

    @contextmanager
    def _enable_name_based_overrides(self) -> Generator:
        """A context manager that enables name-driven overriding of a given ``auto_wrap_policy`` with a list of
        module names.

        The composition of module name-based wrapping directives with a given ``auto_wrap_policy`` is achieved here by:
        1. Generating an object id-based module name mapping lambda and passing it to the standard
            ``lambda_auto_wrap_policy``.
        2. Composing the user's provided ``auto_wrap_policy`` with the above name-based policy using the standard
            ``_or_policy``.

        Yields:
            Generator: A wrapping context that applies the provided ``auto_wrap_policy`` along with any user specified
                name-based complements to that policy.
        """
        auto_wrap_policy_handle = _ConfigAutoWrap.kwargs.pop("auto_wrap_policy", None)
        override_ids = [id(m) for n, m in self.pl_module.named_modules() if n in self.awp_overrides]
        name_based_override_policy: Union[NameDrivenPolicy, Callable]
        if _TORCH_GREATER_EQUAL_2_0 and isinstance(auto_wrap_policy_handle, _Policy):
            name_based_override_policy = NameDrivenPolicy(auto_wrap_policy_handle, override_ids=override_ids)
        else:  # Callable policy implementation path
            name_driven_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda m: id(m) in override_ids)
            name_based_override_policy = partial(_or_policy, policies=[auto_wrap_policy_handle, name_driven_policy])
        _ConfigAutoWrap.kwargs["auto_wrap_policy"] = name_based_override_policy
        try:
            yield
        finally:
            _ConfigAutoWrap.kwargs["auto_wrap_policy"] = auto_wrap_policy_handle

    fts_optim_inspect = partialmethod(fts_optim_transform, inspect_only=True)
