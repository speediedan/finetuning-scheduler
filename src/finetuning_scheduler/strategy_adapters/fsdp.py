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

Classes to extend Fine-Tuning Scheduler support of complex or custom training strategies

"""
import itertools
import os
import re
from collections import Counter
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

import torch
from lightning_lite.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies.fully_sharded_native import _fsdp_available
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden

from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

if _fsdp_available:
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        _get_param_to_unflat_param_names,
        FullyShardedDataParallel,
    )
    from torch.distributed.fsdp.wrap import _ConfigAutoWrap, wrap
else:
    FullyShardedDataParallel = None  # type: ignore[misc,assignment]
    MixedPrecision = None  # type: ignore[misc,assignment]
    BackwardPrefetch = None  # type: ignore[misc,assignment]
    CPUOffload = None  # type: ignore[misc,assignment]


class FSDPStrategyAdapter(StrategyAdapter):
    """_summary_

    Args:
        StrategyAdapter (_type_): _description_
    """

    def __init__(self, awp_overrides: Optional[List] = None, *args: Any, **kwargs: Any) -> None:
        """_summary_

        Args:
            awp_overrides (Optional[List], optional): _description_. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.awp_overrides = awp_overrides or []
        self._fsdp_flat_to_unflat_mapping: Dict
        self._fsdp_unflat_to_flat_mapping: Dict
        self._ft_schedule_module_map: Dict
        self._min_wrap_validated: bool = False
        self.exec_ft_phase = partial(StrategyAdapter.base_ft_phase, translation_func=self.logical_param_translation)

    @property
    def pl_module(self) -> LightningModule:
        return self.fts_handle.pl_module

    @property
    def lightning_restore_optimizer(self) -> bool:
        return False

    def on_before_init_fts(self) -> None:
        """Hook executed in Fine-Tuning Scheduler setup immediately before `init_fts`"""
        # hack to avoid subclassing FSDP strategy for adapter
        setattr(Strategy, "lightning_restore_optimizer", self.lightning_restore_optimizer)
        if self.awp_overrides:
            self._validate_awp_overrides()
        if is_overridden("configure_sharded_model", self.fts_handle.pl_module):
            rank_zero_info(
                "You have overridden the `LightningModule.configure_sharded_model` hook. Fine-Tuning Scheduler"
                " will validate that you have wrapped the provided model in a manner that aligns with the"
                " defined fine-tuning schedule phases. If you would like to have Fine-Tuning Scheduler"
                " automatically wrap your model in a fine-tuning phase-aligned according to a given auto wrap"
                " policy, avoid overriding `configure_sharded_model` in your module and provide the desired"
                " auto wrap policy."
            )
            csm_func = self._wrapped_configure_sharded_model(self.fts_handle.pl_module.configure_sharded_model)
            setattr(self.fts_handle.pl_module, "configure_sharded_model", csm_func)
        else:
            setattr(self.fts_handle.pl_module, "configure_sharded_model", self._fts_auto_configure_sharded_model)

    def on_after_init_fts(self) -> None:
        """Override this hook for FSDP to defer first ft schedule phase initialization until after model
        wrapped."""

    def on_before_fts_fit_start(self) -> None:
        fsdp_fts_cfg_errors = self._validate_fsdp_fts_config()
        if fsdp_fts_cfg_errors:
            raise MisconfigurationException(fsdp_fts_cfg_errors)

    def on_before_restore_optimizers_and_lrs(self) -> None:
        """summary."""
        checkpoint_connector = self.fts_handle.pl_module.trainer._checkpoint_connector
        if not checkpoint_connector._loaded_checkpoint:
            return

        # Restore the optimizer states from the pre-loaded checkpoint.
        self.load_optimizer_state_dict(checkpoint_connector)

    def load_optimizer_state_dict(self, checkpoint_connector: CheckpointConnector) -> None:
        optimizer_states = checkpoint_connector._loaded_checkpoint["optimizer_states"]
        for optimizer, opt_state in zip(self.pls_handle.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)
            # _optimizer_to_device(optimizer, self.root_device)

    def fts_optim_view(self, orig_pl: List) -> List:
        """FSDP requires an FTS adapter transformation of schedule parameters for optimizer operations."""
        return self.fsdp_param_transform(orig_pl)

    def fsdp_param_transform(self, orig_thaw_pl: List) -> List:
        flat_next_tl = {self._fsdp_unflat_to_flat_mapping[p] for p in orig_thaw_pl}
        return [n for n, p in self.fts_handle.pl_module.named_parameters() if p in flat_next_tl]

    def logical_param_translation(self, param_names: List) -> List:
        logical_param_names = []
        for n, p in self.fts_handle.pl_module.named_parameters():
            if n in param_names:
                if self._fsdp_flat_to_unflat_mapping.get(p):
                    logical_param_names.extend(self._fsdp_flat_to_unflat_mapping[p])
                else:
                    logical_param_names.append(n)
        return logical_param_names

    def _validate_fsdp_fts_config(self) -> List:
        # collect all validation errors before returning them to the user to facilitate faster remediation
        return [ce for ce in [self._validate_min_wrap_condition(), *self._validate_fsdp_phases_disjoint()] if ce]

    def _validate_awp_overrides(self) -> None:
        """Expand any regex expressions specified in an ft_schedule phase to fully qualified parameter names. If
        any shared parameter copies are explicitly specified in the schedule, the copies will be pruned from the
        schedule with a warning.

        Raises:
            MisconfigurationException: If a specified parameter or regex does not resolve to at least one parameter.
        """
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
        elems = Counter(list(itertools.chain(*phase_lists)))
        unique_elems = Counter(list(set().union(*phase_lists)))
        elems.subtract(unique_elems)
        dup_elems = set(elems.elements())
        return dup_elems

    def _phase_unaligned_fsdp_params(self) -> Set:
        fsdp_param_sets: dict = {}
        for d, pl in self.fts_handle.ft_schedule.items():
            fsdp_param_sets[d] = set()
            for lp in pl["params"]:
                fsdp_param_sets[d].update(self._fsdp_flat_to_unflat_mapping[self._fsdp_unflat_to_flat_mapping[lp]])
        fsdp_phase_lists = [list(fsdp_param_sets[d]) for d in fsdp_param_sets.keys()]
        return FSDPStrategyAdapter._phasewise_intersection(fsdp_phase_lists)

    def _fsdp_param_phase_overlap_feedback(self, dup_params: Set) -> str:
        def get_fsdp_owner(lp: str) -> str:
            owner = "no owner found"
            for fsdp_mod in FullyShardedDataParallel.fsdp_modules(self.fts_handle.pl_module):
                for p in fsdp_mod.params:
                    if self._fsdp_unflat_to_flat_mapping[lp] is p:
                        owner = fsdp_mod.module._get_name()
            return owner

        dup_params_fsdp_mapping = {lp: get_fsdp_owner(lp) for lp in dup_params}
        warn_msg = (
            "Fine-tuning schedule phases do not have disjoint FSDP-flattened parameter sets. Because the"
            " `requires_grad` attribute of FSDP-flattened parameters currently must be the same for all flattened"
            " parameters, fine-tuning schedules must avoid thawing parameters in the same FSDP-flattened parameter in"
            " different phases. Please either ensure parameters associated with each phase are wrapped in separate"
            " phase-aligned FSDP instances."
            " The following logical parameters are associated with an FSDP-flattened parameter that spans one or more"
            " fine-tuning phases. The mapping of each logical parameter and a representation of its associated FSDP"
            " instance is provided below:"
            f" {os.linesep}{dup_params_fsdp_mapping}"
        )
        return warn_msg

    def _module_overlap_feedback(self, dup_mods: Set) -> str:
        ft_sched = self.fts_handle.ft_schedule
        dup_mod_dict = {
            m: list(
                itertools.chain(
                    *[
                        self._fsdp_flat_to_unflat_mapping[p]
                        for p in self.fts_handle.pl_module.get_submodule(m).parameters()
                    ]
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

    def _validate_fsdp_phases_disjoint(self) -> Tuple:
        """Validate that the defined schedule does not specify any module in multiple phases.

        Raises:
            MisconfigurationException: Provides a list of the parameters specified in more than one phase.
        """
        mod_phase_lists = [list(self._ft_schedule_module_map[d]) for d in self._ft_schedule_module_map.keys()]
        ft_sched_dup_mods = FSDPStrategyAdapter._phasewise_intersection(mod_phase_lists)
        fsdp_dup_params = self._phase_unaligned_fsdp_params()
        fsdp_feedback_msgs = []
        if ft_sched_dup_mods:
            fsdp_feedback_msgs.append(self._module_overlap_feedback(ft_sched_dup_mods))
        if fsdp_dup_params:
            fsdp_feedback_msgs.append(self._fsdp_param_phase_overlap_feedback(fsdp_dup_params))
        return tuple(fsdp_feedback_msgs)

    def _fts_auto_configure_sharded_model(self) -> None:
        for m in self.fts_handle.pl_module.modules():
            # if the model is already wrapped with FSDP, tracing with auto-policy would fail
            if isinstance(m, FullyShardedDataParallel):
                raise MisconfigurationException(
                    "The provided model is already wrapped by FSDP. Cannot apply an FSDP auto-wrapping policy along"
                    " fine-tuning schedule phase boundaries if the model is already wrapped."
                )
        self._gen_ft_sched_module_map()
        self._fts_auto_wrap()
        self._after_configure_sharded_model()

    def _after_configure_sharded_model(self) -> None:
        assert isinstance(self.fts_handle.ft_schedule, Dict)  # TODO: move/consolidate ft_schedule assertions
        self._init_fsdp_param_map()
        _, self.fts_handle._fts_state._curr_thawed_params = self.exec_ft_phase(
            self.fts_handle.pl_module,
            thaw_pl=self.fts_optim_view(self.fts_handle.ft_schedule[0]["params"]),
            init_thaw=True,
        )

    def _wrapped_configure_sharded_model(self, csm_func: Callable) -> Callable:
        @wraps(csm_func)
        def wrapped_func() -> None:
            csm_func()
            self._after_configure_sharded_model()

        return wrapped_func

    def _wrapped_before_optim_setup(self, before_optim_setup_func: Callable) -> Callable:
        @wraps(before_optim_setup_func)
        def wrapped_func(trainer: "Trainer") -> None:
            self._after_configure_sharded_model()
            before_optim_setup_func(trainer)

        return wrapped_func

    def _init_fsdp_param_map(self) -> None:
        # TODO: make weakrefs?
        self._fsdp_flat_to_unflat_mapping = _get_param_to_unflat_param_names(self.fts_handle.pl_module)
        self._fsdp_unflat_to_flat_mapping = {
            up: fpn for fpn, upl in self._fsdp_flat_to_unflat_mapping.items() for up in upl
        }
        if not hasattr(self, "_ft_schedule_module_map"):
            self._gen_ft_sched_module_map()

    def _validate_min_wrap_condition(self) -> Optional[str]:
        wrapped_statuses = []
        for m in self._ft_schedule_module_map[0]:
            is_wrapped = isinstance(self.fts_handle.pl_module.get_submodule(m), FullyShardedDataParallel)
            wrapped_statuses.append(is_wrapped)
        if not any(wrapped_statuses):
            fts_p0_err = (
                "Training an FSDP wrapped model requires one or more FSDP parameters to be included in the optimizer."
                " The `configure_sharded_model method or auto_wrap_policy` you have specified did not wrap any of the"
                " layers specified in fine-tuning phase 0. Ensure your overridden `configure_sharded_model` method or"
                " auto_wrap_policy wraps at least one module included in phase `0`."
            )
            return fts_p0_err

    def _gen_ft_sched_module_map(self) -> None:
        assert isinstance(self.fts_handle.ft_schedule, Dict)
        module_map: Dict = {}
        for depth in self.fts_handle.ft_schedule.keys():  # type: ignore[union-attr]
            phase_params = self.fts_handle.ft_schedule[depth].get("params", [])  # type: ignore[union-attr]
            module_map[depth] = set()
            for p in phase_params:
                module_path, _, param_name = p.rpartition(".")
                mod: torch.nn.Module = self.fts_handle.pl_module.get_submodule(module_path)
                if not hasattr(mod, param_name):
                    raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")
                module_map[depth].add(module_path)
        self._ft_schedule_module_map = module_map

    def get_wrapped_name_from_unwrapped(self, unwrapped_name: str) -> str:
        for n, m in self.fts_handle.pl_module.named_modules():
            if self.fts_handle.pl_module.get_submodule(unwrapped_name) is m:
                return n
        raise MisconfigurationException(f"Module {unwrapped_name} was not found and so cannot be explicitly wrapped.")

    def _apply_awp_overrides(self) -> None:
        for om in self.awp_overrides:
            if isinstance(self.fts_handle.pl_module.get_submodule(om), FullyShardedDataParallel):
                rank_zero_warn(
                    f"You specified a module ({om}) in `awp_overrides` that the provided `auto_wrap_policy`"
                    f" ({_ConfigAutoWrap.kwargs['auto_wrap_policy'].__name__}) has already wrapped. Skipping manual"
                    " application of FSDP wrap to avoid double-wrapping of this module."
                )
            else:
                with self._enable_explicit_wrap():
                    pn, _, cn = self.get_wrapped_name_from_unwrapped(om).rpartition(".")
                    setattr(self.pl_module.get_submodule(pn), cn, wrap(self.pl_module.get_submodule(om)))

    def _fts_auto_wrap(self) -> None:
        for n, m in self.fts_handle.pl_module.named_children():
            setattr(self.fts_handle.pl_module, n, wrap(m))
        if self.awp_overrides:
            self._apply_awp_overrides()

    @contextmanager
    def _enable_explicit_wrap(self) -> Generator:
        auto_wrap_policy_handle = _ConfigAutoWrap.kwargs.get("auto_wrap_policy", None)
        _ConfigAutoWrap.kwargs["auto_wrap_policy"] = None
        try:
            yield
        finally:
            _ConfigAutoWrap.kwargs["auto_wrap_policy"] = auto_wrap_policy_handle
