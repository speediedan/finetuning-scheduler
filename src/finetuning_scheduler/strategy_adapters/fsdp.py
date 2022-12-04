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
import weakref
from contextlib import contextmanager
from copy import deepcopy
from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, List

import torch
from lightning_lite.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.strategies.fully_sharded_native import _fsdp_available
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

FTS_TMP_KEY = "__ftskey__"


class FSDPStrategyAdapter(StrategyAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fsdp_flat_to_unflat_mapping: Dict
        self._fsdp_unflat_to_flat_mapping: Dict
        self.exec_ft_phase = partial(StrategyAdapter.base_ft_phase, translation_func=self.logical_param_translation)

    def on_before_init_fts(self) -> None:
        """Hook executed in Fine-Tuning Scheduler setup immediately before `init_fts`"""
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
            setattr(self.fts_handle.pl_module, "configure_sharded_model", self._phase_aligned_configure_sharded_model)

    def on_after_init_fts(self) -> None:
        """Override this hook for FSDP to defer first ft schedule phase initialization until after model
        wrapped."""

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

    def _phase_aligned_configure_sharded_model(self) -> None:
        for m in self.fts_handle.pl_module.modules():
            # if the model is already wrapped with FSDP, tracing with auto-policy would fail
            if isinstance(m, FullyShardedDataParallel):
                raise MisconfigurationException(
                    "The provided model is already wrapped by FSDP. Cannot apply an FSDP auto-wrapping policy along"
                    " fine-tuning schedule phase boundaries if the model is already wrapped."
                )
        self._phase_constrained_auto_wrap()
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

    def _init_fsdp_param_map(self) -> None:
        # TODO: make weakrefs?
        self._fsdp_flat_to_unflat_mapping = _get_param_to_unflat_param_names(self.fts_handle.pl_module)
        self._fsdp_unflat_to_flat_mapping = {
            up: fpn for fpn, upl in self._fsdp_flat_to_unflat_mapping.items() for up in upl
        }

    def _inspect_policy_trace(self, phase_mods: List) -> Dict:
        phase_fsdp = weakref.proxy(
            wrap(
                torch.nn.ModuleDict(
                    {
                        mp.replace(".", FTS_TMP_KEY): deepcopy(self.fts_handle.pl_module.get_submodule(mp))
                        for mp in phase_mods
                    }
                )
            )
        )
        return dict(
            map(
                (
                    lambda m: (m, True)
                    if isinstance(phase_fsdp.get_submodule(m.replace(".", FTS_TMP_KEY)), FullyShardedDataParallel)
                    else (m, False)
                ),
                phase_mods,
            )
        )

    def _apply_phase_wrap(self, should_wrap: Dict, mod_paths: List) -> None:
        with self._enable_explicit_wrap():
            for modp in mod_paths:
                parent_name, _, child_name = modp.rpartition(".")
                if should_wrap[modp]:
                    setattr(
                        self.fts_handle.pl_module.get_submodule(parent_name),
                        child_name,
                        wrap(self.fts_handle.pl_module.get_submodule(modp)),
                    )

    def _validate_min_wrap_condition(self, should_wrap: Dict, mod_paths: List) -> Dict:
        if any([w for w in should_wrap.values()]):
            return should_wrap
        init_layer_sizes = {}
        for modp in mod_paths:
            init_layer_sizes[modp] = sum(p.numel() for p in self.fts_handle.pl_module.get_submodule(modp).parameters())
        force_wrap_m = max(init_layer_sizes, key=lambda k: init_layer_sizes[k])
        # TODO: update this warning message with override options
        rank_zero_warn(
            "Training an FSDP wrapped model requires one or more FSDP parameters to be included in the optimizer."
            " The auto_wrap_policy you have specified would not wrap any of the layers specified in fine-tuning"
            " phase 0. To enable training in this context, Fine-Tuning Scheduler by default wraps the largest layer"
            f" in phase 0, (in this case {force_wrap_m}). If you would like to override this behavior..."
        )
        should_wrap[force_wrap_m] = True
        return should_wrap

    def _gen_ft_sched_module_map(self) -> Dict:
        assert isinstance(self.fts_handle.ft_schedule, Dict)
        ft_schedule_module_map: Dict = {}
        for depth in self.fts_handle.ft_schedule.keys():  # type: ignore[union-attr]
            phase_params = self.fts_handle.ft_schedule[depth].get("params", [])  # type: ignore[union-attr]
            ft_schedule_module_map[depth] = set()
            for p in phase_params:
                module_path, _, param_name = p.rpartition(".")
                mod: torch.nn.Module = self.fts_handle.pl_module.get_submodule(module_path)
                if not hasattr(mod, param_name):
                    raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")
                ft_schedule_module_map[depth].add(module_path)
        return ft_schedule_module_map

    def _phase_constrained_auto_wrap(self) -> None:
        # link phase params to modules, until PT adds fix for param-level specificaiton
        module_map = self._gen_ft_sched_module_map()  # TODO: move to fts setup?
        for phase, mod_paths in module_map.items():
            should_wrap = self._inspect_policy_trace(mod_paths)
            if phase == 0:
                should_wrap = self._validate_min_wrap_condition(should_wrap, mod_paths)
            self._apply_phase_wrap(should_wrap, mod_paths)
        with self._enable_explicit_wrap():
            for n, m in self.fts_handle.pl_module.named_children():
                setattr(self.fts_handle.pl_module, n, wrap(m))

    @contextmanager
    def _enable_explicit_wrap(self) -> Generator:
        auto_wrap_policy_handle = _ConfigAutoWrap.kwargs.get("auto_wrap_policy", None)
        _ConfigAutoWrap.kwargs["auto_wrap_policy"] = None
        try:
            yield
        finally:
            _ConfigAutoWrap.kwargs["auto_wrap_policy"] = auto_wrap_policy_handle
