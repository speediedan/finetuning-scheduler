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
MemProfiler Configuration Dataclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This module defines the configuration dataclasses for the MemProfiler.
"""
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from finetuning_scheduler.core.dist import rank_zero_warn
from finetuning_scheduler.core.exceptions import configuration_error_cls
from finetuning_scheduler.profiling.accelerator import accelerator_backend, allocator_history_module

# deprecated configuration names and their replacements, accepted until no earlier than 2.17
_DEPRECATED_FUNC_KEYS = {"cuda": "accelerator", "cuda_allocator_history": "allocator_history"}


def _deprecated_key_warning(old: str, new: str) -> None:
    rank_zero_warn(f"The MemProfiler configuration key `{old}` is deprecated in favor of `{new}` and will be"
                   " unsupported no earlier than 2.17.", category=FutureWarning)


@dataclass
class MemProfilerHooks:
    pre_forward_hooks: list[str | Callable] = \
        field(default_factory=lambda: ['finetuning_scheduler.profiling.npp_hooks._hook_npp_pre_forward'])
    post_forward_hooks: list[str | Callable] = \
        field(default_factory=lambda: ['finetuning_scheduler.profiling.npp_hooks._hook_npp_post_forward'])
    # the provided reset_state_hooks will be called with the model and the `save_hook_attrs` list
    reset_state_hooks: list[str | Callable] = \
        field(default_factory=lambda: ['finetuning_scheduler.profiling.npp_hooks._reset_memory_hooks_state'])

@dataclass
class MemProfilerFuncs: # can specify arbitrary list of `memprofilable` decorated function names
    # funcs that will be added to all memory collection types
    default: set[str] = field(default_factory=lambda: {'training_step'})
    cpu: set[str] = field(default_factory=set)
    accelerator: set[str] = field(default_factory=set)
    allocator_history: set[str] = field(default_factory=set)
    fsdp: set[str] = field(default_factory=set)
    # deprecated aliases of `accelerator` and `allocator_history`, merged into them
    cuda: set[str] | None = None
    cuda_allocator_history: set[str] | None = None

    def __post_init__(self) -> None:
        for old, new in _DEPRECATED_FUNC_KEYS.items():
            if (deprecated_funcs := getattr(self, old)) is not None:
                _deprecated_key_warning(f"collect_funcs.{old}", f"collect_funcs.{new}")
                getattr(self, new).update(deprecated_funcs)
                setattr(self, old, None)

    def collect_types(self) -> list[str]:
        """The memory collection types, excluding ``default`` and the deprecated aliases."""
        return [f.name for f in fields(self) if f.name != "default" and f.name not in _DEPRECATED_FUNC_KEYS]

@dataclass
class MemProfilerSchedule:
    # keeping schedule simple as possibile for now, may expand to accommodate more flexible schedules in the future
    warmup_iters: int = 1
    max_iter: int | None = None

@dataclass
class MemProfilerCfg:
    """Configuration dataclass for the MemProfiler.

    :param enabled: Whether to enable memory profiling.
    :param collect_funcs: A MemProfilerFuncs instance specifying the functions to collect per memory collection type.
    :param allocator_history: Whether to collect accelerator memory allocator history (CUDA, ROCm and XPU). Defaults
        to ``False``.
    :param track_fsdp_mem: Whether to collect FSDP memory statistics.
    :param fsdp_mem_track_module_depth: The depth of FSDP modules to track.
    :param fsdp_mem_tracker_tabulate: Whether to print FSDP memory statistics in a tabular format.
    :param fsdp_mem_tracker_units: The units to use for FSDP memory statistics.
    :param fsdp_mem_tracker_root_module: The root module to use for FSDP memory statistics.
    :param dump_fsdp_mem_json: Whether to also save the FSDP memory statistics as one JSON report per rank.
    :param aggregate_fsdp_mem_reports: Whether rank 0 aggregates the ranks' JSON FSDP memory reports across ranks once
        every rank has written its own (requires ``dump_fsdp_mem_json``).
    :param dump_memorystats_pickle: Whether to dump memory statistics to a pickle file.
    :param dump_memorystats_yaml: Whether to dump memory statistics to a yaml file.
    :param schedule: A MemProfilerSchedule instance specifying the schedule for memory collection.
    :param save_dir: The directory to save the memory statistics.
    :param enable_memory_hooks: Whether to enable memory hooks.
    :param enable_saved_tensors_hooks: Whether to enable saved tensors hooks.
    :param memory_hooks: A MemProfilerHooks instance specifying the memory hooks.
    :param saved_tensors_funcs: A list of saved tensors functions.
    :param save_hook_attrs: A list of module state attributes to save.
    :param retain_hooks_for_funcs: A set of functions to retain memory hooks for.
    :param cuda_allocator_history: Deprecated alias of ``allocator_history``, which takes precedence when both are set.
    """
    enabled: bool = False
    # specify funcs to collect per memory collection type, a default list to apply to all types or both composed
    collect_funcs: MemProfilerFuncs = field(default_factory=MemProfilerFuncs)
    allocator_history: bool | None = None  # `False` unless set (`None` tells an unset key from an explicit one)
    track_fsdp_mem: bool = False
    fsdp_mem_track_module_depth: int = 2
    fsdp_mem_tracker_tabulate: bool = False
    fsdp_mem_tracker_units: str = "MiB"
    fsdp_mem_tracker_root_module: str = ""
    dump_fsdp_mem_json: bool = True
    aggregate_fsdp_mem_reports: bool = False
    dump_memorystats_pickle: bool = False
    dump_memorystats_yaml: bool = True
    schedule: MemProfilerSchedule = field(default_factory=MemProfilerSchedule)
    save_dir: str | Path | None = None
    enable_memory_hooks: bool = True
    enable_saved_tensors_hooks: bool = True
    memory_hooks: MemProfilerHooks = field(default_factory=MemProfilerHooks)
    # because it's frequently used for unpacking and to ensure this dataclass remains serializable, we allow
    # specification of 'identity_lambda' which will resolve to `lambda x: x`
    saved_tensors_funcs: list = field(
        default_factory=lambda: list(('finetuning_scheduler.profiling.npp_hooks._npp_hook', 'identity_lambda')))
    # if you add custom hooks, make sure to add the desired module state attributes to save to `save_hook_attrs`
    save_hook_attrs: list = field(default_factory=lambda: ["rss_pre_forward", "rss_post_forward", "rss_diff",
                                                           "npp_pre_forward", "npp_post_forward", "npp_diff"])
    # since we cannot reliably ascertain when all MemProfilerFuncs will be executed, memory hooks will
    # only be removed once the funcs in this set have reached `max_iter`
    retain_hooks_for_funcs: set[str] = field(default_factory=lambda: {'training_step'})
    cuda_allocator_history: bool | None = None  # deprecated alias of `allocator_history`

    def __post_init__(self) -> None:
        if self.cuda_allocator_history is not None:
            _deprecated_key_warning("cuda_allocator_history", "allocator_history")
            if self.allocator_history is None:  # an explicitly set `allocator_history` takes precedence
                self.allocator_history = self.cuda_allocator_history
            self.cuda_allocator_history = None
        self.allocator_history = bool(self.allocator_history)
        if self.aggregate_fsdp_mem_reports and not self.dump_fsdp_mem_json:
            raise configuration_error_cls()("`aggregate_fsdp_mem_reports` aggregates the per-rank JSON FSDP memory"
                                            " reports, so it requires `dump_fsdp_mem_json`.")
        if not self.enabled:
            return
        has_hooks = any(getattr(self.memory_hooks, ht.name) for ht in fields(self.memory_hooks))
        if not has_hooks:
            rank_zero_warn("MemProfilerCfg is configured to enable memory hooks but MemProfilerHooks does not have"
                           " any specified.")
        if self.schedule.max_iter is None:
            self.schedule.max_iter = self.schedule.warmup_iters + 1
        # only explicitly requested accelerator collection warrants a warning when it is unsupported
        funcs = self.collect_funcs
        accel_requested = bool(funcs.accelerator or funcs.allocator_history or self.allocator_history)
        history_requested = bool(funcs.allocator_history or self.allocator_history)
        # compose all non-default func sets with the default set
        for k in funcs.collect_types():
            getattr(funcs, k).update(funcs.default)
        self._disable_unsupported_collection(accel_requested, history_requested)

    def _disable_unsupported_collection(self, accel_requested: bool, history_requested: bool) -> None:
        funcs = self.collect_funcs
        if (backend := accelerator_backend()) is None:
            if accel_requested:
                rank_zero_warn("Disabling accelerator memory profiling functionality since no accelerator detected.")
            funcs.accelerator, funcs.allocator_history = set(), set()
            self.allocator_history = False
        elif allocator_history_module() is None:
            if history_requested:
                rank_zero_warn(f"Disabling allocator history collection since the `{backend}` backend does not"
                               " support allocator history recording.")
            funcs.allocator_history = set()
            self.allocator_history = False


def resolve_funcs(cfg_obj: Any, func_type: str) -> list:
    """The callables ``cfg_obj.<func_type>`` names, each given as a callable, a dotted qualified name, or
    ``'identity_lambda'``.

    Raises:
        FTSConfigurationError: If a name does not import or does not name a callable.
    """
    resolved_funcs = []
    funcs_to_resolve = getattr(cfg_obj, func_type)
    if not isinstance(funcs_to_resolve, list):
        funcs_to_resolve = [funcs_to_resolve]
    for func_or_qualname in funcs_to_resolve:
        if callable(func_or_qualname):
            resolved_funcs.append(func_or_qualname)  # TODO: inspect if signature is appropriate for custom hooks
        elif func_or_qualname == 'identity_lambda':  # special case often used e.g. w/ saved_tensors_hooks
            resolved_funcs.append(lambda x: x)
        else:
            module, _, func = str(func_or_qualname).rpartition(".")
            if not module:
                raise configuration_error_cls()(f"`{func_or_qualname}` is not a qualified function name (module.func).")
            try:
                mod = importlib.import_module(module)
                resolved_func = getattr(mod, func, None)
                if callable(resolved_func):
                    resolved_funcs.append(resolved_func)
                else:
                    raise configuration_error_cls()(f"Custom function {func} from module {module} is not callable!")
            except (AttributeError, ImportError) as e:
                err_msg = f"Unable to import and resolve specified function {func} from module {module}: {e}"
                raise configuration_error_cls()(err_msg)
    return resolved_funcs
