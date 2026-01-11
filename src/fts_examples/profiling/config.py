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
from typing import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch
from lightning.fabric.utilities import rank_zero_warn


@dataclass
class MemProfilerHooks:
    pre_forward_hooks: list[str | Callable] = \
        field(default_factory=lambda: ['fts_examples.profiling.npp_hooks._hook_npp_pre_forward'])
    post_forward_hooks: list[str | Callable] = \
        field(default_factory=lambda: ['fts_examples.profiling.npp_hooks._hook_npp_post_forward'])
    # the provided reset_state_hooks will be called with the model and the `save_hook_attrs` list
    reset_state_hooks: list[str | Callable] = \
        field(default_factory=lambda: ['fts_examples.profiling.npp_hooks._reset_memory_hooks_state'])

@dataclass
class MemProfilerFuncs: # can specify arbitrary list of `memprofilable` decorated function names
    # funcs that will be added to all memory collection types
    default: set[str] = field(default_factory=lambda: {'training_step'})
    cpu: set[str] = field(default_factory=set)
    cuda: set[str] = field(default_factory=set)
    cuda_allocator_history: set[str] = field(default_factory=set)
    fsdp: set[str] = field(default_factory=set)

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
    :param cuda_allocator_history: Whether to collect CUDA memory allocator history.
    :param track_fsdp_mem: Whether to collect FSDP memory statistics.
    :param fsdp_mem_track_module_depth: The depth of FSDP modules to track.
    :param fsdp_mem_tracker_tabulate: Whether to print FSDP memory statistics in a tabular format.
    :param fsdp_mem_tracker_units: The units to use for FSDP memory statistics.
    :param fsdp_mem_tracker_root_module: The root module to use for FSDP memory statistics.
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
    """
    enabled: bool = False
    # specify funcs to collect per memory collection type, a default list to apply to all types or both composed
    collect_funcs: MemProfilerFuncs = field(default_factory=MemProfilerFuncs)
    cuda_allocator_history: bool = False
    track_fsdp_mem: bool = False
    fsdp_mem_track_module_depth: int = 2
    fsdp_mem_tracker_tabulate: bool = False
    fsdp_mem_tracker_units: str = "MiB"
    fsdp_mem_tracker_root_module: str = ""
    dump_memorystats_pickle: bool = False
    dump_memorystats_yaml: bool = True
    schedule: MemProfilerSchedule = field(default_factory=MemProfilerSchedule)
    save_dir: str | Path | None = None
    enable_memory_hooks: bool = True
    enable_saved_tensors_hooks: bool = True
    memory_hooks: MemProfilerHooks = field(default_factory=MemProfilerHooks)
    # because it's frequently used for unpacking and to ensure this dataclass remains serializable, we allow
    # specification of 'identity_lambda' which will resolve to `lambda x: x`
    saved_tensors_funcs: list = field(default_factory=lambda: list(('fts_examples.profiling.npp_hooks._npp_hook',
                                                                    'identity_lambda')))
    # if you add custom hooks, make sure to add the desired module state attributes to save to `save_hook_attrs`
    save_hook_attrs: list = field(default_factory=lambda: ["rss_pre_forward", "rss_post_forward", "rss_diff",
                                                           "npp_pre_forward", "npp_post_forward", "npp_diff"])
    # since we cannot reliably ascertain when all MemProfilerFuncs will be executed, memory hooks will
    # only be removed once the funcs in this set have reached `max_iter`
    retain_hooks_for_funcs: set[str] = field(default_factory=lambda: {'training_step'})

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not torch.cuda.is_available() and any((self.collect_funcs.cuda_allocator_history, self.collect_funcs.cuda,
                                                  self.cuda_allocator_history)):
            rank_zero_warn("Disabling CUDA memory profiling functionality since no CUDA device detected.")
            self.collect_funcs.cuda, self.collect_funcs.cuda_allocator_history = set(), set()
            self.cuda_allocator_history = False
        has_hooks = any(getattr(self.memory_hooks, ht.name) for ht in fields(self.memory_hooks))
        if not has_hooks:
            rank_zero_warn("MemProfilerCfg is configured to enable memory hooks but MemProfilerHooks does not have"
                           " any specified.")
        if self.schedule.max_iter is None:
            self.schedule.max_iter = self.schedule.warmup_iters + 1
        # compose all non-default func sets with the default set
        default_funcs = self.collect_funcs.default
        for k in self.collect_funcs.__dataclass_fields__.keys():
            if k != 'default':
                getattr(self.collect_funcs, k).update(default_funcs)
