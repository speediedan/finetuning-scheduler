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
MemProfiler
^^^^^^^^^^^

A powerful memory profiling utility that synthesizes numerous complementary profiling methods.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, DefaultDict, Callable
from dataclasses import dataclass, field, fields, asdict
from contextlib import redirect_stdout, contextmanager, ExitStack
from collections import defaultdict
from itertools import chain
from functools import wraps
from psutil import Process
from copy import deepcopy
from pathlib import Path

import yaml
import torch
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from fts_examples.cfg_utils import resolve_funcs
from fts_examples.profiling.config import MemProfilerCfg


class MemProfiler:
    """MemProfiler is a powerful memory profiling utility that synthesizes numerous complementary profiling
    methods.

    The following profiling utilities are integrated and simultaneously configured:

    - ``FSDP2MemTracker``
    - `cuda memory snapshot and allocator history tracking <https://pytorch.org/docs/stable/torch_cuda_memory.html>`_
    - host-level memory tracking
    - custom memory hooks (e.g. for activation checkpoint memory tracking via ``saved_tensors_hooks`` etc.)
    """
    def __init__(self, *args, **kwargs) -> None:
        """The MemProfiler is a powerful memory profiling utility that synthesizes numerous complementary profiling
        methods.

        The following profiling utilities are integrated and simultaneously configured:

        - ``FSDP2MemTracker``
        - `cuda memory snapshot and allocator history
          tracking <https://pytorch.org/docs/stable/torch_cuda_memory.html>`_
        - host-level memory tracking
        - custom memory hooks (e.g. for activation checkpoint memory tracking via ``saved_tensors_hooks`` etc.)

        `See this example for usage guidance.
        <https://finetuning-scheduler.readthedocs.io/en/latest/profiling/memprofiler_profiling.html>`_

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Attributes:
            memory_stats (DefaultDict): A dictionary of dictionaries containing memory statistics.
            fsdp_memory_stats (Dict): A dictionary containing FSDP memory statistics.
            rank (int): The rank of the current process.
            module (Any): The module being profiled.
            memprof_log_dir (str): The directory where memory profiles are saved.
            fsdp_mem_tracker (FSDPMemTracker): The FSDP memory tracker.
            saved_tensors_funcs (list[Callable]): A list of functions used to track saved tensors.
            _state (MemProfInternalState): The internal state of the MemProfiler.
        """
        super().__init__()
        self.memory_stats = defaultdict(dict)
        self.fsdp_memory_stats = {}
        self.rank = _get_rank() or 0
        self.module = None
        self.memprof_log_dir = None
        self.fsdp_mem_tracker = None
        self.saved_tensors_funcs: list[Callable] = []
        self._state = _MemProfInternalState()

    def connect(self, obj_ref: Any) -> None:
        """Connects the MemProfiler to the given target module.

        Args:
            obj_ref: The module to be profiled.
        """
        self.module = obj_ref
        self._state.curr_pid = Process(os.getpid())
        if self.memprofiler_cfg.enable_saved_tensors_hooks:
            self.saved_tensors_funcs = resolve_funcs(cfg_obj=self.memprofiler_cfg, func_type='saved_tensors_funcs')
        self._state.base_collect_func_set = self._all_base_collect_funcs()
        self.memprofiler_cfg.retain_hooks_for_funcs = self._state.base_collect_func_set  # conservatively wait for all

    def _all_base_collect_funcs(self) -> set:
        funcs = self.memprofiler_cfg.collect_funcs
        return set(chain(*[funcs.cuda_allocator_history, funcs.cuda, funcs.cpu]))

    @property
    def memprofiler_cfg(self) -> MemProfilerCfg:
        return self.module.memprofiler_cfg

    @property
    def schedule(self) -> int:
        return self.module.memprofiler_cfg.schedule

    @property
    def fsdp_mem_tracker_root_module(self) -> FSDPModule | None:  # type: ignore
        """If ``track_fsdp_mem`` is enabled, this is the root FSDP module used for FSDP2 memory tracking.

        The root module must have ``fully_shard`` applied for FSDP2 memory tracking.
        If ``track_fsdp_mem`` is disabled, this is ``None``.
        """
        if not self.memprofiler_cfg.track_fsdp_mem:
            return None
        try:
            fsdp_tracker_root = self.module.get_submodule(self.memprofiler_cfg.fsdp_mem_tracker_root_module)
            assert issubclass(type(fsdp_tracker_root), FSDPModule), \
                "MemProfiler requires the root FSDP module to have ``fully_shard`` applied for FSDP2 memory tracking."
            return fsdp_tracker_root
        except AttributeError:
            raise MisconfigurationException(
                "Could not find the specified ``fsdp_mem_tracker_root_module``: "
                f"{self.memprofiler_cfg.fsdp_mem_tracker_root_module}. MemProfiler needs to attach to a module that has"
                " ``fully_shard`` applied for FSDP2 memory tracking."
            )

    def remove_memprofiler_hooks(self) -> None:
        """Removes all hooks added by MemProfiler."""
        for handle_list in self._state.hook_handles.values():
            for handle in handle_list:
                handle.remove()

    def exec_reset_state_hooks(self) -> None:
        """Executes all hooks registered for resetting memory tracking state.

        These hooks are responsible for resetting state (e.g. saved tensor sizes, RSS, etc.) tracked by the hooks
        registered in :meth:`add_memprofiler_hooks`.
        """
        for hook in self._state.configured_hooks["reset_state_hooks"]:
            hook(self.module, self.memprofiler_cfg.save_hook_attrs)

    def maybe_init_fsdp_mem_tracker(self) -> None:
        """Initializes FSDP2 memory tracker if ``track_fsdp_mem`` is ``True``."""
        if self.memprofiler_cfg.track_fsdp_mem:
            self.fsdp_mem_tracker = FSDPMemTracker(self.fsdp_mem_tracker_root_module, self.module.trainer.optimizers[0])

    def add_memprofiler_hooks(self) -> None:
        """Adds all hooks registered for memory tracking.

        Currently, hooks are registered for the `pre_forward` and `post_forward` points. Hooks are added to the
        `modules` of the provided `module` and are responsible for tracking memory usage. Supported hooks are
        registered based on the configuration of `memory_hooks` in `memprofiler_cfg`.

        After registering all hooks, this method calls :meth:`exec_reset_state_hooks` to reset the state of all hooks.
        """
        # TODO: extend supported hook points (e.g. backward, etc.) and if/once supporting additional hook points,
        # use a hook_type to registration function mapping
        memory_hooks_cfg = self.memprofiler_cfg.memory_hooks
        for supported_hooks in fields(memory_hooks_cfg):
            if getattr(memory_hooks_cfg, supported_hooks.name):
                self._state.configured_hooks[supported_hooks.name] = resolve_funcs(cfg_obj=memory_hooks_cfg,
                                                                             func_type=supported_hooks.name)
        for module in self.module.modules():
            module.mem_info_handle = self._state.curr_pid.memory_info
            for hook_func in self._state.configured_hooks["pre_forward_hooks"]:
                self._state.hook_handles[hook_func].append(module.register_forward_pre_hook(hook_func))
            for hook_func in self._state.configured_hooks["post_forward_hooks"]:
                self._state.hook_handles[hook_func].append(module.register_forward_hook(hook_func))
        self.exec_reset_state_hooks()

    def init_memprof_log_dir(self) -> None:
        """Initializes the directory where MemProfiler will save profiling artifacts.

        The directory is determined by the following logic:

        1. If ``memprofiler_cfg.save_dir`` is set, use that directory.
        2. Otherwise, use a directory named ``memprofiler`` in the current
           Lightning ``log_dir``.
        """
        base_log_dir = self.module._trainer.log_dir or self.module._trainer.default_root_dir
        self.memprof_log_dir = self.memprofiler_cfg.save_dir or Path(base_log_dir) / "memprofiler"
        self.memprof_log_dir = Path(self.memprof_log_dir)  # ensure the dir is a Path
        self.memprof_log_dir.mkdir(exist_ok=True, parents=True)

    def cuda_allocator_history_snap(self, snap_key: tuple) -> None:
        """Dumps a snapshot of the CUDA memory allocator history.

        The snapshot is saved to a file named ``cuda_alloc_rank_<snap_key>.pickle`` in the directory specified by
        ``memprof_log_dir``.

        Args:
            snap_key: A tuple used to identify the snapshot.
        """
        cuda_snapshot_file = (self.memprof_log_dir / f"cuda_alloc_rank_{snap_key}.pickle")
        torch.cuda.memory._dump_snapshot(cuda_snapshot_file)

    def done(self, iter_idx: int) -> bool:
        """Checks if the profiling is done.

        Args:
            iter_idx: The current iteration index.

        Returns:
            bool: Whether the profiling is done.
        """
        return self.schedule.max_iter and iter_idx > self.schedule.max_iter

    def _process_hooks(self, snap_key) -> None:
        if self.memprofiler_cfg.enable_memory_hooks:
            if len(self._state.hook_handles) == 0:
                self.add_memprofiler_hooks()
            collected = {attr: getattr(self.module, attr, None) for attr in self.memprofiler_cfg.save_hook_attrs}
            self.memory_stats[snap_key].update(collected)

    def _collect_snap(self, snap_key, reset_mem_hooks: bool = False) -> None:
        """Collects a snapshot of the current memory statistics.

        Args:
            snap_key: A tuple of strings identifying the snapshot.
            reset_mem_hooks: Whether to reset the memory hooks after collecting the snapshot.

        Notes:
            The snapshot is saved to `memory_stats` with the key `snap_key`.
        """
        _, fn_name, *_ = snap_key
        snap_key = ".".join(map(str, snap_key))
        mem_cfg = self.memprofiler_cfg
        self._process_hooks(snap_key)
        if fn_name in mem_cfg.collect_funcs.cpu:
            mem = self._state.curr_pid.memory_info()
            self.memory_stats[snap_key].update({"rss": mem.rss, "vms": mem.vms})
        if fn_name in mem_cfg.collect_funcs.cuda:
            self.memory_stats[snap_key].update(torch.cuda.memory_stats())
        if fn_name in mem_cfg.collect_funcs.cuda_allocator_history and mem_cfg.cuda_allocator_history:
            self.cuda_allocator_history_snap(snap_key)
        if mem_cfg.enable_memory_hooks and reset_mem_hooks:
            self.exec_reset_state_hooks()

    @property
    def _should_remove_hooks(self) -> bool:
        return all(func in self._state.done_prof_funcs for func in self.memprofiler_cfg.retain_hooks_for_funcs)

    def teardown_prof(self, fn_name: str, iter_ctx: str) -> None:
        """Tears down profiling state for a function.

        This method is responsible for:

        1. Disabling profiling for the given function and iteration context.
        2. Marking the function as done if it is not being profiled in either the "start" or "end" iteration contexts.
        3. Removing all hooks registered by MemProfiler if the function is marked as done and
           ``retain_hooks_for_funcs`` is ``True``.

        Args:
            fn_name (str): The name of the function to tear down profiling state for.
            iter_ctx (str): The iteration context of the function to tear down profiling state for.
        """
        self._state.can_collect[(fn_name, iter_ctx)] = False
        if not any(self._state.can_collect[(fn_name, iter_ctx)] for iter_ctx in ["start", "end"]):
            self._state.done_prof_funcs.append(fn_name)
        if self.memprofiler_cfg.retain_hooks_for_funcs and self._should_remove_hooks:
            self.remove_memprofiler_hooks()
            self.memprofiler_cfg.enable_memory_hooks = False

    def gen_snap_keys(self, fn_name: str, iter_ctx: str, iter_idx: int | None = None) -> tuple[int, int, tuple]:
        """Generates the MemProfiler snapshot key for a given function and iteration context.

        Args:
            fn_name (str): The name of the function to generate a snapshot key for.
            iter_ctx (str): The iteration context of the function to generate a snapshot key for.
            iter_idx (int | None): The iteration index to use in the snapshot key. If ``None``, the current iteration
                index will be used.

        Returns:
            A tuple of (iteration index, snapshot key). The snapshot key is a tuple of (rank, function name, iteration
            index, iteration context).

        Note:
            The snapshot key format is rank.fn_name.iter_idx.iter_ctx.
        """
        # NOTE [Memprofiler Key Format]
        # snap key format is rank.fn_name.iter_idx.iter_ctx
        # e.g. 0.training_step.0.end keys hook output for the end of `training_step` 0 (first iteration), for rank 0
        # 0.training_step.2.start keys mem stats for the start of `training_step` 2, for rank 0
        if iter_idx is None:
            iter_idx = self._state.snap_indices[(fn_name, iter_ctx)]
        return iter_idx, (self.rank, fn_name, iter_idx, iter_ctx)

    def update_collect_state(self, fn_name: str, iter_ctx: str, iter_idx: int | None = None) -> None:
        """Updates the MemProfiler state for a given function and iteration context.

        Args:
            fn_name (str): The name of the function to update the state for.
            iter_ctx (str): The iteration context of the function to update the state for.
            iter_idx (int, optional): The iteration index to use when generating the snapshot key. If ``None`` is
                provided, the current iteration index will be used.
        """
        self._state.maybe_init_iter_state(fn_name, iter_ctx)
        if self._state.can_collect[(fn_name, iter_ctx)] and not self._state._iter_incremented[(fn_name, iter_ctx)]:
            self._state.snap_indices[(fn_name, iter_ctx)] += 1
            self._state._iter_incremented[(fn_name, iter_ctx)] = True
            curr = self._state
            curr._iter_idx, curr._snap_key = self.gen_snap_keys(fn_name, iter_ctx, iter_idx)

    def snap(self, fn_name: str, iter_ctx: str, reset_mem_hooks: bool = False) -> None:
        """Collects a memory snapshot for the given function and iteration context.

        If the current iteration index is greater than the number of warmup iterations,
        then this function will check if the iteration index is within the profiling
        range. If it is, then a memory snapshot will be collected. If not, the memory
        profiler state for the given function and iteration context will be torn down.

        Args:
            fn_name (str): The name of the function to collect a memory snapshot for.
            iter_ctx (str): The iteration context of the function to collect a memory snapshot for.
            reset_mem_hooks (bool, optional): Whether to reset the memory hooks after collecting the snapshot.
                Defaults to False.
        """
        if self._state._iter_idx > self.schedule.warmup_iters:
            if not self.done(self._state._iter_idx):
                self._collect_snap(self._state._snap_key, reset_mem_hooks)
            else:
                self.teardown_prof(fn_name, iter_ctx)

    def dump_memory_stats(self) -> None:
        """Dumps the collected memory statistics to a pickle file and/or a yaml file.

        If `dump_memorystats_pickle` is True, the memory statistics will be saved to a pickle file
        at `memprof_log_dir / f"rank_{self.rank}_memory_stats.pickle"`.

        If `dump_memorystats_yaml` is True, the memory statistics will be saved to a yaml file
        at `memprof_log_dir / f"rank_{self.rank}_memory_stats.yaml"`.

        If `track_fsdp_mem` is True, the collected FSDP memory statistics will be saved to a yaml file
        at `memprof_log_dir / f"rank_{self.rank}_fsdp_memory_stats.yaml"`.
        """
        if self.memory_stats:
            if self.memprofiler_cfg.dump_memorystats_pickle:
                filename = self.memprof_log_dir / f"rank_{self.rank}_memory_stats.pickle"
                with open(filename, "wb") as f:
                    pickle.dump(self.memory_stats, f)
            if self.memprofiler_cfg.dump_memorystats_yaml:
                filename = self.memprof_log_dir / f"rank_{self.rank}_memory_stats.yaml"
                with open(filename, "w", newline="") as f:
                    yaml.dump(self.memory_stats, f)
        if self.fsdp_memory_stats:
            self.save_fsdp_mem_reports()

    def save_fsdp_mem_reports(self) -> None:
        """Saves the collected FSDP memory statistics to a yaml file.

        For each function name and iteration in the collected FSDP memory statistics, a log file is created
        at `memprof_log_dir / f"fsdp_mem_rank_{self.rank}_{fn_name}_{iter}.log"` containing the FSDP memory
        statistics in both tabular and modulewise formats.

        The modulewise format is displayed up to the specified `fsdp_mem_track_module_depth`, and the units
        for the memory statistics are specified by `fsdp_mem_tracker_units`.

        If `fsdp_mem_tracker_tabulate` is True, the FSDP memory statistics will be displayed in a tabular format.
        Otherwise, a plain text format will be used.

        :return: None
        """
        for (fn_name, iter) in self.fsdp_memory_stats.keys():
            self.fsdp_mem_tracker.memory_tracking = self.fsdp_memory_stats[(fn_name, iter)]
            target_fsdp_depth = self.memprofiler_cfg.fsdp_mem_track_module_depth
            tabulate_mode = self.memprofiler_cfg.fsdp_mem_tracker_tabulate
            report_units = self.memprofiler_cfg.fsdp_mem_tracker_units
            memory_report = self.memprof_log_dir / f"fsdp_mem_rank_{self.rank}_{fn_name}_{iter}.log"
            fs = get_filesystem(memory_report)
            with fs.open(memory_report, "w", newline="") as fp:
                with redirect_stdout(fp):
                    fp.write(f"Iteration {iter} {fn_name} Peak FSDP memory stats: {os.linesep}")
                    fp.write("-" * 80 + os.linesep)
                    self.fsdp_mem_tracker.display_snapshot("peak", tabulate=tabulate_mode)
                    fp.write("-" * 80 + os.linesep)
                    fp.write(f"Iteration {iter} {fn_name} Modulewise FSDP memory stats: {os.linesep}")
                    fp.write("-" * 80 + os.linesep)
                    self.fsdp_mem_tracker.display_modulewise_snapshots(depth=target_fsdp_depth, units=report_units,
                                                                       tabulate=tabulate_mode)
                    fp.write("-" * 80 + os.linesep)

    def _in_collect_range(self) -> bool:
        return self._state._iter_idx > self.schedule.warmup_iters and self._state._iter_idx <= self.schedule.max_iter

    def _collecting(self, fn_name: str, iter_ctx: str = 'start') -> bool:
        return self._state.can_collect[(fn_name, iter_ctx)] and self._in_collect_range()

    def _should_collect_fsdp(self, fn_name: str) -> bool:
        return all((self.fsdp_mem_tracker, fn_name in self.memprofiler_cfg.collect_funcs.fsdp,
                    self._collecting(fn_name)))

    def _should_collect_saved_tensors(self, fn_name: str) -> bool:
        return self.memprofiler_cfg.enable_saved_tensors_hooks and self._collecting(fn_name)

    @contextmanager
    @staticmethod
    def memprofile_snap_ctx(memprofiler, fn_name: str):
        """Context manager for taking memory snapshots at the start and end of a method.

        This context manager takes care of calling the `snap` method of the `MemProfiler` instance
        at the start and end of the context. It also handles the case where we are only collecting
        FSDP memory statistics.

        Args:
            memprofiler (MemProfiler): The MemProfiler instance.
            fn_name (str): The name of the function being profiled.

        Yields:
            None
        """
        try:
            # no need to execute snap logic if we are only collecting fsdp memory stats
            if fn_name not in memprofiler._state.base_collect_func_set:
                yield
            else:
                memprofiler.update_collect_state(fn_name=fn_name, iter_ctx="start")
                memprofiler.snap(iter_ctx="start", fn_name=fn_name)
                if memprofiler._should_collect_saved_tensors(fn_name):
                    with torch.autograd.graph.saved_tensors_hooks(*memprofiler.saved_tensors_funcs):
                        yield
                else:
                    yield
        finally:
            if fn_name in memprofiler._state.base_collect_func_set:
                memprofiler.update_collect_state(fn_name=fn_name, iter_ctx="end")
                memprofiler.snap(iter_ctx="end", fn_name=fn_name, reset_mem_hooks=True)

    @contextmanager
    @staticmethod
    def memprofile_fsdp_ctx(memprofiler, fn_name: str, track_inputs_target: tuple | None = None):
        """Sets the FSDP memory tracker context manager if ``fsdp_mem_tracker_enabled`` is ``True``.

        This context manager takes care of calling the `update_collect_state` and `reset_mod_stats`
        methods of the `MemProfiler` instance as well as the `track_inputs` and `reset` methods of
        the `FSDPMemTracker` instance.

        Args:
            memprofiler (MemProfiler): The MemProfiler instance.
            fn_name (str): The name of the function being profiled.
            track_inputs_target (Tuple | None): The FSDP inputs to track. If ``None``, no inputs will be tracked.

        Yields:
            None
        """
        fsdp_state_collected = False
        try:
            memprofiler.update_collect_state(fn_name=fn_name, iter_ctx="start")
            if track_inputs_target is not None:
                memprofiler.fsdp_mem_tracker.track_inputs(track_inputs_target)
            if not memprofiler._should_collect_fsdp(fn_name):
                yield
            else:
                fsdp_state_collected = True
                with memprofiler.fsdp_mem_tracker:
                    yield
        finally:
            memprofiler.update_collect_state(fn_name=fn_name, iter_ctx="end")
            if fsdp_state_collected:
                memprofiler.fsdp_memory_stats[(fn_name, memprofiler._state._iter_idx)] = \
                    deepcopy(memprofiler.fsdp_mem_tracker.memory_tracking)
                memprofiler.fsdp_mem_tracker.reset_mod_stats()

    @contextmanager
    @staticmethod
    def memprofile_meta_ctx(memprofiler, fn_name: str):
        """Context manager to set metadata for MemProfiler invocations.

        This context manager always sets the iteration state for the given function name
        to enable orchestration of relevant profiling invocations.

        Args:
            memprofiler (MemProfiler): The MemProfiler instance.
            fn_name (str): The name of the function being profiled.

        Yields:
            None
        """
        try:
            memprofiler._state.maybe_init_iter_state(fn_name, 'start')
            yield
        finally:
            memprofiler._state.reset_iter_state(fn_name, 'end')

    @staticmethod
    def memprofilable(func):
        """Decorate a function to enable use with MemProfiler.

        This decorator is used to enable profiling a method with MemProfiler.  If the
        method is decorated with this decorator and the object has an enabled MemProfiler
        instance, the method will be profiled using the MemProfiler instance.

        Args:
            func: The function to decorate.

        Returns:
            The decorated function.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.memprofiler:
                return func(self, *args, **kwargs)
            fn_name = func.__name__
            with MemProfiler.memprofile_ctx(self.memprofiler, fn_name=fn_name):
                return func(self, *args, **kwargs)
        return wrapper

    @contextmanager
    @staticmethod
    def memprofile_ctx(memprofiler, fn_name: str):
        """Context manager to orchestrate profiling for a given function.

        This context manager is responsible for calling the relevant subordinate context managers
        that manage profiling for a given function. The context managers invoked are:

        1. `memprofile_meta_ctx`: Sets metadata for the profiling invocation.
        2. `memprofile_fsdp_ctx`: Collects FSDP memory statistics.
        3. `memprofile_snap_ctx`: Collects memory snapshots at the start and end of a method.

        Args:
            memprofiler (MemProfiler): The MemProfiler instance.
            fn_name (str): The name of the function being profiled.

        Yields:
            None
        """
        ctx_managers = []
        ctx_managers.append(MemProfiler.memprofile_meta_ctx(memprofiler, fn_name=fn_name))
        ctx_managers.append(MemProfiler.memprofile_fsdp_ctx(memprofiler, fn_name=fn_name))
        ctx_managers.append(MemProfiler.memprofile_snap_ctx(memprofiler, fn_name=fn_name))
        with ExitStack() as stack:
            for ctx_manager in ctx_managers:
                stack.enter_context(ctx_manager)
            try:
                yield
            finally:
                stack.close()

def _memprofiler_cfg_mapping_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', asdict(data))

yaml.SafeDumper.add_representer(MemProfilerCfg, _memprofiler_cfg_mapping_representer)


@dataclass
class _MemProfInternalState:
    can_collect: dict[str, bool] = field(default_factory=dict)
    curr_pid: Process | None = None
    snap_indices: dict[str, int] = field(default_factory=dict)
    configured_hooks: dict[str, Any] = field(default_factory=dict)
    hook_handles: DefaultDict[str, list[Any]] = field(default_factory=lambda: defaultdict(list))
    done_prof_funcs: list[str] = field(default_factory=list)
    base_collect_func_set: set | None = None
    _iter_idx: int | None = None
    _snap_key: tuple | None = None
    _iter_incremented: dict[str, int] = field(default_factory=dict)

    def maybe_init_iter_state(self, fn_name: str, iter_ctx: str) -> None:
        if not self.snap_indices.get((fn_name, iter_ctx), None):
            self._iter_incremented[(fn_name, iter_ctx)] = False
            self.snap_indices[(fn_name, iter_ctx)] = 0
            self.can_collect[(fn_name, iter_ctx)] = True

    def reset_iter_state(self, fn_name: str, iter_ctx: str) -> None:
        self._snap_key = None
        for (fn_name, iter_ctx) in self._iter_incremented.keys():
            self._iter_incremented[(fn_name, iter_ctx)] = False
