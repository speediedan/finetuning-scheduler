import os
import pickle
from typing import Any, Dict, Optional, Tuple,  DefaultDict, Callable, List, Set, Union
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
from lightning.fabric.utilities import rank_zero_warn
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.utilities.exceptions import MisconfigurationException

# conditionally import indirectly to avoid duplicating import logic in several different modules
from finetuning_scheduler.strategy_adapters._mp_imports import _TORCH_GREATER_EQUAL_2_5, FSDPModule, FSDPMemTracker
from fts_examples.cfg_utils import resolve_funcs
from finetuning_scheduler.types import AutoStrEnum


class DefaultMemHooks(AutoStrEnum):
    pre_forward = 'fts_examples.profiling.npp_hooks._hook_npp_pre_forward'
    post_forward = 'fts_examples.profiling.npp_hooks._hook_npp_post_forward'
    reset_state = 'fts_examples.profiling.npp_hooks._reset_memory_hooks_state'

@dataclass
class MemProfilerHooks:
    pre_forward_hooks: List[Union[str, Callable]] = field(default_factory=lambda: [DefaultMemHooks.pre_forward.value])
    post_forward_hooks: List[Union[str, Callable]] = field(default_factory=lambda: [DefaultMemHooks.post_forward.value])
    # the provided reset_state_hooks will be called with the model and the `save_hook_attrs` list
    reset_state_hooks: List[Union[str, Callable]] = field(default_factory=lambda: [DefaultMemHooks.reset_state.value])

@dataclass
class MemProfilerFuncs: # can specify arbitrary list of `memprofilable` decorated function names
    # funcs that will be added to all memory collection types
    default: Set[str] = field(default_factory=lambda: {'training_step'})
    cpu: Set[str] = field(default_factory=set)
    cuda: Set[str] = field(default_factory=set)
    cuda_allocator_history: Set[str] = field(default_factory=set)
    fsdp: Set[str] = field(default_factory=set)

@dataclass
class MemProfilerSchedule:
    # keeping schedule simple as possibile for now, may expand to accommodate more flexible schedules in the future
    warmup_iters: int = 1
    max_iter: Optional[int] = None

@dataclass
class MemProfilerCfg:
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
    save_dir: Optional[Union[str, Path]] = None
    enable_memory_hooks: bool = True
    enable_saved_tensors_hooks: bool = True
    memory_hooks: MemProfilerHooks = field(default_factory=MemProfilerHooks)
    # because it's frequently used for unpacking and to ensure this dataclass remains serializable, we allow
    # specification of 'identity_lambda' which will resolve to `lambda x: x`
    saved_tensors_funcs: List = field(default_factory=lambda: list(('fts_examples.profiling.npp_hooks._npp_hook',
                                                                    'identity_lambda')))
    # if you add custom hooks, make sure to add the desired module state attributes to save to `save_hook_attrs`
    save_hook_attrs: List = field(default_factory=lambda: ["rss_pre_forward", "rss_post_forward", "rss_diff",
                                                           "npp_pre_forward", "npp_post_forward", "npp_diff"])
    # since we cannot reliably ascertain when all MemProfilerFuncs will be executed, memory hooks will
    # only be removed once the funcs in this set have reached `max_iter`
    retain_hooks_for_funcs: Set[str] = field(default_factory=lambda: {'training_step'})

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not torch.cuda.is_available() and any((self.collect_funcs.cuda_allocator_history, self.collect_funcs.cuda,
                                                  self.cuda_allocator_history)):
            rank_zero_warn("Disabling CUDA memory profiling functionality since no CUDA device detected.")
            self.collect_funcs.cuda, self.collect_funcs.cuda_allocator_history = set(), set()
            self.cuda_allocator_history = False
        if self.track_fsdp_mem and not _TORCH_GREATER_EQUAL_2_5:
            rank_zero_warn("Disabling FSDP memory profiling functionality since PyTorch version < 2.5.")
            self.track_fsdp_mem = False
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

def memprofiler_cfg_mapping_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', asdict(data))

yaml.SafeDumper.add_representer(MemProfilerCfg, memprofiler_cfg_mapping_representer)


@dataclass
class MemProfInternalState:
    can_collect: Dict[str, bool] = field(default_factory=dict)
    curr_pid: Optional[Process] = None
    snap_indices: Dict[str, int] = field(default_factory=dict)
    configured_hooks: Dict[str, Any] = field(default_factory=dict)
    hook_handles: DefaultDict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    done_prof_funcs: List[str] = field(default_factory=list)
    base_collect_func_set: Optional[Set] = None
    _iter_idx: Optional[int] = None
    _snap_key: Optional[Tuple] = None
    _iter_incremented: Dict[str, int] = field(default_factory=dict)

    def maybe_init_iter_state(self, fn_name: str, iter_ctx: str) -> None:
        if not self.snap_indices.get((fn_name, iter_ctx), None):
            self._iter_incremented[(fn_name, iter_ctx)] = False
            self.snap_indices[(fn_name, iter_ctx)] = 0
            self.can_collect[(fn_name, iter_ctx)] = True

    def reset_iter_state(self, fn_name: str, iter_ctx: str) -> None:
        self._snap_key = None
        for (fn_name, iter_ctx) in self._iter_incremented.keys():
            self._iter_incremented[(fn_name, iter_ctx)] = False

class MemProfiler:
    """MemProfiler is a utility that expedites simultaneous configuration and orchestration of numerous
    complementary profiling methods.

    The following profiling utilities are integrated and simultaneously configured:

    - ``FSDP2MemTracker``
    - `cuda memory snapshot and allocator history tracking <https://pytorch.org/docs/stable/torch_cuda_memory.html>`_
    - host-level memory tracking
    - custom memory hooks (e.g. for activation checkpoint memory tracking via ``saved_tensors_hooks`` etc.)
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.memory_stats = defaultdict(dict)
        self.fsdp_memory_stats = {}
        self.rank = _get_rank() or 0
        self.module = None
        self.memprof_log_dir = None
        self.fsdp_mem_tracker = None
        self.saved_tensors_funcs: List[Callable] = []
        self._state = MemProfInternalState()

    def connect(self, obj_ref: Any) -> None:
        self.module = obj_ref
        self._state.curr_pid = Process(os.getpid())
        if self.memprofiler_cfg.enable_saved_tensors_hooks:
            self.saved_tensors_funcs = resolve_funcs(cfg_obj=self.memprofiler_cfg, func_type='saved_tensors_funcs')
        self._state.base_collect_func_set = self._all_base_collect_funcs()
        self.memprofiler_cfg.retain_hooks_for_funcs = self._state.base_collect_func_set  # conservatively wait for all

    def _all_base_collect_funcs(self) -> Set:
        funcs = self.memprofiler_cfg.collect_funcs
        return set(chain(*[funcs.cuda_allocator_history, funcs.cuda, funcs.cpu]))

    @property
    def memprofiler_cfg(self) -> MemProfilerCfg:
        return self.module.memprofiler_cfg

    @property
    def schedule(self) -> int:
        return self.module.memprofiler_cfg.schedule

    @property
    def fsdp_mem_tracker_root_module(self) -> Optional[FSDPModule]:  # type: ignore
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
        for handle_list in self._state.hook_handles.values():
            for handle in handle_list:
                handle.remove()

    def exec_reset_state_hooks(self) -> None:
        for hook in self._state.configured_hooks["reset_state_hooks"]:
            #hook(self.module.model, self.memprofiler_cfg.save_hook_attrs)
            hook(self.module, self.memprofiler_cfg.save_hook_attrs)

    def maybe_init_fsdp_mem_tracker(self) -> None:
        if self.memprofiler_cfg.track_fsdp_mem:

            self.fsdp_mem_tracker = FSDPMemTracker(self.fsdp_mem_tracker_root_module, self.module.trainer.optimizers[0])

    def add_memprofiler_hooks(self) -> None:
        # TODO: extend supported hook points (e.g. backward, etc.) and if/once supporting additional hook points,
        # use a hook_type to registration function mapping
        memory_hooks_cfg = self.memprofiler_cfg.memory_hooks
        for supported_hooks in fields(memory_hooks_cfg):
            if getattr(memory_hooks_cfg, supported_hooks.name):
                self._state.configured_hooks[supported_hooks.name] = resolve_funcs(cfg_obj=memory_hooks_cfg,
                                                                             func_type=supported_hooks.name)
        #for module in self.module.model.modules():
        for module in self.module.modules():
            module.mem_info_handle = self._state.curr_pid.memory_info
            for hook_func in self._state.configured_hooks["pre_forward_hooks"]:
                self._state.hook_handles[hook_func].append(module.register_forward_pre_hook(hook_func))
            for hook_func in self._state.configured_hooks["post_forward_hooks"]:
                self._state.hook_handles[hook_func].append(module.register_forward_hook(hook_func))
        self.exec_reset_state_hooks()

    def init_memprof_log_dir(self) -> None:
        self.memprof_log_dir = self.memprofiler_cfg.save_dir or Path(self.module._trainer.log_dir) / "memprofiler"
        self.memprof_log_dir = Path(self.memprof_log_dir)  # ensure the dir is a Path
        self.memprof_log_dir.mkdir(exist_ok=True, parents=True)

    def cuda_allocator_history_snap(self, snap_key: Tuple) -> Dict:
        cuda_snapshot_file = (self.memprof_log_dir / f"cuda_alloc_rank_{snap_key}.pickle")
        torch.cuda.memory._dump_snapshot(cuda_snapshot_file)

    def done(self, iter_idx: int) -> bool:
        return self.schedule.max_iter and iter_idx > self.schedule.max_iter

    def _process_hooks(self, snap_key) -> None:
        if self.memprofiler_cfg.enable_memory_hooks:
            if len(self._state.hook_handles) == 0:
                self.add_memprofiler_hooks()
            #collected = {attr: getattr(self.module.model, attr, None) for attr in self.memprofiler_cfg.save_hook_attrs}
            collected = {attr: getattr(self.module, attr, None) for attr in self.memprofiler_cfg.save_hook_attrs}
            self.memory_stats[snap_key].update(collected)

    def _collect_snap(self, snap_key, reset_mem_hooks: bool = False) -> None:
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
        self._state.can_collect[(fn_name, iter_ctx)] = False
        if not any(self._state.can_collect[(fn_name, iter_ctx)] for iter_ctx in ["start", "end"]):
            self._state.done_prof_funcs.append(fn_name)
        if self.memprofiler_cfg.retain_hooks_for_funcs and self._should_remove_hooks:
            self.remove_memprofiler_hooks()
            self.memprofiler_cfg.enable_memory_hooks = False

    def gen_snap_keys(self, fn_name: str, iter_ctx: str, iter_idx: Optional[int] = None) -> Tuple[int, int, Tuple]:
        # NOTE [Memprofiler Key Format]
        # snap key format is rank.fn_name.iter_idx.iter_ctx
        # e.g. 0.training_step.0.end keys hook output for the end of `training_step` 0 (first iteration), for rank 0
        # 0.training_step.2.start keys mem stats for the start of `training_step` 2, for rank 0
        if iter_idx is None:
            iter_idx = self._state.snap_indices[(fn_name, iter_ctx)]
        return iter_idx, (self.rank, fn_name, iter_idx, iter_ctx)

    def update_collect_state(self, fn_name: str, iter_ctx: str, iter_idx: Optional[int] = None) -> None:
        self._state.maybe_init_iter_state(fn_name, iter_ctx)
        if self._state.can_collect[(fn_name, iter_ctx)] and not self._state._iter_incremented[(fn_name, iter_ctx)]:
            self._state.snap_indices[(fn_name, iter_ctx)] += 1
            self._state._iter_incremented[(fn_name, iter_ctx)] = True
            curr = self._state
            curr._iter_idx, curr._snap_key = self.gen_snap_keys(fn_name, iter_ctx, iter_idx)

    def snap(self, fn_name: str, iter_ctx: str, reset_mem_hooks: bool = False) -> None:
        if self._state._iter_idx > self.schedule.warmup_iters:
            if not self.done(self._state._iter_idx):
                self._collect_snap(self._state._snap_key, reset_mem_hooks)
            else:
                self.teardown_prof(fn_name, iter_ctx)

    def dump_memory_stats(self) -> None:
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

    def save_fsdp_mem_reports(self):
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
    def memprofile_fsdp_ctx(memprofiler, fn_name: str, track_inputs_target: Optional[Tuple] = None,):
        """Returns the FSDP memory tracker context manager if self.fsdp_mem_tracker_enabled is true, otherwise
        yields a no-op generator."""
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
        # we always set iterwise metadata to enable orchestration of relevant profiling invocations
        try:
            memprofiler._state.maybe_init_iter_state(fn_name, 'start')
            yield
        finally:
            memprofiler._state.reset_iter_state(fn_name, 'end')

    @staticmethod
    def memprofilable(func):
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
