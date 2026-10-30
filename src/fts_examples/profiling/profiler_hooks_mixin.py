from lightning.fabric.utilities import rank_zero_warn

from finetuning_scheduler.profiling import MemProfiler, MemProfilerCfg
from finetuning_scheduler.profiling.accelerator import record_allocator_history


class ProfilerHooksMixin:
    """Initially supporting only MemProfiler but will likely be extended to support more profiling in the
    future."""
    def __init__(self, memprofiler_cfg: MemProfilerCfg | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memprofiler = None
        self.memprofiler_cfg = memprofiler_cfg
        if self.memprofiler_cfg and self.memprofiler_cfg.enabled:
            self.memprofiler = MemProfiler()
            # the trainer attaches after the module is built, so it is read when the profiler needs it
            self.memprofiler.connect(
                self,
                optimizer=lambda: self.trainer.optimizers[0],
                log_dir=lambda: self._trainer.log_dir or self._trainer.default_root_dir,
            )
            if self.allocator_history:
                record_allocator_history()

    def setup(self, stage):
        super().setup(stage)
        if self.memprofiler:
            self.memprofiler.init_memprof_log_dir()

    def on_train_start(self) -> None:
        if self.memprofiler:
            self.memprofiler.maybe_init_fsdp_mem_tracker()
        super().on_train_start()

    def on_train_end(self) -> None:
        if self.memprofiler:
            self.memprofiler.dump_memory_stats()
        super().on_train_end()

    @property
    def allocator_history(self) -> bool:
        return self.memprofiler and self.memprofiler_cfg.allocator_history

    @property
    def cuda_allocator_history(self) -> bool:
        """Deprecated alias of ``allocator_history``, unsupported no earlier than 2.17."""
        rank_zero_warn("`ProfilerHooksMixin.cuda_allocator_history` is deprecated in favor of"
                       " `ProfilerHooksMixin.allocator_history` and will be unsupported no earlier than 2.17.",
                       category=FutureWarning)
        return self.allocator_history

    @property
    def memprofiler(self) -> MemProfiler | None:
        """Returns the MemProfiler instance if it has been initialized, otherwise None."""
        return getattr(self, '_memprofiler', None)

    @memprofiler.setter
    def memprofiler(self, memprofiler: MemProfiler | None) -> None:
        """Sets the MemProfiler instance if it is not already initialized."""
        if not getattr(self, '_memprofiler', None):
            setattr(self, '_memprofiler', memprofiler)
