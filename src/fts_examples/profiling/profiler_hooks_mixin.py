import torch

from fts_examples.profiling import MemProfiler, MemProfilerCfg


class ProfilerHooksMixin:
    """Initially supporting only MemProfiler but will likely be extended to support more profiling in the
    future."""
    def __init__(self, memprofiler_cfg: MemProfilerCfg | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memprofiler = None
        self.memprofiler_cfg = memprofiler_cfg
        if self.memprofiler_cfg and self.memprofiler_cfg.enabled:
            self.memprofiler = MemProfiler()
            self.memprofiler.connect(self)
            if self.cuda_allocator_history:
                torch.cuda.memory._record_memory_history()

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
    def cuda_allocator_history(self) -> bool:
        return self.memprofiler and self.memprofiler_cfg.cuda_allocator_history

    @property
    def memprofiler(self) -> MemProfiler | None:
        """Returns the MemProfiler instance if it has been initialized, otherwise None."""
        return getattr(self, '_memprofiler', None)

    @memprofiler.setter
    def memprofiler(self, memprofiler: MemProfiler | None) -> None:
        """Sets the MemProfiler instance if it is not already initialized."""
        if not getattr(self, '_memprofiler', None):
            setattr(self, '_memprofiler', memprofiler)
