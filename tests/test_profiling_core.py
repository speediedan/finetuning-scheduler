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
"""The framework-neutral MemProfiler core (``finetuning_scheduler.profiling``), driven by a plain PyTorch loop."""
import subprocess
import sys

import pytest
import torch

pytest.importorskip("psutil")

from finetuning_scheduler.core.exceptions import FTSConfigurationError  # noqa: E402
from finetuning_scheduler.profiling import (MemProfiler, MemProfilerCfg, MemProfilerFuncs,  # noqa: E402
                                            MemProfilerSchedule)

_LIGHTNING_LOADED = (
    "import sys, {module}; "
    "print(sorted(m for m in sys.modules if m.split('.')[0] in ('lightning', 'pytorch_lightning'))[:3])"
)


def _lightning_modules_after_importing(module: str) -> str:
    cmd = [sys.executable, "-c", _LIGHTNING_LOADED.format(module=module)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def test_the_core_imports_without_lightning():
    assert _lightning_modules_after_importing("finetuning_scheduler.profiling") == "[]"
    # the same check sees Lightning through the examples package's Lightning integration
    assert _lightning_modules_after_importing("fts_examples.profiling") != "[]"


class _PlainModule(torch.nn.Module):
    """What the profiler reads off a module: its ``memprofiler_cfg``, and the ``memprofiler`` its profiled methods
    use."""

    def __init__(self, cfg: MemProfilerCfg) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        self.memprofiler_cfg = cfg
        self.memprofiler: MemProfiler | None = None

    @MemProfiler.memprofilable
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        return self.layer(batch).sum()


@pytest.fixture
def no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


def _cpu_cfg(**kwargs) -> MemProfilerCfg:
    return MemProfilerCfg(enabled=True, enable_memory_hooks=False, enable_saved_tensors_hooks=False,
                          collect_funcs=MemProfilerFuncs(cpu={"training_step"}, accelerator=set()),
                          schedule=MemProfilerSchedule(warmup_iters=1, max_iter=3), **kwargs)


def test_a_plain_loop_profiles_with_an_injected_optimizer_and_log_dir(no_accelerator, tmp_path):
    module = _PlainModule(_cpu_cfg())
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module.memprofiler = MemProfiler()
    module.memprofiler.connect(module, optimizer=optimizer, log_dir=tmp_path)
    module.memprofiler.init_memprof_log_dir()
    for _ in range(4):
        optimizer.zero_grad()
        module.training_step(torch.ones(2, 4)).backward()
        optimizer.step()
    module.memprofiler.dump_memory_stats()
    assert module.memprofiler.memprof_log_dir == tmp_path / "memprofiler"
    collected = module.memprofiler.memory_stats
    assert collected and all("training_step" in key for key in collected)
    assert all({"rss", "vms"} <= set(stats) for stats in collected.values())
    assert any((tmp_path / "memprofiler").iterdir())


def test_the_log_dir_is_read_when_needed_and_refused_when_unset(tmp_path):
    seen = []
    memprofiler = MemProfiler()
    memprofiler.connect(_PlainModule(_cpu_cfg()), log_dir=lambda: seen.append(1) or tmp_path)
    assert not seen  # a provider is not called until the directory is needed
    memprofiler.init_memprof_log_dir()
    assert seen and memprofiler.memprof_log_dir == tmp_path / "memprofiler"

    memprofiler = MemProfiler()
    memprofiler.connect(_PlainModule(_cpu_cfg()))
    with pytest.raises(FTSConfigurationError, match="set `memprofiler_cfg.save_dir`, or pass `log_dir`"):
        memprofiler.init_memprof_log_dir()
    memprofiler = MemProfiler()
    memprofiler.connect(_PlainModule(_cpu_cfg(save_dir=str(tmp_path / "explicit"))))
    memprofiler.init_memprof_log_dir()
    assert memprofiler.memprof_log_dir == tmp_path / "explicit"
