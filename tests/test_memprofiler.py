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
"""The MemProfiler's device-generic collection, exercised without an accelerator."""
import warnings
from types import SimpleNamespace

import pytest
import torch
import yaml
from lightning.pytorch import Trainer

pytest.importorskip("psutil")

from finetuning_scheduler.profiling import accelerator as accel  # noqa: E402
from fts_examples.profiling import (ExtendedPyTorchProfiler, MemProfiler, MemProfilerCfg, MemProfilerFuncs,  # noqa: E402
                                    ProfilerHooksMixin)
from tests.helpers.boring_models import BoringModel  # noqa: E402
from tests.helpers.runif import RunIf  # noqa: E402


@pytest.fixture
def no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


@pytest.fixture
def fake_accelerator(monkeypatch):
    """An accelerator backend (``fakeacc``) with memory statistics but without allocator history."""
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cpu"))
    monkeypatch.setattr(accel, "accelerator_type", lambda: "fakeacc")
    for mod in ("finetuning_scheduler.profiling.config", "finetuning_scheduler.profiling.memprofiler"):
        monkeypatch.setattr(f"{mod}.accelerator_backend", lambda: "fakeacc")
    monkeypatch.setattr("finetuning_scheduler.profiling.memprofiler.accelerator_type", lambda: "fakeacc")


def test_deprecated_collect_funcs_keys_merge():
    with pytest.warns(FutureWarning, match="collect_funcs.cuda` is deprecated"):
        funcs = MemProfilerFuncs(cuda={"validation_step"}, accelerator={"training_step"})
    assert funcs.accelerator == {"training_step", "validation_step"}
    assert funcs.cuda is None
    with pytest.warns(FutureWarning, match="collect_funcs.cuda_allocator_history` is deprecated"):
        funcs = MemProfilerFuncs(cuda_allocator_history={"training_step"})
    assert funcs.allocator_history == {"training_step"} and funcs.cuda_allocator_history is None
    assert "cuda" not in funcs.collect_types() and "accelerator" in funcs.collect_types()


def test_deprecated_allocator_history_key(no_accelerator):
    with pytest.warns(FutureWarning, match="`cuda_allocator_history` is deprecated"):
        cfg = MemProfilerCfg(cuda_allocator_history=True)
    assert cfg.allocator_history and cfg.cuda_allocator_history is None
    # an enabled config on a host without an accelerator then disables it
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cfg = MemProfilerCfg(enabled=True, cuda_allocator_history=True)
    assert [w.category for w in record] == [FutureWarning, UserWarning]
    assert "no accelerator detected" in str(record[1].message)
    assert not cfg.allocator_history
    # an explicitly set deprecated key overrides the new key's default, but not an explicitly set new key
    for new, old, expected in ((None, False, False), (None, True, True), (False, True, False), (True, False, True)):
        with pytest.warns(FutureWarning):
            assert MemProfilerCfg(allocator_history=new, cuda_allocator_history=old).allocator_history is expected
    assert MemProfilerCfg().allocator_history is False


def test_no_accelerator_disables_accelerator_collection(no_accelerator, recwarn):
    cfg = MemProfilerCfg(enabled=True)  # default collection does not warn
    assert not [w for w in recwarn.list if "accelerator" in str(w.message)]
    assert cfg.collect_funcs.accelerator == set() and cfg.collect_funcs.allocator_history == set()
    assert cfg.collect_funcs.cpu == {"training_step"} and cfg.collect_funcs.fsdp == {"training_step"}
    with pytest.warns(UserWarning, match="no accelerator detected"):
        MemProfilerCfg(enabled=True, collect_funcs=MemProfilerFuncs(accelerator={"validation_step"}))


def test_backend_without_allocator_history(fake_accelerator):
    with pytest.warns(UserWarning, match="`fakeacc` backend does not support allocator history"):
        cfg = MemProfilerCfg(enabled=True, allocator_history=True)
    assert not cfg.allocator_history and cfg.collect_funcs.allocator_history == set()
    assert cfg.collect_funcs.accelerator == {"training_step"}


def test_accelerator_helpers(monkeypatch):
    assert accel.allocator_history_module("cuda") is torch.cuda.memory
    assert accel.allocator_history_module("xpu") is torch.xpu.memory
    assert accel.allocator_history_module("mps") is None
    assert accel.allocator_history_module("not_a_backend") is None
    monkeypatch.setattr(accel, "accelerator_type", lambda: None)
    assert accel.allocator_history_module() is None and accel.accelerator_backend() is None
    accel.record_allocator_history()  # a no-op without an accelerator
    monkeypatch.setattr(accel, "accelerator_type", lambda: "cuda")
    monkeypatch.setattr(torch.version, "hip", "7.0")
    assert accel.accelerator_backend() == "rocm"
    monkeypatch.setattr(torch.version, "hip", None)
    assert accel.accelerator_backend() == "cuda"
    monkeypatch.undo()
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    assert accel.accelerator_type() is None


def test_accelerator_memory_stats_merge(monkeypatch):
    def unsupported():
        raise RuntimeError("memory stats are not supported")

    monkeypatch.setattr(accel, "accelerator_type", lambda: "cuda")
    monkeypatch.setattr(torch.accelerator, "memory_stats", lambda: {"allocated_bytes.all.current": 1})
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: {"allocated_bytes.all.current": 2, "num_oom_rejections": 0})
    # device-generic values win, the backend adds the keys they lack
    assert accel.accelerator_memory_stats() == {"allocated_bytes.all.current": 1, "num_oom_rejections": 0}
    monkeypatch.setattr(torch.accelerator, "memory_stats", unsupported)
    assert accel.accelerator_memory_stats() == {"allocated_bytes.all.current": 2, "num_oom_rejections": 0}
    monkeypatch.setattr(accel, "accelerator_type", lambda: "fakeacc")  # a backend with no `memory_stats` of its own
    with pytest.raises(RuntimeError, match="not supported"):
        accel.accelerator_memory_stats()
    monkeypatch.setattr(torch.accelerator, "memory_stats", lambda: {"allocated_bytes.all.current": 1})
    assert accel.accelerator_memory_stats() == {"allocated_bytes.all.current": 1}
    # a failing supplement keeps the device-generic statistics
    monkeypatch.setattr(accel, "accelerator_type", lambda: "cuda")
    monkeypatch.setattr(torch.cuda, "memory_stats", unsupported)
    with pytest.warns(UserWarning, match="Skipping the `cuda` backend's own memory statistics"):
        assert accel.accelerator_memory_stats() == {"allocated_bytes.all.current": 1}


def _connected_profiler(cfg: MemProfilerCfg, tmp_path) -> MemProfiler:
    memprofiler = MemProfiler()
    memprofiler.connect(SimpleNamespace(memprofiler_cfg=cfg))
    memprofiler.memprof_log_dir = tmp_path
    return memprofiler


def test_accelerator_stats_collection(fake_accelerator, monkeypatch, tmp_path):
    with pytest.warns(UserWarning, match="does not support allocator history"):
        cfg = MemProfilerCfg(enabled=True, enable_memory_hooks=False, allocator_history=True)
    memprofiler = _connected_profiler(cfg, tmp_path)
    monkeypatch.setattr(torch.accelerator, "memory_stats", lambda: {"allocated_bytes.all.current": 512})
    memprofiler._collect_snap((0, "training_step", 2, "start"))
    memprofiler._collect_snap((0, "training_step", 2, "end"))
    assert memprofiler.memory_stats["0.training_step.2.start"]["allocated_bytes.all.current"] == 512
    assert {"rss", "vms"} <= set(memprofiler.memory_stats["0.training_step.2.end"])
    assert memprofiler._state.accelerator_stats_logged

    def unsupported():
        raise RuntimeError("memory stats are not supported")

    monkeypatch.setattr(torch.accelerator, "memory_stats", unsupported)
    with pytest.warns(UserWarning, match="`fakeacc` backend does not provide them"):
        memprofiler._collect_snap((0, "training_step", 3, "start"))
    assert cfg.collect_funcs.accelerator == set()
    assert "rss" in memprofiler.memory_stats["0.training_step.3.start"]


def test_allocator_history_snap(fake_accelerator, monkeypatch, tmp_path):
    dumped = []
    fake_memory = SimpleNamespace(_record_memory_history=lambda: None, _dump_snapshot=dumped.append)
    monkeypatch.setattr("finetuning_scheduler.profiling.memprofiler.allocator_history_module",
                        lambda device_type: fake_memory)
    memprofiler = _connected_profiler(MemProfilerCfg(), tmp_path)
    memprofiler.allocator_history_snap("0.training_step.2.end")
    with pytest.warns(FutureWarning, match="cuda_allocator_history_snap` is deprecated"):
        memprofiler.cuda_allocator_history_snap("0.training_step.2.start")
    assert dumped == [str(tmp_path / f"fakeacc_alloc_rank_0.training_step.2.{ctx}.pickle") for ctx in ("end", "start")]
    # a backend without allocator history skips the snapshot rather than failing
    monkeypatch.setattr("finetuning_scheduler.profiling.memprofiler.allocator_history_module", lambda device_type: None)
    with pytest.warns(UserWarning, match="`fakeacc` backend does not support allocator history"):
        memprofiler.allocator_history_snap("0.training_step.3.end")
    assert len(dumped) == 2


def test_extended_profiler_activities(monkeypatch):
    from torch.profiler import ProfilerActivity

    monkeypatch.setattr("fts_examples.profiling.extended_profiler._KINETO_AVAILABLE", True)
    for device_type, expected in ((None, [ProfilerActivity.CPU]), ("mps", [ProfilerActivity.CPU]),
                                  ("cuda", [ProfilerActivity.CPU, ProfilerActivity.CUDA]),
                                  ("xpu", [ProfilerActivity.CPU, ProfilerActivity.XPU])):
        monkeypatch.setattr("fts_examples.profiling.extended_profiler.accelerator_type", lambda: device_type)
        assert ExtendedPyTorchProfiler._set_activities(None, {}) == expected


class ProfiledBoringModel(ProfilerHooksMixin, BoringModel):
    @MemProfiler.memprofilable
    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)


def test_memprofiler_cpu_fit(no_accelerator, tmp_path):
    """A CPU fit collects host memory and saved-tensor hook statistics, and no accelerator statistics."""
    cfg = MemProfilerCfg(enabled=True, save_dir=tmp_path / "memprofiler")
    model = ProfiledBoringModel(memprofiler_cfg=cfg)
    assert not model.allocator_history
    with pytest.warns(FutureWarning, match="ProfilerHooksMixin.cuda_allocator_history` is deprecated"):
        assert not model.cuda_allocator_history
    trainer = Trainer(default_root_dir=tmp_path, accelerator="cpu", devices=1, max_steps=4, limit_val_batches=0,
                      logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model)
    with open(tmp_path / "memprofiler" / "rank_0_memory_stats.yaml") as f:
        stats = yaml.safe_load(f)
    assert set(stats) == {f"0.training_step.2.{ctx}" for ctx in ("start", "end")}
    end_stats = stats["0.training_step.2.end"]
    assert {"rss", "vms", "npp_diff", "rss_diff"} <= set(end_stats)
    assert not any(k.startswith("allocated_bytes") for k in end_stats)
    assert not list((tmp_path / "memprofiler").glob("*_alloc_rank_*.pickle"))


@RunIf(min_cuda_gpus=1)
def test_accelerator_stats_match_cuda_stats():
    """On CUDA and ROCm, the device-generic statistics the MemProfiler collects carry every key
    ``torch.cuda.memory_stats`` does, so moving collection to ``torch.accelerator`` loses none."""
    device = torch.device("cuda", torch.cuda.current_device())
    keep = torch.ones(1024, device=device)
    stats = accel.accelerator_memory_stats()
    assert set(torch.cuda.memory_stats()) <= set(stats)
    assert stats["allocated_bytes.all.current"] >= keep.nbytes
    assert accel.allocator_history_module() is torch.cuda.memory
    assert accel.accelerator_backend() == ("rocm" if torch.version.hip else "cuda")
