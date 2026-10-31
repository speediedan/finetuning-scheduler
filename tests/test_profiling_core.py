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
import json
import subprocess
import sys

import pytest
import torch

pytest.importorskip("psutil")

from finetuning_scheduler.core.exceptions import FTSConfigurationError  # noqa: E402
from finetuning_scheduler.profiling import (MemProfiler, MemProfilerCfg, MemProfilerFuncs,  # noqa: E402
                                            MemProfilerSchedule, aggregate_fsdp_mem_reports,
                                            collector_availability)

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


class _ShardedModule(torch.nn.Module):
    """A two-layer module whose input width sets its activation memory, so calls given different batches peak
    differently."""

    def __init__(self, cfg: MemProfilerCfg) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 2))
        self.memprofiler_cfg = cfg
        self.memprofiler: MemProfiler | None = None

    @MemProfiler.memprofilable
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        return self.layers(batch).sum()


def _fsdp_cfg(save_dir, **kwargs) -> MemProfilerCfg:
    return MemProfilerCfg(enabled=True, enable_memory_hooks=False, enable_saved_tensors_hooks=False,
                          track_fsdp_mem=True, save_dir=str(save_dir),
                          collect_funcs=MemProfilerFuncs(default=set(), fsdp={"training_step"}),
                          schedule=MemProfilerSchedule(warmup_iters=1, max_iter=4), **kwargs)


def test_fsdp2_reports_record_each_calls_peak_and_phase(no_accelerator, single_process_pg, tmp_path):
    """A plain loop under FSDP2 on CPU: each profiled call's report carries that call's own peak, a phase change
    rebuilds the tracker with the optimizer then current, and the JSON report and its aggregate agree with it."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cpu", (1,))
    module = _ShardedModule(_fsdp_cfg(tmp_path, aggregate_fsdp_mem_reports=True, fsdp_mem_tracker_units="B"))
    for layer in module.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(module, mesh=mesh)
    state = {"phase": 0, "optimizer": torch.optim.Adam(module.parameters())}
    module.memprofiler = MemProfiler()
    module.memprofiler.connect(module, optimizer=lambda: state["optimizer"], log_dir=tmp_path,
                               phase=lambda: state["phase"])
    module.memprofiler.init_memprof_log_dir()
    module.memprofiler.maybe_init_fsdp_mem_tracker()
    first_tracker = module.memprofiler.fsdp_mem_tracker
    batch_rows = {1: 4, 2: 64, 3: 4, 4: 4, 5: 4}
    for step in range(1, 6):
        if step == 4:  # a phase transition that replaces the optimizer
            state["phase"], state["optimizer"] = 1, torch.optim.SGD(module.parameters(), lr=0.1)
        optimizer = state["optimizer"]
        optimizer.zero_grad()
        module.training_step(torch.ones(batch_rows[step], 8)).backward()
        optimizer.step()
    memprofiler = module.memprofiler
    memprofiler.dump_memory_stats()

    assert memprofiler.fsdp_mem_tracker is not first_tracker
    assert memprofiler.fsdp_mem_tracker._optm is state["optimizer"]
    phases = {key: phase for key, phase in memprofiler.fsdp_mem_phases.items()}
    assert phases == {("training_step", 2): 0, ("training_step", 3): 0, ("training_step", 4): 1}
    peak_totals = {key: sum(cats["Total"] for cats in peak.values())
                   for key, peak in memprofiler.fsdp_peak_mem_stats.items()}
    # the wide batch peaks higher; one shared (last) peak for every call would make these equal
    assert peak_totals[("training_step", 2)] > peak_totals[("training_step", 3)]

    with open(tmp_path / "rank_0_fsdp_memory_report.json") as f:
        report = json.load(f)
    assert (report["rank"], report["world_size"], report["backend"]) == (0, 1, "cpu")
    assert report["collectors"]["fsdp_mem_tracker"] and not report["collectors"]["accelerator_memory_stats"]
    entries = {(e["fn_name"], e["iter"]): e for e in report["entries"]}
    assert {key: e["phase"] for key, e in entries.items()} == phases
    assert {key: sum(c["Total"] for c in e["peak"].values()) for key, e in entries.items()} == peak_totals
    assert all(e["modules"] for e in entries.values())
    aggregate = json.loads((tmp_path / "fsdp_memory_report_aggregate.json").read_text())
    assert [(e["iter"], e["phase"], e["peak"]["Total"]["max"]) for e in aggregate["entries"]] == \
        [(i, phases[("training_step", i)], peak_totals[("training_step", i)]) for i in (2, 3, 4)]
    peak_text = {i: (tmp_path / f"fsdp_mem_rank_0_training_step_{i}.log").read_text().split("Modulewise")[0]
                 for i in (2, 3)}
    assert "Peak FSDP memory stats" in peak_text[2]
    assert peak_text[2] != peak_text[3].replace("Iteration 3", "Iteration 2")


def _write_rank_report(report_dir, rank: int, world_size: int, peaks: dict, version: int = 1) -> None:
    entries = [{"fn_name": "training_step", "iter": it, "phase": None, "modules": {},
                "peak": {f"cuda:{rank}": {"Total": total, "Activation": total // 2}}} for it, total in peaks.items()]
    report = {"version": version, "rank": rank, "world_size": world_size, "backend": "cuda", "collectors": {},
              "units": "B", "entries": entries}
    (report_dir / f"rank_{rank}_fsdp_memory_report.json").write_text(json.dumps(report))


def test_fsdp2_reports_aggregate_across_ranks(tmp_path):
    _write_rank_report(tmp_path, 0, 2, {2: 100})
    _write_rank_report(tmp_path, 1, 2, {2: 300})
    aggregate = aggregate_fsdp_mem_reports(tmp_path)
    (entry,) = aggregate["entries"]
    assert entry["peak"]["Total"] == {"per_rank": [100, 300], "max": 300, "min": 100, "mean": 200}
    assert entry["peak"]["Activation"]["max"] == 150
    assert json.loads((tmp_path / "fsdp_memory_report_aggregate.json").read_text()) == aggregate


@pytest.mark.parametrize("setup, match", [
    ({0: {2: 1}}, r"ranks \[1\] are missing"),
    ({0: {2: 1}, 1: {2: 1, 3: 1}}, "Rank 1 profiled different calls"),
    ({0: {2: 1}, "v": {2: 1}}, "this reader understands version 1 only"),
    ({}, "No `rank_<n>_fsdp_memory_report.json`"),
], ids=["missing_rank", "different_calls", "unknown_version", "no_reports"])
def test_fsdp2_report_aggregation_refuses_what_it_cannot_compare(tmp_path, setup, match):
    for rank, peaks in setup.items():
        if rank == "v":
            _write_rank_report(tmp_path, 1, 2, peaks, version=2)
        else:
            _write_rank_report(tmp_path, rank, 2, peaks)
    with pytest.raises(FTSConfigurationError, match=match):
        aggregate_fsdp_mem_reports(tmp_path)


def test_aggregation_without_json_reports_is_refused():
    with pytest.raises(FTSConfigurationError, match="requires `dump_fsdp_mem_json`"):
        MemProfilerCfg(aggregate_fsdp_mem_reports=True, dump_fsdp_mem_json=False)


def test_collector_availability_is_measured(no_accelerator):
    availability = collector_availability()
    assert availability == {"accelerator_memory_stats": False, "allocator_history": availability["allocator_history"],
                            "fsdp_mem_tracker": torch.distributed.is_available(), "host_memory": True}
