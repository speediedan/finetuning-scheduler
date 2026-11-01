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
MemProfiler FSDP2 Reports
^^^^^^^^^^^^^^^^^^^^^^^^^

Machine-readable ``FSDP2MemTracker`` reports: one JSON document per rank, their aggregation across ranks, and the
availability of each MemProfiler collector on the current backend.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from finetuning_scheduler.core.exceptions import configuration_error_cls
from finetuning_scheduler.profiling.accelerator import (accelerator_backend, accelerator_memory_stats,
                                                        allocator_history_module)

FSDP_MEM_REPORT_VERSION = 1
FSDP_MEM_AGGREGATE_FILENAME = "fsdp_memory_report_aggregate.json"


def fsdp_mem_report_filename(rank: int) -> str:
    """The name of the JSON ``FSDP2MemTracker`` report rank ``rank`` writes."""
    return f"rank_{rank}_fsdp_memory_report.json"


def collector_availability() -> dict[str, bool]:
    """Which MemProfiler collectors the current process can use.

    Each entry is measured in this process rather than read from a table, so a report records what the backend it
    ran on actually provided:

    - ``accelerator_memory_stats``: the allocator answered a memory statistics query (an allocator that has not
      allocated yet answers with no statistics, which still counts);
    - ``allocator_history``: the backend provides allocator history recording and snapshots;
    - ``fsdp_mem_tracker``: ``torch.distributed`` is available, which ``FSDP2MemTracker`` needs;
    - ``host_memory``: ``psutil``, which host memory tracking uses, is importable.

    Returns:
        dict[str, bool]: Each collector's availability.
    """
    accelerator_stats = accelerator_backend() is not None
    if accelerator_stats:
        try:
            accelerator_memory_stats()
        except RuntimeError:
            accelerator_stats = False
    return {
        "accelerator_memory_stats": accelerator_stats,
        "allocator_history": allocator_history_module() is not None,
        "fsdp_mem_tracker": torch.distributed.is_available(),
        "host_memory": importlib.util.find_spec("psutil") is not None,
    }


def _label(key: Any) -> str:
    # tracker categories and module states are `str` enums whose value is the label `display_snapshot` prints
    return str(getattr(key, "value", key))


def device_snapshot_json(snapshot: dict[Any, dict[Any, int]]) -> dict[str, dict[str, int]]:
    """A tracker snapshot (``{device: {category: bytes}}``) with plain string keys."""
    return {str(device): {_label(cat): int(nbytes) for cat, nbytes in cats.items()}
            for device, cats in snapshot.items()}


def fsdp_mem_entry(fn_name: str, iter_idx: int, phase: int | None, peak: dict[Any, dict[Any, int]],
                   memory_tracking: Any, depth: int) -> dict[str, Any]:
    """One profiled call's JSON record: its peak snapshot and its module-wise state snapshots.

    Args:
        fn_name: The profiled function.
        iter_idx: The profiled iteration.
        phase: The training phase the call ran in, if the loop reports one.
        peak: The tracker's peak snapshot for this call.
        memory_tracking: The tracker's ``memory_tracking`` for this call (module to module memory statistics).
        depth: Modules deeper than this (dot count plus one, as ``display_modulewise_snapshots`` counts) are omitted.

    Returns:
        dict[str, Any]: The record.
    """
    modules: dict[str, dict[str, list[dict[str, dict[str, int]]]]] = {}
    for mod_stats in memory_tracking.values():
        if mod_stats.mod_fqn.count(".") + 1 > depth:
            continue
        modules[mod_stats.mod_fqn] = {_label(state): [device_snapshot_json(snap) for snap in snaps]
                                      for state, snaps in mod_stats.snapshots.items()}
    return {"fn_name": fn_name, "iter": iter_idx, "phase": phase, "peak": device_snapshot_json(peak),
            "modules": modules}


def fsdp_mem_report(rank: int, world_size: int, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """A rank's JSON report: its identity, the backend and collectors it ran with, and its entries (bytes)."""
    return {"version": FSDP_MEM_REPORT_VERSION, "rank": rank, "world_size": world_size,
            "backend": accelerator_backend() or "cpu", "collectors": collector_availability(), "units": "B",
            "entries": entries}


def _entry_key(entry: dict[str, Any]) -> tuple[str, int, int | None]:
    return entry["fn_name"], entry["iter"], entry["phase"]


def _per_rank_totals(entry: dict[str, Any]) -> dict[str, int]:
    """A peak snapshot summed over the devices a rank used, per category."""
    totals: dict[str, int] = {}
    for cats in entry["peak"].values():
        for cat, nbytes in cats.items():
            totals[cat] = totals.get(cat, 0) + nbytes
    return totals


def load_fsdp_mem_reports(report_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """Every rank's JSON report in ``report_dir``, ordered by rank.

    Raises:
        FTSConfigurationError: If there is no report, a report has a version this reader does not know, two
            reports claim one rank, the ranks disagree on the world size, or a rank's report is missing.
    """
    paths = sorted(Path(report_dir).glob("rank_*_fsdp_memory_report.json"))
    if not paths:
        raise configuration_error_cls()(f"No `rank_<n>_fsdp_memory_report.json` FSDP2 reports found in {report_dir}.")
    reports: dict[int, dict[str, Any]] = {}
    for path in paths:
        with open(path) as f:
            report = json.load(f)
        if report.get("version") != FSDP_MEM_REPORT_VERSION:
            raise configuration_error_cls()(f"{path} has FSDP2 report version {report.get('version')!r}; this reader"
                                            f" understands version {FSDP_MEM_REPORT_VERSION} only.")
        if report["rank"] in reports:
            raise configuration_error_cls()(f"Two FSDP2 reports in {report_dir} claim rank {report['rank']}.")
        reports[report["rank"]] = report
    world_sizes = {report["world_size"] for report in reports.values()}
    if len(world_sizes) != 1:
        raise configuration_error_cls()(f"The FSDP2 reports in {report_dir} disagree on the world size: {world_sizes}.")
    if missing := set(range(world_sizes.pop())) - set(reports):
        raise configuration_error_cls()(f"FSDP2 reports for ranks {sorted(missing)} are missing from {report_dir}.")
    return [reports[rank] for rank in sorted(reports)]


def aggregate_fsdp_mem_reports(report_dir: str | os.PathLike, write: bool = True) -> dict[str, Any]:
    """Aggregates every rank's JSON ``FSDP2MemTracker`` report in ``report_dir`` across ranks.

    For each profiled call (function, iteration and phase) every rank recorded, each peak category (its bytes summed
    over the devices the rank used) is reported per rank, with its maximum, minimum and mean across ranks. The
    maximum is the figure that decides whether a configuration fits, since the most loaded rank runs out first.

    Args:
        report_dir: The directory holding the ``rank_<n>_fsdp_memory_report.json`` reports.
        write: Whether to also write the aggregate to ``report_dir / "fsdp_memory_report_aggregate.json"``.

    Returns:
        dict[str, Any]: The aggregate.

    Raises:
        FTSConfigurationError: If the reports cannot be loaded (see :func:`load_fsdp_mem_reports`), or the ranks did
            not profile the same calls, which a per-call comparison across ranks cannot be made from.
    """
    reports = load_fsdp_mem_reports(report_dir)
    entries_by_rank = [{_entry_key(entry): entry for entry in report["entries"]} for report in reports]
    keys = set(entries_by_rank[0])
    for report, rank_entries in zip(reports[1:], entries_by_rank[1:]):
        if set(rank_entries) != keys:
            raise configuration_error_cls()(
                f"Rank {report['rank']} profiled different calls than rank 0 (function, iteration, phase): only on"
                f" rank 0 {sorted(keys - set(rank_entries), key=str)}, only on rank {report['rank']}"
                f" {sorted(set(rank_entries) - keys, key=str)}."
            )
    aggregated = []
    for key in sorted(keys, key=lambda k: (k[0], k[1], -1 if k[2] is None else k[2])):
        per_rank = [_per_rank_totals(rank_entries[key]) for rank_entries in entries_by_rank]
        categories = sorted(set().union(*per_rank))
        peak = {}
        for cat in categories:
            values = [totals.get(cat, 0) for totals in per_rank]
            peak[cat] = {"per_rank": values, "max": max(values), "min": min(values), "mean": mean(values)}
        aggregated.append({"fn_name": key[0], "iter": key[1], "phase": key[2], "peak": peak})
    aggregate = {"version": FSDP_MEM_REPORT_VERSION, "world_size": len(reports), "units": "B",
                 "backends": [report["backend"] for report in reports], "entries": aggregated}
    if write:
        with open(Path(report_dir) / FSDP_MEM_AGGREGATE_FILENAME, "w") as f:
            json.dump(aggregate, f, indent=2)
    return aggregate
