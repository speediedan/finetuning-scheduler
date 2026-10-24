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
MemProfiler Accelerator Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Device-generic helpers that let the MemProfiler collect accelerator memory statistics and allocator history without
assuming CUDA.
"""
from types import ModuleType
from typing import Any

import torch
from lightning.fabric.utilities import rank_zero_warn


def accelerator_type() -> str | None:
    """The device type of the current accelerator (``"cuda"`` on both CUDA and ROCm builds), or ``None`` when no
    accelerator is available."""
    if not torch.accelerator.is_available():
        return None
    acc = torch.accelerator.current_accelerator()
    return acc.type if acc is not None else None


def accelerator_backend() -> str | None:
    """A label for the current accelerator backend: ``"rocm"`` for a ROCm (HIP) build of PyTorch, whose
    accelerator device type is ``"cuda"``, and the device type otherwise."""
    device_type = accelerator_type()
    if device_type == "cuda" and torch.version.hip:
        return "rocm"
    return device_type


def allocator_history_module(device_type: str | None = None) -> ModuleType | None:
    """The ``torch.<device type>.memory`` module providing allocator history recording and snapshots
    (``_record_memory_history`` and ``_dump_snapshot``) for ``device_type``, or ``None`` if that backend provides
    neither.

    Args:
        device_type: The accelerator device type to look up. Defaults to the current accelerator's.

    Returns:
        ModuleType | None: The backend's memory module, if it supports allocator history.
    """
    device_type = device_type or accelerator_type()
    if device_type is None:
        return None
    memory_mod = getattr(getattr(torch, device_type, None), "memory", None)
    if all(hasattr(memory_mod, attr) for attr in ("_record_memory_history", "_dump_snapshot")):
        return memory_mod
    return None


def accelerator_memory_stats() -> dict[str, Any]:
    """The current accelerator's memory allocator statistics.

    The device-generic ``torch.accelerator.memory_stats`` provides them, and any key the backend's own
    ``torch.<device type>.memory_stats`` adds is included as well (``num_oom_rejections`` on CUDA as of PyTorch
    2.15), so collection through the device-generic API loses no statistic the backend reports. The two calls sample
    the allocator one after the other, so a supplementary key can be one call newer than the device-generic keys
    beside it. A supplement that fails is skipped with a warning, keeping the device-generic statistics. A backend
    without device-generic statistics falls back to its own ``memory_stats``, if it has one.

    Returns:
        dict[str, Any]: The flattened allocator statistics.

    Raises:
        RuntimeError: If the backend provides no memory statistics.
    """
    device_type = accelerator_type()
    backend_stats_fn = None if device_type is None else getattr(getattr(torch, device_type, None), "memory_stats", None)
    try:
        stats = dict(torch.accelerator.memory_stats())
    except RuntimeError:
        if backend_stats_fn is None:
            raise
        return dict(backend_stats_fn())
    if backend_stats_fn is not None:
        try:
            supplement = backend_stats_fn()
        except RuntimeError as err:  # the supplement is additive, so its failure keeps the device-generic statistics
            rank_zero_warn(f"Skipping the `{device_type}` backend's own memory statistics, which failed: {err}")
        else:
            stats.update({k: v for k, v in supplement.items() if k not in stats})
    return stats


def record_allocator_history() -> None:
    """Starts recording allocator history on the current accelerator, if its backend supports it."""
    if (memory_mod := allocator_history_module()) is not None:
        memory_mod._record_memory_history()
