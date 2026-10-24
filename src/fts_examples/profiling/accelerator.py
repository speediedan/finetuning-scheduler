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

import torch


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


def record_allocator_history() -> None:
    """Starts recording allocator history on the current accelerator, if its backend supports it."""
    if (memory_mod := allocator_history_module()) is not None:
        memory_mod._record_memory_history()
