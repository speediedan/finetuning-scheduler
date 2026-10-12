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
Fine-Tuning Scheduler Training Hosts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The narrow seam between the scheduling core and whatever runs the training loop. A host owns the module, the
optimizers and lr schedulers, the loop counters, the process collectives and the device set; the core asks it, and never
reaches into a training framework itself.

.. warning::

    Experimental: the ``TrainingHost`` protocol may change in minor releases until the direct PyTorch API stabilizes.

"""
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class LRStepConfig:
    """How the host steps an lr scheduler.

    Attributes:
        interval: Step after every optimizer step (``"step"``) or every epoch (``"epoch"``).
        frequency: Step every ``frequency`` intervals.
        monitor: The metric passed to a scheduler that steps on a metric (e.g. ``ReduceLROnPlateau``).
    """

    interval: Literal["step", "epoch"] = "epoch"
    frequency: int = 1
    monitor: str | None = None


@runtime_checkable
class TrainingHost(Protocol):
    """What the scheduling core needs from the code that runs the training loop."""

    @property
    def module(self) -> nn.Module: ...

    def get_optimizers(self) -> list[Any]: ...

    def set_optimizer(self, idx: int, optimizer: Any) -> None: ...

    def get_lr_schedulers(self) -> list[tuple[Any, LRStepConfig]]: ...

    def set_lr_scheduler(self, idx: int, scheduler: Any, step_cfg: LRStepConfig) -> None: ...

    @property
    def current_epoch(self) -> int: ...

    @property
    def global_step(self) -> int: ...

    @property
    def max_epochs(self) -> int | None: ...

    def set_max_epochs(self, max_epochs: int) -> None: ...

    @property
    def is_global_zero(self) -> bool: ...

    @property
    def world_size(self) -> int: ...

    def barrier(self, name: str | None = None) -> None: ...

    def broadcast_object(self, obj: Any, src: int = 0) -> Any: ...

    def reduce_max(self, value: int) -> int: ...

    def reduce_any(self, value: bool) -> bool: ...

    def all_gather_object(self, obj: Any) -> list[Any]: ...

    @property
    def log_dir(self) -> str | os.PathLike | None: ...

    def on_compile_reset(self) -> None: ...

    def rng_devices(self) -> list[torch.device]: ...

    def advance_epoch(self) -> None:
        """Count a completed epoch (a no-op for a host whose framework counts epochs itself)."""
        ...

    def advance_step(self, steps: int = 1) -> None:
        """Count completed optimizer steps (a no-op for a host whose framework counts them itself)."""
        ...

    def load_counters(self, current_epoch: int, global_step: int) -> None:
        """Restore the epoch and step counters on resume."""
        ...


class TorchHost:
    """The :class:`TrainingHost` for a plain PyTorch training loop.

    Collectives run only when ``torch.distributed`` is initialized; the host never creates or destroys a process group.
    The loop counters are owned by the controller, which advances them through :meth:`advance_epoch` and
    :meth:`advance_step`.

    Args:
        module: The module being trained (possibly sharded with FSDP2 or tensor parallelism, or split across devices).
        process_group: The process group for collectives; the default group if ``None``.
        log_dir: Where schedules and other artifacts are written.
        rng_devices: The CUDA devices whose generators phase hooks fork; by default every device holding the module's
            parameters, plus the current CUDA device once CUDA is initialized.
    """

    def __init__(self, module: nn.Module, *, process_group: Any | None = None,
                 log_dir: str | os.PathLike | None = None,
                 rng_devices: Sequence[torch.device | int | str] | None = None) -> None:
        self._module = module
        self._process_group = process_group
        self._log_dir = log_dir
        self._rng_devices = None if rng_devices is None else [torch.device(d) if not isinstance(d, int)
                                                                  else torch.device("cuda", d) for d in rng_devices]
        self._optimizers: list[Any] = []
        self._lr_schedulers: list[tuple[Any, LRStepConfig]] = []
        self._current_epoch = 0
        self._global_step = 0
        self._max_epochs: int | None = None

    # -- module, optimizers and lr schedulers ------------------------------------------------------------------------
    @property
    def module(self) -> nn.Module:
        return self._module

    def get_optimizers(self) -> list[Any]:
        return list(self._optimizers)

    def set_optimizer(self, idx: int, optimizer: Any) -> None:
        if idx == len(self._optimizers):
            self._optimizers.append(optimizer)
        else:
            self._optimizers[idx] = optimizer

    def get_lr_schedulers(self) -> list[tuple[Any, LRStepConfig]]:
        return list(self._lr_schedulers)

    def set_lr_scheduler(self, idx: int, scheduler: Any, step_cfg: LRStepConfig) -> None:
        if idx == len(self._lr_schedulers):
            self._lr_schedulers.append((scheduler, step_cfg))
        else:
            self._lr_schedulers[idx] = (scheduler, step_cfg)

    # -- loop counters -----------------------------------------------------------------------------------------------
    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def global_step(self) -> int:
        return self._global_step

    def advance_epoch(self) -> None:
        self._current_epoch += 1

    def advance_step(self, steps: int = 1) -> None:
        self._global_step += steps

    def load_counters(self, current_epoch: int, global_step: int) -> None:
        self._current_epoch, self._global_step = current_epoch, global_step

    @property
    def max_epochs(self) -> int | None:
        return self._max_epochs

    def set_max_epochs(self, max_epochs: int) -> None:
        self._max_epochs = max_epochs

    # -- collectives -------------------------------------------------------------------------------------------------
    @property
    def _distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    @property
    def is_global_zero(self) -> bool:
        return not self._distributed or dist.get_rank() == 0

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self._process_group) if self._distributed else 1

    def barrier(self, name: str | None = None) -> None:
        if self._distributed:
            dist.barrier(group=self._process_group)

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        if not self._distributed:
            return obj
        holder = [obj]
        dist.broadcast_object_list(holder, src=src, group=self._process_group)
        return holder[0]

    def _reduce_device(self) -> torch.device:
        backend = str(dist.get_backend(self._process_group))
        if "nccl" in backend and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def reduce_max(self, value: int) -> int:
        if not self._distributed:
            return value
        t = torch.tensor(int(value), device=self._reduce_device())
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self._process_group)
        return int(t.item())

    def reduce_any(self, value: bool) -> bool:
        return bool(self.reduce_max(int(value)))

    def all_gather_object(self, obj: Any) -> list[Any]:
        if not self._distributed:
            return [obj]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, obj, group=self._process_group)
        return gathered

    # -- artifacts, compilation and devices --------------------------------------------------------------------------
    @property
    def log_dir(self) -> str | os.PathLike | None:
        return self._log_dir

    def on_compile_reset(self) -> None:
        """Discard compiled graphs at a phase transition (a thaw changes which parameters require grad, which
        dynamo guards on)."""
        torch._dynamo.reset()

    def rng_devices(self) -> list[torch.device]:
        if self._rng_devices is not None:
            return list(self._rng_devices)
        devices = {p.device for p in self._module.parameters() if p.device.type == "cuda"}
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            devices.add(torch.device("cuda", torch.cuda.current_device()))
        return sorted(devices, key=lambda d: (d.type, d.index if d.index is not None else -1))
