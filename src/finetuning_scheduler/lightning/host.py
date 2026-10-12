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
Fine-Tuning Scheduler Lightning Host
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~finetuning_scheduler.core.host.TrainingHost` for a Lightning ``Trainer``. This module is where the
Lightning integration's uses of ``Trainer`` and ``Strategy`` internals are to be gathered.

.. warning::

    Experimental: the ``TrainingHost`` protocol may change in minor releases until the direct PyTorch API stabilizes.

"""
import os
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import LRSchedulerConfig

from finetuning_scheduler.core.host import LRStepConfig


class LightningHost:
    """Present a Lightning ``Trainer`` and ``LightningModule`` as a
    :class:`~finetuning_scheduler.core.host.TrainingHost`.

    Args:
        trainer: The ``Trainer`` running the fit.
        pl_module: The ``LightningModule`` being trained.
    """

    def __init__(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.trainer = trainer
        self.pl_module = pl_module

    @property
    def module(self) -> torch.nn.Module:
        return self.pl_module

    def get_optimizers(self) -> list[Any]:
        return list(self.trainer.strategy.optimizers)

    def set_optimizer(self, idx: int, optimizer: Any) -> None:
        optimizers = list(self.trainer.strategy.optimizers)
        optimizers[idx:idx + 1] = [optimizer]
        self.trainer.strategy.optimizers = optimizers

    def get_lr_schedulers(self) -> list[tuple[Any, LRStepConfig]]:
        return [(cfg.scheduler, LRStepConfig(interval=cfg.interval, frequency=cfg.frequency,  # type: ignore[arg-type]
                                             monitor=cfg.monitor)) for cfg in self.trainer.lr_scheduler_configs]

    def set_lr_scheduler(self, idx: int, scheduler: Any, step_cfg: LRStepConfig) -> None:
        configs = list(self.trainer.strategy.lr_scheduler_configs)
        new_cfg = LRSchedulerConfig(scheduler=scheduler, interval=step_cfg.interval, frequency=step_cfg.frequency,
                                    monitor=step_cfg.monitor)
        configs[idx:idx + 1] = [new_cfg]
        self.trainer.strategy.lr_scheduler_configs = configs

    @property
    def current_epoch(self) -> int:
        return self.trainer.current_epoch

    @property
    def global_step(self) -> int:
        return self.trainer.global_step

    @property
    def max_epochs(self) -> int | None:
        return self.trainer.fit_loop.max_epochs

    def set_max_epochs(self, max_epochs: int) -> None:
        self.trainer.fit_loop.max_epochs = max_epochs

    def advance_epoch(self) -> None:
        """Lightning's fit loop counts epochs."""

    def advance_step(self, steps: int = 1) -> None:
        """Lightning's fit loop counts optimizer steps."""

    def load_counters(self, current_epoch: int, global_step: int) -> None:
        raise NotImplementedError("Under Lightning, resume through the Trainer's checkpointing (`ckpt_path`).")

    @property
    def is_global_zero(self) -> bool:
        return self.trainer.is_global_zero

    @property
    def world_size(self) -> int:
        return self.trainer.world_size

    def barrier(self, name: str | None = None) -> None:
        self.trainer.strategy.barrier(name)

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return self.trainer.strategy.broadcast(obj, src)

    def reduce_max(self, value: int) -> int:
        t = torch.tensor(int(value), device=self.trainer.strategy.root_device)
        if self.trainer.world_size > 1:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return int(t.item())

    def reduce_any(self, value: bool) -> bool:
        return bool(self.reduce_max(int(value)))

    def all_gather_object(self, obj: Any) -> list[Any]:
        if self.trainer.world_size == 1:
            return [obj]
        gathered: list[Any] = [None] * self.trainer.world_size
        torch.distributed.all_gather_object(gathered, obj)
        return gathered

    @property
    def log_dir(self) -> str | os.PathLike | None:
        return self.trainer.log_dir

    def on_compile_reset(self) -> None:
        compiler_ctx = getattr(self.pl_module, "_compiler_ctx", None)
        if compiler_ctx and compiler_ctx.get("compiler", None) == "dynamo":
            torch._dynamo.reset()

    def rng_devices(self) -> list[torch.device]:
        root = self.trainer.strategy.root_device
        return [root] if root.type == "cuda" else []
