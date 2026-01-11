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
Fine-Tuning Scheduler Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Type definitions required for Fine-Tuning Scheduler.

"""
from typing import Any, Protocol, runtime_checkable, Type
from typing_extensions import TypeAlias
from enum import Enum

import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from lightning.fabric.utilities.types import Optimizable
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


@runtime_checkable
class ParamGroupAddable(Optimizable, Protocol):
    """To structurally type ``optimizer`` s that support adding parameter groups."""

    def add_param_group(self, param_group: dict[Any, Any]) -> None:
        ...  # pragma: no cover


# TODO: improve FTSLRSchedulerType naming/typing once corresponding changes made upstream in Lightning
supported_lrs = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
    "CosineAnnealingWarmRestarts",
    "ConstantLR",
    "LinearLR",
]
FTSLRSchedulerTypeTuple = tuple(getattr(torch.optim.lr_scheduler, lr_class) for lr_class in supported_lrs)
FTSLRSchedulerType = Type[LRScheduler] | Type[ReduceLROnPlateau]

BaseCallbackDepType: TypeAlias = Type[EarlyStopping] | Type[ModelCheckpoint]

class AutoStrEnum(Enum):
    def _generate_next_value_(name, start, count, last_values) -> str:  # type: ignore
        return name  # type: ignore[return-value]
