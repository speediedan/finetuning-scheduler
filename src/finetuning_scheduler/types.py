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
from typing import Any, Dict, Protocol, runtime_checkable, Type, Union

import torch
from lightning.fabric.utilities.types import _TORCH_LRSCHEDULER, Optimizable, ReduceLROnPlateau


@runtime_checkable
class ParamGroupAddable(Optimizable, Protocol):
    """To structurally type ``optimizer`` s that support adding parameter groups."""

    def add_param_group(self, param_group: Dict[Any, Any]) -> None:
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
FTSLRSchedulerType = Union[Type[_TORCH_LRSCHEDULER], Type[ReduceLROnPlateau]]
