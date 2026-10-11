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
Fine-Tuning Scheduler Core Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Framework-neutral type definitions for the Fine-Tuning Scheduler core.

"""
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol, overload, runtime_checkable

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


@runtime_checkable
class ParamGroupAddable(Protocol):
    """To structurally type ``optimizer`` s that support adding parameter groups.

    The members other than ``add_param_group`` mirror Lightning's ``Optimizable`` protocol, so any optimizer that
    satisfies one satisfies the other.
    """

    param_groups: list[dict[Any, Any]]
    defaults: dict[Any, Any]
    state: defaultdict[Tensor, Any]

    @overload
    def step(self, closure: None = ...) -> None:
        ...  # pragma: no cover

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...  # pragma: no cover

    def step(self, closure: Callable[[], float] | None = ...) -> float | None:
        ...  # pragma: no cover

    def state_dict(self) -> dict[str, dict[Any, Any]]:
        ...  # pragma: no cover

    def load_state_dict(self, state_dict: dict[str, dict[Any, Any]]) -> None:
        ...  # pragma: no cover

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
FTSLRSchedulerType = type[LRScheduler] | type[ReduceLROnPlateau]


class AutoStrEnum(Enum):
    def _generate_next_value_(name, start, count, last_values) -> str:  # type: ignore
        return name  # type: ignore[return-value]
