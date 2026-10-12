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
from typing import TYPE_CHECKING, Any

from finetuning_scheduler.core.types import (AutoStrEnum, FTSLRSchedulerType, FTSLRSchedulerTypeTuple,
                                             ParamGroupAddable, supported_lrs)

if TYPE_CHECKING:
    from finetuning_scheduler.lightning.types import BaseCallbackDepType


def __getattr__(name: str) -> Any:
    # the Lightning-bound type loads on first access, so importing this module does not import Lightning
    if name == "BaseCallbackDepType":
        from finetuning_scheduler.lightning.types import BaseCallbackDepType

        globals()[name] = BaseCallbackDepType
        return BaseCallbackDepType
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AutoStrEnum",
    "BaseCallbackDepType",
    "FTSLRSchedulerType",
    "FTSLRSchedulerTypeTuple",
    "ParamGroupAddable",
    "supported_lrs",
]
