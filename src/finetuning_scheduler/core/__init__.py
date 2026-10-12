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
Fine-Tuning Scheduler Core
^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework-neutral scheduling core. Modules in this package import only ``torch``, ``yaml``, ``fsspec`` and the
standard library; they never import Lightning.

"""
from finetuning_scheduler.core.exceptions import (configuration_error_cls, FTSConfigurationError,
                                                  register_configuration_error)
from finetuning_scheduler.core.state import FTSState
from finetuning_scheduler.core.types import (AutoStrEnum, FTSLRSchedulerType, FTSLRSchedulerTypeTuple,
                                             ParamGroupAddable, supported_lrs)

__all__ = [
    "AutoStrEnum",
    "configuration_error_cls",
    "FTSConfigurationError",
    "FTSLRSchedulerType",
    "FTSLRSchedulerTypeTuple",
    "FTSState",
    "ParamGroupAddable",
    "register_configuration_error",
    "supported_lrs",
]
