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
r"""Fine-Tuning Scheduler.

Used to implement flexible fine-tuning training schedules
"""
# The implementation lives in ``finetuning_scheduler.lightning.fts``. This module keeps the established import path, and
# the API documentation page and anchors, which document the names listed in ``__all__``.
from typing import Any

from finetuning_scheduler.lightning import fts as _impl
from finetuning_scheduler.lightning.fts import FinetuningScheduler, STRATEGY_ADAPTERS  # noqa: F401

__all__ = ["FinetuningScheduler"]


def __getattr__(name: str) -> Any:
    # forward every other name, private ones included, so existing imports from this path keep resolving
    try:
        return getattr(_impl, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
