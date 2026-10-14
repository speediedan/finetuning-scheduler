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
Fine-Tuning Scheduler Testing Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conformance cases for :class:`~finetuning_scheduler.core.views.TunableView` implementations
(:mod:`~finetuning_scheduler.testing.conformance`) and known-good fixture views
(:mod:`~finetuning_scheduler.testing.fixtures`). Importing this package does not import pytest or Lightning, and
importing :mod:`finetuning_scheduler` does not import this package.

.. warning:: This is an experimental API and may change in a future release.
"""
from finetuning_scheduler.testing.conformance import (ConformanceCase, ConformanceError, ConformanceReport,
                                                      pytest_params, TunableViewConformance)

__all__ = ["ConformanceCase", "ConformanceError", "ConformanceReport", "pytest_params", "TunableViewConformance"]
