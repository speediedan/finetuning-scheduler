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
Tunable View Protocol
^^^^^^^^^^^^^^^^^^^^^

The tunable view protocol: the contract between the logical parameter names a schedule uses and the physical
parameters an optimizer, checkpoint and sharding see, with a lifecycle role for each tensor. It is independent of
Fine-Tuning Scheduler, which is one consumer of it.

- :mod:`~finetuning_scheduler.tunable_view.views`: the :class:`~finetuning_scheduler.tunable_view.views.TunableView`
  protocol, its support record, the defaults of its optional members and the identity view;
- :mod:`~finetuning_scheduler.tunable_view.roles`: the role vocabulary and the ``role:`` selector grammar;
- :mod:`~finetuning_scheduler.tunable_view.errors`: the errors the protocol raises;
- :mod:`~finetuning_scheduler.tunable_view.conformance`: the conformance cases every implementation passes, whatever
  consumes it.

The package imports only ``torch`` and the standard library, and nothing else from Fine-Tuning Scheduler, so it can be
published as a distribution of its own.

.. warning:: This is an experimental API and may change in a future release.
"""
from .conformance import ConformanceCase, ConformanceError, ConformanceReport, ViewConformance
from .errors import TunableViewError, Unresolvable
from .roles import ROLE_KINDS
from .views import (as_view, BoundView, CanonicalView, resolve_view, TunableSupport, TunableView, TunableViewExtras,
                    VIEW_ATTR, ViewDefaults)

__all__ = [
    "as_view",
    "BoundView",
    "CanonicalView",
    "ConformanceCase",
    "ConformanceError",
    "ConformanceReport",
    "resolve_view",
    "ROLE_KINDS",
    "TunableSupport",
    "TunableView",
    "TunableViewError",
    "TunableViewExtras",
    "Unresolvable",
    "VIEW_ATTR",
    "ViewConformance",
    "ViewDefaults",
]
