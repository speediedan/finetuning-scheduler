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
Fine-Tuning Scheduler Tunable Views
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tunable view protocol (:mod:`finetuning_scheduler.tunable_view`) as Fine-Tuning Scheduler consumes it. A
:class:`TunableView` is the contract between a model's *logical* parameter names (the names a schedule uses) and its
*physical* parameters (the tensors the optimizer, checkpoints and sharding see). Every name here is the protocol's
except :class:`ScheduleUnresolvable`, which Fine-Tuning Scheduler raises when a schedule names something a view does not
resolve.

A view that does not satisfy the protocol (one lacking a required member, say) raises
:class:`~venngram.tuning.errors.TunableViewError`, while a schedule or configuration that is invalid
raises :class:`~finetuning_scheduler.core.exceptions.FTSConfigurationError`.

Views are discovered, in order, from an explicit ``view`` argument, then a view attached to the model as its
``tunable_view`` attribute, then :class:`CanonicalView` (see :func:`resolve_view`).

.. warning:: This is an experimental API and may change in a future release.
"""
from finetuning_scheduler.core.exceptions import FTSConfigurationError
# imported from venngram directly rather than through the `finetuning_scheduler.tunable_view` submodule aliases, which
# exist only at runtime (`sys.modules`), so a static checker cannot resolve them
from venngram.tuning.errors import TunableViewError, Unresolvable  # noqa: F401 (re-exported)
from venngram.tuning.views import (as_view, BoundView, canonical_fqn, CanonicalView, instrument_state_keys,
                                   OPTIONAL_MEMBERS, pair_groups, REQUIRED_MEMBERS, resolve_view, TunableSupport,
                                   TunableView, TunableViewExtras, uncheckpointed_keys, VIEW_ATTR, ViewDefaults)


class ScheduleUnresolvable(Unresolvable, FTSConfigurationError):
    """Raised when a schedule's names cannot be resolved to physical parameters; names every unresolved input.

    It is both the protocol's :class:`~venngram.tuning.errors.Unresolvable` (so catching that
    catches this) and an :class:`~finetuning_scheduler.core.exceptions.FTSConfigurationError`.
    """


__all__ = [
    "as_view",
    "BoundView",
    "canonical_fqn",
    "CanonicalView",
    "instrument_state_keys",
    "OPTIONAL_MEMBERS",
    "pair_groups",
    "REQUIRED_MEMBERS",
    "resolve_view",
    "ScheduleUnresolvable",
    "TunableSupport",
    "TunableView",
    "TunableViewExtras",
    "uncheckpointed_keys",
    "VIEW_ATTR",
    "ViewDefaults",
]
