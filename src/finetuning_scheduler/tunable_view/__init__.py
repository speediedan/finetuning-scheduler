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

The tunable view protocol now lives in `venngram <https://github.com/speediedan/venngram>`_ (``venngram.tuning``),
beside the address grammar that schedules use, so both projects that consume it (Fine-Tuning Scheduler and
interpretune) depend on one neutral package. This package re-exports it: ``finetuning_scheduler.tunable_view`` and its
submodules ``views``, ``roles``, ``errors`` and ``conformance`` are venngram's own modules, so every name and every
object is identical.

.. warning:: This is an experimental API and may change in a future release.
"""
import sys

import venngram.roles as roles
from venngram.tuning import conformance, errors, views
from venngram.tuning import *  # noqa: F401, F403
from venngram.tuning import __all__ as _protocol_all

# the submodules are venngram's modules themselves, so `finetuning_scheduler.tunable_view.roles` and
# `venngram.roles` are one object (patching one patches the other)
for _name, _module in {"roles": roles, "views": views, "errors": errors, "conformance": conformance}.items():
    sys.modules[f"{__name__}.{_name}"] = _module

__all__ = list(_protocol_all)
