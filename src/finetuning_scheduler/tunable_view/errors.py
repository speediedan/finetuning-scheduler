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
Tunable View Errors
^^^^^^^^^^^^^^^^^^^

The errors the tunable view protocol raises. Both are ``ValueError`` subclasses. A consumer may translate them into its
own error types at its boundary (Fine-Tuning Scheduler raises its ``FTSConfigurationError`` for them).

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Sequence


class TunableViewError(ValueError):
    """Raised when a view, a role or a role selector does not satisfy the protocol."""


class Unresolvable(TunableViewError):
    """Raised when logical names cannot be resolved to physical parameters; names every unresolved input.

    Args:
        names: Every name that did not resolve.
        detail: Optional context appended to the message.
    """

    def __init__(self, names: Sequence[str], detail: str | None = None) -> None:
        self.names = list(names)
        self.detail = detail
        msg = f"These names do not resolve to any parameter: {self.names}."
        super().__init__(f"{msg} {detail}" if detail else msg)


__all__ = ["TunableViewError", "Unresolvable"]
