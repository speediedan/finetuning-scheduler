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
Fine-Tuning Scheduler Parameter Roles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The role vocabulary and ``role:`` selector grammar of the tunable view protocol
(:mod:`~venngram.roles`), as Fine-Tuning Scheduler applies them to schedules. The constants
are the protocol's; each function is the protocol's, raising
:class:`~finetuning_scheduler.core.exceptions.FTSConfigurationError` where the protocol raises
:class:`~venngram.errors.VenngramError`, because here a role or selector comes from a
schedule or a configuration.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager

from finetuning_scheduler.core.exceptions import FTSConfigurationError
import venngram.roles as _roles
from venngram.errors import VenngramError
from venngram.roles import (FROZEN_ROLE_KINDS, is_role_selector, NAMED_OPTIONAL, NAMED_REQUIRED, ROLE_KINDS,
                            SELECTOR_PREFIX, UNCHECKPOINTED_ROLE_KINDS)


@contextmanager
def _configuration_errors() -> Iterator[None]:
    try:
        yield
    except VenngramError as err:
        raise FTSConfigurationError(str(err)) from err


def parse_role(role: str) -> tuple[str, str | None]:
    """Split a role into its kind and optional name (:func:`~venngram.roles.parse_role`).

    Raises:
        FTSConfigurationError: If the role is invalid.
    """
    with _configuration_errors():
        return _roles.parse_role(role)


def role_kind(role: str) -> str:
    """The kind of a role.

    Raises:
        FTSConfigurationError: If the role is invalid.
    """
    with _configuration_errors():
        return _roles.role_kind(role)


def is_frozen_role(role: str) -> bool:
    """Whether tensors with ``role`` are never thawed or placed in an optimizer (``fitted`` and ``instrument``).

    Raises:
        FTSConfigurationError: If the role is invalid.
    """
    with _configuration_errors():
        return _roles.is_frozen_role(role)


def parse_selector(selector: str) -> tuple[str, str | None]:
    """Parse a ``role:<kind>[:<name>]`` selector (:func:`~venngram.roles.parse_selector`).

    Raises:
        FTSConfigurationError: If ``selector`` is not a valid selector.
    """
    with _configuration_errors():
        return _roles.parse_selector(selector)


def selector_matches(selector: str, role: str) -> bool:
    """Whether ``role`` matches ``selector`` (:func:`~venngram.roles.selector_matches`).

    Raises:
        FTSConfigurationError: If the selector or the role is invalid.
    """
    with _configuration_errors():
        return _roles.selector_matches(selector, role)


def select(selector: str, names: Iterable[str], roles: Mapping[str, str]) -> list[str]:
    """The ``names`` whose role matches ``selector``, in order (:func:`~venngram.roles.select`).

    Raises:
        FTSConfigurationError: If the selector or a role is invalid.
    """
    with _configuration_errors():
        return _roles.select(selector, names, roles)


def validate_roles(roles: Mapping[str, str], parameters: Iterable[str], buffers: Iterable[str]) -> dict[str, str]:
    """Validate a role mapping against a module's parameter and buffer names
    (:func:`~venngram.roles.validate_roles`).

    Raises:
        FTSConfigurationError: If a role is invalid or a key names neither a parameter nor a buffer.
    """
    with _configuration_errors():
        return _roles.validate_roles(roles, parameters, buffers)


def declared_kinds(roles: Mapping[str, str]) -> frozenset[str]:
    """The role kinds a role mapping uses, always including ``base``.

    Raises:
        FTSConfigurationError: If a role is invalid.
    """
    with _configuration_errors():
        return _roles.declared_kinds(roles)


__all__ = [
    "declared_kinds",
    "FROZEN_ROLE_KINDS",
    "is_frozen_role",
    "is_role_selector",
    "NAMED_OPTIONAL",
    "NAMED_REQUIRED",
    "parse_role",
    "parse_selector",
    "ROLE_KINDS",
    "role_kind",
    "select",
    "selector_matches",
    "SELECTOR_PREFIX",
    "UNCHECKPOINTED_ROLE_KINDS",
    "validate_roles",
]
