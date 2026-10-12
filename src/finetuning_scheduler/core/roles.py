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

Lifecycle roles for the tensors a :class:`~finetuning_scheduler.core.views.TunableView` exposes, and the ``role:``
selectors a schedule uses to address them.

A role is one of:

- ``base``: the model's own parameters (the default for anything unlisted);
- ``adapter`` or ``adapter:<name>``: parameter-efficient adapters (LoRA and similar);
- ``intervention`` or ``intervention:<name>``: modules that intervene in the forward pass (steering, ReFT);
- ``latent:<name>``: learned latent dictionaries (for example spliced transcoders);
- ``fitted``: run-fitted, non-gradient state, never thawed or optimized but checkpointed;
- ``instrument``: published, re-loadable artifacts, never thawed, optimized or checkpointed.

A selector is ``role:<kind>`` (every role of that kind, named or not) or ``role:<kind>:<name>`` (exactly that role).

.. warning:: This is an experimental API and may change in a future release.
"""
import re
from collections.abc import Iterable, Mapping

from finetuning_scheduler.core.exceptions import FTSConfigurationError

# every role kind, in documentation order
ROLE_KINDS = ("base", "adapter", "intervention", "latent", "fitted", "instrument")
NAMED_OPTIONAL = frozenset({"adapter", "intervention"})
NAMED_REQUIRED = frozenset({"latent"})
# role kinds that are never thawed or placed in an optimizer
FROZEN_ROLE_KINDS = frozenset({"fitted", "instrument"})
# role kinds that are never checkpointed
UNCHECKPOINTED_ROLE_KINDS = frozenset({"instrument"})
SELECTOR_PREFIX = "role:"
_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


def parse_role(role: str) -> tuple[str, str | None]:
    """Split a role into its kind and optional name, validating both.

    Args:
        role: A role such as ``"base"``, ``"adapter:char"`` or ``"latent:transcoder"``.

    Returns:
        The role's kind and its name (``None`` when the role has none).

    Raises:
        FTSConfigurationError: If the kind is unknown, a name is missing where the kind requires one, a name is given
            where the kind takes none, or the name has characters other than letters, digits, ``_``, ``.`` and ``-``.
    """
    if not isinstance(role, str):
        raise FTSConfigurationError(f"A role must be a string, got {type(role).__name__} ({role!r}).")
    kind, sep, name = role.partition(":")
    if kind not in ROLE_KINDS:
        raise FTSConfigurationError(f"Unknown role {role!r}: the role kinds are {', '.join(ROLE_KINDS)}.")
    if not sep:
        if kind in NAMED_REQUIRED:
            raise FTSConfigurationError(f"Role {role!r} needs a name (for example `{kind}:transcoder`).")
        return kind, None
    if kind not in NAMED_OPTIONAL | NAMED_REQUIRED:
        raise FTSConfigurationError(f"Role {role!r}: the `{kind}` role takes no name.")
    if not _NAME_RE.fullmatch(name):
        raise FTSConfigurationError(
            f"Role {role!r}: a role name may contain only letters, digits, `_`, `.` and `-`.")
    return kind, name


def role_kind(role: str) -> str:
    """The kind of a (validated) role."""
    return parse_role(role)[0]


def is_frozen_role(role: str) -> bool:
    """Whether tensors with ``role`` are never thawed or placed in an optimizer (``fitted`` and ``instrument``)."""
    return role_kind(role) in FROZEN_ROLE_KINDS


def is_role_selector(entry: str) -> bool:
    """Whether a schedule ``params`` entry is a ``role:`` selector rather than a name or pattern."""
    return isinstance(entry, str) and entry.startswith(SELECTOR_PREFIX)


def parse_selector(selector: str) -> tuple[str, str | None]:
    """Parse a ``role:<kind>[:<name>]`` selector into the kind and the optional name it matches.

    Raises:
        FTSConfigurationError: If ``selector`` is not a valid selector.
    """
    if not is_role_selector(selector):
        raise FTSConfigurationError(f"{selector!r} is not a role selector (`role:<kind>` or `role:<kind>:<name>`).")
    body = selector[len(SELECTOR_PREFIX):]
    kind, sep, name = body.partition(":")
    if kind not in ROLE_KINDS:
        raise FTSConfigurationError(f"Selector {selector!r} names an unknown role kind: the role kinds are "
                                    f"{', '.join(ROLE_KINDS)}.")
    if not sep:
        return kind, None
    parse_role(body)  # validates the name against the kind
    return kind, name


def selector_matches(selector: str, role: str) -> bool:
    """Whether ``role`` matches ``selector``: ``role:<kind>`` matches every role of that kind (named or not), and
    ``role:<kind>:<name>`` matches exactly that role."""
    sel_kind, sel_name = parse_selector(selector)
    kind, name = parse_role(role)
    return kind == sel_kind and (sel_name is None or name == sel_name)


def select(selector: str, names: Iterable[str], roles: Mapping[str, str]) -> list[str]:
    """The ``names`` whose role matches ``selector``, in order. A name absent from ``roles`` has the role ``base``.

    Args:
        selector: A ``role:`` selector.
        names: Candidate tensor names.
        roles: A mapping from tensor names to roles.
    """
    parse_selector(selector)
    return [n for n in names if selector_matches(selector, roles.get(n, "base"))]


def validate_roles(roles: Mapping[str, str], parameters: Iterable[str], buffers: Iterable[str]) -> dict[str, str]:
    """Validate a role mapping against a module's parameter and buffer names.

    Args:
        roles: A mapping from parameter or buffer names to roles.
        parameters: The module's parameter names.
        buffers: The module's buffer names.

    Returns:
        A copy of ``roles``.

    Raises:
        FTSConfigurationError: If a role is invalid or a key names neither a parameter nor a buffer (every offending
            key is named).
    """
    known = set(parameters) | set(buffers)
    unknown = [k for k in roles if k not in known]
    if unknown:
        raise FTSConfigurationError(f"These role keys name neither a parameter nor a buffer of the module: {unknown}.")
    for role in roles.values():
        parse_role(role)
    return dict(roles)


def declared_kinds(roles: Mapping[str, str]) -> frozenset[str]:
    """The role kinds a role mapping uses, always including ``base``."""
    return frozenset({"base"} | {role_kind(r) for r in roles.values()})
