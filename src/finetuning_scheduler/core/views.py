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

A :class:`TunableView` is the contract between a model's *logical* parameter names (the names a schedule uses) and its
*physical* parameters (the tensors the optimizer, checkpoints and sharding see). It lets a schedule address a model
through names other than ``named_parameters()``, for example the names an interpretability library gives a wrapped
model, where one logical name may stand for several physical tensors.

A view needs only four members (:class:`TunableView`). The optional members (:class:`TunableViewExtras`) default to
Fine-Tuning Scheduler's implementations when a view does not define them; :class:`ViewDefaults` provides every default
for implementers who prefer to inherit. :class:`CanonicalView` is the identity view every plain model gets.

Views are discovered, in order, from an explicit ``view`` argument, then a view attached to the model as its
``fts_tunable_view`` attribute, then :class:`CanonicalView` (see :func:`resolve_view`).

.. warning:: This is an experimental API and may change in a future release.
"""
import itertools
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from torch import nn, Tensor

from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.roles import declared_kinds, role_kind, UNCHECKPOINTED_ROLE_KINDS, validate_roles

# the module attribute a view is attached under for discovery
VIEW_ATTR = "fts_tunable_view"
REQUIRED_MEMBERS = ("root", "support", "to_physical", "to_logical")
OPTIONAL_MEMBERS = ("logical_named_parameters", "coupled", "depth_groups", "roles", "checkpoint_state_dict",
                    "load_checkpoint_state_dict", "promote")


@dataclass(frozen=True)
class TunableSupport:
    """What a view declares about its model.

    Attributes:
        trainable: Whether the model can be fine-tuned through this view. When ``False``, Fine-Tuning Scheduler refuses
            the model, naming ``reason``.
        reason: Why the model cannot be fine-tuned (required when ``trainable`` is ``False``).
        naming_scheme: The scheme the view's logical names follow. ``"physical"`` means logical names are the root's
            own parameter names; any other value (for example ``"tl"``) requires the view to define
            ``logical_named_parameters``.
        logical_names_are_views: Whether logical names stand for reshaped or sliced views of physical tensors.
        declared_roles: The role kinds the view's tensors use (see :mod:`~finetuning_scheduler.core.roles`).
    """

    trainable: bool
    reason: str | None = None
    naming_scheme: str = "physical"
    logical_names_are_views: bool = False
    declared_roles: frozenset[str] = field(default_factory=lambda: frozenset({"base"}))

    def __post_init__(self) -> None:
        if not self.trainable and not self.reason:
            raise FTSConfigurationError("A `TunableSupport` with `trainable=False` must give a `reason`.")


class Unresolvable(FTSConfigurationError):
    """Raised when logical names cannot be resolved to physical parameters; names every unresolved input."""

    def __init__(self, names: Sequence[str], detail: str | None = None) -> None:
        self.names = list(names)
        msg = f"These names do not resolve to any parameter: {self.names}."
        super().__init__(f"{msg} {detail}" if detail else msg)


@runtime_checkable
class TunableView(Protocol):
    """The required members of a tunable view.

    Any object with these members is a view; inheriting from a Fine-Tuning Scheduler class is not required.
    """

    @property
    def root(self) -> nn.Module:
        """The module whose ``named_parameters()`` are the physical names the optimizer and checkpoints use."""
        ...

    def support(self) -> TunableSupport:
        """The view's support record.

        It is read once, at setup.
        """
        ...

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        """Map each logical name to the physical parameter names it stands for (one or more).

        Raises:
            Unresolvable: If any name does not resolve, naming every such name.
        """
        ...

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        """Map physical parameter names to logical names, without duplicates and in order; physical names without a
        logical name pass through unchanged."""
        ...


@runtime_checkable
class TunableViewExtras(Protocol):
    """The optional members of a tunable view.

    Fine-Tuning Scheduler supplies a default for each one a view lacks.
    """

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """Logical names with a representative tensor, in registration order (for schedule generation and
        display)."""
        ...

    def coupled(self, physical: Sequence[str]) -> list[str]:
        """Physical names thawed implicitly with ``physical`` (for example a block's norms)."""
        ...

    def depth_groups(self) -> list[list[str]]:
        """Logical names grouped into phases for a generated schedule, the first group thawed first."""
        ...

    def roles(self) -> Mapping[str, str]:
        """Physical parameter or buffer names mapped to roles; anything unlisted is ``base``."""
        ...

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """The state to checkpoint."""
        ...

    def load_checkpoint_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        """Load state produced by :meth:`checkpoint_state_dict`."""
        ...

    def promote(self, physical: Sequence[str], role: str) -> Mapping[str, str]:
        """Promote ``physical`` tensors to ``role``, returning each promoted tensor's old name mapped to its new
        name."""
        ...


def uncheckpointed_keys(roles: Mapping[str, str]) -> set[str]:
    """The tensor names whose role is never checkpointed (``instrument``)."""
    return {k for k, r in roles.items() if role_kind(r) in UNCHECKPOINTED_ROLE_KINDS}


# module-name segments that wrappers insert and that state dicts (DSD, activation checkpointing) leave out
_WRAPPER_SEGMENTS = ("_orig_mod.", "_checkpoint_wrapped_module.", "_fsdp_wrapped_module.")


def canonical_fqn(name: str) -> str:
    """``name`` without wrapper segments (``torch.compile``'s ``_orig_mod.``, activation checkpointing's).

    >>> canonical_fqn("_orig_mod.block._checkpoint_wrapped_module.weight")
    'block.weight'
    """
    for seg in _WRAPPER_SEGMENTS:
        name = name.replace(seg, "")
    return name


def instrument_state_keys(root: nn.Module, roles: Mapping[str, str], keys: Collection[str]) -> set[str]:
    """The state-dict ``keys`` that hold ``instrument`` tensors, matched by tensor identity (so every alias of a
    tied tensor counts) and by wrapper-free name (so keys from modules wrapped by ``torch.compile`` or by
    activation checkpointing match)."""
    instrument = uncheckpointed_keys(roles)
    if not instrument:
        return set()
    named = list(itertools.chain(root.named_parameters(remove_duplicate=False),
                                 root.named_buffers(remove_duplicate=False)))
    aliases: dict[int, list[str]] = {}
    for n, t in named:
        aliases.setdefault(id(t), []).append(n)
    ids = {id(t) for n, t in named if n in instrument}
    names = set(instrument) | {a for i in ids for a in aliases[i]}
    canonical = {canonical_fqn(n) for n in names}
    return {k for k in keys if canonical_fqn(k) in canonical}


def pair_groups(names: Sequence[str]) -> list[list[str]]:
    """Group ``names`` in pairs starting from the last, the order Fine-Tuning Scheduler's generated schedules use.

    >>> pair_groups(["a", "b", "c"])
    [['c', 'b'], ['a']]
    """
    reversed_names = list(names)[::-1]
    return [reversed_names[i:i + 2] for i in range(0, len(reversed_names), 2)]


class ViewDefaults:
    """Every optional view member's default, for views that prefer to inherit them.

    Subclasses provide :attr:`root`, :meth:`support`, :meth:`to_physical` and :meth:`to_logical`.
    """

    root: nn.Module

    def support(self) -> TunableSupport:
        raise NotImplementedError

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        raise NotImplementedError

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        raise NotImplementedError

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """The root's ``named_parameters()`` when the naming scheme is ``"physical"``.

        Raises:
            FTSConfigurationError: For any other naming scheme, which must define this member.
        """
        scheme = self.support().naming_scheme
        if scheme != "physical":
            raise FTSConfigurationError(
                f"A view whose naming scheme is {scheme!r} must define `logical_named_parameters`; the default serves"
                " only the 'physical' scheme."
            )
        return self.root.named_parameters()

    def coupled(self, physical: Sequence[str]) -> list[str]:
        """No implicit co-thaws."""
        return []

    def depth_groups(self) -> list[list[str]]:
        """Logical names in pairs, starting from the last registered one."""
        return pair_groups([n for n, _ in self.logical_named_parameters()])

    def roles(self) -> Mapping[str, str]:
        """Everything is ``base``."""
        return {}

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """The root's ``state_dict()``, without ``instrument`` tensors (which are never checkpointed)."""
        state = self.root.state_dict()
        instrument = instrument_state_keys(self.root, self.roles(), state.keys())
        return {k: v for k, v in state.items() if k not in instrument}

    def load_checkpoint_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        """Load into the root. A strict load is strict by role: the keys missing from ``state_dict`` must be
        exactly the ``instrument`` keys, and it may hold no key the root lacks. It is checked before anything is
        loaded.

        Raises:
            RuntimeError: If a strict load's keys do not match, naming the keys (as ``load_state_dict`` does).
        """
        expected = set(self.root.state_dict())
        instrument = instrument_state_keys(self.root, self.roles(), expected)
        if not strict or not instrument:
            self.root.load_state_dict(state_dict, strict=strict)
            return
        missing = sorted(expected - set(state_dict) - instrument)
        unexpected = sorted(set(state_dict) - expected)
        if missing or unexpected:
            raise RuntimeError(f"Error(s) in loading state_dict for {type(self.root).__name__}: missing keys"
                               f" {missing}, unexpected keys {unexpected}.")
        self.root.load_state_dict(state_dict, strict=False)

    def promote(self, physical: Sequence[str], role: str) -> Mapping[str, str]:
        """Keep every name.

        A promoted parameter that shares its storage with another tensor (a tied weight, for example) is replaced by a
        frozen copy of itself, so training it leaves the other tensor unchanged; otherwise only the role changes.
        """
        named = list(itertools.chain(self.root.named_parameters(remove_duplicate=False),
                                     self.root.named_buffers(remove_duplicate=False)))
        holders: dict[int, set[str]] = {}
        for name, t in named:
            if t.device.type != "meta":
                holders.setdefault(t.untyped_storage().data_ptr(), set()).add(name)
        params = dict(self.root.named_parameters(remove_duplicate=False))
        for name in physical:
            t = params[name]
            if t.device.type != "meta" and holders.get(t.untyped_storage().data_ptr(), set()) - {name}:
                owner, _, attr = name.rpartition(".")
                setattr(self.root.get_submodule(owner), attr, nn.Parameter(t.detach().clone(), requires_grad=False))
        return {n: n for n in physical}


class BoundView:
    """A view with every optional member filled in: the view's own member when it defines one, Fine-Tuning
    Scheduler's default otherwise. Its support record is read once and cached.

    Args:
        view: The view to bind.
    """

    def __init__(self, view: Any) -> None:
        missing = [m for m in REQUIRED_MEMBERS if not hasattr(view, m)]
        if missing:
            raise FTSConfigurationError(
                f"{type(view).__name__} is not a tunable view: it lacks {', '.join(f'`{m}`' for m in missing)}."
            )
        self.view = view
        self._support: TunableSupport | None = None
        self._promoted: dict[str, str] = {}

    @property
    def root(self) -> nn.Module:
        return self.view.root

    def support(self) -> TunableSupport:
        if self._support is None:
            support = self.view.support()
            if not isinstance(support, TunableSupport):
                raise FTSConfigurationError(f"{type(self.view).__name__}.support() must return a `TunableSupport`, got"
                                            f" {type(support).__name__}.")
            self._support = support
        return self._support

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        return self.view.to_physical(logical)

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        return self.view.to_logical(physical)

    def _member(self, name: str) -> Any:
        # the view's own member, or else the default bound to this view, so the defaults see the promoted roles
        fn = getattr(self.view, name, None)
        default = getattr(ViewDefaults, name)
        if callable(fn) and getattr(fn, "__func__", None) is not default:
            return fn
        return default.__get__(self)

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        return self._member("logical_named_parameters")()

    def coupled(self, physical: Sequence[str]) -> list[str]:
        return list(self._member("coupled")(physical))

    def depth_groups(self) -> list[list[str]]:
        return [list(g) for g in self._member("depth_groups")()]

    def roles(self) -> Mapping[str, str]:
        """The view's roles, with the roles of tensors promoted since (see :meth:`promote`)."""
        return {**self.view_roles(), **self._promoted}

    def view_roles(self) -> dict[str, str]:
        """The roles the view itself reports, without promotions."""
        return dict(self._member("roles")())

    @property
    def promoted(self) -> dict[str, str]:
        """The tensors promoted through :meth:`promote`, mapped to their new roles."""
        return dict(self._promoted)

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return self._member("checkpoint_state_dict")()

    def load_checkpoint_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        self._member("load_checkpoint_state_dict")(state_dict, strict=strict)

    def promote(self, physical: Sequence[str], role: str) -> Mapping[str, str]:
        """Promote ``physical`` to ``role`` through the view (or the default), and record the new role.

        Raises:
            FTSConfigurationError: If the view renamed a promoted tensor (a promotion keeps names in this release).
        """
        mapping = dict(self._member("promote")(physical, role))
        renamed = {old: new for old, new in mapping.items() if old != new}
        if renamed:
            raise FTSConfigurationError(
                f"{type(self.view).__name__}.promote renamed the tensors it promoted ({renamed}); a promotion must keep"
                " each tensor's name in this release.")
        self._promoted.update(dict.fromkeys(physical, role))
        return mapping

    def defines(self, member: str) -> bool:
        """Whether the bound view has ``member`` (its own or inherited from :class:`ViewDefaults`)."""
        return callable(getattr(self.view, member, None))

    def overrides(self, member: str) -> bool:
        """Whether the bound view implements ``member`` differently from Fine-Tuning Scheduler's default."""
        impl = getattr(type(self.view), member, None)
        return callable(impl) and impl is not getattr(ViewDefaults, member)


def as_view(view: Any) -> BoundView:
    """Bind ``view`` (returning it unchanged if it is already bound)."""
    return view if isinstance(view, BoundView) else BoundView(view)


def resolve_view(model: nn.Module, view: Any | None = None) -> BoundView:
    """Find the view for ``model``: ``view`` when given, else a view attached as ``model.fts_tunable_view``, else
    :class:`CanonicalView` over ``model``.

    Raises:
        FTSConfigurationError: If the chosen object is not a tunable view.
    """
    if view is None:
        view = getattr(model, VIEW_ATTR, None)
    if view is None:
        view = CanonicalView(model)
    return as_view(view)


class CanonicalView(ViewDefaults):
    """The identity view over ``root.named_parameters()``.

    A parametrized tensor (``torch.nn.utils.parametrize``) is also reachable by its unparametrized name:
    ``layer.weight`` resolves to ``layer.parametrizations.weight.original`` (or to every ``original<i>`` of a
    multi-tensor parametrization).

    Args:
        root: The module to view.
        roles: Optional roles for the root's parameters and buffers (anything unlisted is ``base``).

    Raises:
        FTSConfigurationError: If a role is invalid or a role key names neither a parameter nor a buffer.
    """

    def __init__(self, root: nn.Module, roles: Mapping[str, str] | None = None) -> None:
        self.root = root
        self._roles = validate_roles(roles or {}, (n for n, _ in root.named_parameters()),
                                     (n for n, _ in root.named_buffers()))

    def support(self) -> TunableSupport:
        return TunableSupport(trainable=True, naming_scheme="physical", declared_roles=declared_kinds(self._roles))

    def _parametrized(self) -> dict[str, tuple[str, ...]]:
        """Unparametrized names of parametrized tensors mapped to their underlying physical names."""
        out: dict[str, list[str]] = {}
        for name, _ in self.root.named_parameters():
            if name.startswith("parametrizations."):  # a parametrization on the root module itself
                head, tail = "", name[len("parametrizations."):]
            else:
                head, sep, tail = name.partition(".parametrizations.")
                if not sep:
                    continue
            tensor_name, _, original = tail.partition(".")
            if original.startswith("original"):
                logical = f"{head}.{tensor_name}" if head else tensor_name
                out.setdefault(logical, []).append(name)
        return {k: tuple(v) for k, v in out.items()}

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        physical = {n for n, _ in self.root.named_parameters()}
        parametrized = self._parametrized()
        resolved: dict[str, tuple[str, ...]] = {}
        unresolved = []
        for name in logical:
            if name in physical:
                resolved[name] = (name,)
            elif name in parametrized:
                resolved[name] = parametrized[name]
            else:
                unresolved.append(name)
        if unresolved:
            raise Unresolvable(unresolved)
        return resolved

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(physical))

    def roles(self) -> Mapping[str, str]:
        return dict(self._roles)
