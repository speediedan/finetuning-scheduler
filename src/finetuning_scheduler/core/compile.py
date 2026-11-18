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
Fine-Tuning Scheduler Schedule Compilation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compile a schedule's logical names, once, into the physical parameters each phase thaws.

:func:`compile_schedule` resolves each phase's entries through a :class:`~finetuning_scheduler.core.views.TunableView`:
addresses (`venngram <https://github.com/speediedan/venngram>`_ addresses, ``role:`` selectors among them) select
physical tensors through the view's roles, its structure and the schedule's named sets; names and regexes resolve
against the view's logical names and then map to physical names; and the view's coupled tensors (for example a block's
norms) join the first phase that pulls them. An address of gradients (``of: grad``) thaws what it selects and compiles
to a gradient mask over each tensor's first axis (:attr:`CompiledPhase.masks`). It refuses a schedule whose phases share
storage, that schedules a ``fitted`` or ``artifact`` tensor, or that names anything the view cannot resolve, naming
the offending input. The schedule itself keeps its logical names.

.. warning:: This is an experimental API and may change in a future release.
"""
import dataclasses
import functools
import hashlib
import itertools
import json
import pathlib
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from torch import Tensor

from finetuning_scheduler.core.dist import rank_zero_debug, rank_zero_info, rank_zero_warn
from finetuning_scheduler.core.exceptions import FTSConfigurationError
import venngram
from venngram import is_address, ParamSpace, select
from venngram.address import parse as parse_address
from venngram.address import references
from venngram.errors import VenngramError
from venngram.resolve import gradient_masks, split_of
from venngram.support import SupportRecord, TuningSupport

from finetuning_scheduler.core.grad_masks import Transform
from finetuning_scheduler.core.grad_transforms import transform_id
from finetuning_scheduler.core.roles import is_frozen_role, role_kind
from finetuning_scheduler.core.schedule import (parse_promote, requires_grad_capable, resolve_param_entries,
                                                 shared_params_ref)
from finetuning_scheduler.core.views import as_view, BoundView, ScheduleUnresolvable, Unresolvable


@dataclass(frozen=True)
class CompiledPhase:
    """One phase's compiled parameters.

    Attributes:
        depth: The phase.
        logical: The phase's resolved logical names (role selectors are expanded into ``physical`` directly).
        physical: The physical parameters the phase names, through the view or a role selector.
        coupled: The physical parameters thawed with them that no earlier phase claimed.
        refrozen: The physical parameters the phase refreezes when it starts, with the view's coupled tensors among
            those still trainable.
        masks: The gradient masks the phase sets (its ``of: grad`` entries): each tensor's kept indices along its first
            axis (a weight's output rows), or ``None`` for a tensor an earlier phase masked that this phase trains
            whole. A tensor whose every index is kept has no mask. A masked tensor an earlier phase thawed and this
            phase selects again widens its mask: it is listed here, not in ``physical``, and stays in the earlier
            phase's parameter groups.
        transforms: The gradient transforms the phase sets (its ``grad_transforms``): each tensor's transforms, in the
            order they apply, with the indices along its first axis its entry's unit set keeps (``None`` without one).
            ``None`` when the phase states none, so the transforms in effect before it carry on.
    """

    depth: int
    logical: tuple[str, ...]
    physical: tuple[str, ...]
    coupled: tuple[str, ...] = ()
    refrozen: tuple[str, ...] = ()
    masks: Mapping[str, tuple[int, ...] | None] = field(default_factory=dict)
    transforms: Mapping[str, tuple[Transform, ...]] | None = None

    @property
    def thawed(self) -> tuple[str, ...]:
        """Every physical parameter the phase thaws: its own, then its coupled ones."""
        return self.physical + self.coupled


@dataclass(frozen=True)
class CompiledSchedule:
    """A schedule compiled through a view.

    Attributes:
        phases: The compiled phases, by depth.
        roles: The view's role map (physical parameter or buffer names to roles; unlisted names are ``base``).
        naming_scheme: The view's naming scheme.
        digest: A hash of every phase's physical parameters and of the roles, identifying what the schedule thaws
            independently of how it names it.
        overlaps: Pairs of physical names within one phase that share storage (reported, not refused).
        logical_to_physical: Each compiled logical name mapped to its physical parameters.
        promotions: Per phase, the physical tensors it promotes when it starts, mapped to their new roles.
    """

    phases: tuple[CompiledPhase, ...]
    roles: Mapping[str, str]
    naming_scheme: str
    digest: str
    overlaps: tuple[tuple[str, str], ...] = ()
    logical_to_physical: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    promotions: Mapping[int, Mapping[str, str]] = field(default_factory=dict)

    def roles_at(self, depth: int) -> dict[str, str]:
        """The roles in effect once phase ``depth`` has started: the view's roles, then every promotion through
        ``depth``."""
        roles = dict(self.roles)
        for d in sorted(self.promotions):
            if d <= depth:
                roles.update(self.promotions[d])
        return roles

    def physical_for(self, depth: int) -> tuple[str, ...]:
        """The physical parameters phase ``depth`` thaws (its own, then its coupled ones)."""
        for cp in self.phases:
            if cp.depth == depth:
                return cp.thawed
        raise KeyError(f"The compiled schedule has no phase {depth}.")

    def trainable_at(self, depth: int) -> tuple[str, ...]:
        """The physical parameters trainable once phase ``depth`` is in place: the phases' thaws and ``refreeze``
        entries through ``depth``, applied in order (a tensor thawed again after a refreeze is trainable)."""
        if not any(cp.depth == depth for cp in self.phases):
            raise KeyError(f"The compiled schedule has no phase {depth}.")
        trainable: dict[str, None] = {}
        for cp in (cp for cp in self.phases if cp.depth <= depth):
            trainable.update(dict.fromkeys(cp.thawed))
            for n in cp.refrozen:
                trainable.pop(n, None)
        return tuple(trainable)

    def masks_at(self, depth: int) -> dict[str, tuple[int, ...]]:
        """The gradient masks in effect once phase ``depth`` is in place: each masked tensor's kept first-axis
        indices, after the phases' masks, widenings and ``refreeze`` entries through ``depth``, applied in order (a
        refrozen tensor loses its mask)."""
        if not any(cp.depth == depth for cp in self.phases):
            raise KeyError(f"The compiled schedule has no phase {depth}.")
        masks: dict[str, tuple[int, ...]] = {}
        for cp in (cp for cp in self.phases if cp.depth <= depth):
            for n, kept in cp.masks.items():
                if kept is None:
                    masks.pop(n, None)
                else:
                    masks[n] = kept
            for n in cp.refrozen:
                masks.pop(n, None)
        return masks

    def transforms_at(self, depth: int) -> dict[str, tuple[Transform, ...]]:
        """The gradient transforms in effect once phase ``depth`` is in place: those of the last phase through
        ``depth`` that states ``grad_transforms``, less the tensors refrozen since (a refrozen tensor loses its
        transforms, and a later thaw does not restore them)."""
        if not any(cp.depth == depth for cp in self.phases):
            raise KeyError(f"The compiled schedule has no phase {depth}.")
        transforms: dict[str, tuple[Transform, ...]] = {}
        for cp in (cp for cp in self.phases if cp.depth <= depth):
            if cp.transforms is not None:
                transforms = dict(cp.transforms)
            for n in cp.refrozen:
                transforms.pop(n, None)
        return transforms

    def physical_of(self, logical: Sequence[str]) -> list[str]:
        """The physical parameters compiled logical names stand for, in order and without duplicates."""
        unknown = [n for n in logical if n not in self.logical_to_physical]
        if unknown:
            raise ScheduleUnresolvable(unknown, "They are not names this compiled schedule resolved.")
        return list(dict.fromkeys(p for n in logical for p in self.logical_to_physical[n]))

    def module_map(self) -> dict[int, set[str]]:
        """The modules owning each phase's thawed parameters (a parametrized tensor's owner is the parametrized
        module)."""
        return {cp.depth: {owner_module(n) for n in cp.thawed} for cp in self.phases}


def owner_module(physical: str) -> str:
    """The name of the module owning parameter ``physical``.

    >>> owner_module("encoder.0.weight"), owner_module("0.parametrizations.weight.original"), owner_module("bias")
    ('encoder.0', '0', '')
    """
    if physical.startswith("parametrizations."):  # a parametrization on the root module itself
        return ""
    head, sep, _ = physical.partition(".parametrizations.")
    if sep:
        return head
    return physical.rpartition(".")[0]


def _storage_span(t: Tensor) -> tuple[tuple[str, int], int, int] | None:
    """A tensor's storage key and byte range, or ``None`` when its storage cannot be compared the same way on every
    rank (a ``DTensor``, whose local shard differs by rank), when it is empty, or when it has none (meta
    tensors)."""
    if hasattr(t, "to_local") or t.device.type == "meta" or t.numel() == 0:
        return None
    storage = t.untyped_storage()
    start = int(t.storage_offset()) * t.element_size()
    return (str(t.device), storage.data_ptr()), start, start + int(t.nbytes)


def _overlapping(entries: Sequence[tuple[str, int]],
                 tensors: Mapping[str, Tensor]) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """Pairs of ``(name, depth)`` entries whose tensors' storage byte ranges overlap (two ranges on one storage
    overlap when each starts before the other ends)."""
    spans: dict[tuple[str, int], list[tuple[int, int, tuple[str, int]]]] = {}
    for entry in entries:
        span = _storage_span(tensors[entry[0]])
        if span is not None:
            key, start, end = span
            spans.setdefault(key, []).append((start, end, entry))
    pairs = []
    for group in spans.values():
        group.sort(key=lambda g: (g[0], g[1]))
        for i, (s_a, e_a, a) in enumerate(group):
            for s_b, e_b, b in group[i + 1:]:
                if s_b >= e_a:
                    break
                if s_a < e_b:
                    pairs.append((a, b))
    return pairs


def resolves_exactly(bound: BoundView, entry: str, physical: Collection[str]) -> bool:
    """Whether the view resolves ``entry``, a name it does not list, to parameters of its root (for example a
    parametrized tensor's unparametrized name).

    ``entry`` may be a regex, which a view need not expect: any failure to map it (a ``KeyError`` from a view that
    indexes a mapping, for example) means it does not resolve exactly, and it is matched as a regex instead.
    """
    try:
        mapped = bound.to_physical([entry]).get(entry)
    except Exception:
        return False
    return bool(mapped) and all(p in physical for p in mapped)


def _unmapped(bound: BoundView, logical: Sequence[str]) -> list[str]:
    """The names ``to_physical`` fails on one at a time (all of ``logical`` when none fails alone)."""
    failed = []
    for name in logical:
        try:
            bound.to_physical([name])
        except Exception:
            failed.append(name)
    return failed or list(logical)


@dataclass
class _Addressing:
    """What addresses resolve against: the view's physical tensors (known by their logical names too), its structure,
    and the schedule's named sets. The roles change as phases promote tensors, so the space is built per call."""

    bound: BoundView
    physical: Sequence[str]
    sets: Mapping[str, Any]
    widths: Mapping[str, int] = field(default_factory=dict)  # each tensor's first-axis length, for gradient masks
    _aliases: dict[str, tuple[str, ...]] | None = None
    _structure: Any = None

    def space(self, roles: Mapping[str, str]) -> ParamSpace:
        if self._aliases is None:
            self._aliases = {}
            if self.bound.support().naming_scheme != "physical":
                names = [n for n, _ in self.bound.logical_named_parameters()]
                for name, physical in self.bound.to_physical(names).items():
                    for p in physical:
                        self._aliases[p] = self._aliases.get(p, ()) + (name,)
            self._structure = self.bound.structure()
        return ParamSpace(self.physical, roles=roles, aliases=self._aliases, structure=self._structure)

    def select(self, entry: Any, depth: int, roles: Mapping[str, str]) -> list[str]:
        try:
            return select(entry, self.space(roles), self.sets)
        except VenngramError as err:
            raise _unresolved(entry, depth, err) from err

    def split(self, entry: Any, depth: int) -> tuple[Any, Any]:
        """``entry`` as its value part and its gradient part (``of: grad``), either ``None`` when it has none: a
        set may unite the two (venngram's :func:`~venngram.resolve.split_of`), and a name or regex is all value."""
        if not is_address(entry):
            return entry, None
        try:
            return split_of(entry, self.sets)
        except VenngramError as err:
            raise _unresolved(entry, depth, err) from err

    def of_grad(self, entry: Any, depth: int) -> bool:
        """Whether ``entry`` addresses gradients (``of: grad``, written in it or in a set it references), in whole
        or in part."""
        return self.split(entry, depth)[1] is not None

    def masks(self, entry: Any, grad: Any, depth: int, roles: Mapping[str, str]) -> dict[str, tuple[int, ...]]:
        """The kept first-axis indices of each tensor ``entry``'s gradient part ``grad`` selects (a union's
        disjuncts each give their own mask, united per tensor)."""
        try:
            return gradient_masks(grad, self.space(roles), self.sets, widths=self.widths)
        except VenngramError as err:
            raise _unresolved(entry, depth, err) from err


def _unresolved(entry: Any, depth: int, err: VenngramError) -> FTSConfigurationError:
    detail = str(err).partition(" does not resolve: ")[2] or str(err)
    return FTSConfigurationError(f"The address '{entry}' in phase {depth} does not resolve: {detail}")


def _phase_masks(entries: Sequence[Any], resolved: Sequence[tuple[bool, list[str]]],
                 mapped: Mapping[str, Sequence[str]], depth: int, roles: Mapping[str, str],
                 addressing: _Addressing) -> dict[str, tuple[int, ...]]:
    """Phase ``depth``'s gradient masks: for each tensor the gradient parts (``of: grad``) of its entries select,
    the union of their kept indices, unless a value part selects the whole tensor or the union keeps every index
    (no mask)."""
    kept: dict[str, set[int]] = {}
    whole: set[str] = set()
    for entry, (is_sel, names) in zip(entries, resolved):
        value, grad = addressing.split(entry, depth)
        if grad is None:
            whole.update(names if is_sel else (q for n in names for q in mapped[n]))
            continue
        for p, rows in addressing.masks(entry, grad, depth, roles).items():
            kept.setdefault(p, set()).update(rows)
        if value is not None:  # a set uniting a value part and a gradient part
            whole.update(addressing.select(value, depth, roles))
    return {p: tuple(sorted(rows)) for p, rows in kept.items()
            if p not in whole and len(rows) < addressing.widths[p]}


def _refuse_gradient_entries(entries: Sequence[Any], depth: int, key: str, addressing: _Addressing) -> None:
    """``refreeze`` and ``promote`` act on whole tensors, so an ``of: grad`` address there is refused by name."""
    grads = [e for e in entries if addressing.of_grad(e, depth)]
    if grads:
        raise FTSConfigurationError(
            f"Phase {depth}'s `{key}` addresses gradients (`of: grad`): {grads}. `{key}` acts on whole tensors, so"
            " address them without `of: grad`; a gradient mask belongs in a phase's `params`.")


def _expand(entries: Sequence[Any], depth: int, bound: BoundView, logical_names: Sequence[str],
            roles: Mapping[str, str], shared: tuple, addressing: _Addressing) -> list[tuple[bool, list[str]]]:
    """Resolve a phase's entries in order: each becomes ``(True, physical names)`` for an address (a ``role:``
    selector is one) or ``(False, logical names)`` for a name or regex.

    A name the view lists resolves to itself; a name the view does not list but resolves exactly is kept as is; anything
    else is a regex over the listed names.
    """
    listed = set(logical_names)
    all_physical = addressing.physical
    resolved: list[tuple[bool, list[str]]] = []
    for entry in entries:
        if is_address(entry):
            resolved.append((True, addressing.select(entry, depth, roles)))
        elif entry not in listed and resolves_exactly(bound, entry, all_physical):
            resolved.append((False, [entry]))
        else:
            resolved.append((False, resolve_param_entries([entry], depth, logical_names, shared)))
    return resolved


def _check_phase_tensors(cp: CompiledPhase, roles: Mapping[str, str], tensors: Mapping[str, Tensor],
                         views_of_logical: bool, legacy: bool) -> None:
    frozen = [n for n in cp.physical if is_frozen_role(roles.get(n, "base"))]
    leaked = [n for n in cp.coupled if is_frozen_role(roles.get(n, "base"))]
    if frozen:
        raise FTSConfigurationError(
            f"Phase {cp.depth} schedules tensors whose role is never thawed (`fitted` or `artifact`): {frozen}.")
    if leaked:
        msg = f"Phase {cp.depth}'s coupled tensors include tensors whose role is never thawed: {leaked}."
        if not legacy:
            raise FTSConfigurationError(msg)
        rank_zero_warn(msg)
    if views_of_logical:
        packed = [n for n in cp.thawed if tensors[n].is_quantized or not tensors[n].is_floating_point()]
        if packed:
            raise FTSConfigurationError(
                f"Phase {cp.depth} schedules quantized or packed tensors through logical names that are views of them,"
                f" which cannot be thawed safely: {packed}.")
    # a physical name thaws its own tensor: refuse only storage torch cannot make require gradients, which would
    # otherwise fail when the phase starts, possibly after earlier phases have trained
    untrainable = {n: str(tensors[n].dtype).removeprefix("torch.") for n in cp.thawed
                   if not requires_grad_capable(tensors[n])}
    if untrainable:
        shown = ", ".join(f"{n} ({dtype})" for n, dtype in untrainable.items())
        raise FTSConfigurationError(
            f"Phase {cp.depth} schedules tensors whose storage cannot require gradients (quantized or packed): {shown}."
            " A frozen quantized base under trainable adapters (QLoRA, for example) is supported: schedule the"
            " adapters, or other floating-point tensors, rather than the packed ones.")


def _repeats(lists: Sequence[Sequence[Any]]) -> list[Any]:
    """Each name once per extra occurrence across ``lists``, as the Lightning callback lists them."""
    counts = Counter(itertools.chain(*lists))
    counts.subtract(Counter(set().union(*map(set, lists))))
    return list(counts.elements())


def _check_disjoint_names(within: Sequence[Sequence[str]], across: Sequence[Sequence[tuple[str, int]]],
                          legacy: bool) -> None:
    """Refuse (or, with ``legacy``, warn about) a name resolved twice within a phase or a physical parameter
    scheduled by two phases with no ``refreeze`` of it between them, listing every repeat in the Lightning
    callback's format.

    ``across`` holds each phase's physical parameters as ``(name, refreezes so far)``, so a thaw after a refreeze is
    not a repeat. Names are compared rather than storage, so the check gives the same answer on every rank.
    """
    dup_params = [n for phase in within for n in _repeats([phase])] + [n for n, _ in _repeats(across)]
    if dup_params:
        msg = ("Phases are not disjoint. The following parameters are specified in multiple phases: "
               f"{', '.join(dup_params)}")
        if not legacy:
            raise FTSConfigurationError(msg)
        rank_zero_warn(msg)


def _digest(phases: Sequence[CompiledPhase], roles: Mapping[str, str],
            promotions: Mapping[int, Mapping[str, str]]) -> str:
    payload: dict[str, Any] = {"phases": [[cp.depth, list(cp.physical), list(cp.coupled)] for cp in phases],
                               "roles": sorted(roles.items())}
    if promotions:  # only when present, so a schedule without promotions keeps its digest
        payload["promotions"] = [[d, sorted(p.items())] for d, p in sorted(promotions.items())]
    if any(cp.refrozen for cp in phases):  # likewise for a schedule that refreezes nothing
        payload["refrozen"] = [[cp.depth, list(cp.refrozen)] for cp in phases if cp.refrozen]
    if any(cp.masks for cp in phases):  # and for one that masks no gradients
        payload["masks"] = [[cp.depth, [[n, None if m is None else list(m)] for n, m in sorted(cp.masks.items())]]
                            for cp in phases if cp.masks]
    if any(cp.transforms is not None for cp in phases):  # and for one that transforms no gradients
        payload["transforms"] = [[cp.depth, [[n, [[transform_id(fn), None if u is None else list(u)] for fn, u in ts]]
                                             for n, ts in sorted(cp.transforms.items())]]
                                 for cp in phases if cp.transforms is not None]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _compile_promotions(depth: int, promote: Mapping[str, str], bound: BoundView, logical_names: Sequence[str],
                        roles: Mapping[str, str], shared: tuple, addressing: _Addressing) -> dict[str, str]:
    """Resolve phase ``depth``'s ``promote`` mapping to physical tensors and their new roles, checking each is an
    ``artifact`` tensor promoted once."""
    parse_promote(depth, promote)
    _refuse_gradient_entries(list(promote), depth, "promote", addressing)
    promoted: dict[str, str] = {}
    for source, target in promote.items():
        ((is_sel, names),) = _expand([source], depth, bound, logical_names, roles, shared, addressing)
        physical = list(dict.fromkeys(names if is_sel else (q for qs in bound.to_physical(names).values() for q in qs)))
        not_artifact = {p: roles.get(p, "base") for p in physical if role_kind(roles.get(p, "base")) != "artifact"}
        if not_artifact:
            raise FTSConfigurationError(
                f"Phase {depth} promotes tensors whose role is not `artifact` (an earlier phase may already have"
                f" promoted them): {dict(sorted(not_artifact.items()))}. Only `artifact` tensors can be promoted,"
                " to a `latent:<name>` role.")
        twice = sorted(p for p in physical if p in promoted)
        if twice:
            raise FTSConfigurationError(f"Phase {depth}'s `promote` names these tensors more than once: {twice}.")
        promoted.update(dict.fromkeys(physical, target))
    return promoted


def _compile_refreeze(depth: int, entries: Sequence[str], bound: BoundView, logical_names: Sequence[str],
                      roles: Mapping[str, str], shared: tuple, addressing: _Addressing, live: Mapping[str, int],
                      refrozen: Mapping[str, int]) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    """Resolve phase ``depth``'s ``refreeze`` entries as ``params`` entries resolve, to the physical tensors it
    refreezes (with the view's coupled tensors that are still trainable) and each logical name's physical tensors,
    checking each is trainable: ``live`` maps the tensors trainable so far to the phase that thawed them, and
    ``refrozen`` those refrozen since to the phase that refroze them."""
    _refuse_gradient_entries(entries, depth, "refreeze", addressing)
    resolved = _expand(entries, depth, bound, logical_names, roles, shared, addressing)
    logical = list(dict.fromkeys(n for is_sel, names in resolved if not is_sel for n in names))
    try:
        mapped = dict(bound.to_physical(logical)) if logical else {}
    except Unresolvable as err:
        raise ScheduleUnresolvable(err.names, f"(phase {depth}'s `refreeze`)") from err
    named = list(dict.fromkeys(p for is_sel, names in resolved
                               for p in (names if is_sel else (q for n in names for q in mapped[n]))))
    again = {p: refrozen[p] for p in named if p in refrozen}
    if again:
        raise FTSConfigurationError(
            f"Phase {depth} refreezes tensors an earlier phase refroze and no phase has thawed since (tensor: phase):"
            f" {dict(sorted(again.items()))}.")
    unthawed = sorted(p for p in named if p not in live)
    if unthawed:
        raise FTSConfigurationError(f"Phase {depth} refreezes tensors no earlier phase thawed: {unthawed}.")
    coupled = [c for c in dict.fromkeys(bound.coupled(named)) if c not in named and c in live]
    return named + coupled, {k: tuple(v) for k, v in mapped.items()}


def _compile_transforms(depth: int, transforms: Mapping[str, Any], bound: BoundView, logical_names: Sequence[str],
                        roles: Mapping[str, str], shared: tuple, addressing: _Addressing,
                        trainable: Collection[str]) -> dict[str, tuple[Transform, ...]]:
    """Resolve phase ``depth``'s ``grad_transforms`` entries as ``params`` entries resolve, to each physical
    tensor's transforms in entry order, with the indices its entry's gradient unit set keeps (``None`` without one,
    or when it keeps every index), checking each tensor is ``trainable`` once the phase is in place."""
    out: dict[str, list[Transform]] = {}
    for entry, fns in transforms.items():
        ((is_sel, names),) = _expand([entry], depth, bound, logical_names, roles, shared, addressing)
        try:
            mapped = dict(bound.to_physical(names)) if names and not is_sel else {}
        except Unresolvable as err:
            raise ScheduleUnresolvable(err.names, f"(phase {depth}'s `grad_transforms`)") from err
        physical = list(dict.fromkeys(names if is_sel else (q for n in names for q in mapped[n])))
        untrained = sorted(p for p in physical if p not in trainable)
        if untrained:
            raise FTSConfigurationError(
                f"Phase {depth}'s `grad_transforms` entry {entry!r} selects tensors that are not trainable in the"
                f" phase: {untrained}.")
        grad = addressing.split(entry, depth)[1] if is_address(entry) else None
        units = addressing.masks(entry, grad, depth, roles) if grad is not None else {}
        listed = list(fns) if isinstance(fns, (list, tuple)) else [fns]
        for p in physical:
            rows = units.get(p)
            kept = tuple(rows) if rows is not None and len(rows) < addressing.widths[p] else None
            out.setdefault(p, []).extend((fn, kept) for fn in listed)
    return {p: tuple(ts) for p, ts in out.items()}


_ADDRESSES_WARNED = False


def _warn_addresses_experimental(schedule: Mapping[int, Mapping[str, Any]],
                                 addresses: Mapping[str, Any] | None) -> None:
    """Warn once per process the first time a schedule uses an address or named sets: the grammar is venngram's and
    experimental. Names and regexes never warn."""
    global _ADDRESSES_WARNED
    if _ADDRESSES_WARNED:
        return
    used = [e for phase in schedule.values()
            for e in [*phase.get("params", []), *(phase.get("promote") or {}), *(phase.get("refreeze") or [])]
            if is_address(e)]
    if not used and not addresses:
        return
    _ADDRESSES_WARNED = True
    example = f" (`{used[0]}`)" if used else ""
    rank_zero_warn(f"This schedule uses addresses{example}, which are experimental: their grammar and address books"
                   f" come from venngram {venngram.__version__} and may change in a future release. Parameter names"
                   " and regexes are unaffected.", stacklevel=3)


SUPPORT_RECORD_PATH = pathlib.Path(__file__).with_name("support_record.yaml")


@functools.cache
def support_record() -> SupportRecord:
    """Fine-Tuning Scheduler's venngram support record (:mod:`venngram.support`): what a schedule's address books
    may ask of it.

    It is shipped as a document (:data:`SUPPORT_RECORD_PATH`), so a book can be checked against it without
    Fine-Tuning Scheduler installed (``python -m venngram.conformance <location> --support <record>``).
    """
    return SupportRecord.from_file(SUPPORT_RECORD_PATH)


def referenced_sets(schedule: Mapping[int, Mapping[str, Any]]) -> set[str]:
    """The named sets a schedule's ``params``, ``promote`` and ``refreeze`` entries reference (``@<set>``),
    directly."""
    out: set[str] = set()
    for phase in schedule.values():
        for entry in [*(phase.get("params") or []), *(phase.get("promote") or {}), *(phase.get("refreeze") or [])]:
            if isinstance(entry, str) and is_address(entry):
                out.update(references(parse_address(entry)))
    return out


def _is_gradient(entry: Any, sets: Mapping[str, Any]) -> bool:
    try:
        return split_of(entry, sets)[1] is not None
    except VenngramError:  # a missing set, or a disjunct mixing values and gradients: compilation names it
        return False


def gradient_entries(schedule: Mapping[int, Mapping[str, Any]],
                     sets: Mapping[str, Any] | None = None) -> dict[int, list[str]]:
    """Each phase's ``params`` entries that address gradients, in whole or in part (``of: grad``, written in the
    entry or in a set of ``sets`` it references), for the phases that have any."""
    out: dict[int, list[str]] = {}
    for depth, phase in schedule.items():
        grads = [e for e in phase.get("params") or [] if isinstance(e, str) and is_address(e)
                 and _is_gradient(e, sets or {})]
        if grads:
            out[depth] = grads
    return out


@functools.cache
def _mask_support(tuning: TuningSupport) -> TuningSupport:
    """``tuning`` as it judges the sets a schedule uses under ``of: grad``, where a unit set is a gradient mask
    rather than units to tune.

    venngram's support check reads every unit set as units to tune, which the record refuses (it honours no ``units``
    field, and declares ``units: none``), and its unit patterns name no gradient mask, so the refusal is lifted here
    for those sets only.
    """
    return dataclasses.replace(tuning, fields=tuning.fields | {"units"}, units="unit-adapter")


def _check_book_requirements(addresses: Mapping[str, Any] | None, naming_scheme: str,
                             used: Collection[str] = (), used_as_grad: Collection[str] = ()) -> None:
    """Refuse address books written in another vocabulary than the view's (``requires.naming_scheme``), and books
    whose sets in ``used`` and ``used_as_grad`` (and the sets those reference) ask for something
    :func:`support_record` does not honour.

    A set in ``used_as_grad`` is referenced by an ``of: grad`` entry, so its unit sets are gradient masks.
    """
    record = support_record()
    assert record.tuning is not None
    for book in getattr(addresses, "books", ()):
        try:
            book.check_requires(naming_scheme=naming_scheme)
        except VenngramError as err:
            raise FTSConfigurationError(f"The schedule's `addresses` do not apply through this view: {err}") from err
        own = set(book.sets())
        refusals = record.check_book(book, sets=sorted(set(used) & own))
        if set(used_as_grad) & own:
            masked = _mask_support(record.tuning).check_book(book, sets=sorted(set(used_as_grad) & own))
            refusals += [r for r in masked if r not in refusals]
        if refusals:
            raise FTSConfigurationError("The schedule's `addresses` ask for what Fine-Tuning Scheduler does not tune:"
                                        "\n  " + "\n  ".join(refusals))


def compile_schedule(schedule: Mapping[int, Mapping[str, Any]], view: Any, *, legacy: bool = False,
                     addresses: Mapping[str, Any] | None = None) -> CompiledSchedule:
    """Compile ``schedule`` through ``view`` into the physical parameters each phase thaws.

    Per phase, in depth order: the phase's ``promote`` mapping changes the roles of the ``artifact`` tensors it names
    (to ``latent:<name>`` roles, from then on); addresses (``role:`` selectors among them) select through the roles in
    effect, the view's ``structure()`` and the named sets in ``addresses``; names and
    regexes resolve against the view's logical names (shared-parameter copies are pruned, as for the Lightning
    callback) and map to physical names through one ``to_physical`` call; the view's coupled tensors join the phase
    unless an earlier phase claimed them (first come, first served, as the runtime thaw behaves). A physical name an
    earlier phase claimed only as a coupled tensor is likewise dropped from a later phase (with a debug message).
    A phase's ``refreeze`` entries resolve the same way, to tensors still trainable when it starts. A refrozen tensor
    can be thawed again by a later phase; the checks above count a tensor's thaws only while it is trainable.

    An address of gradients (``of: grad``) thaws what it selects, as any address does, and compiles to a gradient mask
    over each tensor's first axis (venngram's :func:`~venngram.resolve.gradient_masks`): the union of the phase's
    masks of that tensor, or none when another of the phase's entries selects it whole. A later phase may select a
    masked tensor again, which widens its mask (to the whole tensor, when it selects it whole) rather than thawing it a
    second time; a ``refreeze`` drops the mask with the tensor. A book's unit sets are judged as gradient masks in the
    sets an ``of: grad`` entry uses, and as units to tune (refused) elsewhere.

    A phase's ``grad_transforms`` entries resolve the same way, to tensors trainable once the phase is in place, each
    with its transforms and the rows its entry's gradient unit set keeps (:attr:`CompiledPhase.transforms`); they apply
    until a later phase states its own (:meth:`CompiledSchedule.transforms_at`).

    Args:
        schedule: The schedule (phase depth mapped to its configuration; only ``params``, ``promote``, ``refreeze``
            and ``grad_transforms`` are read).
        view: A tunable view (bound or not).
        legacy: Warn instead of refusing where the legacy adapter path tolerated it: storage shared across phases, and
            coupled tensors whose role is never thawed.
        addresses: The named sets ``@<set>`` references resolve against (see :func:`load_address_sets`).

    Returns:
        The compiled schedule.

    Raises:
        FTSConfigurationError: If an address book requires another naming scheme than the view's; an entry resolves
            to nothing or an address does not resolve; phases share storage
            (unless ``legacy``); a phase schedules a ``fitted`` or ``artifact`` tensor; a phase promotes a tensor
            that is not an ``artifact`` one; a phase schedules quantized or packed tensors through logical names
            that are views of them, or thaws a tensor whose storage cannot require gradients (see
            :func:`~finetuning_scheduler.core.schedule.requires_grad_capable`); or a phase refreezes a tensor that is
            not trainable when it starts, both thaws and refreezes a tensor, or leaves nothing trainable; or a
            ``refreeze`` or ``promote`` entry addresses gradients; or a ``grad_transforms`` entry selects a tensor that
            is not trainable in its phase.
    """
    _warn_addresses_experimental(schedule, addresses)
    bound: BoundView = as_view(view)
    support = bound.support()
    if not support.trainable:
        raise FTSConfigurationError(f"The model cannot be fine-tuned through {type(bound.view).__name__}: "
                                    f"{support.reason}")
    grads = gradient_entries(schedule, addresses)
    by_value = {d: {**p, "params": [e for e in p.get("params") or [] if e not in grads.get(d, ())]}
                for d, p in schedule.items()}
    _check_book_requirements(addresses, support.naming_scheme, referenced_sets(by_value),
                             referenced_sets({d: {"params": es} for d, es in grads.items()}))
    root = bound.root
    tensors = dict(root.named_parameters())
    widths = {n: t.shape[0] if t.dim() else 1 for n, t in tensors.items()}
    roles = bound.roles()
    logical_names = [n for n, _ in bound.logical_named_parameters()]
    shared = shared_params_ref(root) if support.naming_scheme == "physical" else ([], set())
    # the tensors trainable after the phases so far, with the phase that thawed each: thawed by name, or with a name
    live_primary: dict[str, int] = {}
    within: list[list[str]] = []
    across: list[list[tuple[str, int]]] = []
    live_coupled: dict[str, int] = {}
    refrozen: dict[str, int] = {}  # the tensors refrozen and not thawed since, with the phase that refroze each
    masked: dict[str, tuple[int, ...]] = {}  # the trainable tensors' gradient masks so far
    lives: Counter[str] = Counter()  # how often each tensor has been refrozen: a thaw after a refreeze is a new one
    logical_to_physical: dict[str, tuple[str, ...]] = {}
    phases: list[CompiledPhase] = []
    overlaps: list[tuple[str, str]] = []
    roles_now = dict(roles)  # the view's roles plus the promotions so far
    promotions: dict[int, dict[str, str]] = {}
    addressing = _Addressing(bound, list(tensors), addresses or {}, widths)
    for depth in sorted(schedule):
        if schedule[depth].get("promote"):
            promotions[depth] = _compile_promotions(depth, schedule[depth]["promote"], bound, logical_names, roles_now,
                                                    shared, addressing)
            roles_now.update(promotions[depth])
        entries = schedule[depth].get("params", [])
        resolved = _expand(entries, depth, bound, logical_names, roles_now, shared, addressing)
        logical = list(dict.fromkeys(n for is_sel, names in resolved if not is_sel for n in names))
        try:
            mapped = dict(bound.to_physical(logical)) if logical else {}
        except Unresolvable as err:
            raise ScheduleUnresolvable(err.names, f"(phase {depth})") from err
        except Exception as err:  # a view that does not raise `Unresolvable` itself: name what it failed on
            raise ScheduleUnresolvable(_unmapped(bound, logical),
                               f"(phase {depth}; {type(bound.view).__name__}.to_physical raised"
                               f" {type(err).__name__}: {err})") from err
        logical_to_physical.update({k: tuple(v) for k, v in mapped.items()})
        primary = list(dict.fromkeys(p for is_sel, names in resolved
                                     for p in (names if is_sel else (q for n in names for q in mapped[n]))))
        # the disjointness check counts, within a phase, the names its entries resolved to (a name resolved twice
        # counts twice, as the Lightning callback counts it) and, across phases, the physical parameters
        within.append([n for is_sel, names in resolved if not is_sel for n in names])
        unknown = [p for p in primary if p not in tensors]
        if unknown:
            raise FTSConfigurationError(
                f"Phase {depth} resolved to names that are not parameters of the view's root module: {unknown}.")
        refreeze: list[str] = []
        if schedule[depth].get("refreeze"):
            refreeze, refreeze_mapped = _compile_refreeze(depth, schedule[depth]["refreeze"], bound, logical_names,
                                                          roles_now, shared, addressing,
                                                          {**live_primary, **live_coupled}, refrozen)
            logical_to_physical.update(refreeze_mapped)
        dropped = [p for p in primary if p in live_coupled]
        if dropped:
            rank_zero_debug(f"Phase {depth}: {dropped} were already thawed with an earlier phase's coupled tensors.")
            primary = [p for p in primary if p not in live_coupled]
        kept = _phase_masks(entries, resolved, mapped, depth, roles_now, addressing)
        # a tensor an earlier phase masked that this phase selects again keeps training in its earlier groups, with
        # the union of the two masks (no mask once this phase trains it whole)
        phase_masks: dict[str, tuple[int, ...] | None] = {}
        reselected = [p for p in primary if p in masked]
        for p in reselected:
            union = tuple(sorted(set(masked[p]) | set(kept.get(p, range(widths[p])))))
            if union != masked[p]:
                phase_masks[p] = union if len(union) < widths[p] else None
        primary = [p for p in primary if p not in masked]
        phase_masks.update({p: kept[p] for p in primary if p in kept})
        coupled = [c for c in dict.fromkeys(bound.coupled(primary))
                   if c not in primary and c not in live_primary and c not in live_coupled]
        unknown = [c for c in coupled if c not in tensors]
        if unknown:
            raise FTSConfigurationError(
                f"The view's coupled tensors for phase {depth} are not parameters of its root module: {unknown}.")
        both = sorted(set(refreeze) & (set(primary) | set(coupled) | set(reselected)))
        if both:
            raise FTSConfigurationError(f"Phase {depth} both thaws and refreezes these tensors: {both}.")
        cp = CompiledPhase(depth, tuple(logical), tuple(primary), tuple(coupled), tuple(refreeze), phase_masks)
        _check_phase_tensors(cp, roles_now, tensors, support.logical_names_are_views, legacy)
        # storage is shared only with what is trainable when the phase starts: a refrozen tensor no longer counts
        tagged = list(live_primary.items()) + [(p, depth) for p in primary if p not in live_primary]
        for (a, da), (b, db) in _overlapping(tagged, tensors):
            if da == db == depth:
                overlaps.append((a, b))
            elif depth in (da, db) and a != b:  # a name repeated across phases is refused by name, after the loop
                msg = (f"Phases {min(da, db)} and {max(da, db)} schedule tensors that share storage: '{a}' and '{b}'."
                       " A tensor can be thawed by one phase only, unless a phase refreezes it in between.")
                if not legacy:
                    raise FTSConfigurationError(msg)
                rank_zero_warn(msg)
        across.append([(p, lives[p]) for p in primary])
        live_primary.update(dict.fromkeys(primary, depth))
        live_coupled.update(dict.fromkeys(coupled, depth))
        for p in [*primary, *coupled]:
            refrozen.pop(p, None)
        for p, m in phase_masks.items():
            if m is None:
                masked.pop(p, None)
            else:
                masked[p] = m
        for p in refreeze:
            live_primary.pop(p, None)
            live_coupled.pop(p, None)
            masked.pop(p, None)
            refrozen[p] = depth
            lives[p] += 1
        if refreeze and not (live_primary or live_coupled):
            raise FTSConfigurationError(f"Phase {depth} leaves nothing trainable: it refreezes every trainable tensor.")
        if schedule[depth].get("grad_transforms") is not None:
            cp = replace(cp, transforms=_compile_transforms(depth, schedule[depth]["grad_transforms"], bound,
                                                            logical_names, roles_now, shared, addressing,
                                                            {**live_primary, **live_coupled}))
        phases.append(cp)
    _check_disjoint_names(within, across, legacy)
    if overlaps:
        rank_zero_debug(f"Tensors sharing storage within a phase: {overlaps}")
    return CompiledSchedule(phases=tuple(phases), roles=dict(roles), naming_scheme=support.naming_scheme,
                            digest=_digest(phases, roles, promotions), overlaps=tuple(overlaps),
                            logical_to_physical=logical_to_physical, promotions=promotions)


def view_default_schedule(view: Any, order: Literal["output_first", "input_first"] = "output_first") -> dict:
    """Generate a default schedule over a view's logical names.

    Logical names standing for a tensor whose role is never thawed (``fitted`` or ``artifact``), or for a tensor whose
    storage cannot require gradients (a quantized or packed base under trainable adapters, for example), are left out
    (with a warning when they also stand for trainable tensors), and groups left empty are dropped.

    Args:
        view: A tunable view (bound or not).
        order: ``"output_first"`` (the default, as Fine-Tuning Scheduler has always generated schedules) uses the
            view's ``depth_groups()``; ``"input_first"`` thaws the same groups in the reverse order.

    Returns:
        The schedule, mapping each phase to its ``params``.
    """
    if order not in ("output_first", "input_first"):
        raise FTSConfigurationError(f"`order` must be 'output_first' or 'input_first', got {order!r}.")
    bound = as_view(view)
    groups = bound.depth_groups()
    roles = bound.roles()
    tensors = dict(bound.root.named_parameters())
    packed = {n for n, t in tensors.items() if not requires_grad_capable(t)}
    if packed or any(is_frozen_role(r) for r in roles.values()):
        names = list(dict.fromkeys(n for g in groups for n in g))
        mapped = bound.to_physical(names)

        def _excluded(p: str) -> bool:
            return is_frozen_role(roles.get(p, "base")) or p in packed

        excluded = {n for n in names if any(_excluded(p) for p in mapped[n])}
        # a name standing for excluded and trainable tensors alike cannot be scheduled (compilation refuses it)
        mixed = sorted(n for n in excluded if not all(_excluded(p) for p in mapped[n]))
        if mixed:
            rank_zero_warn(f"The generated schedule leaves out {mixed}: each stands for tensors whose role is never"
                           " thawed (`fitted` or `artifact`), or whose storage cannot require gradients, as well as"
                           " trainable ones, so it cannot be scheduled.")
        if left_packed := sorted(n for n in excluded if n not in mixed and any(p in packed for p in mapped[n])):
            rank_zero_info(f"The generated schedule leaves out {len(left_packed)} names whose storage cannot require"
                           f" gradients (quantized or packed), such as {left_packed[0]!r}.")
        groups = [kept for g in groups if (kept := [n for n in g if n not in excluded])]
    if order == "input_first":
        groups = [list(reversed(g)) for g in reversed(groups)]
    return {depth: {"params": list(g)} for depth, g in enumerate(groups)}


__all__ = ["CompiledPhase", "CompiledSchedule", "compile_schedule", "gradient_entries", "owner_module",
           "view_default_schedule"]
