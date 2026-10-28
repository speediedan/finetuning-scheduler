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
norms) join the first phase that pulls them. It refuses a schedule whose phases share storage, that schedules a
``fitted`` or ``instrument`` tensor, or that names anything the view cannot resolve, naming the offending input. The
schedule itself keeps its logical names.

.. warning:: This is an experimental API and may change in a future release.
"""
import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from torch import Tensor

from finetuning_scheduler.core.dist import rank_zero_debug, rank_zero_warn
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from venngram import is_address, ParamSpace, select
from venngram.errors import VenngramError

from finetuning_scheduler.core.roles import is_frozen_role, role_kind
from finetuning_scheduler.core.schedule import parse_promote, resolve_param_entries, shared_params_ref
from finetuning_scheduler.core.views import as_view, BoundView, ScheduleUnresolvable, Unresolvable


@dataclass(frozen=True)
class CompiledPhase:
    """One phase's compiled parameters.

    Attributes:
        depth: The phase.
        logical: The phase's resolved logical names (role selectors are expanded into ``physical`` directly).
        physical: The physical parameters the phase names, through the view or a role selector.
        coupled: The physical parameters thawed with them that no earlier phase claimed.
    """

    depth: int
    logical: tuple[str, ...]
    physical: tuple[str, ...]
    coupled: tuple[str, ...] = ()

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
            detail = str(err).partition(" does not resolve: ")[2] or str(err)
            raise FTSConfigurationError(f"The address '{entry}' in phase {depth} does not resolve: {detail}") from err


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
            f"Phase {cp.depth} schedules tensors whose role is never thawed (`fitted` or `instrument`): {frozen}.")
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


def _repeats(lists: Sequence[Sequence[str]]) -> list[str]:
    """Each name once per extra occurrence across ``lists``, as the Lightning callback lists them."""
    counts = Counter(itertools.chain(*lists))
    counts.subtract(Counter(set().union(*map(set, lists))))
    return list(counts.elements())


def _check_disjoint_names(within: Sequence[Sequence[str]], across: Sequence[Sequence[str]], legacy: bool) -> None:
    """Refuse (or, with ``legacy``, warn about) a name resolved twice within a phase or a physical parameter
    scheduled by two phases, listing every repeat in the Lightning callback's format.

    Names are compared rather than storage, so the check gives the same answer on every rank.
    """
    dup_params = [n for phase in within for n in _repeats([phase])] + _repeats(across)
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
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _compile_promotions(depth: int, promote: Mapping[str, str], bound: BoundView, logical_names: Sequence[str],
                        roles: Mapping[str, str], shared: tuple, addressing: _Addressing) -> dict[str, str]:
    """Resolve phase ``depth``'s ``promote`` mapping to physical tensors and their new roles, checking each is an
    ``instrument`` tensor promoted once."""
    parse_promote(depth, promote)
    promoted: dict[str, str] = {}
    for source, target in promote.items():
        ((is_sel, names),) = _expand([source], depth, bound, logical_names, roles, shared, addressing)
        physical = list(dict.fromkeys(names if is_sel else (q for qs in bound.to_physical(names).values() for q in qs)))
        not_instrument = {p: roles.get(p, "base") for p in physical if role_kind(roles.get(p, "base")) != "instrument"}
        if not_instrument:
            raise FTSConfigurationError(
                f"Phase {depth} promotes tensors whose role is not `instrument` (an earlier phase may already have"
                f" promoted them): {dict(sorted(not_instrument.items()))}. Only `instrument` tensors can be promoted,"
                " to a `latent:<name>` role.")
        twice = sorted(p for p in physical if p in promoted)
        if twice:
            raise FTSConfigurationError(f"Phase {depth}'s `promote` names these tensors more than once: {twice}.")
        promoted.update(dict.fromkeys(physical, target))
    return promoted


def compile_schedule(schedule: Mapping[int, Mapping[str, Any]], view: Any, *, legacy: bool = False,
                     addresses: Mapping[str, Any] | None = None) -> CompiledSchedule:
    """Compile ``schedule`` through ``view`` into the physical parameters each phase thaws.

    Per phase, in depth order: the phase's ``promote`` mapping changes the roles of the ``instrument`` tensors it names
    (to ``latent:<name>`` roles, from then on); addresses (``role:`` selectors among them) select through the roles in
    effect, the view's ``structure()`` and the named sets in ``addresses``; names and
    regexes resolve against the view's logical names (shared-parameter copies are pruned, as for the Lightning
    callback) and map to physical names through one ``to_physical`` call; the view's coupled tensors join the phase
    unless an earlier phase claimed them (first come, first served, as the runtime thaw behaves). A physical name an
    earlier phase claimed only as a coupled tensor is likewise dropped from a later phase (with a debug message).

    Args:
        schedule: The schedule (phase depth mapped to its configuration; only ``params`` and ``promote`` are read).
        view: A tunable view (bound or not).
        legacy: Warn instead of refusing where the legacy adapter path tolerated it: storage shared across phases, and
            coupled tensors whose role is never thawed.
        addresses: The named sets ``@<set>`` references resolve against (see :func:`load_address_sets`).

    Returns:
        The compiled schedule.

    Raises:
        FTSConfigurationError: If an entry resolves to nothing or an address does not resolve; phases share storage
            (unless ``legacy``); a phase schedules a ``fitted`` or ``instrument`` tensor; a phase promotes a tensor
            that is not an ``instrument`` one; or a phase schedules quantized or packed tensors through logical names
            that are views of them.
    """
    bound: BoundView = as_view(view)
    support = bound.support()
    if not support.trainable:
        raise FTSConfigurationError(f"The model cannot be fine-tuned through {type(bound.view).__name__}: "
                                    f"{support.reason}")
    root = bound.root
    tensors = dict(root.named_parameters())
    roles = bound.roles()
    logical_names = [n for n, _ in bound.logical_named_parameters()]
    shared = shared_params_ref(root) if support.naming_scheme == "physical" else ([], set())
    claimed_primary: dict[str, int] = {}
    within: list[list[str]] = []
    claimed_coupled: dict[str, int] = {}
    logical_to_physical: dict[str, tuple[str, ...]] = {}
    phases: list[CompiledPhase] = []
    overlaps: list[tuple[str, str]] = []
    roles_now = dict(roles)  # the view's roles plus the promotions so far
    promotions: dict[int, dict[str, str]] = {}
    addressing = _Addressing(bound, list(tensors), addresses or {})
    for depth in sorted(schedule):
        if schedule[depth].get("promote"):
            promotions[depth] = _compile_promotions(depth, schedule[depth]["promote"], bound, logical_names, roles_now,
                                                    shared, addressing)
            roles_now.update(promotions[depth])
        resolved = _expand(schedule[depth].get("params", []), depth, bound, logical_names, roles_now, shared,
                           addressing)
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
        dropped = [p for p in primary if p in claimed_coupled]
        if dropped:
            rank_zero_debug(f"Phase {depth}: {dropped} were already thawed with an earlier phase's coupled tensors.")
            primary = [p for p in primary if p not in claimed_coupled]
        coupled = [c for c in dict.fromkeys(bound.coupled(primary))
                   if c not in primary and c not in claimed_primary and c not in claimed_coupled]
        unknown = [c for c in coupled if c not in tensors]
        if unknown:
            raise FTSConfigurationError(
                f"The view's coupled tensors for phase {depth} are not parameters of its root module: {unknown}.")
        cp = CompiledPhase(depth, tuple(logical), tuple(primary), tuple(coupled))
        _check_phase_tensors(cp, roles_now, tensors, support.logical_names_are_views, legacy)
        tagged = list(claimed_primary.items()) + [(p, depth) for p in primary if p not in claimed_primary]
        for (a, da), (b, db) in _overlapping(tagged, tensors):
            if da == db == depth:
                overlaps.append((a, b))
            elif depth in (da, db) and a != b:  # a name repeated across phases is refused by name, after the loop
                msg = (f"Phases {min(da, db)} and {max(da, db)} schedule tensors that share storage: '{a}' and '{b}'."
                       " A tensor can be thawed by one phase only.")
                if not legacy:
                    raise FTSConfigurationError(msg)
                rank_zero_warn(msg)
        claimed_primary.update(dict.fromkeys(primary, depth))
        claimed_coupled.update(dict.fromkeys(coupled, depth))
        phases.append(cp)
    _check_disjoint_names(within, [list(dict.fromkeys(cp.physical)) for cp in phases], legacy)
    if overlaps:
        rank_zero_debug(f"Tensors sharing storage within a phase: {overlaps}")
    return CompiledSchedule(phases=tuple(phases), roles=dict(roles), naming_scheme=support.naming_scheme,
                            digest=_digest(phases, roles, promotions), overlaps=tuple(overlaps),
                            logical_to_physical=logical_to_physical, promotions=promotions)


def view_default_schedule(view: Any, order: Literal["output_first", "input_first"] = "output_first") -> dict:
    """Generate a default schedule over a view's logical names.

    Logical names standing for a tensor whose role is never thawed (``fitted`` or ``instrument``) are left out (with a
    warning when they also stand for trainable tensors), and groups left empty are dropped.

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
    if any(is_frozen_role(r) for r in roles.values()):
        names = list(dict.fromkeys(n for g in groups for n in g))
        mapped = bound.to_physical(names)
        frozen = {n for n in names if any(is_frozen_role(roles.get(p, "base")) for p in mapped[n])}
        # a name standing for frozen and trainable tensors alike cannot be scheduled (compilation refuses it)
        mixed = sorted(n for n in frozen if not all(is_frozen_role(roles.get(p, "base")) for p in mapped[n]))
        if mixed:
            rank_zero_warn(f"The generated schedule leaves out {mixed}: each stands for tensors whose role is never"
                           " thawed (`fitted` or `instrument`) as well as trainable ones, so it cannot be scheduled.")
        groups = [kept for g in groups if (kept := [n for n in g if n not in frozen])]
    if order == "input_first":
        groups = [list(reversed(g)) for g in reversed(groups)]
    return {depth: {"params": list(g)} for depth, g in enumerate(groups)}


__all__ = ["CompiledPhase", "CompiledSchedule", "compile_schedule", "owner_module", "view_default_schedule"]
