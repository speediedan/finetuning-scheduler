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
Fine-Tuning Scheduler View Conformance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conformance cases for :class:`~finetuning_scheduler.core.views.TunableView` implementations.

:class:`TunableViewConformance` takes a factory returning a fresh view and checks it against the contract Fine-Tuning
Scheduler relies on. Which cases apply is decided by the view's own support record: a view declaring
``trainable=False`` runs only the refusal case; role cases run only for the role kinds a view declares; the
round-trip case runs for views whose naming scheme is not ``"physical"``; the parametrization case runs when the model
has parametrized tensors. A run in which no case applied fails, so a suite cannot pass vacuously.

The cases need only PyTorch (no pytest, no inputs and no forward pass): the optimizer-step case fills each trainable
tensor's gradient with ones. :func:`pytest_params` turns the cases into ``pytest.mark.parametrize`` arguments:

.. code-block:: python

    @pytest.mark.parametrize("case", pytest_params(make_my_view), ids=lambda case: case.__name__)
    def test_my_view_conformance(case):
        case()

.. warning:: This is an experimental API and may change in a future release.
"""
import copy
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn.utils import parametrize

from finetuning_scheduler.core.compile import compile_schedule, view_default_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.phase import thaw_params
from finetuning_scheduler.core.roles import declared_kinds, is_frozen_role, role_kind, validate_roles
from finetuning_scheduler.core.views import as_view, BoundView, instrument_state_keys, TunableSupport

NO_SUCH_NAME = "__fts_conformance_no_such_name__"


class ConformanceError(AssertionError):
    """A conformance case failed, or no case applied."""


@dataclass(frozen=True)
class ConformanceCase:
    """One conformance case.

    Attributes:
        name: The case's name.
        kind: The property family it checks (``mapping``, ``storage``, ``thaw``, ``step``, ``checkpoint``,
            ``refusal``, ``roles``, ``promotion`` or ``parametrization``).
        applies: Whether the case applies to a view, given its support record and the view itself.
        run: Runs the case, raising :class:`ConformanceError` on failure.
    """

    name: str
    kind: str
    applies: Callable[[TunableSupport, BoundView], bool]
    run: Callable[[], None]


@dataclass
class ConformanceReport:
    """The outcome of :meth:`TunableViewConformance.run`.

    Attributes:
        ran: The cases that ran.
        skipped: The cases that did not apply.
        failures: Each failed case's message.
    """

    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.ran) and not self.failures


def _check(condition: bool, msg: str) -> None:
    if not condition:
        raise ConformanceError(msg)


def _freeze(root: torch.nn.Module) -> None:
    for p in root.parameters():
        p.requires_grad = False


def _trainable(root: torch.nn.Module) -> set[str]:
    return {n for n, p in root.named_parameters() if p.requires_grad}


def _instrument_parameters(view: BoundView) -> list[str]:
    params = {n for n, _ in view.root.named_parameters()}
    return sorted(n for n, r in view.roles().items() if n in params and role_kind(r) == "instrument")


def _has_frozen_parameter(view: BoundView) -> bool:
    """Whether a parameter (not only a buffer) has a role that is never thawed."""
    params = {n for n, _ in view.root.named_parameters()}
    return any(n in params and is_frozen_role(r) for n, r in view.roles().items())


class TunableViewConformance:
    """Conformance cases for one view implementation.

    Args:
        make_view: Returns a fresh view over a fresh model (cases change parameters and ``requires_grad``).
        sample_logical: The logical names the per-name cases exercise (all of the view's logical names by default).
        step_fn: Performs the optimizer-step case's step as ``step_fn(view, optimizer)``, for views whose tensors
            receive gradients only through a forward pass. By default, every trainable tensor's gradient is set to
            ones and the optimizer steps.
    """

    def __init__(self, make_view: Callable[[], Any], *, sample_logical: Sequence[str] | None = None,
                 step_fn: Callable[[Any, torch.optim.Optimizer], None] | None = None) -> None:
        self.make_view = make_view
        self.sample_logical = list(sample_logical) if sample_logical is not None else None
        self.step_fn = step_fn

    def _fresh(self) -> BoundView:
        return as_view(self.make_view())

    def _sample(self, bound: BoundView) -> list[str]:
        if self.sample_logical is not None:
            return self.sample_logical
        return list(dict.fromkeys(n for n, _ in bound.logical_named_parameters()))

    def cases(self) -> list[ConformanceCase]:
        """Every case, in run order."""
        trainable = lambda s, v: s.trainable  # noqa: E731
        return [
            ConformanceCase("mapping_completeness", "mapping", trainable, self._mapping_completeness),
            ConformanceCase("mapping_round_trip", "mapping", lambda s, v: s.trainable and s.naming_scheme != "physical",
                            self._mapping_round_trip),
            ConformanceCase("default_schedule_compiles", "storage", trainable, self._default_schedule_compiles),
            ConformanceCase("thaw_is_exact_and_deterministic", "thaw", trainable, self._thaw_exact),
            ConformanceCase("step_changes_only_thawed", "step", trainable, self._step_changes_only_thawed),
            ConformanceCase("checkpoint_round_trip", "checkpoint", trainable, self._checkpoint_round_trip),
            ConformanceCase("unmatched_rule_refuses", "refusal", trainable, self._unmatched_rule_refuses),
            ConformanceCase("untrainable_refuses", "refusal", lambda s, v: not s.trainable, self._untrainable_refuses),
            ConformanceCase("roles_valid", "roles", lambda s, v: s.trainable and (s.declared_roles != {"base"}
                                                                                  or bool(v.roles())),
                            self._roles_valid),
            ConformanceCase("frozen_roles_refuse", "roles", lambda s, v: s.trainable and _has_frozen_parameter(v),
                            self._frozen_roles_refuse),
            ConformanceCase("promotion", "promotion", lambda s, v: s.trainable and bool(_instrument_parameters(v)),
                            self._promotion),
            ConformanceCase("parametrization_mapping", "parametrization",
                            lambda s, v: s.trainable and any(parametrize.is_parametrized(m) for m in v.root.modules()),
                            self._parametrization_mapping),
        ]

    def run(self, select: Iterable[str] | None = None, *, raise_on_failure: bool = False) -> ConformanceReport:
        """Run the applicable cases (optionally only those named in ``select``).

        Raises:
            ConformanceError: If no case applied, or (with ``raise_on_failure``) if any case failed.
        """
        selected = set(select) if select is not None else None
        report = ConformanceReport()
        probe = self._fresh()
        support = probe.support()
        for case in self.cases():
            if (selected is not None and case.name not in selected) or not case.applies(support, probe):
                report.skipped.append(case.name)
                continue
            report.ran.append(case.name)
            try:
                case.run()
            except Exception as err:  # every exception is a failure of the case
                report.failures[case.name] = f"{type(err).__name__}: {err}"
        if not report.ran:
            raise ConformanceError("No conformance case applied to this view, so nothing was checked.")
        if raise_on_failure and report.failures:
            raise ConformanceError(f"Conformance failures: {report.failures}")
        return report

    # ------------------------------------------------------------------------------------------------------------------
    # cases
    # ------------------------------------------------------------------------------------------------------------------

    def _mapping_completeness(self) -> None:
        """Every logical name maps to one or more of the root's parameters."""
        view = self._fresh()
        params = {n for n, _ in view.root.named_parameters()}
        logical = list(dict.fromkeys(n for n, _ in view.logical_named_parameters()))
        _check(bool(logical), "The view lists no logical names.")
        mapped = view.to_physical(logical)
        empty = [n for n in logical if not mapped.get(n)]
        _check(not empty, f"These logical names map to no physical parameter: {empty}.")
        unknown = sorted({p for n in logical for p in mapped[n] if p not in params})
        _check(not unknown, f"These physical names are not parameters of the view's root module: {unknown}.")

    def _mapping_round_trip(self) -> None:
        """Every physical parameter a logical name reaches is reached again through its own logical name."""
        view = self._fresh()
        logical = list(dict.fromkeys(n for n, _ in view.logical_named_parameters()))
        reached = list(dict.fromkeys(p for ps in view.to_physical(logical).values() for p in ps))
        lost = []
        for p in reached:
            back = view.to_logical([p])
            if p not in {q for ps in view.to_physical(back).values() for q in ps}:
                lost.append(p)
        _check(not lost, f"These physical names do not survive a round trip through their logical names: {lost}.")

    def _default_schedule_compiles(self) -> None:
        """The view's own default schedule compiles: no two phases share storage."""
        view = self._fresh()
        compile_schedule(view_default_schedule(view), view)

    def _thaw_exact(self) -> None:
        """Compiling a logical name is deterministic, and thawing the result flips exactly those tensors."""
        for name in self._sample(self._fresh()):
            view = self._fresh()
            physical = view.to_physical([name])[name]
            if any(is_frozen_role(view.roles().get(p, "base")) for p in physical):
                continue
            first = compile_schedule({0: {"params": [name]}}, view).physical_for(0)
            second = compile_schedule({0: {"params": [name]}}, view).physical_for(0)
            _check(first == second, f"Compiling '{name}' twice gave different tensors: {first} and {second}.")
            _freeze(view.root)
            thaw_params(view.root, list(first))
            thawed = _trainable(view.root)
            _check(thawed == set(first), f"Thawing '{name}' made {sorted(thawed)} trainable, expected {sorted(first)}.")

    def _step_changes_only_thawed(self) -> None:
        """An optimizer step over phase 0 of the default schedule changes only phase 0's tensors (and never a
        ``fitted`` or ``instrument`` tensor)."""
        view = self._fresh()
        compiled = compile_schedule(view_default_schedule(view), view)
        _freeze(view.root)
        thaw = list(compiled.physical_for(0))
        thaw_params(view.root, thaw)
        before = {n: t.detach().clone() for n, t in view.root.state_dict(keep_vars=True).items()}
        params = [p for p in view.root.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=1.0)
        if self.step_fn is not None:
            self.step_fn(view.view, optimizer)
        else:
            for p in params:
                p.grad = torch.ones_like(p)
            optimizer.step()
        after = view.root.state_dict(keep_vars=True)
        changed = {n for n, t in after.items() if not torch.equal(t.detach(), before[n])}
        # names are compared through the root's state_dict, which lists every alias of a tied tensor
        thawed_ids = {id(view.root.get_parameter(n)) for n in thaw}
        unexpected = sorted(n for n in changed if id(after[n]) not in thawed_ids)
        _check(not unexpected, f"An optimizer step over phase 0 changed tensors phase 0 does not thaw: {unexpected}.")
        roles = view.roles()
        frozen_changed = sorted(n for n in changed if is_frozen_role(roles.get(n, "base")))
        _check(not frozen_changed, f"An optimizer step changed `fitted` or `instrument` tensors: {frozen_changed}.")

    def _checkpoint_round_trip(self) -> None:
        """The view's checkpoint state leaves out ``instrument`` tensors (under any alias), keeps ``fitted`` ones,
        and restores every checkpointed parameter and buffer exactly through a strict load."""
        view = self._fresh()
        roles = view.roles()
        state = copy.deepcopy(view.checkpoint_state_dict())
        own = view.root.state_dict(keep_vars=True)
        instrument = instrument_state_keys(view.root, roles, own.keys())
        saved_instrument = sorted(instrument & set(state))
        _check(not saved_instrument, f"The checkpoint state includes `instrument` tensors: {saved_instrument}.")
        fitted = {k for k, r in roles.items() if r == "fitted" and k in own}
        unsaved_fitted = sorted(fitted - set(state))
        _check(not unsaved_fitted, f"The checkpoint state leaves out `fitted` tensors: {unsaved_fitted}.")
        reference = {k: t.detach().clone() for k, t in own.items() if k not in instrument}
        with torch.no_grad():
            for t in own.values():
                if t.is_floating_point():
                    t.add_(1.0)
        view.load_checkpoint_state_dict(state, strict=True)
        after = view.root.state_dict(keep_vars=True)
        differ = sorted(k for k, ref in reference.items() if not torch.equal(after[k].detach(), ref))
        _check(not differ, f"Loading the view's checkpoint state did not restore: {differ}.")

    def _unmatched_rule_refuses(self) -> None:
        """A schedule entry that matches nothing is refused."""
        view = self._fresh()
        try:
            compile_schedule({0: {"params": [NO_SUCH_NAME]}}, view)
        except FTSConfigurationError:
            return
        raise ConformanceError(f"A schedule entry matching nothing ('{NO_SUCH_NAME}') was not refused.")

    def _untrainable_refuses(self) -> None:
        """A view declaring ``trainable=False`` is refused, with its reason."""
        raw = self.make_view()
        reason = as_view(raw).support().reason
        try:
            compile_schedule({0: {"params": [NO_SUCH_NAME]}}, raw)
        except FTSConfigurationError as err:
            _check(bool(reason) and str(reason) in str(err), f"The refusal does not give the view's reason: {err}")
            return
        raise ConformanceError("A view declaring `trainable=False` was not refused.")

    def _roles_valid(self) -> None:
        """Roles name the root's parameters or buffers, use valid roles, and the support record declares every role
        kind they use."""
        view = self._fresh()
        roles = view.roles()
        validate_roles(roles, (n for n, _ in view.root.named_parameters()), (n for n, _ in view.root.named_buffers()))
        undeclared = sorted(declared_kinds(roles) - view.support().declared_roles)
        _check(not undeclared, f"The support record does not declare these role kinds the view uses: {undeclared}.")

    def _frozen_roles_refuse(self) -> None:
        """Scheduling a ``fitted`` or ``instrument`` parameter, through its logical name, is refused."""
        view = self._fresh()
        params = {n for n, _ in view.root.named_parameters()}
        frozen = [n for n, r in view.roles().items() if n in params and is_frozen_role(r)]
        for p in frozen:
            fresh = self._fresh()
            logical = fresh.to_logical([p])
            try:
                compile_schedule({0: {"params": logical}}, fresh)
            except FTSConfigurationError:
                continue
            raise ConformanceError(f"Scheduling '{p}' (role {fresh.roles()[p]!r}) through {logical} was not refused.")

    def _promotion(self) -> None:
        """Promoting the ``instrument`` parameters to a ``latent`` role keeps their names, reports the new role,
        leaves each with storage no other tensor shares, and checkpoints them from then on."""
        view = self._fresh()
        names = _instrument_parameters(view)
        role = "latent:conformance"
        renamed = {old: new for old, new in view.promote(names, role).items() if old != new}
        _check(not renamed, f"Promotion renamed tensors: {renamed}.")
        roles = view.roles()
        unreported = sorted(n for n in names if roles.get(n) != role)
        _check(not unreported, f"After promotion, these tensors do not report the role {role!r}: {unreported}.")
        named = list(view.root.named_parameters(remove_duplicate=False)) + list(
            view.root.named_buffers(remove_duplicate=False))
        promoted = dict(view.root.named_parameters(remove_duplicate=False))
        shared = sorted(n for n in names if promoted[n].device.type != "meta" and any(
            other != n and t.device.type != "meta"
            and t.untyped_storage().data_ptr() == promoted[n].untyped_storage().data_ptr() for other, t in named))
        _check(not shared, f"Promoted tensors still share storage with other tensors: {shared}.")
        state = view.checkpoint_state_dict()
        own = view.root.state_dict(keep_vars=True)
        unsaved = sorted(k for k in own if k not in state and k in names)
        _check(not unsaved, f"Promoted tensors are left out of the checkpoint state: {unsaved}.")

    def _parametrization_mapping(self) -> None:
        """Each parametrized tensor resolves by its unparametrized name to its underlying tensors."""
        view = self._fresh()
        missing = []
        for mod_name, mod in view.root.named_modules():
            if not parametrize.is_parametrized(mod):
                continue
            for tensor_name in mod.parametrizations.keys():  # type: ignore[union-attr]
                owner = f"{mod_name}." if mod_name else ""
                logical, prefix = f"{owner}{tensor_name}", f"{owner}parametrizations.{tensor_name}."
                originals = {n for n, _ in view.root.named_parameters() if n.startswith(prefix)}
                try:  # through the schedule path, not only the view's mapping
                    got = set(compile_schedule({0: {"params": [logical]}}, self._fresh()).physical_for(0))
                except FTSConfigurationError:
                    got = set()
                if got != originals:
                    missing.append(logical)
        _check(not missing, f"These parametrized tensors do not resolve to their underlying tensors: {missing}.")


def pytest_params(make_view: Callable[[], Any], **kwargs: Any) -> list[Any]:
    """The applicable cases as callables for ``pytest.mark.parametrize`` (each named after its case, for ``ids``).

    Each callable runs one case and raises :class:`ConformanceError` on failure. With no applicable case, the list
    holds one callable that raises (the vacuity guard).

    Args:
        make_view: Returns a fresh view over a fresh model.
        kwargs: Passed to :class:`TunableViewConformance`.
    """
    suite = TunableViewConformance(make_view, **kwargs)
    probe = as_view(make_view())
    support = probe.support()

    def _bind(case: ConformanceCase) -> Callable[[], None]:
        def _run() -> None:
            report = suite.run(select=[case.name])
            if report.failures:
                raise ConformanceError(report.failures[case.name])

        _run.__name__ = case.name
        return _run

    applicable = [c for c in suite.cases() if c.applies(support, probe)]
    if not applicable:
        def _vacuous() -> None:
            raise ConformanceError("No conformance case applied to this view, so nothing was checked.")

        return [_vacuous]
    return [_bind(c) for c in applicable]
