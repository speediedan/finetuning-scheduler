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
Tunable View Conformance
^^^^^^^^^^^^^^^^^^^^^^^^

Conformance cases for :class:`~finetuning_scheduler.tunable_view.views.TunableView` implementations: the cases that
hold for any consumer of the protocol.

:class:`ViewConformance` takes a factory returning a fresh view and checks it against the protocol: its logical names
map to the root's parameters and back, its checkpoint state leaves out ``instrument`` tensors and restores everything
else exactly through a strict load, its roles are valid and declared, and a promotion keeps names. Which cases apply is
decided by the view's own support record, and a run in which no case applied fails, so a suite cannot pass vacuously.
A consumer extends the suite with the cases its own use of a view needs (Fine-Tuning Scheduler's
``TunableViewConformance`` adds schedule compilation, thawing, optimizer step and refusal cases).

The cases need only PyTorch (no pytest, no inputs and no forward pass). :meth:`ViewConformance.pytest_params` turns
them into ``pytest.mark.parametrize`` arguments:

.. code-block:: python

    @pytest.mark.parametrize("case", ViewConformance(make_my_view).pytest_params(), ids=lambda case: case.__name__)
    def test_my_view_conformance(case):
        case()

.. warning:: This is an experimental API and may change in a future release.
"""
import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from .roles import declared_kinds, role_kind, validate_roles
from .views import as_view, BoundView, instrument_state_keys, TunableSupport


class ConformanceError(AssertionError):
    """A conformance case failed, or no case applied."""


@dataclass(frozen=True)
class ConformanceCase:
    """One conformance case.

    Attributes:
        name: The case's name.
        kind: The property family it checks (``mapping``, ``checkpoint``, ``roles`` or ``promotion`` for the
            protocol's cases; a consumer's suite may add others).
        applies: Whether the case applies to a view, given its support record and the view itself.
        run: Runs the case, raising :class:`ConformanceError` on failure.
    """

    name: str
    kind: str
    applies: Callable[[TunableSupport, BoundView], bool]
    run: Callable[[], None]


@dataclass
class ConformanceReport:
    """The outcome of :meth:`ViewConformance.run`.

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


def _instrument_parameters(view: BoundView) -> list[str]:
    params = {n for n, _ in view.root.named_parameters()}
    return sorted(n for n, r in view.roles().items() if n in params and role_kind(r) == "instrument")


class ViewConformance:
    """The protocol's conformance cases for one view implementation.

    Args:
        make_view: Returns a fresh view over a fresh model (cases change parameters).
    """

    def __init__(self, make_view: Callable[[], Any]) -> None:
        self.make_view = make_view

    def _fresh(self) -> BoundView:
        return as_view(self.make_view())

    def protocol_cases(self) -> dict[str, ConformanceCase]:
        """The protocol's cases by name, for a consumer's suite to place among its own."""
        trainable = lambda s, v: s.trainable  # noqa: E731
        return {c.name: c for c in (
            ConformanceCase("mapping_completeness", "mapping", trainable, self._mapping_completeness),
            ConformanceCase("mapping_round_trip", "mapping", lambda s, v: s.trainable and s.naming_scheme != "physical",
                            self._mapping_round_trip),
            ConformanceCase("checkpoint_round_trip", "checkpoint", trainable, self._checkpoint_round_trip),
            ConformanceCase("roles_valid", "roles", lambda s, v: s.trainable and (s.declared_roles != {"base"}
                                                                                  or bool(v.roles())),
                            self._roles_valid),
            ConformanceCase("promotion", "promotion", lambda s, v: s.trainable and bool(_instrument_parameters(v)),
                            self._promotion),
        )}

    def cases(self) -> list[ConformanceCase]:
        """Every case, in run order."""
        return list(self.protocol_cases().values())

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

    def pytest_params(self) -> list[Any]:
        """The applicable cases as callables for ``pytest.mark.parametrize`` (each named after its case, for
        ``ids``).

        Each callable runs one case and raises :class:`ConformanceError` on failure. With no applicable case, the list
        holds one callable that raises (the vacuity guard).
        """
        probe = self._fresh()
        support = probe.support()

        def _bind(case: ConformanceCase) -> Callable[[], None]:
            def _run() -> None:
                report = self.run(select=[case.name])
                if report.failures:
                    raise ConformanceError(report.failures[case.name])

            _run.__name__ = case.name
            return _run

        applicable = [c for c in self.cases() if c.applies(support, probe)]
        if not applicable:
            def _vacuous() -> None:
                raise ConformanceError("No conformance case applied to this view, so nothing was checked.")

            return [_vacuous]
        return [_bind(c) for c in applicable]

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

    def _roles_valid(self) -> None:
        """Roles name the root's parameters or buffers, use valid roles, and the support record declares every role
        kind they use."""
        view = self._fresh()
        roles = view.roles()
        validate_roles(roles, (n for n, _ in view.root.named_parameters()), (n for n, _ in view.root.named_buffers()))
        undeclared = sorted(declared_kinds(roles) - view.support().declared_roles)
        _check(not undeclared, f"The support record does not declare these role kinds the view uses: {undeclared}.")

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


__all__ = ["ConformanceCase", "ConformanceError", "ConformanceReport", "ViewConformance"]
