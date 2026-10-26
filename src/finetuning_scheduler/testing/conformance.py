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

Conformance cases for :class:`~finetuning_scheduler.core.views.TunableView` implementations used with Fine-Tuning
Scheduler.

:class:`TunableViewConformance` extends the protocol's suite
(:class:`~finetuning_scheduler.tunable_view.conformance.ViewConformance`, whose cases hold for any consumer) with the
cases Fine-Tuning Scheduler's use of a view relies on: the view's default schedule compiles, thawing a compiled name
flips exactly its tensors, an optimizer step changes only thawed tensors, and schedules that name nothing, schedule a
frozen role, or use a view declaring ``trainable=False`` are refused. Which cases apply is decided by the view's own
support record: a view declaring ``trainable=False`` runs only the refusal case; role cases run only for the role kinds
a view declares; the round-trip case runs for views whose naming scheme is not ``"physical"``; the parametrization case
runs when the model has parametrized tensors. A run in which no case applied fails, so a suite cannot pass vacuously.

The cases need only PyTorch (no pytest, no inputs and no forward pass): the optimizer-step case fills each trainable
tensor's gradient with ones. :func:`pytest_params` turns the cases into ``pytest.mark.parametrize`` arguments:

.. code-block:: python

    @pytest.mark.parametrize("case", pytest_params(make_my_view), ids=lambda case: case.__name__)
    def test_my_view_conformance(case):
        case()

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.nn.utils import parametrize

from finetuning_scheduler.core.compile import compile_schedule, view_default_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.phase import thaw_params
from finetuning_scheduler.core.roles import is_frozen_role
from finetuning_scheduler.core.views import as_view, BoundView
from finetuning_scheduler.tunable_view.conformance import (ConformanceCase, ConformanceError, ConformanceReport,
                                                           ViewConformance)

NO_SUCH_NAME = "__fts_conformance_no_such_name__"


def _check(condition: bool, msg: str) -> None:
    if not condition:
        raise ConformanceError(msg)


def _freeze(root: torch.nn.Module) -> None:
    for p in root.parameters():
        p.requires_grad = False


def _trainable(root: torch.nn.Module) -> set[str]:
    return {n for n, p in root.named_parameters() if p.requires_grad}


def _has_frozen_parameter(view: BoundView) -> bool:
    """Whether a parameter (not only a buffer) has a role that is never thawed."""
    params = {n for n, _ in view.root.named_parameters()}
    return any(n in params and is_frozen_role(r) for n, r in view.roles().items())


class TunableViewConformance(ViewConformance):
    """Conformance cases for one view implementation: the protocol's cases and Fine-Tuning Scheduler's.

    Args:
        make_view: Returns a fresh view over a fresh model (cases change parameters and ``requires_grad``).
        sample_logical: The logical names the per-name cases exercise (all of the view's logical names by default).
        step_fn: Performs the optimizer-step case's step as ``step_fn(view, optimizer)``, for views whose tensors
            receive gradients only through a forward pass. By default, every trainable tensor's gradient is set to
            ones and the optimizer steps.
    """

    def __init__(self, make_view: Callable[[], Any], *, sample_logical: Sequence[str] | None = None,
                 step_fn: Callable[[Any, torch.optim.Optimizer], None] | None = None) -> None:
        super().__init__(make_view)
        self.sample_logical = list(sample_logical) if sample_logical is not None else None
        self.step_fn = step_fn

    def _sample(self, bound: BoundView) -> list[str]:
        if self.sample_logical is not None:
            return self.sample_logical
        return list(dict.fromkeys(n for n, _ in bound.logical_named_parameters()))

    def cases(self) -> list[ConformanceCase]:
        """Every case, in run order."""
        protocol = self.protocol_cases()
        trainable = lambda s, v: s.trainable  # noqa: E731
        return [
            protocol["mapping_completeness"],
            protocol["mapping_round_trip"],
            ConformanceCase("default_schedule_compiles", "storage", trainable, self._default_schedule_compiles),
            ConformanceCase("thaw_is_exact_and_deterministic", "thaw", trainable, self._thaw_exact),
            ConformanceCase("step_changes_only_thawed", "step", trainable, self._step_changes_only_thawed),
            protocol["checkpoint_round_trip"],
            ConformanceCase("unmatched_rule_refuses", "refusal", trainable, self._unmatched_rule_refuses),
            ConformanceCase("untrainable_refuses", "refusal", lambda s, v: not s.trainable, self._untrainable_refuses),
            protocol["roles_valid"],
            ConformanceCase("frozen_roles_refuse", "roles", lambda s, v: s.trainable and _has_frozen_parameter(v),
                            self._frozen_roles_refuse),
            protocol["promotion"],
            ConformanceCase("parametrization_mapping", "parametrization",
                            lambda s, v: s.trainable and any(parametrize.is_parametrized(m) for m in v.root.modules()),
                            self._parametrization_mapping),
        ]

    # ------------------------------------------------------------------------------------------------------------------
    # Fine-Tuning Scheduler's cases
    # ------------------------------------------------------------------------------------------------------------------
    def _default_schedule_compiles(self) -> None:
        """The view's own default schedule compiles: no two phases share storage."""
        view = self._fresh()
        compile_schedule(view_default_schedule(view), view)

    def _thaw_exact(self) -> None:
        """Compiling a logical name is deterministic, and thawing the result flips exactly those tensors.

        One view serves every name: each thaw starts from a frozen root, and determinism is checked by compiling a
        name twice on the same view, so a fresh view per name (a whole model and wrapper each) would add only cost.
        """
        view = self._fresh()
        for name in self._sample(view):
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
    return TunableViewConformance(make_view, **kwargs).pytest_params()


__all__ = ["ConformanceCase", "ConformanceError", "ConformanceReport", "pytest_params", "TunableViewConformance"]
