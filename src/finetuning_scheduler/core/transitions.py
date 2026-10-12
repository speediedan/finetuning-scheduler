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
Fine-Tuning Scheduler Transitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Framework-neutral phase-transition logic: the plateau (early-stopping) criteria evaluated within a phase, the
composition of plateau and epoch-driven transitions, and the pluggable transition policies and phase gates the direct
PyTorch API composes.

.. warning::

    Experimental: the policy and gate types may change in minor releases until the PyTorch API stabilizes.

"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from finetuning_scheduler.core.dist import rank_zero_warn
from finetuning_scheduler.core.exceptions import FTSConfigurationError


@dataclass(frozen=True)
class PlateauOutcome:
    """The result of one plateau evaluation.

    Attributes:
        should_stop: Whether training should stop.
        reason: Why training should stop, or the improvement message, or ``None``.
        best_score: The best monitored value after this evaluation.
        wait_count: Evaluations without improvement after this one, before any phase-transition reset.
        phase_complete: Whether patience ran out in a phase that is not the final one, so the phase should end.
    """

    should_stop: bool
    reason: str | None
    best_score: Tensor
    wait_count: int
    phase_complete: bool


def plateau_criteria(
    current: Tensor,
    *,
    best_score: Tensor,
    wait_count: int,
    patience: int,
    min_delta: float,
    monitor_op: Callable[[Tensor, Tensor], Tensor],
    monitor: str,
    order: str,
    final_phase: bool,
    check_finite: bool,
    stopping_threshold: float | None,
    divergence_threshold: float | None,
    improvement_message: Callable[[Tensor], str],
    stop_message: str = "Signaling Trainer to stop.",
) -> PlateauOutcome:
    """Evaluate the plateau (early-stopping) criteria for one monitored value.

    Mirrors Lightning's ``EarlyStopping`` criteria, except that running out of patience in a phase that is not the
    final one completes the phase rather than stopping training.

    Args:
        current: The monitored value.
        best_score: The best value so far.
        wait_count: Evaluations without improvement so far.
        patience: Evaluations without improvement that end a phase (or, in the final phase, training).
        min_delta: The signed minimum change that counts as an improvement (negative when minimizing).
        monitor_op: The comparison defining improvement (``torch.lt`` when minimizing, ``torch.gt`` when maximizing).
        monitor: The monitored metric's name, for messages.
        order: The comparison's symbol (``"<"`` or ``">"``), for messages.
        final_phase: Whether the current phase is the final scheduled phase.
        check_finite: Whether a non-finite value stops training.
        stopping_threshold: Stop once the value improves past this threshold.
        divergence_threshold: Stop once the value is worse than this threshold.
        improvement_message: Builds the improvement message from ``current``, before ``best_score`` changes.
        stop_message: The sentence ending a stop reason; the default is the Lightning callback's.

    Returns:
        The :class:`PlateauOutcome`.
    """
    should_stop, phase_complete, reason = False, False, None
    if check_finite and not torch.isfinite(current):
        should_stop = True
        reason = (
            f"Monitored metric {monitor} = {current} is not finite."
            f" Previous best value was {best_score:.3f}. {stop_message}"
        )
    elif stopping_threshold is not None and monitor_op(current, stopping_threshold):  # type: ignore[arg-type]
        should_stop = True
        reason = (
            f"Stopping threshold reached: {monitor} = {current} {order} {stopping_threshold}."
            f" {stop_message}"
        )
    elif divergence_threshold is not None and monitor_op(-current, -divergence_threshold):  # type: ignore[arg-type]
        should_stop = True
        reason = (
            f"Divergence threshold reached: {monitor} = {current} {order} {divergence_threshold}."
            f" {stop_message}"
        )
    elif monitor_op(current - min_delta, best_score.to(current.device)):
        reason = improvement_message(current)
        best_score, wait_count = current, 0
    else:
        wait_count += 1
        if wait_count >= patience:
            if final_phase:
                should_stop = True
                reason = (
                    f"Monitored metric {monitor} did not improve in the last {wait_count} records."
                    f" Best score: {best_score.item():.3f}. {stop_message}"
                )
            else:
                phase_complete = True
    return PlateauOutcome(should_stop, reason, best_score, wait_count, phase_complete)


def current_max_epoch(schedule: dict, curr_depth: int, depth_remaining: int, max_epochs: int | float) -> int | float:
    """The epoch bounding the current phase: its ``max_transition_epoch``, or the run's ``max_epochs`` in the final
    phase."""
    return schedule[curr_depth]["max_transition_epoch"] if depth_remaining > 0 else max_epochs


def epoch_transition_due(curr_max_epoch: int | float, current_epoch: int) -> bool:
    """Whether an epoch-driven transition is due (a negative ``curr_max_epoch`` disables epoch-driven
    transitions)."""
    return 0 <= curr_max_epoch <= current_epoch


def compose_transition(plateau_phase_complete: bool, final_phase: bool, epoch_due: bool) -> bool:
    """Compose plateau and epoch-driven transitions: either ends a phase, except that an epoch-driven transition never
    ends the final phase."""
    return plateau_phase_complete or (not final_phase and epoch_due)


class Decision(IntEnum):
    """A transition policy's verdict for one unit (epoch or step), ordered by precedence: the maximum wins when
    policies are composed and when ranks are reduced."""

    CONTINUE = 0
    ADVANCE = 1
    STOP = 2
    VETO = 3


@dataclass(frozen=True)
class TransitionContext:
    """What a transition policy sees at the end of a unit.

    Attributes:
        depth: The current phase.
        max_depth: The final phase.
        is_final_phase: Whether ``depth`` is the final phase.
        unit: The unit that just ended, ``"epoch"`` or ``"step"``.
        epoch: The number of completed epochs, i.e. the index of the next epoch.
        global_step: The number of completed optimizer steps.
        phase_start_epoch: ``epoch`` when the current phase began.
        phase_start_step: ``global_step`` when the current phase began.
        max_transition_epoch: The phase's ``max_transition_epoch`` (negative when unset).
        max_transition_step: The phase's ``max_transition_step`` (negative when unset).
        max_epochs: The run's epoch limit, if known.
        metrics: The metrics reported for this unit.
        phase_config: The phase's validated payload, if any.
        metadata: The run's user metadata.
    """

    depth: int
    max_depth: int
    is_final_phase: bool
    unit: Literal["epoch", "step"]
    epoch: int
    global_step: int
    phase_start_epoch: int
    phase_start_step: int
    max_transition_epoch: int
    max_transition_step: int
    max_epochs: int | None = None
    metrics: Mapping[str, Tensor] = field(default_factory=dict)
    phase_config: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


PolicyResult = Decision | tuple[Decision, str | None]


@runtime_checkable
class TransitionPolicy(Protocol):
    """Decides, at the end of each unit it observes, whether to continue, advance, stop or veto."""

    granularity: frozenset[str]

    def evaluate(self, ctx: TransitionContext) -> PolicyResult: ...

    def reset(self, depth: int) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...

    def config(self) -> dict[str, Any]: ...


def as_decision(result: PolicyResult | bool) -> tuple[Decision, str | None]:
    """Normalize a policy's result (a ``Decision``, a ``(Decision, reason)`` pair or a ``bool`` meaning
    advance)."""
    if isinstance(result, tuple):
        return Decision(result[0]), result[1]
    if isinstance(result, bool):
        return (Decision.ADVANCE if result else Decision.CONTINUE), None
    return Decision(result), None


class _StatelessPolicy:
    granularity: frozenset[str] = frozenset()

    def reset(self, depth: int) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        pass

    def config(self) -> dict[str, Any]:
        return {"class_path": f"{type(self).__module__}.{type(self).__qualname__}"}


class EpochTransition(_StatelessPolicy):
    """Advance once the phase's ``max_transition_epoch`` is reached; never ends the final phase."""

    granularity = frozenset({"epoch"})

    def evaluate(self, ctx: TransitionContext) -> PolicyResult:
        if not ctx.is_final_phase and epoch_transition_due(ctx.max_transition_epoch, ctx.epoch):
            return Decision.ADVANCE, f"max_transition_epoch ({ctx.max_transition_epoch}) reached"
        return Decision.CONTINUE


class StepTransition(_StatelessPolicy):
    """Advance once the phase's ``max_transition_step`` (a global step count) is reached; never ends the final
    phase."""

    granularity = frozenset({"step"})

    def evaluate(self, ctx: TransitionContext) -> PolicyResult:
        if not ctx.is_final_phase and 0 <= ctx.max_transition_step <= ctx.global_step:
            return Decision.ADVANCE, f"max_transition_step ({ctx.max_transition_step}) reached"
        return Decision.CONTINUE


class PlateauTransition:
    """Advance when the monitored metric stops improving; in the final phase, stop.

    Mirrors Lightning's ``EarlyStopping`` arguments and semantics (including the ``min_delta`` sign convention). As in
    the Lightning integration, a phase transition resets the patience count but keeps the best score.

    Args:
        monitor: The metric to monitor.
        mode: ``"min"`` or ``"max"``.
        patience: Evaluations without improvement that end a phase.
        min_delta: The minimum change that counts as an improvement.
        check_finite: Stop when the metric is not finite.
        stopping_threshold: Stop once the metric improves past this value.
        divergence_threshold: Stop once the metric is worse than this value.
        strict: Raise if the metric is missing; otherwise continue with a warning.
        granularity: Evaluate at the end of each ``"epoch"`` or each ``"step"``.
    """

    order_dict = {"min": "<", "max": ">"}

    def __init__(self, monitor: str, mode: Literal["min", "max"] = "min", patience: int = 3, min_delta: float = 0.0,
                 check_finite: bool = True, stopping_threshold: float | None = None,
                 divergence_threshold: float | None = None, strict: bool = True,
                 granularity: Literal["epoch", "step"] = "epoch") -> None:
        if mode not in self.order_dict:
            raise FTSConfigurationError(f"`mode` can be {', '.join(self.order_dict)}, got {mode}")
        self.monitor, self.mode, self.patience = monitor, mode, patience
        self.monitor_op = torch.lt if mode == "min" else torch.gt
        self.min_delta = abs(min_delta) * (1 if self.monitor_op == torch.gt else -1)
        self.check_finite, self.strict = check_finite, strict
        self.stopping_threshold, self.divergence_threshold = stopping_threshold, divergence_threshold
        self.granularity = frozenset({granularity})
        self.wait_count = 0
        self.best_score = torch.tensor(torch.inf if mode == "min" else -torch.inf)
        self._warned_missing = False

    def _improvement_message(self, current: Tensor) -> str:
        if torch.isfinite(self.best_score):
            return (f"Metric {self.monitor} improved by {abs(self.best_score - current).item():.3f} >="
                    f" min_delta = {abs(self.min_delta)}. New best score: {current.item():.3f}")
        return f"Metric {self.monitor} improved. New best score: {current.item():.3f}"

    def evaluate(self, ctx: TransitionContext) -> PolicyResult:
        if self.monitor not in ctx.metrics:
            msg = f"PlateauTransition is monitoring `{self.monitor}`, which was not reported for this {ctx.unit}."
            if self.strict:
                raise FTSConfigurationError(msg + " Report it, or set `strict=False`.")
            if not self._warned_missing:
                self._warned_missing = True
                rank_zero_warn(msg + " Continuing.")
            return Decision.CONTINUE
        current = torch.as_tensor(ctx.metrics[self.monitor], dtype=torch.float32).detach().squeeze().cpu()
        outcome = plateau_criteria(
            current, best_score=self.best_score, wait_count=self.wait_count, patience=self.patience,
            min_delta=self.min_delta, monitor_op=self.monitor_op, monitor=self.monitor,
            order=self.order_dict[self.mode], final_phase=ctx.is_final_phase, check_finite=self.check_finite,
            stopping_threshold=self.stopping_threshold, divergence_threshold=self.divergence_threshold,
            improvement_message=self._improvement_message, stop_message="Stopping training.",
        )
        self.best_score, self.wait_count = outcome.best_score, outcome.wait_count
        if outcome.should_stop:
            return Decision.STOP, outcome.reason
        if outcome.phase_complete:
            return Decision.ADVANCE, f"{self.monitor} did not improve in the last {self.wait_count} records"
        return Decision.CONTINUE, outcome.reason

    def reset(self, depth: int) -> None:
        self.wait_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {"wait_count": self.wait_count, "best_score": self.best_score.clone()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.best_score = torch.as_tensor(state_dict["best_score"])

    def config(self) -> dict[str, Any]:
        return {"class_path": f"{type(self).__module__}.{type(self).__qualname__}",
                "init_args": {"monitor": self.monitor, "mode": self.mode, "patience": self.patience,
                              "min_delta": abs(self.min_delta), "check_finite": self.check_finite,
                              "stopping_threshold": self.stopping_threshold,
                              "divergence_threshold": self.divergence_threshold, "strict": self.strict,
                              "granularity": next(iter(self.granularity))}}


class AnyOf:
    """Compose policies: each unit's decision is the maximum over the policies observing that unit."""

    def __init__(self, *policies: TransitionPolicy) -> None:
        self.policies = list(policies)
        self.granularity = frozenset().union(*(p.granularity for p in policies))

    def evaluate(self, ctx: TransitionContext) -> PolicyResult:
        best, reason = Decision.CONTINUE, None
        for policy in self.policies:
            if ctx.unit not in policy.granularity:
                continue
            decision, why = as_decision(policy.evaluate(ctx))
            if decision > best or (decision == best and reason is None):
                best, reason = decision, why
        return best, reason

    def reset(self, depth: int) -> None:
        for p in self.policies:
            p.reset(depth)

    def state_dict(self) -> dict[str, Any]:
        return {str(i): p.state_dict() for i, p in enumerate(self.policies)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        for i, p in enumerate(self.policies):
            p.load_state_dict(state_dict.get(str(i), {}))

    def config(self) -> dict[str, Any]:
        return {"class_path": f"{type(self).__module__}.{type(self).__qualname__}",
                "policies": [p.config() for p in self.policies]}


class CallableTransition(_StatelessPolicy):
    """Wrap a function of the :class:`TransitionContext` returning a ``Decision``, a ``(Decision, reason)`` pair or
    a ``bool`` (``True`` meaning advance)."""

    def __init__(self, fn: Callable[[TransitionContext], PolicyResult | bool],
                 granularity: Literal["epoch", "step"] = "epoch", name: str | None = None) -> None:
        self.fn, self.name = fn, name
        self.granularity = frozenset({granularity})

    def evaluate(self, ctx: TransitionContext) -> PolicyResult:
        return as_decision(self.fn(ctx))

    def config(self) -> dict[str, Any]:
        fn = self.fn
        return {"class_path": f"{type(self).__module__}.{type(self).__qualname__}",
                "fn": f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}", "name": self.name}


@dataclass(frozen=True)
class GateEvent:
    """What a gate sees at a phase boundary, after the phase's best checkpoint has been restored.

    Attributes:
        depth: The phase being closed.
        final: Whether it is the final phase (the run is ending).
        best_score: The best monitored value, if a checkpoint policy tracks one.
        best_path: The best checkpoint's path, if any.
        best_depth: The phase the best checkpoint came from, if any.
        metrics: The metrics of the last unit.
        phase_config: The phase's validated payload, if any.
        metadata: The run's user metadata.
    """

    depth: int
    final: bool
    best_score: Tensor | None = None
    best_path: str | None = None
    best_depth: int | None = None
    metrics: Mapping[str, Tensor] = field(default_factory=dict)
    phase_config: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateResult:
    """A gate's verdict: accept the phase, or veto it (the run stops and the previous accepted phase's best is
    restored).

    Attributes:
        accept: Whether the phase is accepted.
        record: Details recorded in the run's metadata and checkpoints.
        reason: Why, for logs and the stop reason.
    """

    accept: bool
    record: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None
