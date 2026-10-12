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

Framework-neutral phase-transition logic: the plateau (early-stopping) criteria evaluated within a phase.

"""
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


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

    Returns:
        The :class:`PlateauOutcome`.
    """
    should_stop, phase_complete, reason = False, False, None
    if check_finite and not torch.isfinite(current):
        should_stop = True
        reason = (
            f"Monitored metric {monitor} = {current} is not finite."
            f" Previous best value was {best_score:.3f}. Signaling Trainer to stop."
        )
    elif stopping_threshold is not None and monitor_op(current, stopping_threshold):  # type: ignore[arg-type]
        should_stop = True
        reason = (
            f"Stopping threshold reached: {monitor} = {current} {order} {stopping_threshold}."
            " Signaling Trainer to stop."
        )
    elif divergence_threshold is not None and monitor_op(-current, -divergence_threshold):  # type: ignore[arg-type]
        should_stop = True
        reason = (
            f"Divergence threshold reached: {monitor} = {current} {order} {divergence_threshold}."
            " Signaling Trainer to stop."
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
                    f" Best score: {best_score.item():.3f}. Signaling Trainer to stop."
                )
            else:
                phase_complete = True
    return PlateauOutcome(should_stop, reason, best_score, wait_count, phase_complete)
