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
Fine-Tuning Scheduler Phase Executors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler` carries out a phase once it has decided on it:
thawing the phase's parameters and configuring the optimizer and lr schedulers for them, resuming at a depth, and
finishing a transition.

The tuning scheduler decides *when* phases change (transition policies, gates, best-checkpoint restores); an executor
decides *how* a phase is put in place. The default, :class:`TuningSchedulerExecutor`, is the tuning scheduler's own
implementation. A training interface that already implements phase execution (the Lightning callback, whose strategy
adapters customize it) can supply its own executor, so the tuning scheduler drives the same calls the interface made
before.

.. warning:: This is an experimental API and may change in a future release.
"""
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from finetuning_scheduler.core.tuning_scheduler import TuningScheduler


@runtime_checkable
class PhaseExecutor(Protocol):
    """Puts the phases a tuning scheduler decides on in place."""

    def configure_phase(self, depth: int, restored_depth: int, depth_sync: bool, pre_state: Any) -> None:
        """Thaw phase ``depth``'s parameters and configure the optimizer and lr schedulers for them.

        Args:
            depth: The phase to configure.
            restored_depth: The depth of the state the optimizer holds (the best checkpoint's, after a restore).
            depth_sync: Configure every phase after ``restored_depth``, not only ``depth``.
            pre_state: The lr state captured before any restore, re-applied to the existing parameter groups.
        """
        ...

    def resume_at(self, depth: int) -> None:
        """Thaw every phase through ``depth`` without configuring the optimizer, whose parameter groups a resumed
        state already holds."""
        ...

    def after_transition(self, depth: int) -> None:
        """Finish a transition to ``depth`` (or a resume at it), after its phase is configured."""
        ...


class TuningSchedulerExecutor:
    """The tuning scheduler's own phase execution.

    Args:
        tuning_scheduler: The tuning scheduler whose phases it executes.
    """

    def __init__(self, tuning_scheduler: "TuningScheduler") -> None:
        self.tuning_scheduler = tuning_scheduler

    def configure_phase(self, depth: int, restored_depth: int, depth_sync: bool, pre_state: Any) -> None:
        self.tuning_scheduler._configure_phase(depth, restored_depth, depth_sync=depth_sync, pre_state=pre_state)

    def resume_at(self, depth: int) -> None:
        for d in range(depth + 1):
            self.tuning_scheduler._thaw_phase(d)

    def after_transition(self, depth: int) -> None:
        """Nothing: the tuning scheduler's phase configuration is complete."""


__all__ = ["TuningSchedulerExecutor", "PhaseExecutor"]
