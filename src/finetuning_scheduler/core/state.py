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
Fine-Tuning Scheduler State
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework-neutral container for the Fine-Tuning Scheduler's internal state.

"""
from dataclasses import dataclass, field


@dataclass
class FTSState:
    """Dataclass to encapsulate the :class:`~finetuning_scheduler.fts.FinetuningScheduler` internal state."""

    _resume_fit_from_ckpt: bool = False
    _ft_epoch: int = 0
    _ft_global_steps: int = 0
    _curr_depth: int = 0
    _best_ckpt_depth: int = 0
    _ft_sync_props: tuple = (
        ("epoch_progress.current.completed", "_ft_epoch"),
        ("epoch_loop.global_step", "_ft_global_steps"),
    )
    _ft_sync_objects: tuple | None = None
    _ft_init_epoch: int | None = None
    _curr_thawed_params: list = field(default_factory=list)
    _fts_ckpt_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._fts_ckpt_metadata = {
            "current_ckpt_depth": self._curr_depth,
            "best_ckpt_depth": self._best_ckpt_depth,
            "best_ckpt_pgs": {},
        }
