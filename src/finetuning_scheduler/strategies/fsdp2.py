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
Fine-Tuning Scheduler FSDP2 Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~finetuning_scheduler.strategies.base.DistributedAdapter` for a model with FSDP2 (``fully_shard``)
modules, alone or with tensor parallelism: its state is read and written through ``torch.distributed.checkpoint`` by
every rank, as for :class:`~finetuning_scheduler.strategies.dtensor.DTensorAdapter`. Its validation also reports the
FSDP units that reshard after forward and hold parameters thawing in different phases (see
:func:`~finetuning_scheduler.strategies.plan.phase_feedback`).

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Mapping, Sequence

from torch import nn

from finetuning_scheduler.strategies.base import Feedback
from finetuning_scheduler.strategies.dtensor import DTensorAdapter


class FSDP2Adapter(DTensorAdapter):
    """FSDP2 (``fully_shard``) modules anywhere in the model, checkpointed through DCP by every rank."""

    name = "fsdp2"

    def detect(self, model: nn.Module) -> bool:
        from torch.distributed.fsdp import FSDPModule

        return any(isinstance(m, FSDPModule) for m in model.modules())

    def validate(self, model: nn.Module, phases: Mapping[int, Sequence[str]]) -> list[Feedback]:
        from finetuning_scheduler.strategies.plan import phase_feedback

        feedback = super().validate(model, phases)
        if any(f.level == "error" for f in feedback):  # names the model lacks: what each unit trains is unknown
            return feedback
        return feedback + phase_feedback(model, phases)
