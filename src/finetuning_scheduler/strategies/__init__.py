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
Fine-Tuning Scheduler Distributed Adapters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Torch-native adapters for how a model's parameters are laid out across processes (see
:class:`~finetuning_scheduler.strategies.base.DistributedAdapter`). They import only PyTorch and the scheduling core.

.. warning:: This is an experimental API and may change in a future release.
"""
from finetuning_scheduler.strategies.base import DistributedAdapter, Feedback, select_distributed_adapter
from finetuning_scheduler.strategies.dtensor import DTensorAdapter
from finetuning_scheduler.strategies.fsdp2 import FSDP2Adapter
from finetuning_scheduler.strategies.unsharded import UnshardedAdapter

__all__ = ["DistributedAdapter", "DTensorAdapter", "Feedback", "FSDP2Adapter", "select_distributed_adapter",
           "UnshardedAdapter"]
