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
Fine-Tuning Scheduler Unsharded Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~finetuning_scheduler.strategies.base.DistributedAdapter` for a model whose parameters are plain tensors
(one device, or a layer split across devices in one process): its state is one ``torch.save`` file.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

from torch import nn
from torch.optim import Optimizer

from finetuning_scheduler.core.checkpoint import load_payload, save_payload
from finetuning_scheduler.strategies.base import Feedback, Quiesce


class UnshardedAdapter:
    """Plain parameters, saved to one file (by global rank zero, under the controller)."""

    name = "unsharded"
    sharded = False

    def detect(self, model: nn.Module) -> bool:
        return not any(type(p).__name__ == "DTensor" for p in model.parameters())

    def validate(self, model: nn.Module, phases: Mapping[int, Sequence[str]]) -> list[Feedback]:
        return []

    def state_dict(self, model: nn.Module, optimizer: Optimizer | None, *, exclude: Callable[[Any], set[str]],
                   quiesce: Quiesce = nullcontext) -> tuple[dict[str, Any], dict[str, Any] | None]:
        model_sd = model.state_dict()
        excluded = exclude(model_sd.keys())
        return ({k: v for k, v in model_sd.items() if k not in excluded},
                optimizer.state_dict() if optimizer is not None else None)

    def save(self, state: Mapping[str, Any], path: str) -> None:
        save_payload(state, path)

    def load_state(self, path: str, model: nn.Module, optimizer: Optimizer | None, *, load_optimizer: bool,
                   exclude: Callable[[Any], set[str]], quiesce: Quiesce = nullcontext) -> None:
        state = load_payload(path)
        excluded = exclude(model.state_dict().keys())
        model.load_state_dict(state["model"], strict=not excluded)
        if load_optimizer and optimizer is not None and state.get("optim") is not None:
            optimizer.load_state_dict(state["optim"])
