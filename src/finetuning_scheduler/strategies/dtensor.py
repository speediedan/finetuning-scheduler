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
Fine-Tuning Scheduler DTensor Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~finetuning_scheduler.strategies.base.DistributedAdapter` for a model whose parameters are ``DTensor``\ s
without FSDP2 (tensor parallelism): its state is read and written through ``torch.distributed.checkpoint`` (DCP) and
its state-dict utilities, by every rank.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

from torch import nn
from torch.optim import Optimizer

from finetuning_scheduler.strategies.base import Feedback, mixed_dtensor_feedback, Quiesce


class DTensorAdapter:
    """``DTensor`` parameters (tensor parallelism), checkpointed through DCP by every rank."""

    name = "dtensor"
    sharded = True

    def detect(self, model: nn.Module) -> bool:
        return any(type(p).__name__ == "DTensor" for p in model.parameters())

    def validate(self, model: nn.Module, phases: Mapping[int, Sequence[str]]) -> list[Feedback]:
        return mixed_dtensor_feedback(model, phases)

    def state_dict(self, model: nn.Module, optimizer: Optimizer | None, *, exclude: Callable[[Any], set[str]],
                   quiesce: Quiesce = nullcontext) -> tuple[dict[str, Any], dict[str, Any] | None]:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_state_dict

        # DSD initializes an optimizer with no state by stepping it, which would change the live training state
        with quiesce():
            if optimizer is not None and optimizer.state:
                model_sd, optim_sd = get_state_dict(model, optimizer)
            else:
                model_sd, optim_sd = get_model_state_dict(model), None
        excluded = exclude(model_sd.keys())
        return {k: v for k, v in model_sd.items() if k not in excluded}, optim_sd

    def save(self, state: Mapping[str, Any], path: str) -> None:
        import torch.distributed.checkpoint as dcp

        dcp.save({k: v for k, v in state.items() if v is not None}, checkpoint_id=path)

    def load_state(self, path: str, model: nn.Module, optimizer: Optimizer | None, *, load_optimizer: bool,
                   exclude: Callable[[Any], set[str]], quiesce: Quiesce = nullcontext) -> None:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (get_model_state_dict, get_state_dict,
                                                             set_model_state_dict, set_state_dict, StateDictOptions)

        load_optimizer = load_optimizer and optimizer is not None
        if load_optimizer:
            # DSD initializes the optimizer's state only when no parameter has a gradient
            for p in model.parameters():
                p.grad = None
        unsaved: list[str] = []
        with quiesce():
            if load_optimizer:
                assert optimizer is not None
                model_sd, optim_sd = get_state_dict(model, optimizer)
                # parameters with no state in the checkpoint (never stepped) must get none, not DSD's zeros
                saved = set(dcp.FileSystemReader(path).read_metadata().state_dict_metadata)
                optim_state: dict = optim_sd["state"]  # type: ignore[assignment]
                unsaved = [fqn for fqn in optim_state if not any(k.startswith(f"optim.state.{fqn}.") for k in saved)]
                for fqn in unsaved:
                    del optim_state[fqn]
                state = {"model": model_sd, "optim": optim_sd}
            else:
                state = {"model": get_model_state_dict(model)}
            # excluded (instrument) tensors are never saved, so they are the only model keys a load may miss
            excluded = exclude(state["model"].keys())
            state["model"] = {k: v for k, v in state["model"].items() if k not in excluded}
            dcp.load(state, checkpoint_id=path)
            if load_optimizer:
                assert optimizer is not None
                # `dcp.load` has already required every saved key; relax the per-parameter checks only for what is
                # legitimately absent (excluded tensors, and optimizer state of never-stepped parameters)
                options = StateDictOptions(strict=not (unsaved or excluded))
                set_state_dict(model, optimizer, model_state_dict=state["model"],
                               optim_state_dict=state["optim"],  # type: ignore[arg-type]
                               options=options)
            else:
                set_model_state_dict(model, state["model"], options=StateDictOptions(strict=not excluded))
