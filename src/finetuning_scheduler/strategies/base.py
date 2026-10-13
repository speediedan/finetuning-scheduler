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

A :class:`DistributedAdapter` is the torch-native seam between scheduling and how a model's parameters are laid out
across processes: it recognizes a layout (a plain module, FSDP2, tensor parallelism), checks a compiled schedule
against it, and reads and writes the training state's tensors in the form that layout needs (one file, or a
``torch.distributed.checkpoint`` directory written by every rank).

.. warning:: This is an experimental API and may change in a future release.
"""
import os
from collections.abc import Callable, Mapping, Sequence
from pprint import pformat
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from torch import nn
from torch.optim import Optimizer

from finetuning_scheduler.core.exceptions import FTSConfigurationError

# a context the adapter enters around state-dict utilities that may step the optimizer to initialize its state
Quiesce = Callable[[], AbstractContextManager]


@dataclass(frozen=True)
class Feedback:
    """A finding from :meth:`DistributedAdapter.validate`.

    Attributes:
        level: ``"info"``, ``"warning"`` or ``"error"``.
        message: What was found, and what to do about it.
    """

    level: Literal["info", "warning", "error"]
    message: str


@runtime_checkable
class DistributedAdapter(Protocol):
    """How a model's parameters are laid out across processes, and how its training state is saved and loaded.

    Attributes:
        name: The adapter's name (``"unsharded"``, ``"fsdp2"``, ``"dtensor"``).
        sharded: Whether checkpoints are ``torch.distributed.checkpoint`` directories written by every rank (rather
            than one file written by global rank zero).
    """

    name: str
    sharded: bool

    def detect(self, model: nn.Module) -> bool:
        """Whether ``model`` has this adapter's layout."""
        ...

    def validate(self, model: nn.Module, phases: Mapping[int, Sequence[str]]) -> list[Feedback]:
        """Check each phase's physical parameter names (``phases``) against the model's layout."""
        ...

    def state_dict(self, model: nn.Module, optimizer: Optimizer | None, *, exclude: Callable[[Any], set[str]],
                   quiesce: Quiesce = nullcontext) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """The model's and (when it has state) the optimizer's state, leaving out the model keys ``exclude``
        returns for the model's state keys."""
        ...

    def save(self, state: Mapping[str, Any], path: str) -> None:
        """Write ``state`` (``{"model": ..., "optim": ...}``) to ``path``."""
        ...

    def load_state(self, path: str, model: nn.Module, optimizer: Optimizer | None, *, load_optimizer: bool,
                   exclude: Callable[[Any], set[str]], quiesce: Quiesce = nullcontext) -> None:
        """Load the state :meth:`save` wrote to ``path`` into ``model`` (and ``optimizer`` with
        ``load_optimizer``).

        Keys ``exclude`` returns were never saved and are the only model keys a load may
        miss.
        """
        ...


def select_distributed_adapter(model: nn.Module, explicit: DistributedAdapter | None = None) -> DistributedAdapter:
    """The adapter for ``model``: ``explicit`` when given, else the first of FSDP2, DTensor (tensor parallelism)
    and unsharded whose :meth:`~DistributedAdapter.detect` recognizes it.

    Raises:
        FTSConfigurationError: If ``explicit`` is an unsharded adapter that does not recognize ``model`` (a sharded
            model saved to one file would keep only one rank's shards). A sharded adapter reads and writes any layout.
    """
    if explicit is not None:
        if not explicit.sharded and not explicit.detect(model):
            raise FTSConfigurationError(
                f"The distributed adapter {type(explicit).__name__} ({explicit.name!r}) does not recognize this model's"
                " layout: its parameters are sharded, and an unsharded checkpoint would keep one rank's shards only."
                " Pass a sharded adapter (FSDP2Adapter or DTensorAdapter), or none to detect it.")
        return explicit
    from finetuning_scheduler.strategies.dtensor import DTensorAdapter
    from finetuning_scheduler.strategies.fsdp2 import FSDP2Adapter
    from finetuning_scheduler.strategies.unsharded import UnshardedAdapter

    for adapter in (FSDP2Adapter(), DTensorAdapter()):
        if adapter.detect(model):
            return adapter
    return UnshardedAdapter()


def unresolved_feedback(model: nn.Module, phases: Mapping[int, Sequence[str]]) -> list[Feedback]:
    """An error naming each phase's parameters the model no longer has (for example names a wrapper renamed after
    the schedule was validated, such as ``torch.compile``'s ``_orig_mod.``), which would otherwise thaw nothing."""
    names = {n for n, _ in model.named_parameters()}
    missing = {d: [n for n in phase if n not in names] for d, phase in phases.items()}
    missing = {d: ns for d, ns in missing.items() if ns}
    if not missing:
        return []
    detail = "; ".join(f"phase {d}: {ns}" for d, ns in missing.items())
    return [Feedback("error", f"The schedule names parameters the model does not have ({detail}). A wrapper applied"
                              " after the schedule was validated (in `configure_model`, for example) may have renamed"
                              " them.")]


def mixed_dtensor_feedback(model: nn.Module, phases: Mapping[int, Sequence[str]]) -> list[Feedback]:
    """Phases whose parameters mix ``DTensor`` and plain tensors, which some optimizer operations do not accept."""
    all_params = dict(model.named_parameters())
    mixed: dict[int, dict[str, str]] = {}
    for depth, names in phases.items():
        plain = [n for n in names if n in all_params and type(all_params[n]).__name__ != "DTensor"]
        if plain and len(plain) < len([n for n in names if n in all_params]):
            mixed[depth] = {n: type(all_params[n].data).__name__ for n in plain}
    if not mixed:
        return []
    detail = "".join(f"  Phase {d}:{os.linesep}    {pformat(names, indent=4)}{os.linesep}"
                     for d, names in mixed.items())
    return [Feedback("info", "\nThe current fine-tuning schedule and sharding plan produce mixed DTensor/Non-DTensor"
                             " parameter group(s) in at least one phase.\n"
                             "Be aware some optimizer operations may require you to either:\n"
                             "  1. allow implicit replication (experimental)\n"
                             "  2. shard the modules owning the optimized non-DTensor (usually ``torch.Tensor``)"
                             " parameters.\n"
                             "     (HINT: sharding your root model usually addresses this concern for all phases if"
                             " needed).\n\n"
                             f"Found Non-DTensor parameters in mixed parameter groups: \n{detail}")]
