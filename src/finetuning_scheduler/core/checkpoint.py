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
Fine-Tuning Scheduler Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What the direct PyTorch API needs from checkpointing: a policy that tracks the best monitored value and saves the
training state when it improves, pins the best as of each phase boundary (so a vetoed phase can be rolled back), and
reads and writes checkpoint payloads.

.. warning::

    Experimental: the checkpoint policy protocol may change in minor releases until the direct PyTorch API
    stabilizes. Checkpoints are written with ``torch.save`` and read with ``weights_only=False``: load only checkpoints
    you trust.

"""
import os
import shutil
from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from finetuning_scheduler.core.dist import rank_zero_warn


def save_payload(payload: Mapping[str, Any], path: str | os.PathLike) -> str:
    """Write a checkpoint payload to ``path`` (creating its directory) and return the path."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(dict(payload), path)
    return path


def load_payload(path: str | os.PathLike) -> dict[str, Any]:
    """Read a checkpoint payload written by :func:`save_payload` onto the CPU."""
    return torch.load(os.fspath(path), map_location="cpu", weights_only=False)


@runtime_checkable
class CheckpointPolicy(Protocol):
    """Decides when the controller saves the training state, and where the best checkpoints are.

    The controller calls :meth:`observe` with each epoch's metrics and saves the training state to
    :meth:`path_for` ``("best")`` when it returns ``True``; it restores from :attr:`best_path` at phase boundaries
    when ``restore_best`` is set, and from :meth:`boundary_path` when a gate vetoes a phase.
    """

    monitor: str | None

    def observe(self, metrics: Mapping[str, Any], depth: int) -> bool: ...

    def path_for(self, tag: str) -> str: ...

    @property
    def best_path(self) -> str | None: ...

    @property
    def best_depth(self) -> int | None: ...

    @property
    def best_score(self) -> Tensor | None: ...

    def pin_boundary(self, depth: int, initial_path: str | None = None) -> None: ...

    def boundary_path(self, depth: int) -> str | None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...

    def config(self) -> dict[str, Any]: ...


class BestCheckpoint:
    """Keep the single best checkpoint by a monitored metric, as Lightning's ``ModelCheckpoint(save_top_k=1)``
    does: the first observation always saves, later ones only on strict improvement, and a non-finite value counts
    as the worst possible.

    It also pins the best as of each phase's start (copied to ``boundary.pt``, so at most two files are kept), which a
    gate's veto rolls back to.

    Args:
        dirpath: Where checkpoints are written.
        monitor: The metric to monitor.
        mode: ``"min"`` or ``"max"``.
        include: Save every parameter (``"all"``) or only the scheduled ones plus buffers (``"scheduled"``), so a
            large frozen base is not rewritten on every save.
        save_initial: Also save the state at setup, so a veto of the first trained phase has something to roll back to.
        metadata: Extra metadata stored in every checkpoint.
    """

    def __init__(self, dirpath: str | os.PathLike, monitor: str, mode: Literal["min", "max"] = "min",
                 include: Literal["all", "scheduled"] = "all", save_initial: bool = False,
                 metadata: Mapping[str, Any] | None = None) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"`mode` can be 'min' or 'max', got {mode}")
        if include not in ("all", "scheduled"):
            raise ValueError(f"`include` can be 'all' or 'scheduled', got {include}")
        self.dirpath = os.fspath(dirpath)
        self.monitor, self.mode, self.include = monitor, mode, include
        self.save_initial = save_initial
        self.metadata = dict(metadata or {})
        self._best_score: Tensor | None = None
        self._best_depth: int | None = None
        self._best_path: str | None = None
        self._boundaries: dict[int, tuple[str | None, int | None]] = {}
        self._warned_missing = False

    def observe(self, metrics: Mapping[str, Any], depth: int) -> bool:
        """Record ``metrics``' monitored value; return whether it is a new best (the caller then saves)."""
        if self.monitor not in metrics:
            if not self._warned_missing:
                self._warned_missing = True
                rank_zero_warn(f"BestCheckpoint is monitoring `{self.monitor}`, which was not reported; nothing saved.")
            return False
        current = torch.as_tensor(metrics[self.monitor], dtype=torch.float32).detach().squeeze().cpu()
        if not torch.isfinite(current):
            current = torch.tensor(torch.inf if self.mode == "min" else -torch.inf)
        op = torch.lt if self.mode == "min" else torch.gt
        if self._best_score is not None and not bool(op(current, self._best_score)):
            return False
        self._best_score, self._best_depth = current, depth
        self._best_path = self.path_for("best")
        return True

    def path_for(self, tag: str) -> str:
        return os.path.join(self.dirpath, f"{tag}.pt")

    @property
    def best_path(self) -> str | None:
        return self._best_path

    @property
    def best_depth(self) -> int | None:
        return self._best_depth

    @property
    def best_score(self) -> Tensor | None:
        return self._best_score

    def pin_boundary(self, depth: int, initial_path: str | None = None) -> None:
        """Pin the current best as the rollback point for phase ``depth`` (copying it, since the best is
        overwritten as training continues).

        Before any best exists, ``initial_path`` (a state saved at setup) is
        pinned instead.
        """
        source, source_depth = self._best_path, self._best_depth
        if source is None and initial_path is not None:
            source, source_depth = initial_path, depth
        pinned = None
        if source is not None and os.path.exists(source):
            pinned = self.path_for("boundary")
            if os.path.abspath(source) != os.path.abspath(pinned):
                shutil.copyfile(source, pinned)
        self._boundaries = {depth: (pinned, source_depth)}

    def boundary_path(self, depth: int) -> str | None:
        return self._boundaries.get(depth, (None, None))[0]

    def state_dict(self) -> dict[str, Any]:
        return {"best_score": self._best_score, "best_depth": self._best_depth, "best_path": self._best_path,
                "boundaries": dict(self._boundaries)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._best_score, self._best_depth = state_dict["best_score"], state_dict["best_depth"]
        self._best_path, self._boundaries = state_dict["best_path"], dict(state_dict["boundaries"])

    def config(self) -> dict[str, Any]:
        return {"class_path": f"{type(self).__module__}.{type(self).__qualname__}",
                "init_args": {"monitor": self.monitor, "mode": self.mode, "include": self.include,
                              "save_initial": self.save_initial, "metadata": self.metadata}}
