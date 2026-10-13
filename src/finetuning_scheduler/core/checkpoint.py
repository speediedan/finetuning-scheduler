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

What the PyTorch API needs from checkpointing: a policy that keeps the best checkpoints by a monitored value, pins the
best as of each phase boundary (so a vetoed phase can be rolled back), and reads and writes checkpoint payloads.

.. warning::

    Experimental: the checkpoint policy protocol may change in minor releases until the PyTorch API
    stabilizes. Checkpoints are written with ``torch.save`` and read with ``weights_only=False``: load only checkpoints
    you trust.

"""
import os
import shutil
import uuid
from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from finetuning_scheduler.core.dist import rank_zero_warn


def save_payload(payload: Mapping[str, Any], path: str | os.PathLike) -> str:
    """Write a checkpoint payload to ``path`` (creating its directory) and return the path.

    The payload is written
    beside ``path`` and then moved over it, so an interrupted write never destroys an existing checkpoint.
    """
    path = os.path.normpath(os.fspath(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return path


def load_payload(path: str | os.PathLike) -> dict[str, Any]:
    """Read a checkpoint payload written by :func:`save_payload` onto the CPU."""
    return torch.load(os.fspath(path), map_location="cpu", weights_only=False)


def remove_checkpoint(path: str) -> None:
    """Remove a checkpoint file or directory, if there is one."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def replace_checkpoint(source: str, target: str) -> None:
    """Move a freshly written checkpoint file or directory over ``target`` (whatever ``target`` holds).

    An existing
    ``target`` is first moved aside and removed only once ``source`` is in place, so an interruption leaves either the
    old checkpoint or the new one at ``target`` (or, at worst, the old one at ``<target>.old``).
    """
    source, target = os.path.normpath(source), os.path.normpath(target)
    if not os.path.lexists(target):
        os.replace(source, target)
        return
    old = f"{target}.old"
    remove_checkpoint(old)
    os.replace(target, old)
    os.replace(source, target)
    remove_checkpoint(old)


def copy_checkpoint(source: str, target: str) -> None:
    """Copy a checkpoint file, or a sharded checkpoint's directory, to ``target`` (replacing it, whether it is a
    file or a directory).

    The copy is made beside ``target`` first, so ``target`` is replaced only by a complete
    copy.
    """
    source, target = os.path.normpath(source), os.path.normpath(target)
    tmp = f"{target}.tmp"
    remove_checkpoint(tmp)  # also clears a copy an interrupted earlier run left behind
    if os.path.isdir(source):
        shutil.copytree(source, tmp)
    else:
        shutil.copyfile(source, tmp)
    replace_checkpoint(tmp, target)


@runtime_checkable
class CheckpointPolicy(Protocol):
    """Decides when the controller saves the training state, and where the best checkpoints are.

    The controller calls :meth:`observe` with each epoch's metrics and saves the training state to
    :meth:`path_for` ``("best")`` when it returns ``True``; it restores from :attr:`best_path` at phase boundaries
    when ``restore_best`` is set, and from :meth:`boundary_path` when a gate vetoes a phase.

    A policy may also define these methods, which the controller calls when they exist:

    - ``prune(remove: bool)``, after each checkpoint write and boundary pin, to remove the checkpoints the policy no
      longer needs (``remove`` is ``True`` on the one rank that removes files);
    - ``mark_resume_point(remove: bool)``, after :meth:`~finetuning_scheduler.core.controller.FTSController.save` and
      after a saved state is loaded, to keep the checkpoints that state refers to;
    - ``verify_resume()``, when a saved state is loaded, to raise ``FileNotFoundError`` if a checkpoint the state
      needs no longer exists.
    """

    monitor: str | None

    def observe(self, metrics: Mapping[str, Any], depth: int) -> bool:
        """Record an epoch's ``metrics`` at phase ``depth``; return whether to save a checkpoint for them."""
        ...

    def path_for(self, tag: str) -> str:
        """The path to write the checkpoint named ``tag`` to: ``"best"`` (the one :meth:`observe` just accepted) or
        ``"initial"``."""
        ...

    @property
    def best_path(self) -> str | None:
        """The best checkpoint's path, or ``None`` before the first save."""
        ...

    @property
    def best_depth(self) -> int | None:
        """The phase depth at which the best checkpoint was saved."""
        ...

    @property
    def best_score(self) -> Tensor | None:
        """The best monitored value so far."""
        ...

    def pin_boundary(self, depth: int, initial_path: str | None = None, copy: bool = True) -> None:
        """Pin the current best, or ``initial_path`` before any best, as phase ``depth``'s rollback point."""
        ...

    def boundary_path(self, depth: int) -> str | None:
        """The checkpoint a gate's veto of phase ``depth`` rolls back to, or ``None`` if none was pinned."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """The policy's own state, saved with the training state."""
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the policy's state from :meth:`state_dict`'s output."""
        ...

    def config(self) -> dict[str, Any]:
        """The policy's settings, recorded in each checkpoint and compared on resume."""
        ...


class BestCheckpoint:
    """Keep the ``save_top_k`` best checkpoints by a monitored metric, as Lightning's ``ModelCheckpoint`` does:
    while fewer than ``save_top_k`` are kept every observation saves; after that, only one strictly better than the
    worst kept, which it replaces. A non-finite value counts as the worst possible. With the default
    ``save_top_k=1``, the first observation saves and later ones only on strict improvement.

    Every checkpoint gets a new, unique name and is never rewritten, so a saved training state always refers to the
    checkpoints it was saved with. The best as of each phase's start is pinned by reference as the rollback point for a
    gate's veto, and kept until the next phase starts. A sharded model's checkpoints are directories.

    When resuming, the checkpoints the latest :meth:`~finetuning_scheduler.core.controller.FTSController.save` refers
    to are kept until the next save, so a resume from the latest save always finds them. An older save can be resumed
    while its checkpoints remain: if its best or its rollback point has been removed,
    :meth:`~finetuning_scheduler.core.controller.FTSController.load` raises ``FileNotFoundError``, and any other kept
    checkpoint that has been removed is dropped with a warning. Checkpoints that an interrupted run wrote after the
    save being resumed are left in place.

    Args:
        dirpath: Where checkpoints are written.
        monitor: The metric to monitor.
        mode: ``"min"`` or ``"max"``.
        save_top_k: How many of the best checkpoints to keep, or ``-1`` to keep every one.
        include: Save every parameter (``"all"``) or only the scheduled ones plus buffers (``"scheduled"``), so a
            large frozen base is not rewritten on every save (single-process checkpoints; sharded checkpoints are
            written by DCP and hold every shard).
        save_initial: Also save the state at setup, so a veto of the first trained phase has something to roll back to.
        metadata: Extra metadata stored in every checkpoint.
    """

    def __init__(self, dirpath: str | os.PathLike, monitor: str, mode: Literal["min", "max"] = "min",
                 save_top_k: int = 1, include: Literal["all", "scheduled"] = "all", save_initial: bool = False,
                 metadata: Mapping[str, Any] | None = None) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"`mode` can be 'min' or 'max', got {mode}")
        if include not in ("all", "scheduled"):
            raise ValueError(f"`include` can be 'all' or 'scheduled', got {include}")
        if isinstance(save_top_k, bool) or not isinstance(save_top_k, int) or save_top_k == 0 or save_top_k < -1:
            raise ValueError(f"`save_top_k` can be a positive integer or -1 (keep every checkpoint), got {save_top_k}")
        self.dirpath = os.fspath(dirpath)
        self.monitor, self.mode, self.save_top_k, self.include = monitor, mode, save_top_k, include
        self.save_initial = save_initial
        self.metadata = dict(metadata or {})
        self._entries: list[tuple[str, Tensor, int]] = []  # the kept checkpoints in saving order: (path, score, depth)
        self._seq = 0
        self._boundaries: dict[int, tuple[str | None, int | None]] = {}
        # this process's view only: every checkpoint the policy has named or referred to, and those a resume needs
        self._known: set[str] = set()
        self._protected: set[str] = set()
        self._warned_missing = False

    def observe(self, metrics: Mapping[str, Any], depth: int) -> bool:
        """Record ``metrics``' monitored value; return whether it is kept (the caller then saves it to
        :meth:`path_for` ``("best")``)."""
        if self.monitor not in metrics:
            if not self._warned_missing:
                self._warned_missing = True
                rank_zero_warn(f"BestCheckpoint is monitoring `{self.monitor}`, which was not reported; nothing saved.")
            return False
        current = torch.as_tensor(metrics[self.monitor], dtype=torch.float32).detach().squeeze().cpu()
        if not torch.isfinite(current):
            current = torch.tensor(torch.inf if self.mode == "min" else -torch.inf)
        if self.save_top_k != -1 and len(self._entries) >= self.save_top_k:
            worst = self._worst()
            better = torch.lt if self.mode == "min" else torch.gt
            if not bool(better(current, worst[1])):
                return False
            self._entries.remove(worst)  # its file is removed by `prune`, once the new checkpoint is written
        self._seq += 1
        self._entries.append((self._unique_path(f"best-{self._seq}"), current, depth))
        return True

    def _unique_path(self, stem: str) -> str:
        path = os.path.join(self.dirpath, f"{stem}-{uuid.uuid4().hex[:8]}.pt")
        self._known.add(path)
        return path

    def _worst(self) -> tuple[str, Tensor, int]:
        pick = max if self.mode == "min" else min  # the first saved of equals, as Lightning evicts
        return pick(self._entries, key=lambda e: float(e[1]))

    def _best(self) -> tuple[str, Tensor, int] | None:
        pick = min if self.mode == "min" else max
        return pick(self._entries, key=lambda e: float(e[1])) if self._entries else None

    def path_for(self, tag: str) -> str:
        """The path to write the checkpoint ``tag`` to: for ``"best"``, the one :meth:`observe` just kept; for any
        other tag (``"initial"``), a new unique path."""
        if tag == "best":
            if not self._entries:
                raise ValueError("No checkpoint is kept yet: save a best only after `observe` returns `True`.")
            return self._entries[-1][0]
        return self._unique_path(tag)

    @property
    def best_path(self) -> str | None:
        best = self._best()
        return best[0] if best else None

    @property
    def best_depth(self) -> int | None:
        best = self._best()
        return best[2] if best else None

    @property
    def best_score(self) -> Tensor | None:
        best = self._best()
        return best[1] if best else None

    @property
    def best_k_models(self) -> dict[str, Tensor]:
        """The kept checkpoints' paths and scores, in the order they were saved."""
        return {path: score for path, score, _ in self._entries}

    def pin_boundary(self, depth: int, initial_path: str | None = None, copy: bool = True) -> None:
        """Pin the current best as the rollback point for phase ``depth``, by reference: checkpoints are never
        rewritten, and a pinned one is kept until the next pin, even once it is no longer among the best.

        Before any best exists, ``initial_path`` (a state saved at setup) is pinned instead. ``copy`` is part of the
        protocol and unused here, since a pin writes no file.
        """
        source, source_depth = self.best_path, self.best_depth
        if source is None and initial_path is not None:
            source, source_depth = initial_path, depth
        pinned = source if source is not None and os.path.exists(source) else None
        if pinned is not None:
            self._known.add(pinned)
        self._boundaries = {depth: (pinned, source_depth)}

    def boundary_path(self, depth: int) -> str | None:
        return self._boundaries.get(depth, (None, None))[0]

    def _referenced(self) -> set[str]:
        return {path for path, _, _ in self._entries} | {path for path, _ in self._boundaries.values() if path}

    def prune(self, remove: bool = True) -> None:
        """Forget, and with ``remove`` delete, the checkpoints this policy wrote or referred to that are no longer
        kept, pinned or needed by the latest resume point."""
        for path in sorted(self._known - self._referenced() - self._protected):
            if remove:
                remove_checkpoint(path)
            self._known.discard(path)

    def mark_resume_point(self, remove: bool = True) -> None:
        """Keep the checkpoints the current state refers to until the next resume point (a later
        :meth:`~finetuning_scheduler.core.controller.FTSController.save`), then :meth:`prune`."""
        self._protected = self._referenced()
        self.prune(remove)

    def verify_resume(self) -> None:
        """Check that the checkpoints a loaded state refers to still exist.

        A missing best or rollback point raises
        ``FileNotFoundError``; any other kept checkpoint that is missing is dropped with a warning.
        """
        hint = (" A run that continued past this save has removed it. Resume from that run's latest save, whose"
                " checkpoints are always kept.")
        best = self.best_path
        if best is not None and not os.path.exists(best):
            raise FileNotFoundError(f"The best checkpoint this state refers to, {best!r}, no longer exists.{hint}")
        for depth, (path, _) in self._boundaries.items():
            if path is not None and not os.path.exists(path):
                raise FileNotFoundError(
                    f"The checkpoint a veto of phase {depth} rolls back to, {path!r}, no longer exists.{hint}")
        missing = [path for path, _, _ in self._entries if not os.path.exists(path)]
        if missing:
            rank_zero_warn(f"Dropping {len(missing)} kept checkpoint(s) that no longer exist: {missing}.")
            self._entries = [e for e in self._entries if e[0] not in missing]

    def state_dict(self) -> dict[str, Any]:
        return {"entries": [{"path": path, "score": score, "depth": depth} for path, score, depth in self._entries],
                "seq": self._seq, "boundaries": dict(self._boundaries)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._entries = [(e["path"], e["score"], e["depth"]) for e in state_dict["entries"]]
        self._seq, self._boundaries = state_dict["seq"], dict(state_dict["boundaries"])
        self._known |= self._referenced()

    def config(self) -> dict[str, Any]:
        return {"class_path": f"{type(self).__module__}.{type(self).__qualname__}",
                "init_args": {"monitor": self.monitor, "mode": self.mode, "save_top_k": self.save_top_k,
                              "include": self.include, "save_initial": self.save_initial, "metadata": self.metadata}}
