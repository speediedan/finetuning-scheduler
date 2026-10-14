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
Fine-Tuning Scheduler Controller Bridge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running :class:`~finetuning_scheduler.core.controller.FTSController` under the Lightning callback, host-driven (see
:meth:`~finetuning_scheduler.core.controller.FTSController.attach`): Lightning owns the loop, checkpointing and early
stopping, the callback decides when a phase ends, and the controller sequences each transition.

The bridges keep what strategy adapters and callback subclasses observe unchanged: :class:`LightningPhaseExecutor`
carries out phases through the callback's own ``step_pg``, ``thaw_to_depth`` and the strategy adapter's
``on_after_phase_transition``, and :class:`FTSCheckpointPolicy` restores the best checkpoint through the callback's
``restore_best_ckpt``.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from finetuning_scheduler.core.controller import FTSController
from finetuning_scheduler.lightning.host import LightningHost

if TYPE_CHECKING:
    from torch import Tensor

    from finetuning_scheduler.lightning.fts import FinetuningScheduler


class LightningPhaseExecutor:
    """Carries out a controller's phases through the callback's own phase execution, so the strategy adapter's
    hooks are called as the callback calls them.

    Args:
        fts: The callback.
    """

    def __init__(self, fts: "FinetuningScheduler") -> None:
        self.fts = fts

    def configure_phase(self, depth: int, restored_depth: int, depth_sync: bool, pre_state: Any) -> None:
        """The callback's ``step_pg`` (``restored_depth`` is the callback's own best checkpoint depth, which
        ``step_pg`` reads itself)."""
        # called as the callback's own `step` calls it (`depth_sync` only when not the default), for its subclasses
        sync_kwargs = {} if depth_sync else {"depth_sync": False}
        self.fts.step_pg(depth=depth, optimizer=self.fts.trainer.optimizers[0],  # type: ignore[arg-type]
                         pre_reinit_state=pre_state, **sync_kwargs)

    def resume_at(self, depth: int) -> None:
        # the callback resumes at its own depth with `thaw_to_depth()`, which callback subclasses may override
        if depth == self.fts.curr_depth:
            self.fts.thaw_to_depth()
        else:
            self.fts.thaw_to_depth(depth)

    def after_transition(self, depth: int) -> None:
        self.fts.strategy_adapter.on_after_phase_transition()


class FTSCheckpointPolicy:
    """The Lightning checkpoint callback seen as a controller checkpoint policy: Lightning writes the checkpoints,
    so the policy only reports the best one and restores it through the callback.

    Args:
        fts: The callback.
    """

    def __init__(self, fts: "FinetuningScheduler") -> None:
        self.fts = fts
        self.monitor: str | None = getattr(self._checkpoint_callback, "monitor", None)

    @property
    def _checkpoint_callback(self) -> Any:
        return self.fts.trainer.checkpoint_callback

    def restore(self) -> int | None:
        """Restore the best checkpoint through the callback's ``restore_best_ckpt``; return its depth."""
        self.fts.restore_best_ckpt()
        return self.fts._fts_state._best_ckpt_depth

    def observe(self, metrics: Mapping[str, Any], depth: int) -> bool:
        """Lightning's checkpoint callback decides when to save."""
        return False

    def path_for(self, tag: str) -> str:
        raise NotImplementedError("Lightning's checkpoint callback writes the checkpoints.")

    @property
    def best_path(self) -> str | None:
        return getattr(self._checkpoint_callback, "best_model_path", None) or None

    @property
    def best_depth(self) -> int | None:
        return self.fts._fts_state._best_ckpt_depth

    @property
    def best_score(self) -> "Tensor | None":
        return getattr(self._checkpoint_callback, "best_model_score", None)

    def pin_boundary(self, depth: int, initial_path: str | None = None, copy: bool = True) -> None:
        """Phase gates are not available under Lightning yet, so no rollback point is pinned."""

    def boundary_path(self, depth: int) -> str | None:
        return None

    def state_dict(self) -> dict[str, Any]:
        """Lightning checkpoints the checkpoint callback's state."""
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Lightning restores the checkpoint callback's state."""

    def config(self) -> dict[str, Any]:
        return {"monitor": self.monitor}


def attach_controller(fts: "FinetuningScheduler", phase_hooks: Sequence[tuple] = (), *,
                      run_start_hooks: bool = True) -> FTSController:
    """A host-driven controller for the callback's fit, over its validated schedule (the callback's own optimizer,
    lr schedulers and phase-0 thaw are adopted as they are).

    Args:
        fts: The callback.
        phase_hooks: ``(fn, when, isolate_rng)`` phase hooks to register.
        run_start_hooks: Run phase 0's start hooks (``False`` when the fit resumes a run).
    """
    trainer, pl_module = fts.trainer, fts.pl_module
    assert isinstance(fts.ft_schedule, dict)
    controller = FTSController(
        pl_module, optimizer=trainer.optimizers[0], host=LightningHost(trainer, pl_module),
        checkpoint=FTSCheckpointPolicy(fts), restore_best=fts.restore_best, max_depth=fts.max_depth,
        reinit_optim_cfg=fts.reinit_optim_cfg, reinit_lr_cfg=fts.reinit_lr_cfg,
        compile_reset=False,  # the callback resets a compiled module itself, once its transition is complete
    )
    for fn, when, isolate_rng in phase_hooks:
        controller.add_phase_hook(fn, when=when, isolate_rng=isolate_rng)
    controller.attach(LightningPhaseExecutor(fts), fts.ft_schedule, run_start_hooks=run_start_hooks)
    return controller


__all__ = ["attach_controller", "FTSCheckpointPolicy", "LightningPhaseExecutor"]
