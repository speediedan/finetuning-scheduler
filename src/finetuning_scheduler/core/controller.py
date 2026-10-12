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
Fine-Tuning Scheduler Controller
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`FTSController` schedules multi-phase fine-tuning in a plain PyTorch training loop. The loop stays yours: the
controller is driven by explicit calls, in the spirit of ``torch.optim.lr_scheduler``, and never runs training itself.

.. warning::

    Experimental: the direct PyTorch API may change in minor releases until it stabilizes.

"""
import os
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from finetuning_scheduler.core import optim as core_optim
from finetuning_scheduler.core import phase as core_phase
from finetuning_scheduler.core import schedule as core_schedule
from finetuning_scheduler.core.config import config_hash, describe_callable
from finetuning_scheduler.core.dist import rank_zero_info, rank_zero_warn
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.fingerprint import batch_fingerprint
from finetuning_scheduler.core.host import LRStepConfig, TorchHost, TrainingHost
from finetuning_scheduler.core.rng import fork_rngs
from finetuning_scheduler.core.transitions import (AnyOf, as_decision, Decision, epoch_transition_due,
                                                   TransitionContext, TransitionPolicy)

STATE_VERSION = 1


@dataclass(frozen=True)
class PhaseEvent:
    """What a phase hook sees.

    Attributes:
        depth: The phase starting (for start hooks) or ending (for end hooks).
        epoch: The number of completed epochs.
        global_step: The number of completed optimizer steps.
        phase_config: The phase's payload: the ``phase_config_type`` instance if one was given, otherwise the mapping.
        metadata: The run's user metadata.
        controller: The controller, for access to the module, optimizer and schedule.
    """

    depth: int
    epoch: int
    global_step: int
    phase_config: Any
    metadata: Mapping[str, Any]
    controller: "FTSController" = field(repr=False)


def _is_compiled(module: nn.Module) -> bool:
    return hasattr(module, "_orig_mod") or any(hasattr(m, "_orig_mod") for m in module.modules())


class FTSController:
    """Schedule multi-phase fine-tuning in a plain PyTorch training loop.

    The minimal loop is::

        fts = FTSController(model, "schedule.yaml", optimizer=partial(torch.optim.AdamW, lr=1e-4))
        fts.setup()
        for epoch in fts.epochs(max_epochs=10):
            for batch in loader:
                loss = model(batch).loss
                loss.backward()
                fts.optimizer.step()
                fts.optimizer.zero_grad()
            fts.epoch_end({"val_loss": evaluate(model)})

    Phases advance at epoch boundaries when the schedule's ``max_transition_epoch`` is reached or a transition policy
    (for example :class:`~finetuning_scheduler.core.transitions.PlateauTransition`) decides so.

    Args:
        model: The module to fine-tune.
        schedule: A schedule mapping, a path to a schedule YAML file, or ``None`` to generate the default schedule.
        optimizer: An optimizer factory called with phase 0's parameter groups (preferred), or an optimizer instance,
            which is reconfigured to optimize exactly phase 0's parameters unless ``enforce_phase0_params`` is
            ``False``.
        lr_scheduler: An lr scheduler factory called with the optimizer, an lr scheduler instance, or ``None``.
        lr_scheduler_cfg: How :meth:`step_lr_schedulers` steps the scheduler: ``interval`` (``"epoch"`` or
            ``"step"``), ``frequency`` and ``monitor``.
        transition: Transition policies composed with the schedule's ``max_transition_epoch`` and
            ``max_transition_step`` keys. With none, every phase but the final one must set one of those keys.
        reinit_optim_cfg: An optimizer reinitialization configuration applied at every phase transition (implicit
            mode), as with the Lightning callback; per-phase ``new_optimizer`` keys serve explicit schedules.
        reinit_lr_cfg: An lr scheduler reinitialization configuration applied at every phase transition (implicit
            mode); per-phase ``new_lr_scheduler`` keys serve explicit schedules.
        max_depth: The deepest phase to run; ``-1`` runs every phase.
        base_max_lr: The default learning rate of phases after phase 0 that do not set ``lr``.
        enforce_phase0_params: Reconfigure an optimizer instance whose parameters differ from phase 0's.
        apply_lambdas_new_pgs: Apply ``LambdaLR``-style lr lambdas to the groups new phases add.
        frozen_bn_track_running_stats: Keep frozen ``BatchNorm`` layers tracking running statistics.
        no_decay: Parameter name fragments whose groups get ``weight_decay=0``.
        phase_config_type: A callable that validates each phase's ``phase_config`` mapping (called with its items as
            keyword arguments); hooks then receive its result.
        allow_untested: Allow lr scheduler types Fine-Tuning Scheduler has not been tested with.
        isolate_rng: Run phase hooks without disturbing the training random streams (see
            :func:`~finetuning_scheduler.core.rng.fork_rngs`).
        fingerprint_batches: How many batches per phase :meth:`observe_batch` fingerprints.
        metadata: User metadata, part of the configuration hash and handed to hooks.
        log_dir: Where a generated schedule is saved.
        host: The :class:`~finetuning_scheduler.core.host.TrainingHost`; a
            :class:`~finetuning_scheduler.core.host.TorchHost` by default.
        compile_reset: Reset dynamo's compiled graphs at each transition; by default when the module is compiled.
        allow_meta_frozen: Allow meta-device parameters that no phase schedules (they are warned about).
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: str | os.PathLike | Mapping[int, Mapping[str, Any]] | None = None,
        *,
        optimizer: Optimizer | Callable[[list[dict[str, Any]]], Optimizer],
        lr_scheduler: Any | Callable[[Optimizer], Any] | None = None,
        lr_scheduler_cfg: Mapping[str, Any] | None = None,
        transition: TransitionPolicy | Sequence[TransitionPolicy] | None = None,
        reinit_optim_cfg: Mapping[str, Any] | None = None,
        reinit_lr_cfg: Mapping[str, Any] | None = None,
        max_depth: int = -1,
        base_max_lr: float = 1e-5,
        enforce_phase0_params: bool = True,
        apply_lambdas_new_pgs: bool = False,
        frozen_bn_track_running_stats: bool = True,
        no_decay: Sequence[str] | None = None,
        phase_config_type: Callable[..., Any] | None = None,
        allow_untested: bool = False,
        isolate_rng: bool = True,
        fingerprint_batches: int = 4,
        metadata: Mapping[str, Any] | None = None,
        log_dir: str | os.PathLike | None = None,
        host: TrainingHost | None = None,
        compile_reset: bool | None = None,
        allow_meta_frozen: bool = False,
    ) -> None:
        self.model = model
        self._schedule_src = schedule
        self._optimizer_src = optimizer
        self._lr_scheduler_src = lr_scheduler
        self._lr_step_cfg = LRStepConfig(**dict(lr_scheduler_cfg or {}))
        policies = [] if transition is None else ([transition] if not isinstance(transition, Sequence) else
                                                  list(transition))
        self._user_policies = policies
        self._policy = AnyOf(*policies)
        self.reinit_optim_cfg = dict(reinit_optim_cfg) if reinit_optim_cfg else None
        self.reinit_lr_cfg = dict(reinit_lr_cfg) if reinit_lr_cfg else None
        self._max_depth_arg = max_depth
        self.base_max_lr = base_max_lr
        self.enforce_phase0_params = enforce_phase0_params
        self.apply_lambdas_new_pgs = apply_lambdas_new_pgs
        self.frozen_bn_track_running_stats = frozen_bn_track_running_stats
        self.no_decay = list(no_decay) if no_decay else None
        self.phase_config_type = phase_config_type
        self.allow_untested = allow_untested
        self.isolate_rng = isolate_rng
        self.fingerprint_batches = fingerprint_batches
        self._metadata = dict(metadata or {})
        self.log_dir = log_dir
        self._host = host if host is not None else TorchHost(model, log_dir=log_dir)
        self._compile_reset = compile_reset
        self.allow_meta_frozen = allow_meta_frozen

        self._schedule: dict = {}
        self._phase_configs: dict[int, Any] = {}
        self._max_depth = -1
        self._depth = 0
        self._phase_start_epoch = 0
        self._phase_start_step = 0
        self._pending: Decision = Decision.CONTINUE
        self._pending_reason: str | None = None
        self._should_stop = False
        self._stop_reason: str | None = None
        self._fingerprints: dict[int, list[str]] = {}
        self._hooks: dict[str, OrderedDict[int, tuple[Callable, bool | None]]] = {"start": OrderedDict(),
                                                                                  "end": OrderedDict()}
        self._epoch_end_called = False
        self._counting_suspended = False
        self._steps_since_step_end = 0
        self._step_hook: RemovableHandle | None = None
        self._is_setup = False
        self._finished = False

    # ------------------------------------------------------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------------------------------------------------------
    def setup(self) -> None:
        """Load, generate or validate the schedule; freeze the model and thaw phase 0; build or reconfigure the
        optimizer and lr scheduler; then run the phase-0 start hooks."""
        if self._is_setup:
            raise FTSConfigurationError("`FTSController.setup()` has already been called.")
        self._refuse_unsupported_model()
        self._schedule = self._load_schedule()
        self._validate_schedule()
        self._max_depth = core_schedule.resolve_max_depth(self._max_depth_arg, len(self._schedule))
        self._refuse_unschedulable_params()
        core_phase.freeze_module_tree(self.model, train_bn=False,
                                      bn_track_running_stats=self.frozen_bn_track_running_stats)
        self._thaw_phase(0, init_thaw=True)
        optimizer, schedulers = self._configure_optimizer_and_schedulers()
        self._host.set_optimizer(0, optimizer)
        for i, sched in enumerate(schedulers):
            self._host.set_lr_scheduler(i, sched, self._lr_step_cfg)
        self._step_hook = optimizer.register_step_post_hook(self._on_optimizer_step)
        self._check_schedule_agreement()
        self._fingerprints.setdefault(0, [])
        self._is_setup = True
        self._run_hooks("start", 0)
        rank_zero_info(f"Fine-tuning schedule set up with {len(self._schedule)} phases (max_depth {self._max_depth}).")

    def _refuse_unsupported_model(self) -> None:
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            raise FTSConfigurationError(
                "`DistributedDataParallel` registers only the parameters that require grad when it wraps the module,"
                " so parameters thawed in later phases would never be gradient-synchronized. Pass the unwrapped module"
                " and shard it with FSDP2 (`fully_shard`) instead."
            )
        if isinstance(self._optimizer_src, core_optim.ZeroRedundancyOptimizer):
            raise FTSConfigurationError(
                "`ZeroRedundancyOptimizer` is not supported by the direct PyTorch API; shard with FSDP2 instead."
            )

    def _load_schedule(self) -> dict:
        src = self._schedule_src
        if src is None:
            schedule = core_schedule.build_default_schedule(self.model)
            if self.log_dir is not None and self._host.is_global_zero:
                core_schedule.save_schedule(f"{type(self.model).__name__}_ft_schedule.yaml", schedule, self.log_dir)
            return self._host.broadcast_object(schedule)
        if isinstance(src, (str, os.PathLike)):
            return core_schedule.load_yaml_schedule(src)
        if isinstance(src, Mapping):
            return {k: dict(deepcopy(v)) for k, v in src.items()}
        raise FTSConfigurationError(
            f"`schedule` must be a mapping, a path to a schedule YAML file or None, got {type(src).__name__}."
        )

    def _validate_schedule(self) -> None:
        schedule, _ = core_schedule.convert_phase_keys(self._schedule)
        if not core_schedule.ensure_contiguous_phase_keys(schedule):
            raise FTSConfigurationError(
                core_schedule.NON_CONTIGUOUS_KEYS_MSG + "(phases must be numbered 0, 1, 2, ...)."
            )
        self._schedule = schedule
        core_schedule.parse_phase_extensions(schedule)
        core_schedule.check_schedule_vs_reinit_lr(self._schedule_src is not None, self.reinit_lr_cfg)
        reinit = core_schedule.collect_reinit_cfgs(
            schedule, {"new_lr_scheduler": self.reinit_lr_cfg, "new_optimizer": self.reinit_optim_cfg}
        )
        if any(reinit.values()):
            core_schedule.refuse_phase0_reinit(reinit)
            for key, cfg in reinit.items():
                implicit = bool(self.reinit_lr_cfg if key == "new_lr_scheduler" else self.reinit_optim_cfg)
                for depth, target in ([(None, cfg)] if implicit else list(cfg.items())):
                    if key == "new_optimizer":
                        self._validate_optimizer_reinit(target, depth)
                    else:
                        self._validate_lr_scheduler_reinit(target, depth, implicit)
        named_params = dict(self.model.named_parameters()).keys()
        shared = core_schedule.shared_params_ref(self.model)
        max_epoch_wm = -1
        for depth in schedule:
            schedule[depth].setdefault("max_transition_epoch", -1)
            core_schedule.parse_phase_lr(schedule, depth, self.base_max_lr)
            core_schedule.resolve_phase_params(schedule, depth, named_params, shared)
            if depth > 0:
                max_epoch_wm = core_schedule.update_epoch_watermark(schedule, depth, max_epoch_wm,
                                                                    epoch_transitions_only=not self._user_policies)
        core_schedule.validate_phases_disjoint(schedule)
        if not self._user_policies:
            final = len(schedule) - 1
            unset = [d for d in schedule if d < final and schedule[d]["max_transition_epoch"] < 0
                     and schedule[d]["max_transition_step"] < 0]
            if unset:
                raise FTSConfigurationError(
                    f"Phases {unset} set neither `max_transition_epoch` nor `max_transition_step`, and no transition"
                    " policy was given, so they would never end. Set one of the keys or pass `transition=` (for example"
                    " `PlateauTransition('val_loss')`)."
                )
        self._phase_configs = {}
        for depth, phase in schedule.items():
            raw = phase.get("phase_config")
            if raw is None:
                continue
            try:
                self._phase_configs[depth] = self.phase_config_type(**raw) if self.phase_config_type else dict(raw)
            except Exception as err:
                raise FTSConfigurationError(f"Phase {depth}'s `phase_config` failed validation: {err}") from err

    @staticmethod
    def _validate_optimizer_reinit(target: dict, depth: int | None) -> None:
        core_schedule.validate_reinit_keys(target, "optimizer_init", depth)
        optimizer_init = target["optimizer_init"]
        optimizer_class = core_optim.import_reinit_class(optimizer_init)
        core_optim.check_supported_reinit_optimizer(optimizer_class)
        core_optim.check_optimizer_init(optimizer_class, optimizer_init)

    def _validate_lr_scheduler_reinit(self, target: dict, depth: int | None, implicit: bool) -> None:
        core_schedule.validate_reinit_keys(target, "lr_scheduler_init", depth)
        if "init_pg_lrs" in target and implicit:
            raise FTSConfigurationError(
                "Specifying a `init_pg_lrs` key in the lr scheduler configuration passed via `reinit_lr_cfg` (i.e. "
                "implicit mode training) is not a valid configuration since the same lr scheduler configuration "
                "is intended to be reinitialized at every fine-tuning phase with implicit mode fine-tuning."
            )
        step_cfg = target.get("pl_lrs_cfg", {})
        unsupported = sorted(set(step_cfg) - {"interval", "frequency", "monitor"})
        if unsupported:
            raise FTSConfigurationError(
                f"The direct PyTorch API steps lr schedulers with `interval`, `frequency` and `monitor` only; remove"
                f" {unsupported} from `pl_lrs_cfg`."
            )
        if step_cfg.get("interval", "epoch") not in ("step", "epoch"):
            raise FTSConfigurationError(
                f'`pl_lrs_cfg` "interval" must be "step" or "epoch", got "{step_cfg["interval"]}"'
            )
        if "init_pg_lrs" in target:
            assert depth is not None
            core_schedule.parse_reinit_pg_lrs(depth, target["init_pg_lrs"])
        lr_scheduler_init = target["lr_scheduler_init"]
        lrs_class = core_optim.import_reinit_class(lr_scheduler_init)
        core_optim.check_supported_lr(lrs_class, self.allow_untested)
        core_optim.check_lr_scheduler_init(lrs_class, lr_scheduler_init, implicit)

    def _refuse_unschedulable_params(self) -> None:
        scheduled = {n for d, phase in self._schedule.items() if d <= self._max_depth for n in phase["params"]}
        named = dict(self.model.named_parameters())
        meta = [n for n, p in named.items() if p.is_meta]
        meta_scheduled = sorted(set(meta) & scheduled)
        if meta_scheduled or (meta and not self.allow_meta_frozen):
            raise FTSConfigurationError(
                f"Parameters on the meta device cannot be trained or restored: {meta_scheduled or meta}. Materialize"
                " them first, or pass `allow_meta_frozen=True` when only unscheduled parameters are on the meta device."
            )
        if meta:
            rank_zero_warn(f"Unscheduled parameters remain on the meta device: {meta}.")
        bad = []
        for n in sorted(scheduled):
            p = named[n]
            plain = type(p) in (nn.Parameter, torch.Tensor) or type(p).__name__ == "DTensor"
            if not (p.is_floating_point() or p.is_complex()) or not plain:
                bad.append(f"{n} ({p.dtype}, {type(p).__name__})")
        if bad:
            raise FTSConfigurationError(
                "Scheduled parameters must be floating point (or complex) tensors or `DTensor`s; these cannot be"
                f" trained: {bad}. Leave quantized base weights out of the schedule and schedule their adapters."
            )

    def _configure_optimizer_and_schedulers(self) -> tuple[Optimizer, list[Any]]:
        phase0 = self._schedule[0]
        src = self._optimizer_src
        if isinstance(src, Optimizer):
            if phase0.get("pg_overrides"):
                raise FTSConfigurationError(
                    "Phase 0's `pg_overrides` needs an optimizer factory: an optimizer instance's groups were built"
                    " before the schedule was read."
                )
            optimizer = src
            schedulers = self._build_schedulers(optimizer)
            no_grad_cnt, init_cnt, total_cnt, diff = core_optim.inspect_phase0_opt_state(optimizer, self.model,
                                                                                         self._schedule, list, list)
            if diff or no_grad_cnt > 0:
                if not self.enforce_phase0_params:
                    rank_zero_warn(core_optim.phase0_divergence_warning(init_cnt, no_grad_cnt, total_cnt, diff))
                else:
                    orig_num_pgs = core_optim.clean_optim_lr_pgs([optimizer], schedulers)
                    core_optim.add_optimizer_groups(
                        self.model, optimizer, phase0["params"], no_decay=self.no_decay,
                        lr=phase0.get("lr", optimizer.defaults["lr"]), apply_lambdas=self.apply_lambdas_new_pgs,
                        schedulers=schedulers,
                    )
                    core_optim.reconfigure_lrs_for_phase0(schedulers, orig_num_pgs)
                    rank_zero_info(core_optim.PHASE_0_DIVERGENCE_MSG + (
                        "Since `enforce_phase0_params` is `True` (the default), the optimizer was reconfigured to"
                        " optimize the parameters (and only those parameters) scheduled in phase 0."))
        elif callable(src):
            optimizer = src(core_optim.reinit_phase0_pgs(self.model, phase0["params"], self.no_decay,
                                                         phase0.get("pg_overrides")))
            if not isinstance(optimizer, Optimizer):
                raise FTSConfigurationError(f"The optimizer factory returned {type(optimizer).__name__}, not an"
                                            " `Optimizer`.")
            schedulers = self._build_schedulers(optimizer)
        else:
            raise FTSConfigurationError("`optimizer` must be an `Optimizer` or a factory taking parameter groups.")
        for sched in schedulers:
            core_optim.check_supported_lr(type(sched), self.allow_untested)
            if getattr(sched, "optimizer", optimizer) is not optimizer:
                raise FTSConfigurationError("The lr scheduler wraps a different optimizer than the one scheduled.")
        return optimizer, schedulers

    def _build_schedulers(self, optimizer: Optimizer) -> list[Any]:
        src = self._lr_scheduler_src
        if src is None:
            return []
        if not isinstance(src, type) and hasattr(src, "step") and hasattr(src, "state_dict"):
            return [src]
        if callable(src):
            return [src(optimizer)]
        raise FTSConfigurationError("`lr_scheduler` must be an lr scheduler, a factory taking the optimizer, or None.")

    def _check_schedule_agreement(self) -> None:
        digests = self._host.all_gather_object(config_hash({"schedule": self._schedule}))
        if len(set(digests)) > 1:
            raise FTSConfigurationError(
                "Ranks resolved different schedules (their models' parameter names differ). Every rank must schedule"
                " the same parameters."
            )

    # ------------------------------------------------------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------------------------------------------------------
    def epochs(self, max_epochs: int) -> Iterator[int]:
        """Iterate over epoch indices, calling :meth:`epoch_start` before each epoch and :meth:`epoch_end` after it
        (with no metrics) unless the loop called it; stops early once :attr:`should_stop`, then calls
        :meth:`train_end`."""
        self._require_setup()
        self._host.set_max_epochs(max_epochs)
        try:
            while self._host.current_epoch < max_epochs and not self._should_stop:
                self.epoch_start()
                yield self._host.current_epoch
                if not self._epoch_end_called:
                    self.epoch_end()
        finally:
            self.train_end()

    def epoch_start(self) -> None:
        """Begin an epoch: advance to the next phase when a transition is pending or the schedule's
        ``max_transition_epoch`` is reached (at most one phase per epoch)."""
        self._require_setup()
        self._epoch_end_called = False
        if self.is_final_phase:
            return
        phase = self._schedule[self._depth]
        due = epoch_transition_due(phase["max_transition_epoch"], self._host.current_epoch)
        if self._pending >= Decision.ADVANCE or due:
            reason = self._pending_reason if self._pending >= Decision.ADVANCE else (
                f"max_transition_epoch ({phase['max_transition_epoch']}) reached")
            self._pending, self._pending_reason = Decision.CONTINUE, None
            self._advance(reason)

    def epoch_end(self, metrics: Mapping[str, Any] | None = None) -> Decision:
        """End an epoch: evaluate the epoch-granular transition policies on ``metrics``, agree across ranks (the
        greatest decision wins), and count the epoch.

        Returns:
            The agreed decision. ``ADVANCE`` takes effect at the next :meth:`epoch_start`; ``STOP`` and ``VETO`` set
            :attr:`should_stop`.
        """
        self._require_setup()
        self._epoch_end_called = True
        decision, reason = self._evaluate("epoch", metrics, completed_epochs=self._host.current_epoch + 1)
        self._host.advance_epoch()
        if decision >= Decision.STOP:
            self._stop(reason or decision.name.lower())
        elif decision == Decision.ADVANCE and not self.is_final_phase:
            self._pending, self._pending_reason = decision, reason
        return decision

    def step_end(self, metrics: Mapping[str, Any] | None = None) -> Decision:
        """End an optimizer step: evaluate the step-granular transition policies and the schedule's
        ``max_transition_step`` on ``metrics``, agree across ranks, and apply an ``ADVANCE`` at once (between steps).

        Call it after each optimizer step when a step-granular transition is configured.

        Returns:
            The agreed decision.
        """
        self._require_setup()
        self._steps_since_step_end = 0
        decision, reason = self._evaluate("step", metrics, completed_epochs=self._host.current_epoch)
        if not self.is_final_phase:
            mts = self._schedule[self._depth]["max_transition_step"]
            if decision < Decision.ADVANCE and 0 <= mts <= self._host.global_step:
                decision, reason = Decision.ADVANCE, f"max_transition_step ({mts}) reached"
        if decision >= Decision.STOP:
            self._stop(reason or decision.name.lower())
        elif decision == Decision.ADVANCE and not self.is_final_phase:
            self._advance(reason)
        return decision

    @property
    def _step_granular(self) -> bool:
        return "step" in self._policy.granularity or any(
            p["max_transition_step"] >= 0 for d, p in self._schedule.items() if d < self._max_depth)

    def train_end(self) -> None:
        """End training: run the final phase's end hooks (once)."""
        if self._is_setup and not self._finished:
            self._finished = True
            self._run_hooks("end", self._depth)

    def step_lr_schedulers(self, interval: Literal["step", "epoch"], metrics: Mapping[str, Any] | None = None) -> None:
        """Step every lr scheduler configured for ``interval`` (honoring its ``frequency``, and passing its
        ``monitor`` metric to schedulers such as ``ReduceLROnPlateau``)."""
        self._require_setup()
        count = self._host.global_step if interval == "step" else self._host.current_epoch
        for sched, cfg in self._host.get_lr_schedulers():
            if cfg.interval != interval or count % cfg.frequency != 0:
                continue
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if not cfg.monitor or metrics is None or cfg.monitor not in metrics:
                    raise FTSConfigurationError(
                        "`ReduceLROnPlateau` needs `lr_scheduler_cfg={'monitor': ...}` and that metric passed to"
                        " `step_lr_schedulers`."
                    )
                sched.step(float(metrics[cfg.monitor]))
            else:
                sched.step()

    def _evaluate(self, unit: Literal["epoch", "step"], metrics: Mapping[str, Any] | None,
                  completed_epochs: int) -> tuple[Decision, str | None]:
        if unit not in self._policy.granularity:
            local, reason = Decision.CONTINUE, None
        else:
            phase = self._schedule[self._depth]
            ctx = TransitionContext(
                depth=self._depth, max_depth=self._max_depth, is_final_phase=self.is_final_phase, unit=unit,
                epoch=completed_epochs, global_step=self._host.global_step, phase_start_epoch=self._phase_start_epoch,
                phase_start_step=self._phase_start_step, max_transition_epoch=phase["max_transition_epoch"],
                max_transition_step=phase["max_transition_step"], max_epochs=self._host.max_epochs,
                metrics=dict(metrics or {}), phase_config=self._phase_configs.get(self._depth),
                metadata=self._metadata,
            )
            local, reason = as_decision(self._policy.evaluate(ctx))
        agreed = Decision(self._host.reduce_max(int(local)))
        if agreed != local:
            rank_zero_warn(
                f"Ranks disagreed on the {unit}'s transition decision ({local.name} here); following the greatest,"
                f" {agreed.name}. Report the same (synchronized) metrics on every rank to avoid this."
            )
            reason = reason or f"{agreed.name.lower()} agreed across ranks"
        return agreed, reason

    def _stop(self, reason: str) -> None:
        self._should_stop, self._stop_reason = True, reason
        rank_zero_info(f"Fine-tuning stopping in phase {self._depth}: {reason}.")

    # ------------------------------------------------------------------------------------------------------------------
    # phase transitions
    # ------------------------------------------------------------------------------------------------------------------
    def _advance(self, reason: str | None) -> None:
        prev = self._depth
        self._run_hooks("end", prev)
        self._depth += 1
        self._phase_start_epoch, self._phase_start_step = self._host.current_epoch, self._host.global_step
        self._fingerprints.setdefault(self._depth, [])
        self._run_hooks("start", self._depth)
        optimizer = self.optimizer
        schedulers = [s for s, _ in self._host.get_lr_schedulers()]
        pre_state = core_optim.capture_lr_state(schedulers[0] if schedulers else None, optimizer)
        new_optimizer_cfg, phases = core_phase.thaw_plan(self._schedule, self._depth, self._depth - 1,
                                                         depth_sync=False, reinit_optim_cfg=self.reinit_optim_cfg)
        for i, phase in phases:
            if i > self._depth:
                continue
            self._thaw_phase(i)
            if new_optimizer_cfg and i == 0:
                # a reinitialized optimizer starts from phase 0's groups; later phases are replayed onto it
                optimizer = self._reinit_optimizer(new_optimizer_cfg, phase["params"])
                schedulers = [s for s, _ in self._host.get_lr_schedulers()]
                continue
            if i == self._depth and self._depth > 0:
                core_optim.restore_latest_lr_state(optimizer, schedulers, *pre_state)
            core_optim.add_optimizer_groups(
                self.model, optimizer, phase["params"], no_decay=self.no_decay,
                lr=phase.get("lr", optimizer.defaults["lr"]), apply_lambdas=self.apply_lambdas_new_pgs,
                schedulers=schedulers, pg_overrides=phase.get("pg_overrides"),
            )
            new_scheduler_cfg = core_phase.lr_reinit_cfg_for(phase, self._depth, self.reinit_lr_cfg)
            if new_scheduler_cfg:
                self._reinit_lr_scheduler(new_scheduler_cfg, optimizer)
                schedulers = [s for s, _ in self._host.get_lr_schedulers()]
            else:
                core_optim.warn_unapplied_lr_lambdas(schedulers, self.apply_lambdas_new_pgs)
        if self._compile_reset or (self._compile_reset is None and _is_compiled(self.model)):
            self._host.on_compile_reset()
        self._policy.reset(self._depth)
        rank_zero_info(f"Fine-tuning advanced from phase {prev} to phase {self._depth}"
                       f"{f' ({reason})' if reason else ''}.")

    def _reinit_optimizer(self, new_optimizer: dict, init_params: list) -> Optimizer:
        optimizer_init = new_optimizer["optimizer_init"]
        optimizer_class = core_optim.import_reinit_class(optimizer_init)
        reinit_pgs = core_optim.reinit_phase0_pgs(self.model, init_params, self.no_decay,
                                                  self._schedule[0].get("pg_overrides"))
        optimizer = core_optim.build_phase0_optimizer(optimizer_class, optimizer_init.get("init_args", {}),
                                                      reinit_pgs)
        self._retire_optimizer(self.optimizer)
        self._host.set_optimizer(0, optimizer)
        self._step_hook = optimizer.register_step_post_hook(self._on_optimizer_step)  # type: ignore[attr-defined]
        for sched, _ in self._host.get_lr_schedulers():
            sched.optimizer = optimizer
        rank_zero_info(f"Optimizer reinitialized at phase {self._depth}: {optimizer_class.__name__}.")
        return optimizer  # type: ignore[return-value]

    def _reinit_lr_scheduler(self, new_lr_scheduler: dict, optimizer: Optimizer) -> None:
        lr_scheduler_init = new_lr_scheduler["lr_scheduler_init"]
        lrs_class = core_optim.import_reinit_class(lr_scheduler_init)
        core_optim.apply_reinit_pg_lrs(optimizer, new_lr_scheduler)
        new_scheduler = lrs_class(optimizer=optimizer, **lr_scheduler_init.get("init_args", {}))
        existing = self._host.get_lr_schedulers()
        step_cfg = LRStepConfig(**{**(vars(existing[0][1]) if existing else vars(self._lr_step_cfg)),
                                   **new_lr_scheduler.get("pl_lrs_cfg", {})})
        if existing:
            self._retire_lr_scheduler(existing[0][0])
        self._host.set_lr_scheduler(0, new_scheduler, step_cfg)
        rank_zero_info(f"Lr scheduler reinitialized at phase {self._depth}: {lrs_class.__name__}.")

    def _retire_optimizer(self, optimizer: Optimizer) -> None:
        if self._step_hook is not None:
            self._step_hook.remove()
        depth = self._depth

        def _refuse(*_: Any, **__: Any) -> None:
            raise FTSConfigurationError(
                f"This optimizer was replaced when phase {depth} reinitialized the optimizer; step"
                " `FTSController.optimizer` instead."
            )

        optimizer.register_step_pre_hook(_refuse)

    def _retire_lr_scheduler(self, scheduler: Any) -> None:
        depth = self._depth

        def _refuse(*_: Any, **__: Any) -> None:
            raise FTSConfigurationError(
                f"This lr scheduler was replaced when phase {depth} reinitialized it; use"
                " `FTSController.lr_scheduler` (or `step_lr_schedulers`) instead."
            )

        scheduler.step = _refuse

    def _thaw_phase(self, depth: int, init_thaw: bool = False) -> None:
        params = self._schedule[depth]["params"]
        core_phase.thaw_params(self.model, params, init_thaw=init_thaw)
        if not self.frozen_bn_track_running_stats:
            owners = {n.rsplit(".", 1)[0] if "." in n else "" for n in params}
            for name, mod in self.model.named_modules():
                if name in owners and isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.track_running_stats = True

    # ------------------------------------------------------------------------------------------------------------------
    # hooks, fingerprints and step counting
    # ------------------------------------------------------------------------------------------------------------------
    def add_phase_hook(self, fn: Callable[[PhaseEvent], None], when: Literal["start", "end"] = "start", *,
                       isolate_rng: bool | None = None) -> RemovableHandle:
        """Call ``fn(event)`` whenever a phase starts (phase 0 at :meth:`setup`) or ends (the final phase at
        :meth:`train_end`). Hooks run in registration order, with the training random streams isolated unless
        ``isolate_rng`` (or the controller's default) is ``False``.

        Returns:
            A handle whose ``remove()`` unregisters the hook.
        """
        if when not in self._hooks:
            raise FTSConfigurationError(f"`when` must be 'start' or 'end', got {when!r}.")
        handle = RemovableHandle(self._hooks[when])
        self._hooks[when][handle.id] = (fn, isolate_rng)
        return handle

    def _run_hooks(self, when: str, depth: int) -> None:
        event = PhaseEvent(depth=depth, epoch=self._host.current_epoch, global_step=self._host.global_step,
                           phase_config=self._phase_configs.get(depth), metadata=self._metadata, controller=self)
        for fn, isolate in list(self._hooks[when].values()):
            isolate = self.isolate_rng if isolate is None else isolate
            with self._suspend_step_counting(), (fork_rngs(self._host.rng_devices()) if isolate else nullcontext()):
                fn(event)

    def observe_batch(self, batch: Any) -> None:
        """Fingerprint ``batch`` if fewer than ``fingerprint_batches`` batches of the current phase have been."""
        prints = self._fingerprints.setdefault(self._depth, [])
        if len(prints) < self.fingerprint_batches:
            prints.append(batch_fingerprint(batch))

    def observe_batches(self, batches: Iterable[Any]) -> Iterator[Any]:
        """Pass ``batches`` through, fingerprinting each with :meth:`observe_batch`."""
        for batch in batches:
            self.observe_batch(batch)
            yield batch

    @contextmanager
    def _suspend_step_counting(self) -> Iterator[None]:
        prev, self._counting_suspended = self._counting_suspended, True
        try:
            yield
        finally:
            self._counting_suspended = prev

    def _on_optimizer_step(self, optimizer: Optimizer, args: Any, kwargs: Any) -> None:
        if self._counting_suspended:
            return
        self._host.advance_step(1)
        self._steps_since_step_end += 1
        if self._steps_since_step_end > 1 and self._step_granular and not self.is_final_phase:
            raise FTSConfigurationError(
                "Two optimizer steps ran without `FTSController.step_end()` between them, but a step-granular"
                " transition is configured. Call `step_end()` after each optimizer step."
            )

    # ------------------------------------------------------------------------------------------------------------------
    # state and configuration
    # ------------------------------------------------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """The controller's resumable state: depth, counters, pending decision, stop state, policy state,
        fingerprints and the schedule and configuration hashes.

        Optimizer and lr scheduler state are saved separately.
        """
        return {
            "version": STATE_VERSION,
            "depth": self._depth,
            "epoch": self._host.current_epoch,
            "global_step": self._host.global_step,
            "phase_start_epoch": self._phase_start_epoch,
            "phase_start_step": self._phase_start_step,
            "pending": int(self._pending),
            "pending_reason": self._pending_reason,
            "should_stop": self._should_stop,
            "stop_reason": self._stop_reason,
            "policy": self._policy.state_dict(),
            "fingerprints": deepcopy(self._fingerprints),
            "schedule_hash": config_hash({"schedule": self._schedule}),
            "config_hash": self.config_hash,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore state saved by :meth:`state_dict`, thawing (and adding optimizer groups for) every phase up to
        the saved depth. Call it after :meth:`setup` and before loading the optimizer's and lr schedulers' state.

        Raises:
            FTSConfigurationError: If the saved schedule differs from this controller's.
        """
        self._require_setup()
        if state_dict["schedule_hash"] != config_hash({"schedule": self._schedule}):
            raise FTSConfigurationError("The saved state belongs to a different fine-tuning schedule.")
        if state_dict["config_hash"] != self.config_hash:
            rank_zero_warn("The saved state was produced with a different controller configuration.")
        optimizer = self.optimizer
        schedulers = [s for s, _ in self._host.get_lr_schedulers()]
        for depth in range(1, state_dict["depth"] + 1):
            self._thaw_phase(depth)
            phase = self._schedule[depth]
            core_optim.add_optimizer_groups(
                self.model, optimizer, phase["params"], no_decay=self.no_decay,
                lr=phase.get("lr", optimizer.defaults["lr"]), apply_lambdas=self.apply_lambdas_new_pgs,
                schedulers=schedulers, pg_overrides=phase.get("pg_overrides"),
            )
        self._depth = state_dict["depth"]
        self._host.load_counters(state_dict["epoch"], state_dict["global_step"])  # type: ignore[attr-defined]
        self._phase_start_epoch, self._phase_start_step = state_dict["phase_start_epoch"], state_dict[
            "phase_start_step"]
        self._pending, self._pending_reason = Decision(state_dict["pending"]), state_dict["pending_reason"]
        self._should_stop, self._stop_reason = state_dict["should_stop"], state_dict["stop_reason"]
        self._policy.load_state_dict(state_dict["policy"])
        self._fingerprints = deepcopy(state_dict["fingerprints"])

    def config_dict(self) -> dict[str, Any]:
        """The configuration the hash covers (paths and package versions excluded)."""
        opt = self._optimizer_src
        sched = self._lr_scheduler_src
        return {
            "schedule": self._schedule,
            "max_depth": self._max_depth if self._is_setup else self._max_depth_arg,
            "base_max_lr": self.base_max_lr,
            "enforce_phase0_params": self.enforce_phase0_params,
            "apply_lambdas_new_pgs": self.apply_lambdas_new_pgs,
            "frozen_bn_track_running_stats": self.frozen_bn_track_running_stats,
            "no_decay": self.no_decay,
            "optimizer": ({"class_path": describe_callable(type(opt)), "defaults": opt.defaults}
                          if isinstance(opt, Optimizer) else describe_callable(opt)),
            "lr_scheduler": (None if sched is None else describe_callable(sched)
                             if isinstance(sched, type) or not hasattr(sched, "state_dict")
                             else {"class_path": describe_callable(type(sched))}),
            "lr_scheduler_cfg": vars(self._lr_step_cfg),
            "transition": self._policy.config(),
            "reinit_optim_cfg": self.reinit_optim_cfg,
            "reinit_lr_cfg": self.reinit_lr_cfg,
            "isolate_rng": self.isolate_rng,
            "fingerprint_batches": self.fingerprint_batches,
            "metadata": self._metadata,
        }

    @property
    def config_hash(self) -> str:
        """The SHA-256 of :meth:`config_dict`'s canonical form."""
        return config_hash(self.config_dict())

    def versions(self) -> dict[str, str]:
        """The package versions in use (recorded with checkpoints, not hashed)."""
        from finetuning_scheduler.__about__ import __version__

        return {"finetuning_scheduler": __version__, "torch": torch.__version__,
                "python": ".".join(map(str, sys.version_info[:3]))}

    # ------------------------------------------------------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------------------------------------------------------
    def _require_setup(self) -> None:
        if not self._is_setup:
            raise FTSConfigurationError("Call `FTSController.setup()` first.")

    @property
    def host(self) -> TrainingHost:
        return self._host

    @property
    def optimizer(self) -> Optimizer:
        return self._host.get_optimizers()[0]

    @property
    def lr_schedulers(self) -> list[Any]:
        return [s for s, _ in self._host.get_lr_schedulers()]

    @property
    def lr_scheduler(self) -> Any | None:
        scheds = self.lr_schedulers
        return scheds[0] if scheds else None

    @property
    def schedule(self) -> dict:
        return self._schedule

    @property
    def curr_depth(self) -> int:
        return self._depth

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def depth_remaining(self) -> int:
        return core_phase.depth_remaining(self._max_depth, self._depth)

    @property
    def is_final_phase(self) -> bool:
        return self.depth_remaining == 0

    @property
    def phase_config(self) -> Any:
        return self._phase_configs.get(self._depth)

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def epoch(self) -> int:
        return self._host.current_epoch

    @property
    def global_step(self) -> int:
        return self._host.global_step

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def fingerprints(self) -> dict[int, list[str]]:
        return self._fingerprints
