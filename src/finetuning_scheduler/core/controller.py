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

    Experimental: the PyTorch API may change in minor releases until it stabilizes.

"""
import itertools
import os
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from finetuning_scheduler.core import optim as core_optim
from finetuning_scheduler.core import phase as core_phase
from finetuning_scheduler.core import schedule as core_schedule
from finetuning_scheduler.core.checkpoint import (CheckpointPolicy, load_payload, remove_checkpoint, replace_checkpoint,
                                                  save_payload)
from finetuning_scheduler.core.compile import CompiledSchedule, compile_schedule, owner_module, view_default_schedule
from finetuning_scheduler.core.config import config_hash, describe_callable
from finetuning_scheduler.core.dist import rank_zero_info, rank_zero_warn
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.fingerprint import batch_fingerprint
from finetuning_scheduler.core.host import LRStepConfig, TorchHost, TrainingHost
from finetuning_scheduler.core.rng import capture_rng_state, fork_rngs, restore_rng_state
from finetuning_scheduler.core.roles import (is_frozen_role, is_role_selector, role_kind, SELECTOR_PREFIX,
                                             selector_matches)
from finetuning_scheduler.strategies.base import DistributedAdapter, select_distributed_adapter
from finetuning_scheduler.strategies.dtensor import DTensorAdapter
from finetuning_scheduler.core.views import BoundView, CanonicalView, instrument_state_keys, resolve_view
from finetuning_scheduler.core.transitions import (AnyOf, as_decision, Decision, epoch_transition_due, GateEvent,
                                                   GateResult, TransitionContext, TransitionPolicy)

STATE_VERSION = 1
# the non-tensor state of a sharded (DCP) checkpoint directory
META_FILE = "fts_meta.pt"



def _validate_artifacts(artifacts: Mapping[str, Mapping[str, str]] | None) -> dict[str, dict[str, str]]:
    """Check the ``artifacts`` mapping: each named artifact has a ``path`` and a 64-hex-digit ``sha256``, and
    optionally a ``revision``; nothing else."""
    if artifacts is None:
        return {}
    if not isinstance(artifacts, Mapping):
        raise FTSConfigurationError(f"`artifacts` must map names to records, got {type(artifacts).__name__}.")
    checked: dict[str, dict[str, str]] = {}
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise FTSConfigurationError(f"Artifact {name!r} must be a mapping, got {type(record).__name__}.")
        unknown = sorted(set(record) - {"path", "sha256", "revision"})
        if unknown:
            raise FTSConfigurationError(f"Artifact {name!r} has unknown keys {unknown} (path, sha256, revision).")
        sha = record.get("sha256")
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise FTSConfigurationError(f"Artifact {name!r} needs a `path`.")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
            raise FTSConfigurationError(f"Artifact {name!r} needs a `sha256` of 64 hexadecimal digits, got {sha!r}.")
        if "revision" in record and not isinstance(record["revision"], str):
            raise FTSConfigurationError(f"Artifact {name!r}'s `revision` must be a string.")
        checked[str(name)] = dict(record)
    return checked

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
        physical_params: The physical parameters the phase thaws (its own, then its coupled ones).
        logical_params: The logical names the phase's schedule entries resolved to (role selectors excluded).
    """

    depth: int
    epoch: int
    global_step: int
    phase_config: Any
    metadata: Mapping[str, Any]
    controller: "FTSController" = field(repr=False)
    physical_params: tuple[str, ...] = ()
    logical_params: tuple[str, ...] = ()


def _is_compiled(module: nn.Module) -> bool:
    """Whether ``module`` or a submodule is compiled, by ``torch.compile(module)`` or in place by
    ``module.compile()``."""
    return any(hasattr(m, "_orig_mod") or getattr(m, "_compiled_call_impl", None) is not None
               for m in module.modules())


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
        checkpoint: A checkpoint policy (for example
            :class:`~finetuning_scheduler.core.checkpoint.BestCheckpoint`), offered each epoch's metrics; the training
            state is saved whenever it reports a new best.
        restore_best: Restore the best checkpoint's model, optimizer and lr scheduler state at each phase boundary, as
            the Lightning callback does. Defaults to ``True`` when a ``checkpoint`` policy is given.
        capture_rng: Include every random generator's state in :meth:`save`, so :meth:`load` resumes the random
            streams exactly.
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
        view: The :class:`~finetuning_scheduler.core.views.TunableView` through which the schedule names the model's
            parameters (experimental). By default, a view attached to the model as ``fts_tunable_view``, else the
            identity :class:`~finetuning_scheduler.core.views.CanonicalView`. Its ``root`` must be ``model``.
        distributed: The :class:`~finetuning_scheduler.strategies.base.DistributedAdapter` for the model's layout
            (experimental). By default, the first of FSDP2, DTensor (tensor parallelism) and unsharded that recognizes
            the model.
        artifacts: External artifacts the run depends on (published, re-loadable tensors such as a lens or a
            transcoder), as ``{name: {"path": ..., "sha256": ..., "revision": ...}}`` (``revision`` optional). Recorded
            in every checkpoint's and export's metadata, not in the configuration hash.
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
        checkpoint: CheckpointPolicy | None = None,
        restore_best: bool | None = None,
        capture_rng: bool = True,
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
        view: Any | None = None,
        distributed: DistributedAdapter | None = None,
        artifacts: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.model = model
        self._artifacts = _validate_artifacts(artifacts)
        self._schedule_src = schedule
        self._optimizer_src = optimizer
        self._lr_scheduler_src = lr_scheduler
        self._lr_step_cfg = LRStepConfig(**dict(lr_scheduler_cfg or {}))
        policies = [] if transition is None else ([transition] if not isinstance(transition, Sequence) else
                                                  list(transition))
        self._user_policies = policies
        self._policy = AnyOf(*policies)
        self._checkpoint = checkpoint
        self._restore_best = (checkpoint is not None) if restore_best is None else restore_best
        self.capture_rng = capture_rng
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
        self._view_src = view
        self._view: BoundView | None = None
        self._distributed_src = distributed
        self._distributed: DistributedAdapter | None = None
        self._compiled: CompiledSchedule | None = None
        self._logical_params: dict[int, tuple[str, ...]] = {}
        # promotions applied so far, and each promoted tensor's value when it was promoted (on the CPU)
        self._promotions: list[dict[str, Any]] = []
        self._promoted_values: dict[str, Tensor] = {}

        self._schedule: dict = {}
        self._phase_configs: dict[int, Any] = {}
        self._max_depth = -1
        self._epoch_watermark = -1
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
        self._gates: OrderedDict[int, tuple[Callable, bool, bool | None]] = OrderedDict()
        self._statefuls: dict[str, tuple[Any, bool]] = {}
        self._gate_records: list[dict[str, Any]] = []
        self._ended_depths: set[int] = set()
        self._last_metrics: dict[str, Any] = {}
        self._restored_count = 0
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
        if self._restore_best and self._checkpoint is None:
            raise FTSConfigurationError("`restore_best=True` needs a `checkpoint` policy to restore from.")
        self._refuse_unsupported_model()
        self._view = self._resolve_view()
        self._distributed = select_distributed_adapter(self.model, self._distributed_src)
        if self._sharded:
            self._refuse_sharded_key_translation()
        self._schedule = self._load_schedule()
        self._validate_schedule()
        assert self._compiled is not None
        if self._compiled.promotions:
            self._refuse_unsupported_promotion()
        self._max_depth = core_schedule.resolve_max_depth(self._max_depth_arg, len(self._schedule))
        self._refuse_unschedulable_params()
        core_phase.reshard_fsdp_modules(self.model)
        self._validate_layout()
        self._warn_unsynchronized_params()
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
        if self._checkpoint is not None and getattr(self._checkpoint, "save_initial", False):
            initial = self._write_checkpoint(self._host.broadcast_object(self._checkpoint.path_for("initial")))
            self._pin_boundary(0, initial_path=initial)
        self._run_hooks("start", 0)
        rank_zero_info(f"Fine-tuning schedule set up with {len(self._schedule)} phases (max_depth {self._max_depth}).")

    def _refuse_unsupported_model(self) -> None:
        # the composable `replicate()` mixes this marker class into a module's class rather than wrapping the module
        replicated = getattr(sys.modules.get("torch.distributed._composable.replicate"), "DDP", None)
        for name, module in self.model.named_modules():
            where = f"`{name}`" if name else "the model"
            if isinstance(module, nn.parallel.DistributedDataParallel):
                raise FTSConfigurationError(
                    f"{where} is wrapped in `DistributedDataParallel`, which registers only the parameters that require"
                    " grad when it wraps the module, so parameters thawed in later phases would never be"
                    " gradient-synchronized. Pass the unwrapped module and shard it with FSDP2 (`fully_shard`) instead."
                )
            if replicated is not None and isinstance(module, replicated):
                raise FTSConfigurationError(
                    f"{where} is under the composable `replicate()`, which builds its `DistributedDataParallel` at the"
                    " first forward, after the controller has frozen the model, so parameters thawed in later phases"
                    " would never be gradient-synchronized. Shard it with FSDP2 (`fully_shard`) instead."
                )
        if isinstance(self._optimizer_src, core_optim.ZeroRedundancyOptimizer):
            raise FTSConfigurationError(
                "`ZeroRedundancyOptimizer` is not supported by `FTSController`; shard with FSDP2 instead."
            )

    def _resolve_view(self) -> BoundView:
        view = resolve_view(self.model, self._view_src)
        if view.root is not self.model:
            raise FTSConfigurationError(
                f"The view's `root` ({type(view.root).__name__}) must be the module given to the controller"
                f" ({type(self.model).__name__}): the optimizer and checkpoints see the root's parameters."
            )
        support = view.support()
        if not support.trainable:
            raise FTSConfigurationError(
                f"The model cannot be fine-tuned through {type(view.view).__name__}: {support.reason}")
        return view

    def _load_schedule(self) -> dict:
        src = self._schedule_src
        if src is None:
            assert self._view is not None
            schedule = view_default_schedule(self._view)
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
        max_epoch_wm = -1
        for depth in schedule:
            schedule[depth].setdefault("max_transition_epoch", -1)
            core_schedule.parse_phase_lr(schedule, depth, self.base_max_lr)
            if depth > 0:
                max_epoch_wm = core_schedule.update_epoch_watermark(schedule, depth, max_epoch_wm,
                                                                    epoch_transitions_only=not self._user_policies)
        # compile once through the view: each phase's `params` become the physical parameters it thaws, and the
        # logical names it resolved are kept for display
        assert self._view is not None
        self._compiled = compile_schedule(schedule, self._view)
        for cp in self._compiled.phases:
            self._logical_params[cp.depth] = cp.logical
            schedule[cp.depth]["params"] = list(cp.thawed)
        self._epoch_watermark = max_epoch_wm
        if not self._user_policies:
            final = core_schedule.resolve_max_depth(self._max_depth_arg, len(schedule))
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
                f"`FTSController` steps lr schedulers with `interval`, `frequency` and `monitor` only; remove"
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

    def _warn_unsynchronized_params(self) -> None:
        """Warn about scheduled parameters no FSDP2 module manages in a data-parallel run (a plan that does not
        shard the root module): nothing synchronizes their gradients, so each rank would train its own copy."""
        if self._host.world_size < 2 or not torch.distributed.is_available():
            return
        from torch.distributed.fsdp import FSDPModule

        fsdp_modules = {name for name, m in self.model.named_modules() if isinstance(m, FSDPModule)}
        if not fsdp_modules:
            return

        def managed(param_name: str) -> bool:  # FSDP2 manages a parameter of the module it shards or any descendant
            parts = param_name.split(".")[:-1]
            return any(".".join(parts[:i]) in fsdp_modules for i in range(len(parts) + 1))

        scheduled = {n for d, phase in self._schedule.items() if d <= self._max_depth for n in phase["params"]}
        # a tensor-parallel DTensor outside every FSDP module is not averaged across data-parallel ranks either
        plain = [n for n, _ in self.model.named_parameters() if n in scheduled and not managed(n)]
        if plain:
            shown = ", ".join(plain[:5]) + (f" and {len(plain) - 5} more" if len(plain) > 5 else "")
            rank_zero_warn(
                f"FSDP2 does not manage these scheduled parameters, so their gradients are not synchronized across"
                f" ranks: {shown}. Apply `fully_shard` to the root module as well (last), so every parameter is"
                " sharded."
            )

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
            self._refuse_frozen_roles_in(src)
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
        :meth:`train_end`.

        Leaving the loop early (``break``, or an exception in its body) does not end training, so no final restore, end
        hooks or gates run, unless :attr:`should_stop` was already set (``if fts.should_stop: break`` ends training
        as usual). Call :meth:`train_end` yourself if another deliberate early exit should end the run. An epoch is
        delimited by :meth:`epoch_start`, which :meth:`epochs` calls; a loop of its own must call it too.

        As with the Lightning callback, when the final phase sets ``max_transition_epoch`` the run lasts the greater of
        ``max_epochs`` and the schedule's highest ``max_transition_epoch`` (with a warning).
        """
        self._require_setup()
        # as the Lightning callback does, the schedule's last phase (not `max_depth`'s) decides the extension
        extended = core_schedule.final_phase_max_epochs(self._schedule, len(self._schedule) - 1,
                                                        self._epoch_watermark, max_epochs)
        max_epochs = extended if extended is not None else max_epochs
        self._host.set_max_epochs(max_epochs)
        while self._host.current_epoch < max_epochs and not self._should_stop:
            self.epoch_start()
            if self._should_stop:  # a gate vetoed the phase at this boundary
                break
            try:
                yield self._host.current_epoch
            except GeneratorExit:
                # the loop was left early: end training only if it had already decided to stop (the common
                # `if fts.should_stop: break`); a failure or an arbitrary early exit does not end it
                if self._should_stop:
                    self.train_end()
                raise
            if not self._epoch_end_called:
                self.epoch_end()
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
        self._last_metrics = dict(metrics or {})
        decision, reason = self._evaluate("epoch", metrics, completed_epochs=self._host.current_epoch + 1)
        # with several ranks, every rank joins the agreement, even one that reported no metrics this epoch
        if self._checkpoint is not None and (metrics is not None or self._host.world_size > 1):
            before = self._checkpoint.state_dict()
            if self._observe_best(metrics):
                try:
                    self._write_checkpoint(self._checkpoint.path_for("best"))
                except BaseException:
                    # a failed write keeps the previous best (and whatever it would have replaced)
                    self._checkpoint.load_state_dict(before)
                    raise
                self._checkpoint_hook("prune")
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
        """End training (once): with gates registered, restore the best (if configured) and run the gates on the
        last phase, after its end hooks; a veto rolls back to the best as of the phase's start."""
        if not self._is_setup or self._finished:
            return
        self._finished = True
        if self._depth in self._ended_depths:
            return  # a veto already closed this phase
        if self._gates and self._restore_best:
            self._restore_best_state()
        self._end_phase(self._depth)
        self._run_gates(self._depth, final=True)

    def step_lr_schedulers(self, interval: Literal["step", "epoch"], metrics: Mapping[str, Any] | None = None) -> None:
        """Step every lr scheduler configured for ``interval`` (honoring its ``frequency``, and passing its
        ``monitor`` metric to schedulers such as ``ReduceLROnPlateau``).

        For ``"epoch"``, call it once per epoch, before or after :meth:`epoch_end`: the epoch being run counts as
        completed either way, so a ``frequency`` of ``n`` steps after every ``n``-th epoch, as under Lightning. For
        ``"step"``, call it after each optimizer step; ``frequency`` counts optimizer steps since the run began
        (Lightning counts batches within the epoch, which differs only when an epoch's step count is not a
        multiple of the frequency).
        """
        self._require_setup()
        if interval == "step":
            count = self._host.global_step
        else:
            count = self._host.current_epoch if self._epoch_end_called else self._host.current_epoch + 1
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
            # no policy evaluates this unit (the same configuration on every rank), so there is nothing to agree on
            return Decision.CONTINUE, None
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
        """Close phase ``k`` and open ``k + 1``: restore the best (if configured), run phase ``k``'s end hooks and
        its gates (a veto stops here), advance and pin the boundary best, run the new phase's start hooks, then
        thaw and configure the optimizer for it."""
        prev = self._depth
        core_phase.reshard_fsdp_modules(self.model)
        schedulers = [s for s, _ in self._host.get_lr_schedulers()]
        pre_state = core_optim.capture_lr_state(schedulers[0] if schedulers else None, self.optimizer)
        restored_depth = self._restore_best_state() if self._restore_best else None
        self._end_phase(prev)
        if not self._run_gates(prev, final=False):
            return
        self._depth += 1
        self._phase_start_epoch, self._phase_start_step = self._host.current_epoch, self._host.global_step
        self._fingerprints.setdefault(self._depth, [])
        if self._checkpoint is not None:
            self._pin_boundary(self._depth)
        self._run_hooks("start", self._depth)
        self._apply_promotions(self._depth)
        if restored_depth is not None:
            self._configure_phase(self._depth, restored_depth, depth_sync=True, pre_state=pre_state)
        else:
            self._configure_phase(self._depth, self._depth - 1, depth_sync=False, pre_state=pre_state)
        if self._compile_reset or (self._compile_reset is None and _is_compiled(self.model)):
            self._host.on_compile_reset()
        self._policy.reset(self._depth)
        rank_zero_info(f"Fine-tuning advanced from phase {prev} to phase {self._depth}"
                       f"{f' ({reason})' if reason else ''}.")

    def _apply_promotions(self, depth: int) -> None:
        """Promote the tensors phase ``depth`` names in its ``promote`` key (when the phase starts, after its start
        hooks and before its thaw), through the view's ``promote`` or the default materialization, and record each
        promoted tensor's value, which restoring a checkpoint saved before the promotion puts back."""
        assert self._compiled is not None and self._view is not None
        promoted = self._compiled.promotions.get(depth)
        if not promoted or any(r["depth"] == depth for r in self._promotions):  # none, or applied before a reload
            return
        by_role: dict[str, list[str]] = {}
        for name, role in promoted.items():
            by_role.setdefault(role, []).append(name)
        for role, names in by_role.items():
            before = dict(self.model.named_parameters(remove_duplicate=False))
            unsaved = set(self._view.checkpoint_state_dict())
            self._view.promote(names, role)
            after = dict(self.model.named_parameters(remove_duplicate=False))
            # every checkpoint key the promotion adds (the promoted tensors, and aliases a materialization untied),
            # by its value now: a checkpoint saved before the promotion lacks it
            state = self._view.checkpoint_state_dict()
            added = [k for k in state if k not in unsaved]
            if not added:
                raise FTSConfigurationError(
                    f"After promoting {sorted(names)}, {type(self._view.view).__name__}'s checkpoint state still leaves"
                    " them out, so they would never be checkpointed.")
            for k in added:
                self._promoted_values[k] = state[k].detach().to("cpu", copy=True)
            self._promotions.append({"depth": depth, "names": sorted(names), "from": "instrument", "to": role,
                                     "materialized": sorted(n for n in names if after[n] is not before[n])})
            rank_zero_info(f"Phase {depth} promoted {sorted(names)} from `instrument` to `{role}`.")

    def _with_promoted_values(self, model_state: Mapping[str, Any]) -> Mapping[str, Any]:
        """``model_state`` (a checkpoint's), completed with the value at promotion of each tensor promoted since
        the checkpoint was saved: an ``instrument`` tensor then, it was not checkpointed."""
        missing = [k for k in self._promoted_values if k not in model_state]
        return {**model_state, **{k: self._promoted_values[k] for k in missing}} if missing else model_state

    def _refuse_unsupported_promotion(self) -> None:
        """Promotion needs an unsharded model, and a view that translates checkpoint keys must own promotion, since
        only it can report the promoted tensors in its checkpoint state."""
        assert self._compiled is not None and self._view is not None
        phases = sorted(self._compiled.promotions)
        if self._sharded:
            raise FTSConfigurationError(
                f"Phases {phases} `promote` tensors of a sharded model, which this release does not support.")
        translating = [m for m in ("checkpoint_state_dict", "load_checkpoint_state_dict") if self._view.overrides(m)]
        if translating and not self._view.overrides("promote"):
            raise FTSConfigurationError(
                f"Phases {phases} `promote` tensors, but {type(self._view.view).__name__} overrides {translating} and"
                " not `promote`: a view that translates checkpoint keys must define `promote`, so its checkpoint state"
                " includes the tensors it promotes.")

    def _configure_phase(self, depth: int, restored_depth: int, depth_sync: bool,
                         pre_state: tuple[dict, list] | None) -> None:
        """Thaw ``depth``'s parameters (and, after a restore or an optimizer reinitialization, every phase the
        optimizer lacks) and configure the optimizer and lr schedulers for them, as the callback's ``step_pg``
        does."""
        # after any phase hooks and gates, which may have run a forward-only pass, and on `load_state_dict`'s replay
        core_phase.reshard_fsdp_modules(self.model)
        optimizer = self.optimizer
        schedulers = [s for s, _ in self._host.get_lr_schedulers()]
        new_optimizer_cfg, phases = core_phase.thaw_plan(self._schedule, depth, restored_depth, depth_sync=depth_sync,
                                                         reinit_optim_cfg=self.reinit_optim_cfg)
        for i, phase in phases:
            if i > depth:
                continue
            self._thaw_phase(i)
            if new_optimizer_cfg and i == 0:
                # a reinitialized optimizer starts from phase 0's groups; later phases are replayed onto it
                optimizer = self._reinit_optimizer(new_optimizer_cfg, phase["params"])
                schedulers = [s for s, _ in self._host.get_lr_schedulers()]
                continue
            if i == depth and depth > 0 and pre_state is not None:
                core_optim.restore_latest_lr_state(optimizer, schedulers, *pre_state)
            core_optim.add_optimizer_groups(
                self.model, optimizer, phase["params"], no_decay=self.no_decay,
                lr=phase.get("lr", optimizer.defaults["lr"]), apply_lambdas=self.apply_lambdas_new_pgs,
                schedulers=schedulers, pg_overrides=phase.get("pg_overrides"),
            )
            new_scheduler_cfg = core_phase.lr_reinit_cfg_for(phase, depth, self.reinit_lr_cfg)
            if new_scheduler_cfg:
                self._reinit_lr_scheduler(new_scheduler_cfg, optimizer)
                schedulers = [s for s, _ in self._host.get_lr_schedulers()]
            else:
                core_optim.warn_unapplied_lr_lambdas(schedulers, self.apply_lambdas_new_pgs)

    def _pin_boundary(self, depth: int, initial_path: str | None = None) -> None:
        assert self._checkpoint is not None
        self._checkpoint.pin_boundary(depth, initial_path=initial_path, copy=self._host.is_global_zero)
        self._host.barrier("fts_boundary_pinned")
        self._checkpoint_hook("prune")

    def _checkpoint_hook(self, name: Literal["prune", "mark_resume_point"], **kwargs: Any) -> None:
        """Call the checkpoint policy's optional ``name`` hook, if it has one; only global rank zero removes
        files."""
        hook = getattr(self._checkpoint, name, None)
        if hook is not None:
            hook(remove=self._host.is_global_zero, **kwargs)

    def _verify_checkpoint_resume(self) -> None:
        """Have the checkpoint policy check, on global rank zero, that the checkpoints a loaded state refers to
        still exist, then share the outcome (and the policy's state, which a check may trim) with every rank."""
        cp = self._checkpoint
        verify = getattr(cp, "verify_resume", None)
        if cp is None or verify is None:
            return
        error = None
        if self._host.is_global_zero:
            try:
                verify()
            except FileNotFoundError as err:
                error = str(err)
        if self._host.world_size > 1:
            error, state = self._host.broadcast_object((error, cp.state_dict()))
            cp.load_state_dict(state)
        if error is not None:
            raise FileNotFoundError(error)

    def _end_phase(self, depth: int) -> None:
        if depth not in self._ended_depths:
            self._ended_depths.add(depth)
            self._run_hooks("end", depth)

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
        assert self._view is not None
        core_phase.thaw_params(self.model, params, translation_func=self._view.to_logical, init_thaw=init_thaw)
        self._refuse_trainable_frozen_roles()
        if not self.frozen_bn_track_running_stats:
            owners = {owner_module(n) for n in params}
            for name, mod in self.model.named_modules():
                if name in owners and isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.track_running_stats = True

    def _refuse_frozen_roles_in(self, optimizer: Optimizer) -> None:
        """An optimizer instance may not hold a `fitted` or `instrument` parameter, even with
        ``enforce_phase0_params=False``."""
        roles = self.roles
        names = {id(p): n for n, p in self.model.named_parameters()}
        held = sorted(names[id(p)] for g in optimizer.param_groups for p in g["params"]
                      if id(p) in names and is_frozen_role(roles.get(names[id(p)], "base")))
        if held:
            raise FTSConfigurationError(
                f"The optimizer holds parameters whose role is never optimized (`fitted` or `instrument`): {held}.")

    def _refuse_trainable_frozen_roles(self) -> None:
        """Guard: no `fitted` or `instrument` tensor may require grad (a view's coupled tensors could leak one)."""
        roles = self.roles
        leaked = sorted(n for n, p in self.model.named_parameters()
                        if p.requires_grad and is_frozen_role(roles.get(n, "base")))
        if leaked:
            raise FTSConfigurationError(
                f"Tensors whose role is never thawed (`fitted` or `instrument`) require grad: {leaked}.")

    # ------------------------------------------------------------------------------------------------------------------
    # checkpoints, gates and registered state
    # ------------------------------------------------------------------------------------------------------------------
    def register_stateful(self, name: str, obj: Any, *, restore_on_best: bool = False) -> None:
        """Include ``obj`` (anything with ``state_dict``/``load_state_dict``, such as a sampler or a data position)
        in :meth:`save` and :meth:`load`, and in best checkpoints.

        With ``restore_on_best``, it is also restored when the
        best checkpoint is (at a phase boundary or by :meth:`restore_best`).
        """
        if not (hasattr(obj, "state_dict") and hasattr(obj, "load_state_dict")):
            raise FTSConfigurationError(f"`{name}` needs `state_dict()` and `load_state_dict()` to be registered.")
        self._statefuls[name] = (obj, restore_on_best)

    def add_gate(self, fn: Callable[[GateEvent], GateResult | bool], *, rank_zero_only: bool = False,
                 isolate_rng: bool | None = None) -> RemovableHandle:
        """Register a gate: at each phase boundary (after the phase's end hooks, and with the best checkpoint restored
        when ``restore_best`` is set) and at the end of training, ``fn(event)`` accepts or vetoes the phase.

        A veto on any rank stops training and restores the best checkpoint as of the vetoed phase's start (when the
        checkpoint policy has one). ``fn`` returns a :class:`~finetuning_scheduler.core.transitions.GateResult` or a
        ``bool`` (``True`` accepts). With ``rank_zero_only``, it runs on global rank zero and its result is broadcast.

        Returns:
            A handle whose ``remove()`` unregisters the gate.
        """
        handle = RemovableHandle(self._gates)
        self._gates[handle.id] = (fn, rank_zero_only, isolate_rng)
        return handle

    def _run_gates(self, depth: int, final: bool) -> bool:
        if not self._gates:
            return True
        cp = self._checkpoint
        event = GateEvent(depth=depth, final=final, best_score=cp.best_score if cp else None,
                          best_path=cp.best_path if cp else None, best_depth=cp.best_depth if cp else None,
                          metrics=dict(self._last_metrics), phase_config=self._phase_configs.get(depth),
                          metadata=self._metadata)
        accepted, reasons = True, []
        for fn, rank_zero_only, isolate in list(self._gates.values()):
            result: GateResult | None = None
            if not rank_zero_only or self._host.is_global_zero:
                isolate = self.isolate_rng if isolate is None else isolate
                with self._suspend_step_counting(), (fork_rngs(self._host.rng_devices()) if isolate else nullcontext()):
                    raw = fn(event)
                result = raw if isinstance(raw, GateResult) else GateResult(accept=bool(raw))
            if rank_zero_only:
                result = self._host.broadcast_object(result)
            assert result is not None
            self._gate_records.append({"depth": depth, "final": final, "gate": describe_callable(fn),
                                       "accept": result.accept, "reason": result.reason,
                                       "record": dict(result.record)})
            if not result.accept:
                accepted = False
                reasons.append(result.reason or describe_callable(fn))
        if not self._host.reduce_any(not accepted):
            return True
        # every rank reports the same reasons, whichever rank vetoed
        gathered = [r for rank_reasons in self._host.all_gather_object(reasons) for r in rank_reasons]
        reasons = list(dict.fromkeys(gathered))
        boundary = cp.boundary_path(depth) if cp else None
        if boundary is not None:
            self._restore_checkpoint(boundary, restore_optimizer=True)
        self._ended_depths.add(depth)
        self._stop(f"veto: phase {depth} was rejected ({'; '.join(reasons)})"
                   + ("" if boundary else "; no earlier checkpoint to restore"))
        return False

    @property
    def _sharded(self) -> bool:
        if self._distributed is not None:
            return self._distributed.sharded
        return any(type(p).__name__ == "DTensor" for p in self.model.parameters())

    def _validate_layout(self) -> None:
        """Report the distributed adapter's findings on the compiled schedule (after resharding, so a forward-only
        pass before setup cannot show a layout the run never has); an ``error`` refuses."""
        assert self._compiled is not None and self._distributed is not None
        for feedback in self._distributed.validate(self.model, {cp.depth: cp.thawed for cp in self._compiled.phases}):
            if feedback.level == "error":
                raise FTSConfigurationError(feedback.message)
            (rank_zero_warn if feedback.level == "warning" else rank_zero_info)(feedback.message)

    def _instrument_keys(self, keys: Any) -> set[str]:
        return instrument_state_keys(self.model, self.roles, keys)

    def _training_payload(self, with_tensors: bool) -> dict[str, Any]:
        named = dict(self.model.named_parameters())
        include = getattr(self._checkpoint, "include", "all")
        optimizer = self.optimizer
        payload: dict[str, Any] = {
            "version": STATE_VERSION,
            "depth": self._depth,
            "include": include,
            "sharded": not with_tensors,
            "optimizer_class": type(optimizer).__qualname__,
            "param_groups": core_phase.map_param_groups(optimizer.param_groups, {p: n for n, p in named.items()}),
            "lr_schedulers": [s.state_dict() for s in self.lr_schedulers],
            "statefuls": {name: obj.state_dict() for name, (obj, _) in self._statefuls.items()},
            "schedule_hash": config_hash({"schedule": self._schedule}),
            "config_hash": self.config_hash,
            "versions": self.versions(),
            "metadata": {**self._metadata, **getattr(self._checkpoint, "metadata", {}),
                         "artifacts": deepcopy(self._artifacts),
                         "phase_config": self._schedule[self._depth].get("phase_config"),
                         "gate_records": list(self._gate_records), "fingerprints": deepcopy(self._fingerprints),
                         "promotions": deepcopy(self._promotions)},
        }
        if with_tensors:
            assert self._view is not None
            model_state = self._view.checkpoint_state_dict()
            if include == "scheduled":
                roles = self.roles
                scheduled = {n for phase in self._schedule.values() for n in phase["params"]}
                model_state = {k: v for k, v in model_state.items()
                               if k in scheduled or k not in named or roles.get(k) == "fitted"}
            payload["model"] = model_state
            payload["optimizer"] = optimizer.state_dict()
        return payload

    def _write_checkpoint(self, path: str, extra: Mapping[str, Any] | None = None,
                          rng_state: dict | None = None) -> str:
        """Write the training state to ``path``: one file (rank zero) for an unsharded model; for a sharded model,
        a directory written by every rank through DCP, with the non-tensor state in ``fts_meta.pt`` and each rank's
        random generator state (if given) beside it."""
        path = os.path.normpath(os.fspath(path))  # `ckpt/` and `ckpt` are one checkpoint
        core_phase.reshard_fsdp_modules(self.model)
        # registered state (a data-loader position, for example) and random streams are per rank
        statefuls = {name: obj.state_dict() for name, (obj, _) in self._statefuls.items()}
        per_rank = (self._host.all_gather_object({"statefuls": statefuls, "rng": rng_state})
                    if self._host.world_size > 1 else None)
        if self._sharded:
            assert self._distributed is not None
            # written beside `path` and moved over it once complete, so an interrupted save keeps the old one
            # (a fixed name, so a later save to the same path clears what an interrupted one left behind)
            tmp = self._host.broadcast_object(f"{path}.tmp")
            if self._host.is_global_zero:
                remove_checkpoint(tmp)
            self._host.barrier("fts_checkpoint_tmp")
            model_sd, optim_sd = self._distributed.state_dict(self.model, self.optimizer, exclude=self._instrument_keys,
                                                              quiesce=self._suspend_step_counting)
            optim_saved = optim_sd is not None
            self._distributed.save({"model": model_sd, "optim": optim_sd}, tmp)
            if self._host.is_global_zero:
                save_payload({**self._training_payload(with_tensors=False), "optim_saved": optim_saved,
                              "per_rank": per_rank, **(extra or {})}, os.path.join(tmp, META_FILE))
            if rng_state is not None:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                save_payload({"rng": rng_state}, os.path.join(tmp, f"rng_rank{rank}.pt"))
            self._host.barrier("fts_checkpoint_staged")
            if self._host.is_global_zero:
                replace_checkpoint(tmp, path)
        elif self._host.is_global_zero:
            payload = {**self._training_payload(with_tensors=True), "per_rank": per_rank, **(extra or {})}
            if rng_state is not None:
                payload["rng"] = rng_state
            save_payload(payload, path)
        self._host.barrier("fts_checkpoint_written")
        return path

    def _refuse_sharded_key_translation(self) -> None:
        assert self._view is not None
        translating = [m for m in ("checkpoint_state_dict", "load_checkpoint_state_dict") if self._view.overrides(m)]
        if translating:
            raise FTSConfigurationError(
                f"{type(self._view.view).__name__} overrides {translating}; checkpoint key translation is not supported"
                " for sharded models yet.")

    @staticmethod
    def _read_meta(path: str) -> dict[str, Any]:
        return load_payload(os.path.join(path, META_FILE)) if os.path.isdir(path) else load_payload(path)

    def _restore_checkpoint(self, path: str, restore_optimizer: bool, statefuls: Literal["all", "on_best"] = "on_best",
                            meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Restore the model (and, when its type still matches, the optimizer and lr scheduler state) and
        registered state from a checkpoint written by :meth:`_write_checkpoint`; return its metadata."""
        core_phase.reshard_fsdp_modules(self.model)
        meta = dict(meta) if meta is not None else self._read_meta(path)
        optimizer = self.optimizer
        same_optimizer = meta["optimizer_class"] == type(optimizer).__qualname__
        if restore_optimizer and not same_optimizer:
            rank_zero_info(
                f"The checkpoint's optimizer ({meta['optimizer_class']}) predates an optimizer reinitialization;"
                " restoring the model state only."
            )
        with_optimizer = restore_optimizer and same_optimizer
        if with_optimizer:
            # match the saved parameter groups first, so the saved optimizer state lines up with them
            optimizer.param_groups = core_phase.map_param_groups(meta["param_groups"],
                                                                 dict(self.model.named_parameters()))
        if meta.get("sharded"):
            assert self._distributed is not None
            if with_optimizer:
                optimizer.state.clear()  # a checkpoint saved before any step restores an optimizer with no state
            # a sharded checkpoint loads through DCP whatever this model's layout (an unsharded model included), and
            # instrument tensors are never saved, so they are the only model keys a load may miss
            reader = self._distributed if self._distributed.sharded else DTensorAdapter()
            reader.load_state(path, self.model, optimizer,
                                         load_optimizer=with_optimizer and meta.get("optim_saved", True),
                                         exclude=self._instrument_keys, quiesce=self._suspend_step_counting)
        else:
            if with_optimizer:
                optimizer.load_state_dict(meta["optimizer"])
            assert self._view is not None
            self._view.load_checkpoint_state_dict(self._with_promoted_values(meta["model"]),
                                                  strict=meta["include"] == "all")
        if with_optimizer:
            for sched, state_ in zip(self.lr_schedulers, meta["lr_schedulers"]):
                sched.load_state_dict(state_)
        saved_statefuls = self._for_this_rank(meta, "statefuls") or meta["statefuls"]
        for name, (obj, restore_on_best) in self._statefuls.items():
            if name in saved_statefuls and (statefuls == "all" or restore_on_best):
                obj.load_state_dict(saved_statefuls[name])
        return meta

    def _for_this_rank(self, meta: Mapping[str, Any], key: str) -> Any:
        """This rank's entry of a checkpoint's per-rank state (``None`` for a single-process checkpoint)."""
        per_rank = meta.get("per_rank")
        if not per_rank:
            return None
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return per_rank[rank][key] if rank < len(per_rank) else None

    def _observe_best(self, metrics: Mapping[str, Any] | None) -> bool:
        """Offer ``metrics`` to the checkpoint policy (``None`` counts as no improvement here).

        Saving is collective, so every rank must agree: rank zero's decision is used, and its policy state is shared, so
        the ranks keep one record of the best.
        """
        assert self._checkpoint is not None
        improved = self._checkpoint.observe(metrics, self._depth) if metrics is not None else False
        if self._host.world_size > 1:
            improved, state = self._host.broadcast_object((improved, self._checkpoint.state_dict()))
            self._checkpoint.load_state_dict(state)
        return bool(improved)

    def _restore_best_state(self) -> int | None:
        """Restore the best checkpoint's model, optimizer and lr scheduler state (as the callback does at a phase
        boundary); return the depth it was saved at, or ``None`` when there is none yet."""
        cp = self._checkpoint
        if cp is None or cp.best_path is None:
            return None
        self._host.barrier("fts_restore_best")
        meta = self._restore_checkpoint(cp.best_path, restore_optimizer=True)
        self._restored_count += 1
        return meta["depth"]

    def restore_best(self) -> bool:
        """Load the best checkpoint's model state (and registered state marked ``restore_on_best``), for example
        before a final evaluation. The schedule's depth and the optimizer are left as they are.

        Returns:
            Whether a best checkpoint existed.
        """
        self._require_setup()
        cp = self._checkpoint
        if cp is None or cp.best_path is None:
            return False
        self._host.barrier("fts_restore_best")
        self._restore_checkpoint(cp.best_path, restore_optimizer=False)
        return True

    def export_best(self, fn: Callable[[dict[str, Tensor], dict[str, Any], str | None], Any] | None = None, *,
                    roles: Sequence[str] = ("adapter",), path: str | os.PathLike | None = None) -> Any:
        """Export the best checkpoint's tensors whose role matches ``roles`` (for example only the adapters, so a
        large base and the optimizer state need not be kept).

        The tensors come from the best checkpoint, not the live model; for a sharded model, global rank zero reads
        their full values from the checkpoint directory. Global rank zero then writes
        ``{"tensors", "roles", "metadata"}`` to ``path`` with ``torch.save`` or, given ``fn``, calls
        ``fn(tensors, metadata, path)`` instead (to write another format, for example).

        Args:
            fn: Writes the export itself; its return value is returned on global rank zero.
            roles: Role kinds (``"adapter"``) or selectors (``"role:adapter:char"``) to export.
            path: Where to write; by default ``best_export.pt`` beside the checkpoints.

        Returns:
            On global rank zero, ``fn``'s result or the path written; on other ranks, ``None`` with ``fn``, else the
            path.

        Raises:
            FTSConfigurationError: If there is no best checkpoint yet, no tensor matches ``roles``, a matched tensor is
                never checkpointed (``instrument``), or the view translates checkpoint keys.
        """
        self._require_setup()
        cp = self._checkpoint
        if cp is None or cp.best_path is None:
            raise FTSConfigurationError("There is no best checkpoint to export yet.")
        assert self._view is not None
        if self._view.overrides("checkpoint_state_dict"):
            raise FTSConfigurationError(
                f"{type(self._view.view).__name__} translates checkpoint keys, which `export_best` does not support"
                " yet.")
        selectors = [r if is_role_selector(r) else f"{SELECTOR_PREFIX}{r}" for r in roles]
        effective = self.roles
        names = [n for n in itertools.chain((n for n, _ in self.model.named_parameters()),
                                            (n for n, _ in self.model.named_buffers()))
                 if any(selector_matches(sel, effective.get(n, "base")) for sel in selectors)]
        if not names:
            raise FTSConfigurationError(f"No tensor's role matches {list(roles)}.")
        never_saved = sorted(n for n in names if role_kind(effective.get(n, "base")) == "instrument")
        if never_saved:
            raise FTSConfigurationError(f"`instrument` tensors are never checkpointed, so they cannot be exported: "
                                        f"{never_saved}.")
        dest = os.fspath(path) if path is not None else os.path.join(getattr(cp, "dirpath", "."), "best_export.pt")
        result: Any = None if fn is not None else dest
        self._host.barrier("fts_export_best")
        if self._host.is_global_zero:
            tensors = self._read_checkpoint_tensors(cp.best_path, names)
            metadata = {**self._metadata, "artifacts": deepcopy(self._artifacts), "source": cp.best_path,
                        "best_depth": cp.best_depth,
                        "best_score": float(cp.best_score) if cp.best_score is not None else None,
                        "config_hash": self.config_hash, "versions": self.versions()}
            export_roles = {n: effective.get(n, "base") for n in names}
            if fn is not None:
                result = fn(tensors, {**metadata, "roles": export_roles}, os.fspath(path) if path is not None else None)
            else:
                save_payload({"tensors": tensors, "roles": export_roles, "metadata": metadata}, dest)
        self._host.barrier("fts_export_best_written")
        return result

    def _read_checkpoint_tensors(self, path: str, names: Sequence[str]) -> dict[str, Tensor]:
        """The full values of ``names`` in the checkpoint at ``path`` (on this process alone, so call it on one
        rank), without touching the live model."""
        meta = self._read_meta(path)
        if not meta.get("sharded"):
            missing = [n for n in names if n not in meta["model"]]
            if missing:  # left out of a scheduled-only checkpoint (`include="scheduled"`)
                raise FTSConfigurationError(f"The best checkpoint does not hold {missing}.")
            return {n: meta["model"][n] for n in names}
        import torch.distributed.checkpoint as dcp

        saved = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
        missing = [n for n in names if f"model.{n}" not in saved]
        if missing:
            raise FTSConfigurationError(f"The best checkpoint does not hold {missing}.")
        model_state = {n: torch.empty(saved[f"model.{n}"].size, dtype=saved[f"model.{n}"].properties.dtype)
                       for n in names}
        dcp.load({"model": model_state}, checkpoint_id=path, no_dist=True)
        return model_state

    def save(self, path: str | os.PathLike) -> str:
        """Save everything needed to resume exactly: the controller's state, the model, optimizer, lr schedulers,
        registered state, the checkpoint policy's state and (with ``capture_rng``) every random generator's state.

        An unsharded model is saved to the file ``path`` (by global rank zero); a sharded (FSDP2/DTensor) model to the
        directory ``path``, written by every rank through DCP. Either way the new checkpoint replaces ``path`` only once
        it is complete. Each rank's registered state and random streams are saved, and restored to the same rank.

        Save inside the training loop for an exact resume. A state saved after :meth:`train_end` (for example after
        :meth:`epochs` returned) records what ending did: the final best-checkpoint restore and gates, when configured.
        Resuming it reopens the run, so the current phase's end hooks run again at its real end.
        """
        self._require_setup()
        rng_state = capture_rng_state(self._host.rng_devices()) if self.capture_rng else None
        path = self._write_checkpoint(os.fspath(path), extra={"controller": self.state_dict()}, rng_state=rng_state)
        self._checkpoint_hook("mark_resume_point")  # keep the checkpoints this save refers to until the next one
        return path

    def load(self, path: str | os.PathLike) -> None:
        """Resume from :meth:`save`: rebuild the optimizer and lr schedulers phase by phase up to the saved depth
        (replaying any reinitialization), then load every saved state. Call it after :meth:`setup`.

        Raises:
            FTSConfigurationError: If the checkpoint belongs to a different fine-tuning schedule.
        """
        self._require_setup()
        path = os.fspath(path)
        meta = self._read_meta(path)
        self.load_state_dict(meta["controller"])
        self._restore_checkpoint(path, restore_optimizer=True, statefuls="all", meta=meta)
        if meta.get("sharded"):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            rng_file = os.path.join(path, f"rng_rank{rank}.pt")
            if os.path.exists(rng_file):
                restore_rng_state(load_payload(rng_file)["rng"])
        elif self._for_this_rank(meta, "rng") is not None:
            restore_rng_state(self._for_this_rank(meta, "rng"))
        elif "rng" in meta:
            restore_rng_state(meta["rng"])

    # ------------------------------------------------------------------------------------------------------------------
    # hooks, fingerprints and step counting
    # ------------------------------------------------------------------------------------------------------------------
    def add_phase_hook(self, fn: Callable[[PhaseEvent], None], when: Literal["start", "end"] = "start", *,
                       isolate_rng: bool | None = None) -> RemovableHandle:
        """Call ``fn(event)`` whenever a phase starts (phase 0 at :meth:`setup`) or ends (the final phase at
        :meth:`train_end`). Hooks run in registration order, with the training random streams isolated unless
        ``isolate_rng`` (or the controller's default) is ``False``.

        Start hooks for phase 0 run at the end of :meth:`setup`, after phase 0 is thawed and the optimizer built. Start
        hooks for a later phase run at its boundary before its parameters are thawed and its parameter groups added
        (after any best-checkpoint restore, the previous phase's end hooks and the gates), so they can prepare the
        model for the thaw; the new phase's parameters are listed in ``event.controller.schedule[event.depth]``.

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
                           phase_config=self._phase_configs.get(depth), metadata=self._metadata, controller=self,
                           physical_params=tuple(self._schedule[depth]["params"]),
                           logical_params=self._logical_params.get(depth, ()))
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
    def suspend_step_counting(self) -> Iterator[None]:
        """Do not count optimizer steps taken inside this context.

        Some state-dict utilities step the optimizer to initialize its state (for example
        ``torch.distributed.checkpoint.state_dict.get_state_dict`` on an optimizer with no state yet). Wrap your own
        calls to them in this context; the controller's own checkpointing already does.
        """
        prev, self._counting_suspended = self._counting_suspended, True
        try:
            yield
        finally:
            self._counting_suspended = prev

    _suspend_step_counting = suspend_step_counting

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
            "view_digest": self._compiled.digest if self._compiled is not None else None,
            "config_hash": self.config_hash,
            "gate_records": list(self._gate_records),
            "promotions": deepcopy(self._promotions),
            # the promoted tensors' values at promotion, which a resume cannot recompute (start hooks are not replayed)
            "promoted_values": dict(self._promoted_values),
            "ended_depths": sorted(self._ended_depths),
            "checkpoint": self._checkpoint.state_dict() if self._checkpoint is not None else None,
            "last_metrics": dict(self._last_metrics),
            "finished": self._finished,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore state saved by :meth:`state_dict`, rebuilding the optimizer and lr schedulers phase by phase up
        to the saved depth (replaying any reinitialization). Call it after :meth:`setup` and before loading the
        optimizer's and lr schedulers' state (:meth:`load` does all of this).

        Raises:
            FTSConfigurationError: If the saved state belongs to a different fine-tuning schedule.
            FileNotFoundError: If the checkpoint policy's best checkpoint or rollback point no longer exists.
        """
        self._require_setup()
        if state_dict["schedule_hash"] != config_hash({"schedule": self._schedule}):
            raise FTSConfigurationError("The saved state belongs to a different fine-tuning schedule.")
        saved_digest = state_dict.get("view_digest")
        if saved_digest is not None and self._compiled is not None and saved_digest != self._compiled.digest:
            raise FTSConfigurationError(
                "The saved state was compiled through a view that maps the schedule to different tensors or roles; it"
                " would thaw different parameters on resume.")
        if state_dict["config_hash"] != self.config_hash:
            rank_zero_warn("The saved state was produced with a different controller configuration.")
        if self._checkpoint is not None and state_dict.get("checkpoint") is not None:
            self._checkpoint.load_state_dict(state_dict["checkpoint"])
            self._verify_checkpoint_resume()
            # keep what this state refers to until the next save, beside what the latest save refers to
            self._checkpoint_hook("mark_resume_point", keep_previous=True)
        for depth in range(1, state_dict["depth"] + 1):
            self._depth = depth
            self._apply_promotions(depth)  # before the state loads: promoted tensors are checkpointed only from then
            self._configure_phase(depth, depth - 1, depth_sync=False, pre_state=None)
        self._promoted_values.update(state_dict.get("promoted_values", {}))
        self._depth = state_dict["depth"]
        self._host.load_counters(state_dict["epoch"], state_dict["global_step"])  # type: ignore[attr-defined]
        self._phase_start_epoch, self._phase_start_step = state_dict["phase_start_epoch"], state_dict[
            "phase_start_step"]
        self._pending, self._pending_reason = Decision(state_dict["pending"]), state_dict["pending_reason"]
        self._should_stop, self._stop_reason = state_dict["should_stop"], state_dict["stop_reason"]
        self._policy.load_state_dict(state_dict["policy"])
        self._fingerprints = deepcopy(state_dict["fingerprints"])
        self._gate_records = list(state_dict.get("gate_records", []))
        self._ended_depths = set(state_dict.get("ended_depths", []))
        if state_dict.get("finished") and not self._should_stop:
            # saved after `train_end`: resuming reopens the run, so the phase's end hooks run again at its real end
            self._ended_depths.discard(self._depth)
        self._last_metrics = dict(state_dict.get("last_metrics", {}))

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
            "checkpoint": self._checkpoint.config() if self._checkpoint is not None else None,
            "restore_best": self._restore_best,
            "isolate_rng": self.isolate_rng,
            "fingerprint_batches": self.fingerprint_batches,
            "metadata": self._metadata,
            **self._view_config(),
        }

    def _view_config(self) -> dict[str, Any]:
        """The view's description, when one other than the default identity view is used (omitted otherwise, so the
        hash of a view-less configuration is unchanged)."""
        bound = self._view
        if bound is None:
            try:
                bound = resolve_view(self.model, self._view_src)
            except FTSConfigurationError:  # an invalid view is refused at setup
                return {"view": {"class_path": describe_callable(type(self._view_src))}}
        roles = bound.view_roles()  # without promotions, so the hash is the same before and after one
        if type(bound.view) is CanonicalView and not roles:
            return {}
        support = bound.support()
        return {"view": {"class_path": describe_callable(type(bound.view)), "naming_scheme": support.naming_scheme,
                         "logical_names_are_views": support.logical_names_are_views,
                         "declared_roles": sorted(support.declared_roles), "roles": dict(sorted(roles.items()))}}

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
    def view(self) -> BoundView:
        """The (bound) view the schedule names the model's parameters through."""
        self._require_setup()
        assert self._view is not None
        return self._view

    @property
    def compiled_schedule(self) -> CompiledSchedule:
        """The schedule compiled through the view: each phase's physical parameters."""
        self._require_setup()
        assert self._compiled is not None
        return self._compiled

    @property
    def roles(self) -> dict[str, str]:
        """Physical parameter and buffer names mapped to roles (anything unlisted is ``base``)."""
        if self._compiled is not None:
            assert self._view is not None
            return {**self._compiled.roles, **self._view.promoted}
        return dict(self._view.roles()) if self._view is not None else {}

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
        """Whether training should stop, after an agreed ``STOP`` or a gate's veto (see :attr:`stop_reason`)."""
        return self._should_stop

    @property
    def stop_reason(self) -> str | None:
        """Why :attr:`should_stop` was set, or ``None`` while training continues."""
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

    @property
    def gate_records(self) -> list[dict[str, Any]]:
        return self._gate_records

    @property
    def restored_count(self) -> int:
        """How many times the best checkpoint was restored at a phase boundary."""
        return self._restored_count
