"""
Fine-Tuning Scheduler
=====================

Package used to implement multi-phase fine-tuned training schedules

"""
import importlib
from typing import TYPE_CHECKING, Any

from finetuning_scheduler.__about__ import *  # noqa: F401, F403

# eager: ``core.optim`` installs an import-time warning filter that must precede any optimizer import
from finetuning_scheduler.core import optim as _core_optim  # noqa: F401
from finetuning_scheduler.core.state import FTSState

# the PyTorch API (experimental): Lightning-free, so imported eagerly
from finetuning_scheduler.core.controller import FTSController, PhaseEvent
from finetuning_scheduler.core.host import LRStepConfig, TorchHost, TrainingHost
from finetuning_scheduler.core.transitions import (AnyOf, CallableTransition, Decision, EpochTransition, GateEvent,
                                                   GateResult, PlateauTransition, StepTransition, TransitionContext,
                                                   TransitionPolicy)
from finetuning_scheduler.core.views import (CanonicalView, TunableSupport, TunableView, TunableViewExtras,
                                             ViewDefaults)

if TYPE_CHECKING:  # pragma: no cover
    from finetuning_scheduler.lightning.fts import FinetuningScheduler
    from finetuning_scheduler.lightning.fts_supporters import (CallbackDepMixin, CallbackResolverMixin, FTSCheckpoint,
                                                     FTSEarlyStopping, ScheduleImplMixin, ScheduleParsingMixin)

# The Lightning-bound names, with the module defining each. They (and the Lightning-bound submodules) load on first
# access, so importing this package does not import Lightning.
_LAZY_ATTRS = {
    "FinetuningScheduler": "finetuning_scheduler.lightning.fts",
    "FTSCheckpoint": "finetuning_scheduler.lightning.fts_supporters",
    "FTSEarlyStopping": "finetuning_scheduler.lightning.fts_supporters",
    "ScheduleImplMixin": "finetuning_scheduler.lightning.fts_supporters",
    "ScheduleParsingMixin": "finetuning_scheduler.lightning.fts_supporters",
    "CallbackDepMixin": "finetuning_scheduler.lightning.fts_supporters",
    "CallbackResolverMixin": "finetuning_scheduler.lightning.fts_supporters",
}
_LAZY_SUBMODULES = ("fts", "fts_supporters", "lightning", "strategy_adapters", "types")
_LIGHTNING_DISTRIBUTIONS = ("lightning", "pytorch_lightning", "lightning_fabric", "lightning_utilities", "torchmetrics")


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        target = _LAZY_ATTRS[name]
    elif name in _LAZY_SUBMODULES:
        target = f"{__name__}.{name}"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        # import the full dotted name: resolving through the parent package's ``fromlist`` can recurse into this hook
        module = importlib.import_module(target)
    except ModuleNotFoundError as err:
        if (err.name or "").split(".")[0] not in _LIGHTNING_DISTRIBUTIONS:
            raise
        raise ImportError(
            f"`{__name__}.{name}` requires Lightning, which could not be imported ({err}). Install"
            " `finetuning-scheduler[lightning]` (or `finetuning-scheduler[pytorch-lightning]` for the standalone"
            " package) to use the Lightning integration."
        ) from err
    value = getattr(module, name) if name in _LAZY_ATTRS else module
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(_LAZY_SUBMODULES))


__all__ = [
    "AnyOf",
    "CallableTransition",
    "Decision",
    "EpochTransition",
    "FTSController",
    "GateEvent",
    "GateResult",
    "LRStepConfig",
    "PhaseEvent",
    "PlateauTransition",
    "StepTransition",
    "TorchHost",
    "TrainingHost",
    "TransitionContext",
    "TransitionPolicy",
    "CanonicalView",
    "TunableSupport",
    "TunableView",
    "TunableViewExtras",
    "ViewDefaults",
    "FTSState",
    "FTSCheckpoint",
    "FTSEarlyStopping",
    "ScheduleImplMixin",
    "ScheduleParsingMixin",
    "CallbackDepMixin",
    "CallbackResolverMixin",
    "FinetuningScheduler",
]
