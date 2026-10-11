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
"""Characterization tests for the Lightning-free core extraction (#50).

Each test pins behavior that downstream users, saved configurations or the Lightning integration rely on today. They
pass on the pre-extraction code and must keep passing, unmodified, through every extraction step.
"""
import importlib
import inspect
import logging
import os
import warnings

import pytest
import torch
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.utilities.exceptions import MisconfigurationException

import finetuning_scheduler
from finetuning_scheduler import FinetuningScheduler
from finetuning_scheduler.fts_supporters import ScheduleImplMixin

LEGACY_IMPORT_SURFACE = {
    "finetuning_scheduler": [
        "FTSState",
        "FTSCheckpoint",
        "FTSEarlyStopping",
        "ScheduleImplMixin",
        "ScheduleParsingMixin",
        "CallbackDepMixin",
        "CallbackResolverMixin",
        "FinetuningScheduler",
        "fts",
        "fts_supporters",
        "strategy_adapters",
        "types",
    ],
    "finetuning_scheduler.fts": ["FinetuningScheduler", "STRATEGY_ADAPTERS"],
    "finetuning_scheduler.fts_supporters": [
        "FTSState",
        "FTSCheckpoint",
        "FTSEarlyStopping",
        "ScheduleImplMixin",
        "ScheduleParsingMixin",
        "CallbackDepMixin",
        "CallbackResolverMixin",
        "UniqueKeyLoader",
        "STRATEGY_ADAPTERS",
        "_discover_strategy_adapters",
        "CALLBACK_DEP_PARENTS",
        "CALLBACK_ATTRS",
        "TARGET_CALLBACK_REF",
        "FTSCallbackDepType",
    ],
    "finetuning_scheduler.types": [
        "ParamGroupAddable",
        "FTSLRSchedulerType",
        "FTSLRSchedulerTypeTuple",
        "AutoStrEnum",
        "BaseCallbackDepType",
        "supported_lrs",
    ],
    "finetuning_scheduler.strategy_adapters": [
        "StrategyAdapter",
        "FSDPStrategyAdapter",
        "ModelParallelStrategyAdapter",
    ],
    "finetuning_scheduler.strategy_adapters.base": ["StrategyAdapter"],
    "finetuning_scheduler.strategy_adapters.model_parallel": ["ActCkptEnum", "ModelParallelStrategyAdapter"],
}

# static methods called on the class itself by tests and by downstream packages (e.g. ``save_schedule``)
SCHEDULE_IMPL_STATICMETHODS = ["save_schedule", "load_yaml_schedule", "sync", "_add_groups"]

RANK_ZERO_LOGGER = "lightning.pytorch.utilities.rank_zero"


@pytest.mark.parametrize("module_name", list(LEGACY_IMPORT_SURFACE), ids=list(LEGACY_IMPORT_SURFACE))
def test_legacy_import_surface(module_name):
    """Every module path and name consumers import today keeps resolving."""
    module = importlib.import_module(module_name)
    missing = [name for name in LEGACY_IMPORT_SURFACE[module_name] if not hasattr(module, name)]
    assert not missing, f"{module_name} no longer provides {missing}"


def test_legacy_object_identity():
    """Re-exported names are the same objects, and the strategy adapter registry is a single shared dict."""
    from finetuning_scheduler import fts, fts_supporters

    assert finetuning_scheduler.FinetuningScheduler is fts.FinetuningScheduler
    assert finetuning_scheduler.CallbackResolverMixin is fts_supporters.CallbackResolverMixin
    assert finetuning_scheduler.FTSState is fts_supporters.FTSState
    assert fts.STRATEGY_ADAPTERS is fts_supporters.STRATEGY_ADAPTERS


@pytest.mark.parametrize("name", SCHEDULE_IMPL_STATICMETHODS)
def test_schedule_impl_staticmethods(name):
    """Schedule helpers stay static methods callable on the mixin class itself."""
    assert isinstance(inspect.getattr_static(ScheduleImplMixin, name), staticmethod)


@pytest.mark.parametrize("name", ["FinetuningScheduler", "FTSCheckpoint", "FTSEarlyStopping", "CallbackResolverMixin"])
def test_cli_import_paths(name):
    """Saved CLI configurations keep resolving the public classes to their top-level import path."""
    jsonargparse_util = pytest.importorskip("jsonargparse._util")
    assert jsonargparse_util.get_import_path(getattr(finetuning_scheduler, name)) == f"finetuning_scheduler.{name}"


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def test_save_schedule_rank_gating_and_logger(tmp_path):
    """Only global rank zero writes a schedule, and its info message lands on the Lightning rank-zero logger."""
    schedule = {0: {"params": ["0.weight"]}}
    rz_logger = logging.getLogger(RANK_ZERO_LOGGER)
    handler, orig_level = _ListHandler(), rz_logger.level
    rz_logger.addHandler(handler)
    rz_logger.setLevel(logging.INFO)
    try:
        rank_zero_only.rank = 1
        assert ScheduleImplMixin.save_schedule("sched.yaml", schedule, tmp_path) is None
        assert not os.listdir(tmp_path)
        rank_zero_only.rank = 0
        saved = ScheduleImplMixin.save_schedule("sched.yaml", schedule, tmp_path)
        assert saved is not None and os.path.isfile(saved)
        assert any("fine-tuning schedule dumped" in r.getMessage() for r in handler.records)
    finally:
        rz_logger.removeHandler(handler)
        rz_logger.setLevel(orig_level)


def test_logging_level_targets_rank_zero_logger():
    """The callback's ``logging_level`` configures the Lightning rank-zero logger."""
    rz_logger = logging.getLogger(RANK_ZERO_LOGGER)
    orig_level = rz_logger.level
    try:
        FinetuningScheduler(logging_level=logging.WARNING)
        assert rz_logger.level == logging.WARNING
        FinetuningScheduler(logging_level=logging.DEBUG)
        assert rz_logger.level == logging.DEBUG
    finally:
        rz_logger.setLevel(orig_level)


def _reinit_import_warnings(class_path: str) -> tuple[str, list[warnings.WarningMessage]]:
    fts = FinetuningScheduler()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(MisconfigurationException) as excinfo:
            fts._import_reinit_class({"class_path": class_path}, "lr_scheduler")
    return str(excinfo.value), list(record)


@pytest.mark.parametrize(
    "class_path, expected_prefix",
    [
        ("torch.optim.SGD", "The provided lr scheduler"),
        ("torch.optim.lr_scheduler.NoSuchScheduler", "Could not import specified reinitialization"),
    ],
    ids=["unsupported_lrs", "import_failure"],
)
def test_reinit_class_errors_are_unwrapped_and_attributed(class_path, expected_prefix):
    """An unsupported scheduler is reported as such (not re-wrapped as an import failure), with exactly one
    warning, attributed to FTS code rather than to a logging or distributed helper."""
    msg, record = _reinit_import_warnings(class_path)
    assert msg.startswith(expected_prefix)
    assert len(record) == 1
    assert str(record[0].message).startswith(expected_prefix)
    pkg_dir = os.path.dirname(finetuning_scheduler.__file__)
    assert record[0].filename.startswith(pkg_dir)
    assert os.path.basename(record[0].filename) not in {"dist.py", "rank_zero.py", "exceptions.py"}


def test_core_types_are_torch_types():
    """The supported scheduler tuple is built from ``torch.optim.lr_scheduler`` classes."""
    from finetuning_scheduler.types import FTSLRSchedulerTypeTuple, supported_lrs

    assert [cls.__name__ for cls in FTSLRSchedulerTypeTuple] == supported_lrs
    assert all(cls.__module__.startswith("torch.optim.lr_scheduler") for cls in FTSLRSchedulerTypeTuple)
    assert issubclass(torch.optim.lr_scheduler.StepLR, FTSLRSchedulerTypeTuple)
