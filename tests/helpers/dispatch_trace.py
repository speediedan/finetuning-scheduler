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
"""Recording the order in which the Lightning callback dispatches to its strategy adapter.

:func:`traced` builds a subclass of a strategy adapter that records each call of the adapter hooks a third-party or
built-in adapter overrides (:data:`ADAPTER_METHODS`, plus ``exec_ft_phase``, which adapters bind per instance), and
:class:`TracingFinetuningScheduler` records the callback's own dispatch methods (:data:`CALLBACK_METHODS`), which
tests also override. Each record is ``[source, method, epoch, depth, args, kwargs]``, with arguments summarized to
JSON-comparable values, so a run's trace can be compared with a committed snapshot.

A traced adapter keeps its class's parameter-naming classification (the recording overrides are not naming overrides).
"""
import functools
import json
import os
from pathlib import Path
from typing import Any

from finetuning_scheduler import FinetuningScheduler
from finetuning_scheduler.strategy_adapters import FSDPStrategyAdapter, ModelParallelStrategyAdapter, StrategyAdapter
from tests.helpers.legacy_naming_adapter import LegacyNamingAdapter

ADAPTER_METHODS = (
    "on_before_init_fts",
    "on_after_init_fts",
    "on_before_fts_fit_start",
    "on_after_phase_transition",
    "on_before_restore_optimizers_and_lrs",
    "before_restore_model",
    "phase0_optimizer_override",
    "fts_optim_transform",
    "fts_optim_inspect",
    "logical_param_translation",
    "_maybe_set_bn_track_running_stats",
    "get_named_params_for_schedule_validation",
    "validate_ft_sched",
    "gen_ft_schedule",
)
CALLBACK_METHODS = ("step", "step_pg", "restore_best_ckpt", "thaw_to_depth", "reinit_optimizer", "reinit_lr_scheduler")
SNAPSHOTS = Path(__file__).parent / "dispatch_traces.json"
# set to regenerate the committed snapshots from the current code (review the diff before committing it)
UPDATE_ENV = "FTS_UPDATE_DISPATCH_TRACES"


def summarize(value: Any) -> Any:
    """A JSON-comparable summary of a dispatch argument: names and scalars as they are, paths (which vary by run) as
    ``"<path>"``, anything else by type."""
    if isinstance(value, os.PathLike) or (isinstance(value, str) and os.path.isabs(value)):
        return "<path>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return sorted(value)
    return type(value).__name__


def _record(fts: Any, source: str, name: str, args: tuple, kwargs: dict) -> None:
    trace = getattr(fts, "dispatch_trace", None)
    if trace is None:
        return
    trainer = getattr(fts, "trainer", None)
    epoch = trainer.current_epoch if trainer is not None else None
    depth = fts._fts_state._curr_depth if hasattr(fts, "_fts_state") else None
    trace.append([source, name, epoch, depth, [summarize(a) for a in args],
                  {k: summarize(v) for k, v in sorted(kwargs.items())}])


def _recording(name: str, owner: list) -> Any:
    # dispatches through `super()`, so descriptors such as `fts_optim_inspect` (a `partialmethod`) bind as they would
    def method(self, *args, **kwargs):
        _record(getattr(self, "fts_handle", None), "adapter", name, args, kwargs)
        return getattr(super(owner[0], self), name)(*args, **kwargs)

    method.__name__ = name
    return method


def traced(adapter_cls: type[StrategyAdapter]) -> type[StrategyAdapter]:
    """A subclass of ``adapter_cls`` recording its dispatch into the connected callback's ``dispatch_trace``."""
    owner: list = []
    namespace: dict[str, Any] = {
        name: _recording(name, owner) for name in ADAPTER_METHODS if hasattr(adapter_cls, name)
    }
    # keep the class's own classification: the recording overrides do not name parameters
    namespace["_fts_naming_overrides"] = adapter_cls.naming_overrides()
    namespace["__module__"] = __name__

    def get_exec(self):
        fn = self.__dict__["_traced_exec_ft_phase"]

        @functools.wraps(fn)
        def exec_ft_phase(*args, **kwargs):
            _record(getattr(self, "fts_handle", None), "adapter", "exec_ft_phase", args, kwargs)
            return fn(*args, **kwargs)

        return exec_ft_phase

    def set_exec(self, fn):
        self.__dict__["_traced_exec_ft_phase"] = fn

    namespace["exec_ft_phase"] = property(get_exec, set_exec)
    cls = type(f"Traced{adapter_cls.__name__}", (adapter_cls,), namespace)
    owner.append(cls)
    return cls


TracedStrategyAdapter = traced(StrategyAdapter)
TracedLegacyNamingAdapter = traced(LegacyNamingAdapter)
TracedFSDPStrategyAdapter = traced(FSDPStrategyAdapter)
TracedModelParallelStrategyAdapter = traced(ModelParallelStrategyAdapter)


def _callback_recording(name: str) -> Any:
    fn = getattr(FinetuningScheduler, name)

    @functools.wraps(fn)
    def method(self, *args, **kwargs):
        _record(self, "callback", name, args, kwargs)
        return fn(self, *args, **kwargs)

    return method


class TracingFinetuningScheduler(FinetuningScheduler):
    """The callback, recording its own dispatch methods and (with a traced adapter) the adapter's."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dispatch_trace: list = []


for _name in CALLBACK_METHODS:
    setattr(TracingFinetuningScheduler, _name, _callback_recording(_name))


def check_snapshot(scenario: str, trace: list) -> None:
    """Compare ``trace`` with the committed snapshot for ``scenario`` (or rewrite it, under :data:`UPDATE_ENV`)."""
    trace = json.loads(json.dumps(trace))
    snapshots = json.loads(SNAPSHOTS.read_text()) if SNAPSHOTS.exists() else {}
    if os.environ.get(UPDATE_ENV):
        snapshots[scenario] = trace
        SNAPSHOTS.write_text(json.dumps(snapshots, indent=1, sort_keys=True) + "\n")
        return
    assert scenario in snapshots, f"No dispatch snapshot for {scenario!r}; generate it with {UPDATE_ENV}=1."
    expected = snapshots[scenario]
    for i, (got, want) in enumerate(zip(trace, expected)):
        assert got == want, f"{scenario}: dispatch record {i} differs:\n  got  {got}\n  want {want}"
    assert len(trace) == len(expected), f"{scenario}: {len(trace)} dispatch records, expected {len(expected)}"
