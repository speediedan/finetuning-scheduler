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
"""Characterize the legacy :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` naming surface.

A third-party adapter schedules logical parameter names by overriding the adapter's naming methods (see
:mod:`tests.helpers.legacy_naming_adapter`). These tests pin what that surface produces today: the phase path, the
physical parameters thawed (coupled normalization parameters included), the parameter groups, the translated thaw log
lines and the generated schedule, plus the construction, reflection and subclass checks such an adapter's own tests
rely on. A later change to the adapter layer must leave every pin unchanged.
"""
import inspect
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback

from finetuning_scheduler import FinetuningScheduler, TuningScheduler, TunableSupport
from finetuning_scheduler.strategy_adapters import StrategyAdapter
from tests.helpers.legacy_naming_adapter import (COUPLED_NORMS, LegacyNamingAdapter, LOGICAL_TO_PHYSICAL,
                                                 LogicalNamesBoringModel)
from tests.test_phase_characterization import _rank_zero_debug_records

ADAPTER_PATH = "tests.helpers.legacy_naming_adapter:LegacyNamingAdapter"
LOGICAL_SCHEDULE = {
    0: {"params": ["head.W"], "max_transition_epoch": 1},
    1: {"params": ["blocks.1.W"], "max_transition_epoch": 2},
    2: {"params": ["blocks.0.W"], "max_transition_epoch": 4},
}
HEAD = ["model.4.bias", "model.4.weight"]
BLOCK1 = ["model.2.bias", "model.2.weight", "model.3.bias", "model.3.weight"]  # linear plus its coupled norm
BLOCK0 = ["model.0.bias", "model.0.weight", "model.1.bias", "model.1.weight"]
# (epoch, depth, thawed parameters, the scheduler's record of thawed parameters, optimizer parameter groups)
EXPECTED_PATH = [
    (0, 0, HEAD, HEAD, 1),
    (1, 1, sorted(BLOCK1 + HEAD), sorted(BLOCK1 + HEAD), 2),
    (2, 2, sorted(BLOCK0 + BLOCK1 + HEAD), sorted(BLOCK0 + BLOCK1 + HEAD), 3),
    (3, 2, sorted(BLOCK0 + BLOCK1 + HEAD), sorted(BLOCK0 + BLOCK1 + HEAD), 3),
]
# the thaw log lines name parameters through the adapter's translation (logical where mapped, physical otherwise); phase
# 0 is thawed twice before training starts (once when the schedule is initialized, again by the first depth step)
EXPECTED_THAW_LINES = [
    "Initializing with the following module parameters: ['head.W']",
    "Thawed the following module parameters: ['head.W']",
    "Thawed the following module parameters: ['blocks.1.W', 'model.3.weight', 'model.3.bias']",
    "Thawed the following module parameters: ['blocks.0.W', 'model.1.weight', 'model.1.bias']",
]


class PathRecorder(Callback):
    def __init__(self):
        self.rows = []

    def on_train_epoch_start(self, trainer, pl_module):
        fts = next(c for c in trainer.callbacks if isinstance(c, FinetuningScheduler))
        thawed = sorted(n for n, p in pl_module.named_parameters() if p.requires_grad)
        self.rows.append((trainer.current_epoch, fts.curr_depth, thawed, sorted(fts._fts_state._curr_thawed_params),
                          len(trainer.optimizers[0].param_groups)))


def _fts(**kwargs):
    return FinetuningScheduler(custom_strategy_adapters={"single_device": ADAPTER_PATH}, **kwargs)


def _trainer(tmp_path, callbacks, max_epochs=4, **kwargs):
    kwargs.setdefault("logger", False)
    return Trainer(default_root_dir=tmp_path, callbacks=callbacks, max_epochs=max_epochs, accelerator="cpu", devices=1,
                   limit_train_batches=2, limit_val_batches=2, enable_progress_bar=False, enable_model_summary=False,
                   **kwargs)


def test_legacy_adapter_phase_path(tmp_path):
    """A logical-name schedule follows the pinned phase path, thawing each logical name's physical parameters plus
    its coupled norms, restoring the best checkpoint through the redirected ``state_dict`` at each transition."""
    seed_everything(42)
    fts, recorder = _fts(ft_schedule=LOGICAL_SCHEDULE, epoch_transitions_only=True), PathRecorder()
    with _rank_zero_debug_records() as messages:
        _trainer(tmp_path, [fts, recorder]).fit(LogicalNamesBoringModel())
    assert type(fts.strategy_adapter) is LegacyNamingAdapter
    assert recorder.rows == EXPECTED_PATH
    thaw_lines = [m for m in messages if m.startswith(("Initializing with the following", "Thawed the following"))]
    assert thaw_lines == EXPECTED_THAW_LINES


def test_legacy_adapter_generated_schedule(tmp_path):
    """The adapter's schedule generation, through the pre-2.15 ``ScheduleImplMixin.save_schedule`` import path,
    writes logical names."""
    fts = _fts(gen_ft_sched_only=True)
    with pytest.raises(SystemExit):
        _trainer(tmp_path, [fts], logger=True).fit(LogicalNamesBoringModel())
    sched_file = tmp_path / "lightning_logs" / "version_0" / "LogicalNamesBoringModel_ft_schedule.yaml"
    assert yaml.safe_load(sched_file.read_text()) == {0: {"params": ["head.W"]}, 1: {"params": ["blocks.1.W"]},
                                                       2: {"params": ["blocks.0.W"]}}


def test_legacy_adapter_rejects_unknown_logical_name(tmp_path):
    fts = _fts(ft_schedule={0: {"params": ["head.W"], "max_transition_epoch": 1}, 1: {"params": ["blocks.9.W"]}})
    with pytest.raises(Exception, match="blocks.9.W"):
        _trainer(tmp_path, [fts]).fit(LogicalNamesBoringModel())


def test_legacy_adapter_construction_and_reflection():
    """What a third-party adapter's own tests rely on: an argument-free base ``__init__``, a mock handle exposing
    only ``pl_module`` and ``trainer``, plain functions on the class, and the subclass relationship."""
    adapter = LegacyNamingAdapter()
    assert adapter.exec_ft_phase.func is StrategyAdapter.base_ft_phase  # a partial over the static base function
    module = LogicalNamesBoringModel()
    adapter.fts_handle = SimpleNamespace(pl_module=module, trainer=None)  # no ft_schedule, no _fts_state
    adapter.on_before_init_fts()
    assert module.state_dict == module.model.state_dict
    assert adapter.fts_optim_transform(["blocks.1.W"]) == ["model.2.weight", "model.2.bias", "model.3.weight",
                                                          "model.3.bias"]
    assert adapter.logical_param_translation(["model.2.weight", "model.2.bias", "model.3.bias"]) == [
        "blocks.1.W", "model.3.bias"]
    functions = dict(inspect.getmembers(LegacyNamingAdapter, inspect.isfunction))
    assert {"before_restore_model", "fts_optim_transform", "logical_param_translation"} <= set(functions)
    assert issubclass(LegacyNamingAdapter, StrategyAdapter)
    base_fn = inspect.getattr_static(StrategyAdapter, "base_ft_phase")
    assert isinstance(base_fn, staticmethod)
    assert list(inspect.signature(StrategyAdapter.base_ft_phase).parameters) == ["module", "thaw_pl",
                                                                               "translation_func", "init_thaw"]


def test_base_adapter_naming_defaults():
    """The base adapter's naming methods are identities over the module's own parameter names."""
    adapter = StrategyAdapter()
    module = LogicalNamesBoringModel()
    adapter.fts_handle = SimpleNamespace(pl_module=module, trainer=None)
    names = [n for n, _ in module.named_parameters()]
    assert adapter.fts_optim_transform(names) == names
    assert adapter.logical_param_translation(names) == names
    assert list(adapter.get_named_params_for_schedule_validation()) == names


class EquivalentView:
    """The legacy adapter's naming as a tunable view: the same logical names, physical tensors and coupled
    norms."""

    def __init__(self, root):
        self.root = root

    def support(self):
        return TunableSupport(trainable=True, naming_scheme="logical-blocks")

    def to_physical(self, logical):
        return {n: tuple(LOGICAL_TO_PHYSICAL[n]) for n in logical}

    def to_logical(self, physical):
        rev = {p: lg for lg, ps in LOGICAL_TO_PHYSICAL.items() for p in ps}
        return list(dict.fromkeys(rev.get(p, p) for p in physical))

    def logical_named_parameters(self):
        for lg, ps in LOGICAL_TO_PHYSICAL.items():
            yield lg, self.root.get_parameter(ps[0])

    def coupled(self, physical):
        pulled = [lg for lg, ps in LOGICAL_TO_PHYSICAL.items() if set(ps) & set(physical)]
        return [c for lg in pulled for c in COUPLED_NORMS.get(lg, [])]


def test_equivalent_view_follows_the_legacy_path_on_the_plain_loop():
    """The legacy adapter's phase path (pinned above, under Lightning) is reproduced by an equivalent view driving
    ``TuningScheduler``: the same depth, thawed tensors and parameter groups at every epoch."""
    seed_everything(42)
    model = LogicalNamesBoringModel()
    sgd = partial(torch.optim.SGD, lr=1e-3)
    fts = TuningScheduler(model, LOGICAL_SCHEDULE, optimizer=sgd, view=EquivalentView(model))
    fts.setup()
    path = []
    for epoch in fts.epochs(max_epochs=4):
        path.append((epoch, fts.curr_depth, sorted(n for n, p in model.named_parameters() if p.requires_grad),
                     len(fts.optimizer.param_groups)))
        fts.epoch_end()
    assert path == [(e, d, thawed, groups) for e, d, thawed, _, groups in EXPECTED_PATH]
