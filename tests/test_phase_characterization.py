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
"""Characterization of the phase-transition behavior the direct-PyTorch controller extracts into the core (#51).

These pin today's Lightning-layer behavior, quirks included, before any of it moves: the plateau decisions of
``FTSEarlyStopping`` (which become ``PlateauTransition``) and the thaw bookkeeping of ``StrategyAdapter.base_ft_phase``
(which downstream packages bind with ``functools.partial``).
"""
import logging
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch import seed_everything, Trainer

from finetuning_scheduler import FinetuningScheduler, FTSEarlyStopping
from finetuning_scheduler.strategy_adapters import StrategyAdapter
from tests.test_finetuning_scheduler_callback import FinetuningSchedulerBoringModel

# (should_stop, reason, best_score, wait_count, es_phase_complete) after each evaluated value
PLATEAU_CASES = {
    "min_nonfinal": (dict(mode="min", patience=2, min_delta=0.1), False, [1.0, 0.95, 0.95, 0.5, 0.45, 0.44], [
        (False, "Metric val_loss improved. New best score: 1.000", 1.0, 0, False),
        (False, None, 1.0, 1, False),
        (False, None, 1.0, 0, True),
        (False, "Metric val_loss improved by 0.500 >= min_delta = 0.1. New best score: 0.500", 0.5, 0, True),
        (False, None, 0.5, 1, True),
        (False, None, 0.5, 0, True),
    ]),
    "min_final": (dict(mode="min", patience=2, min_delta=0.1), True, [1.0, 0.95, 0.95, 0.5, 0.45, 0.44], [
        (False, "Metric val_loss improved. New best score: 1.000", 1.0, 0, False),
        (False, None, 1.0, 1, False),
        (True, "Monitored metric val_loss did not improve in the last 2 records. Best score: 1.000. Signaling Trainer"
         " to stop.", 1.0, 2, False),
        (False, "Metric val_loss improved by 0.500 >= min_delta = 0.1. New best score: 0.500", 0.5, 0, False),
        (False, None, 0.5, 1, False),
        (True, "Monitored metric val_loss did not improve in the last 2 records. Best score: 0.500. Signaling Trainer"
         " to stop.", 0.5, 2, False),
    ]),
    "max_nonfinal": (dict(mode="max", patience=1, min_delta=0.0), False, [0.5, 0.75, 0.75, 0.875], [
        (False, "Metric val_loss improved. New best score: 0.500", 0.5, 0, False),
        (False, "Metric val_loss improved by 0.250 >= min_delta = 0.0. New best score: 0.750", 0.75, 0, False),
        (False, None, 0.75, 0, True),
        (False, "Metric val_loss improved by 0.125 >= min_delta = 0.0. New best score: 0.875", 0.875, 0, True),
    ]),
    "stopping_threshold": (dict(mode="min", patience=3, stopping_threshold=0.25), False, [1.0, 0.5, 0.25, 0.125], [
        (False, "Metric val_loss improved. New best score: 1.000", 1.0, 0, False),
        (False, "Metric val_loss improved by 0.500 >= min_delta = 0.0. New best score: 0.500", 0.5, 0, False),
        (False, "Metric val_loss improved by 0.250 >= min_delta = 0.0. New best score: 0.250", 0.25, 0, False),
        (True, "Stopping threshold reached: val_loss = 0.125 < 0.25. Signaling Trainer to stop.", 0.25, 0, False),
    ]),
    # the divergence message reuses the mode's comparison symbol (``<`` for ``min``), as Lightning's does
    "divergence_threshold": (dict(mode="min", patience=3, divergence_threshold=2.0), False, [1.0, 1.5, 2.5], [
        (False, "Metric val_loss improved. New best score: 1.000", 1.0, 0, False),
        (False, None, 1.0, 1, False),
        (True, "Divergence threshold reached: val_loss = 2.5 < 2.0. Signaling Trainer to stop.", 1.0, 1, False),
    ]),
    "non_finite": (dict(mode="min", patience=3), False, [1.0, float("nan")], [
        (False, "Metric val_loss improved. New best score: 1.000", 1.0, 0, False),
        (True, "Monitored metric val_loss = nan is not finite. Previous best value was 1.000. Signaling Trainer to"
         " stop.", 1.0, 0, False),
    ]),
    "non_finite_unchecked": (dict(mode="min", patience=1, check_finite=False), False, [1.0, float("inf"), 0.5], [
        (False, "Metric val_loss improved. New best score: 1.000", 1.0, 0, False),
        (False, None, 1.0, 0, True),
        (False, "Metric val_loss improved by 0.500 >= min_delta = 0.0. New best score: 0.500", 0.5, 0, True),
    ]),
}


def _phase_es(es_cls=FTSEarlyStopping, final_phase=False, **kwargs):
    es = es_cls(monitor="val_loss", **kwargs)
    es.finetuningscheduler_callback = SimpleNamespace(curr_depth=0)
    es.final_phase = final_phase
    es._reset_es_phase()
    return es


@pytest.mark.parametrize("case", list(PLATEAU_CASES), ids=list(PLATEAU_CASES))
def test_plateau_decisions(case):
    """Each evaluation's stop decision, reason, best score, wait count and phase-complete flag.

    Note that a phase
    transition resets ``wait_count`` but keeps ``best_score`` for the next phase.
    """
    kwargs, final_phase, values, expected = PLATEAU_CASES[case]
    es = _phase_es(final_phase=final_phase, **kwargs)
    observed = []
    for v in values:
        should_stop, reason = es._evaluate_stopping_criteria(torch.tensor(v))
        observed.append((should_stop, reason, float(es.best_score), es.wait_count, es.es_phase_complete))
    assert observed == expected


def test_plateau_improvement_message_override():
    """A subclass's ``_improvement_message`` supplies the improvement reason, computed before ``best_score``
    changes."""

    class CustomES(FTSEarlyStopping):
        def _improvement_message(self, current):
            return f"custom: {float(self.best_score):.3f} -> {float(current):.3f}"

    es = _phase_es(CustomES, mode="min", patience=1)
    assert es._evaluate_stopping_criteria(torch.tensor(0.5)) == (False, "custom: inf -> 0.500")
    assert es._evaluate_stopping_criteria(torch.tensor(0.25)) == (False, "custom: 0.500 -> 0.250")


class _ThawNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(2, 2)
        self.b = torch.nn.Linear(2, 2)
        self.c = torch.nn.Linear(2, 2)


@contextmanager
def _rank_zero_debug_records():
    """Capture Lightning's rank-zero debug records (its ``lightning.pytorch`` logger does not propagate)."""
    rz_logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")
    records: list[str] = []
    handler = logging.Handler(level=logging.DEBUG)
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    orig_level = rz_logger.level
    rz_logger.addHandler(handler)
    rz_logger.setLevel(logging.DEBUG)
    rank_zero_only.rank = 0
    try:
        yield records
    finally:
        rz_logger.removeHandler(handler)
        rz_logger.setLevel(orig_level)


@pytest.mark.parametrize("init_thaw", [False, True], ids=["thaw", "init_thaw"])
def test_base_ft_phase(init_thaw):
    """Newly thawed names follow ``named_parameters`` order.

    The currently thawed list holds the previously thawed names, then the new ones. Names already thawed are not
    reported as new.
    """
    module = _ThawNet()
    for n, p in module.named_parameters():
        p.requires_grad = n.startswith("c.")
    with _rank_zero_debug_records() as messages:
        thawed, curr = StrategyAdapter.base_ft_phase(module, ["c.bias", "a.weight", "a.bias"], init_thaw=init_thaw)
    assert thawed == ["a.weight", "a.bias"]
    assert curr == ["c.weight", "c.bias", "a.weight", "a.bias"]
    assert {n for n, p in module.named_parameters() if p.requires_grad} == {"a.weight", "a.bias", "c.weight", "c.bias"}
    prefix = "Initializing with" if init_thaw else "Thawed"
    assert any(m.startswith(f"{prefix} the following module parameters: ") for m in messages)
    assert any(m.startswith("The following module parameters are currently thawed: ") for m in messages)


def test_base_ft_phase_translation_partial():
    """A ``functools.partial`` binding ``translation_func`` (as downstream adapters do) translates only the logged
    names; the returned names are the module's own."""
    module = _ThawNet()
    for p in module.parameters():
        p.requires_grad = False
    thaw = partial(StrategyAdapter.base_ft_phase, translation_func=lambda names: [f"tl::{n}" for n in names])
    with _rank_zero_debug_records() as messages:
        thawed, curr = thaw(module, ["b.weight"])
    assert thawed == curr == ["b.weight"]
    assert any("tl::b.weight" in m for m in messages)
    assert StrategyAdapter.base_ft_phase(module, ["b.weight"]) == ([], ["b.weight"])


def test_epoch_watermark_message(tmpdir):
    """A phase whose ``max_transition_epoch`` is at or below an earlier phase's (ignoring phase 0) is reported once
    per such phase, with the EarlyStopping addendum when transitions are not epoch-only."""
    seed_everything(42)
    schedule = {
        0: {"params": ["model.3.weight", "model.3.bias"], "max_transition_epoch": 5},  # phase 0: not a watermark
        1: {"params": ["model.2.weight", "model.2.bias"], "max_transition_epoch": 3},
        2: {"params": ["model.1.weight", "model.1.bias"], "max_transition_epoch": 2},
        3: {"params": ["model.0.weight", "model.0.bias"], "max_transition_epoch": 4},
    }
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[FinetuningScheduler(ft_schedule=schedule)], devices=1,
                      max_epochs=1, limit_train_batches=1, limit_val_batches=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    rz_logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")
    records: list[str] = []
    handler = logging.Handler(level=logging.INFO)
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    rz_logger.addHandler(handler)
    try:
        trainer.fit(FinetuningSchedulerBoringModel())
    finally:
        rz_logger.removeHandler(handler)
    watermark = [m for m in records if "is less than or equal to a previous max_transition_epoch" in m]
    assert watermark == [
        "Specified max_transition_epoch of depth 2(2) is less than or equal to a previous max_transition_epoch (3), "
        "depth may execute only a single epoch depending upon EarlyStopping criteria."
    ]
