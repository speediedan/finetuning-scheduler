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
"""Tests for gradient addresses (``of: grad``) on the PyTorch API: a phase trains only the rows its gradient masks
keep (#198).

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
import logging
from functools import partial

import pytest
import torch
import yaml

from finetuning_scheduler import CanonicalView, TuningScheduler
from finetuning_scheduler.core.checkpoint import BestCheckpoint
from finetuning_scheduler.core.compile import _mask_support, compile_schedule, gradient_entries, support_record
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.schedule import load_address_sets
from finetuning_scheduler.strategies.dtensor import DTensorAdapter
from tests.test_tuning_scheduler_stages import _assert_same, _data, _model, _optimized, _schedule, _values

SGD = partial(torch.optim.SGD, lr=0.1, momentum=0.9, weight_decay=0.1)
ADAMW = partial(torch.optim.AdamW, lr=0.05, weight_decay=0.5)
MID_ROWS = "module:1 & units:[0, 2] & of:grad"  # rows 0 and 2 of the middle layer's weight and bias
# mask the middle layer, then train its weight whole beside a mask of the head, then refreeze the head's weight and
# thaw it whole again
MASKED = {
    0: {"params": [MID_ROWS], "max_transition_epoch": 2},
    1: {"params": ["1.weight", "module:2 & leaf:weight & units:[1] & of:grad"], "max_transition_epoch": 4, "lr": 0.05},
    2: {"params": ["2.bias"], "refreeze": ["2.weight"], "max_transition_epoch": 5},
    3: {"params": ["2.weight"]},
}


def _compiled(schedule, model=None, addresses=None):
    model = _model() if model is None else model
    return compile_schedule(_schedule(schedule), CanonicalView(model), addresses=addresses)


def _fts(schedule=MASKED, model=None, optimizer=SGD, **kwargs):
    fts = TuningScheduler(_model() if model is None else model, _schedule(schedule), optimizer=optimizer, **kwargs)
    fts.setup()
    return fts


def _backward(fts, epoch, accumulate=1):
    x, y = _data(epoch)
    for chunk_x, chunk_y in zip(x.chunk(accumulate), y.chunk(accumulate)):
        ((fts.model(chunk_x) - chunk_y).pow(2).mean() / accumulate).backward()


def _train_epoch(fts, epoch, metrics=None, accumulate=1):
    _backward(fts, epoch, accumulate)
    fts.optimizer.step()
    fts.optimizer.zero_grad()
    fts.step_lr_schedulers("epoch")
    return fts.epoch_end(metrics)


def _run(fts, epochs=6, metrics=None, start=0):
    for epoch in fts.epochs(epochs):
        if epoch >= start:
            _train_epoch(fts, epoch, metrics(epoch) if metrics else None)
    return _values(fts.model)


def _moved_rows(before, after):
    """Each tensor's first-axis indices whose values changed."""
    return {n: [i for i in range(after[n].shape[0]) if not torch.equal(before[n][i], after[n][i])] for n in after}


def test_an_of_grad_address_thaws_what_it_selects_and_compiles_to_masks():
    compiled = _compiled(MASKED)
    first, second, third, fourth = compiled.phases
    assert first.physical == ("1.weight", "1.bias") and first.masks == {"1.weight": (0, 2), "1.bias": (0, 2)}
    # the masked weight trains whole from phase 1 on, in phase 0's parameter groups
    assert second.physical == ("2.weight",) and second.masks == {"1.weight": None, "2.weight": (1,)}
    assert third.refrozen == ("2.weight",) and not third.masks and not fourth.masks
    assert compiled.masks_at(0) == {"1.weight": (0, 2), "1.bias": (0, 2)}
    assert compiled.masks_at(1) == {"1.bias": (0, 2), "2.weight": (1,)}
    assert compiled.masks_at(2) == compiled.masks_at(3) == {"1.bias": (0, 2)}  # a refreeze drops the mask
    assert compiled.trainable_at(1) == ("1.weight", "1.bias", "2.weight")
    with pytest.raises(KeyError, match="no phase 4"):
        compiled.masks_at(4)


@pytest.mark.parametrize("params, masks", [
    ([MID_ROWS, "module:1 & units:[1] & of:grad"], {"1.weight": (0, 1, 2), "1.bias": (0, 1, 2)}),
    ([MID_ROWS, "1.bias"], {"1.weight": (0, 2)}),  # another entry selects the bias whole
    (["module:1 & units:0:4 & of:grad"], {}),  # a mask keeping every row is no mask
    (["module:1 & of:grad"], {}),  # nor is a gradient address without a unit set
    # one set: a union's disjuncts each give their own mask, and a value part trains its tensors whole
    (["module:1 & leaf:weight & units:[0] & of:grad | module:1 & leaf:bias & units:[1] & of:grad"],
     {"1.weight": (0,), "1.bias": (1,)}),
    (["module:1 & leaf:bias & of:value | module:1 & leaf:weight & units:[0, 2] & of:grad"], {"1.weight": (0, 2)}),
], ids=["union", "whole", "every_row", "no_units", "union_masks_in_one_set", "mixed_set"])
def test_a_phases_masks_are_the_union_of_its_entries(params, masks):
    (cp,) = _compiled({0: {"params": params}}).phases
    assert cp.physical == ("1.weight", "1.bias") and dict(cp.masks) == masks


def test_a_later_phase_widens_a_mask_rather_than_thawing_twice():
    compiled = _compiled({0: {"params": [MID_ROWS], "max_transition_epoch": 1},
                          1: {"params": ["module:1 & units:[1] & of:grad"], "max_transition_epoch": 2},
                          2: {"params": ["module:1 & units:[1] & of:grad"]}})
    assert compiled.phases[1].physical == () and compiled.masks_at(1)["1.weight"] == (0, 1, 2)
    assert not compiled.phases[2].masks  # selecting rows already kept changes nothing
    with pytest.raises(FTSConfigurationError, match="specified in multiple phases: 1.weight"):
        _compiled({0: {"params": ["1.weight"]}, 1: {"params": [MID_ROWS]}})  # a whole tensor is not masked again


def test_the_digest_sees_masks_only_when_present():
    plain = _compiled({0: {"params": ["1.weight", "1.bias"]}})
    assert _compiled({0: {"params": ["module:1 & of:grad"]}}).digest == plain.digest
    assert _compiled({0: {"params": [MID_ROWS]}}).digest != plain.digest
    assert _compiled({0: {"params": [MID_ROWS]}}).digest != _compiled(
        {0: {"params": ["module:1 & units:[0] & of:grad"]}}).digest


@pytest.mark.parametrize("schedule, match", [
    ({0: {"params": ["1.weight"], "max_transition_epoch": 1}, 1: {"params": ["2.weight"], "refreeze": [MID_ROWS]}},
     "Phase 1's `refreeze` addresses gradients"),
    ({0: {"params": ["1.weight"]}, 1: {"params": ["1.bias"], "refreeze": ["1.weight"]},
      2: {"params": [MID_ROWS], "refreeze": ["1.bias"]}}, "Phase 2 both thaws and refreezes these tensors"),
    ({0: {"params": [MID_ROWS], "max_transition_epoch": 1}, 1: {"params": [MID_ROWS], "refreeze": ["1.weight"]}},
     r"Phase 1 both thaws and refreezes these tensors: \['1.weight'\]"),
    ({0: {"params": ["module:1 & units:[0]"]}}, "masked as gradients with `of: grad`"),
], ids=["refreeze_of_grad", "thaw_and_refreeze", "reselect_and_refreeze", "units_without_of_grad"])
def test_refused_gradient_addresses(schedule, match):
    with pytest.raises(FTSConfigurationError, match=match):
        _compiled(schedule)


@pytest.mark.parametrize("entry, physical, masks", [
    # a unit set under a difference keeps the rest of the tensor: the bias without its row 0, the weight whole
    ("module:1 & of:grad - module:1 & leaf:bias & units:[0] & of:grad", ("1.weight", "1.bias"), {"1.bias": (1, 2, 3)}),
    # intersected unit sets keep the rows both keep
    ("(module:1 & units:[0, 1] & of:grad) & (module:1 & units:[1] & of:grad)", ("1.weight", "1.bias"),
     {"1.weight": (1,), "1.bias": (1,)}),
], ids=["units_under_a_difference", "units_intersected"])
def test_unit_sets_under_a_difference_or_intersection_resolve_exactly(entry, physical, masks):
    """Venngram resolves gradient masks under a difference or an intersection exactly (it refused them before)."""
    compiled = _compiled({0: {"params": [entry]}})
    assert compiled.physical_for(0) == physical
    assert compiled.masks_at(0) == masks


def test_the_support_record_honours_gradient_masks_and_refuses_unit_sets_to_tune():
    tuning = support_record().tuning
    assert {"of", "units"} <= tuning.fields and tuning.units == "grad-mask"
    book = {"sets": {"rows": {"module": "1", "units": [0, 2]}, "rows_grad": "@rows & of:grad",
                     "top_rows": {"module": "2", "leaf": "weight", "units": [1], "of": "grad"}}}
    sets = load_address_sets(book)
    compiled = _compiled({0: {"params": ["@top_rows"], "max_transition_epoch": 1}, 1: {"params": ["@rows_grad"]}},
                         addresses=sets)
    assert compiled.masks_at(1) == {"2.weight": (1,), "1.weight": (0, 2), "1.bias": (0, 2)}
    assert gradient_entries({0: {"params": ["@top_rows", "1.bias"]}, 1: {"params": ["@rows"]}}, sets) == {
        0: ["@top_rows"]}
    # the same unit set used as values is units to tune, which the record refuses by name
    with pytest.raises(FTSConfigurationError, match=r"(?s)does not tune:.*set `rows`: it selects units"):
        _compiled({0: {"params": ["@rows"]}}, addresses=sets)


@pytest.mark.parametrize("optimizer", [SGD, ADAMW, partial(torch.optim.Adam, lr=0.05, weight_decay=0.1)],
                         ids=["sgd_momentum_decay", "adamw_decay", "adam_coupled_decay"])
def test_masked_rows_hold_through_momentum_and_weight_decay(optimizer):
    """A zeroed gradient alone would let decoupled decay (AdamW), coupled decay and momentum move a masked row."""
    fts = _fts(optimizer=optimizer)
    seen = []
    for epoch in fts.epochs(7):
        seen.append(fts.curr_depth)
        masks = fts.compiled_schedule.masks_at(fts.curr_depth)
        assert fts.gradient_masks == masks
        trainable = set(fts.compiled_schedule.trainable_at(fts.curr_depth))
        assert _optimized(fts) == sorted(trainable)
        before = _values(fts.model)
        _train_epoch(fts, epoch, accumulate=2)
        moved = _moved_rows(before, _values(fts.model))
        for n, rows in moved.items():
            expected = list(masks[n]) if n in masks else list(range(before[n].shape[0])) if n in trainable else []
            assert rows == expected, n
    assert seen == [0, 0, 1, 1, 2, 3, 3]


def test_masked_rows_gather_no_optimizer_state():
    fts = _fts(optimizer=ADAMW)
    for epoch in range(2):
        _backward(fts, epoch)
        fts.optimizer.step()
        fts.optimizer.zero_grad()
    for p in fts.model[1].parameters():
        state = fts.optimizer.state[p]
        assert all(state[k][[1, 3]].abs().sum() == 0 and state[k][[0, 2]].abs().sum() > 0
                   for k in ("exp_avg", "exp_avg_sq"))


def test_masked_gradients_are_zero_before_clipping_under_accumulation():
    fts = _fts()
    _backward(fts, 0, accumulate=4)
    grad = fts.model.get_parameter("1.weight").grad
    assert grad[[1, 3]].abs().sum() == 0 and grad[[0, 2]].abs().sum() > 0
    kept = torch.cat([p.grad[[0, 2]].flatten() for p in fts.model[1].parameters()]).norm()
    # the clipped norm is the kept rows' alone
    assert torch.allclose(torch.nn.utils.clip_grad_norm_(fts.model.parameters(), max_norm=1e-3), kept)


def test_an_unmasked_tensor_trains_every_row_after_its_hook_is_removed():
    """Phase 1 trains the masked weight whole and phase 3 thaws the refrozen head weight whole: their hooks are gone,
    so a gradient on a formerly masked row survives the backward pass."""
    fts = _fts()
    for epoch in fts.epochs(7):
        _train_epoch(fts, epoch)
        if fts.curr_depth == 3:
            break
    assert fts.gradient_masks == {"1.bias": (0, 2)}
    _backward(fts, 0)
    assert all(fts.model.get_parameter(n).grad[r].abs().sum() > 0 for n, r in (("1.weight", 1), ("2.weight", 0)))
    assert fts.model.get_parameter("1.bias").grad[[1, 3]].abs().sum() == 0


def test_a_reinitialized_optimizer_holds_the_masked_rows():
    schedule = {**_schedule(MASKED), 1: {**MASKED[1], "new_optimizer": {"optimizer_init": {
        "class_path": "torch.optim.AdamW", "init_args": {"lr": 0.05, "weight_decay": 0.5}}}}}
    fts = _fts(schedule)
    before = _values(fts.model)
    _run(fts, 4)
    assert isinstance(fts.optimizer, torch.optim.AdamW) and fts.curr_depth == 1
    assert _moved_rows(before, _values(fts.model))["1.bias"] == [0, 2]


def test_each_phases_masks_are_reported(caplog):
    with caplog.at_level(logging.INFO, logger="finetuning_scheduler"):
        _run(_fts(), 3)
    reports = [r.getMessage() for r in caplog.records if "through gradient masks" in r.getMessage()]
    prefix = "trains part of {} tensors through gradient masks (rows kept of rows):"
    assert reports == [f"Phase 0 {prefix.format(2)} 1.weight (2 of 4), 1.bias (2 of 4).",
                       f"Phase 1 {prefix.format(1)} 2.weight (1 of 2)."]


def test_a_sharded_model_refuses_gradient_masks():
    with pytest.raises(FTSConfigurationError, match=r"Phases \[0, 1\] mask the gradients .* of a sharded model"):
        _fts(distributed=DTensorAdapter())


def _val(epoch):
    return {"loss": [1.0, 0.9, 0.95, 0.8, 0.85, 0.7, 0.75][epoch]}


def _make(tmp_path, name, optimizer, checkpoint, **kwargs):
    if checkpoint:
        kwargs["checkpoint"] = BestCheckpoint(tmp_path / name, "loss")
    return _fts(optimizer=optimizer, lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma=0.5),
                **kwargs)


@pytest.mark.parametrize("split", [0, 1, 2, 4, 5])
@pytest.mark.parametrize("optimizer, checkpoint", [(SGD, False), (ADAMW, False), (ADAMW, True)],
                         ids=["sgd", "adamw", "adamw_best"])
def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path, split, optimizer, checkpoint):
    """Saved before, at or after a mask widens, a refreeze drops one or a tensor is thawed whole again, a resumed
    run reinstalls the masks in effect and ends where an uninterrupted one does."""
    metrics = _val if checkpoint else None
    expected = _run(_make(tmp_path, "ref", optimizer, checkpoint), 7, metrics)
    first = _make(tmp_path, "first", optimizer, checkpoint)
    for epoch in first.epochs(7):
        _train_epoch(first, epoch, metrics(epoch) if metrics else None)
        if epoch == split:
            saved = first.save(tmp_path / "s.pt")
            break
    resumed = _make(tmp_path, "resumed", optimizer, checkpoint, model=_model(seed=7))
    resumed.load(saved)
    assert resumed.curr_depth == first.curr_depth and resumed.gradient_masks == first.gradient_masks
    _assert_same(expected, _run(resumed, 7, metrics, start=split + 1))


def test_a_state_with_other_masks_is_refused(tmp_path):
    saved = _fts().save(tmp_path / "s.pt")
    other = {**_schedule(MASKED), 0: {**MASKED[0], "params": ["module:1 & units:[0] & of:grad"]}}
    with pytest.raises(FTSConfigurationError, match="different fine-tuning schedule|different tensors"):
        _fts(other).load(saved)


def test_a_gradient_phase_exports_as_a_set_of_gradients(tmp_path):
    fts = _fts({0: {"params": [MID_ROWS], "max_transition_epoch": 1}, 1: {"params": ["2.weight"]}})
    book = fts.export_book(tmp_path / "masked.yaml")
    assert book["schema_version"] == 2 and book["sets"]["phase_0"] == {"module": "1", "units": [0, 2], "of": "grad"}
    assert yaml.safe_load((tmp_path / "masked.yaml").read_text()) == book
    replay = _compiled({0: {"params": ["@phase_0"], "max_transition_epoch": 1}, 1: {"params": ["@phase_1"]}},
                       addresses=load_address_sets(str(tmp_path / "masked.yaml")))
    assert replay.masks_at(1) == fts.compiled_schedule.masks_at(1) == {"1.weight": (0, 2), "1.bias": (0, 2)}


@pytest.mark.parametrize("schedule", [
    MASKED,  # phase 1 mixes a name (values) with a gradient address
    {0: {"params": [MID_ROWS, "module:2 & leaf:weight & units:[1] & of:grad"], "max_transition_epoch": 1},
     1: {"params": ["module:2 & leaf:bias & of:grad"]}},  # several gradient entries with unit sets
], ids=["mixed_set", "union_masks"])
def test_every_phase_exports_as_one_set_that_compiles_back_to_its_masks(tmp_path, schedule):
    fts = _fts(schedule)
    book = fts.export_book(tmp_path / "book.yaml")
    sets = load_address_sets(str(tmp_path / "book.yaml"))
    # the record honours mixed sets and united masks, judged as compilation judges a set of gradients
    assert not _mask_support(support_record().tuning).check_book(sets.books[0])
    replay = _compiled({d: {**p, "params": [f"@phase_{d}"]} for d, p in schedule.items()}, addresses=sets)
    for depth in schedule:
        assert replay.masks_at(depth) == fts.compiled_schedule.masks_at(depth)
        assert set(replay.phases[depth].physical) == set(fts.compiled_schedule.phases[depth].physical)
    if schedule is MASKED:  # a value entry beside a gradient one states `of: value`
        assert {"of": "value"} in book["sets"]["phase_1"]["any"][0]["all"]
