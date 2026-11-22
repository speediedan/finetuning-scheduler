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
"""Tests for a chosen ``restore``: a function that chooses, at a phase boundary, the state the next phase starts
from (#196).

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
import pytest
import torch

from finetuning_scheduler import RestoreEvent, TuningScheduler
from finetuning_scheduler.core.checkpoint import BestCheckpoint, load_payload
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from tests.test_tuning_scheduler_stages import (_assert_in_place, _assert_same, _fts, _make, _model, _run,
                                                _schedule, _train_epoch, _val, _values, BASE, HEAD, MID, SGD)


def lowest(event: RestoreEvent) -> str:
    """The candidate with the lowest recorded score, else ``none``."""
    scored = {c: s for c, s in event.scores.items() if s is not None}
    return min(scored, key=lambda c: float(scored[c])) if scored else "none"


def highest(event: RestoreEvent) -> str:
    scored = {c: s for c, s in event.scores.items() if s is not None}
    return max(scored, key=lambda c: float(scored[c]))


# train the head, refreeze it and train the middle, then thaw the head again from whichever earlier phase's best
# scored lowest (by `_val`: phase 0's best is 0.9, phase 1's 0.8)
CHOSEN = {
    0: {"params": HEAD, "max_transition_epoch": 2, "phase_config": {"stage": "head"}},
    1: {"params": MID, "refreeze": HEAD, "max_transition_epoch": 5, "lr": 0.05},
    2: {"params": HEAD, "refreeze": MID, "lr": 0.02,
        "restore": {"choose": "lowest", "candidates": ["best:0", "best:1", "none"]}},
}


def _chosen(tmp_path, name="ckpt", chooser=lowest, schedule=CHOSEN, **kwargs):
    return _fts(schedule, checkpoint=BestCheckpoint(tmp_path / name, "loss"), restore_best=False,
                restore_choosers={"lowest": chooser}, **kwargs)


def test_the_chooser_picks_the_state_the_phase_starts_from(tmp_path):
    seen, ended, started = [], {}, {}
    fts = _chosen(tmp_path, chooser=lambda e: seen.append(e) or lowest(e))
    fts.add_phase_hook(lambda e: ended.update({e.depth: _values(fts.model)}), when="end")
    fts.add_phase_hook(lambda e: started.update({e.depth: _values(fts.model)}), when="start")
    fts.add_gate(lambda e: True)
    _run(fts, metrics=_val)
    assert fts.curr_depth == 2
    (event,) = seen
    assert event.depth == 2 and event.candidates == ("best:0", "best:1", "none")
    cp = fts._checkpoint
    assert {c: None if s is None else float(s) for c, s in event.scores.items()} == pytest.approx(
        {"best:0": 0.9, "best:1": 0.8, "none": None})
    assert event.scores["best:1"] is not None and torch.equal(event.scores["best:1"], cp.best_score_for_depth(1))
    assert event.best_depth == 1 and event.metrics == {"loss": 0.85}
    # the ended phase's gates ran before the choice
    assert [(r["depth"], r["accept"]) for r in event.gate_records] == [(0, True), (1, True)]
    assert event.phase_configs == {0: {"stage": "head"}} and event.tuning_scheduler is fts
    # the end hooks and gates see the ended phase's last state; the phase starts from phase 1's best
    best_1 = load_payload(cp.best_path_for_depth(1))["model"]
    assert not torch.equal(ended[1]["1.weight"], best_1["1.weight"])
    _assert_same({n: best_1[n] for n in started[2]}, started[2])
    (record,) = fts.restore_choices
    assert record["fn"].endswith("<lambda>")
    assert {k: v for k, v in record.items() if k != "fn"} == {
        "depth": 2, "chooser": "lowest", "candidates": ["best:0", "best:1", "none"], "choice": "best:1"}
    assert fts.restored_count == 1
    _assert_in_place(fts)


def test_another_choice_restores_another_phases_best(tmp_path):
    fts = _chosen(tmp_path, chooser=highest)
    started = {}
    fts.add_phase_hook(lambda e: started.update({e.depth: _values(fts.model)}), when="start")
    _run(fts, metrics=_val)
    assert fts.restore_choices[0]["choice"] == "best:0"
    best_0 = load_payload(fts._checkpoint.best_path_for_depth(0))["model"]
    _assert_same({n: best_0[n] for n in started[2]}, started[2])


def test_choosing_none_keeps_the_state_the_phase_ended_with(tmp_path):
    ended, started = {}, {}
    fts = _chosen(tmp_path, chooser=lambda e: "none")
    fts.add_phase_hook(lambda e: ended.update({e.depth: _values(fts.model)}), when="end")
    fts.add_phase_hook(lambda e: started.update({e.depth: _values(fts.model)}), when="start")
    _run(fts, metrics=_val)
    _assert_same(ended[1], started[2])
    assert fts.restored_count == 0


def test_choosing_the_initial_state(tmp_path):
    model = _model()
    initial = _values(model)
    schedule = {**CHOSEN, 2: {**CHOSEN[2], "restore": {"choose": "lowest", "candidates": ["initial", "none"]}}}
    fts = _chosen(tmp_path, chooser=lambda e: "initial", schedule=schedule, model=model)
    started = {}
    fts.add_phase_hook(lambda e: started.update({e.depth: _values(fts.model)}), when="start")
    _run(fts, metrics=_val)
    _assert_same(initial, started[2])


def test_a_choice_outside_the_candidates_is_refused(tmp_path):
    fts = _chosen(tmp_path, chooser=lambda e: "best")
    with pytest.raises(FTSConfigurationError, match=r"chose 'best' for phase 2, which is not one of its candidates"):
        _run(fts, metrics=_val)


def test_a_chooser_named_by_import_path(tmp_path):
    schedule = {**CHOSEN, 2: {**CHOSEN[2], "restore": {"choose": f"{__name__}.lowest",
                                                         "candidates": ["best:0", "best:1"]}}}
    fts = _fts(schedule, checkpoint=BestCheckpoint(tmp_path, "loss"), restore_best=False)
    _run(fts, metrics=_val)
    assert fts.restore_choices[0]["choice"] == "best:1"
    assert fts.restore_choices[0]["fn"] == f"{__name__}.lowest"
    assert "restore_choosers" not in fts.config_dict()  # the schedule names it


def test_the_default_candidates_are_best_and_none(tmp_path):
    seen = []
    schedule = {0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "restore": {"choose": "c"}}}
    fts = _fts(schedule, checkpoint=BestCheckpoint(tmp_path, "loss"),
               restore_choosers={"c": lambda e: seen.append(e.candidates) or "best"})
    _run(fts, 2, metrics=lambda e: {"loss": 1.0 - e / 10})
    assert seen == [("best", "none")] and fts.restored_count == 1


def test_choosing_among_none_needs_no_checkpoint():
    schedule = {0: {"params": HEAD, "max_transition_epoch": 1},
                1: {"params": MID, "restore": {"choose": "c", "candidates": ["none"]}}}
    fts = _fts(schedule, restore_choosers={"c": lambda e: "none"})
    _run(fts, 2)
    assert fts.restore_choices[0]["choice"] == "none"


@pytest.mark.parametrize("restore, match", [
    ({"choose": "c", "from": ["best"]}, r"unknown keys \['from'\]"),
    ({"candidates": ["best"]}, "needs the name of a restore chooser"),
    ({"choose": "c", "candidates": []}, "non-empty list of states"),
    ({"choose": "c", "candidates": "best"}, "non-empty list of states"),
    ({"choose": "c", "candidates": ["best", "best"]}, "more than once"),
    ({"choose": "c", "candidates": [{"choose": "d"}]}, "are states"),
    ({"choose": "c", "candidates": ["best:1"]}, "names phase 1's best, which is not an earlier phase"),
    ({"choose": "c", "candidates": ["latest"]}, "must be one of"),
], ids=["unknown_key", "no_name", "empty", "string", "repeated", "nested", "not_earlier", "unknown_state"])
def test_refused_chosen_restores(tmp_path, restore, match):
    schedule = {0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "restore": restore}}
    with pytest.raises(FTSConfigurationError, match=match):
        _fts(schedule, checkpoint=BestCheckpoint(tmp_path, "loss"), restore_choosers={"c": lowest})


def test_phase_0_may_not_choose(tmp_path):
    with pytest.raises(FTSConfigurationError, match="Phase 0 may not set `restore`"):
        _fts({0: {"params": HEAD, "restore": {"choose": "c"}}}, checkpoint=BestCheckpoint(tmp_path, "loss"),
             restore_choosers={"c": lowest})


@pytest.mark.parametrize("name", ["missing", "no_such_module_xyz.chooser", f"{__name__}.no_such_chooser"])
def test_an_unknown_chooser_is_refused_at_setup(tmp_path, name):
    schedule = {0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "restore": {"choose": name}}}
    with pytest.raises(FTSConfigurationError, match=rf"neither registered.*'{name}' \(phases \[1\]\)"):
        _fts(schedule, checkpoint=BestCheckpoint(tmp_path, "loss"))


def test_saved_candidates_need_a_checkpoint_policy():
    schedule = {0: {"params": HEAD, "max_transition_epoch": 1},
                1: {"params": MID, "restore": {"choose": "c", "candidates": ["best", "none"]}}}
    with pytest.raises(FTSConfigurationError, match=r"Phases \[1\] `restore` a saved state"):
        _fts(schedule, restore_choosers={"c": lowest})


def test_registration_is_validated():
    fts = TuningScheduler(_model(), _schedule(CHOSEN), optimizer=SGD, restore_choosers={"lowest": lowest})
    with pytest.raises(FTSConfigurationError, match="already registered"):
        fts.add_restore_chooser("lowest", highest)
    with pytest.raises(FTSConfigurationError, match="needs a name and a callable"):
        fts.add_restore_chooser("other", "not callable")  # type: ignore[arg-type]
    with pytest.raises(FTSConfigurationError, match="maps names to restore choosers"):
        TuningScheduler(_model(), _schedule(CHOSEN), optimizer=SGD, restore_choosers=[lowest])  # type: ignore


def test_the_configuration_hash_describes_the_chooser(tmp_path):
    kwargs = {"checkpoint": BestCheckpoint(tmp_path, "loss"), "restore_best": False}
    a = TuningScheduler(_model(), _schedule(CHOSEN), optimizer=SGD, restore_choosers={"lowest": lowest}, **kwargs)
    b = TuningScheduler(_model(), _schedule(CHOSEN), optimizer=SGD, restore_choosers={"lowest": highest}, **kwargs)
    assert a.config_dict()["restore_choosers"] == {"lowest": f"{__name__}.lowest"}
    assert a.config_hash != b.config_hash
    # and without one, the configuration is what it always was
    assert "restore_choosers" not in TuningScheduler(_model(), _schedule(CHOSEN), optimizer=SGD).config_dict()


def test_the_choice_is_saved_and_recorded_in_checkpoints(tmp_path):
    fts = _chosen(tmp_path)
    _run(fts, metrics=_val)
    choices = fts.restore_choices
    assert fts.state_dict()["restore_choices"] == choices
    saved = fts.save(tmp_path / "s.pt")
    assert load_payload(saved)["metadata"]["restore_choices"] == choices


def test_a_run_without_a_chosen_restore_saves_what_it_always_has(tmp_path):
    fts = _fts(CHOSEN | {2: {k: v for k, v in CHOSEN[2].items() if k != "restore"}},
               checkpoint=BestCheckpoint(tmp_path, "loss"))
    _run(fts, metrics=_val)
    assert "restore_choices" not in fts.state_dict()
    assert "restore_choices" not in load_payload(fts.save(tmp_path / "s.pt"))["metadata"]


@pytest.mark.parametrize("split", range(6))
def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path, split):
    """Saved at any epoch (before or after the chosen boundary), a resumed run ends where an uninterrupted one
    does, keeps the choices made before the save and calls the chooser only at boundaries it has not crossed."""
    calls = []

    def counted(event):
        calls.append(event.depth)
        return lowest(event)

    kwargs = {"checkpoint": "best", "restore_best": False, "restore_choosers": {"lowest": counted}}
    ref = _make(CHOSEN, tmp_path, "ref", **kwargs)
    expected = _run(ref, metrics=_val, lr_step=True)
    expected_choices = ref.restore_choices

    first = _make(CHOSEN, tmp_path, "first", **kwargs)
    for epoch in first.epochs(6):
        _train_epoch(first, epoch, _val(epoch), lr_step=True)
        if epoch == split:
            saved = first.save(tmp_path / "s.pt")
            break
    calls.clear()
    crossed = first.curr_depth == 2

    resumed = _make(CHOSEN, tmp_path, "resumed", **kwargs, model=_model(seed=7))
    resumed.load(saved)
    assert resumed.restore_choices == (expected_choices if crossed else [])
    final = _run(resumed, metrics=_val, start=split + 1, lr_step=True)
    _assert_same(expected, final)
    _assert_in_place(resumed)
    assert resumed.restore_choices == expected_choices
    assert calls == ([] if crossed else [2])


def test_unchanged_schedules_compile_as_before():
    """A schedule naming a fixed state is unaffected: `restore_targets` reads it as its own single target."""
    from finetuning_scheduler.core.schedule import restore_targets

    assert restore_targets(None) == ()
    assert restore_targets("best:1") == ("best:1",)
    assert restore_targets({"choose": "c"}) == ("best", "none")
    assert restore_targets({"choose": "c", "candidates": ["initial"]}) == ("initial",)
    fts = _fts({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": BASE}})
    _run(fts, 2)
    assert fts.restore_choices == []
