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
"""Tests for ``FTSController``'s checkpoint policy, best-checkpoint restore, gates and resume (#51)."""
import os
import random
from functools import partial

import pytest
import torch

from finetuning_scheduler.core.checkpoint import BestCheckpoint, load_payload
from finetuning_scheduler.core.controller import FTSController
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.transitions import GateResult, PlateauTransition
from tests.test_controller import _run, _train_epoch, Net, SCHEDULE


class Counter:
    def __init__(self):
        self.n = 0

    def state_dict(self):
        return {"n": self.n}

    def load_state_dict(self, state):
        self.n = state["n"]


def test_best_checkpoint_policy(tmp_path):
    """First observation saves; then only strict improvement; non-finite is worst; ``max`` mode flips the order."""
    cp = BestCheckpoint(tmp_path, "val_loss")
    assert cp.observe({"val_loss": 1.0}, 0) and cp.best_path == cp.path_for("best")
    assert os.path.dirname(cp.best_path) == str(tmp_path) and os.path.basename(cp.best_path).startswith("best-1-")
    assert not cp.observe({"val_loss": 1.0}, 0)
    assert not cp.observe({"val_loss": float("nan")}, 1)
    assert cp.observe({"val_loss": 0.5}, 1) and (float(cp.best_score), cp.best_depth) == (0.5, 1)
    acc = BestCheckpoint(tmp_path, "acc", mode="max")
    assert acc.observe({"acc": 0.1}, 0) and not acc.observe({"acc": 0.05}, 0) and acc.observe({"acc": 0.2}, 0)
    with pytest.warns(UserWarning, match="was not reported"):
        assert not cp.observe({"other": 0.1}, 0)
    restored = BestCheckpoint(tmp_path, "val_loss")
    restored.load_state_dict(cp.state_dict())
    assert (restored.best_path, restored.best_depth) == (cp.best_path, 1)
    assert cp.path_for("initial") != cp.path_for("initial")  # every checkpoint gets a new name
    with pytest.raises(ValueError, match="No checkpoint is kept yet"):
        BestCheckpoint(tmp_path, "x").path_for("best")
    assert "dirpath" not in str(cp.config())  # paths stay out of the configuration hash
    with pytest.raises(ValueError, match="`mode`"):
        BestCheckpoint(tmp_path, "x", mode="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`include`"):
        BestCheckpoint(tmp_path, "x", include="some")  # type: ignore[arg-type]


def test_restore_best_at_boundaries(tmp_path):
    """At each transition the best checkpoint's weights are restored (here, saved before the plateau epochs)."""
    torch.manual_seed(0)
    model = Net()
    cp = BestCheckpoint(tmp_path, "val_loss")
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1), checkpoint=cp)
    fts.setup()
    seen_best = []

    def at_start(event):
        if event.depth > 0:
            best = load_payload(cp.best_path)
            seen_best.append(all(torch.equal(best["model"][n], p) for n, p in model.state_dict().items()))

    fts.add_phase_hook(at_start)
    losses = [1.0, 2.0, 0.5, 3.0, 3.0]  # epochs 1 and 3 are worse: their weights must not survive the boundary
    _run(fts, model, 5, metrics_fn=lambda e: {"val_loss": losses[e]})
    assert seen_best == [True, True] and fts.restored_count == 2
    with pytest.raises(FTSConfigurationError, match="needs a `checkpoint` policy"):
        FTSController(Net(), SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1), restore_best=True).setup()


@pytest.mark.parametrize("rank_zero_only", [False, True], ids=["all_ranks", "rank_zero"])
def test_gate_veto_rolls_back(tmp_path, rank_zero_only):
    """A vetoed phase stops training and restores the best as of the phase's start; records are kept."""
    torch.manual_seed(0)
    model = Net()
    cp = BestCheckpoint(tmp_path, "val_loss")
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1), checkpoint=cp)
    fts.add_gate(lambda e: GateResult(accept=e.depth != 1, reason="phase 1 hurt the probe", record={"score": 0.3}),
                 rank_zero_only=rank_zero_only)
    fts.setup()
    pinned = {}
    fts.add_phase_hook(lambda e: pinned.setdefault(e.depth, {k: v.clone() for k, v in model.state_dict().items()}))
    _run(fts, model, 6, metrics_fn=lambda e: {"val_loss": 1.0 - 0.1 * e})
    assert fts.should_stop and fts.stop_reason.startswith("veto: phase 1") and fts.curr_depth == 1
    boundary = load_payload(cp.boundary_path(1))
    assert all(torch.equal(boundary["model"][k], v) for k, v in model.state_dict().items())
    assert [r["accept"] for r in fts.gate_records] == [True, False]
    assert fts.gate_records[1]["record"] == {"score": 0.3}


def test_final_gate_and_bool_result(tmp_path):
    model = Net()
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1),
                        checkpoint=BestCheckpoint(tmp_path, "val_loss"))
    events = []
    handle = fts.add_gate(lambda e: events.append((e.depth, e.final)) or True)
    fts.setup()
    _run(fts, model, 5, metrics_fn=lambda e: {"val_loss": 1.0})
    assert events == [(0, False), (1, False), (2, True)] and not fts.should_stop
    handle.remove()


def test_statefuls_and_public_restore_best(tmp_path):
    model = Net()
    counter, on_best = Counter(), Counter()
    cp = BestCheckpoint(tmp_path, "val_loss")
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1), checkpoint=cp, restore_best=False)
    fts.register_stateful("plain", counter)
    fts.register_stateful("on_best", on_best, restore_on_best=True)
    with pytest.raises(FTSConfigurationError, match="needs `state_dict"):
        fts.register_stateful("bad", object())
    fts.setup()
    assert not fts.restore_best()
    counter.n = on_best.n = 7
    _train_epoch(fts, model)
    fts.epoch_end({"val_loss": 1.0})  # saved as best, with both counters at 7
    counter.n = on_best.n = 9
    best_l2 = load_payload(cp.best_path)["model"]["l2.weight"]
    _train_epoch(fts, model)
    assert fts.restore_best()
    assert torch.equal(model.l2.weight, best_l2) and (counter.n, on_best.n) == (9, 7)


def test_scheduled_only_checkpoints(tmp_path):
    model = Net()
    sched = {0: {"params": ["l2.weight", "l2.bias"], "max_transition_epoch": 1}, 1: {"params": ["l1.weight"]}}
    cp = BestCheckpoint(tmp_path, "val_loss", include="scheduled")
    fts = FTSController(model, sched, optimizer=partial(torch.optim.SGD, lr=0.1), checkpoint=cp)
    fts.setup()
    _train_epoch(fts, model)
    fts.epoch_end({"val_loss": 1.0})
    assert set(load_payload(cp.best_path)["model"]) == {"l2.weight", "l2.bias", "l1.weight"}


def test_save_load_resumes_exactly(tmp_path):
    """Saving mid-run (after an optimizer reinitialization and boundary restores) and loading into a fresh
    controller continues exactly as the uninterrupted run, random streams included."""
    reinit_schedule = {**SCHEDULE, 1: {**SCHEDULE[1], "new_optimizer": {"optimizer_init": {
        "class_path": "Adam", "init_args": {"lr": 0.01}}}}}

    def make(model, dirpath):
        fts = FTSController(model, reinit_schedule, optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9),
                            checkpoint=BestCheckpoint(dirpath, "val_loss"), transition=None)
        counter = Counter()
        fts.register_stateful("counter", counter)
        fts.setup()
        return fts, counter

    def run(fts, model, counter, max_epochs, trace):
        for epoch in fts.epochs(max_epochs):
            counter.n += 1
            _train_epoch(fts, model, seed=epoch)
            trace.append((epoch, fts.curr_depth, round(torch.rand(1).item(), 6), random.random()))
            fts.epoch_end({"val_loss": [1.0, 0.8, 0.9, 0.7, 0.6][epoch]})

    torch.manual_seed(3), random.seed(3)
    ref_model = Net()
    init = {k: v.clone() for k, v in ref_model.state_dict().items()}
    ref, ref_counter = make(ref_model, tmp_path / "ref")
    ref_trace: list = []
    run(ref, ref_model, ref_counter, 5, ref_trace)

    torch.manual_seed(3), random.seed(3)
    model = Net()
    model.load_state_dict(init)
    first, counter = make(model, tmp_path / "run")
    trace: list = []
    run(first, model, counter, 2, trace)
    first.save(tmp_path / "resume.pt")
    torch.rand(5), random.random()  # disturb the streams; load must restore them

    resumed_model = Net()
    resumed, resumed_counter = make(resumed_model, tmp_path / "run")
    resumed.load(tmp_path / "resume.pt")
    assert isinstance(resumed.optimizer, torch.optim.Adam) and resumed_counter.n == 2
    run(resumed, resumed_model, resumed_counter, 5, trace)
    assert trace == ref_trace
    for (n, p), (_, q) in zip(ref_model.named_parameters(), resumed_model.named_parameters()):
        assert torch.equal(p, q), n


def test_checkpoint_files_are_replaced_only_when_complete(tmp_path):
    """Writes go beside the target and are moved over it, and a copy replaces a file or a directory alike."""
    from finetuning_scheduler.core.checkpoint import copy_checkpoint, save_payload

    target = tmp_path / "ckpt.pt"
    save_payload({"v": 1}, target)
    save_payload({"v": 2}, target)
    assert load_payload(target) == {"v": 2} and sorted(p.name for p in tmp_path.iterdir()) == ["ckpt.pt"]
    shard_dir = tmp_path / "sharded"
    shard_dir.mkdir()
    (shard_dir / "part").write_text("x")
    copy_checkpoint(str(shard_dir), str(target))  # a directory over an existing file
    assert (target / "part").read_text() == "x"
    copy_checkpoint(str(tmp_path / "sharded" / "part"), str(target))  # and a file over a directory
    assert target.is_file()
    assert not [p for p in tmp_path.iterdir() if ".tmp-" in p.name]


def test_resume_keeps_the_metrics_a_pending_boundary_sees(tmp_path):
    """A state saved while a transition is pending resumes with the metrics the boundary's gate receives."""
    events = {}

    def run(save_at=None, load_from=None):
        torch.manual_seed(0)
        model = Net()
        fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1),
                            transition=PlateauTransition("loss", patience=1),
                            checkpoint=BestCheckpoint(tmp_path / "c", "loss"))
        seen = []
        fts.add_gate(lambda e: seen.append(dict(e.metrics)) or True)
        fts.setup()
        if load_from is not None:
            fts.load(load_from)
        for epoch in fts.epochs(4):
            _train_epoch(fts, model, steps=1, seed=epoch)
            fts.epoch_end({"loss": 1.0})
            if epoch == save_at:
                fts.save(tmp_path / "pending.pt")
                return seen
        return seen

    events["straight"] = run()
    run(save_at=1)  # loss plateaus with patience 1, so an ADVANCE is pending after epoch 1
    events["resumed"] = run(load_from=tmp_path / "pending.pt")
    assert events["straight"] and events["resumed"]
    assert {k: float(v) for k, v in events["resumed"][0].items()} == {"loss": 1.0}
    assert {k: float(v) for k, v in events["straight"][0].items()} == {"loss": 1.0}


def test_resuming_a_finished_run_reopens_its_phase(tmp_path):
    """A state saved after ``epochs`` returned (training ended) resumes with the current phase open: its end hooks
    run again at its real end."""
    ends = []
    torch.manual_seed(0)
    model = Net()
    first = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1))
    first.setup()
    _run(first, model, 2)
    first.save(tmp_path / "ended.pt")
    model2 = Net()
    fts = FTSController(model2, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1))
    fts.add_phase_hook(lambda e: ends.append(e.depth), when="end")
    fts.setup()
    fts.load(tmp_path / "ended.pt")
    _run(fts, model2, 5)
    assert ends == [1, 2]  # phase 1 (current when saved) ends at its real boundary, then the final phase


def test_interrupted_writes_leave_a_usable_checkpoint(tmp_path):
    """A temporary file or directory left by an interrupted write is cleared by the next write, and replacing a
    checkpoint moves the old one aside until the new one is in place."""
    from finetuning_scheduler.core.checkpoint import copy_checkpoint, replace_checkpoint, save_payload

    target = tmp_path / "ckpt.pt"
    (tmp_path / "ckpt.pt.tmp").write_text("left by a crashed write")
    save_payload({"v": 1}, target)
    assert load_payload(target) == {"v": 1} and not (tmp_path / "ckpt.pt.tmp").exists()
    new = tmp_path / "new"
    new.mkdir()
    (new / "part").write_text("new")
    (tmp_path / "ckpt.pt.old").write_text("left by a crashed replace")
    replace_checkpoint(str(new), str(target))
    assert (target / "part").read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt.pt"]
    copy_checkpoint(str(target), str(tmp_path / "copy") + "/")  # a trailing separator names the same target
    assert (tmp_path / "copy" / "part").read_text() == "new"


def test_best_checkpoint_top_k():
    """Top-k keeps the ``save_top_k`` best: while not full every observation saves, then only one strictly better
    than the worst kept, which it replaces (the first saved of equals goes first); ``-1`` keeps every one."""
    cp = BestCheckpoint("ckpts", "loss", save_top_k=2)
    assert cp.observe({"loss": 3.0}, 0) and cp.observe({"loss": float("nan")}, 0)  # not full: both kept
    first, nan = list(cp.best_k_models)
    assert cp.observe({"loss": 2.0}, 1)  # replaces the non-finite one, the worst
    assert nan not in cp.best_k_models and len(cp.best_k_models) == 2
    assert not cp.observe({"loss": 3.0}, 1)  # only as good as the worst: not kept
    assert cp.observe({"loss": 1.0}, 2) and first not in cp.best_k_models
    assert (float(cp.best_score), cp.best_depth) == (1.0, 2)
    assert [float(v) for v in cp.best_k_models.values()] == [2.0, 1.0]
    ties = BestCheckpoint("ckpts", "acc", mode="max", save_top_k=2)
    assert ties.observe({"acc": 0.5}, 0) and ties.observe({"acc": 0.5}, 0)
    tied_first = list(ties.best_k_models)[0]
    assert ties.best_path == tied_first  # the first saved of equals is the best
    assert ties.observe({"acc": 0.6}, 0) and tied_first not in ties.best_k_models  # and is evicted first
    keep_all = BestCheckpoint("ckpts", "loss", save_top_k=-1)
    assert all(keep_all.observe({"loss": v}, 0) for v in (1.0, 2.0, 3.0)) and len(keep_all.best_k_models) == 3
    restored = BestCheckpoint("ckpts", "loss", save_top_k=2)
    restored.load_state_dict(cp.state_dict())
    assert restored.best_k_models.keys() == cp.best_k_models.keys() and restored.best_path == cp.best_path
    assert cp.config()["init_args"]["save_top_k"] == 2
    for bad in (0, -2, True, 1.5):
        with pytest.raises(ValueError, match="`save_top_k`"):
            BestCheckpoint("ckpts", "loss", save_top_k=bad)  # type: ignore[arg-type]


def _ckpt_names(dirpath):
    return sorted(p.name for p in dirpath.iterdir())


@pytest.mark.parametrize("save_top_k", [1, 2])
def test_only_needed_checkpoints_are_kept(tmp_path, save_top_k):
    """A replaced checkpoint is removed once the new one is written; a phase's rollback point is kept while the
    phase runs, and the setup state only while it is one."""
    torch.manual_seed(0)
    model = Net()
    cp = BestCheckpoint(tmp_path, "val_loss", save_top_k=save_top_k, save_initial=True)
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1), checkpoint=cp)
    fts.setup()
    initial = cp.boundary_path(0)
    assert initial is not None and _ckpt_names(tmp_path) == [os.path.basename(initial)]
    seen = []

    def kept(event):
        seen.append(_ckpt_names(tmp_path))
        expected = {os.path.basename(p) for p in cp.best_k_models} | {os.path.basename(cp.boundary_path(event.depth))}
        assert set(seen[-1]) == expected

    fts.add_phase_hook(kept)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6]  # every epoch improves, so each rollback point soon leaves the best
    _run(fts, model, 5, metrics_fn=lambda e: {"val_loss": losses[e]})
    assert len(seen) == 2 and not os.path.exists(initial)
    assert set(_ckpt_names(tmp_path)) == ({os.path.basename(p) for p in cp.best_k_models}
                                          | {os.path.basename(cp.boundary_path(2))})
    assert len(cp.best_k_models) == save_top_k


def _resumable_run(dirpath, save_top_k, init, gate=None, saves=None, stop_after=None, load_from=None,
                   losses=(1.0, 2.0, 0.5, 3.0, 3.0)):
    """Train ``Net`` under ``SCHEDULE`` from ``init`` with a best checkpoint, saving after the epochs ``saves``
    maps to a path, stopping after ``stop_after`` (an interruption) or resuming from ``load_from``."""
    model = Net()
    model.load_state_dict(init)
    cp = BestCheckpoint(dirpath, "val_loss", save_top_k=save_top_k)
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1), checkpoint=cp, transition=None)
    if gate is not None:
        fts.add_gate(gate)
    fts.setup()
    if load_from is not None:
        fts.load(load_from)
    for epoch in fts.epochs(5):
        _train_epoch(fts, model, seed=epoch)
        fts.epoch_end({"val_loss": losses[epoch]})
        if saves and epoch in saves:
            fts.save(saves[epoch])
        if epoch == stop_after:
            break
    return model, fts


def _same_params(a, b):
    return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


@pytest.mark.parametrize("save_top_k", [1, 2])
def test_resume_from_a_save_older_than_the_latest_best(tmp_path, save_top_k):
    """A run interrupted after writing a newer best than its latest save refers to resumes from that save as if
    uninterrupted: the next boundary restores the save's best, not the interrupted run's (#52)."""
    torch.manual_seed(0)
    init = Net().state_dict()
    ref, _ = _resumable_run(tmp_path / "ref", save_top_k, init)
    # the save after epoch 0 refers to epoch 0's best; epoch 2 then writes a better one before the run stops
    _resumable_run(tmp_path / "run", save_top_k, init, saves={0: tmp_path / "s.pt"}, stop_after=2)
    resumed, fts = _resumable_run(tmp_path / "run", save_top_k, init, load_from=tmp_path / "s.pt")
    assert fts.restored_count == 2 and _same_params(ref, resumed)


def test_resumed_veto_rolls_back_to_the_saved_rollback_point(tmp_path):
    """A veto after resuming restores the rollback point the save refers to, even after the interrupted run pinned
    a later one (#52)."""
    torch.manual_seed(0)
    init = Net().state_dict()
    veto_phase_1 = {"on": True}

    def gate(event):
        return GateResult(accept=not (veto_phase_1["on"] and event.depth == 1), reason="phase 1 rejected")

    losses = (1.0, 0.9, 0.8, 0.7, 0.6)
    ref, ref_fts = _resumable_run(tmp_path / "ref", 1, init, gate=gate, losses=losses)
    assert ref_fts.stop_reason.startswith("veto: phase 1")
    veto_phase_1["on"] = False  # the interrupted run accepts phase 1 and pins phase 2's rollback point
    _, first = _resumable_run(tmp_path / "run", 1, init, gate=gate, losses=losses, saves={1: tmp_path / "s.pt"},
                              stop_after=3)
    assert first.curr_depth == 2
    veto_phase_1["on"] = True
    resumed, fts = _resumable_run(tmp_path / "run", 1, init, gate=gate, losses=losses, load_from=tmp_path / "s.pt")
    assert fts.stop_reason.startswith("veto: phase 1") and _same_params(ref, resumed)


def test_resuming_a_save_whose_checkpoints_are_gone(tmp_path):
    """Only the latest save's checkpoints are kept: resuming an older save whose best was removed fails at once,
    and a kept non-best checkpoint that was removed is dropped with a warning (#52)."""
    torch.manual_seed(0)
    init = Net().state_dict()
    losses = (1.0, 0.9, 0.5, 0.7, 0.6)  # phase 2's rollback point is epoch 2's best; epoch 3's is kept beside it
    saves = {0: tmp_path / "s0.pt", 3: tmp_path / "s3.pt"}
    _resumable_run(tmp_path / "run", 2, init, losses=losses, saves=saves)
    with pytest.raises(FileNotFoundError, match="best checkpoint this state refers to"):
        _resumable_run(tmp_path / "run", 2, init, losses=losses, load_from=saves[0])
    saved = load_payload(saves[3])["controller"]["checkpoint"]
    worst = max(saved["entries"], key=lambda e: float(e["score"]))["path"]
    assert worst not in {path for path, _ in saved["boundaries"].values()}
    os.remove(worst)
    with pytest.warns(UserWarning, match="Dropping 1 kept checkpoint"):
        _, fts = _resumable_run(tmp_path / "run", 2, init, losses=losses, load_from=saves[3])
    assert worst not in fts._checkpoint.best_k_models
