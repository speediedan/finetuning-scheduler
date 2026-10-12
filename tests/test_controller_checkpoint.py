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
"""Tests for the direct PyTorch controller's checkpoint policy, best-checkpoint restore, gates and resume (#51)."""
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
    assert cp.observe({"val_loss": 1.0}, 0) and cp.best_path == str(tmp_path / "best.pt")
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
    assert "dirpath" not in str(cp.config())  # paths stay out of the configuration hash
    with pytest.raises(ValueError, match="`mode`"):
        BestCheckpoint(tmp_path, "x", mode="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`include`"):
        BestCheckpoint(tmp_path, "x", include="some")  # type: ignore[arg-type]


def test_restore_best_at_boundaries(tmp_path):
    """At each transition the best checkpoint's weights are restored (here, saved before the plateau epochs)."""
    torch.manual_seed(0)
    model = Net()
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1),
                        checkpoint=BestCheckpoint(tmp_path, "val_loss"))
    fts.setup()
    seen_best = []

    def at_start(event):
        if event.depth > 0:
            best = load_payload(tmp_path / "best.pt")
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
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1),
                        checkpoint=BestCheckpoint(tmp_path, "val_loss"))
    fts.add_gate(lambda e: GateResult(accept=e.depth != 1, reason="phase 1 hurt the probe", record={"score": 0.3}),
                 rank_zero_only=rank_zero_only)
    fts.setup()
    pinned = {}
    fts.add_phase_hook(lambda e: pinned.setdefault(e.depth, {k: v.clone() for k, v in model.state_dict().items()}))
    _run(fts, model, 6, metrics_fn=lambda e: {"val_loss": 1.0 - 0.1 * e})
    assert fts.should_stop and fts.stop_reason.startswith("veto: phase 1") and fts.curr_depth == 1
    boundary = load_payload(tmp_path / "boundary.pt")
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
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1),
                        checkpoint=BestCheckpoint(tmp_path, "val_loss"), restore_best=False)
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
    best_l2 = load_payload(tmp_path / "best.pt")["model"]["l2.weight"]
    _train_epoch(fts, model)
    assert fts.restore_best()
    assert torch.equal(model.l2.weight, best_l2) and (counter.n, on_best.n) == (9, 7)


def test_scheduled_only_checkpoints(tmp_path):
    model = Net()
    sched = {0: {"params": ["l2.weight", "l2.bias"], "max_transition_epoch": 1}, 1: {"params": ["l1.weight"]}}
    fts = FTSController(model, sched, optimizer=partial(torch.optim.SGD, lr=0.1),
                        checkpoint=BestCheckpoint(tmp_path, "val_loss", include="scheduled"))
    fts.setup()
    _train_epoch(fts, model)
    fts.epoch_end({"val_loss": 1.0})
    assert set(load_payload(tmp_path / "best.pt")["model"]) == {"l2.weight", "l2.bias", "l1.weight"}


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
