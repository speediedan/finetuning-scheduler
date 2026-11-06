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
"""Tests for the per-phase ``freeze`` and ``restore`` schedule keys (#193).

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
import os
from functools import partial

import pytest
import torch
from torch import nn

from finetuning_scheduler import CanonicalView, TuningScheduler
from finetuning_scheduler.core.checkpoint import BestCheckpoint
from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.schedule import refuse_direct_api_keys

SGD = partial(torch.optim.SGD, lr=0.1, momentum=0.9, weight_decay=0.01)
HEAD = ["2.weight", "2.bias"]
MID = ["1.weight", "1.bias"]
BASE = ["0.weight", "0.bias"]
# train the head, freeze it and train the middle, freeze that and train the base
STAGES = {
    0: {"params": HEAD, "max_transition_epoch": 2},
    1: {"params": MID, "freeze": HEAD, "max_transition_epoch": 4, "lr": 0.05},
    2: {"params": BASE, "freeze": ["1\\..*"], "lr": 0.02},
}
# each condition starts from the same state: the initial one
SWEEP = {
    0: {"params": HEAD, "max_transition_epoch": 2},
    1: {"params": MID, "freeze": HEAD, "restore": "initial", "max_transition_epoch": 4, "lr": 0.05},
    2: {"params": BASE, "freeze": MID, "restore": "initial", "lr": 0.02},
}


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))


def _data(epoch):
    gen = torch.Generator().manual_seed(100 + epoch)
    return torch.randn(8, 4, generator=gen), torch.randn(8, 2, generator=gen)


def _schedule(schedule):
    return {k: dict(v) for k, v in schedule.items()}


def _fts(schedule=STAGES, model=None, optimizer=SGD, **kwargs):
    model = _model() if model is None else model
    fts = TuningScheduler(model, _schedule(schedule), optimizer=optimizer, **kwargs)
    fts.setup()
    return fts


def _train_epoch(fts, epoch, metrics=None, lr_step=False):
    """One epoch of real gradients, stepped by the optimizer the way a loop does."""
    x, y = _data(epoch)
    (fts.model(x) - y).pow(2).mean().backward()
    fts.optimizer.step()
    fts.optimizer.zero_grad()
    if lr_step:
        fts.step_lr_schedulers("epoch")
    return fts.epoch_end(metrics)


def _values(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _optimized(fts):
    names = {id(p): n for n, p in fts.model.named_parameters()}
    return sorted(names[id(p)] for g in fts.optimizer.param_groups for p in g["params"])


def _assert_in_place(fts):
    """The module's trainable set and the optimizer's parameters are what the compiled schedule says of the
    phase."""
    expected = sorted(fts.compiled_schedule.trainable_at(fts.curr_depth))
    assert sorted(n for n, p in fts.model.named_parameters() if p.requires_grad) == expected
    assert _optimized(fts) == expected
    held = {id(p) for g in fts.optimizer.param_groups for p in g["params"]}
    assert all(id(p) in held for p in fts.optimizer.state)


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for n in a:
        assert torch.equal(a[n], b[n]), n


def _run(fts, epochs=6, metrics=None, start=0, **kwargs):
    for epoch in fts.epochs(epochs):
        if epoch >= start:
            _train_epoch(fts, epoch, metrics(epoch) if metrics else None, **kwargs)
    return _values(fts.model)


def test_compile_resolves_freeze_like_params():
    fts = _fts()
    compiled = fts.compiled_schedule
    assert [cp.frozen for cp in compiled.phases] == [(), tuple(HEAD), tuple(MID)]
    assert sorted(compiled.trainable_at(0)) == sorted(HEAD)
    assert sorted(compiled.trainable_at(1)) == sorted(MID)
    assert sorted(compiled.trainable_at(2)) == sorted(BASE)
    with pytest.raises(KeyError):
        compiled.trainable_at(3)
    # the schedule the hashes read carries the physical names
    assert fts.schedule[2]["freeze"] == MID


def test_freeze_entries_can_be_addresses():
    model = _model()
    view = CanonicalView(model, roles={"2.weight": "latent:head", "2.bias": "latent:head"})
    schedule = {0: {"params": ["role:latent:head"], "max_transition_epoch": 1},
                1: {"params": MID, "freeze": ["role:latent:head"]}}
    with pytest.warns(UserWarning, match="experimental"):
        compiled = compile_schedule(schedule, view)
    assert sorted(compiled.phases[1].frozen) == sorted(HEAD)


def test_freeze_takes_the_views_coupled_tensors_along():
    class Coupled(CanonicalView):
        def coupled(self, names):
            return ["2.bias"] if "2.weight" in names else []

    model = _model()
    schedule = {0: {"params": ["2.weight"], "max_transition_epoch": 1},
                1: {"params": MID, "freeze": ["2.weight"], "max_transition_epoch": 2},
                2: {"params": BASE}}
    compiled = compile_schedule(schedule, Coupled(model))
    assert compiled.phases[0].thawed == ("2.weight", "2.bias")
    assert compiled.phases[1].frozen == ("2.weight", "2.bias")


def test_digest_sees_freeze_only_when_present():
    view = CanonicalView(_model())
    without = compile_schedule({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID}}, view)
    with_freeze = compile_schedule({0: {"params": HEAD, "max_transition_epoch": 1},
                                    1: {"params": MID, "freeze": HEAD}}, view)
    assert without.digest != with_freeze.digest
    assert all(cp.frozen == () for cp in without.phases)


@pytest.mark.parametrize("schedule, match", [
    ({0: {"params": HEAD, "freeze": HEAD}}, "Phase 0 may not `freeze`"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "freeze": BASE}},
     "no earlier phase thawed"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "freeze": MID}},
     "no earlier phase thawed"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": HEAD, "freeze": HEAD}},
     "both thaws and freezes"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "freeze": HEAD, "max_transition_epoch": 2},
      2: {"params": BASE, "freeze": HEAD}}, "already froze"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "freeze": HEAD, "max_transition_epoch": 2},
      2: {"params": HEAD}}, "an earlier phase froze"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"freeze": HEAD}}, "leaves nothing trainable"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "freeze": "2.weight"}}, "non-empty list"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "freeze": []}}, "non-empty list"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "freeze": [1]}}, "is not a string"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "freeze": ["nothing.*"]}}, "nothing"),
    ({0: {"params": HEAD, "restore": "none"}}, "Phase 0 may not set `restore`"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "restore": "latest"}},
     "must be one of"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "restore": True}}, "must be one of"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "restore": "best"}}, "`checkpoint` policy"),
    ({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "restore": "initial"}},
     "`checkpoint` policy"),
])
def test_refused_schedules(schedule, match):
    with pytest.raises(FTSConfigurationError, match=match):
        _fts(schedule)


def test_restore_none_needs_no_checkpoint():
    fts = _fts({0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "restore": "none"}})
    _run(fts, 2)
    assert fts.curr_depth == 1


def test_named_state_needs_a_policy_that_can_keep_one(tmp_path):
    class Plain:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            if name in ("pin_state", "state_path"):
                raise AttributeError(name)
            return getattr(self.inner, name)

    with pytest.raises(FTSConfigurationError, match="cannot keep one"):
        _fts(SWEEP, checkpoint=Plain(BestCheckpoint(tmp_path, "loss")))


def test_lightning_refuses_the_keys():
    with pytest.raises(FTSConfigurationError, match="freeze.*restore|restore.*freeze"):
        refuse_direct_api_keys(_schedule(SWEEP))


def test_externally_driven_refuses_the_keys():
    model = _model()
    fts = TuningScheduler(model, optimizer=SGD)
    with pytest.raises(FTSConfigurationError, match="freeze"):
        fts.attach(object(), _schedule(STAGES))


@pytest.mark.parametrize("optimizer", [
    SGD, partial(torch.optim.AdamW, lr=0.01, weight_decay=0.1), partial(torch.optim.Adam, lr=0.01),
    partial(torch.optim.RMSprop, lr=0.01, momentum=0.9, weight_decay=0.01),
], ids=["sgd", "adamw", "adam", "rmsprop"])
def test_frozen_tensors_hold_and_leave_the_optimizer(optimizer):
    """Momentum and weight decay would keep moving a frozen tensor left in the optimizer."""
    fts = _fts(optimizer=optimizer)
    seen = []
    for epoch in fts.epochs(6):
        _assert_in_place(fts)
        seen.append(fts.curr_depth)
        before = _values(fts.model)
        _train_epoch(fts, epoch)
        after = _values(fts.model)
        frozen = {n for cp in fts.compiled_schedule.phases[:fts.curr_depth + 1] for n in cp.frozen}
        trainable = set(fts.compiled_schedule.trainable_at(fts.curr_depth))
        for n in after:
            assert torch.equal(before[n], after[n]) == (n not in trainable), n
            assert fts.model.get_parameter(n).grad is None
        assert frozen.isdisjoint(trainable)
    assert seen == [0, 0, 1, 1, 2, 2]


def test_zeroed_gradients_do_not_move_a_frozen_tensor():
    fts = _fts()
    for epoch in fts.epochs(4):
        x, y = _data(epoch)
        (fts.model(x) - y).pow(2).mean().backward()
        fts.optimizer.step()
        fts.optimizer.zero_grad(set_to_none=False)
        fts.epoch_end()
        if fts.curr_depth == 1:
            break
    held = _values(fts.model)["2.weight"]
    for epoch in range(2):
        x, y = _data(epoch)
        (fts.model(x) - y).pow(2).mean().backward()
        fts.optimizer.step()
        fts.optimizer.zero_grad(set_to_none=False)
    assert torch.equal(held, fts.model.get_parameter("2.weight"))


def test_a_partly_frozen_group_keeps_its_options():
    """Freezing some of a group's parameters leaves the group, with its options, on the rest."""
    schedule = {0: {"params": HEAD, "max_transition_epoch": 1},
                1: {"params": MID, "freeze": ["2.bias"], "lr": 0.05}}
    fts = _fts(schedule)
    _run(fts, 2)
    assert [(len(g["params"]), g["lr"]) for g in fts.optimizer.param_groups] == [(1, 0.1), (2, 0.05)]
    _assert_in_place(fts)


def test_no_decay_groups_empty_one_at_a_time():
    schedule = {0: {"params": HEAD, "max_transition_epoch": 1},
                1: {"params": MID, "freeze": ["2.bias"], "max_transition_epoch": 2, "lr": 0.05},
                2: {"params": BASE, "freeze": ["2.weight"], "lr": 0.02}}
    fts = _fts(schedule, no_decay=["bias"])
    _run(fts, 3)
    _assert_in_place(fts)
    assert [(len(g["params"]), g["lr"], g["weight_decay"]) for g in fts.optimizer.param_groups] == [
        (1, 0.05, 0.01), (1, 0.05, 0.0), (1, 0.02, 0.01), (1, 0.02, 0.0)]


@pytest.mark.parametrize("lr_scheduler, attr", [
    (partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5), "base_lrs"),
    (partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda e: 0.5 ** e), "lr_lambdas"),
    (partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=8), "base_lrs"),
    (partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9), "base_lrs"),
    (partial(torch.optim.lr_scheduler.MultiplicativeLR, lr_lambda=lambda e: 0.9), "lr_lambdas"),
], ids=["step", "lambda", "cosine", "exponential", "multiplicative"])
def test_lr_schedulers_stay_aligned_with_the_groups(lr_scheduler, attr):
    fts = _fts(lr_scheduler=lr_scheduler)
    for epoch in fts.epochs(6):
        _train_epoch(fts, epoch, lr_step=True)
        assert len(getattr(fts.lr_scheduler, attr)) == len(fts.optimizer.param_groups) == 1
    assert fts.optimizer.param_groups[0]["params"][0] is fts.model.get_parameter("0.weight")
    assert len(fts.lr_scheduler.get_last_lr()) == 1


def test_reduce_on_plateau_stays_aligned_with_the_groups():
    fts = _fts(lr_scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, patience=0),
               lr_scheduler_cfg={"monitor": "loss"})
    for epoch in fts.epochs(6):
        _train_epoch(fts, epoch)
        fts.step_lr_schedulers("epoch", {"loss": 1.0})
        assert len(fts.lr_scheduler.min_lrs) == len(fts.optimizer.param_groups) == 1


def test_a_phase_can_only_freeze():
    schedule = {0: {"params": HEAD + MID, "max_transition_epoch": 1}, 1: {"freeze": MID}}
    fts = _fts(schedule)
    _run(fts, 2)
    assert fts.curr_depth == 1
    _assert_in_place(fts)
    assert sorted(fts.export_book()["sets"]) == ["phase_0"]


def test_frozen_bn_stops_tracking_with_the_option_off():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    schedule = {0: {"params": ["2.weight", "2.bias", "1.weight", "1.bias"], "max_transition_epoch": 1},
                1: {"params": ["0.weight", "0.bias"], "freeze": ["1.weight", "1.bias"]}}
    fts = _fts(schedule, model=model, frozen_bn_track_running_stats=False)
    assert model[1].track_running_stats
    _run(fts, 2)
    assert fts.curr_depth == 1 and not model[1].track_running_stats


def test_phase_event_names_what_the_phase_freezes():
    seen = {}
    fts = TuningScheduler(_model(), _schedule(STAGES), optimizer=SGD)
    fts.add_phase_hook(
        lambda e: seen.update({e.depth: (e.frozen_params, fts.model.get_parameter("2.bias").requires_grad)}))
    fts.setup()
    _run(fts)
    # start hooks run before the phase's thaw and freeze
    assert seen == {0: ((), True), 1: (tuple(HEAD), True), 2: (tuple(MID), False)}


def test_hashes_see_both_keys():
    plain = {0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID}}
    frozen = {0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "freeze": HEAD}}
    restored = {0: {"params": HEAD, "max_transition_epoch": 1}, 1: {"params": MID, "restore": "none"}}
    hashes = {_fts(s).config_hash for s in (plain, frozen, restored)}
    assert len(hashes) == 3


def test_a_state_from_another_freeze_is_refused(tmp_path):
    fts = _fts()
    _train_epoch(fts, 0)
    saved = fts.save(tmp_path / "s.pt")
    other = {**STAGES, 2: {"params": BASE, "lr": 0.02}}
    with pytest.raises(FTSConfigurationError, match="different fine-tuning schedule"):
        _fts(other).load(saved)


def _val(epoch):
    return {"loss": [1.0, 0.9, 0.95, 0.8, 0.85, 0.7, 0.75, 0.6][epoch]}


def _make(schedule, tmp_path, name, **kwargs):
    checkpoint = kwargs.pop("checkpoint", None)
    if checkpoint == "best":
        checkpoint = BestCheckpoint(tmp_path / name, "loss")
    return _fts(schedule, checkpoint=checkpoint,
                lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma=0.5), **kwargs)


@pytest.mark.parametrize("split", range(6))
@pytest.mark.parametrize("schedule, checkpoint, restore_best", [
    (STAGES, None, None),
    (STAGES, "best", None),
    (STAGES, "best", False),
    (SWEEP, "best", False),
    (SWEEP, "best", None),
], ids=["freeze", "freeze_best", "freeze_ckpt_only", "sweep", "sweep_best"])
def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path, split, schedule, checkpoint, restore_best):
    """The acceptance test: saved at any epoch (before, at or after a freeze or a restore), a resumed run ends where an
    uninterrupted one does, with the same model, optimizer and lr scheduler state."""
    kwargs = {"checkpoint": checkpoint}
    if restore_best is not None:
        kwargs["restore_best"] = restore_best
    ref = _make(schedule, tmp_path, "ref", **kwargs)
    expected = _run(ref, metrics=_val if checkpoint else None, lr_step=True)

    first = _make(schedule, tmp_path, "first", **kwargs)
    for epoch in first.epochs(6):
        _train_epoch(first, epoch, _val(epoch) if checkpoint else None, lr_step=True)
        if epoch == split:
            saved = first.save(tmp_path / "s.pt")
            break

    resumed = _make(schedule, tmp_path, "resumed", **kwargs, model=_model(seed=7))
    resumed.load(saved)
    _assert_in_place(resumed)
    assert resumed.curr_depth == first.curr_depth
    final = _run(resumed, metrics=_val if checkpoint else None, start=split + 1, lr_step=True)

    _assert_same(expected, final)
    _assert_in_place(resumed)
    assert resumed.curr_depth == ref.curr_depth == 2
    ref_state, state = ref.optimizer.state_dict(), resumed.optimizer.state_dict()
    assert [g["lr"] for g in ref_state["param_groups"]] == [g["lr"] for g in state["param_groups"]]
    assert ref_state["state"].keys() == state["state"].keys()
    for k, v in ref_state["state"].items():
        assert torch.equal(v["momentum_buffer"], state["state"][k]["momentum_buffer"])
    assert ref.lr_scheduler.state_dict() == resumed.lr_scheduler.state_dict()


@pytest.mark.parametrize("split", [1, 3])
def test_resume_state_matches_an_uninterrupted_one(split):
    """The same for a loop that checkpoints the model and optimizer itself."""
    expected = _run(_fts(), lr_step=False)
    first = _fts()
    for epoch in first.epochs(6):
        _train_epoch(first, epoch)
        if epoch == split:
            state = first.resume_state()
            model_state, optim_state = first.model.state_dict(), first.optimizer.state_dict()
            break
    resumed = _fts(model=_model(seed=7))
    resumed.load_resume_state(state)
    resumed.optimizer.load_state_dict(optim_state)
    resumed.model.load_state_dict(model_state)
    _assert_same(expected, _run(resumed, start=split + 1))
    _assert_in_place(resumed)


def test_restore_initial_starts_each_stage_from_the_same_state(tmp_path):
    ends, starts = {}, {}
    model = _model()
    initial = _values(model)
    fts = TuningScheduler(model, _schedule(SWEEP), optimizer=SGD, checkpoint=BestCheckpoint(tmp_path, "loss"),
                          restore_best=False, lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1,
                                                                   gamma=0.5))
    fts.add_phase_hook(lambda e: starts.update({e.depth: _values(model)}), when="start")
    fts.add_phase_hook(lambda e: ends.update({e.depth: _values(model)}), when="end")
    fts.setup()
    lrs = {}
    for epoch in fts.epochs(6):
        if epoch in (0, 2, 4):
            lrs[fts.curr_depth] = [g["lr"] for g in fts.optimizer.param_groups]
            assert not fts.optimizer.state  # momentum starts over with the state
            # every tensor but those thawed so far, restored and untouched, is at its initial value
            _assert_same(initial, _values(model))
        _assert_in_place(fts)
        _train_epoch(fts, epoch, {"loss": 1.0 - epoch / 10}, lr_step=True)
    # start hooks see the state the phase starts from; end hooks the phase's outcome
    for depth in (1, 2):
        _assert_same(initial, starts[depth])
    assert not torch.equal(ends[0]["2.weight"], initial["2.weight"])
    assert not torch.equal(ends[1]["1.weight"], initial["1.weight"])
    assert torch.equal(ends[1]["2.weight"], initial["2.weight"])
    # the lr scheduler restarts: a stage does not inherit the decay of the stage before it
    assert lrs == {0: [0.1], 1: [0.05], 2: [0.02]}
    assert fts.restored_count == 0


def test_restore_overrides_restore_best_for_its_boundary(tmp_path):
    schedule = {0: {"params": HEAD, "max_transition_epoch": 1},
                1: {"params": MID, "restore": "none", "max_transition_epoch": 2},
                2: {"params": BASE}}
    fts = _fts(schedule, checkpoint=BestCheckpoint(tmp_path, "loss"))
    _run(fts, 3, metrics=lambda e: {"loss": 1.0 - e / 10})
    assert fts.restored_count == 1  # the default at the second boundary, and not at the first

    schedule = {0: {"params": HEAD, "max_transition_epoch": 1},
                1: {"params": MID, "restore": "best", "max_transition_epoch": 2},
                2: {"params": BASE}}
    fts = _fts(schedule, checkpoint=BestCheckpoint(tmp_path, "loss"), restore_best=False)
    _run(fts, 3, metrics=lambda e: {"loss": 1.0 - e / 10})
    assert fts.restored_count == 1  # at the first boundary, and not at the second


def test_restore_none_keeps_the_state_the_phase_ended_with(tmp_path):
    schedule = {0: {"params": HEAD, "max_transition_epoch": 2}, 1: {"params": MID, "restore": "none"}}
    fts = _fts(schedule, checkpoint=BestCheckpoint(tmp_path, "loss"))
    ended, started, after_first = {}, {}, {}
    fts.add_phase_hook(lambda e: ended.update(_values(fts.model)), when="end")
    fts.add_phase_hook(lambda e: started.update(_values(fts.model)), when="start")
    # the second epoch is worse than the first, so `best` would go back to the first
    for epoch in fts.epochs(3):
        if epoch == 2:
            break
        _train_epoch(fts, epoch, {"loss": [1.0, 2.0][epoch]})
        if epoch == 0:
            after_first.update(_values(fts.model))
    assert fts.curr_depth == 1 and fts.restored_count == 0
    _assert_same(ended, started)
    assert not torch.equal(after_first["2.weight"], fts.model.get_parameter("2.weight"))


def test_a_best_from_before_a_freeze_replays_it(tmp_path):
    """Restoring a best saved before the freeze brings back parameter groups holding the frozen tensors, and the
    phases after it are replayed onto them."""
    fts = _fts(checkpoint=BestCheckpoint(tmp_path, "loss"))
    seen = []
    for epoch in fts.epochs(6):
        _assert_in_place(fts)
        seen.append(fts.curr_depth)
        _train_epoch(fts, epoch, {"loss": 1.0 + epoch})  # the first epoch stays the best
    assert seen == [0, 0, 1, 1, 2, 2] and fts.restored_count == 2


def test_new_optimizer_after_a_freeze_holds_only_what_trains():
    schedule = {0: {"params": HEAD, "max_transition_epoch": 1},
                1: {"params": MID, "freeze": HEAD, "max_transition_epoch": 2, "lr": 0.05},
                2: {"params": BASE, "lr": 0.02,
                    "new_optimizer": {"optimizer_init": {"class_path": "torch.optim.Adam", "init_args": {"lr": 0.01}}}}}
    fts = _fts(schedule)
    _run(fts, 3)
    assert isinstance(fts.optimizer, torch.optim.Adam)
    _assert_in_place(fts)


def test_the_initial_state_outlives_pruning_and_a_resume(tmp_path):
    fts = _make(SWEEP, tmp_path, "run", checkpoint="best", restore_best=False)
    initial = fts._checkpoint.state_path("initial")
    for epoch in fts.epochs(6):
        _train_epoch(fts, epoch, _val(epoch), lr_step=True)
        assert (tmp_path / "run" / initial.rsplit("/", 1)[1]).exists()
        if epoch == 2:
            saved = fts.save(tmp_path / "s.pt")
            break
    state = torch.load(saved, weights_only=False)["tuning_scheduler"]["checkpoint"]
    assert state["states"] == {"initial": initial}


def test_a_missing_initial_state_is_named_on_resume(tmp_path):
    fts = _make(SWEEP, tmp_path, "run", checkpoint="best", restore_best=False)
    initial = fts._checkpoint.state_path("initial")
    _train_epoch(fts, 0, _val(0))
    saved = fts.save(tmp_path / "s.pt")
    os.remove(initial)
    resumed = _make(SWEEP, tmp_path, "run", checkpoint="best", restore_best=False)
    with pytest.raises(FileNotFoundError, match="the state 'initial'"):
        resumed.load(saved)
