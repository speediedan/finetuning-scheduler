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
"""Tests for the direct PyTorch controller (#51).

Lightning-free: these run in the ``core-no-lightning`` job too.
"""
import random
import warnings
from functools import partial

import pytest
import torch
from torch import nn

from finetuning_scheduler.core.controller import FTSController
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.host import TorchHost
from finetuning_scheduler.core.transitions import CallableTransition, Decision, PlateauTransition


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0, self.l1, self.l2 = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(torch.relu(self.l0(x)))))


SCHEDULE = {
    0: {"params": ["l2.weight", "l2.bias"], "max_transition_epoch": 1},
    1: {"params": ["l1.weight", "l1.bias"], "max_transition_epoch": 3, "lr": 0.01},
    2: {"params": ["l0.weight", "l0.bias"]},
}


def _trainable(model):
    return sorted(n for n, p in model.named_parameters() if p.requires_grad)


def _train_epoch(fts, model, steps=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(8, 4, generator=g)
        loss = (model(x) - x.sum(1, keepdim=True)).pow(2).mean()
        loss.backward()
        fts.optimizer.step()
        fts.optimizer.zero_grad()


def _run(fts, model, max_epochs, metrics_fn=None):
    seen = []
    for epoch in fts.epochs(max_epochs):
        seen.append((epoch, fts.curr_depth, _trainable(model), len(fts.optimizer.param_groups)))
        _train_epoch(fts, model, seed=epoch)
        fts.epoch_end(metrics_fn(epoch) if metrics_fn else None)
    return seen


def test_epoch_driven_phases():
    model = Net()
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1))
    fts.setup()
    seen = _run(fts, model, 5)
    assert [(e, d, g) for e, d, _, g in seen] == [(0, 0, 1), (1, 1, 2), (2, 1, 2), (3, 2, 3), (4, 2, 3)]
    assert seen[1][2] == ["l1.bias", "l1.weight", "l2.bias", "l2.weight"]
    assert [pg["lr"] for pg in fts.optimizer.param_groups] == [0.1, 0.01, 1e-5]  # phase 2 defaults to base_max_lr
    assert (fts.epoch, fts.global_step, fts.is_final_phase) == (5, 10, True)


def test_optimizer_instance_reconfigured_for_phase0():
    model = Net()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)  # every parameter: not phase 0's
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    fts = FTSController(model, SCHEDULE, optimizer=opt, lr_scheduler=sched)
    fts.setup()
    assert fts.optimizer is opt and fts.lr_scheduler is sched
    assert [len(pg["params"]) for pg in opt.param_groups] == [2]
    assert _trainable(model) == ["l2.bias", "l2.weight"]


def test_optimizer_instance_not_enforced_warns():
    model = Net()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    fts = FTSController(model, SCHEDULE, optimizer=opt, enforce_phase0_params=False)
    with pytest.warns(UserWarning, match="enforce_phase0_params"):
        fts.setup()
    assert [len(pg["params"]) for pg in opt.param_groups] == [6]


def test_plateau_transitions_and_final_stop():
    """Scripted metrics drive plateau transitions; in the final phase the plateau stops the run."""
    model = Net()
    sched = {0: {"params": ["l2.weight", "l2.bias"]}, 1: {"params": ["l1.weight", "l1.bias"]}}
    fts = FTSController(model, sched, optimizer=partial(torch.optim.SGD, lr=0.1),
                        transition=PlateauTransition("val_loss", patience=1))
    fts.setup()
    losses = [1.0, 1.0, 0.5, 0.5, 0.5]
    seen = _run(fts, model, 10, metrics_fn=lambda e: {"val_loss": losses[e]})
    assert [(e, d) for e, d, _, _ in seen] == [(0, 0), (1, 0), (2, 1), (3, 1)]
    assert fts.should_stop and "did not improve" in fts.stop_reason


def test_no_transition_defined_refused():
    with pytest.raises(FTSConfigurationError, match=r"Phases \[0\] set neither"):
        FTSController(Net(), {0: {"params": ["l2.weight"]}, 1: {"params": ["l1.weight"]}},
                      optimizer=partial(torch.optim.SGD, lr=0.1)).setup()


def test_default_schedule_with_policy(tmp_path):
    model = Net()
    fts = FTSController(model, None, optimizer=partial(torch.optim.SGD, lr=0.1), log_dir=tmp_path,
                        transition=CallableTransition(lambda ctx: ctx.epoch >= ctx.phase_start_epoch + 1))
    fts.setup()
    assert (tmp_path / "Net_ft_schedule.yaml").exists()
    assert len(fts.schedule) == 3 and fts.schedule[0]["params"] == ["l2.bias", "l2.weight"]
    seen = _run(fts, model, 4)
    assert [d for _, d, _, _ in seen] == [0, 1, 2, 2]


def test_hooks_order_rng_isolation_and_removal():
    model = Net()
    calls = []
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1))

    def start_hook(event):
        calls.append(("start", event.depth, event.epoch))
        torch.rand(100), random.random()  # must not disturb the training streams

    fts.add_phase_hook(start_hook)
    handle = fts.add_phase_hook(lambda e: calls.append(("removed", e.depth)), when="end")
    fts.add_phase_hook(lambda e: calls.append(("end", e.depth)), when="end")
    handle.remove()
    torch.manual_seed(0)
    random.seed(0)
    fts.setup()
    after_setup = (torch.rand(1).item(), random.random())
    torch.manual_seed(0)
    random.seed(0)
    assert (torch.rand(1).item(), random.random()) == after_setup
    _run(fts, model, 4)
    assert calls == [("start", 0, 0), ("end", 0), ("start", 1, 1), ("end", 1), ("start", 2, 3), ("end", 2)]
    with pytest.raises(FTSConfigurationError, match="`when` must be"):
        fts.add_phase_hook(start_hook, when="middle")  # type: ignore[arg-type]


def test_hook_optimizer_steps_not_counted():
    model = Net()
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.0))
    fts.add_phase_hook(lambda e: e.controller.optimizer.step())
    fts.setup()
    assert fts.global_step == 0


def test_fingerprints():
    model = Net()
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1), fingerprint_batches=2)
    fts.setup()
    batches = [torch.full((2,), float(i)) for i in range(3)]
    assert list(fts.observe_batches(batches)) == batches
    assert len(fts.fingerprints[0]) == 2 and fts.fingerprints[0][0] != fts.fingerprints[0][1]


def test_resume_matches_uninterrupted_run():
    """Restoring controller, optimizer and model state at an epoch boundary continues exactly as an uninterrupted
    run."""
    torch.manual_seed(1)
    ref_model = Net()
    init_state = {k: v.clone() for k, v in ref_model.state_dict().items()}
    ref = FTSController(ref_model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9))
    ref.setup()
    ref_seen = _run(ref, ref_model, 5)

    model = Net()
    model.load_state_dict(init_state)
    first = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9))
    first.setup()
    seen = _run(first, model, 2)
    saved = (first.state_dict(), first.optimizer.state_dict(), {k: v.clone() for k, v in model.state_dict().items()})

    resumed_model = Net()
    resumed_model.load_state_dict(saved[2])
    resumed = FTSController(resumed_model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9))
    resumed.setup()
    resumed.load_state_dict(saved[0])
    resumed.optimizer.load_state_dict(saved[1])
    seen += _run(resumed, resumed_model, 5)
    assert seen == ref_seen
    for (n, p), (_, q) in zip(ref_model.named_parameters(), resumed_model.named_parameters()):
        assert torch.equal(p, q), n


def test_load_state_dict_refuses_other_schedule():
    a = FTSController(Net(), SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1))
    a.setup()
    other = {**SCHEDULE, 0: {**SCHEDULE[0], "max_transition_epoch": 2}}
    b = FTSController(Net(), other, optimizer=partial(torch.optim.SGD, lr=0.1))
    b.setup()
    with pytest.raises(FTSConfigurationError, match="different fine-tuning schedule"):
        b.load_state_dict(a.state_dict())


def test_config_hash():
    def make(**kw):
        fts = FTSController(Net(), SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1), **kw)
        fts.setup()
        return fts

    assert make().config_hash == make().config_hash
    assert make().config_hash != make(base_max_lr=1e-4).config_hash
    assert make(metadata={"run": "a"}).config_hash != make(metadata={"run": "b"}).config_hash
    assert set(make().versions()) == {"finetuning_scheduler", "torch", "python"}


def test_step_lr_schedulers():
    model = Net()
    fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=1.0),
                        lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5),
                        lr_scheduler_cfg={"interval": "step", "frequency": 2})
    fts.setup()
    _train_epoch(fts, model, steps=1)
    fts.step_lr_schedulers("step")  # global_step 1: not a multiple of the frequency
    fts.step_lr_schedulers("epoch")  # wrong interval
    assert fts.optimizer.param_groups[0]["lr"] == 1.0
    _train_epoch(fts, model, steps=1)
    fts.step_lr_schedulers("step")
    assert fts.optimizer.param_groups[0]["lr"] == 0.5
    rlrop = FTSController(Net(), SCHEDULE, optimizer=partial(torch.optim.SGD, lr=1.0),
                          lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau)
    rlrop.setup()
    with pytest.raises(FTSConfigurationError, match="needs `lr_scheduler_cfg"):
        rlrop.step_lr_schedulers("epoch", {"val_loss": 1.0})


def test_rank_disagreement_follows_greatest():
    class SkewedHost(TorchHost):
        def reduce_max(self, value):
            return max(value, int(Decision.ADVANCE))

    model = Net()
    fts = FTSController(model, {0: {"params": ["l2.weight"]}, 1: {"params": ["l1.weight"]}},
                        optimizer=partial(torch.optim.SGD, lr=0.1), host=SkewedHost(model),
                        transition=CallableTransition(lambda ctx: Decision.CONTINUE))
    fts.setup()
    with pytest.warns(UserWarning, match="Ranks disagreed"):
        assert fts.epoch_end() == Decision.ADVANCE
    fts.epoch_start()
    assert fts.curr_depth == 1


def test_phase_config_validation():
    class Cfg:
        def __init__(self, weight: float):
            if weight < 0:
                raise ValueError("weight must be non-negative")
            self.weight = weight

    sched = {0: {"params": ["l2.weight"], "max_transition_epoch": 1, "phase_config": {"weight": 0.5}},
             1: {"params": ["l1.weight"], "phase_config": {"weight": 2.0}}}
    fts = FTSController(Net(), sched, optimizer=partial(torch.optim.SGD, lr=0.1), phase_config_type=Cfg)
    fts.setup()
    assert fts.phase_config.weight == 0.5
    bad = {0: {**sched[0], "phase_config": {"weight": -1.0}}, 1: sched[1]}
    with pytest.raises(FTSConfigurationError, match="Phase 0's `phase_config` failed validation"):
        FTSController(Net(), bad, optimizer=partial(torch.optim.SGD, lr=0.1), phase_config_type=Cfg).setup()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda m: setattr(m.l2, "weight", nn.Parameter(torch.zeros(1, 4, dtype=torch.int8), requires_grad=False)),
         "must be floating point"),
        (lambda m: setattr(m.l0, "weight", nn.Parameter(torch.empty(4, 4, device="meta"))), "meta device"),
    ],
    ids=["int8", "meta"],
)
def test_unschedulable_params_refused(mutate, match):
    model = Net()
    mutate(model)
    with pytest.raises(FTSConfigurationError, match=match):
        FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1)).setup()


def test_meta_frozen_allowed_when_unscheduled():
    model = Net()
    model.l0.weight = nn.Parameter(torch.empty(4, 4, device="meta"))
    sched = {0: {"params": ["l2.weight", "l2.bias"], "max_transition_epoch": 1},
             1: {"params": ["l1.weight", "l1.bias"]}}
    fts = FTSController(model, sched, optimizer=partial(torch.optim.SGD, lr=0.1), allow_meta_frozen=True)
    with pytest.warns(UserWarning, match="meta device"):
        fts.setup()


@pytest.mark.parametrize(
    "schedule, match",
    [
        ({0: {"params": ["l2.weight"], "max_transition_epoch": 1}, 2: {"params": ["l1.weight"]}}, "non-contiguous"),
        ({0: {"params": ["l2.weight"], "max_transition_epoch": 1,
              "new_optimizer": {"optimizer_init": {"class_path": "torch.optim.Adam"}}},
          1: {"params": ["l1.weight"]}}, "initial training phase which is an invalid"),
        ("not/a/schedule.yaml", "schedule.yaml"),
    ],
    ids=["noncontiguous", "phase0_reinit", "missing_file"],
)
def test_schedule_refusals(schedule, match):
    with pytest.raises((FTSConfigurationError, FileNotFoundError), match=match):
        FTSController(Net(), schedule, optimizer=partial(torch.optim.SGD, lr=0.1)).setup()


def test_setup_guards():
    fts = FTSController(Net(), SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1))
    with pytest.raises(FTSConfigurationError, match="Call `FTSController.setup"):
        fts.epoch_end()
    fts.setup()
    with pytest.raises(FTSConfigurationError, match="already been called"):
        fts.setup()
    with pytest.raises(FTSConfigurationError, match="must be an `Optimizer` or a factory"):
        FTSController(Net(), SCHEDULE, optimizer=3).setup()  # type: ignore[arg-type]
    with pytest.raises(FTSConfigurationError, match="not an `Optimizer`"):
        FTSController(Net(), SCHEDULE, optimizer=lambda pgs: object()).setup()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FTSController(Net(), SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1)).setup()


def test_pg_overrides():
    """Phase-0 overrides reach the factory's groups; later phases' overrides reach the groups they add, while a
    ``no_decay`` group keeps ``weight_decay=0``."""
    model = Net()
    sched = {
        0: {"params": ["l2.weight", "l2.bias"], "max_transition_epoch": 1, "pg_overrides": {"weight_decay": 0.1}},
        1: {"params": ["l1.weight", "l1.bias"], "pg_overrides": {"weight_decay": 0.01, "momentum": 0.5}},
    }
    fts = FTSController(model, sched, optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9), no_decay=["bias"])
    fts.setup()
    assert [(g["weight_decay"], g["momentum"]) for g in fts.optimizer.param_groups] == [(0.1, 0.9), (0.0, 0.9)]
    _run(fts, model, 2)
    assert [(g["weight_decay"], g["momentum"]) for g in fts.optimizer.param_groups[2:]] == [(0.01, 0.5), (0.0, 0.5)]
    with pytest.raises(FTSConfigurationError, match="needs an optimizer factory"):
        FTSController(Net(), sched, optimizer=torch.optim.SGD(Net().parameters(), lr=0.1)).setup()


def test_step_transitions_and_guard():
    """``max_transition_step`` and step-granular policies advance between steps via ``step_end``; skipping
    ``step_end`` while a step transition is configured is refused."""
    model = Net()
    sched = {0: {"params": ["l2.weight", "l2.bias"], "max_transition_step": 3},
             1: {"params": ["l1.weight", "l1.bias"], "max_transition_step": 5},
             2: {"params": ["l0.weight", "l0.bias"]}}
    fts = FTSController(model, sched, optimizer=partial(torch.optim.SGD, lr=0.1))
    fts.setup()
    depths = []
    for _ in range(7):
        _train_epoch(fts, model, steps=1)
        fts.step_end()
        depths.append(fts.curr_depth)
    assert depths == [0, 0, 1, 1, 2, 2, 2]
    guarded = FTSController(Net(), sched, optimizer=partial(torch.optim.SGD, lr=0.1))
    guarded.setup()
    with pytest.raises(FTSConfigurationError, match="without `FTSController.step_end"):
        _train_epoch(guarded, guarded.model, steps=2)
    policy = CallableTransition(lambda ctx: ctx.global_step >= ctx.phase_start_step + 2, granularity="step")
    fts2 = FTSController(Net(), {0: {"params": ["l2.weight"]}, 1: {"params": ["l1.weight"]}},
                         optimizer=partial(torch.optim.SGD, lr=0.1), transition=policy)
    fts2.setup()
    for _ in range(3):
        _train_epoch(fts2, fts2.model, steps=1)
        fts2.step_end()
    assert fts2.curr_depth == 1


def test_lr_cast_message():
    with pytest.raises(FTSConfigurationError, match="explicit schedule could not be cast to a float"):
        FTSController(Net(), {0: {"params": ["l2.weight"], "max_transition_epoch": 1},
                              1: {"params": ["l1.weight"], "lr": "fast"}},
                      optimizer=partial(torch.optim.SGD, lr=0.1)).setup()


REINIT_SCHEDULE = {
    0: {"params": ["l2.weight", "l2.bias"], "max_transition_epoch": 1},
    1: {"params": ["l1.weight", "l1.bias"], "max_transition_epoch": 2, "lr": 0.05,
        "new_optimizer": {"optimizer_init": {"class_path": "Adam", "init_args": {"lr": 0.01}}},
        "new_lr_scheduler": {"lr_scheduler_init": {"class_path": "StepLR", "init_args": {"step_size": 1}},
                             "pl_lrs_cfg": {"interval": "epoch"}}},
    2: {"params": ["l0.weight", "l0.bias"]},
}


def test_explicit_reinitialization():
    """A phase's ``new_optimizer`` replaces the optimizer (phase 0's groups rebuilt, later phases replayed) and its
    ``new_lr_scheduler`` replaces the scheduler; the retired objects refuse to step."""
    model = Net()
    fts = FTSController(model, REINIT_SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1),
                        lr_scheduler=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9))
    fts.setup()
    old_opt, old_sched = fts.optimizer, fts.lr_scheduler
    _run(fts, model, 2)
    assert isinstance(fts.optimizer, torch.optim.Adam) and fts.optimizer is not old_opt
    assert isinstance(fts.lr_scheduler, torch.optim.lr_scheduler.StepLR) and fts.lr_scheduler.optimizer is fts.optimizer
    assert [len(g["params"]) for g in fts.optimizer.param_groups] == [2, 2]
    assert fts.global_step == 4  # counting moved to the new optimizer
    with pytest.raises(FTSConfigurationError, match="replaced when phase 1 reinitialized the optimizer"):
        old_opt.step()
    with pytest.raises(FTSConfigurationError, match="replaced when phase 1 reinitialized it"):
        old_sched.step()
    _run(fts, model, 3)
    assert fts.curr_depth == 2 and [len(g["params"]) for g in fts.optimizer.param_groups] == [2, 2, 2]


def test_implicit_reinitialization():
    model = Net()
    fts = FTSController(model, {0: {"params": ["l2.weight", "l2.bias"], "max_transition_epoch": 1},
                                1: {"params": ["l1.weight", "l1.bias"]}},
                        optimizer=partial(torch.optim.SGD, lr=0.1),
                        lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1),
                        reinit_optim_cfg={"optimizer_init": {"class_path": "torch.optim.AdamW",
                                                             "init_args": {"lr": 0.02}}},
                        reinit_lr_cfg={"lr_scheduler_init": {"class_path": "torch.optim.lr_scheduler.ConstantLR",
                                                             "init_args": {"factor": 0.5, "total_iters": 3}}})
    with pytest.raises(FTSConfigurationError, match="Specifying both `ft_schedule` and `reinit_lr_cfg`"):
        fts.setup()
    fts = FTSController(model, None, optimizer=partial(torch.optim.SGD, lr=0.1),
                        lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1),
                        transition=CallableTransition(lambda ctx: ctx.epoch >= ctx.phase_start_epoch + 1),
                        reinit_optim_cfg={"optimizer_init": {"class_path": "torch.optim.AdamW",
                                                             "init_args": {"lr": 0.02}}},
                        reinit_lr_cfg={"lr_scheduler_init": {"class_path": "torch.optim.lr_scheduler.ConstantLR",
                                                             "init_args": {"factor": 0.5, "total_iters": 3}}})
    fts.setup()
    _run(fts, model, 2)
    assert isinstance(fts.optimizer, torch.optim.AdamW)
    assert isinstance(fts.lr_scheduler, torch.optim.lr_scheduler.ConstantLR)


@pytest.mark.parametrize(
    "phase1, kwargs, match",
    [
        ({"new_lr_scheduler": {"lr_scheduler_init": {"class_path": "StepLR", "init_args": {"step_size": 1}},
                               "pl_lrs_cfg": {"name": "x"}}}, {}, r"remove \['name'\]"),
        ({"new_lr_scheduler": {"lr_scheduler_init": {"class_path": "StepLR", "init_args": {"step_size": 1}},
                               "pl_lrs_cfg": {"interval": "batch"}}}, {}, "must be \"step\" or \"epoch\""),
        ({"new_optimizer": {"optimizer_init": {"class_path": "NotAnOptimizer"}}}, {}, "fully qualified"),
        ({}, {"reinit_lr_cfg": {"lr_scheduler_init": {"class_path": "StepLR", "init_args": {"step_size": 1}},
                                "init_pg_lrs": [0.1]}}, "init_pg_lrs"),
    ],
    ids=["pl_lrs_key", "interval", "class_path", "implicit_init_pg_lrs"],
)
def test_reinitialization_refusals(phase1, kwargs, match):
    schedule = None if kwargs else {0: {"params": ["l2.weight"], "max_transition_epoch": 1},
                                    1: {"params": ["l1.weight"], **phase1}}
    transition = CallableTransition(lambda ctx: False) if kwargs else None
    with pytest.raises(FTSConfigurationError, match=match):
        FTSController(Net(), schedule, optimizer=partial(torch.optim.SGD, lr=0.1), transition=transition,
                      **kwargs).setup()


@pytest.mark.parametrize("when", ["before_epoch_end", "after_epoch_end"])
def test_epoch_lr_stepping_does_not_depend_on_placement(when):
    """With ``frequency=2``, the scheduler steps after epochs 2 and 4 (as under Lightning), whether the loop calls
    ``step_lr_schedulers`` before or after ``epoch_end``."""
    model = Net()
    fts = FTSController(model, {0: {"params": ["l2.weight", "l2.bias"]}}, optimizer=partial(torch.optim.SGD, lr=1.0),
                        lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5),
                        lr_scheduler_cfg={"interval": "epoch", "frequency": 2})
    fts.setup()
    lrs = []
    for epoch in fts.epochs(4):
        lrs.append(fts.optimizer.param_groups[0]["lr"])
        _train_epoch(fts, model, steps=1, seed=epoch)
        if when == "before_epoch_end":
            fts.step_lr_schedulers("epoch")
            fts.epoch_end()
        else:
            fts.epoch_end()
            fts.step_lr_schedulers("epoch")
    assert lrs == [1.0, 1.0, 0.5, 0.5]


def test_transition_check_honors_max_depth():
    """Phase 1 is the final phase when ``max_depth=1``, so it need not set a transition."""
    schedule = {0: {"params": ["l2.weight", "l2.bias"], "max_transition_epoch": 1}, 1: {"params": ["l1.weight"]},
                2: {"params": ["l0.weight"]}}
    FTSController(Net(), schedule, optimizer=partial(torch.optim.SGD, lr=0.1), max_depth=1).setup()
    with pytest.raises(FTSConfigurationError, match=r"Phases \[1\] set neither"):
        FTSController(Net(), schedule, optimizer=partial(torch.optim.SGD, lr=0.1)).setup()


def test_leaving_the_loop_early_does_not_end_training():
    ended = []
    for exit_by in ("break", "error", "completion"):
        model = Net()
        fts = FTSController(model, SCHEDULE, optimizer=partial(torch.optim.SGD, lr=0.1))
        fts.add_phase_hook(lambda e: ended.append((exit_by, e.depth)), when="end")
        fts.setup()
        try:
            for epoch in fts.epochs(4):
                _train_epoch(fts, model, steps=1, seed=epoch)
                if exit_by == "break":
                    break
                if exit_by == "error":
                    raise RuntimeError("the loop failed")
        except RuntimeError:
            pass
    assert [e for e in ended if e[0] != "completion"] == []
    assert ("completion", 2) in ended


def test_suspend_step_counting_for_user_state_dict_calls():
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model = Net()
    fts = FTSController(model, {0: {"params": ["l2.weight", "l2.bias"], "max_transition_step": 4},
                                1: {"params": ["l1.weight", "l1.bias"]}}, optimizer=partial(torch.optim.SGD, lr=0.1))
    fts.setup()
    _train_epoch(fts, model, steps=1)
    fts.step_end()
    with fts.suspend_step_counting():
        get_state_dict(model, fts.optimizer)  # steps a stateless optimizer to initialize its state
    assert fts.global_step == 1


def test_in_place_compile_is_detected():
    from finetuning_scheduler.core.controller import _is_compiled

    model = Net()
    assert not _is_compiled(model)
    model.compile()
    assert _is_compiled(model)


def test_max_epochs_extension_follows_the_schedules_last_phase():
    """As the Lightning callback does, ``max_epochs`` is extended only when the schedule's last phase sets
    ``max_transition_epoch``, even when ``max_depth`` stops earlier."""
    schedule = {0: {"params": ["l2.weight", "l2.bias"], "max_transition_epoch": 1},
                1: {"params": ["l1.weight", "l1.bias"], "max_transition_epoch": 3}, 2: {"params": ["l0.weight"]}}
    model = Net()
    fts = FTSController(model, schedule, optimizer=partial(torch.optim.SGD, lr=0.1), max_depth=1)
    fts.setup()
    assert len(_run(fts, model, 2)) == 2


def test_breaking_after_should_stop_ends_training():
    """``if fts.should_stop: break`` still ends training: the final end hooks run."""
    ended = []
    model = Net()
    fts = FTSController(model, {0: {"params": ["l2.weight", "l2.bias"]}}, optimizer=partial(torch.optim.SGD, lr=0.1),
                        transition=PlateauTransition("loss", patience=1))
    fts.add_phase_hook(lambda e: ended.append(e.depth), when="end")
    fts.setup()
    for epoch in fts.epochs(10):
        _train_epoch(fts, model, steps=1, seed=epoch)
        fts.epoch_end({"loss": 1.0})
        if fts.should_stop:
            break
    assert ended == [0]
