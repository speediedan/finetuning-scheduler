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
        ({0: {"params": ["l2.weight"], "max_transition_epoch": 1},
          1: {"params": ["l1.weight"], "new_optimizer": {"optimizer_init": {}}}}, "not yet supported"),
        ("not/a/schedule.yaml", "schedule.yaml"),
    ],
    ids=["noncontiguous", "reinit", "missing_file"],
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
