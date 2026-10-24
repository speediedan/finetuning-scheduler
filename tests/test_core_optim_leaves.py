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
"""Tests for the core's phase-0 enforcement and reinitialization leaves (#51)."""
import enum
import warnings

import pytest
import torch
from torch import nn

from finetuning_scheduler.core.config import canonical_json
from finetuning_scheduler.core.optim import (apply_reinit_pg_lrs, build_phase0_optimizer, capture_lr_state,
                                             clean_optim_lr_pgs, inspect_phase0_opt_state, reconfigure_lrs_for_phase0,
                                             warn_unapplied_lr_lambdas)


def _net():
    return nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))


SCHEDULE = {0: {"params": ["1.weight", "1.bias"]}, 1: {"params": ["0.weight", "0.bias"]}}


def test_inspect_phase0_opt_state():
    """Counts, and a name summary only when the optimizer's trainable set differs from phase 0's."""
    net = _net()
    for n, p in net.named_parameters():
        p.requires_grad = n.startswith("1.")
    aligned = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
    assert inspect_phase0_opt_state(aligned, net, SCHEDULE, list, list) == (0, 2, 4, {})
    net[0].weight.requires_grad = True  # trainable in the optimizer but not scheduled in phase 0
    misaligned = torch.optim.SGD(net.parameters(), lr=0.1)
    no_grad, init_cnt, total_cnt, summary = inspect_phase0_opt_state(
        misaligned, net, SCHEDULE, list, lambda names: [f"logical::{n}" for n in names])
    assert (no_grad, init_cnt, total_cnt) == (1, 2, 4)
    assert summary == {"optim_params": ["logical::0.weight", "logical::1.weight", "logical::1.bias"],
                       "phase_0_params": ["logical::1.weight", "logical::1.bias"]}


def test_inspect_phase0_opt_state_inspects_each_phase_once():
    """``optim_inspect`` (a strategy adapter's, which can walk the module) is called once per phase plus once for
    phase 0's check, however many parameters the module has (#122)."""
    net = nn.Sequential(*[nn.Linear(2, 2) for _ in range(50)])
    names = [n for n, _ in net.named_parameters()]
    schedule = {0: {"params": names[:50]}, 1: {"params": names[50:]}}
    for n, p in net.named_parameters():
        p.requires_grad = n in schedule[0]["params"]
    calls = []

    def inspect(pl):
        calls.append(len(pl))
        return list(pl)

    optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
    assert inspect_phase0_opt_state(optimizer, net, schedule, inspect, list) == (0, 50, 100, {})
    assert len(calls) == 1 + len(schedule)


@pytest.mark.parametrize("plateau", [False, True], ids=["steplr", "rlrop"])
def test_clean_and_reconfigure_for_phase0(plateau):
    net = _net()
    opt = torch.optim.SGD([{"params": [net[0].weight]}, {"params": [net[0].bias]}], lr=0.1)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, min_lr=[0.0, 0.0]) if plateau
             else torch.optim.lr_scheduler.StepLR(opt, step_size=1))
    assert clean_optim_lr_pgs([opt], [sched]) == [2]
    assert opt.param_groups == [] and sched.last_epoch == -1
    assert plateau or sched.base_lrs == []
    opt.add_param_group({"params": [net[1].weight], "lr": 0.5})
    if plateau:
        sched.min_lrs = [0.0, 0.0, 0.01]
    reconfigure_lrs_for_phase0([sched], [2])
    assert sched._last_lr == [0.5]
    if plateau:
        assert sched.min_lrs == [0.01]
    else:
        assert sched.last_epoch == 0  # the initial step ran again


def test_capture_lr_state():
    net = _net()
    opt = torch.optim.SGD(net.parameters(), lr=0.25)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    state, lrs = capture_lr_state(sched, opt)
    assert lrs == [0.25] and state["base_lrs"] == [0.25]
    state["base_lrs"].append(1.0)
    assert sched.base_lrs == [0.25]  # a copy, not a view
    assert capture_lr_state(None, opt) == ({}, [0.25])


def test_build_phase0_optimizer_sets_initial_lr():
    net = _net()
    opt = build_phase0_optimizer(torch.optim.SGD, {"lr": 0.1},
                                 [{"params": [net[0].weight]}, {"params": [net[0].bias], "initial_lr": 0.7}])
    assert [g["initial_lr"] for g in opt.param_groups] == [0.1, 0.7]


@pytest.mark.parametrize(
    "directive, expected_lr, expected_initial",
    [
        ({}, [0.2, 0.1], [0.2, 0.1]),  # each group's initial_lr, falling back to its lr
        ({"use_current_optimizer_pg_lrs": True}, [0.5, 0.1], [0.2, 0.1]),  # current lrs
        ({"init_pg_lrs": [0.3, 0.4]}, [0.3, 0.4], [0.3, 0.4]),  # explicit lrs also reset initial_lr
    ],
    ids=["initial_lr", "current", "init_pg_lrs"],
)
def test_apply_reinit_pg_lrs(directive, expected_lr, expected_initial):
    net = _net()
    opt = torch.optim.SGD([{"params": [net[0].weight], "lr": 0.5, "initial_lr": 0.2},
                           {"params": [net[0].bias], "lr": 0.1}], lr=0.1)
    apply_reinit_pg_lrs(opt, directive)
    assert [g["lr"] for g in opt.param_groups] == expected_lr
    assert [g.get("initial_lr", g["lr"]) for g in opt.param_groups] == expected_initial


@pytest.mark.parametrize("apply_lambdas, expected", [(False, 1), (True, 0)], ids=["warn", "apply"])
def test_warn_unapplied_lr_lambdas(apply_lambdas, expected):
    opt = torch.optim.SGD(_net().parameters(), lr=0.1)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.9), torch.optim.lr_scheduler.StepLR(opt, 1)]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warn_unapplied_lr_lambdas(schedulers, apply_lambdas)
    assert sum("has lr_lambdas" in str(w.message) for w in record) == expected


def test_config_enum_values():
    class Mode(enum.Enum):
        MIN = "min"

    assert canonical_json({"mode": Mode.MIN}) == '{"mode":"min"}'
