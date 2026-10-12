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
"""Tests for the core's phase leaves, including twins of the Lightning helpers they copy (#51)."""
import pytest
import torch
from lightning.pytorch.callbacks import BaseFinetuning
from torch import nn

from finetuning_scheduler.core.phase import (depth_remaining, flatten_modules, freeze_module_tree, lr_reinit_cfg_for,
                                             map_param_groups, thaw_plan)


def _bn_net():
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)))


def _state(module):
    return ([(n, p.requires_grad) for n, p in module.named_parameters()],
            [m.track_running_stats for m in module.modules() if isinstance(m, nn.BatchNorm1d)])


@pytest.mark.parametrize("train_bn", [False, True], ids=["frozen_bn", "train_bn"])
@pytest.mark.parametrize("bn_track_running_stats", [False, True], ids=["no_track", "track"])
def test_freeze_module_tree_matches_lightning(train_bn, bn_track_running_stats):
    """Identical to Lightning's ``BaseFinetuning.freeze`` followed by the scheduler's frozen-``BatchNorm`` handling
    (which re-enables running statistics on every ``BatchNorm`` layer when ``frozen_bn_track_running_stats``)."""
    ours, theirs = _bn_net(), _bn_net()
    freeze_module_tree(ours, train_bn=train_bn, bn_track_running_stats=bn_track_running_stats)
    BaseFinetuning.freeze(theirs, train_bn=train_bn)
    if bn_track_running_stats:
        for m in BaseFinetuning.flatten_modules(theirs):
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = True
    assert _state(ours) == _state(theirs)


def test_flatten_modules_matches_lightning():
    net = nn.ModuleDict({"a": _bn_net(), "b": nn.Linear(2, 2)})
    assert flatten_modules(net) == BaseFinetuning.flatten_modules(net)
    assert flatten_modules([net["a"], [net["b"]]]) == BaseFinetuning.flatten_modules([net["a"], [net["b"]]])


def test_map_param_groups_matches_lightning():
    net = nn.Linear(3, 2)
    opt = torch.optim.SGD([{"params": [net.weight], "lr": 0.1}, {"params": [net.bias], "lr": 0.01}], lr=0.1)
    to_names = {p: n for n, p in net.named_parameters()}
    recorded = map_param_groups(opt.param_groups, to_names)
    assert recorded == BaseFinetuning._apply_mapping_to_param_groups(opt.param_groups, to_names)
    assert [g["params"] for g in recorded] == [["weight"], ["bias"]]
    rebuilt = map_param_groups(recorded, dict(net.named_parameters()))
    assert [g["params"] for g in rebuilt] == [[net.weight], [net.bias]] and rebuilt[1]["lr"] == 0.01


SCHEDULE = {0: {"params": ["a"]}, 1: {"params": ["b"], "new_lr_scheduler": {"lr_scheduler_init": "x"}},
            2: {"params": ["c"], "new_optimizer": {"optimizer_init": "y"}}, 3: {"params": ["d"]}}


@pytest.mark.parametrize(
    "depth, best_ckpt_depth, depth_sync, reinit_optim_cfg, expected_cfg, expected_phases",
    [
        (1, 0, True, None, None, [1, 2, 3]),  # sync from the restored depth onward
        (1, 0, False, None, None, [1]),  # target depth only
        (2, 1, False, None, {"optimizer_init": "y"}, [0, 1, 2, 3]),  # a reinitialized optimizer revisits phase 0
        (0, 0, True, {"optimizer_init": "z"}, None, [1, 2, 3]),  # never reinitialize at depth 0
        (3, 2, False, {"optimizer_init": "z"}, {"optimizer_init": "z"}, [0, 1, 2, 3]),  # implicit config wins
    ],
)
def test_thaw_plan(depth, best_ckpt_depth, depth_sync, reinit_optim_cfg, expected_cfg, expected_phases):
    new_cfg, phases = thaw_plan(SCHEDULE, depth, best_ckpt_depth, depth_sync, reinit_optim_cfg)
    assert new_cfg == expected_cfg
    assert [d for d, _ in phases] == expected_phases
    assert all(cfg is SCHEDULE[d] for d, cfg in phases)


def test_lr_reinit_cfg_for_and_depth_remaining():
    assert lr_reinit_cfg_for(SCHEDULE[1], 1, None) == {"lr_scheduler_init": "x"}
    assert lr_reinit_cfg_for(SCHEDULE[1], 1, {"implicit": True}) == {"implicit": True}
    assert lr_reinit_cfg_for(SCHEDULE[1], 0, {"implicit": True}) is None
    assert lr_reinit_cfg_for(SCHEDULE[3], 3, None) is None
    assert depth_remaining(3, 1) == 2 and depth_remaining(1, 3) == 0


@pytest.mark.parametrize(
    "plateau_complete, final_phase, curr_max_epoch, current_epoch, expected",
    [
        (False, False, 3, 2, False),
        (False, False, 3, 3, True),  # epoch-driven
        (False, True, 3, 5, False),  # never epoch-driven in the final phase
        (True, True, -1, 0, True),  # plateau ends any phase
        (False, False, -1, 99, False),  # a negative max_transition_epoch disables epoch-driven transitions
    ],
)
def test_transition_composition(plateau_complete, final_phase, curr_max_epoch, current_epoch, expected):
    from finetuning_scheduler.core.transitions import compose_transition, current_max_epoch, epoch_transition_due

    due = epoch_transition_due(curr_max_epoch, current_epoch)
    assert compose_transition(plateau_complete, final_phase, due) is expected
    assert current_max_epoch(SCHEDULE | {1: {"params": ["b"], "max_transition_epoch": 4}}, 1, 2, 10) == 4
    assert current_max_epoch(SCHEDULE, 3, 0, 10) == 10
