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
"""Numerical and optimizer/checkpoint invariants for the shared tutorial mask modules."""
import copy
from functools import partial

import pytest
import torch
from torch import nn

from fts_examples.tunable_views.masks import MaskedDeltaLinear, sigmoid_budget


@pytest.mark.parametrize("budget", [0.0, 0.001, 1.0, 2.5, 4.999, 5.0])
def test_budget_mass_and_implicit_gradient(budget):
    scores = torch.tensor([-3., -0.5, 0., 1., 4.], dtype=torch.float64, requires_grad=True)
    mask = sigmoid_budget(scores, budget, 0.7)
    torch.testing.assert_close(mask.sum(), torch.tensor(budget, dtype=torch.float64), atol=1e-10, rtol=1e-10)
    assert torch.autograd.gradcheck(lambda s: sigmoid_budget(s, budget, 0.7), (scores,))
    torch.testing.assert_close(sigmoid_budget(scores + 100, budget, 0.7), mask)


@pytest.mark.parametrize("scores,budget,temperature", [
    (torch.tensor([]), 0, 1), (torch.zeros(2, 2), 1, 1), (torch.tensor([1, 2]), 1, 1),
    (torch.tensor([float("nan")]), 1, 1), (torch.zeros(2), -1, 1),
    (torch.zeros(2), 3, 1), (torch.zeros(2), 1, 0), (torch.zeros(2), 1, float("inf")),
])
def test_invalid_mask_inputs(scores, budget, temperature):
    with pytest.raises(ValueError):
        sigmoid_budget(scores, budget, temperature)


def make_mask():
    torch.manual_seed(17)
    return MaskedDeltaLinear(nn.Linear(3, 5, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64),
                             row_indices=[4, 1, 3], budget=1.5, temperature=0.7)


def test_forward_matches_dense_masked_delta_for_batched_inputs():
    layer = make_mask()
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    weight = layer.base.weight.detach().clone()
    weight[layer.row_indices] += layer.weight_delta * layer.mask()[:, None]
    expected = torch.nn.functional.linear(x, weight, layer.base.bias)
    torch.testing.assert_close(layer(x), expected)
    assert [(n, p.requires_grad) for n, p in layer.named_parameters()] == [
        ("mask_scores", True), ("base.weight", False), ("base.bias", False)]


def test_hard_masks_are_nested_and_ties_are_deterministic():
    layer = make_mask()
    previous = set()
    for budget in range(4):
        layer.set_budget(budget, hard=True)
        chosen = set(layer.row_indices[layer.mask().bool()].tolist())
        assert len(chosen) == budget and previous <= chosen
        assert chosen == set([4, 1, 3][:budget])
        previous = chosen
    with pytest.raises(ValueError, match="integer"):
        layer.set_budget(1.5, hard=True)


def test_adamw_and_resume_preserve_frozen_and_excluded_coordinates():
    layer = make_mask()
    x = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 5, dtype=torch.float64)
    frozen = layer.base.weight.detach().clone()
    fixed_delta = layer.weight_delta.clone()
    optimizer = torch.optim.AdamW([layer.mask_scores], lr=0.1, weight_decay=0.3)

    def step(module, optim):
        optim.zero_grad()
        (module(x) - target).square().mean().backward()
        optim.step()

    step(layer, optimizer)
    layer.set_budget(2)
    state = copy.deepcopy(layer.state_dict())
    optim_state = copy.deepcopy(optimizer.state_dict())
    resumed = make_mask()
    resumed.set_budget(0, hard=True)
    resumed.load_state_dict(state)
    resumed_optim = torch.optim.AdamW([resumed.mask_scores], lr=9)
    resumed_optim.load_state_dict(optim_state)
    step(layer, optimizer)
    step(resumed, resumed_optim)
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(layer.base.weight, frozen, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight_delta, fixed_delta, rtol=0, atol=0)
    torch.testing.assert_close(layer(x)[..., [0, 2]], layer.base(x)[..., [0, 2]], rtol=0, atol=0)
    assert not torch.equal(layer.mask_scores, state["mask_scores"])


def test_a_schedule_cannot_train_the_fixed_base():
    layer = make_mask()
    layer.base.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="schedule mask_scores only"):
        layer(torch.ones(3, dtype=torch.float64))


def test_scores_follow_an_fts_name_based_two_phase_schedule(tmp_path):
    from finetuning_scheduler import TuningScheduler

    model = nn.Sequential(make_mask(), nn.Linear(5, 5, dtype=torch.float64).requires_grad_(False))
    schedule = {
        0: {"params": ["0.mask_scores"], "max_transition_epoch": 1},
        1: {"params": ["1.weight", "1.bias"], "lr": 0.1},
    }
    fts = TuningScheduler(model, schedule, optimizer=partial(torch.optim.AdamW, lr=0.1),
                          restore_best=False, log_dir=tmp_path)
    fts.setup()
    for epoch in fts.epochs(2):
        assert model[1].weight.requires_grad == bool(epoch)
        fts.optimizer.zero_grad()
        model(torch.randn(2, 3, dtype=torch.float64)).square().mean().backward()
        fts.optimizer.step()
        fts.epoch_end()
    assert fts.curr_depth == 1 and not model[0].base.weight.requires_grad
