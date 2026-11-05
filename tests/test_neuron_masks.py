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

from fts_examples.tunable_views.masks import GlobalMaskBudget, MaskedDeltaLinear, sigmoid_budget


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_frozen_factor_rows_preserve_donor_endpoint_and_resume(dtype):
    from fts_examples.tunable_views.lora import LoRALinear

    torch.manual_seed(19)
    donor = LoRALinear(nn.Linear(32, 48, dtype=dtype), r=8, alpha=8 * 8**0.5)
    nn.init.normal_(donor.lora_B.weight, std=0.2)
    x = torch.randn(2, 7, 32, dtype=dtype)
    layer = MaskedDeltaLinear(copy.deepcopy(donor.base), budget=48,
                             factors=(donor.lora_A.weight, donor.lora_B.weight, donor.scale))
    layer.set_budget(48, hard=True)
    torch.testing.assert_close(layer(x), donor(x), atol=0, rtol=0)
    layer.set_budget(0, hard=True)
    torch.testing.assert_close(layer(x), donor.base(x), atol=0, rtol=0)
    assert layer.weight_delta is None
    fixed = {n: t.clone() for n, t in layer.named_buffers() if n.startswith("factor_")}
    layer.set_budget(24)
    optim = torch.optim.AdamW([layer.mask_scores], lr=0.1)
    layer(x).float().square().mean().backward()
    assert layer.mask_scores.grad.abs().sum() > 0
    optim.step()
    saved = copy.deepcopy(layer.state_dict())
    clone = copy.deepcopy(layer)
    with torch.no_grad():
        clone.mask_scores.zero_()
    clone.load_state_dict(saved)
    torch.testing.assert_close(clone(x), layer(x), atol=0, rtol=0)
    for name, tensor in fixed.items():
        torch.testing.assert_close(getattr(layer, name), tensor, atol=0, rtol=0)
    assert not any(p.requires_grad for p in layer.base.parameters())
    selected = MaskedDeltaLinear(copy.deepcopy(donor.base), budget=3, row_indices=[47, 2, 9],
                                factors=(donor.lora_A.weight, donor.lora_B.weight, donor.scale))
    selected.set_budget(3, hard=True)
    actual = selected(x)
    torch.testing.assert_close(actual[..., [47, 2, 9]], donor(x)[..., [47, 2, 9]], atol=0, rtol=0)
    torch.testing.assert_close(actual[..., [0, 3]], donor.base(x)[..., [0, 3]], atol=0, rtol=0)
    # Buffers own their storage, so later changes to the source donor cannot change attribution.
    with torch.no_grad():
        donor.lora_A.weight.zero_()
    torch.testing.assert_close(layer.factor_a, fixed["factor_a"], atol=0, rtol=0)


def test_fixed_delta_representation_is_unambiguous_and_validated():
    base = nn.Linear(3, 5)
    factors = (torch.ones(2, 3), torch.ones(5, 2), 1.)
    for kwargs in ({}, {"weight_delta": torch.zeros(5, 3), "factors": factors},
                   {"factors": (torch.ones(2, 4), torch.ones(5, 2), 1.)},
                   {"factors": (torch.ones(2, 3), torch.ones(5, 2), float("nan"))}):
        with pytest.raises(ValueError):
            MaskedDeltaLinear(base, budget=2, **kwargs)


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


def make_global_mask():
    model = nn.Sequential(make_mask(), make_mask())
    model.add_module("budget", GlobalMaskBudget({"first": model[0], "second": model[1]},
                                              budget=2.5, temperature=0.7))
    return model


def test_global_budget_couples_projections_and_nested_hard_selection():
    model = make_global_mask()
    with torch.no_grad():
        model[0].mask_scores.copy_(torch.tensor([0., 3., 1.]))
        model[1].mask_scores.copy_(torch.tensor([5., 4., 2.]))
    masks = model.budget.masks()
    torch.testing.assert_close(sum(mask.sum() for mask in masks.values()), torch.tensor(2.5, dtype=torch.float64))
    # A loss on one projection also changes the other projection's scores through the shared threshold.
    masks["first"][0].backward()
    assert model[1].mask_scores.grad.abs().sum() > 0
    previous = set()
    for count in range(7):
        model.budget.set_budget(count, hard=True)
        chosen = {(path, int(row)) for path, mask in model.budget.masks().items()
                  for row in model.budget._mask_layers[path].row_indices[mask.bool()]}
        assert len(chosen) == count and previous <= chosen
        if count == 2:
            assert chosen == {("second", 4), ("second", 1)}
        previous = chosen
    with pytest.raises(RuntimeError, match="GlobalMaskBudget"):
        model[0].set_budget(1)
    with torch.no_grad():
        model[0].mask_scores.zero_()
        model[1].mask_scores.zero_()
    model.budget.set_budget(4, hard=True)
    assert model.budget.masks()["first"].tolist() == [1, 1, 1]
    assert model.budget.masks()["second"].tolist() == [1, 0, 0]


def test_global_forward_cache_gradient_and_cleanup():
    model = make_global_mask()
    x = torch.randn(2, 3, dtype=torch.float64)

    def loss():
        return model[0](x).square().sum() + model[1](x).square().sum()

    direct = loss()
    direct_grads = torch.autograd.grad(direct, [model[0].mask_scores, model[1].mask_scores])
    with model.budget.masked_forward():
        cached = model.budget.masks()
        assert model[0].mask() is cached["first"]
        value = loss()
        with pytest.raises(RuntimeError, match="nested"):
            with model.budget.masked_forward():
                pass
        with pytest.raises(RuntimeError, match="inside"):
            model.budget.set_budget(1)
    assert model.budget._cached_masks is None
    actual = torch.autograd.grad(value, [model[0].mask_scores, model[1].mask_scores])
    for result, expected in zip(actual, direct_grads):
        torch.testing.assert_close(result, expected)
    with pytest.raises(ValueError, match="forward failed"):
        with model.budget.masked_forward():
            raise ValueError("forward failed")
    assert model.budget._cached_masks is None


def test_global_checkpoint_resume_and_copy_preserve_bindings_and_immutable_weights():
    model = make_global_mask()
    # The controller must not register duplicate copies of projections or their scores.
    assert list(dict(model.named_parameters())) == ["0.mask_scores", "0.base.weight", "0.base.bias",
                                                   "1.mask_scores", "1.base.weight", "1.base.bias"]
    params = [model[0].mask_scores, model[1].mask_scores]
    optimizer = torch.optim.AdamW(params, lr=0.1, weight_decay=0.3)
    x = torch.randn(2, 3, dtype=torch.float64)
    frozen = {name: value.clone() for name, value in model.state_dict().items()
              if isinstance(value, torch.Tensor) and ("base." in name or "weight_delta" in name)}

    def step(module, optim):
        optim.zero_grad()
        with module.budget.masked_forward():
            (module[0](x) - module[1](x).detach()).square().sum().backward()
        optim.step()

    step(model, optimizer)
    model.budget.set_budget(1.7)
    resumed = make_global_mask()
    resumed.load_state_dict(copy.deepcopy(model.state_dict()))
    resumed_optim = torch.optim.AdamW([resumed[0].mask_scores, resumed[1].mask_scores], lr=9)
    resumed_optim.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(resumed, resumed_optim)
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
    for name, value in frozen.items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
    clone = copy.deepcopy(model)
    assert clone[0]._global_budget is clone.budget
    assert clone.budget._mask_layers["first"] is clone[0]
    assert clone[0]._global_budget is not model.budget
    state = copy.deepcopy(model.state_dict())
    state["budget._extra_state"].reverse()
    with pytest.raises(ValueError, match="bindings differ"):
        resumed.load_state_dict(state)


def test_global_invalid_bindings_and_budget_do_not_bind_partially():
    a, b = make_mask(), make_mask().float()
    for layers, budget in [({}, 0), ({"a": a, "b": a}, 1), ({"a": a, "b": b}, 1), ({"a": a}, 4)]:
        with pytest.raises(ValueError):
            GlobalMaskBudget(layers, budget=budget)
        assert getattr(a, "_global_budget", None) is None
    controller = GlobalMaskBudget({"a": a}, budget=1)
    with pytest.raises(ValueError, match="already bound"):
        GlobalMaskBudget({"a": a}, budget=1)
    with pytest.raises(ValueError, match="integer"):
        controller.set_budget(0.5, hard=True)
