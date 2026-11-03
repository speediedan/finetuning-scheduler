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
"""Small score and fixed-delta modules shared by the neuron-mask tutorials.

The budget-constrained sigmoid follows the published Matryoshka Attribution method. This is an independent
implementation of the method, with a row-indexed linear wrapper for the examples. FTS schedules the score tensors; row
selection is performed by this module. Reference:
https://arxiv.org/abs/2609.25518
.
"""
import math
from collections.abc import Sequence

import torch
from torch import nn, Tensor
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


def _validate_budget(budget: float, size: int, temperature: float) -> None:
    if not math.isfinite(budget) or not 0 <= budget <= size:
        raise ValueError(f"budget must be finite and between 0 and {size}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


class _BudgetSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, budget, temperature):
        # Solve the monotone threshold constraint without differentiating the search trajectory.
        values = scores if scores.dtype == torch.float64 else scores.float()
        if budget == 0 or budget == scores.numel():
            mask = torch.full_like(values, float(budget != 0))
        else:
            tail = temperature * (math.log(scores.numel()) + 40)
            lower, upper = values.min() - tail, values.max() + tail
            for _ in range(80):
                threshold = (lower + upper) / 2
                mass = torch.sigmoid((values - threshold) / temperature).sum()
                lower = torch.where(mass > budget, threshold, lower)
                upper = torch.where(mass > budget, upper, threshold)
            mask = torch.sigmoid((values - (lower + upper) / 2) / temperature)
        ctx.save_for_backward(mask)
        ctx.temperature = temperature
        ctx.input_dtype = scores.dtype
        return mask.to(scores.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        sensitivity = mask * (1 - mask)
        total = sensitivity.sum()
        safe_total = total.clamp_min(torch.finfo(sensitivity.dtype).tiny)
        centered = grad_output - (grad_output * sensitivity).sum() / safe_total
        gradient = sensitivity * centered / ctx.temperature
        return gradient.to(ctx.input_dtype), None, None


def sigmoid_budget(scores: Tensor, budget: float, temperature: float = 1.0) -> Tensor:
    """Return a differentiable sigmoid mask with the requested total mass.

    A shared threshold enforces the budget; backward differentiates that constraint, including the
    coupling between scores. Zero/full budgets have a constant mask and zero gradient. Only first
    derivatives are supported. Hard evaluation masks are provided separately by ``MaskedDeltaLinear``.
    """
    if scores.ndim != 1 or not scores.numel() or not scores.is_floating_point():
        raise ValueError("scores must be a nonempty floating-point vector")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    _validate_budget(budget, scores.numel(), temperature)
    return _BudgetSigmoid.apply(scores, float(budget), float(temperature))


class MaskedDeltaLinear(nn.Module):
    """A frozen linear projection plus fixed delta rows gated by trainable scores.

    Only ``mask_scores`` is trainable. ``row_indices`` identifies eligible output neurons; all other
    output coordinates receive exactly zero delta. Base weights, selected delta rows, eligibility,
    budget, temperature and hard/soft evaluation mode travel in ``state_dict``. Construct the same
    shapes before loading a checkpoint. A schedule that thaws the base is rejected on forward.

    This wraps an ordinary ``nn.Linear``; quantized/fused projections and column masks are outside
    this first tutorial primitive. Fixed-delta attribution is distinct from training new base weights.
    """

    def __init__(self, base: nn.Linear, weight_delta: Tensor, *, budget: float,
                 row_indices: Sequence[int] | Tensor | None = None, temperature: float = 1.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        if weight_delta.shape != base.weight.shape or not weight_delta.is_floating_point():
            raise ValueError("weight_delta must have the base weight's shape and a floating-point dtype")
        if not torch.isfinite(weight_delta).all():
            raise ValueError("weight_delta must be finite")
        rows = torch.arange(base.out_features, device=base.weight.device) if row_indices is None else torch.as_tensor(
            row_indices, device=base.weight.device)
        if rows.ndim != 1 or not rows.numel() or rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("row_indices must be a nonempty integer vector")
        if rows.unique().numel() != rows.numel() or rows.min() < 0 or rows.max() >= base.out_features:
            raise ValueError("row_indices must be unique and within the projection output")
        _validate_budget(budget, rows.numel(), temperature)
        self.base = base.requires_grad_(False)
        dtype = torch.float64 if base.weight.dtype == torch.float64 else torch.float32
        self.register_buffer("row_indices", rows.to(torch.long).clone())
        self.register_buffer("weight_delta", weight_delta.detach().to(device=base.weight.device, dtype=dtype)
                             .index_select(0, self.row_indices).clone())
        self.mask_scores = nn.Parameter(torch.zeros(rows.numel(), device=base.weight.device, dtype=dtype))
        self.register_buffer("mask_budget", torch.tensor(float(budget), device=base.weight.device, dtype=dtype))
        self.register_buffer("temperature", torch.tensor(float(temperature), device=base.weight.device, dtype=dtype))
        self.register_buffer("hard", torch.tensor(False, device=base.weight.device))

    def set_budget(self, budget: float, *, hard: bool = False) -> None:
        """Change the mask budget; hard evaluation requires an integer count of eligible rows."""
        _validate_budget(budget, self.mask_scores.numel(), float(self.temperature))
        if hard and not float(budget).is_integer():
            raise ValueError("hard evaluation requires an integer budget")
        self.mask_budget.fill_(budget)
        self.hard.fill_(hard)

    def mask(self) -> Tensor:
        """The soft training mask or deterministic nested hard mask, with ties resolved by row-list order."""
        if not bool(self.hard):
            return sigmoid_budget(self.mask_scores, float(self.mask_budget), float(self.temperature))
        if not torch.isfinite(self.mask_scores).all():
            raise ValueError("scores must be finite")
        count = float(self.mask_budget)
        _validate_budget(count, self.mask_scores.numel(), float(self.temperature))
        if not count.is_integer():
            raise ValueError("hard evaluation requires an integer budget")
        order = torch.argsort(self.mask_scores, descending=True, stable=True)
        return torch.zeros_like(self.mask_scores).scatter(0, order[:int(count)], 1)

    def forward(self, x: Tensor) -> Tensor:
        if any(p.requires_grad for p in self.base.parameters()):
            raise RuntimeError("MaskedDeltaLinear base weights must stay frozen; schedule mask_scores only")
        output = self.base(x)
        selected = F.linear(x.to(self.weight_delta.dtype), self.weight_delta) * self.mask()
        update = torch.zeros_like(output).index_copy(-1, self.row_indices, selected.to(output.dtype))
        return output + update
