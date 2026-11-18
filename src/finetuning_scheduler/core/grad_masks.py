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
r"""
Fine-Tuning Scheduler Gradient Masks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train part of a tensor: the indices along its first axis (a weight's output rows, a vector's entries) that a schedule's
gradient addresses (``of: grad``) keep.

:class:`GradientMasks` holds the masks in effect on a module's parameters. Each masked tensor gets a hook registered
with ``register_post_accumulate_grad_hook`` that zeroes the gradient of every index the mask does not keep once the
gradient has accumulated, so the mask holds under gradient accumulation and acts before gradient clipping and the
optimizer step: the clipped norm and the optimizer's statistics see only the kept rows.

A zero gradient does not keep every optimizer from moving a row. Decoupled weight decay (``AdamW``) shrinks it, coupled
weight decay adds to its gradient inside the step, and other optimizers may update a row from state alone. So the masked
rows are also copied before each optimizer step and written back after it, which holds them for any optimizer. The cost
is a copy of the masked rows per step. The optimizer's state for a masked row is what it derives from a zero gradient:
momentum and adaptive moments stay zero, unless coupled weight decay adds to the gradient inside the step, so a mask a
later phase widens starts its newly kept rows from that state.

The same hook applies a tensor's gradient transforms (a phase's ``grad_transforms``,
:mod:`~finetuning_scheduler.core.grad_transforms`), in order, before its mask, so no transform refills a masked row.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle


def _rows(t: Tensor) -> Tensor:
    """``t`` indexed by its first axis (a 0-d tensor as one row)."""
    return t.view(1) if t.dim() == 0 else t


#: A gradient transform with the units its entry keeps (see :mod:`~finetuning_scheduler.core.grad_transforms`).
Transform = tuple[Callable[[str, Tensor, Sequence[int] | None], Tensor], tuple[int, ...] | None]


def _on_grad(name: str, transforms: tuple[Transform, ...], masked: Tensor, param: Tensor) -> None:
    """Apply ``transforms`` to ``param``'s accumulated gradient, in order, then zero its ``masked`` rows."""
    if param.grad is None:
        return
    if transforms:
        grad = param.grad
        for fn, units in transforms:
            out = fn(name, grad, units)
            if not isinstance(out, Tensor) or out.shape != grad.shape:
                raise RuntimeError(f"The gradient transform {fn!r} on {name!r} must return a tensor of the gradient's"
                                   f" shape {tuple(grad.shape)}, got {getattr(out, 'shape', type(out).__name__)}.")
            grad = out
        if grad is not param.grad:
            param.grad.copy_(grad)
    if masked.numel():
        _rows(param.grad).index_fill_(0, masked.to(param.grad.device), 0)


@dataclass
class _Mask:
    param: nn.Parameter
    kept: tuple[int, ...] | None  # None: no mask, transforms only
    masked: Tensor  # the indices the mask does not keep
    transforms: tuple[Transform, ...]
    handle: RemovableHandle


class GradientMasks:
    """The gradient masks and transforms in effect on ``module``'s parameters (see the module docstring).

    Args:
        module: The module whose parameters are masked or transformed, by name.
    """

    def __init__(self, module: nn.Module) -> None:
        self._module = module
        self._live: dict[str, _Mask] = {}
        self._step_hooks: list[RemovableHandle] = []
        self._saved: list[tuple[_Mask, Tensor]] = []

    @property
    def masks(self) -> dict[str, tuple[int, ...]]:
        """Each masked parameter's kept first-axis indices."""
        return {n: m.kept for n, m in self._live.items() if m.kept is not None}

    @property
    def transforms(self) -> dict[str, tuple[Transform, ...]]:
        """Each transformed parameter's gradient transforms, in the order they apply, with their units."""
        return {n: m.transforms for n, m in self._live.items() if m.transforms}

    def sync(self, masks: Mapping[str, tuple[int, ...]],
             transforms: Mapping[str, tuple[Transform, ...]] | None = None) -> None:
        """Make ``masks`` and ``transforms`` the ones in effect: remove the hooks of parameters no longer masked or
        transformed (refrozen, or trained whole), or masked or transformed differently, and install a hook on each
        parameter newly masked or transformed. A parameter replaced since its hook was installed (by a promotion
        that materializes it, for example) gets a new one.

        Each masked or transformed parameter must require grad when it is masked or transformed.
        """
        transforms = transforms or {}
        params = dict(self._module.named_parameters())
        for name in list(self._live):
            m = self._live[name]
            if (masks.get(name) != m.kept or tuple(transforms.get(name, ())) != m.transforms
                    or params.get(name) is not m.param):
                m.handle.remove()
                del self._live[name]
        for name in dict.fromkeys([*masks, *transforms]):
            if name in self._live:
                continue
            param, kept, fns = params[name], masks.get(name), tuple(transforms.get(name, ()))
            masked = sorted(set(range(_rows(param).shape[0])) - set(kept)) if kept is not None else []
            index = torch.tensor(masked, dtype=torch.long, device=param.device)
            handle = param.register_post_accumulate_grad_hook(partial(_on_grad, name, fns, index))
            self._live[name] = _Mask(param, None if kept is None else tuple(kept), index, fns, handle)

    def attach(self, optimizer: Optimizer) -> None:
        """Hold the masked rows through ``optimizer``'s steps, in place of the optimizer attached before."""
        for handle in self._step_hooks:
            handle.remove()
        self._step_hooks = [optimizer.register_step_pre_hook(self._save),
                            optimizer.register_step_post_hook(self._restore)]

    def remove(self) -> None:
        """Remove every hook: the masks and transforms no longer apply."""
        self.sync({})
        for handle in self._step_hooks:
            handle.remove()
        self._step_hooks = []

    def _save(self, optimizer: Optimizer, args: Any, kwargs: Any) -> None:
        with torch.no_grad():
            self._saved = [(m, _rows(m.param.detach()).index_select(0, m.masked.to(m.param.device)))
                           for m in self._live.values() if m.masked.numel()]

    def _restore(self, optimizer: Optimizer, args: Any, kwargs: Any) -> None:
        with torch.no_grad():
            for m, rows in self._saved:
                _rows(m.param.detach()).index_copy_(0, m.masked.to(m.param.device), rows)
        self._saved = []


__all__ = ["GradientMasks"]
