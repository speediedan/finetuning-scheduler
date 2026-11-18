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
Fine-Tuning Scheduler Gradient Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transform the gradients a schedule addresses, beyond masks: a phase's ``grad_transforms`` key (PyTorch API) maps
``params``-style entries (names, regexes or addresses) to transforms applied to each step's accumulated gradient of
the tensors each entry selects.

A transform is a callable ``transform(name, grad, units)`` taking a parameter's name, its accumulated gradient and the
indices along its first axis that the entry's unit set keeps (``None`` for an entry without one), and returning the
transformed gradient, of the gradient's shape. It runs inside the gradient hook, so it should not keep state: a resumed
run rebuilds the transforms from the schedule. The built-ins below act on the ``units`` rows when given, else on the
whole gradient:

- :class:`Scale`: multiply by a factor (a per-component learning-rate multiplier, without splitting parameter groups);
- :class:`Clamp`: bound each element;
- :class:`Project` and :class:`Reject`: keep a gradient in, or move it out of, the span of directions along its first
  axis (for a weight, its output side), a ``fraction`` of the way.

:class:`Scale`, :class:`Project` and :class:`Reject` have the meaning venngram declares for the ``scale``, ``project``
and ``reject`` ops on a parameter (``reject`` with fraction ``a`` and an orthonormal span ``U`` gives
:math:`(I - aUU^\top)G`).

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor


def _rows(t: Tensor) -> Tensor:
    """``t`` indexed by its first axis (a 0-d tensor as one row)."""
    return t.view(1) if t.dim() == 0 else t


def _on_units(grad: Tensor, units: Sequence[int] | None, fn) -> Tensor:
    """``fn`` applied to ``grad``'s ``units`` rows (every row when ``units`` is ``None``), the rest unchanged."""
    if units is None:
        return fn(grad)
    out = grad.clone()
    index = torch.tensor(list(units), dtype=torch.long, device=grad.device)
    rows = _rows(out)
    rows.index_copy_(0, index, fn(rows.index_select(0, index)))
    return out


@dataclass(frozen=True)
class Scale:
    """Multiply the gradient (its ``units`` rows, when given) by ``factor``."""

    factor: float

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return _on_units(grad, units, lambda g: g * self.factor)


@dataclass(frozen=True)
class Clamp:
    """Bound each element of the gradient (its ``units`` rows, when given) to ``[min, max]`` (either may be
    ``None``)."""

    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError("`Clamp` needs a `min`, a `max` or both.")

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return _on_units(grad, units, lambda g: g.clamp(min=self.min, max=self.max))


@dataclass(frozen=True, eq=False)
class _Span:
    directions: Tensor
    fraction: float = 1.0
    _basis: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        d = torch.as_tensor(self.directions)
        if d.dim() not in (1, 2) or d.numel() == 0:
            raise ValueError(f"`{type(self).__name__}` takes a direction or a matrix of directions (one per column),"
                             f" not a tensor of shape {tuple(d.shape)}.")
        if self.fraction < 0:
            raise ValueError(f"`{type(self).__name__}`'s `fraction` must be >= 0, got {self.fraction}.")

    def basis(self, like: Tensor) -> Tensor:
        """An orthonormal basis of the directions' span (one column per direction), on ``like``'s device and in
        ``like``'s floating dtype."""
        key = (like.device, like.dtype)
        if key not in self._basis:
            d = torch.as_tensor(self.directions).to(device=like.device, dtype=torch.float64)
            q, _ = torch.linalg.qr(d.reshape(d.shape[0], -1))
            self._basis[key] = q.to(like.dtype)
        return self._basis[key]

    def in_span(self, name: str, grad: Tensor) -> Tensor:
        """The component of ``grad`` in the span, along its first axis."""
        g = _rows(grad).reshape(_rows(grad).shape[0], -1)
        if g.shape[0] != torch.as_tensor(self.directions).shape[0]:
            raise ValueError(f"`{type(self).__name__}` on {name!r}: its directions have"
                             f" {torch.as_tensor(self.directions).shape[0]} entries, but the gradient has"
                             f" {g.shape[0]} rows.")
        u = self.basis(g)
        return (u @ (u.T @ g)).reshape(grad.shape)


@dataclass(frozen=True, eq=False)
class Project(_Span):
    """Keep the gradient in the span of ``directions`` along its first axis, ``fraction`` of the way: with ``fraction``
    :math:`a` and the projection :math:`P` onto the span, :math:`G - a(G - PG)` (:math:`a = 1` keeps only the component
    in it).

    ``directions`` is a vector with one entry per row, or a matrix of such vectors, one per column. The gradient's
    ``units`` are not used: a projection acts on the whole tensor.
    """

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return grad - self.fraction * (grad - self.in_span(name, grad))


@dataclass(frozen=True, eq=False)
class Reject(_Span):
    """Remove ``fraction`` of the gradient's component in the span of ``directions`` along its first axis: with
    ``fraction`` :math:`a` and the projection :math:`P` onto the span, :math:`G - aPG` (:math:`a = 1` trains
    orthogonally to the span; above 1 over-rejects).

    ``directions`` is a vector with one entry per row, or a matrix of such vectors, one per column. The gradient's
    ``units`` are not used: a rejection acts on the whole tensor.
    """

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return grad - self.fraction * self.in_span(name, grad)


def transform_id(transform: object) -> str:
    """A stable description of ``transform`` for the compiled schedule's digest: a built-in's parameters, or a
    callable's qualified name."""
    if isinstance(transform, (Scale, Clamp)):
        return repr(transform)
    if isinstance(transform, _Span):
        d = torch.as_tensor(transform.directions)
        return (f"{type(transform).__name__}(directions={tuple(d.shape)}:{d.double().sum().item():.12g},"
                f" fraction={transform.fraction})")
    target = getattr(transform, "__func__", transform)
    qualname = getattr(target, "__qualname__", None) or type(transform).__qualname__
    module = getattr(target, "__module__", None) or type(transform).__module__
    return f"{module}.{qualname}"


__all__ = ["Clamp", "Project", "Reject", "Scale", "transform_id"]
