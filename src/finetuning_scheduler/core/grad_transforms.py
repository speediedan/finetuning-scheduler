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
the tensors each entry selects, or names a treatment of the schedule's address books whose ``gradients`` ops it
applies (see below).

A transform is a callable ``transform(name, grad, units)`` taking a parameter's name, its accumulated gradient and the
indices along its first axis that the entry's unit set keeps (``None`` for an entry without one), and returning the
transformed gradient, of the gradient's shape. It runs inside the gradient hook. A resumed run rebuilds the transforms
from the schedule, so a transform that keeps state (a random generator, running statistics) is *stateful*: it defines
``state_dict()`` and ``load_state_dict(state)``, and the tuning scheduler saves the state of each stateful transform in
effect with its own (:meth:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.state_dict`) and loads it into
the rebuilt one on resume, so a resumed run matches an uninterrupted one. A transform with state it does not expose this
way starts afresh on resume. The built-ins below act on the ``units`` slices along their ``dim`` (the first axis by
default) when given, else on the whole gradient:

- :class:`Scale`: multiply by a factor (a per-component learning-rate multiplier, without splitting parameter groups);
- :class:`Replace` and :class:`Add`: set each element to a value, or add one to it;
- :class:`Clamp`: bound each element;
- :class:`Noise`: add Gaussian noise, drawn from a generator of its own (stateful: its generator follows the
  checkpoints);
- :class:`Project` and :class:`Reject`: keep a gradient in, or move it out of, the span of directions along one axis
  (the first by default: for a ``torch.nn.Linear`` weight, its output side), a ``fraction`` of the way.

:class:`Scale`, :class:`Replace`, :class:`Add`, :class:`Project` and :class:`Reject` have the meaning venngram declares
for the ``scale``, ``replace``, ``add``, ``project`` and ``reject`` ops on a parameter (``reject`` with fraction ``a``
and an orthonormal span ``U`` gives :math:`(I - aUU^\top)G`).

**Treatments.** A phase's ``grad_transforms`` may instead name a treatment of the schedule's address books (a
venngram treatment written in the book it treats, :mod:`venngram.treatment`), so a schedule read from YAML can transform
gradients too. Its ``gradients`` section's ops apply in order, each to the gradients of the tensors its target set
selects, as these built-ins (:data:`GRADIENT_OPS`): a ``dose`` or per-unit ``doses`` for ``scale``, ``replace`` and
``add``, and a ``vector`` or ``subspace`` operand for ``project`` and ``reject``. A ``param_units`` target acts along
each tensor's unit axis, which the view's component map records
(:meth:`venngram.maps.structure.MapStructure.unit_dim`), and a span's directions run along a weight's ``out`` side,
which it records too (the first axis without one). A treatment with ``stages`` or ``weights``, an op Fine-Tuning
Scheduler does not apply, or a value taken ``from`` evidence or through a value transform is refused, as its support
record declares.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from finetuning_scheduler.core.grad_masks import Transform

#: The venngram ops Fine-Tuning Scheduler applies to parameter gradients (a treatment's ``gradients`` section), as its
#: support record's ``gradient_ops`` declares.
GRADIENT_OPS = ("add", "project", "reject", "replace", "scale")


def _rows(t: Tensor) -> Tensor:
    """``t`` indexed by its first axis (a 0-d tensor as one row)."""
    return t.view(1) if t.dim() == 0 else t


def _along(t: Tensor, dim: int, name: str, who: str) -> Tensor:
    """``t`` (a 0-d tensor as one entry) with its ``dim`` axis first, as a view."""
    rows = _rows(t)
    if not 0 <= dim < rows.dim():
        raise ValueError(f"`{who}` on {name!r}: dimension {dim} is not an axis of a gradient of shape"
                         f" {tuple(t.shape)}.")
    return rows.movedim(dim, 0)


def _on_units(grad: Tensor, units: Sequence[int] | None, fn, dim: int = 0, name: str = "", who: str = "") -> Tensor:
    """``fn`` applied to ``grad``'s ``units`` slices along ``dim`` (all of ``grad`` when ``units`` is ``None``),
    the rest unchanged."""
    if units is None:
        return fn(grad)
    out = grad.clone()
    index = torch.tensor(list(units), dtype=torch.long, device=grad.device)
    rows = _along(out, dim, name, who)
    rows.index_copy_(0, index, fn(rows.index_select(0, index)))
    return out


@dataclass(frozen=True)
class Scale:
    """Multiply the gradient (its ``units`` slices along ``dim``, when given) by ``factor``."""

    factor: float
    dim: int = 0

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return _on_units(grad, units, lambda g: g * self.factor, self.dim, name, "Scale")


@dataclass(frozen=True)
class Replace:
    """Set each element of the gradient (its ``units`` slices along ``dim``, when given) to ``value``: zero keeps
    them out of a step (venngram's declared form of a gradient mask)."""

    value: float
    dim: int = 0

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return _on_units(grad, units, lambda g: torch.full_like(g, self.value), self.dim, name, "Replace")


@dataclass(frozen=True)
class Add:
    """Add ``value`` to each element of the gradient (its ``units`` slices along ``dim``, when given)."""

    value: float
    dim: int = 0

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return _on_units(grad, units, lambda g: g + self.value, self.dim, name, "Add")


@dataclass(frozen=True)
class Clamp:
    """Bound each element of the gradient (its ``units`` slices along ``dim``, when given) to ``[min, max]``
    (either may be ``None``)."""

    min: float | None = None
    max: float | None = None
    dim: int = 0

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError("`Clamp` needs a `min`, a `max` or both.")

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return _on_units(grad, units, lambda g: g.clamp(min=self.min, max=self.max), self.dim, name, "Clamp")


@dataclass(frozen=True, eq=False)
class Noise:
    """Add Gaussian noise of standard deviation ``std`` to each element of the gradient (its ``units`` slices along
    ``dim``, when given), drawn on the CPU from a generator of its own, seeded with ``seed``, so the draws do not
    depend on the device or on the global random state.

    It is stateful: its generator's state is saved with the tuning scheduler's and restored on resume, so a resumed run
    draws what an uninterrupted one would. One instance used for several tensors draws for each in the order their
    gradients accumulate.
    """

    std: float
    seed: int = 0
    dim: int = 0
    _generator: torch.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError(f"`Noise`'s `std` must be >= 0, got {self.std}.")
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        object.__setattr__(self, "_generator", generator)

    def _noisy(self, g: Tensor) -> Tensor:
        return g + self.std * torch.randn(g.shape, generator=self._generator).to(device=g.device, dtype=g.dtype)

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return _on_units(grad, units, self._noisy, self.dim, name, "Noise")

    def state_dict(self) -> dict[str, Tensor]:
        """The generator's state."""
        return {"generator": self._generator.get_state()}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore the generator's state saved by :meth:`state_dict`."""
        self._generator.set_state(state["generator"])


def is_stateful(transform: object) -> bool:
    """Whether ``transform`` keeps state it exposes for checkpoints (``state_dict`` and ``load_state_dict``)."""
    return callable(getattr(transform, "state_dict", None)) and callable(getattr(transform, "load_state_dict", None))


@dataclass(frozen=True, eq=False)
class _Span:
    directions: Tensor
    fraction: float = 1.0
    dim: int = 0
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
        """The component of ``grad`` in the span, along its ``dim`` axis."""
        moved = _along(grad, self.dim, name, type(self).__name__)
        g = moved.reshape(moved.shape[0], -1)
        n = torch.as_tensor(self.directions).shape[0]
        if g.shape[0] != n:
            along = "rows" if self.dim == 0 else f"entries along dimension {self.dim}"
            raise ValueError(f"`{type(self).__name__}` on {name!r}: its directions have {n} entries, but the gradient"
                             f" has {g.shape[0]} {along}.")
        u = self.basis(g)
        return (u @ (u.T @ g)).reshape(moved.shape).movedim(0, self.dim).reshape(grad.shape)


@dataclass(frozen=True, eq=False)
class Project(_Span):
    """Keep the gradient in the span of ``directions`` along its ``dim`` axis, ``fraction`` of the way: with
    ``fraction`` :math:`a` and the projection :math:`P` onto the span, :math:`G - a(G - PG)` (:math:`a = 1` keeps only
    the component in it).

    ``directions`` is a vector with one entry per index along ``dim``, or a matrix of such vectors, one per column. The
    gradient's ``units`` are not used: a projection acts on the whole tensor.
    """

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return grad - self.fraction * (grad - self.in_span(name, grad))


@dataclass(frozen=True, eq=False)
class Reject(_Span):
    """Remove ``fraction`` of the gradient's component in the span of ``directions`` along its ``dim`` axis: with
    ``fraction`` :math:`a` and the projection :math:`P` onto the span, :math:`G - aPG` (:math:`a = 1` trains
    orthogonally to the span; above 1 over-rejects).

    ``directions`` is a vector with one entry per index along ``dim``, or a matrix of such vectors, one per column. The
    gradient's ``units`` are not used: a rejection acts on the whole tensor.
    """

    def __call__(self, name: str, grad: Tensor, units: Sequence[int] | None) -> Tensor:
        return grad - self.fraction * self.in_span(name, grad)


_VALUED = {"scale": Scale, "replace": Replace, "add": Add}
_SPANS = {"project": Project, "reject": Reject}


def op_transforms(op: str, *, dim: int = 0, dose: float | None = None, doses: Mapping[int, float] | None = None,
                  directions: Tensor | None = None, fraction: float = 1.0) -> list[Transform]:
    """The built-ins applying venngram's op ``op`` (one of :data:`GRADIENT_OPS`) to one tensor's gradient, each
    with the units it acts on (``None``: the whole tensor).

    Args:
        op: The op's name.
        dim: The axis the op's units (``doses``) or ``directions`` run along.
        dose: A ``scale``, ``replace`` or ``add`` op's value for the whole tensor.
        doses: Its value per unit along ``dim`` instead: one built-in per distinct value, on the units that take it.
        directions: A ``project`` or ``reject`` op's directions, one per column.
        fraction: Its ``fraction``.
    """
    if op in _SPANS:
        return [(_SPANS[op](directions, fraction=fraction, dim=dim), None)]
    if doses is None:
        return [(_VALUED[op](dose, dim=dim), None)]
    by_value: dict[float, list[int]] = {}
    for unit, value in sorted(doses.items()):
        by_value.setdefault(value, []).append(unit)
    return [(_VALUED[op](value, dim=dim), tuple(units)) for value, units in by_value.items()]


def transform_id(transform: object) -> str:
    """A stable description of ``transform`` for the compiled schedule's digest: a built-in's parameters, or a
    callable's qualified name."""
    if isinstance(transform, (Scale, Replace, Add, Clamp, Noise)):
        return repr(transform)
    if isinstance(transform, _Span):
        d = torch.as_tensor(transform.directions)
        return (f"{type(transform).__name__}(directions={tuple(d.shape)}:{d.double().sum().item():.12g},"
                f" fraction={transform.fraction}, dim={transform.dim})")
    target = getattr(transform, "__func__", transform)
    qualname = getattr(target, "__qualname__", None) or type(transform).__qualname__
    module = getattr(target, "__module__", None) or type(transform).__module__
    return f"{module}.{qualname}"


__all__ = ["Add", "Clamp", "GRADIENT_OPS", "is_stateful", "Noise", "op_transforms", "Project", "Reject", "Replace",
           "Scale", "transform_id"]
