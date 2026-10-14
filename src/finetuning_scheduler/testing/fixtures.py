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
Fine-Tuning Scheduler Fixture Views
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Four small, known-good :class:`~finetuning_scheduler.core.views.TunableView` shapes. Each function builds a fresh
model and returns its view, so they can be passed directly as the ``make_view`` factory of
:class:`~finetuning_scheduler.testing.conformance.TunableViewConformance`. They are also templates for implementers.

- :func:`identity_view`: :class:`~finetuning_scheduler.core.views.CanonicalView` over a model with a ``BatchNorm``, a
  tied (shared) parameter and a buffer.
- :func:`one_to_many_view`: logical names that each stand for several physical tensors, two logical names over one
  tied tensor, logical tensors that are reshaped views, coupled norms, and a physical tensor no logical name reaches.
- :func:`roles_view`: every role kind, including tensors outside the base model, ``fitted`` buffers and a ``fitted``
  frozen parameter.
- :func:`parametrized_view`: a parametrized weight, reachable by its unparametrized name.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Iterator, Mapping, Sequence

import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize

from finetuning_scheduler.core.views import CanonicalView, TunableSupport, Unresolvable, ViewDefaults

D, V = 8, 16


class TiedNet(nn.Module):
    """An embedding tied to the output projection, a ``BatchNorm`` (with buffers) and a hidden layer."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.hidden = nn.Linear(D, D)
        self.norm = nn.BatchNorm1d(D)
        self.out = nn.Linear(D, V, bias=False)
        self.out.weight = self.embed.weight
        self.register_buffer("scale", torch.ones(D))


def identity_view() -> CanonicalView:
    """The canonical view over :class:`TiedNet`."""
    return CanonicalView(TiedNet())


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.mlp_in = nn.Linear(D, 2 * D)
        self.mlp_out = nn.Linear(2 * D, D)
        # a fused copy minted alongside the split weights, exposed only physically (no logical name reaches it)
        self.fused_qk = nn.Parameter(torch.zeros(2 * D, D))


class TinyTransformer(nn.Module):
    def __init__(self, n_blocks: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(n_blocks))
        self.ln_final = nn.LayerNorm(D)
        self.unembed = nn.Linear(D, V, bias=False)
        self.unembed.weight = self.embed.weight


class OneToManyView(ViewDefaults):
    """Interpretability-style logical names over :class:`TinyTransformer`.

    ``blocks.<i>.attn.W_QK`` stands for the block's query and key weights and biases; ``blocks.<i>.mlp.W_in`` and
    ``W_out`` for each MLP projection; ``embed.W_E`` and ``unembed.W_U`` both for the one tied embedding. Thawing any
    of a block's tensors also thaws the block's norm, and thawing the embedding thaws the final norm. Logical tensors
    are reshaped views of the physical ones.
    """

    def __init__(self, root: TinyTransformer) -> None:
        self.root = root
        self._n_blocks = len(root.blocks)
        self._table: dict[str, tuple[str, ...]] = {"embed.W_E": ("embed.weight",), "unembed.W_U": ("embed.weight",)}
        for i in range(self._n_blocks):
            b = f"blocks.{i}"
            self._table[f"{b}.attn.W_QK"] = (f"{b}.q.weight", f"{b}.q.bias", f"{b}.k.weight", f"{b}.k.bias")
            self._table[f"{b}.mlp.W_in"] = (f"{b}.mlp_in.weight", f"{b}.mlp_in.bias")
            self._table[f"{b}.mlp.W_out"] = (f"{b}.mlp_out.weight", f"{b}.mlp_out.bias")
        self._reverse: dict[str, str] = {}
        for logical, physical in self._table.items():
            for p in physical:
                self._reverse.setdefault(p, logical)

    def support(self) -> TunableSupport:
        return TunableSupport(trainable=True, naming_scheme="tiny-logical", logical_names_are_views=True)

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        unresolved = [n for n in logical if n not in self._table]
        if unresolved:
            raise Unresolvable(unresolved)
        return {n: self._table[n] for n in logical}

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(self._reverse.get(p, p) for p in physical))

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        for logical, physical in self._table.items():
            yield logical, self.root.get_parameter(physical[0]).view(-1)

    def coupled(self, physical: Sequence[str]) -> list[str]:
        out: list[str] = []
        for p in physical:
            if p.startswith("blocks."):
                b = ".".join(p.split(".")[:2])
                out += [f"{b}.ln.weight", f"{b}.ln.bias"]
            elif p == "embed.weight":
                out += ["ln_final.weight", "ln_final.bias"]
        return list(dict.fromkeys(out))

    def depth_groups(self) -> list[list[str]]:
        """Output side first, one block at a time; the tied embedding's two names stay in one group."""
        groups = [["unembed.W_U", "embed.W_E"]]
        for i in reversed(range(self._n_blocks)):
            b = f"blocks.{i}"
            groups.append([f"{b}.mlp.W_out", f"{b}.mlp.W_in", f"{b}.attn.W_QK"])
        return groups


def one_to_many_view() -> OneToManyView:
    """The logical view over a fresh :class:`TinyTransformer`."""
    return OneToManyView(TinyTransformer())


class InterventionModel(nn.Module):
    """A base model plus modules that live beside it: adapters, an intervention, a latent dictionary, an instrument
    lens and fitted state."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.adapter_backbone = nn.Linear(D, D, bias=False)
        self.adapter_char = nn.Linear(D, D, bias=False)
        self.jspace = nn.Parameter(torch.zeros(D))
        self.transcoder = nn.Linear(D, 2 * D)
        self.lens = nn.Linear(D, V)
        self.procrustes = nn.Parameter(torch.eye(D), requires_grad=False)
        self.register_buffer("enforce_mean", torch.zeros(D))
        self.register_buffer("enforce_mask", torch.ones(D))


ROLES = {
    "adapter_backbone.weight": "adapter:backbone",
    "adapter_char.weight": "adapter:char",
    "jspace": "intervention:jspace",
    "transcoder.weight": "latent:transcoder",
    "transcoder.bias": "latent:transcoder",
    "lens.weight": "instrument",
    "lens.bias": "instrument",
    "procrustes": "fitted",
    "enforce_mean": "fitted",
    "enforce_mask": "fitted",
}


def roles_view() -> CanonicalView:
    """The canonical view over :class:`InterventionModel`, with :data:`ROLES`."""
    return CanonicalView(InterventionModel(), roles=ROLES)


class _Orthogonalish(nn.Module):
    def forward(self, w: Tensor) -> Tensor:
        return w - w.transpose(0, 1)


def parametrized_view() -> CanonicalView:
    """The canonical view over a model whose first layer's weight is parametrized."""
    model = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, 2))
    parametrize.register_parametrization(model[0], "weight", _Orthogonalish())
    return CanonicalView(model)


ALL_FIXTURES = {"identity": identity_view, "one_to_many": one_to_many_view, "roles": roles_view,
                "parametrized": parametrized_view}
