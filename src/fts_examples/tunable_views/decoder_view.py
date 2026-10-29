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
A Tunable View for Decoder Language Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The view of the :doc:`worked example</examples/tunable_views/worked_example>`. It keeps the model's own parameter names
(so names and regexes keep working) and adds two things a schedule can address:

- **roles**: every LoRA tensor is an ``adapter``, everything else ``base``;
- **components**: each tensor's architecture-neutral component (``attn.q``, ``mlp.down``, ``norm.attn``, ...), read
  from a small table per model family, so ``component:mlp.* & layers:-1:`` means the same tensors on a Llama-style
  model and an OPT-style one, although their modules are named differently.

Layers come from the default structure (the model's one block list), which needs no table.

.. warning:: This is an experimental API and may change in a future release.
"""
import re
from collections.abc import Mapping

from torch import nn
from venngram.physical import PhysicalStructure
from venngram.resolve import Location

from finetuning_scheduler.core.views import CanonicalView

# module-path patterns (matched on the tensor's owning module, ending at a segment boundary) mapped to the component
# each names, per model family; the first match wins
COMPONENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "llama": (
        (r"self_attn\.q_proj", "attn.q"), (r"self_attn\.k_proj", "attn.k"), (r"self_attn\.v_proj", "attn.v"),
        (r"self_attn\.o_proj", "attn.o"), (r"mlp\.gate_proj", "mlp.gate"), (r"mlp\.up_proj", "mlp.up"),
        (r"mlp\.down_proj", "mlp.down"), (r"input_layernorm", "norm.attn"), (r"post_attention_layernorm", "norm.mlp"),
    ),
    "opt": (
        (r"self_attn\.q_proj", "attn.q"), (r"self_attn\.k_proj", "attn.k"), (r"self_attn\.v_proj", "attn.v"),
        (r"self_attn\.out_proj", "attn.o"), (r"fc1", "mlp.up"), (r"fc2", "mlp.down"),
        (r"self_attn_layer_norm", "norm.attn"), (r"final_layer_norm", "norm.mlp"),
    ),
}


class LoRALinear(nn.Module):
    """A frozen-by-schedule ``base`` linear layer plus a low-rank update, ``lora_B(lora_A(x))``, that starts at
    zero."""

    def __init__(self, base: nn.Linear, r: int = 4) -> None:
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


def add_lora(model: nn.Module, targets: tuple[str, ...] = ("q_proj", "v_proj"), r: int = 4) -> nn.Module:
    """Wrap every ``nn.Linear`` named in ``targets`` in a :class:`LoRALinear`, in place."""
    for module in list(model.modules()):
        for target in targets:
            child = getattr(module, target, None)
            if isinstance(child, nn.Linear):
                setattr(module, target, LoRALinear(child, r))
    return model


class DecoderStructure:
    """Places each tensor at its layer (the default's block list) and names its component from ``table``."""

    vocabulary = True

    def __init__(self, root: nn.Module, table: tuple[tuple[str, str], ...]) -> None:
        self._layers = PhysicalStructure(root)
        self._table = [(re.compile(rf"(?:^|\.){pattern}(?:\.|$)"), component) for pattern, component in table]

    def locate(self, name: str) -> Location:
        layer = self._layers.locate(name).layer
        owner = name.rpartition(".")[0]
        component = next((c for rx, c in self._table if layer is not None and rx.search(owner)), None)
        return Location(layer=layer, component=component)

    def depth(self, stack: str | None = None):
        return self._layers.depth(stack)


class DecoderView(CanonicalView):
    """The identity view of a decoder language model, with LoRA tensors as ``adapter`` roles and a component
    vocabulary.

    Args:
        root: The model.
        family: The key of its component table in :data:`COMPONENTS` (``"llama"`` or ``"opt"``).
    """

    def __init__(self, root: nn.Module, family: str) -> None:
        if family not in COMPONENTS:
            raise ValueError(f"No component table for family {family!r}; known: {sorted(COMPONENTS)}")
        super().__init__(root, roles=self._adapter_roles(root))
        self.family = family

    @staticmethod
    def _adapter_roles(root: nn.Module) -> Mapping[str, str]:
        return {n: "adapter" for n, _ in root.named_parameters() if ".lora_A." in n or ".lora_B." in n}

    def structure(self) -> DecoderStructure:
        return DecoderStructure(self.root, COMPONENTS[self.family])
