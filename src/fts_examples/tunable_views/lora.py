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
LoRA for the Tunable Views Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A minimal LoRA wrapper for the tunable views examples (``notebooks/dev/tunable_views/lora_placement.ipynb`` and
:mod:`~fts_examples.tunable_views.worked_example`). Its tensors are named ``lora_A`` and ``lora_B``, as PEFT names
them, so venngram's :class:`~venngram.maps.MappedView` gives them the ``adapter`` role through its
:data:`~venngram.maps.PEFT_LORA_RULES`, and places every tensor (adapter or not) at its layer and architecture-neutral
component from the model's component map. The examples therefore need no view code of their own.

.. warning:: This is an experimental API and may change in a future release.
"""
import torch
from torch import nn


class LoRALinear(nn.Module):
    """A ``base`` linear layer plus a low-rank update, ``(alpha / r) * lora_B(lora_A(x))``, that starts at zero.

    The update's tensors are float32 on the base layer's device, whatever the base layer's dtype.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float | None = None) -> None:
        super().__init__()
        self.base = base
        self.scale = (alpha if alpha is not None else r) / r
        kwargs = {"bias": False, "device": base.weight.device, "dtype": torch.float32}
        self.lora_A = nn.Linear(base.in_features, r, **kwargs)
        self.lora_B = nn.Linear(r, base.out_features, **kwargs)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        update = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scale
        return self.base(x) + update.to(x.dtype)


def add_lora(model: nn.Module, targets: tuple[str, ...] = ("q_proj", "v_proj"), r: int = 4,
             alpha: float | None = None) -> nn.Module:
    """Wrap every ``nn.Linear`` named in ``targets`` in a :class:`LoRALinear`, in place."""
    for module in list(model.modules()):
        for target in targets:
            child = getattr(module, target, None)
            if isinstance(child, nn.Linear):
                setattr(module, target, LoRALinear(child, r, alpha))
    return model
