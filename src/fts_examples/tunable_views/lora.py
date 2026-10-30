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

A minimal LoRA wrapper for the tunable views examples (the notebooks in ``notebooks/dev/tunable_views/`` and
:mod:`~fts_examples.tunable_views.two_families`). It wraps ``nn.Linear`` and the transposed ``Conv1D`` projections
GPT-2 uses, by attribute name or by module path. Its tensors are named ``lora_A`` and ``lora_B``, as PEFT names
them, so venngram's :class:`~venngram.maps.MappedView` gives them the ``adapter`` role through its
:data:`~venngram.maps.PEFT_LORA_RULES`, and places every tensor (adapter or not) at its layer and architecture-neutral
component from the model's component map.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Iterable

import torch
from torch import nn


def _features(base: nn.Module) -> tuple[int, int]:
    """``(in_features, out_features)`` of an ``nn.Linear``, or of a ``Conv1D`` (whose weight is ``(in, out)``)."""
    if isinstance(base, nn.Linear):
        return base.in_features, base.out_features
    if type(base).__name__ == "Conv1D":
        return base.weight.shape[0], base.weight.shape[1]
    raise TypeError(f"LoRA wraps nn.Linear or Conv1D projections, not {type(base).__name__}")


class LoRALinear(nn.Module):
    """A ``base`` projection plus a low-rank update, ``(alpha / r) * lora_B(lora_A(x))``, that starts at zero.

    The update's tensors are float32 on the base layer's device, whatever the base layer's dtype.
    """

    def __init__(self, base: nn.Module, r: int = 4, alpha: float | None = None) -> None:
        super().__init__()
        in_features, out_features = _features(base)
        self.base = base
        self.scale = (alpha if alpha is not None else r) / r
        kwargs = {"bias": False, "device": base.weight.device, "dtype": torch.float32}
        self.lora_A = nn.Linear(in_features, r, **kwargs)
        self.lora_B = nn.Linear(r, out_features, **kwargs)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        update = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scale
        return self.base(x) + update.to(x.dtype)


def add_lora(model: nn.Module, targets: tuple[str, ...] = ("q_proj", "v_proj"), r: int = 4,
             alpha: float | None = None, *, modules: Iterable[str] | None = None) -> nn.Module:
    """Wrap projections in a :class:`LoRALinear`, in place.

    Args:
        model: The model.
        targets: Attribute names: every ``nn.Linear`` child with one of these names is wrapped. Ignored when
            ``modules`` is given.
        r: The update's rank.
        alpha: Its scale numerator (``r`` when ``None``).
        modules: Module paths to wrap instead (``nn.Linear`` or ``Conv1D``), for example the owners of the tensors an
            address selects.
    """
    if modules is not None:
        for path in modules:
            parent, _, attr = path.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            setattr(owner, attr, LoRALinear(getattr(owner, attr), r, alpha))
        return model
    for module in list(model.modules()):
        for target in targets:
            child = getattr(module, target, None)
            if isinstance(child, nn.Linear):
                setattr(module, target, LoRALinear(child, r, alpha))
    return model
