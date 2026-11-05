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
"""Mapped tensor roles and output-coordinate manifests for the neuron-mask tutorials.

The score tensor belongs to the same verified architectural component as its wrapped projection. FTS selects that
tensor, while MaskedDeltaLinear maps its entries to explicit projection-output coordinates. A row of an MLP up
projection is an intermediate neuron; a row of an attention query or MLP down projection belongs to that projection's
output space. These axes are not interchangeable.
"""
from dataclasses import dataclass

from torch import nn
from venngram.maps import map_for, MappedView, overlays_for

from fts_examples.tunable_views.masks import MaskedDeltaLinear


@dataclass(frozen=True)
class MaskCoordinates:
    """The physical score tensor, architectural placement and output rows gated by its entries."""

    score_name: str
    module_path: str
    layer: int | None
    component: str
    rows: tuple[int, ...]
    output_width: int
    axis: str = "projection_output"


def mask_view(model: nn.Module, *, architecture: str | None = None) -> MappedView:
    """Use the registry's component map, assigning only real mask scores the adapter:mask role.

    Ordinary LoRA retains its adapter role; fixed deltas, budgets and frozen base tensors retain their base role and
    checkpoint behavior. Call after installing wrappers, before compiling a schedule. The view does not implement native
    coordinate selection or change trainability. A Lightning module preserving the backbone's module paths can pass its
    architecture explicitly.
    """
    roles = {f"{path}.mask_scores" if path else "mask_scores": "adapter:mask"
             for path, module in model.named_modules() if isinstance(module, MaskedDeltaLinear)}
    if not roles:
        raise ValueError("mask_view needs at least one MaskedDeltaLinear projection")
    maps = [map_for(model, architecture), *overlays_for(model)] if architecture is not None else None
    view = MappedView(model, maps=maps, roles=roles)
    mask_coordinates(view)  # Refuse unplaced tensors before a schedule can use them.
    return view


def mask_coordinates(view: MappedView) -> tuple[MaskCoordinates, ...]:
    """Describe every score entry in physical row coordinates, refusing an unplaced projection.

    A global budget couples projections: freezing one score tensor does not fix its mask values
    while other scores change the shared threshold. The manifest describes eligible coordinates,
    not a claim that every eligible row is selected or independently frozen by FTS.
    """
    result = []
    for path, module in view.root.named_modules():
        if not isinstance(module, MaskedDeltaLinear):
            continue
        name = f"{path}.mask_scores" if path else "mask_scores"
        location = view.structure().locate(name)
        if not location.component:
            raise ValueError(f"mask score {name!r} has no component in the mapped architecture")
        result.append(MaskCoordinates(name, path, location.layer, location.component,
                                      tuple(module.row_indices.tolist()), module.base.out_features))
    return tuple(result)
