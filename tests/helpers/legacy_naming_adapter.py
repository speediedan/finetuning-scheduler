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
"""A legacy parameter-naming strategy adapter for contract tests.

It reproduces, without importing any downstream package, the ways an existing third-party adapter uses the
:class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` surface to schedule *logical* parameter names:

- an argument-free base ``__init__`` and an ``exec_ft_phase`` bound with ``functools.partial`` over
  ``StrategyAdapter.base_ft_phase`` and a translation function;
- ``on_before_init_fts`` building the name mapping and redirecting the module's ``state_dict``/``load_state_dict``;
- a one-to-many ``fts_optim_transform`` that also thaws coupled normalization parameters implicitly;
- ``logical_param_translation`` mapping physical names back to logical ones (unmapped names pass through);
- logical names for schedule validation, and unbound calls to the base ``validate_ft_sched`` and to
  ``ScheduleImplMixin.save_schedule`` through its pre-2.15 import path.

The adapter's tests pin what this surface produces today, so later changes to the adapter layer are measured against
it.
"""
import os
from functools import partial

import torch
from torch import nn

from finetuning_scheduler.fts_supporters import ScheduleImplMixin
from finetuning_scheduler.strategy_adapters.base import StrategyAdapter
from tests.helpers.boring_models import BoringModel

# logical name -> the physical (canonical) parameters it stands for (one-to-many)
LOGICAL_TO_PHYSICAL = {
    "blocks.0.W": ["model.0.weight", "model.0.bias"],
    "blocks.1.W": ["model.2.weight", "model.2.bias"],
    "head.W": ["model.4.weight", "model.4.bias"],
}
# the normalization parameters thawed implicitly with each block (they have no logical name)
COUPLED_NORMS = {
    "blocks.0.W": ["model.1.weight", "model.1.bias"],
    "blocks.1.W": ["model.3.weight", "model.3.bias"],
}


class LogicalNamesBoringModel(BoringModel):
    """Two blocks, each a ``Linear`` followed by a ``LayerNorm``, and a ``Linear`` head."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32), nn.Linear(32, 32), nn.LayerNorm(32),
                                   nn.Linear(32, 2))

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=False)
        return {"x": loss}

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        optimizer = torch.optim.SGD(parameters, lr=1e-3)
        return [optimizer]


class LegacyNamingAdapter(StrategyAdapter):
    """Schedules :data:`LOGICAL_TO_PHYSICAL`'s logical names through the legacy naming overrides."""

    def __init__(self, use_logical_names: bool = True, implicit_norm_thaw: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_logical_names = use_logical_names
        self.implicit_norm_thaw = implicit_norm_thaw
        self._physical_to_logical: dict[str, str] | None = None
        self.exec_ft_phase = partial(StrategyAdapter.base_ft_phase, translation_func=self.logical_param_translation)

    def _ensure_mapping(self) -> None:
        if self._physical_to_logical is None:
            self._physical_to_logical = {p: lg for lg, phys in LOGICAL_TO_PHYSICAL.items() for p in phys}

    def on_before_init_fts(self) -> None:
        assert self.pl_module is not None
        # redirect the module's state-dict methods to the wrapped model's, as a key-translating wrapper would
        setattr(self.pl_module, "state_dict", self.pl_module.model.state_dict)
        setattr(self.pl_module, "load_state_dict", self.pl_module.model.load_state_dict)
        self._ensure_mapping()

    def fts_optim_transform(self, orig_pl: list[str], inspect_only: bool = False) -> list[str]:
        if not self.use_logical_names:
            return orig_pl
        physical: list[str] = []
        for name in orig_pl:
            if name not in LOGICAL_TO_PHYSICAL:
                raise ValueError(f"Logical parameter '{name}' not found in mapping.")
            physical.extend(LOGICAL_TO_PHYSICAL[name])
        if self.implicit_norm_thaw:
            explicit = set(physical)
            physical.extend(p for name in orig_pl for p in COUPLED_NORMS.get(name, []) if p not in explicit)
        return physical

    def logical_param_translation(self, param_names: list[str]) -> list[str]:
        if not self.use_logical_names:
            return param_names
        self._ensure_mapping()
        assert self._physical_to_logical is not None
        translated = [self._physical_to_logical.get(n, n) for n in param_names]
        return list(dict.fromkeys(translated))  # de-duplicate, preserving order

    def get_named_params_for_schedule_validation(self) -> dict[str, torch.Tensor]:
        if not self.use_logical_names:
            return super().get_named_params_for_schedule_validation()
        named = dict(self.pl_module.model.named_parameters(prefix="model"))
        return {lg: named[phys[0]] for lg, phys in LOGICAL_TO_PHYSICAL.items()}

    def validate_ft_sched(self) -> tuple[int, int]:
        self._ensure_mapping()
        return StrategyAdapter.validate_ft_sched(self)  # an unbound call to the base implementation

    def gen_ft_schedule(self, dump_loc: str | os.PathLike) -> os.PathLike | None:
        if not self.use_logical_names:
            return StrategyAdapter.gen_ft_schedule(self, dump_loc)
        names = list(LOGICAL_TO_PHYSICAL)[::-1]
        layer_config = {i: {"params": [n]} for i, n in enumerate(names)}
        schedule_name = f"{self.pl_module.__class__.__name__}_ft_schedule.yaml"
        return ScheduleImplMixin.save_schedule(schedule_name, layer_config, dump_loc)
