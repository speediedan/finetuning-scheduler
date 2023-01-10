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
Fine-Tuning Scheduler Strategy Adapters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Base adapter class to extend Fine-Tuning Scheduler support of complex or custom training strategies

"""
from typing import Callable, List, Optional, Tuple

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
from torch.nn import Module


class StrategyAdapter:
    """Base class for all strategies that change the behaviour of the training, validation and test- loop."""

    fts_handle: Callback

    def __init__(
        self,
    ) -> None:
        self.fts_restore_optimizer: bool = False
        self.exec_ft_phase = StrategyAdapter.base_ft_phase

    def connect(self, fts_parent: Callback) -> None:
        self.fts_handle = fts_parent

    @property
    def pl_module(self) -> LightningModule:
        return self.fts_handle.pl_module

    @property
    def pls_handle(self) -> Strategy:
        assert self.pl_module._trainer is not None
        return self.pl_module._trainer.strategy

    @property
    def lightning_restore_optimizer(self) -> bool:
        """Override to disable Lightning restoring optimizers/schedulers.

        This is useful for plugins which manage restoring optimizers/schedulers.
        """
        return True

    def on_before_init_fts(self) -> None:
        """Hook executed in Fine-Tuning Scheduler setup immediately before `init_fts`"""

    def on_after_init_fts(self) -> None:
        """Hook executed in Fine-Tuning Scheduler setup immediately after `init_fts`"""
        _, self.fts_handle._fts_state._curr_thawed_params = self.exec_ft_phase(
            self.pl_module,
            thaw_pl=self.fts_optim_view(self.fts_handle.ft_schedule[0]["params"]),
            init_thaw=True,
        )

    def on_before_fts_fit_start(self) -> None:
        """_summary_"""
        pass

    def on_before_restore_optimizers_and_lrs(self) -> None:
        """summary."""
        pass

    def fts_optim_view(self, orig_pl: List) -> List:
        """By default, no transformation of schedule parameters is required for optimizer operations."""
        return orig_pl

    def logical_param_translation(self, param_names: List) -> List:
        """By default, no transformation of schedule parameters is required for optimizer operations."""
        return param_names

    @staticmethod
    def base_ft_phase(
        module: Module, thaw_pl: List, translation_func: Optional[Callable] = None, init_thaw: bool = False
    ) -> Tuple[List, List]:
        """Thaw/unfreeze the provided list of parameters in the provided :class:`~torch.nn.Module`

        Args:
            module (:class:`~torch.nn.Module`): The :class:`~torch.nn.Module` that will have parameters selectively
                unfrozen/thawed.
            thaw_pl: The list of parameters that should be thawed/unfrozen in the :class:`~torch.nn.Module`
            init_thaw: If ``True``, modifies message to user accordingly. Defaults to ``False``.

        Returns:
            Tuple[List, List]: A Tuple of two lists.
                1. The list of newly thawed/unfrozen parameters thawed by this function
                2. A list of all currently thawed/unfrozen parameters in the target :class:`~torch.nn.Module`
        """
        thawed_p_names = []
        curr_thawed = []
        for n, p in module.named_parameters():
            if not p.requires_grad and n in thaw_pl:
                p.requires_grad = True
                thawed_p_names.append(n)
            elif p.requires_grad:
                curr_thawed.append(n)
        if thawed_p_names:
            rank_zero_debug(
                f"{'Initializing with' if init_thaw else 'Thawed'} the following module parameters: "
                f"{translation_func(thawed_p_names) if translation_func else [n for n in thawed_p_names]}"
                # f"{[n for n in thawed_p_names]}"
            )
        curr_thawed.extend(thawed_p_names)
        rank_zero_debug(
            f"The following module parameters are currently thawed: "
            f"{translation_func(curr_thawed) if translation_func else [n for n in curr_thawed]}"
        )
        return thawed_p_names, curr_thawed
