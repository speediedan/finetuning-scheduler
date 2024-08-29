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
Fine-Tuning Scheduler Model Parallel Strategy Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` that extends Fine-Tuning Scheduler's support
for PyTorch's SPMD style APIs (e.g. DeviceMesh, FSDP2).

"""
from typing import Any, TYPE_CHECKING

from lightning.pytorch.utilities.exceptions import MisconfigurationException

from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

# TODO: replace local version once Lightning version available
# from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_5
import operator
from lightning_utilities.core.imports import compare_version

_TORCH_GREATER_EQUAL_2_5 = compare_version("torch", operator.ge, "2.5.0", use_base_version=True)

if TYPE_CHECKING:
    pass


class ModelParallelStrategyAdapter(StrategyAdapter):
    """"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """The only user-facing configuration for."""
        super().__init__(*args, **kwargs)
        if not _TORCH_GREATER_EQUAL_2_5:
            # specifically, depends upon https://github.com/pytorch/pytorch/pull/133502 among other changes
            raise MisconfigurationException(f"{type(self).__name__} requires PyTorch 2.5 or higher.")

    # def on_before_init_fts(self) -> None:
    #     # TODO: if offering auto-wrap functionality hook `configure_model` here
    #     pass

    # def on_after_init_fts(self) -> None:
    #     # TODO: if offering auto-wrap functionality gen module map here
    #     pass
    #     # self._gen_ft_sched_module_map()
    #     # self.scheduled_mod_lists = [list(self._ft_schedule_module_map[d]) for d in
    #     #                             self._ft_schedule_module_map.keys()]

    # def on_before_fts_fit_start(self) -> None:
    #     # TODO: if offering auto-wrap functionality, validate config globally here
    #     pass

    # def _suppress_cm_warns(self) -> None:
    #     # TODO: only required if offering auto-wrap functionality
    #     pass

    # def _validate_awp_overrides(self) -> None:
    #     # TODO: only required if offering auto-wrap functionality
    #     pass

    # def _fts_auto_configure_model(self) -> None:
    #     # TODO: only required if offering auto-wrap functionality
    #     pass

    # def _fts_auto_wrap(self) -> None:
    #     # TODO: only required if offering auto-wrap functionality
    #     pass

    # def _after_configure_model(self) -> None:
    #     # TODO: only required if offering auto-wrap functionality
    #     pass

    # def _wrapped_configure_model(self) -> None:  #csm_func: Callable) -> Callable:
    #     # TODO: only required if offering auto-wrap functionality
    #     pass

    # @contextmanager
    # def _enable_name_based_overrides(self) -> Generator:
        # TODO: only required if offering auto-wrap functionality
        pass

    # TODO: just a stub for testing rn
    # @override
    # def _get_target_bn_modules(self, schedule_phase: int) -> List:
    #     """Enumerate the :external+torch:class:`~torch.nn.modules.batchnorm._BatchNorm` modules for a given
    #     schedule phase.

    #     Args:
    #         schedule_phase (int): The phase of the schedule to evaluate.

    #     Returns:
    #         List[Tuple[str, torch.nn.modules.batchnorm._BatchNorm]]: A list of tuples containing the names and
    #           (possibly FSDP wrapped) instances of `BatchNorm` modules associated with a given schedule phase.
    #     """
    #     target_bn_modules = []
    #     # for m_name in self.scheduled_mod_lists[schedule_phase]:
    #     #     mod = self.pl_module.get_submodule(m_name)
    #     #     if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
    #     #         target_bn_modules.append((m_name, mod))
    #     #     # TODO: once 2.0 is no longer supported, switch to using FSDP_WRAPPED_MODULE constant here
    #     #     elif orig_mod := getattr(mod, '_fsdp_wrapped_module', None):
    #     #         if isinstance(orig_mod, torch.nn.modules.batchnorm._BatchNorm):
    #     #             target_bn_modules.append((m_name, orig_mod))
    #     return target_bn_modules
