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
from typing import Any, Dict, List, Set

import torch

if torch.distributed.is_available():
    from torch.distributed.fsdp.wrap import _Policy, CustomPolicy


class NameDrivenCustomPolicy(CustomPolicy):
    """An auto-wrapping policy extension that applies module name-based override directives on top of a given base
    ``auto_wrap_policy``.

    The composition of module name-based wrapping directives with a given ``auto_wrap_policy`` is
    achieved here by:
        1. Subclassing ``CustomPolicy``, initializing it with an object id-based module name mapping lambda to
            create a module name-driven policy with a base policy handle.
        2. Generating a base wrapping configuration dictionary by running the user's provided ``auto_wrap_policy``
        3. Updating the base configuration dictionary with any wrapping configuration generated by the module
            name-driven policy and returning the composed configuration.
    """

    def __init__(self, auto_wrap_policy_handle: _Policy, override_ids: List):
        """Compose the provided ``auto_wrap_policy`` with any provided override directives.

        Args:
            auto_wrap_policy_handle (Union[Callable, _Policy]): The user's base ``auto_wrap_policy``.
            override_ids (List): Object ids of the desired modules to wrap even if the provided ``auto_wrap_policy``
                otherwise would not dictate so.
        """
        super().__init__(lambda_fn=lambda m: id(m) in override_ids)
        self._base_awp = auto_wrap_policy_handle

    def _run_policy(
        self,
        root_module: torch.nn.Module,
        ignored_modules: Set[torch.nn.Module],
        root_kwargs: Dict[str, Any],
    ) -> Dict[torch.nn.Module, Dict[str, Any]]:
        target_module_to_kwargs = self._base_awp._run_policy(root_module, ignored_modules, root_kwargs)
        nb_policy_target_module_to_kwargs = super()._run_policy(root_module, ignored_modules, root_kwargs)
        target_module_to_kwargs.update(nb_policy_target_module_to_kwargs)
        return target_module_to_kwargs

NameDrivenPolicy = NameDrivenCustomPolicy
