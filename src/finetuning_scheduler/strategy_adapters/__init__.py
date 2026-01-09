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

Strategy adapters extend Fine-Tuning Scheduler support for complex or custom training strategies.
The built-in adapters (:class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter`,
:class:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter`) handle PyTorch's
advanced distributed training strategies.

Plugin Support
**************

.. warning::
    This is an :ref:`experimental <versioning:API Stability Classifications>` feature which is
    still in development.

Fine-Tuning Scheduler supports custom strategy adapters via Python entry points. Third-party packages
can register custom strategy adapters that will be automatically discovered at runtime.

To register a custom strategy adapter, add an entry point in your package's ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."finetuning_scheduler.strategy_adapters"]
    my_adapter = "my_package.adapters:MyStrategyAdapter"

The entry point name (``my_adapter`` in this example) will be used to reference the adapter,
automatically lowercased. Once registered, the adapter can be used by mapping Lightning strategy
flags to the adapter via the ``custom_strategy_adapters`` parameter. You can use the entry point
name, a fully qualified class path with colon separator (``module:Class``), or dot separator
(``module.Class``):

.. code-block:: python

    from finetuning_scheduler import FinetuningScheduler

    # Map strategy flags to adapters using entry point name
    fts = FinetuningScheduler(
        custom_strategy_adapters={
            "single_device": "my_adapter",  # Entry point name
            "ddp": "my_package.adapters:MyStrategyAdapter",  # Colon-separated
            "fsdp": "my_package.adapters.MyStrategyAdapter",  # Dot-separated
        }
    )

See :ref:`strategy_adapter_entry_points` for complete documentation and examples.
"""
from finetuning_scheduler.strategy_adapters.base import StrategyAdapter
from finetuning_scheduler.strategy_adapters.fsdp import FSDPStrategyAdapter
from finetuning_scheduler.strategy_adapters.model_parallel import ModelParallelStrategyAdapter

__all__ = ["StrategyAdapter", "FSDPStrategyAdapter", "ModelParallelStrategyAdapter"]
