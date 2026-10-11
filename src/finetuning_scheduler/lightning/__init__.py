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
Fine-Tuning Scheduler Lightning Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Importing this package registers Lightning's rank-zero utilities and exception type with the Fine-Tuning Scheduler
core, so core code reports ranks, logs, warns and raises exactly as the rest of a Lightning program does.

"""
import lightning_utilities.core.rank_zero as rank_zero_module

# imported for its side effect: it points ``rank_zero_module.log`` at the Lightning rank-zero logger
from lightning.pytorch.utilities import rank_zero as _pl_rank_zero  # noqa: F401
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from finetuning_scheduler.core import dist as core_dist
from finetuning_scheduler.core.exceptions import FTSConfigurationError, register_configuration_error


class FTSMisconfigurationException(FTSConfigurationError, MisconfigurationException):
    """The :class:`~finetuning_scheduler.core.exceptions.FTSConfigurationError` raised when Fine-Tuning Scheduler
    runs under Lightning.

    It is both a ``ValueError`` (via ``FTSConfigurationError``) and a Lightning ``MisconfigurationException``.
    """


def _lightning_global_rank() -> int:
    rank = getattr(rank_zero_module.rank_zero_only, "rank", None)
    if rank is None:
        raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
    return rank


# Lightning's rank-zero functions add three frames (the function, its rank-zero wrapper and the inner emit helper)
LIGHTNING_RANK_ZERO_BACKEND = core_dist.RankZeroBackend(
    global_rank=_lightning_global_rank,
    warn=rank_zero_module.rank_zero_warn,
    info=rank_zero_module.rank_zero_info,
    debug=rank_zero_module.rank_zero_debug,
    stack_offset=3,
)

register_configuration_error(FTSMisconfigurationException)
core_dist.register_rank_zero_backend(LIGHTNING_RANK_ZERO_BACKEND)

__all__ = ["FTSMisconfigurationException", "LIGHTNING_RANK_ZERO_BACKEND"]
