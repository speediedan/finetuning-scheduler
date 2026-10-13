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

The Lightning integration: the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback
(``finetuning_scheduler.lightning.fts``) and its supporting callbacks and mixins
(``finetuning_scheduler.lightning.fts_supporters``). Their established import paths, ``finetuning_scheduler.fts`` and
``finetuning_scheduler.fts_supporters``, remain available.

Importing this package registers Lightning's rank-zero logger and exception type with the Fine-Tuning Scheduler core,
so core code logs, warns and raises as the rest of a Lightning program does. The rank comes from the process group
once one is initialized, and from Lightning's rank otherwise.

"""
import warnings
from typing import Any

import lightning_utilities.core.rank_zero as rank_zero_module
import torch

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
    # An initialized process group decides: Lightning's own rank is read from environment variables when Lightning is
    # imported, which a launcher passing ranks to `init_process_group` directly never sets, so it would be 0 on every
    # rank of a run this package is merely imported into (#97). Under Lightning's launchers the two agree.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    rank = getattr(rank_zero_module.rank_zero_only, "rank", None)
    if rank is None:
        raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
    return rank


def _warn(message: str | Warning, stacklevel: int = 2, **kwargs: Any) -> None:
    if _lightning_global_rank() == 0:
        warnings.warn(message, stacklevel=stacklevel, **kwargs)


def _info(message: str, stacklevel: int = 2, **kwargs: Any) -> None:
    if _lightning_global_rank() == 0:
        rank_zero_module.log.info(message, stacklevel=stacklevel, **kwargs)


def _debug(message: str, stacklevel: int = 2, **kwargs: Any) -> None:
    if _lightning_global_rank() == 0:
        rank_zero_module.log.debug(message, stacklevel=stacklevel, **kwargs)


# Lightning's rank-zero logger, with the rank decided as above (Lightning's `rank_zero_*` functions would re-check its
# static rank); each sink adds one frame
LIGHTNING_RANK_ZERO_BACKEND = core_dist.RankZeroBackend(
    global_rank=_lightning_global_rank, warn=_warn, info=_info, debug=_debug, stack_offset=1
)

register_configuration_error(FTSMisconfigurationException)
core_dist.register_rank_zero_backend(LIGHTNING_RANK_ZERO_BACKEND)

__all__ = ["FTSMisconfigurationException", "LIGHTNING_RANK_ZERO_BACKEND"]
