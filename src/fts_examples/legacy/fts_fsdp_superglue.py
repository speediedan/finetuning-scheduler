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

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Type, Union

import torch
from lightning.fabric.accelerators.cuda import is_cuda_available
from lightning.fabric.utilities import rank_zero_info
from lightning.pytorch.profilers.pytorch import _KINETO_AVAILABLE, PyTorchProfiler
from torch.profiler.profiler import ProfilerActivity

from fts_examples import _HF_AVAILABLE
from fts_examples.legacy.cli_experiment_utils import instantiate_class
from fts_examples.legacy.fts_superglue import RteBoolqModule

if _HF_AVAILABLE:
    from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Embeddings, DebertaV2Encoder, DebertaV2Layer

# NOTE: We use the non-partial formulation of these auto-wrap policies here principally for expository benefit

##########################################
# FSDP Scheduled Fine-Tuning Demo Policy #
##########################################
deberta_transformer_layer_cls: Set = {DebertaV2Layer, DebertaV2Embeddings, DebertaV2Encoder}


def deberta_awp(
    module: torch.nn.Module,
    recurse: bool,
    unwrapped_params: int,
    transformer_layer_cls: Set[Type[torch.nn.Module]] = deberta_transformer_layer_cls,
) -> bool:
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or remainder
        return isinstance(module, tuple(transformer_layer_cls))


##################################
# Debugging Feedback Demo Policy #
##################################
degenerate_deberta_transformer_layer_cls: Set = {DebertaV2Layer}


def degenerate_deberta_awp(
    module: torch.nn.Module,
    recurse: bool,
    unwrapped_params: int,
    transformer_layer_cls: Set[Type[torch.nn.Module]] = degenerate_deberta_transformer_layer_cls,
) -> bool:
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or remainder
        return isinstance(module, tuple(transformer_layer_cls))


class RteBoolqModuleFSDP(RteBoolqModule):
    # we override `configure_optimizers` because use of the `no_decay` lightning module attribute is not currently
    # supported with FTS FSDP strategy adapter
    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.model.parameters())
        optimizer = instantiate_class(args=parameters, init=self.hparams.optimizer_init)
        scheduler = {
            "scheduler": instantiate_class(args=optimizer, init=self.hparams.lr_scheduler_init),
            **self.hparams.pl_lrs_cfg,
        }
        return [optimizer], [scheduler]


class ExtendedPyTorchProfiler(PyTorchProfiler):
    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        schedule_cfg: Dict = None,
        activities: Optional[List[ProfilerActivity]] = None,
        max_name_column_width: int = 55,
        sort_by_key: Optional[str] = "cpu_time_total",
        *args,
        **kwargs,
    ) -> None:
        """A few minor customizations of the ``PyTorchProfiler`` that should be unnecessary after future upstream
        updates.

        1. We offer the user the ability to customize the ``max_name_column_width`` (useful for long fully-qualified
           module names).
        2. Cleaner profile teardown to improve user feedback visibility when profiling enabled
        3. ``use_cuda`` is deprecated but ``jsonargparse`` will inspect the profiler signature and still pass it by
           default. We set activities for the user before profiler construction to avoid a superfluous warning.
        """
        activities = activities or self._set_activities(kwargs)
        if kwargs.get("use_cuda", None) is not None:
            del kwargs["use_cuda"]
        schedule = torch.profiler.schedule(**schedule_cfg) if schedule_cfg else None
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            schedule=schedule,
            activities=activities,
            sort_by_key=sort_by_key,
            *args,
            **kwargs,
        )
        self.max_name_column_width = max_name_column_width

    def _set_activities(self, kwargs) -> List["ProfilerActivity"]:
        activities: List["ProfilerActivity"] = []
        if not _KINETO_AVAILABLE:
            return activities
        if kwargs.get("use_cpu", True):
            activities.append(ProfilerActivity.CPU)
        if kwargs.get("use_cuda", is_cuda_available()):
            activities.append(ProfilerActivity.CUDA)
        return activities

    def summary(self) -> str:

        if not self._profiler_kwargs.get("enabled", True) or self._emit_nvtx:
            return ""

        self._delete_profilers()

        if not self.function_events:
            return ""

        if self._export_to_chrome and not _KINETO_AVAILABLE:
            filename = f"{self.local_rank}_trace.json"
            path_to_trace = filename if self.dirpath is None else os.path.join(self.dirpath, filename)
            self.function_events.export_chrome_trace(path_to_trace)

        data = self.function_events.key_averages(group_by_input_shapes=self._group_by_input_shapes)
        table = data.table(
            sort_by=self._sort_by_key, row_limit=self._row_limit, max_name_column_width=self.max_name_column_width
        )

        recorded_stats = {"records": table}
        return self._stats_to_str(recorded_stats)

    def _delete_profilers(self) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            try:
                self._cache_functions_events()
            except AssertionError:
                rank_zero_info("Tearing down profiler...")
            self.profiler = None

        if self._schedule is not None:
            self._schedule.reset()

        if self._parent_profiler is not None:
            self._parent_profiler.__exit__(None, None, None)
            self._parent_profiler = None

        if self._register is not None:
            self._register.__exit__(None, None, None)
            self._register = None


cust_profiler_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, skip_first=20)
