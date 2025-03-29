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
from typing import Dict, List, Optional, Union

import torch
#from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_4
from lightning.fabric.accelerators.cuda import is_cuda_available
from lightning.fabric.utilities import rank_zero_info
from lightning.pytorch.profilers.pytorch import _KINETO_AVAILABLE, PyTorchProfiler
from torch.profiler.profiler import ProfilerActivity


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
        activities.append(ProfilerActivity.CPU)
        if is_cuda_available():
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
