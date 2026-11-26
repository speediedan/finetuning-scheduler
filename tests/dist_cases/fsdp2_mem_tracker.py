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
"""Training steps under the FSDP2 memory tracker, on a model with a sharded unit whose forward does not run.

FSDP2's root post-backward callback calls ``post_backward`` on that unit's parameter group too, which upstream's
``FSDPMemTracker`` cannot instrument (speediedan/finetuning-scheduler#246); the profiling core's own tracker can.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker, _FSDPModState
from torch.distributed.fsdp import fully_shard

from finetuning_scheduler.profiling.memprofiler import _FSDPMemTracker


class _TwoBranches(nn.Module):
    """``unused`` is a branch the step never calls, like a multimodal model's vision tower on a text task."""

    def __init__(self) -> None:
        super().__init__()
        self.used, self.unused, self.head = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.used(x)))


def unused_unit_step(ctx, upstream: bool = False, steps: int = 2) -> dict:
    torch.manual_seed(0)
    model = _TwoBranches().to(ctx.device)
    for unit in (model.used, model.unused):
        fully_shard(unit, mesh=ctx.mesh())
    fully_shard(model, mesh=ctx.mesh())
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(steps, 4, 8, generator=gen).to(ctx.device)
    y = torch.randn(steps, 4, 4, generator=gen).to(ctx.device)
    tracker = (FSDPMemTracker if upstream else _FSDPMemTracker)(model, opt)
    error, post_backward = None, []
    try:
        for step in range(steps):  # one tracked call per step, as the profiler tracks each profiled call
            with tracker:
                F.mse_loss(model(x[step]), y[step]).backward()
                opt.step()
                opt.zero_grad()
            # a unit enters `memory_tracking` at its first pre-forward, so the unused one has no entry
            post_backward.append({name: len(tracker.memory_tracking[mod].snapshots.get(_FSDPModState.AFT_POST_BW, []))
                                  if mod in tracker.memory_tracking else None
                                  for name, mod in (("used", model.used), ("unused", model.unused), ("root", model))})
            tracker.reset_mod_stats()
    except AttributeError as err:
        error = f"AttributeError: {err}"
    return {"error": error, "post_backward_snapshots": post_backward}
