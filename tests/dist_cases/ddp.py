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
"""DDP and parameters thawed after the wrap, without FTS.

``DistributedDataParallel`` registers its reducer's parameters when it is constructed, from those that require grad
at that moment. A parameter thawed later gets a gradient on every rank but is never all-reduced, so the ranks drift
apart. Scheduled fine-tuning thaws parameters at every phase transition, so both of its hosts must rebuild the
wrapper when the trainable set changes. These cases pin that upstream behavior: if a release starts tracking late
``requires_grad`` changes, the "stale" case stops diverging and the test says so.
"""
import gc

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _wrap(ctx, model: nn.Module) -> DDP:
    return DDP(model, device_ids=[ctx.device.index] if ctx.device.type == "cuda" else None)


def late_thaw(ctx, rebuild: bool, steps: int = 3, lr: float = 0.1, seed: int = 7) -> dict:
    """Train the head, thaw a middle layer after wrapping (optionally rebuilding the wrapper), then train both."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).to(ctx.device)
    model.requires_grad_(False)
    model[4].requires_grad_(True)
    ddp = _wrap(ctx, model)
    model[2].requires_grad_(True)  # the phase transition, after the wrap
    if rebuild:
        del ddp
        gc.collect()  # the old reducer's autograd hooks must be gone before the new wrapper registers its own
        ddp = _wrap(ctx, model)
    opt = torch.optim.SGD([*model[4].parameters(), *model[2].parameters()], lr=lr)
    gen = torch.Generator().manual_seed(seed + 1 + ctx.rank)  # different data per rank, as data parallelism has
    start = model[2].weight.detach().clone()
    for _ in range(steps):
        x, y = torch.randn(4, 8, generator=gen), torch.randn(4, 4, generator=gen)
        F.mse_loss(ddp(x.to(ctx.device)), y.to(ctx.device)).backward()
        opt.step()
        opt.zero_grad()
    thawed = model[2].weight.detach()
    gathered = [torch.empty_like(thawed) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, thawed)
    return {
        "rank_spread": max((g - gathered[0]).abs().max().item() for g in gathered),
        "moved": (thawed - start).abs().max().item(),
    }
