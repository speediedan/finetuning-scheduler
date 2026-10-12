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
"""Collective smoke cases: each rank contributes a distinct value, so a wrong reduction is visible."""
import torch
import torch.distributed as dist


def all_reduce_sum(ctx, n: int) -> list:
    t = torch.arange(n, device=ctx.device) * (ctx.rank + 1)
    dist.all_reduce(t)
    return t.tolist()


def all_gather_ranks(ctx) -> list:
    out = [torch.zeros(1, device=ctx.device) for _ in range(ctx.world_size)]
    dist.all_gather(out, torch.full((1,), float(ctx.rank), device=ctx.device))
    return [int(t.item()) for t in out]
