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
"""A two-phase schedule on FSDP2, without FTS: train the head, then thaw a middle layer and add it to the
optimizer.

The same schedule runs in one process on the full batch (:func:`reference`). With equal per-rank shards, FSDP2's
gradient averaging makes the two mathematically identical, so the parameters must agree to float tolerance.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import fully_shard


def _model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _data(steps: int, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed + 1)
    return torch.randn(steps, batch, 8, generator=gen), torch.randn(steps, batch, 4, generator=gen)


def _train(model, steps, thaw_at, lr, x, y, step_loss) -> list:
    for p in model.parameters():
        p.requires_grad_(False)
    model[4].requires_grad_(True)
    opt = torch.optim.SGD(model[4].parameters(), lr=lr)
    losses = []
    for step in range(steps):
        if step == thaw_at:  # the phase transition
            model[2].requires_grad_(True)
            opt.add_param_group({"params": list(model[2].parameters())})
        loss = F.mse_loss(model(x[step]), y[step])
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(step_loss(loss))
    return losses


def thaw_schedule(ctx, steps: int = 6, thaw_at: int = 3, lr: float = 0.1, batch: int = 8, seed: int = 7) -> dict:
    model = _model(seed).to(ctx.device)
    for layer in (model[0], model[2], model[4]):
        fully_shard(layer, mesh=ctx.mesh())
    fully_shard(model, mesh=ctx.mesh())
    x, y = _data(steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)

    def global_loss(loss: torch.Tensor) -> float:
        total = loss.detach().clone()
        dist.all_reduce(total)
        return total.item() / ctx.world_size

    losses = _train(model, steps, thaw_at, lr, x[:, shard].to(ctx.device), y[:, shard].to(ctx.device), global_loss)
    params = {name: p.full_tensor().flatten().tolist() for name, p in model.named_parameters()}
    return {"losses": losses, "params": params}


def reference(steps: int = 6, thaw_at: int = 3, lr: float = 0.1, batch: int = 8, seed: int = 7) -> dict:
    model = _model(seed)
    x, y = _data(steps, batch, seed)
    losses = _train(model, steps, thaw_at, lr, x, y, lambda loss: loss.item())
    return {"losses": losses, "params": {name: p.detach().flatten().tolist() for name, p in model.named_parameters()}}
