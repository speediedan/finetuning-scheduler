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
"""``FTSController`` scheduling an FSDP2 model through a view with roles (pool cases, gloo or NCCL).

As in :mod:`tests.dist_cases.fts_controller`, equal per-rank shards make the sharded run match a single-process run on
the full batch: the same phase path, and parameters to float tolerance. The view's roles decide what is thawed and what
the distributed adapter writes to the ``torch.distributed.checkpoint`` directory.
"""
import os

import torch
from torch.distributed.fsdp import fully_shard

from finetuning_scheduler import CanonicalView, FTSController, TorchHost
from finetuning_scheduler.core.checkpoint import BestCheckpoint
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.strategies import FSDP2Adapter
from tests.dist_cases.fts_controller import _data, _model, _params, _train

ROLES = {"4.weight": "adapter:head", "4.bias": "adapter:head", "0.weight": "instrument", "0.bias": "instrument",
         "2.bias": "fitted"}
SCHEDULE = {0: {"params": ["role:adapter"], "max_transition_epoch": 2},
            1: {"params": ["2.weight"], "lr": 0.05, "max_transition_epoch": 4}}


def _controller(model, ckpt_dir, distributed=None, schedule=SCHEDULE):
    from functools import partial

    fts = FTSController(model, {k: dict(v) for k, v in schedule.items()},
                        optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9), host=TorchHost(model),
                        checkpoint=BestCheckpoint(ckpt_dir, "val_loss"), view=CanonicalView(model, roles=ROLES),
                        distributed=distributed)
    fts.setup()
    return fts


def _sharded(ctx, seed):
    model = _model(seed).to(ctx.device)
    for layer in (model[0], model[2], model[4]):
        fully_shard(layer, mesh=ctx.mesh())
    fully_shard(model, mesh=ctx.mesh())
    return model


def _saved_keys(path) -> list[str]:
    import torch.distributed.checkpoint as dcp

    return sorted(k for k in dcp.FileSystemReader(path).read_metadata().state_dict_metadata if k.startswith("model."))


def roles(ctx, ckpt_dir: str, explicit: bool = False, epochs: int = 5, steps: int = 2, batch: int = 8,
          seed: int = 7) -> dict:
    """The roles view on FSDP2: the phase path, the parameters, and the keys of the sharded best checkpoint."""
    model = _sharded(ctx, seed)
    initial = _params(model)
    fts = _controller(model, ckpt_dir, distributed=FSDP2Adapter() if explicit else None)
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    path = _train(fts, model, x, y, shard, ctx.device, epochs)
    params = _params(model)
    return {"path": path, "params": params, "adapter": fts._distributed.name,
            "frozen_unchanged": all(params[n] == initial[n] for n in ("0.weight", "0.bias", "2.bias")),
            "saved_keys": _saved_keys(fts._checkpoint.best_path)}


def reference(ckpt_dir: str, epochs: int = 5, steps: int = 2, batch: int = 8, seed: int = 7) -> dict:
    model = _model(seed)
    fts = _controller(model, ckpt_dir)
    x, y = _data(epochs, steps, batch, seed)
    path = _train(fts, model, x, y, slice(None), torch.device("cpu"), epochs)
    return {"path": path, "params": _params(model), "adapter": fts._distributed.name}


def promotion_refused(ctx, ckpt_dir: str, seed: int = 7) -> dict:
    """Promotion on a sharded model is refused at setup, naming the phase."""
    model = _sharded(ctx, seed)
    schedule = {**SCHEDULE, 1: {**SCHEDULE[1], "promote": {"role:instrument": "latent:probe"}}}
    try:
        _controller(model, os.path.join(ckpt_dir, str(ctx.rank)), schedule=schedule)
    except FTSConfigurationError as err:
        return {"refused": "sharded model" in str(err)}
    return {"refused": False}


def explicit_unsharded_refused(ctx, ckpt_dir: str, seed: int = 7) -> dict:
    """An explicit unsharded adapter on an FSDP2 model is refused: one file would keep one rank's shards only."""
    from finetuning_scheduler.strategies import UnshardedAdapter

    model = _sharded(ctx, seed)
    try:
        _controller(model, os.path.join(ckpt_dir, str(ctx.rank)), distributed=UnshardedAdapter())
    except FTSConfigurationError as err:
        return {"refused": "does not recognize" in str(err)}
    return {"refused": False}


def save_sharded(ctx, ckpt_dir: str, epochs: int = 5, steps: int = 2, batch: int = 8, seed: int = 7) -> dict:
    """Train the roles view on FSDP2 and save the state (a DCP directory); return the full parameters."""
    model = _sharded(ctx, seed)
    fts = _controller(model, os.path.join(ckpt_dir, "best"))
    x, y = _data(epochs, steps, batch, seed)
    shard = slice(ctx.rank * batch // ctx.world_size, (ctx.rank + 1) * batch // ctx.world_size)
    _train(fts, model, x, y, shard, ctx.device, epochs)
    return {"saved": fts.save(os.path.join(ckpt_dir, "resume")), "params": _params(model)}
