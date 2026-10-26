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
"""Model-parallel scheduled fine-tuning in a plain PyTorch loop.

The PyTorch API counterpart of ``mp_examples.py``: the same ``torchtitan`` Llama model, fine-tuning schedule and
parallelism plans, driven by ``TuningScheduler`` in a training loop with no Lightning ``Trainer`` (and no Lightning
import). Launch it with ``torchrun`` on two GPUs::

    torchrun --nproc_per_node=2 torch_mp_example.py --config config/torch_fsdp_plan.yaml
    torchrun --nproc_per_node=2 torch_mp_example.py --config config/torch_tp_plan.yaml

A configuration is read with ``yaml.safe_load`` and merged over ``config/defaults/torch_mp_example_defaults.yaml``.
Relative paths in it resolve against this file's directory, so the example runs from any working directory.
"""
import os

if not os.getenv("SHOW_EXAMPLE_CPP_WARNS", "0") == "1":
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import argparse
import json
import logging
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel
from torch.utils.data import DataLoader, DistributedSampler

from finetuning_scheduler import TuningScheduler, PlateauTransition
from finetuning_scheduler.core.checkpoint import BestCheckpoint
from finetuning_scheduler.core.optim import resolve_class_path
from finetuning_scheduler.strategies.plan import shard_model
from fts_examples.model_parallel.parallel_plans import apply_tp_plan, RandomTokenDataset
from fts_examples.model_parallel.torchtitan_llama import ModelCfg, Transformer

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(EXAMPLE_DIR, "config", "defaults", "torch_mp_example_defaults.yaml")
# the run summary records which of these were imported, so a test can confirm the example stays Lightning-free
LIGHTNING_MODULES = ("lightning", "pytorch_lightning", "lightning_fabric", "lightning_utilities", "torchmetrics",
                     "jsonargparse")


class ModParallelModule(nn.Module):
    """Hold the ``Transformer`` as ``model``, as the Lightning example's ``LightningModule`` does, so both examples
    share the schedule's parameter names and the FSDP2 plan's module names (``model.output.weight``,
    ``model.layers.0``)."""

    def __init__(self, model_cfg: ModelCfg) -> None:
        super().__init__()
        self.model = Transformer(model_cfg)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)


def example_path(path: str) -> str:
    """Resolve a configured path against this example's directory."""
    return path if os.path.isabs(path) else os.path.join(EXAMPLE_DIR, path)


def merge(base: dict, update: Mapping) -> dict:
    """Merge ``update`` over ``base`` in place, mapping by mapping."""
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(args: argparse.Namespace) -> dict:
    """Read the defaults and the given configuration with ``yaml.safe_load``, then apply the command line."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if args.config:
        with open(args.config) as f:
            merge(cfg, yaml.safe_load(f) or {})
    for key in ("max_epochs", "output_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    return cfg


def build_model(cfg: dict, mesh: DeviceMesh, device: torch.device) -> nn.Module:
    """Build the model on the meta device, apply the tensor-parallel plan and then the FSDP2 plan, and only then
    allocate and initialize the weights, sharded, on ``device``."""
    with open(example_path(cfg["model_cfg"])) as f:
        model_cfg = ModelCfg(**yaml.safe_load(f))
    with torch.device("meta"):
        model = ModParallelModule(model_cfg)
    if mesh["tensor_parallel"].size() > 1:
        apply_tp_plan(model.model, device_mesh=mesh, loss_parallel=cfg["loss_parallel"])
    if mesh["data_parallel"].size() > 1 or cfg["fsdp_plan"]:
        # the root is sharded last, so no parameter is left unsharded (and unsynchronized across data-parallel ranks)
        shard_model(model, mesh["data_parallel"], cfg["fsdp_plan"], default_kwargs=cfg["fsdp_default_kwargs"])
    model.to_empty(device=device)
    with torch.no_grad():
        model.model.init_weights()
    return model


def build_tuning_scheduler(cfg: dict, model: nn.Module, checkpoint: BestCheckpoint, log_dir: str) -> TuningScheduler:
    """Build the tuning scheduler from the configuration: the counterpart of the Lightning example's
    ``FinetuningScheduler``, ``FTSEarlyStopping`` and ``FTSCheckpoint`` callbacks and its optimizer and lr scheduler
    configuration."""
    optimizer_cls = resolve_class_path(cfg["optimizer"]["class_path"])
    lr_scheduler = None
    if cfg.get("lr_scheduler"):
        lr_scheduler_cls = resolve_class_path(cfg["lr_scheduler"]["class_path"])
        lr_scheduler = partial(lr_scheduler_cls, **cfg["lr_scheduler"].get("init_args", {}))
    return TuningScheduler(
        model,
        example_path(cfg["ft_schedule"]),
        optimizer=partial(optimizer_cls, **cfg["optimizer"].get("init_args", {})),
        lr_scheduler=lr_scheduler,
        transition=PlateauTransition(**cfg["transition"]),  # with the schedule's `max_transition_epoch` keys
        checkpoint=checkpoint,  # the best state is restored at each phase boundary
        max_depth=cfg["max_depth"],
        log_dir=log_dir,
    )


def local_value(t: torch.Tensor) -> torch.Tensor:
    """A plain tensor holding ``t``'s value (a loss computed under ``loss_parallel`` is a replicated
    ``DTensor``)."""
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach().float()


def mean_across_ranks(total: torch.Tensor, count: int) -> float:
    """Average a per-rank mean over every rank, so each rank reports the same metric to the tuning scheduler."""
    value = (total / max(count, 1)).reshape(1)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value.item() / dist.get_world_size()


def train(cfg: dict, fts: TuningScheduler, model: nn.Module, mesh: DeviceMesh, device: torch.device,
          limit_train_batches: int | None = None, limit_val_batches: int | None = None) -> list[dict[str, Any]]:
    """Run the scheduled fine-tuning loop and return a record of each epoch."""
    dp_mesh, data_cfg = mesh["data_parallel"], cfg["data"]
    dataset = RandomTokenDataset(vocab_size=model.model.model_args.vocab_size, seq_length=data_cfg["seq_length"],
                                 dataset_length=data_cfg["dataset_length"])
    # shard the data across data-parallel ranks only: the ranks of a tensor-parallel group see the same batches
    sampler_kwargs = dict(num_replicas=dp_mesh.size(), rank=dp_mesh.get_local_rank())
    train_sampler = DistributedSampler(dataset, shuffle=True, seed=cfg["seed"], **sampler_kwargs)
    train_loader = DataLoader(dataset, batch_size=data_cfg["batch_size"], sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=data_cfg["batch_size"],
                            sampler=DistributedSampler(dataset, shuffle=False, **sampler_kwargs))
    use_loss_parallel = cfg["loss_parallel"] and mesh["tensor_parallel"].size() > 1

    def loss_on(batch: torch.Tensor) -> torch.Tensor:
        inputs, labels = batch[:, :-1].to(device), batch[:, 1:].to(device)
        output = model(inputs)
        return F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))

    history: list[dict[str, Any]] = []
    fts.setup()
    for epoch in fts.epochs(max_epochs=cfg["max_epochs"]):
        train_sampler.set_epoch(epoch)
        model.train()
        train_total, train_count = torch.zeros((), device=device), 0
        for i, batch in enumerate(train_loader):
            if limit_train_batches is not None and i >= limit_train_batches:
                break
            with loss_parallel() if use_loss_parallel else nullcontext():
                loss = loss_on(batch)
                loss.backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
            train_total, train_count = train_total + local_value(loss), train_count + 1
        model.eval()
        val_total, val_count = torch.zeros((), device=device), 0
        with torch.no_grad(), loss_parallel() if use_loss_parallel else nullcontext():
            for i, batch in enumerate(val_loader):
                if limit_val_batches is not None and i >= limit_val_batches:
                    break
                val_total, val_count = val_total + local_value(loss_on(batch)), val_count + 1
        record = {"epoch": epoch, "depth": fts.curr_depth,
                  "trainable_params": sum(p.requires_grad for p in model.parameters()),
                  "lrs": [group["lr"] for group in fts.optimizer.param_groups],
                  "train_loss": mean_across_ranks(train_total, train_count),
                  "val_loss": mean_across_ranks(val_total, val_count)}
        fts.epoch_end({"val_loss": record["val_loss"]})
        fts.step_lr_schedulers("epoch")
        history.append(record)
        if dist.get_rank() == 0:
            print(f"epoch {epoch} | phase {record['depth']} | trainable params {record['trainable_params']} | lr"
                  f" {max(record['lrs']):.2e} | train_loss {record['train_loss']:.4f} | val_loss"
                  f" {record['val_loss']:.4f}", flush=True)
    return history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="a configuration merged over the example's defaults")
    parser.add_argument("--max-epochs", dest="max_epochs", type=int)
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--limit-train-batches", dest="limit_train_batches", type=int)
    parser.add_argument("--limit-val-batches", dest="limit_val_batches", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args)
    dp, tp = cfg["mesh"]["data_parallel_size"], cfg["mesh"]["tensor_parallel_size"]
    if int(os.environ.get("WORLD_SIZE", 1)) != dp * tp:
        raise SystemExit(f"This configuration needs {dp * tp} processes: `torchrun --nproc_per_node={dp * tp} ...`.")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    try:
        if dist.get_rank() == 0:  # the tuning scheduler reports phase transitions through this logger, from rank zero
            fts_log = logging.getLogger("finetuning_scheduler")
            fts_log.setLevel(logging.INFO)
            fts_log.addHandler(logging.StreamHandler(sys.stdout))
        mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("data_parallel", "tensor_parallel"))
        torch.manual_seed(cfg["seed"])
        model = build_model(cfg, mesh, device)
        log_dir = os.path.join(cfg["output_dir"], cfg["run_name"])
        best = BestCheckpoint(os.path.join(log_dir, "checkpoints"), **cfg["checkpoint"])
        fts = build_tuning_scheduler(cfg, model, best, log_dir)
        history = train(cfg, fts, model, mesh, device, args.limit_train_batches, args.limit_val_batches)
        if dist.get_rank() == 0:
            summary = {"run_name": cfg["run_name"], "mesh": {"data_parallel": dp, "tensor_parallel": tp},
                       "epochs": history, "final_depth": fts.curr_depth,
                       "best_score": None if best.best_score is None else best.best_score.item(),
                       "best_depth": best.best_depth, "stop_reason": fts.stop_reason,
                       "restored_count": fts.restored_count,
                       "param_types": sorted({type(p).__name__ for p in model.parameters()}),
                       "lightning_modules": [m for m in LIGHTNING_MODULES if m in sys.modules]}
            with open(os.path.join(log_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"finished in phase {fts.curr_depth} | best val_loss {summary['best_score']:.4f} (phase"
                  f" {best.best_depth}) | best restored at {fts.restored_count} phase boundaries", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
