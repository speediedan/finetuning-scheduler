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
"""Parallelism plans and data shared by the model-parallel examples.

Nothing here imports Lightning, so both the Lightning example (``mp_examples.py``) and the plain PyTorch example
(``torch_mp_example.py``) use these helpers.
"""
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.distributed._composable import checkpoint
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel, parallelize_module, PrepareModuleInput,
                                               RowwiseParallel, SequenceParallel)
from torch.utils.data import Dataset

from fts_examples.model_parallel.torchtitan_llama import Transformer


# modified version of https://bit.ly/torchtitan_transformer_tp_plan
def apply_tp_plan(model: Transformer, device_mesh: DeviceMesh, loss_parallel: bool) -> Transformer:
    """Apply a tensor-parallel plan to the ``torchtitan`` Llama ``Transformer``.

    Args:
        model: The model to parallelize (on the meta device or materialized).
        device_mesh: A device mesh with a ``"tensor_parallel"`` dimension.
        loss_parallel: Keep the output layer's logits sharded on the vocabulary dimension, for use with
            ``torch.distributed.tensor.parallel.loss_parallel``.

    Returns:
        The parallelized model.
    """
    tp_mesh = device_mesh["tensor_parallel"]

    # 1. Parallelize the pre-transformer block layers
    non_transformerblock_tp_plan = {
        # Parallelize the embedding and shard its outputs (which are the first transformer block's inputs)
        "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1),),
        "norm": SequenceParallel(),  # Parallelize the root norm layer over the sequence dim
    }
    model = parallelize_module(model, tp_mesh, non_transformerblock_tp_plan)

    # 2. Parallelize each transformer block
    for transformer_block in model.layers.values():
        plan = {
            "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
            ),
            "attention_norm": SequenceParallel(),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),

        }

        parallelize_module(transformer_block, tp_mesh, plan)

    # 3. Parallelize the output layer
    output_parallelize_plan = (
        ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        )
    )
    parallelize_module(model.output, tp_mesh, output_parallelize_plan)

    return model


def apply_fsdp_plan(module: nn.Module, mesh: DeviceMesh,
                    fsdp_plan: Mapping[str, Mapping[str, Any] | None] | None = None,
                    fsdp_default_kwargs: Mapping[str, Any] | None = None, shard_root: bool = True) -> nn.Module:
    """Apply ``fully_shard`` to the modules an FSDP2 plan names, then to ``module`` itself.

    The plan has the format of the Lightning example's ``fsdp_plan`` (see
    :attr:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter.fsdp_plan`): its keys are module names
    or regular expressions matched from the start of each module name, and its values are ``fully_shard`` keyword
    arguments, merged over ``fsdp_default_kwargs``. Two of the adapter's aliases are supported: ``act_ckpt`` applies
    composable activation checkpointing to a module before it is sharded (the ``composable`` mode only), and
    ``cpu_offload_policy`` becomes an ``offload_policy``.

    Modules are sharded in plan order, so name children before their parents. With ``shard_root``, ``module`` is
    sharded last, so every parameter becomes a ``DTensor``. A parameter left outside every ``fully_shard`` call would
    be neither sharded nor gradient-synchronized across data-parallel ranks, and some optimizer implementations do not
    support parameter groups that mix ``DTensor`` and plain parameters.

    Args:
        module: The module to shard (on the meta device or materialized), after any tensor-parallel plan.
        mesh: The data-parallel device mesh.
        fsdp_plan: Module names or patterns mapped to ``fully_shard`` keyword arguments (or ``None``).
        fsdp_default_kwargs: Keyword arguments applied to every ``fully_shard`` call the plan makes. Every one but
            ``act_ckpt`` also applies to the root.
        shard_root: Shard ``module`` itself after the planned modules.

    Returns:
        The sharded module.

    Raises:
        ValueError: If a plan key matches no module, or ``act_ckpt`` names a mode other than ``composable``.
    """
    defaults = dict(fsdp_default_kwargs or {})
    names = [n for n, _ in module.named_modules()]
    resolved: dict[str, dict[str, Any]] = {}
    for key, kwargs in (fsdp_plan or {}).items():
        matches = [key] if key in names else [n for n in names if re.match(key, n)]
        if not matches:
            raise ValueError(f"The module or pattern '{key}' in `fsdp_plan` did not match any module.")
        for n in matches:
            # containers without a `forward` cannot be sharded on their own, as with the Lightning adapter
            if not isinstance(module.get_submodule(n), (nn.ModuleList, nn.ModuleDict)):
                resolved.setdefault(n, {**defaults, **(kwargs or {})})
    for n, kwargs in resolved.items():
        _shard(module.get_submodule(n), mesh, kwargs)
    if shard_root:
        _shard(module, mesh, {k: v for k, v in defaults.items() if k != "act_ckpt"})
    return module


def _shard(module: nn.Module, mesh: DeviceMesh, kwargs: Mapping[str, Any]) -> None:
    kwargs = dict(kwargs)
    act_ckpt = kwargs.pop("act_ckpt", None)
    if act_ckpt is not None:
        mode, ac_kwargs = (act_ckpt, None) if isinstance(act_ckpt, str) else (act_ckpt[0], _second(act_ckpt))
        if mode != "composable":
            raise ValueError(f"`act_ckpt` mode '{mode}' is not supported here; use 'composable'.")
        checkpoint(module, **dict(ac_kwargs or {}))
    offload = kwargs.pop("cpu_offload_policy", None)
    if offload is not None:
        kwargs["offload_policy"] = CPUOffloadPolicy(**offload)
    fully_shard(module, mesh=mesh, **kwargs)


def _second(seq: Sequence[Any]) -> Any:
    return seq[1] if len(seq) > 1 else None


class RandomTokenDataset(Dataset):
    """Random token sequences, the same on every rank (a fixed seed), for the model-parallel examples.

    Each item holds ``seq_length + 1`` tokens: the inputs are the first ``seq_length`` and the targets the last
    ``seq_length``. A data-parallel sampler shards the dataset across ranks.

    Args:
        vocab_size: The vocabulary size tokens are drawn from.
        seq_length: The input sequence length.
        dataset_length: The number of sequences.
    """

    def __init__(self, vocab_size: int, seq_length: int, dataset_length: int = 64):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.dataset_length = dataset_length
        self.tokens = torch.randint(
            self.vocab_size,
            size=(len(self), self.seq_length + 1),
            # Set a seed to make this toy dataset the same on each rank
            generator=torch.Generator().manual_seed(42),
        )

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, item: int):
        return self.tokens[item]
