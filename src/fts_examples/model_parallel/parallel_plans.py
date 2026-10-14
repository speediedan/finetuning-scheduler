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

import torch
from torch.distributed.device_mesh import DeviceMesh
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
