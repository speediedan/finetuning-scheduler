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
"""Lightning-free model-parallel test models: a toy transformer and its tensor-parallel plan.

Shared by the Lightning model-parallel tests and the pool cases, which may not import Lightning.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.experimental import implicit_replication
from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel,
                                                SequenceParallel, parallelize_module)
from torch.nn.attention import SDPBackend, sdpa_kernel


# A toy configurable transformer originally based on https://bit.ly/toy_transformer. The intention is to ensure we test
# a more heterogenous set of toy configurable transformer implementations.
@dataclass
class TestModelArgs:
    n_layers: int = 2  # 2
    vocab_size: int = 33278  # 33
    max_seq_len: int = 192  # 10
    dim: int = 192  # 10
    n_heads: int = 2
    dropout_p: float = 0.0  # 0.2  # 0.1
    use_attn_mask: bool = True
    weight_tying: bool = False  # True
    checkpoint_activations: bool = False
    avail_sdp_backends: list[SDPBackend] | SDPBackend | None = None
    sdp_use_implicit_replication: bool = False


class Attention(torch.nn.Module):
    def __init__(self, args: TestModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = torch.nn.Dropout(args.dropout_p)
        self.use_attn_mask = args.use_attn_mask
        self.avail_sdp_backends = args.avail_sdp_backends
        self.sdp_use_implicit_replication = args.sdp_use_implicit_replication

        self.wq = torch.nn.Linear(args.dim, args.dim, bias=False)
        self.wk = torch.nn.Linear(args.dim, args.dim, bias=False)
        self.wv = torch.nn.Linear(args.dim, args.dim, bias=False)
        self.wo = torch.nn.Linear(args.dim, args.dim, bias=False)

    @staticmethod
    @contextmanager
    def sdpa_ctx(avail_sdp_backends, sdp_use_implicit_replication):
        if avail_sdp_backends is None and not sdp_use_implicit_replication:
            yield
        else:
            ctx_managers = []
            if avail_sdp_backends is not None:
                ctx_managers.append(sdpa_kernel(avail_sdp_backends))
            if sdp_use_implicit_replication:
                ctx_managers.append(implicit_replication())
            with ExitStack() as stack:
                for ctx_manager in ctx_managers:
                    stack.enter_context(ctx_manager)
                try:
                    yield
                finally:
                    stack.close()

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        sdpa_args = (queries, keys, values, None, self.dropout_p if self.training else 0, self.use_attn_mask)
        with Attention.sdpa_ctx(self.avail_sdp_backends, self.sdp_use_implicit_replication):
            output = torch.nn.functional.scaled_dot_product_attention(*sdpa_args)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim)
        self.gelu = torch.nn.GELU()
        self.w2 = torch.nn.Linear(hidden_dim, dim)
        self.resid_dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))

class TransformerBlock(torch.nn.Module):
    def __init__(self, args: TestModelArgs):
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = torch.nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class FTSToyTransformer(torch.nn.Module):
    def __init__(self, args: TestModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.max_seq_len is not None
        self.init_args = args
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = torch.nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = torch.nn.Dropout(args.dropout_p)
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.norm = torch.nn.LayerNorm(args.dim)
        self.output = torch.nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight
        self.checkpoint_activations = args.checkpoint_activations
        self.avail_sdp_backends = args.avail_sdp_backends
        self.sdp_use_implicit_replication = args.sdp_use_implicit_replication

    def forward(self, tokens, mask: torch.Tensor | None = None):
        _bsz, seq_len = tokens.size()
        assert seq_len <= self.max_seq_len
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)  # positional embeddings of shape (seq_len, dim)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            if self.checkpoint_activations:
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        h = self.norm(h)
        output = self.output(h)
        return output


# modified version of https://bit.ly/torchtitan_transformer_tp_plan
def gen_apply_transformer_tp_plan(model: nn.Module, device_mesh: DeviceMesh, loss_parallel: bool) -> nn.Module:
    """Apply tensor parallelism."""

    # we're only applying tensor parallelism, composable fsdp is applied subsequently elsewhere if requested
    tp_mesh = device_mesh["tensor_parallel"]

    # 1. Parallelize the embedding and shard its outputs
    # 2. Parallelize the root norm layer over the sequence dim
    non_transformerblock_tp_plan = {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1),),
            "pos_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(0)),
            "norm": SequenceParallel(),
        }

    model = parallelize_module(model, tp_mesh, non_transformerblock_tp_plan)

    # N.B. We're not supporting Float8 parallelism in this test initially (e.g. Float8RowwiseParallel)

    # Apply tensor + sequence parallelism to every transformer block
    for transformer_block in model.layers:
        layer_plan = {
            "attention": PrepareModuleInput(
                input_layouts=Shard(1),
                desired_input_layouts=Replicate(),
            ),
            "attention_norm": SequenceParallel(),
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward.w1": ColwiseParallel(input_layouts=Shard(1)),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    output_parallelize_plan = (
        ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        )
    )
    parallelize_module(model.output, tp_mesh, output_parallelize_plan)

    # Manually set output.weight so that parameters and gradients are shared.
    if model.init_args.weight_tying:
        model.output.weight = model.tok_embeddings.weight

    return model
