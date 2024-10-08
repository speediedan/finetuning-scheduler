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
# Initially based on https://bit.ly/3oQ8Vqf
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from contextlib import ExitStack, contextmanager

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.demos.transformer import WikiText2
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributed.tensor.experimental import implicit_replication
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from tests import _PATH_DATASETS

class LinearWarmupLR(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        super().__init__(optimizer, lr_lambda, last_epoch)


class CustomLRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, epoch):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state_dict):
        ...


class RandomDictDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        a = self.data[index]
        b = a + 2
        return {"a": a, "b": b}

    def __len__(self):
        return self.len


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class RandomIterableDataset(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(self.count):
            yield torch.randn(self.size)


class RandomIterableDatasetWithLen(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self):
        return self.count


class BoringModel(LightningModule):
    def __init__(self):
        """Testing PL Module.

        Use as follows:
        - subclass
        - modify the behavior for what you want

        class TestModel(BaseTestModel):
            def training_step(...):
                # do your own thing

        or:

        model = BaseTestModel()
        model.training_step_end = None  # disable hook
        """
        super().__init__()
        self.model = torch.nn.Linear(32, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss(self, preds: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if labels is None:
            labels = torch.ones_like(preds)
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(preds, labels)

    def step(self, batch: Tensor) -> Tensor:
        output = self(batch)
        return self.loss(output)

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        return {"loss": self.step(batch)}


    def training_step_end(self, training_step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        return training_step_output

    def validation_step(self, batch: Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return {"x": self.step(batch)}


    def test_step(self, batch: Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return {"y": self.step(batch)}

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[LRScheduler]]:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class BoringDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.non_picklable = None
        self.checkpoint_state: Optional[str] = None
        self.random_full = RandomDataset(32, 64 * 4)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.random_train = Subset(self.random_full, indices=range(64))

        if stage in ("fit", "validate") or stage is None:
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))

        if stage == "test" or stage is None:
            self.random_test = Subset(self.random_full, indices=range(64 * 2, 64 * 3))

        if stage == "predict" or stage is None:
            self.random_predict = Subset(self.random_full, indices=range(64 * 3, 64 * 4))

    def train_dataloader(self):
        return DataLoader(self.random_train)

    def val_dataloader(self):
        return DataLoader(self.random_val)

    def test_dataloader(self):
        return DataLoader(self.random_test)

    def predict_dataloader(self):
        return DataLoader(self.random_predict)


class ManualOptimBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        opt = self.optimizers()
        assert isinstance(opt, (Optimizer, LightningOptimizer))
        loss = self.step(batch)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss

class FTSWikiText2(WikiText2):
    """Subclass Lightning's Mini version of WikiText2 with a few customizations."""

    # TODO: avoid this subclass if we don't modify anything other than default path
    def __init__(self, data_dir: Path = Path(_PATH_DATASETS), block_size: int = 32, *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, block_size=block_size, *args, **kwargs)


################################################################################
# Toy Configurable Transformer (non-TransformerLens)
# A toy configurable transformer originally based on
# https://bit.ly/toy_transformer. The intention is to ensure we test a more
# heterogenous set of toy configurable transformer implementations
################################################################################


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
    avail_sdp_backends: Optional[Union[List[SDPBackend], SDPBackend]] = None
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
        #self.device = args.device
        #self.dtype = args.dtype
        #self.tokenizer = args.tokenizer
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

    def forward(self, tokens, mask: Optional[torch.Tensor] = None):
        # we assume target is already shifted w.r.t. inputs
        # _, t = tokens.shape
        # if mask is None:
        #     mask = torch.tril(torch.ones(t, t, device=tokens.device)) == 1
        #     mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
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

    # @torch.inference_mode()
    # def generate(
    #     self,
    #     tokens: Union[str, torch.Tensor] = "",
    #     max_new_tokens: int = 5,
    #     eos_token_id: Optional[int] = None,
    #     output_logits: bool = False,
    #     verbose: bool = True,
    #     **kwargs
    # ) -> Union[SampledOutput, torch.Tensor]:
    #     """Toy generate function to support non-HF/TransformerLens tests with the same interface.

    #     Args:
    #         tokens (Union[str, Int[torch.Tensor, "batch pos"])]): A batch of tokens ([batch, pos]).
    #         max_new_tokens (int): Maximum number of tokens to generate.
    #         eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end of sentence.
    #         output_logits (`bool`, *optional*, defaults to `False`): Whether or not to return the prediction scores.
    #         verbose (bool): If True, show tqdm progress bars for generation.

    #     Returns:
    #         outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens.
    #     """
    #     # To enable a broader range of testing contexts, use the configuration context of the parent_handle
    #     # TODO: update this method to use parent_handle if available for broader range of testing
    #     out_logits = () if output_logits else None

    #     assert isinstance(tokens, torch.Tensor)
    #     batch_size, ctx_length = tokens.shape
    #     gen_device = self.device or tokens.device
    #     tokens = tokens.to(gen_device)

    #     stop_tokens = []
    #     eos_token_for_padding = 0

    #     tokenizer_has_eos_token = (self.tokenizer is not None and self.tokenizer.eos_token_id is not None)
    #     if eos_token_id is None:
    #         assert (tokenizer_has_eos_token), \
    #         "Must pass an `eos_token_id` if tokenizer is None or has no eos_token_id"
    #         eos_token_id = self.tokenizer.eos_token_id

    #     stop_tokens = [eos_token_id]
    #     eos_token_for_padding = eos_token_id

    #     # An array to track which sequences in the batch have finished.
    #     finished_sequences = torch.zeros(
    #         batch_size, dtype=torch.bool, device=gen_device
    #     )

    #     for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
    #         # While generating, we keep generating logits, throw away all but the final logits,
    #         # and then use those logits to sample from the distribution We keep adding the
    #         # sampled tokens to the end of tokens.
    #         # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
    #         # the cache.
    #         logits = self.forward(tokens)
    #         final_logits = logits[:, -1, :]
    #         if output_logits:
    #             out_logits += (final_logits,)

    #         sampled_tokens = final_logits.argmax(-1).to(gen_device)

    #         # For all unfinished sequences, add on the next token. If a sequence was
    #         # finished, throw away the generated token and add eos_token_for_padding
    #         # instead.
    #         sampled_tokens[finished_sequences] = eos_token_for_padding
    #         finished_sequences.logical_or_(
    #             torch.isin(sampled_tokens, torch.tensor(stop_tokens).to(gen_device))
    #         )

    #         tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

    #         if finished_sequences.all():
    #             break

    #     if output_logits:
    #         return SampledOutput(tokens, torch.stack(out_logits, dim=1))
    #     else:
    #         return tokens
