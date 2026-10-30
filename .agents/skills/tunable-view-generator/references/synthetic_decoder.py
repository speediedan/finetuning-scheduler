"""A small decoder that no component map covers: the skill's worked reference.

Its modules have deliberately unusual names (``trunk.cells.{i}.mix.wq``, ``feed.w_lift``, ``pre_feed``), so no bundled
or registered map can describe it, and a draft cannot place every module from names alone. ``synthetic_map.yaml`` is
its map, completed by hand from a draft and verified. The structure is a conventional pre-norm decoder: grouped-query
attention, a gated MLP, RMS norms.

Build it with ``make()``. ``CONFIG`` holds the widths verification reads (under the usual configuration names), and
``inputs(model)`` runs one forward for the order check.
"""
import torch
from torch import nn

CONFIG = {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
          "intermediate_size": 128, "num_hidden_layers": 2, "vocab_size": 256}


class Mixer(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.wq = nn.Linear(hidden, heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(heads * head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(b, t, self.heads * self.head_dim))


class Feed(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(hidden, intermediate, bias=False)
        self.w_lift = nn.Linear(hidden, intermediate, bias=False)
        self.w_drop = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_drop(nn.functional.silu(self.w_gate(x)) * self.w_lift(x))


class Cell(nn.Module):
    def __init__(self, c: dict) -> None:
        super().__init__()
        self.pre_mix = nn.RMSNorm(c["hidden_size"])
        self.mix = Mixer(c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"])
        self.pre_feed = nn.RMSNorm(c["hidden_size"])
        self.feed = Feed(c["hidden_size"], c["intermediate_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.pre_mix(x))
        return x + self.feed(self.pre_feed(x))


class Trunk(nn.Module):
    def __init__(self, c: dict) -> None:
        super().__init__()
        self.glyphs = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.cells = nn.ModuleList(Cell(c) for _ in range(c["num_hidden_layers"]))
        self.settle = nn.RMSNorm(c["hidden_size"])


class SyntheticDecoder(nn.Module):
    def __init__(self, c: dict = CONFIG) -> None:
        super().__init__()
        self.config = dict(c)
        self.trunk = Trunk(c)
        self.readout = nn.Linear(c["hidden_size"], c["vocab_size"], bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.trunk.glyphs(input_ids)
        for cell in self.trunk.cells:
            x = cell(x)
        return self.readout(self.trunk.settle(x))


def make() -> SyntheticDecoder:
    torch.manual_seed(0)
    return SyntheticDecoder()


def inputs(model: nn.Module):
    return model(torch.randint(0, CONFIG["vocab_size"], (1, 4)))
