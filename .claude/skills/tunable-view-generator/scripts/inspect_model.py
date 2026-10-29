#!/usr/bin/env python
"""Print what a tunable view for a model needs to know: its block list, one block's tensors, the tensors outside the
blocks, likely adapter tensors, and a first guess at each block module's architecture-neutral component.

The model comes from a factory (``--model pkg.module:make_model``, called with no arguments) or from a ``transformers``
architecture built from a small config (``--hf-arch LlamaForCausalLM``), randomly initialized so nothing is downloaded.
"""
import argparse
import importlib
import re
from collections import defaultdict

from torch import nn
from venngram.physical import block_lists

# module-name fragments mapped to a guess at the component; a guess only: confirm each against the model's forward
GUESSES = (
    (r"(^|\.)(q_proj|query|wq|to_q)$", "attn.q"), (r"(^|\.)(k_proj|key|wk|to_k)$", "attn.k"),
    (r"(^|\.)(v_proj|value|wv|to_v)$", "attn.v"), (r"(^|\.)(c_attn|qkv|query_key_value|Wqkv)$", "attn.qkv (fused)"),
    (r"(attn|attention)\.(o_proj|out_proj|c_proj|dense|wo|to_out)$", "attn.o"),
    (r"(^|\.)(gate_proj|w1)$", "mlp.gate"), (r"(^|\.)(up_proj|fc1|c_fc|w3|dense_h_to_4h)$", "mlp.up"),
    (r"mlp\.(down_proj|c_proj|w2|dense_4h_to_h)$|(^|\.)fc2$", "mlp.down"),
    (r"(input_layernorm|ln_1|self_attn_layer_norm|attention_norm|norm1)$", "norm.attn"),
    (r"(post_attention_layernorm|ln_2|final_layer_norm|ffn_norm|norm2)$", "norm.mlp"),
)
ADAPTER = re.compile(r"(^|\.)(lora_[AB]|lora_embedding_[AB]|adapter\w*|ia3_\w+|prefix\w*)(\.|$)")


def build(args) -> nn.Module:
    if args.model:
        module, _, attr = args.model.partition(":")
        return getattr(importlib.import_module(module), attr)()
    import transformers

    arch = getattr(transformers, args.hf_arch)
    defaults = arch.config_class()
    small = dict(vocab_size=128, hidden_size=32, n_embd=32, intermediate_size=64, ffn_dim=64, num_hidden_layers=2,
                 n_layer=2, num_attention_heads=4, n_head=4, num_key_value_heads=4, word_embed_proj_dim=32)
    return arch(arch.config_class(**{k: v for k, v in small.items() if hasattr(defaults, k)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="pkg.module:factory returning the model (with any adapters already added)")
    src.add_argument("--hf-arch", help="a transformers architecture, built from a small config")
    args = parser.parse_args()
    model = build(args)
    names = [(n, tuple(p.shape)) for n, p in model.named_parameters()]
    lists = block_lists(model)
    print(f"model: {type(model).__name__}, {len(names)} parameter tensors")
    print(f"block lists (top-level nn.ModuleList): {lists or 'none: `layers` needs a custom structure()'}")
    if len(lists) > 1:
        print("  several: the default structure refuses `layers` until one is chosen (`layers_pattern`)")
    prefix = lists[0][0] if len(lists) == 1 else None
    inside: dict[str, list] = defaultdict(list)
    outside = []
    for n, shape in names:
        m = re.match(rf"^{re.escape(prefix)}\.(\d+)\.(.*)$", n) if prefix else None
        (inside[m.group(2)].append(int(m.group(1))) if m else outside.append((n, shape)))
    if prefix:
        print(f"\none block's tensors (under {prefix}.<i>., present in blocks):")
        for rel, idx in inside.items():
            owner = rel.rpartition(".")[0]
            guess = next((c for rx, c in GUESSES if re.search(rx, owner)), "")
            span = "all" if len(idx) == lists[0][1] else sorted(idx)
            print(f"  {rel:<55} blocks={span}  {('-> ' + guess) if guess else ''}")
    print("\noutside the blocks:")
    for n, shape in outside:
        print(f"  {n} {shape}")
    adapters = [n for n, _ in names if ADAPTER.search(n)]
    print(f"\nlikely adapter tensors: {len(adapters)}" + (f" (e.g. {adapters[0]})" if adapters else ""))
    print("\nfused projections (attn.qkv) cannot be split by a view: name them as one component, or do not tune q/k/v"
          " separately.")


if __name__ == "__main__":
    main()
