#!/usr/bin/env python
"""Draft a venngram component map for an architecture that has none, from its module tree.

The draft is a starting point, not an answer: every row is a guess from module names and types, and the rows it could
not place are listed as ``# UNMAPPED`` comments. Verify it (``find_view.py --map <draft>``), read the problems, fix the
rows, and repeat until verification passes. Confirm each guess against the model's ``forward`` too: a name says what a
module is called, not what it computes.

The model is built on the meta device (nothing is downloaded but a configuration), from a ``transformers`` architecture
with its default configuration (``--hf-arch``), from a Hugging Face model id (``--hf``, only its configuration is
fetched), or from a factory (``--model pkg.module:make_model`` or ``--model path/to/file.py:make_model``).
"""
import argparse
import importlib
import importlib.util
import re
import sys
from pathlib import Path

import torch
from torch import nn
from venngram.maps import known_architectures
from venngram.physical import block_lists

# (pattern over the module path relative to the block, component); first match wins. Names are guesses: confirm each.
BLOCK_GUESSES = (
    (r"(^|\.)(self_attn|attention|attn)$", "attn"),
    (r"(^|\.)(q_proj|query|wq|to_q)$", "attn.q"), (r"(^|\.)(k_proj|key|wk|to_k)$", "attn.k"),
    (r"(^|\.)(v_proj|value|wv|to_v)$", "attn.v"),
    (r"(^|\.)(qkv_proj|c_attn|qkv|query_key_value|Wqkv|in_proj)$", "attn.qkv"),
    (r"(attn|attention)\.(o_proj|out_proj|c_proj|dense|wo|to_out)$", "attn.o"),
    (r"(^|\.)q_norm$", "attn.q_norm"), (r"(^|\.)k_norm$", "attn.k_norm"),
    (r"(^|\.)(mlp|feed_forward|ffn)$", "mlp"),
    (r"(^|\.)(gate_proj|w1)$", "mlp.gate"),
    (r"(^|\.)gate_up_proj$", "mlp.gate_in"),
    (r"(^|\.)(up_proj|fc1|c_fc|w3|dense_h_to_4h)$", "mlp.in"),
    (r"(mlp|feed_forward|ffn)\.(down_proj|c_proj|w2|dense_4h_to_h|fc2)$|(^|\.)fc2$", "mlp.out"),
    (r"(^|\.)(input_layernorm|ln_1|self_attn_layer_norm|attention_norm|norm1|pre_attention_layernorm)$", "ln1"),
    (r"(^|\.)(post_attention_layernorm|ln_2|final_layer_norm|ffn_norm|norm2|pre_feedforward_layernorm)$", "ln2"),
    (r"(^|\.)post_feedforward_layernorm$", "ln2_post"),
)
KIND_BY_COMPONENT = {"attn": "attn", "mlp": "mlp", "ln1": "norm", "ln2": "norm", "ln2_post": "norm",
                     "attn.q_norm": "norm", "attn.k_norm": "norm"}


def load_factory(spec: str):
    """``pkg.module:attr`` or ``path/to/file.py:attr``: the module (imported or loaded from the file) and the attribute."""
    where, _, attr = spec.partition(":")
    if where.endswith(".py"):
        loader = importlib.util.spec_from_file_location(Path(where).stem, where)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
    else:
        module = importlib.import_module(where)
    return module, getattr(module, attr)


def build(args) -> nn.Module:
    if args.model:
        return load_factory(args.model)[1]()
    import transformers

    if args.hf:
        config = transformers.AutoConfig.from_pretrained(args.hf)
        arch = getattr(transformers, (config.architectures or [args.hf_arch])[0])
    else:
        arch = getattr(transformers, args.hf_arch)
        config = arch.config_class()
    with torch.device("meta"):
        return arch(config)


def is_norm(module: nn.Module) -> bool:
    return "norm" in type(module).__name__.lower()


def kind_of(component: str, module: nn.Module) -> str:
    return KIND_BY_COMPONENT.get(component, "norm" if is_norm(module) else "linear")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-arch", help="a transformers architecture class, built from its default configuration")
    src.add_argument("--hf", help="a Hugging Face model id (only its configuration is fetched)")
    src.add_argument("--model", help="pkg.module:factory or path/to/file.py:factory returning the model")
    parser.add_argument("-o", "--output", help="write the draft here (default: stdout)")
    args = parser.parse_args()
    model = build(args)
    arch = type(model).__name__
    if arch in known_architectures():
        print(f"{arch} already has a map: use it (find_view.py), or pass --map to override it.", file=sys.stderr)
    lists = block_lists(model)
    if len(lists) != 1:
        print(f"{arch}: {len(lists)} top-level block lists ({[p for p, _ in lists]}); draft the primary stack's rows by"
              " hand and declare the others under `stacks:`.", file=sys.stderr)
        return 1
    prefix = lists[0][0]
    block = model.get_submodule(f"{prefix}.0")
    lines = [f"# DRAFT component map for {arch}, from module names: verify it, then fix and re-verify.",
             "schema_version: 2", f"architecture: {arch}", "components:"]
    unmapped = []
    rows = []
    for top, module in model.named_children():
        for path, sub in module.named_modules(prefix=top):
            if path == prefix or path.startswith(prefix + "."):
                continue
            if isinstance(sub, nn.Embedding) and "embed" not in [c for c, _, _ in rows]:
                rows.append(("embed", path, "embed"))
            elif is_norm(sub) and next(sub.parameters(recurse=False), None) is not None and path.count(".") <= 1:
                rows.append(("ln_final", path, "norm"))
    if isinstance(getattr(model, "lm_head", None), nn.Module):
        rows.append(("unembed", "lm_head", "unembed"))
    rows.insert(next((i for i, r in enumerate(rows) if r[0] != "embed"), len(rows)), ("blocks.{i}", f"{prefix}.{{i}}",
                                                                                     "block"))
    placed = set()
    for rel, sub in block.named_modules():
        if not rel:
            continue
        guess = next((c for rx, c in BLOCK_GUESSES if re.search(rx, rel)), None)
        if guess is not None and f"blocks.{{i}}.{guess}" not in [r[0] for r in rows]:
            rows.append((f"blocks.{{i}}.{guess}", f"{prefix}.{{i}}.{rel}", kind_of(guess, sub)))
            placed.add(rel)
        elif next(sub.parameters(recurse=False), None) is not None and not any(rel.startswith(p + ".") or rel == p
                                                                               for p in placed):
            unmapped.append(rel)
    width = max(len(c) for c, _, _ in rows) + 1
    for component, module_path, kind in rows:
        lines.append(f"  {component + ':':<{width}} {{module: \"{module_path}\", kind: {kind}}}")
    for rel in unmapped:
        lines.append(f"  # UNMAPPED: {prefix}.{{i}}.{rel} (holds parameters; name its component or confirm it has none)")
    text = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
