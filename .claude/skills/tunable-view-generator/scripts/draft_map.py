#!/usr/bin/env python
"""Draft a venngram component map for an architecture that has none, from its module tree.

A thin command line over venngram's drafter (``venngram.maps.drafter.draft_map``), which names modules by drafting
rules that are data: rules mined from every map venngram knows (so the drafter learns as maps are added), then the
bundled rule documents. The draft is a starting point, not an answer. Each row carries a confidence (the share of
known maps that agree on the name) and, where they disagreed, the alternatives; a module no rule names becomes a
``draft.<path>`` row, marked UNCONFIRMED. Review the least certain rows first, name or confirm each unconfirmed one,
verify the result (``find_view.py --map <draft>``), and repeat until verification passes. Confirm each guess against
the model's ``forward`` too: a name says what a module is called, not what it computes.

The model is built on the meta device (nothing is downloaded but a configuration), from a ``transformers`` architecture
with its default configuration (``--hf-arch``), from a Hugging Face model id (``--hf``, only its configuration is
fetched), or from a factory (``--model pkg.module:make_model`` or ``--model path/to/file.py:make_model``).
"""
import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from venngram.maps import known_architectures
from venngram.maps.drafter import draft_map, map_document


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


def _value(key: str, value) -> str:
    if key == "module":
        return f'"{value}"'
    return str(value).lower() if isinstance(value, bool) else str(value)


def render(draft) -> str:
    """The draft as a map document, each row annotated with how sure the drafter is of it."""
    props = draft.properties or {}
    confidence, alternatives = props.get("confidence", {}), props.get("alternatives", {})
    unconfirmed = set(props.get("unconfirmed", ()))
    doc = map_document(draft)
    lines = [f"# DRAFT component map for {draft.architecture}, from its module tree (drafting rules: "
             f"{', '.join(props.get('drafting_rules', ())) or 'default'}).",
             "# Review the least certain rows first, name or confirm every UNCONFIRMED row, then verify.",
             f"schema_version: {doc['schema_version']}", f"architecture: {doc['architecture']}"]
    if "kinds" in doc:
        lines.append(yaml.safe_dump({"kinds": doc["kinds"]}, sort_keys=False).rstrip())
    lines.append("components:")
    width = max(len(c) for c in doc["components"]) + 1
    for component, row in doc["components"].items():
        body = ", ".join(f"{k}: {_value(k, v)}" for k, v in row.items())
        note = ""
        if component in unconfirmed:
            note = "  # UNCONFIRMED: name its component (or confirm it has none)"
        elif component in confidence and confidence[component] < 1.0:
            alt = f"; alternatives: {', '.join(alternatives[component])}" if component in alternatives else ""
            note = f"  # confidence {confidence[component]:.2f}{alt}"
        lines.append(f"  {component + ':':<{width}} {{{body}}}{note}")
    for stack in props.get("unmapped_stacks", ()):
        lines.append(f"# UNMAPPED STACK: {stack} (map its rows by hand and declare it under `stacks:`)")
    return "\n".join(lines) + "\n"


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
    text = render(draft_map(model, arch))
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
