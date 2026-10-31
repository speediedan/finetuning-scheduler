#!/usr/bin/env python
"""Find (or check) the view for a model: look up its component map, verify the map against the architecture, and say
what to do next.

With a known map (registered in the process, else the best one venngram's registries index: ``VENNGRAM_REGISTRIES``,
by default its seed registry), the whole view is ``MappedView(model, role_rules=...)`` and this prints that. With ``--map`` it verifies a map document instead, such as one ``draft_map.py`` wrote. Verification is
venngram's (``venngram.maps.verify``): every row's module exists, every parameter has exactly one component, widths
match the configuration, and components run in the expected order. It needs no weights: the model is built on the meta
device from a configuration (``--hf-arch`` uses an architecture's default configuration; ``--hf`` fetches only a
model's configuration). ``--model path/to/file.py:make`` verifies a model that is not a ``transformers`` one: its
module's ``CONFIG`` supplies the widths and its ``inputs(model)`` runs the forward the order check observes. Exits
non-zero unless verification passes (``partial`` counts as passing, with what it did not
cover printed). ``--record`` writes the verification record as JSON, for a trial log.
"""
import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from venngram.maps import known_architectures, load_component_map, verify


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-arch", help="a transformers architecture class, with its default configuration")
    src.add_argument("--hf", help="a Hugging Face model id (only its configuration is fetched)")
    src.add_argument("--model", help="pkg.module:factory or path/to/file.py:factory (with CONFIG and inputs beside it)")
    parser.add_argument("--map", help="a component map document to verify (default: the registered one)")
    parser.add_argument("--no-order", action="store_true", help="skip the forward-order check (no small model is run)")
    parser.add_argument("--record", help="write the verification record to this JSON file")
    args = parser.parse_args()
    extra = {}
    if args.model:
        module, factory = load_factory(args.model)
        target = factory()
        arch = type(target).__name__
        extra = {"config": getattr(module, "CONFIG", None), "inputs": getattr(module, "inputs", None)}
    elif args.hf:
        target, arch = args.hf, None
    else:
        import transformers

        cls = getattr(transformers, args.hf_arch)
        target, arch = cls.config_class(), args.hf_arch
    cmap = load_component_map(args.map) if args.map else None
    if cmap is None:
        from transformers import AutoConfig

        name = arch or (AutoConfig.from_pretrained(args.hf).architectures or ["?"])[0]
        if name not in known_architectures():
            print(f"no component map for {name}: draft one with draft_map.py, then verify it with --map.")
            return 2
    report = verify(target, cmap, architecture=arch, order=not args.no_order, **extra)
    print(report)
    if args.record:
        with open(args.record, "w") as f:
            json.dump(report.record(), f, indent=2, default=str)
    if not report.passed:
        print("\nverification failed: fix the rows named above and re-run.")
        return 1
    where = f"load_component_map({args.map!r}) registered with venngram.maps.register" if args.map else "found by venngram.maps.component_map_for"
    print(f"\nverified ({report.status}). The view is MappedView(model, role_rules=...); its map is {where}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
