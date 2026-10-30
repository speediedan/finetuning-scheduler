#!/usr/bin/env python
"""Check a tunable view: run the conformance suite, then compile a schedule through it and print each phase's tensors.

``--view pkg.module:make_view`` names a factory that returns a fresh view over a fresh model (the conformance cases
change parameters, so they need a new one per case). ``--schedule`` is a schedule YAML file (its ``addresses`` key, if
any, is resolved relative to the file). Exits non-zero if a conformance case fails or the schedule does not compile.
"""
import argparse
import importlib
import sys
import warnings
from pathlib import Path

from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.schedule import load_address_sets, load_yaml_schedule, split_addresses
from finetuning_scheduler.testing import TunableViewConformance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", required=True, help="pkg.module:factory returning a fresh view")
    parser.add_argument("--schedule", type=Path, help="a schedule YAML file to compile through the view")
    args = parser.parse_args()
    module, _, attr = args.view.partition(":")
    make_view = getattr(importlib.import_module(module), attr)
    report = TunableViewConformance(make_view).run()
    print(f"conformance: {'ok' if report.ok else 'FAILED'}")
    for case, failure in dict(report.failures).items():
        print(f"  {case}: {failure}")
    if not report.ok:
        return 1
    if args.schedule:
        phases, raw_sets = split_addresses(load_yaml_schedule(args.schedule))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*addresses.*experimental")
            compiled = compile_schedule(phases, make_view(),
                                        addresses=load_address_sets(raw_sets, args.schedule.parent))
        for phase in compiled.phases:
            print(f"phase {phase.depth}: {len(phase.physical)} tensors, {len(phase.coupled)} coupled")
            for name in phase.physical + phase.coupled:
                print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
