#!/usr/bin/env python
"""Script to toggle between standalone and unified Lightning imports.

This script allows for easily switching between unified imports (lightning.pytorch) and standalone imports
(pytorch_lightning) throughout the codebase.
"""

import argparse
from finetuning_scheduler.dynamic_versioning.utils import toggle_lightning_imports


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Toggle between standalone and unified Lightning imports.")
    parser.add_argument(
        "--mode", choices=["unified", "standalone"], default="unified",
        help="Import style to convert to: 'unified' (lightning.pytorch) or 'standalone' (pytorch_lightning).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output.")
    return parser.parse_args()


def main() -> int:
    """Execute the main script functionality."""
    args = parse_args()
    try:
        toggle_lightning_imports(args.mode, debug=args.debug)
        return 0
    except Exception as e:
        print(f"Error toggling imports: {e}")
        return 1


if __name__ == "__main__":
    main()  # pragma: no cover
