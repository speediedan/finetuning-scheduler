#!/usr/bin/env python3
"""Prune packages that are ONLY dependencies of torch from a lockfile.

This reduces the dependency confusion attack surface when using unsafe-best-match
by removing any packages that could potentially be resolved from the nightly index only.

How it works:
1. Parse the lockfile to find all packages and their dependents (the "# via" comments)
2. Identify packages where torch is the ONLY dependent (e.g., "# via torch")
3. Iteratively remove those packages and their exclusive transitive dependencies
4. Packages that are shared with other deps (e.g., "# via torch, transformers") are kept

This ensures that even if a malicious package were introduced to the nightly index,
it would only be resolved if it's also a dependency of other packages (which would
mean it exists on PyPI and would be caught by normal security scanning).

Usage:
    python prune_torch_deps.py <lockfile_path>

Example:
    python prune_torch_deps.py requirements/ci/requirements.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_lockfile(content: str) -> dict[str, dict]:
    """Parse a uv lockfile and extract package blocks with their dependencies.

    Args:
        content: The lockfile content as a string.

    Returns:
        A dict mapping package names to their block info:
        {
            "package_name": {
                "lines": ["line1", "line2", ...],  # All lines in this block
                "dependents": ["dep1", "dep2", ...],  # Packages that depend on this
            }
        }
    """
    packages: dict[str, dict] = {}
    lines = content.splitlines()

    current_pkg: str | None = None
    current_lines: list[str] = []
    current_dependents: list[str] = []
    in_via_section = False

    # Regex to match package line: starts with letter, contains ==
    pkg_pattern = re.compile(r"^([a-zA-Z][a-zA-Z0-9_-]*)")

    for line in lines:
        # Check if this is a new package line (starts with letter, not indented)
        if line and line[0].isalpha():
            # Save previous package if exists
            if current_pkg is not None:
                if current_pkg not in packages:
                    packages[current_pkg] = {"lines": [], "dependents": set()}
                packages[current_pkg]["lines"].extend(current_lines)
                packages[current_pkg]["dependents"].update(current_dependents)

            # Extract package name (before == or space)
            match = pkg_pattern.match(line)
            if match:
                current_pkg = match.group(1).lower()  # Normalize to lowercase
            else:
                current_pkg = None

            current_lines = [line]
            current_dependents = []
            in_via_section = False

        elif current_pkg is not None:
            current_lines.append(line)

            # Parse "# via" comments to find dependents
            # Single-line format: "    # via package_name"
            if line.strip().startswith("# via ") and not line.strip() == "# via":
                dep = line.strip()[6:].strip()  # Remove "# via "
                # Handle potential trailing content (markers, etc.)
                dep = dep.split()[0] if dep else ""
                # Normalize: replace hyphens with underscores for comparison
                if dep and not dep.startswith("("):  # Skip "(pyproject.toml)" style
                    current_dependents.append(dep.lower().replace("-", "_"))
                in_via_section = False

            # Multi-line format start: "    # via"
            elif line.strip() == "# via":
                in_via_section = True

            # Multi-line format entries: "    #   package_name"
            elif in_via_section and line.strip().startswith("#   "):
                dep = line.strip()[4:].strip()  # Remove "#   "
                dep = dep.split()[0] if dep else ""
                if dep and not dep.startswith("("):
                    current_dependents.append(dep.lower().replace("-", "_"))

            # End of via section (any non-comment indented line or empty)
            elif in_via_section and (not line.strip().startswith("#") or line.strip() == ""):
                in_via_section = False

    # Don't forget the last package
    if current_pkg is not None:
        if current_pkg not in packages:
            packages[current_pkg] = {"lines": [], "dependents": set()}
        packages[current_pkg]["lines"].extend(current_lines)
        packages[current_pkg]["dependents"].update(current_dependents)

    # Convert dependent sets to lists
    for pkg_info in packages.values():
        pkg_info["dependents"] = list(pkg_info["dependents"])

    return packages


def find_torch_only_packages(packages: dict[str, dict]) -> set[str]:
    """Find packages that are ONLY dependencies of torch (and its exclusive deps).

    Uses iterative approach to handle transitive dependencies:
    - First pass: find packages where only "torch" is in dependents
    - Subsequent passes: find packages where only already-pruned packages are dependents

    Args:
        packages: Dict from parse_lockfile()

    Returns:
        Set of package names to prune
    """
    pruned: set[str] = set()
    max_iterations = 10  # Safety limit

    for _ in range(max_iterations):
        newly_pruned = set()

        for pkg_name, pkg_info in packages.items():
            if pkg_name in pruned:
                continue

            dependents = pkg_info["dependents"]
            if not dependents:
                continue

            # Check if all dependents are either "torch" or already pruned
            # For first-level deps: package is only required by "torch"
            # For transitive deps: package is only required by already-pruned packages
            all_dependents_prunable = all(
                dep == "torch" or dep.replace("-", "_") in pruned for dep in dependents
            )

            if all_dependents_prunable:
                newly_pruned.add(pkg_name)

        if not newly_pruned:
            break

        pruned.update(newly_pruned)

    return pruned


def prune_packages(content: str, packages_to_prune: set[str]) -> str:
    """Remove specified packages from the lockfile content.

    Args:
        content: Original lockfile content
        packages_to_prune: Set of package names to remove

    Returns:
        Updated lockfile content with packages removed
    """
    lines = content.splitlines()
    result_lines: list[str] = []
    skip_block = False

    # Normalize package names for comparison
    normalized_prune = {pkg.lower().replace("-", "_") for pkg in packages_to_prune}

    pkg_pattern = re.compile(r"^([a-zA-Z][a-zA-Z0-9_-]*)")

    for line in lines:
        # Check if this is a new package line
        if line and line[0].isalpha():
            match = pkg_pattern.match(line)
            if match:
                pkg_name = match.group(1).lower().replace("-", "_")
                skip_block = pkg_name in normalized_prune
            else:
                skip_block = False

        if not skip_block:
            result_lines.append(line)

    return "\n".join(result_lines)


def prune_torch_only_deps(lockfile_path: str) -> list[str]:
    """Main function to prune torch-only dependencies from a lockfile.

    Args:
        lockfile_path: Path to the lockfile to process

    Returns:
        List of pruned package names
    """
    path = Path(lockfile_path)
    content = path.read_text()

    # Parse the lockfile
    packages = parse_lockfile(content)

    # Find packages to prune
    to_prune = find_torch_only_packages(packages)

    if not to_prune:
        return []

    # Prune and write back
    new_content = prune_packages(content, to_prune)
    path.write_text(new_content)

    return sorted(to_prune)


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <lockfile_path>", file=sys.stderr)
        return 1

    lockfile_path = sys.argv[1]

    if not Path(lockfile_path).exists():
        print(f"Error: File not found: {lockfile_path}", file=sys.stderr)
        return 1

    print(f"  Post-processing: pruning torch-only dependencies from {lockfile_path}...")

    pruned = prune_torch_only_deps(lockfile_path)

    if pruned:
        print("  Pruned torch-only dependencies:")
        for pkg in pruned:
            print(f"    - {pkg}")
        print("  Lockfile updated with torch-only dependencies removed")
    else:
        print("  No torch-only dependencies found to prune")

    return 0


if __name__ == "__main__":
    sys.exit(main())
