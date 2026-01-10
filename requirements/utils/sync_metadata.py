#!/usr/bin/env python3
"""Sync minimum version metadata from utils.py to pyproject.toml.

This script reads the actual version constraints from
src/finetuning_scheduler/dynamic_versioning/utils.py and updates
the informational [tool.fts.min-versions] section in pyproject.toml
to match.

This ensures the user-facing metadata stays synchronized with the
actual version constraints used during installation.
"""

import re
import sys
from pathlib import Path


def extract_torch_version(utils_content: str) -> str:
    """Extract torch version constraint from BASE_DEPENDENCIES.

    Args:
        utils_content: Content of utils.py

    Returns:
        Full torch version constraint (e.g., ">=2.6.0")
    """
    # Look for: BASE_DEPENDENCIES = [..., "torch>=X.Y.Z", ...]
    match = re.search(r'BASE_DEPENDENCIES\s*=\s*\[.*?"torch(>=?[0-9.]+(?:,[<>=0-9.]+)?)"', utils_content, re.DOTALL)
    if not match:
        raise ValueError("Could not find torch version in BASE_DEPENDENCIES")
    return match.group(1)


def extract_lightning_version(utils_content: str) -> str:
    """Extract lightning version constraint from LIGHTNING_VERSION.

    Args:
        utils_content: Content of utils.py

    Returns:
        Full lightning version constraint (e.g., ">=2.6.0,<2.6.1")
    """
    # Look for: LIGHTNING_VERSION = ">=X.Y.Z,<X.Y.Z+1"
    match = re.search(r'LIGHTNING_VERSION\s*=\s*"(>=?[0-9.]+(?:,[<>=0-9.]+)?)"', utils_content)
    if not match:
        raise ValueError("Could not find LIGHTNING_VERSION")
    return match.group(1)


def extract_python_version(pyproject_content: str) -> str:
    """Extract minimum Python version from pyproject.toml requires-python.

    Args:
        pyproject_content: Content of pyproject.toml

    Returns:
        Minimum python version (e.g., "3.10")
    """
    # Look for: requires-python = ">=X.Y"
    match = re.search(r'requires-python\s*=\s*">=([0-9.]+)"', pyproject_content)
    if not match:
        raise ValueError("Could not find requires-python in pyproject.toml")
    return match.group(1)


def update_metadata_section(pyproject_content: str, torch_ver: str, lightning_ver: str, python_ver: str) -> str:
    """Update the [tool.fts.min-versions] section with new versions.

    Args:
        pyproject_content: Content of pyproject.toml
        torch_ver: Minimum torch version
        lightning_ver: Minimum lightning version
        python_ver: Minimum python version

    Returns:
        Updated pyproject.toml content
    """
    # Use a line-by-line approach for safer replacement
    lines = pyproject_content.split('\n')
    updated_lines = []
    in_metadata_section = False
    metadata_updated = False

    for i, line in enumerate(lines):
        if '[tool.fts.min-versions]' in line:
            in_metadata_section = True
            updated_lines.append(line)
        elif in_metadata_section:
            # Update torch, lightning, python lines
            if line.strip().startswith('torch'):
                updated_lines.append(f'torch = "{torch_ver}"')
                metadata_updated = True
            elif line.strip().startswith('lightning'):
                updated_lines.append(f'lightning = "{lightning_ver}"')
            elif line.strip().startswith('python'):
                updated_lines.append(f'python = "{python_ver}"  # Defined in pyproject.toml requires-python')
                in_metadata_section = False  # End of section
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    if not metadata_updated:
        raise ValueError("Could not find [tool.fts.min-versions] section to update")

    return '\n'.join(updated_lines)


def main():
    """Main entry point."""
    # Locate files
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    utils_path = repo_root / "src" / "finetuning_scheduler" / "dynamic_versioning" / "utils.py"
    pyproject_path = repo_root / "pyproject.toml"

    if not utils_path.exists():
        print(f"Error: Could not find {utils_path}", file=sys.stderr)
        return 1

    if not pyproject_path.exists():
        print(f"Error: Could not find {pyproject_path}", file=sys.stderr)
        return 1

    # Read files
    utils_content = utils_path.read_text()
    pyproject_content = pyproject_path.read_text()

    # Extract versions
    try:
        torch_ver = extract_torch_version(utils_content)
        lightning_ver = extract_lightning_version(utils_content)
        python_ver = extract_python_version(pyproject_content)
    except ValueError as e:
        print(f"Error extracting versions: {e}", file=sys.stderr)
        return 1

    print("Extracted versions from utils.py:")
    print(f"  torch: {torch_ver}")
    print(f"  lightning: {lightning_ver}")
    print(f"  python: {python_ver} (from pyproject.toml requires-python)")

    # Update pyproject.toml
    try:
        updated_content = update_metadata_section(pyproject_content, torch_ver, lightning_ver, python_ver)
    except ValueError as e:
        print(f"Error updating pyproject.toml: {e}", file=sys.stderr)
        return 1

    # Check if anything changed
    if updated_content == pyproject_content:
        print("✓ Metadata already up to date")
        return 0

    # Write updated content
    pyproject_path.write_text(updated_content)
    print(f"✓ Updated {pyproject_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
