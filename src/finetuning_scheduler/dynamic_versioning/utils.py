#!/usr/bin/env python
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
import os
import re
from pathlib import Path
from typing import ValuesView

# -----------------------------------------------------------------------------
# Lightning Configuration
# -----------------------------------------------------------------------------
#
# These version constraints are the single source of truth for minimum versions.
# They are used by setup.py to generate dynamic dependencies at build time.
#
# For visibility, these values are also documented in pyproject.toml under
# [tool.fts.min-versions] (informational only - not used during installation).

# Shared version constraint for all Lightning packages
LIGHTNING_VERSION = ">=2.6.0,<2.6.1"

LIGHTNING_PACKAGE_MAPPING = {
    "lightning.pytorch": "pytorch_lightning",
    "lightning.fabric": "lightning_fabric",
}

# Package and repository mapping
LIGHTNING_PACKAGES = {
    "unified": {
        "package": "lightning",
        "repo": "Lightning-AI/lightning",
        "version": LIGHTNING_VERSION  # Use shared version constraint
    },
    "standalone": {
        "package": "pytorch-lightning",
        "repo": "Lightning-AI/pytorch-lightning",
        "version": LIGHTNING_VERSION  # Use shared version constraint
    }
}

# Base dependencies (torch + Lightning are handled dynamically)
# These are the core dependencies that are always installed
#
# Note: For visibility, minimum versions are also documented in pyproject.toml
# under [tool.fts.min-versions] (informational only).
BASE_DEPENDENCIES = [
    "torch>=2.6.0",
]

# Files to exclude from modification to prevent self-modification
EXCLUDE_FILES_FROM_CONVERSION = [
    "setup_tools.py",
    "dynamic_versioning/utils.py",
    "dynamic_versioning/toggle_lightning_mode.py",
    "test_toggle_lightning_mode.py",
    "test_setup_tools.py",
    "test_dynamic_versioning_utils.py"
]

def get_base_dependencies() -> list[str]:
    """Get the base dependencies list.

    Returns:
        List of base dependency strings (excluding Lightning, which is added dynamically)
    """
    return BASE_DEPENDENCIES.copy()


def get_requirement_files(standalone: bool = False) -> list[str]:
    """Get installation requirements with dynamic Lightning configuration.

    Note: Lightning commit pinning is now handled at install time via UV_OVERRIDE
    environment variable rather than at package metadata time. This simplifies the
    install_requires list to just include version constraints.

    Args:
        standalone: Whether to use standalone pytorch-lightning package

    Returns:
        List of requirement strings
    """
    # Start with base dependencies
    reqs = get_base_dependencies()

    # Add Lightning dependency based on package type
    package_type = "standalone" if standalone else "unified"
    lightning_req = get_lightning_requirement(package_type)
    reqs.append(lightning_req)

    return reqs


def get_lightning_requirement(package_type: str = "unified") -> str:
    """Get the Lightning requirement string based on configuration.

    Note: Commit pinning is handled at install time via UV_OVERRIDE, not here.

    Args:
        package_type: Either "unified" or "standalone"

    Returns:
        The requirement string for the Lightning package with version constraint
    """
    pkg_info = LIGHTNING_PACKAGES[package_type]
    package_name = pkg_info["package"]
    return f"{package_name}{pkg_info['version']}"


def _retrieve_files(directory: str, *ext: str, exclude_files: list[str] | None = None) -> list[str]:
    """Find all files in a directory with optional extension filtering and exclusion."""
    exclude_files = exclude_files or []
    all_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            file_path = os.path.join(root, fname)
            relative_path = os.path.relpath(file_path, directory)

            # Normalize path separators to handle both Windows and Unix paths
            norm_relative_path = relative_path.replace('\\', '/')

            # Check if any excluded path is contained in the normalized file path
            if any(exclude_path in norm_relative_path for exclude_path in exclude_files):
                print(f"Skipping {file_path} to prevent self-modification")
                continue

            if not ext or any(os.path.split(fname)[1].lower().endswith(e) for e in ext):
                all_files.append(file_path)
    return all_files

def _replace_imports(lines: list[str], mapping: list[tuple[str, str]], lightning_by: str = "") -> list[str]:
    """Replace imports of unified packages to standalone."""
    out = lines[:]
    for source_import, target_import in mapping:
        for i, ln in enumerate(out):
            if "from" in ln and "import" in ln:
                out[i] = re.sub(rf"(^|\s)from\s+{re.escape(source_import)}(\.|\s)", rf"\1from {target_import}\2", ln)
            if ln.strip().startswith("import "):
                out[i] = re.sub(rf"import\s+{re.escape(source_import)}(\.|\s|$|,)", rf"import {target_import}\1", ln)
            if lightning_by:
                out[i] = out[i].replace("from lightning import ", f"from {lightning_by} import ")
                out[i] = out[i].replace("import lightning ", f"import {lightning_by} ")
    return out

def _check_import_format(file_content: str, source_imports: list[str]) -> bool:
    """Check if imports in a file already match the expected format."""
    for import_name in source_imports:
        if re.search(rf"(^|\s)from\s+{re.escape(import_name)}(\.|\s)", file_content, re.MULTILINE) or \
           re.search(rf"import\s+{re.escape(import_name)}(\.|\s|$|,)", file_content, re.MULTILINE):
            return False
    return True

def _process_lightning_imports(src_dirs: ValuesView, source_imports: list[str],
                              mapping_pairs: list[tuple[str, str]], target_format: str, debug: bool = False) -> None:
    """Process Lightning imports in python files across directories.

    Args:
        src_dirs: Directories to process
        source_imports: List of import patterns to check for in files
        mapping_pairs: List of (source, target) import mapping tuples
        target_format: Format name for debug messages ("standalone" or "unified")
        debug: Whether to output debug information
    """
    for in_place_path in src_dirs:
        files_to_process = _retrieve_files(str(in_place_path), '.py', exclude_files=EXCLUDE_FILES_FROM_CONVERSION)
        if not files_to_process:
            continue
        for file_path in files_to_process:
            try:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()
                if _check_import_format(content, source_imports=source_imports):
                    if debug:
                        print(f"No imports needed conversion to {target_format} format in {file_path}.")
                    continue
                with open(file_path, 'w', encoding='utf-8') as f:
                    lines = _replace_imports(
                        content.splitlines(True),
                        mapping_pairs
                    )
                    f.writelines(lines)
                print(f"Updated imports in {file_path}")
            except UnicodeDecodeError:
                if debug:
                    print(f"Skipping binary file: {file_path}")
                continue

def use_standalone_pl(
    src_dirs: ValuesView,
    mapping: dict[str, str] = LIGHTNING_PACKAGE_MAPPING,
    debug: bool = False
) -> None:
    """Replace unified Lightning imports with standalone imports."""
    _process_lightning_imports(src_dirs, list(mapping.keys()), list(zip(mapping.keys(), mapping.values())),
                               "standalone", debug)

def use_unified_pl(src_dirs: ValuesView,
                   mapping: dict[str, str] = LIGHTNING_PACKAGE_MAPPING, debug: bool = False) -> None:
    """Replace standalone Lightning imports with unified imports."""
    _process_lightning_imports(src_dirs, list(mapping.values()), list(zip(mapping.values(), mapping.keys())),
                               "unified", debug)

def get_project_paths() -> tuple[Path, dict[str, Path]]:
    """Get project paths for imports conversion and package setup."""
    current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    if "site-packages" in str(current_file_dir) or "dist-packages" in str(current_file_dir):
        project_root = current_file_dir.parent.parent
        install_paths = {
            "source": current_file_dir.parent,
            "examples": current_file_dir.parent.parent / "fts_examples",
        }
    else:
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        install_paths = {}
        for p, d in zip(["source", "tests", "require"], ["src", "tests", "requirements"]):
            install_paths[p] = project_root / d
    install_paths = {k: v for k, v in install_paths.items() if v is not None and v.exists()}
    return project_root, install_paths

def _is_package_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def toggle_lightning_imports(mode: str = "unified", debug: bool = False) -> None:
    """Toggle between standalone and unified Lightning imports."""
    try:
        if mode == "unified" and not _is_package_installed("lightning"):
            print("Warning: Cannot toggle to unified imports because the 'lightning' package is not installed.")
            print("Please install the unified Lightning package with: uv pip install lightning")
            return
        elif mode == "standalone" and not _is_package_installed("pytorch_lightning"):
            print("Warning: Cannot toggle to standalone imports because the 'pytorch-lightning' package is not "
                  "installed.")
            print("Please install the standalone Lightning package with: uv pip install pytorch-lightning")
            return

        _, install_paths = get_project_paths()

        if mode == "standalone":
            print("Converting to standalone imports (e.g. lightning.pytorch -> pytorch_lightning)...")
            use_standalone_pl(install_paths.values(), debug=debug)
        else:
            print("Converting to unified imports (e.g. pytorch_lightning -> lightning.pytorch)...")
            use_unified_pl(install_paths.values(), debug=debug)

        print(f"Successfully toggled to {mode} Lightning imports.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to toggle Lightning imports: {e}")
