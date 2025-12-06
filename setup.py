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
# Initially based on https://bit.ly/3L7HOQK
"""Setup script for finetuning-scheduler.

This setup.py handles dynamic dependency resolution for Lightning package switching.
Most configuration is in pyproject.toml, but we need setup.py for:
1. Dynamic Lightning package selection (unified vs standalone)
2. Dynamic Lightning commit pinning for CI/dev builds
"""

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import setuptools

_PACKAGE_NAME = os.environ.get("PACKAGE_NAME")
_PACKAGE_MODES = ("pytorch", "lightning")
_PATH_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
_INSTALL_PATHS = {}
for p, d in zip(["source", "tests", "require"], ["src", "tests", "requirements"]):
    _INSTALL_PATHS[p] = _PATH_ROOT / d
_CORE_FTS_LOC = _INSTALL_PATHS["source"] / "finetuning_scheduler"
_DYNAMIC_VERSIONING_LOC = _CORE_FTS_LOC / "dynamic_versioning"


def _load_py_module(name: str, location: str) -> ModuleType:
    location = location / name
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


setup_tools = _load_py_module(name="setup_tools.py", location=_CORE_FTS_LOC)
dynamic_versioning_utils = _load_py_module(name="utils.py", location=_DYNAMIC_VERSIONING_LOC)


def _setup_args(standalone: bool = False) -> dict:
    """Prepare dynamic setup arguments.

    Only version, description, readme, and dependencies are dynamic. Optional dependencies are defined in
    pyproject.toml.
    """
    about = _load_py_module("__about__.py", _CORE_FTS_LOC)
    long_description = setup_tools._load_readme_description(
        _PATH_ROOT, homepage=about.__homepage__, version=about.__version__
    )

    # Load dynamic requirements (base deps + Lightning)
    install_requires = dynamic_versioning_utils.get_requirement_files(standalone)

    return dict(
        version=about.__version__,
        description=about.__docs__,
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=install_requires,
    )


if __name__ == "__main__":
    use_standalone = _PACKAGE_NAME is not None and _PACKAGE_NAME == "pytorch"
    if _PACKAGE_NAME is not None and _PACKAGE_NAME not in _PACKAGE_MODES:
        raise ValueError(f"Unexpected package name: {_PACKAGE_NAME}. Possible choices are: {list(_PACKAGE_MODES)}")
    install_msg = "Installing finetuning-scheduler to depend upon"

    # Toggle to appropriate import style based on package mode
    if use_standalone:
        # Convert to standalone imports (lightning.pytorch -> pytorch_lightning)
        dynamic_versioning_utils.use_standalone_pl(_INSTALL_PATHS.values())
        install_msg += " the standalone version of Lightning: pytorch-lightning."
    else:
        # Convert to unified imports (pytorch_lightning -> lightning.pytorch) if necessary
        dynamic_versioning_utils.use_unified_pl(_INSTALL_PATHS.values())
        install_msg += " the default Lightning unified package: lightning."
    print(install_msg)

    setup_args = _setup_args(use_standalone)
    setuptools.setup(**setup_args)
    print("Finished setup configuration.")
