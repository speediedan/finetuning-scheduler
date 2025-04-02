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

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

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

def _prepare_extras() -> Dict[str, Any]:
    extras = {
        "examples": setup_tools._load_requirements(path_dir=_INSTALL_PATHS["require"], file_name="examples.txt"),
        "extra": setup_tools._load_requirements(path_dir=_INSTALL_PATHS["require"], file_name="extra.txt"),
        "test": setup_tools._load_requirements(path_dir=_INSTALL_PATHS["require"], file_name="test.txt"),
        "ipynb": setup_tools._load_requirements(path_dir=_INSTALL_PATHS["require"], file_name="ipynb.txt"),
        "cli": setup_tools._load_requirements(path_dir=_INSTALL_PATHS["require"], file_name="cli.txt"),
    }
    for ex in ["extra", "examples"]:
        extras[ex].extend(extras["cli"])
    extras["dev"] = extras["extra"] + extras["test"] + extras["ipynb"]
    extras["all"] = extras["dev"] + extras["examples"]
    return extras


def _setup_args(standalone: bool = False) -> Dict[str, Any]:
    about = _load_py_module("__about__.py", _CORE_FTS_LOC)
    long_description = setup_tools._load_readme_description(
        _PATH_ROOT, homepage=about.__homepage__, version=about.__version__
    )
    # Only include dynamic metadata that can't be defined in pyproject.toml
    base_setup = dict(version=about.__version__, description=about.__docs__, long_description=long_description,
                      long_description_content_type="text/markdown",
    )

    # Load our dynamic requirements
    install_requires = dynamic_versioning_utils.get_requirement_files(standalone)

    base_setup["install_requires"] = install_requires
    base_setup["extras_require"] = _prepare_extras()
    return base_setup


if __name__ == "__main__":
    # No need to load assistant.py since we now have the functions in setup_tools
    use_standalone = _PACKAGE_NAME is not None and _PACKAGE_NAME == "pytorch"
    if _PACKAGE_NAME is not None and _PACKAGE_NAME not in _PACKAGE_MODES:
        raise ValueError(f"Unexpected package name: {_PACKAGE_NAME}. Possible choices are: {list(_PACKAGE_MODES)}")
    install_msg = "Installing finetuning-scheduler to depend upon"

    # Toggle to appropriate import style based on package mode using setup_tools directly
    if use_standalone:
        # Convert to standalone imports (lightning.pytorch -> pytorch_lightning)
        dynamic_versioning_utils.use_standalone_pl(_INSTALL_PATHS.values())
        lightning_dep = "pytorch_lightning"
        install_msg += " the standalone version of Lightning: pytorch-lightning."
    else:
        # Convert to unified imports (pytorch_lightning -> lightning.pytorch) if necessary
        dynamic_versioning_utils.use_unified_pl(_INSTALL_PATHS.values())
        lightning_dep = "lightning"
        install_msg += " the default Lightning unified package: lightning."
    print(install_msg)

    # Print additional info about Lightning installation method
    use_commit = os.environ.get("USE_CI_COMMIT_PIN", "").lower() in ("1", "true", "yes")
    if use_commit:
        print("Using Lightning from specific commit (dev/ci mode)")
    else:
        print("Using Lightning from PyPI")

    setup_args = _setup_args(use_standalone)
    setuptools.setup(**setup_args)
    print("Finished setup configuration.")
