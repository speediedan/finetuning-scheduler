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
import setuptools.command.egg_info

_PACKAGE_NAME = os.environ.get("PACKAGE_NAME")
_PACKAGE_MODES = ("pytorch", "lightning")
_PACKAGE_MAPPING = {
    "lightning.pytorch": "pytorch_lightning",
    "lightning.fabric": "lightning_fabric",
}

_PATH_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
_INSTALL_PATHS = {}
for p, d in zip(["source", "tests", "require"], ["src", "tests", "requirements"]):
    _INSTALL_PATHS[p] = _PATH_ROOT / d

_CORE_FTS_LOC = _INSTALL_PATHS["source"] / "finetuning_scheduler"


def _load_py_module(name: str, location: str) -> ModuleType:
    # location = os.path.join(location, name)
    location = location / name
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


setup_tools = _load_py_module(name="setup_tools.py", location=_CORE_FTS_LOC)


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
    base_setup = dict(
        name="finetuning-scheduler",
        version=about.__version__,
        description=about.__docs__,
        author=about.__author__,
        author_email=about.__author_email__,
        url=about.__homepage__,
        download_url="https://github.com/speediedan/finetuning-scheduler",
        license=about.__license__,
        packages=setuptools.find_namespace_packages(where="src"),
        package_dir={"": "src"},
        package_data={
            "fts_examples.stable.config": ["*.yaml"],
            "fts_examples.stable.config.advanced.fsdp": ["*.yaml"],
            "fts_examples.stable.config.advanced.reinit_lr": ["*.yaml"],
            "fts_examples.stable.config.advanced.reinit_optim_lr": ["*.yaml"],
            "fts_examples.legacy.config": ["*.yaml"],
            "fts_examples.legacy.config.advanced.fsdp": ["*.yaml"],
            "fts_examples.legacy.config.advanced.reinit_lr": ["*.yaml"],
        },
        include_package_data=True,
        long_description=long_description,
        long_description_content_type="text/markdown",
        zip_safe=False,
        keywords=[
            "deep learning",
            "pytorch",
            "AI",
            "machine learning",
            "pytorch-lightning",
            "lightning",
            "fine-tuning",
            "finetuning",
        ],
        python_requires=">=3.8",
        setup_requires=[],
        extras_require=_prepare_extras(),
        project_urls={
            "Bug Tracker": "https://github.com/speediedan/finetuning-scheduler/issues",
            "Documentation": "https://finetuning-scheduler.readthedocs.io/en/latest/",
            "Source Code": "https://github.com/speediedan/finetuning-scheduler",
        },
        classifiers=[
            "Environment :: Console",
            "Natural Language :: English",
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
    )

    base_reqs = "standalone_base.txt" if standalone else "base.txt"
    install_requires = setup_tools._load_requirements(
        _INSTALL_PATHS["require"], file_name=base_reqs, standalone=standalone
    )
    # install_requires = setup_tools._load_requirements(
    #     _INSTALL_PATHS["require"],
    #     file_name=base_reqs,
    #     standalone=standalone,
    #     pl_commit="9f173241195450fd8e39a01f28ae2ec94371cd15",
    # )
    base_setup["install_requires"] = install_requires
    return base_setup


if __name__ == "__main__":
    assistant = _load_py_module(name="assistant.py", location=_PATH_ROOT / ".actions")
    use_standalone = _PACKAGE_NAME is not None and _PACKAGE_NAME == "pytorch"
    if _PACKAGE_NAME is not None and _PACKAGE_NAME not in _PACKAGE_MODES:
        raise ValueError(f"Unexpected package name: {_PACKAGE_NAME}. Possible choices are: {list(_PACKAGE_MODES)}")
    install_msg = "Installing finetuning-scheduler to depend upon"
    if use_standalone:
        # install standalone
        mapping = _PACKAGE_MAPPING.copy()
        assistant.use_standalone_pl(mapping, _INSTALL_PATHS.values())
        lightning_dep = "pytorch_lightning"
        install_msg += " the standalone version of Lightning: pytorch-lightning."
    else:
        lightning_dep = "lightning"
        install_msg += " the default Lightning unified package: lightning."
    print(install_msg)
    setup_args = _setup_args(use_standalone)
    setuptools.setup(**setup_args)
    print("Finished setup configuration.")
