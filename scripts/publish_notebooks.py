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
"""Publish the example notebooks: copy ``dev`` to ``publish``, adding a Colab badge and an install cell.

A shim over the notebook publisher vendored from interpretune (``scripts/vendored/notebook_tooling``), configured by
``[tool.notebook-publishing]`` in ``pyproject.toml``. The pre-commit ``publish-notebooks`` hook runs it when a dev
notebook changes.

Usage:
    python scripts/publish_notebooks.py [--dry-run] [--check-only] [--force]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE = REPO_ROOT / "scripts" / "vendored" / "notebook_tooling" / "notebook_publishing.py"


def _load_engine():
    spec = importlib.util.spec_from_file_location("_notebook_publishing_sibling", ENGINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve `from __future__` annotations through sys.modules
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    sys.exit(_load_engine().main(root=REPO_ROOT))
