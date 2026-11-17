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
"""Nbpress, the tutorial notebooks' publisher and renderer, is pinned in two places that install it separately: the
docs requirements (Read the Docs, where conf.py stages the notebooks with its renderer) and the `rev` of its pre-
commit hooks.

A bump that misses one runs two versions.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_nbpress_is_pinned_once_in_each_place_and_at_one_commit():
    docs = re.findall(
        r"^nbpress @ git\+https://github\.com/speediedan/nbpress@([0-9a-f]{40})$",
        (REPO_ROOT / "requirements" / "docs.txt").read_text(),
        flags=re.MULTILINE,
    )
    precommit = re.findall(
        r"repo: https://github\.com/speediedan/nbpress\n\s+rev: ([0-9a-f]{40})\b",
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(),
    )
    assert len(docs) == 1 and len(precommit) == 1, {"requirements/docs.txt": docs, ".pre-commit-config.yaml": precommit}
    assert docs == precommit, {"requirements/docs.txt": docs, ".pre-commit-config.yaml": precommit}
