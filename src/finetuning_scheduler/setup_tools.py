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
import warnings
from pathlib import Path
from typing import Any, Generator
from contextlib import contextmanager

_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(__file__))).parent
_TH = os.path.join(_PROJECT_ROOT, "tests/helpers")


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#") -> list[str]:
    """Load requirements from a file.

    >>> _load_requirements(_TH, file_name="req.txt")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    direct dependency req 'git+https://github.com/t/test.git@test' has been pruned from the provided requirements
    direct dependency req 'http://github.com/user/repo/tarball/master' has been pruned from the provided requirements
    ['ok']
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filter all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith(("http", "git+")) or "@http" in ln:
            print(f"direct dependency req '{ln}' has been pruned from the provided requirements")
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def _load_readme_description(path_dir: str, homepage: str, version: str) -> str:
    """Load readme as description.

    >>> _load_readme_description(_PROJECT_ROOT, "", "")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<div align="center">...'
    """
    path_readme = os.path.join(path_dir, "README.md")
    version_prefix = "v"  # standard prefix used for tagging versions
    text = open(path_readme, encoding="utf-8").read()

    github_source_url = f"{homepage}/raw/{version_prefix}{version}"
    # replace relative repository path to absolute link to the release
    # do not replace all "docs" as in the readme we refer to some other sources with particular path to docs
    text = text.replace("docs/source/_static/", f"{github_source_url}/docs/source/_static/")

    # readthedocs badge
    text = text.replace("badge/?version=stable", f"badge/?version={version}")
    text = text.replace(
        "finetuning-scheduler.readthedocs.io/en/latest/", f"finetuning-scheduler.readthedocs.io/en/{version}"
    )
    # codecov badge
    text = text.replace("/branch/main/graph/badge.svg", f"/release/{version}/graph/badge.svg")
    # replace github badges for release ones
    text = text.replace("badge.svg?branch=main&event=push", f"badge.svg?tag={version}")
    # Azure...
    text = text.replace("?branchName=main", f"?branchName=refs%2Ftags%2F{version}")
    text = re.sub(r"\?definitionId=\d+&branchName=main", f"?definitionId=2&branchName=refs%2Ftags%2F{version}", text)

    skip_begin = r"<!-- following section will be skipped from PyPI description -->"
    skip_end = r"<!-- end skipping PyPI description -->"
    text = re.sub(rf"{skip_begin}.+?{skip_end}", "<!--  -->", text, flags=re.IGNORECASE + re.DOTALL)
    return text

@contextmanager
def disable_always_warnings() -> Generator[None, None, None]:
    # can be used to re-enable filterwarnings in cases where `simplefilter('always')` is used to force a warning
    # which can render normal filterwarnings use ineffective (e.g. https://github.com/pytorch/pytorch/pull/123619)
    """A context manager that temporarily disables the use of `simplefilter('always')` within its context.

    >>> import warnings
    >>> with disable_always_warnings():
    ...     # This would normally show a warning with 'always'
    ...     warnings.simplefilter('always')
    ...     warnings.warn("Test warning", UserWarning)
    """
    original_simplefilter = warnings.simplefilter
    def disable_always_simplefilter(*args: Any, **kwargs: Any) -> None:
        if args[0] == "always":
            return
        original_simplefilter(*args, **kwargs)
    warnings.simplefilter = disable_always_simplefilter
    try:
        yield
    finally:
        warnings.simplefilter = original_simplefilter
