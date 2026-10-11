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
"""Filesystem helpers for the framework-neutral Fine-Tuning Scheduler core."""
import os
from typing import Any

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs


def get_filesystem(path: str | os.PathLike, **kwargs: Any) -> AbstractFileSystem:
    """Return the ``fsspec`` filesystem for ``path`` (local paths resolve to the local filesystem)."""
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs
