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
"""Heading anchors for MyST that match Jupyter's, named by import path in ``conf.py`` (``myst_heading_slug_func``)
so Sphinx can cache the configuration."""


def jupyter_heading_slug(title: str) -> str:
    """Jupyter's heading anchor: spaces become hyphens and case is kept, so a notebook's in-page links such as
    ``[Basic Usage](#Basic-Usage)`` resolve in the docs as they do in Jupyter."""
    return title.strip().replace(" ", "-")
