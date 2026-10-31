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
"""An offline venngram registry for tests (#176).

venngram looks component maps up through registries (by default its seed registry on the Hugging Face Hub). Tests
instead publish fixture copies of the seed maps (``tests/fixtures/venngram_maps``, copied from venngram's own test
fixtures) to a local store, index them, and point ``VENNGRAM_REGISTRIES`` at that index, so lookups take venngram's
real path without the network, a registry change cannot move a test, and subprocesses (the skill's scripts, notebook
kernels) see the same registry.
"""
import os
from pathlib import Path

FIXTURE_MAPS = Path(__file__).parents[1] / "fixtures" / "venngram_maps"


def build_registry(root: Path) -> str:
    """Publish the fixture maps under ``root`` and index them; return the index's location."""
    from venngram import registry
    from venngram.store import publish

    maps = root / "maps"
    maps.mkdir(parents=True)
    for path in sorted(FIXTURE_MAPS.glob("*.yaml")):
        (maps / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    publish("local", str(maps), maps, documents=[("component_map", p.name) for p in sorted(maps.glob("*.yaml"))])
    registry.write_index(registry.crawl([f"local:{maps}"]), root / "index", name="fts-test-seeds")
    return f"local:{root / 'index'}"


def use_offline_registry(root: Path) -> str:
    """Build the registry under ``root`` and make it this process's (and its subprocesses') only registry."""
    from venngram import registry

    location = build_registry(root)
    os.environ["VENNGRAM_REGISTRIES"] = location
    os.environ.setdefault("VENNGRAM_CACHE", str(root / "cache"))
    registry.set_registries(None)  # read VENNGRAM_REGISTRIES
    registry.clear_cache()
    return location
