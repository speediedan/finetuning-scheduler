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
"""Examples of tunable views and addresses: giving a model a view, then scheduling it by address (#158).

The examples find venngram's maps (of architectures and of adapters) through its seed registry pinned to one revision
(:data:`SEED_REGISTRY`), so a rendered notebook says which registry generation it used and a later registry change
cannot move it.
"""
import os

#: venngram's seed registry of verified maps, pinned to a revision (#176), the rules-shape rebuild of venngram#49.
SEED_REGISTRY = "hub-dataset:speediedan/venngram-registry/index@d927d8736e4b2681a1f4291784ef697a53710b2a"


def use_seed_registry() -> list[str]:
    """Look venngram's maps up in :data:`SEED_REGISTRY`, unless ``VENNGRAM_REGISTRIES`` already names the
    registries (as the tests do, with a local one).

    Returns the registries now consulted.
    """
    from venngram import registry

    if "VENNGRAM_REGISTRIES" not in os.environ:
        registry.set_registries([SEED_REGISTRY])
    return registry.registries()
