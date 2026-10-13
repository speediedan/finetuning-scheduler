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
"""Cases served by the persistent distributed test pool (``tests/helpers/dist_pool.py``).

Everything in this package runs inside pool workers, which import torch and nothing from Lightning or
``tests.helpers`` (whose package import pulls Lightning in). A case is a module-level function
``case(ctx, **kwargs)`` that returns plain, picklable data; the test's assertions run in the main process.
``_fake_pg`` also runs cases in the main process itself, once per simulated rank on the fake process group.
"""
