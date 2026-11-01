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
import pickle
from pathlib import Path


def recorded_allocations(snapshot: Path) -> int:
    """The number of allocations an allocator history snapshot recorded, across its devices' traces.

    A snapshot dumped while nothing records allocator history still writes a file, so a file's existence says nothing
    about whether it holds a history to view.
    """
    with open(snapshot, "rb") as f:
        snap = pickle.load(f)
    return sum(1 for trace in snap.get("device_traces", []) for event in trace if event.get("action") == "alloc")
