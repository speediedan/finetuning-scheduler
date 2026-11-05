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
"""Gates for tests that build real ``transformers`` decoders, which the oldest supported version predates."""
import pytest


def _has_hf_models() -> bool:
    try:
        import transformers
    except ImportError:
        return False
    return hasattr(transformers, "Qwen3Config")


needs_hf_models = pytest.mark.skipif(not _has_hf_models(), reason="a transformers with Llama and Qwen3 models required")
