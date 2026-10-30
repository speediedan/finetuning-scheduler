#!/bin/bash
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

# Run the special tests (those that need a process of their own) through gpu-epsilon's shared runner, configured by
# [tool.gpu-epsilon.special-tests] in pyproject.toml: one process per node id, each outcome read from per-rank records.
# `python -m gpu_epsilon.special_tests --help` lists the options. Examples, from the repository root:
#   ./tests/special_tests.sh                                   # every standalone test (--mark_type defaults to standalone)
#   ./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_f'
#   ./tests/special_tests.sh --mark_type=standalone --collect_dir='src/fts_examples' --filter_pattern='model_parallel_examples'
#   ./tests/special_tests.sh --mark_type=standalone --select_file=tests/.standalone_smoke   # the pull request gate's set
#   ./tests/special_tests.sh --mark_type=exp_patch --filter_pattern='test_f' --experiment_patch_mask="1 0 0 1"
#   ./tests/special_tests.sh --mark_type=standalone --allow-failures   # every test, even after a failure (still exits 1)

set -eo pipefail

config="$(dirname "$0")/../pyproject.toml"
_gpu_lease_wrap="$(dirname "$0")/../scripts/gpu_lease_wrap.sh"
# shellcheck source=../scripts/gpu_lease_wrap.sh
[[ -r "${_gpu_lease_wrap}" ]] && source "${_gpu_lease_wrap}"
type gpu_lease_reexec &>/dev/null || gpu_lease_reexec() { :; }

if ! python -c "import gpu_epsilon.special_tests" 2>/dev/null; then
  >&2 echo "gpu-epsilon is not installed: python -m pip install -r requirements/ci/gpu-epsilon.txt"
  exit 1
fi
# Validate the arguments first, so a bad flag is reported at once rather than after waiting for the GPU lease, then
# serialize the whole run against other GPU work on the host (no-op unless GPU_LEASE_CMD is set).
python -m gpu_epsilon.special_tests --config "${config}" --check-args "$@"
gpu_lease_reexec "$0" "$@"
exec python -m gpu_epsilon.special_tests --config "${config}" "$@"
