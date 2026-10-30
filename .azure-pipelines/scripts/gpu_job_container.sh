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
#
# The steps of the any-vendor GPU job (.azure-pipelines/templates/gpu-job.yml, #57), run INSIDE its container with the
# checkout at the working directory. They are the GPU phases of every pipeline that uses the template (gpu-tests.yml,
# gpu-proof.yml), on whichever vendor's image and devices the agent's GPU_VENDOR chose. Phase manifests, records and
# reports go to /evidence.
#
#   gpu_job_container.sh install | collect | verify | lease | phase <name> | coverage | report | release
set -eo pipefail

out=/evidence/fts-phases
pid=/tmp/azp_host_lease.pid
venv() { source /tmp/venvs/fts_dev/bin/activate; }

case "${1:?command}" in
  install)
    venv
    git config --global --add safe.directory "$PWD"
    python -c "import torch; print(f'torch: {torch.__version__} cuda: {torch.version.cuda} hip: {torch.version.hip}')"
    export UV_OVERRIDE="${PWD}/requirements/ci/overrides.txt"
    # torch-excludes.txt keeps the image's own torch build (CUDA or ROCm) from being replaced by a PyPI wheel
    uv pip install -e . -r requirements/ci/requirements.txt -r requirements/ci/gpu-epsilon.txt \
      --excludes /tmp/venvs/fts_dev/torch-excludes.txt
    uv pip list
    ;;
  collect)
    # Collection only, with no device visible, so it needs no lease (as in gpu-tests.yml).
    venv
    mkdir -p "$out"
    phases=(--phase gpu-nonstandalone --phase examples --phase examples-multigpu)
    # the template passes the boolean parameter as written (True/False)
    [[ "${FTS_POOL_REVERSE,,}" == true ]] && phases+=(--phase pool-reverse)
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python -m tests.gpu_ci.phase_manifest collect "${phases[@]}" \
      --out-dir "$out"
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python -m tests.gpu_ci.phase_manifest collect --phase standalone \
      --scope "${FTS_STANDALONE_SCOPE:?}" --out-dir "$out"
    ;;
  lease)
    # The whole server and host memory, through the host's lease tool when it is mounted (scripts/gpu_lease_wrap.sh).
    # GPU_LEASE_DEVICES, when the host passes it, names the devices for a tool that cannot discover them itself.
    source scripts/gpu_lease_wrap.sh
    ci_host_lease_hold /gpu_leases /gpu_lease_tool "$pid" 3600 "azure-${BUILD_BUILDID:?}" --gpus all --cpu-heavy
    ;;
  verify)
    # Backend truth under the lease: the torch build must match the vendor the agent reported, with both devices.
    venv
    kill -0 "$(cat "$pid")"
    python requirements/collect_env_details.py
    python - <<'PY'
import os
import torch
expected = os.environ["FTS_GPU_VENDOR"]
actual = "amd" if torch.version.hip else "nvidia" if torch.version.cuda else "cpu"
assert actual == expected, (expected, actual, torch.__version__)
assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
assert os.environ.get("PL_RUN_CUDA_TESTS") == "1", "PL_RUN_CUDA_TESTS is not set"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip,
       "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
PY
    ;;
  phase)
    venv
    phase="${2:?phase}"
    eval "$(python -m tests.gpu_ci.phase_manifest run-env --phase "$phase" --out-dir "$out")"
    case "$phase" in
      gpu-nonstandalone)
        FTS_CUDA_TESTS_ONLY=1 python -m coverage run --source src/finetuning_scheduler -m pytest \
          src/finetuning_scheduler tests -v --junitxml=/evidence/test-results.xml --durations=50
        ;;
      standalone)
        echo "standalone scope: ${FTS_STANDALONE_SCOPE}"
        if [[ "$FTS_STANDALONE_SCOPE" == smoke ]]; then
          python -m gpu_epsilon.special_tests --mark_type=standalone --select_file=tests/.standalone_smoke
        else
          python -m gpu_epsilon.special_tests --mark_type=standalone
        fi
        ;;
      examples)
        python -m pytest src/fts_examples -v --maxfail=1 --durations=0 -W ignore:\`np.object\`:DeprecationWarning \
          -W ignore:'`np.int` is':DeprecationWarning
        ;;
      examples-multigpu)
        mkdir -p /tmp/kernel_cache
        PYTORCH_KERNEL_CACHE_PATH=/tmp/kernel_cache python -m gpu_epsilon.special_tests --mark_type=standalone \
          --collect_dir='src/fts_examples' --filter_pattern='model_parallel_examples'
        ;;
      pool-reverse)
        modules=$(grep -l dist_pool_fixture tests/test_*.py)
        echo "pool modules: ${modules}"
        FTS_REVERSE_ORDER=1 python -m pytest ${modules} -v --durations=20
        ;;
      *) echo "unknown phase ${phase}" >&2; exit 2 ;;
    esac
    ;;
  coverage)
    # The GPU coverage upload: the two phases it has always covered, each in its own data file, combined here. Either
    # vendor's run covers the same lines (the AMD agent's parity runs lost none), so the upload names the vendor only
    # in its run name. Without a token (a pipeline that does not map the secret), the report is written but not sent.
    venv
    data=()
    for phase in gpu-nonstandalone standalone; do
      f="${out}/coverage/${phase}/.coverage"
      if [[ -f "$f" ]]; then data+=("$f"); else echo "##vso[task.logissue type=warning]no coverage data for ${phase}"; fi
    done
    python -m coverage combine --keep "${data[@]}"
    python -m coverage report
    python -m coverage xml
    if [[ -z "${CODECOV_TOK:-}" ]]; then
      echo "##vso[task.logissue type=warning]no Codecov token in this pipeline: coverage written, not uploaded"
      exit 0
    fi
    # The CLI is verified before it runs: its checksum file must carry a good signature from Codecov's signing key, by
    # fingerprint (keybase.io, which used to serve the key, no longer does), and the binary must match the checksum.
    codecov_key=27034E7FDB850E0BBC2C62FF806BB28AED779869
    curl -fsS https://uploader.codecov.io/verification.gpg | gpg --no-default-keyring --keyring trustedkeys.gpg --import
    curl -fsSO https://cli.codecov.io/latest/linux/codecov
    curl -fsSO https://cli.codecov.io/latest/linux/codecov.SHA256SUM
    curl -fsSO https://cli.codecov.io/latest/linux/codecov.SHA256SUM.sig
    sig=$(gpg --no-default-keyring --keyring trustedkeys.gpg --status-fd 1 --verify codecov.SHA256SUM.sig \
      codecov.SHA256SUM 2>/dev/null) || true
    grep -q "^\[GNUPG:\] VALIDSIG ${codecov_key} " <<< "$sig" \
      || { echo "##vso[task.logissue type=error]the Codecov CLI checksums are not signed by ${codecov_key}"; exit 1; }
    sha256sum -c codecov.SHA256SUM
    chmod +x codecov
    ./codecov upload-process --slug 'speediedan/finetuning-scheduler' -t "$CODECOV_TOK" \
      --commit-sha "${BUILD_SOURCEVERSION:?}" --git-service 'github' -n "GPU-coverage-${FTS_GPU_VENDOR:?}" -F gpu -F pytest \
      --env 'linux,azure' -f 'coverage.xml'
    ;;
  report)
    # One report per collected phase, under the vendor the agent reported. Exits non-zero when any phase is not
    # clean, so a vendor-marked or proof job is decided by its reports and not only by its test steps. A skip or xfail
    # is clean only when tests/gpu_ci/reviewed_exclusions.json gives its reason.
    venv
    status=0
    for manifest in "$out"/manifests/*.json; do
      phase=$(basename "$manifest" .json)
      if [[ ! -d "$out/records/$phase" ]]; then
        echo "##vso[task.logissue type=error]phase ${phase}: collected, but its step did not run"
        status=1
        continue
      fi
      python -m tests.gpu_ci.phase_manifest report --phase "$phase" --out-dir "$out" --vendor "${FTS_GPU_VENDOR:?}" \
        --evidence-class pipeline --lease-mode "$(cat "${pid}.mode" 2>/dev/null)" --image "${FTS_IMAGE:?}" \
        --reviewed-exclusions tests/gpu_ci/reviewed_exclusions.json \
        || { echo "##vso[task.logissue type=error]phase ${phase}: report is not clean"; status=1; }
    done
    exit $status
    ;;
  release)
    source scripts/gpu_lease_wrap.sh
    ci_host_lease_release /gpu_leases /gpu_lease_tool "$pid"
    ;;
  *) echo "unknown command $1" >&2; exit 2 ;;
esac
