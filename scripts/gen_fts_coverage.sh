#!/bin/bash
#
# Utility script to generate local FTS coverage for a given environment
set -eo pipefail

unset repo_home
unset target_env_name
unset no_rebuild_base
unset include_experimental
unset uv_install_flags
unset no_commit_pin
unset venv_dir
unset dry_run
unset oldest
unset no_special
unset run_all_and_examples
unset allow_failures
unset torch_backend
unset phase_status
declare -a from_source_specs=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/infra_utils.sh"

orig_args=("$@")

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo-home input]
   [ --target-env-name input ]
   [ --oldest ]                     # Use oldest CI requirements (Python 3.11, requirements-oldest.txt)
   [ --torch-backend input ]  (cpu, auto, cu130, etc. default: auto)
   [ --no-rebuild-base ]
   [ --no-special ]                 # Skip special tests (standalone/experimental), run only main test suite
   [ --run-all-and-examples ]       # Run all FTS example tests (both standalone and non-standalone)
   [ --allow-failures ]             # Run every phase after a failure (the run still exits 1)
   [ --include-experimental ]
   [ --uv-install-flags "flags" ]
   [ --no-commit-pin ]
   [ --venv-dir input ]
   [ --from-source "package:path[:extras][:ENV_VAR=value]" ]
   [ --dry-run ]
   [ --help ]
   Examples:
	# generate fts_latest coverage without rebuilding the fts_latest base environment:
	#   ./gen_fts_coverage.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --no-rebuild-base
	# generate oldest CI build coverage (matches CI oldest matrix):
	#   ./gen_fts_coverage.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_oldest --oldest --no-special --venv-dir=/mnt/cache/\${USER}/.venvs
	# generate fts_release coverage, rebuilding the base fts_release environment:
	#   ./gen_fts_coverage.sh --repo-home=\${HOME}/repos/fts-release --target-env-name=fts_release
	# generate fts_latest coverage with explicit venv directory (recommended for hardlink performance):
	#   ./gen_fts_coverage.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --venv-dir=/mnt/cache/\${USER}/.venvs
	# generate fts_release coverage without using CI commit pinning:
	#   ./gen_fts_coverage.sh --repo-home=\${HOME}/repos/fts-release --target-env-name=fts_release --no-commit-pin
	# dry-run mode: setup environment and show what tests would run without executing them:
	#   ./gen_fts_coverage.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --dry-run
	# run all examples tests (standalone and non-standalone), continuing after failures:
	#   ./gen_fts_coverage.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --run-all-and-examples --allow-failures
EOF
exit 1
}

args=$(getopt -o '' --long repo-home:,target-env-name:,oldest,torch-backend:,no-rebuild-base,no-special,run-all-and-examples,allow-failures,include-experimental,uv-install-flags:,no-commit-pin,venv-dir:,from-source:,dry-run,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo-home|--repo_home)  repo_home=$2    ; shift 2  ;;
    --target-env-name)  target_env_name=$2  ; shift 2 ;;
    --oldest)   oldest=1 ; shift  ;;
    --torch-backend)   torch_backend=$2   ; shift 2 ;;
    --no-rebuild-base)   no_rebuild_base=1 ; shift  ;;
    --no-special)   no_special=1 ; shift  ;;
    --run-all-and-examples)   run_all_and_examples=1 ; shift  ;;
    --allow-failures)   allow_failures=1 ; shift  ;;
    --include-experimental)   include_experimental=1 ; shift  ;;
    --uv-install-flags)   uv_install_flags=$2 ; shift 2 ;;
    --no-commit-pin)   no_commit_pin=1 ; shift  ;;
    --venv-dir)   venv_dir=$2 ; shift 2 ;;
    --from-source)
        from_source_specs+=("$2")
        shift 2
        ;;
    --dry-run)   dry_run=1 ; shift  ;;
    --help)    usage      ; shift   ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

# Serialize the whole coverage run against other GPU work on the host (no-op unless GPU_LEASE_CMD
# is set). Held once for the entire run; every phase the driver runs inherits it. Taken after
# argument parsing, so a bad flag is reported at once rather than after waiting for the lease.
gpu_lease_reexec "$0" "${orig_args[@]}"
# this run holds the GPU lease (when the host uses one), so it opts in to the CUDA tests (see tests/helpers/runif.py)
export PL_RUN_CUDA_TESTS=1

d=`date +%Y%m%d%H%M%S`
tmp_coverage_dir="/tmp"
coverage_session_log="${tmp_coverage_dir}/gen_fts_coverage_${target_env_name}_${d}.log"
echo "Use 'tail -f ${coverage_session_log}' to monitor progress"

# Log message with timestamp
log_msg() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] $1" >> $coverage_session_log
}

# Determine venv path using infra_utils function
venv_path=$(determine_venv_path "${venv_dir}" "${target_env_name}")
echo "Target venv path: ${venv_path}"

# Define arrays of supported versions
supported_fts_latest=(fts_latest_pt2_6_x fts_latest_pt2_7_x fts_latest_pt2_8_x fts_latest_pt2_9_x fts_latest_pt2_10_x)
supported_fts_release=(fts_release_pt2_6_x fts_release_pt2_7_x fts_release_pt2_8_x fts_release_pt2_9_x fts_release_pt2_10_x)

# Enable extended globbing for pattern matching
shopt -s extglob

# Helper to join array with '|' for extended globbing patterns
join_by_pipe() {
    local IFS="|"
    echo "$*"
}

# Create extended globbing patterns
supported_fts_latest_pattern="@($(join_by_pipe "${supported_fts_latest[@]}"))"
supported_fts_release_pattern="@($(join_by_pipe "${supported_fts_release[@]}"))"
all_supported_pattern="@($(join_by_pipe "${supported_fts_latest[@]}" "${supported_fts_release[@]}"))"

env_rebuild(){
    # Build command arguments array
    local -a cmd_args=("${repo_home}/scripts/build_fts_env.sh" "--repo-home=${repo_home}" "--target-env-name=$1")

    # Add oldest flag if specified
    if [[ $oldest -eq 1 ]]; then
        cmd_args+=("--oldest")
    fi

    # Add uv_install_flags if specified
    if [[ -n "${uv_install_flags}" ]]; then
        cmd_args+=("--uv-install-flags=${uv_install_flags}")
    fi

    # Add no-commit-pin flag if specified
    if [[ $no_commit_pin -eq 1 ]]; then
        cmd_args+=("--no-commit-pin")
    fi

    # Add venv-dir flag if specified
    if [[ -n "${venv_dir}" ]]; then
        cmd_args+=("--venv-dir=${venv_dir}")
    fi

    # Add torch-backend flag if specified
    if [[ -n "${torch_backend}" ]]; then
        cmd_args+=("--torch-backend=${torch_backend}")
    fi

    # Add from-source parameters
    for spec in "${from_source_specs[@]}"; do
        cmd_args+=("--from-source=${spec}")
    done

    # Log the build command before execution
    log_msg "Executing build command: ${cmd_args[*]}"

    case $1 in
        fts_latest|fts_oldest)
            log_msg "Final build command: ${cmd_args[*]}"
            "${cmd_args[@]}"
            ;;
        fts_release)
            log_msg "Final build command: ${cmd_args[*]}"
            "${cmd_args[@]}"
            ;;
        $all_supported_pattern)
            log_msg "Final build command: ${cmd_args[*]}"
            "${cmd_args[@]}"
            log_msg "Logic successfully executed for $1"
            ;;
        *)
            log_msg "ERROR: No matching environment found, exiting..."
            exit 1
            ;;
    esac
}

log_key_package_versions(){
    log_msg "=== Key Package Versions ==="
    log_msg "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
    log_msg "lightning: $(python -c 'import lightning; print(lightning.__version__)' 2>/dev/null || echo 'not installed')"
    log_msg "pytorch_lightning: $(python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)' 2>/dev/null || echo 'not installed')"
    log_msg "finetuning_scheduler: $(python -c 'import finetuning_scheduler; print(finetuning_scheduler.__version__)' 2>/dev/null || echo 'not installed')"
    log_msg "transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'not installed')"
    log_msg "============================="
}

collect_env_coverage(){
    temp_special_log="${tmp_coverage_dir}/special_test_output_$1_${d}.log"
    maybe_deactivate
    source "${venv_path}/bin/activate"
    log_msg "Current venv prompt is now ${VIRTUAL_ENV_PROMPT}"
	cd ${repo_home}

    # Log key package versions after environment activation
    log_key_package_versions

    # If dry-run mode, show test collection and exit
    if [[ $dry_run -eq 1 ]]; then
        log_msg ""
        log_msg "=== DRY-RUN MODE: Collecting tests without execution ==="
        log_msg "Tests that would be collected:"
        python -m pytest src/finetuning_scheduler tests --collect-only -q >> $coverage_session_log 2>&1
        log_msg ""
        log_msg "Standalone tests that would run:"
        PL_RUN_STANDALONE_TESTS=1 python -m pytest tests --collect-only -q >> $coverage_session_log 2>&1 || true
        if [[ $include_experimental -eq 1 ]]; then
            log_msg ""
            log_msg "Experimental patch tests that would run:"
            FTS_EXPERIMENTAL_PATCH_TESTS=1 python -m pytest tests --collect-only -q -k 'test_f' >> $coverage_session_log 2>&1 || true
        fi
    fi

    if ! python -c "import gpu_epsilon.phases" 2>/dev/null; then
        log_msg "ERROR: gpu-epsilon's phase driver is not installed: uv pip install -r requirements/ci/gpu-epsilon.txt"
        exit 1
    fi

    case $1 in
	    fts_latest|fts_oldest|fts_release|$all_supported_pattern)
	        ;;
	    *)
	        log_msg "ERROR: No matching environment found, exiting..."
	        exit 1
	        ;;
	esac

    # The phases run in pyproject.toml order. The driver erases the coverage data first and writes the report last,
    # even after a failed phase. Without --allow-failures it stops at the first failed phase; with it every phase runs
    # and the run still fails, naming each failed phase. Its JSON record of the run lands beside the session log.
    local -a driver_args=(--phases "$(selected_phases)" --log_file "${coverage_session_log}" --log-dir "${tmp_coverage_dir}")
    [[ $allow_failures -eq 1 ]] && driver_args+=(--allow-failures)
    [[ $dry_run -eq 1 ]] && driver_args+=(--dry-run)
    log_msg "Running phases $(selected_phases) through gpu-epsilon's phase driver"
    # each special-test phase's console output goes to the temp log; the session log carries every phase
    python -m gpu_epsilon.phases "${driver_args[@]}" > "${temp_special_log}" 2>&1 || phase_status=$?
    [[ $dry_run -eq 1 ]] && log_msg "=== DRY-RUN COMPLETE ==="
    return 0
}

# The phases this run selects, for gpu-epsilon's phase driver ([tool.gpu-epsilon.phases] in pyproject.toml)
selected_phases(){
    local phases="main"
    if [[ $no_special -ne 1 ]]; then
        phases+=",standalone"
        [[ $run_all_and_examples -eq 1 ]] && phases+=",examples,examples-standalone"
        [[ $include_experimental -eq 1 ]] && phases+=",exp_patch"
    fi
    echo "${phases}"
}

env_rebuild_collect(){
	if [[ $no_rebuild_base -eq 1 ]]; then
		log_msg "Skipping rebuild of the base FTS env ${target_env_name}"
	else
		log_msg "Beginning FTS env rebuild for $1"
		env_rebuild "$1"
	fi
	log_msg "Collecting coverage for the FTS env $1"
    log_msg ""
	collect_env_coverage "$1"
}


## Main coverage collection logic
start_time=$(date +%s)
log_msg "FTS coverage collection executing at ${d} PT"
if [[ $dry_run -eq 1 ]]; then
    log_msg "*** DRY-RUN MODE ENABLED ***"
fi
log_msg "Generating base coverage for the FTS env ${target_env_name}"
env_rebuild_collect "${target_env_name}"
case ${target_env_name} in
    fts_latest|fts_oldest|$supported_fts_latest_pattern)
        log_msg "No env-specific additional coverage currently required for ${target_env_name}"
        ;;
    fts_release|$supported_fts_release_pattern)
        log_msg "No env-specific additional coverage currently required for ${target_env_name}"
        ;;
    # fts_release_ptx_y_z)  # special path to be used when releasing a previous patch version after a new minor version available
    #     log_msg "Generating env-specific coverage for the FTS env fts_release_pta_b_c"
    #     env_rebuild_collect "fts_release_pta_b_c"
    #     log_msg "Generating env-specific coverage for the FTS env fts_release_ptd_e_f"
    #     env_rebuild_collect "fts_release_ptd_e_f"
    #     ;;
    *)
        log_msg "ERROR: No matching environment found, exiting..."
        exit 1
        ;;
esac
if [[ $dry_run -eq 1 ]]; then
    log_msg "Dry-run complete. No tests were executed."
elif [[ ${phase_status:-0} -ne 0 ]]; then
    log_msg "Coverage phases failed (exit ${phase_status}); the phase summary above names each failed phase."
fi
show_elapsed_time $coverage_session_log "FTS coverage collection"
exit ${phase_status:-0}
