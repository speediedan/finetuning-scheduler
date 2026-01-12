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
declare -a from_source_specs=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/infra_utils.sh"

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo-home input]
   [ --target-env-name input ]
   [ --oldest ]                     # Use oldest CI requirements (Python 3.10, requirements-oldest.txt)
   [ --no-rebuild-base ]
   [ --no-special ]                 # Skip special tests (standalone/experimental), run only main test suite
   [ --run-all-and-examples ]       # Run all FTS example tests (both standalone and non-standalone)
   [ --allow-failures ]             # Continue running special tests after failures
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

args=$(getopt -o '' --long repo-home:,repo_home:,target-env-name:,target_env_name:,oldest,no-rebuild-base,no_rebuild_base,no-special,run-all-and-examples,allow-failures,include-experimental:,include_experimental,uv-install-flags:,uv_install_flags:,no-commit-pin,no_commit_pin,venv-dir:,from-source:,dry-run,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo-home|--repo_home)  repo_home=$2    ; shift 2  ;;
    --target-env-name|--target_env_name)  target_env_name=$2  ; shift 2 ;;
    --oldest)   oldest=1 ; shift  ;;
    --no-rebuild-base|--no_rebuild_base)   no_rebuild_base=1 ; shift  ;;
    --no-special)   no_special=1 ; shift  ;;
    --run-all-and-examples)   run_all_and_examples=1 ; shift  ;;
    --allow-failures)   allow_failures=1 ; shift  ;;
    --include-experimental|--include_experimental)   include_experimental=1 ; shift  ;;
    --uv-install-flags|--uv_install_flags)   uv_install_flags=$2 ; shift 2 ;;
    --no-commit-pin|--no_commit_pin)   no_commit_pin=1 ; shift  ;;
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
    local -a cmd_args=("${repo_home}/scripts/build_fts_env.sh" "--repo_home=${repo_home}" "--target_env_name=$1")

    # Add oldest flag if specified
    if [[ $oldest -eq 1 ]]; then
        cmd_args+=("--oldest")
    fi

    # Add uv_install_flags if specified
    if [[ -n "${uv_install_flags}" ]]; then
        cmd_args+=("--uv_install_flags=${uv_install_flags}")
    fi

    # Add no_commit_pin flag if specified
    if [[ $no_commit_pin -eq 1 ]]; then
        cmd_args+=("--no_commit_pin")
    fi

    # Add venv-dir flag if specified
    if [[ -n "${venv_dir}" ]]; then
        cmd_args+=("--venv-dir=${venv_dir}")
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
        python -m pytest src/finetuning_scheduler tests --collect-only -q 2>&1 >> $coverage_session_log
        log_msg ""
        log_msg "Standalone tests that would run (pattern: test_f):"
        PL_RUN_STANDALONE_TESTS=1 python -m pytest tests -m standalone --collect-only -q -k 'test_f' 2>&1 >> $coverage_session_log || true
        if [[ $include_experimental -eq 1 ]]; then
            log_msg ""
            log_msg "Experimental patch tests that would run:"
            PL_RUN_EXP_PATCH_TESTS=1 python -m pytest tests -m exp_patch --collect-only -q -k 'test_f' 2>&1 >> $coverage_session_log || true
        fi
        log_msg "=== DRY-RUN COMPLETE ==="
        return 0
    fi

    case $1 in
	    fts_latest|fts_oldest|fts_release|$all_supported_pattern)
            log_msg "Erasing previous coverage data"
			python -m coverage erase
            log_msg "Running main test suite with coverage"
			python -m coverage run --append --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v 2>&1 >> $coverage_session_log
            # Skip special tests if --no-special flag is set
            if [[ $no_special -eq 1 ]]; then
                log_msg "Skipping special tests (--no-special flag set)"
            else
                # Prepare allow_failures flag for special_tests.sh
                local failures_flag=""
                if [[ $allow_failures -eq 1 ]]; then
                    failures_flag="--allow-failures"
                fi

                log_msg "Running standalone tests (pattern: test_f)"
                (./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_f' --log_file=${coverage_session_log} ${failures_flag} 2>&1 >> ${temp_special_log}) > /dev/null

                if [[ $run_all_and_examples -eq 1 ]]; then
                    log_msg "Running all FTS example tests (non-standalone)"
                    python -m coverage run --append --source src/finetuning_scheduler -m pytest src/fts_examples/test_examples.py -v 2>&1 >> $coverage_session_log

                    log_msg "Running all FTS example tests (standalone)"
                    (./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_examples' --collect_dir=src/fts_examples --log_file=${coverage_session_log} ${failures_flag} 2>&1 >> ${temp_special_log}) > /dev/null
                fi

                if [[ $include_experimental -eq 1 ]]; then
                    log_msg "Running tests that require experimental patches using $1"
                    (./tests/special_tests.sh --mark_type=exp_patch --filter_pattern='test_f' --log_file=${coverage_session_log} --experiment_patch_mask="1 0 0 1" ${failures_flag} 2>&1 >> ${temp_special_log}) > /dev/null
                else
                    log_msg "Skipping tests that require experimental patches."
                fi
            fi
	        ;;
	    *)
	        log_msg "ERROR: No matching environment found, exiting..."
	        exit 1
	        ;;
	esac
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
else
    log_msg "Writing collected coverage stats for FTS env ${target_env_name}"
    python -m coverage report -m >> $coverage_session_log
fi
show_elapsed_time $coverage_session_log "FTS coverage collection"
