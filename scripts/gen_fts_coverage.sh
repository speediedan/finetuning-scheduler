#!/bin/bash
#
# Utility script to generate local FTS coverage for a given environment
set -eo pipefail

unset repo_home
unset target_env_name
unset torch_dev_ver
unset torchvision_dev_ver
unset torch_test_channel
unset no_rebuild_base
unset include_experimental
unset pip_install_flags
unset no_commit_pin

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo_home input]
   [ --target_env_name input ]
   [ --torch_dev_ver input ]
   [ --torchvision_dev_ver input ]
   [ --torch_test_channel ]
   [ --no_rebuild_base ]
   [ --include_experimental ]
   [ --pip_install_flags "flags" ]
   [ --no_commit_pin ]
   [ --help ]
   Examples:
	# generate fts_latest coverage without rebuilding the fts_latest base environment:
	#   ./gen_fts_coverage.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --no_rebuild_base
	# generate fts_latest coverage with a given torch_dev_version:
	#   ./gen_fts_coverage.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_dev_ver=dev20240201 --torchvision_dev_ver=dev20240201
    # generate fts_latest coverage, rebuilding base fts_latest with PyTorch test channel and run tests that require experimental patches:
    #   ./gen_fts_coverage.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_test_channel --include_experimental
	# generate fts_release coverage, rebuilding the base fts_release environment with PyTorch stable channel:
	#   ./gen_fts_coverage.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release
	# generate fts_release coverage, rebuilding the base fts_release environment with PyTorch test channel:
	#   ./gen_fts_coverage.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release --torch_test_channel
	# generate fts_latest coverage with no pip cache:
	#   ./gen_fts_coverage.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --pip_install_flags="--no-cache-dir"
	# generate fts_release coverage without using CI commit pinning:
	#   ./gen_fts_coverage.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release --no_commit_pin
EOF
exit 1
}

args=$(getopt -o '' --long repo_home:,target_env_name:,torch_dev_ver:,torchvision_dev_ver:,torch_test_channel,no_rebuild_base,include_experimental,pip_install_flags:,no_commit_pin,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo_home)  repo_home=$2    ; shift 2  ;;
    --target_env_name)  target_env_name=$2  ; shift 2 ;;
    --torch_dev_ver)   torch_dev_ver=$2   ; shift 2 ;;
    --torchvision_dev_ver)   torchvision_dev_ver=$2   ; shift 2 ;;
    --torch_test_channel)   torch_test_channel=1 ; shift  ;;
    --no_rebuild_base)   no_rebuild_base=1 ; shift  ;;
    --include_experimental)   include_experimental=1 ; shift  ;;
    --pip_install_flags)   pip_install_flags=$2 ; shift 2 ;;
    --no_commit_pin)   no_commit_pin=1 ; shift  ;;
    --help)    usage      ; shift   ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

d=`date +%Y%m%d%H%M%S`
tmp_coverage_dir="/tmp"
coverage_session_log="${tmp_coverage_dir}/gen_fts_coverage_${target_env_name}_${d}.log"

env_rebuild(){
    # Prepare pip_install_flags parameter if set
    pip_flags_param=""
    if [[ -n "${pip_install_flags}" ]]; then
        pip_flags_param="--pip_install_flags=\"${pip_install_flags}\""
    fi

    # Add no_commit_pin flag if specified
    no_commit_pin_param=""
    if [[ $no_commit_pin -eq 1 ]]; then
        no_commit_pin_param="--no_commit_pin"
    fi

    case $1 in
        fts_latest)
            if [[ -n ${torch_dev_ver} ]]; then
                if [[ -n ${torchvision_dev_ver} ]]; then
                    torchvision_dev_ver=${torch_dev_ver}
                fi
                ${repo_home}/scripts/build_fts_env.sh --repo_home=${repo_home} --target_env_name=$1 --torch_dev_ver=${torch_dev_ver} --torchvision_dev_ver=${torchvision_dev_ver} ${pip_flags_param} ${no_commit_pin_param}
			elif [[ $torch_test_channel -eq 1 ]]; then
                ${repo_home}/scripts/build_fts_env.sh --repo_home=${repo_home} --target_env_name=$1 --torch_test_channel ${pip_flags_param} ${no_commit_pin_param}
            else
                ${repo_home}/scripts/build_fts_env.sh --repo_home=${repo_home} --target_env_name=$1 ${pip_flags_param} ${no_commit_pin_param}
            fi
            ;;
        fts_release)
            if [[ $torch_test_channel -eq 1 ]]; then
                ${repo_home}/scripts/build_fts_env.sh --repo_home=${repo_home} --target_env_name=$1 --torch_test_channel ${pip_flags_param} ${no_commit_pin_param}
            else
                ${repo_home}/scripts/build_fts_env.sh --repo_home=${repo_home} --target_env_name=$1 ${pip_flags_param} ${no_commit_pin_param}
            fi
            ;;
        # fts_release_ptx_y_z)  # minor versions eligible for patch releases
        fts_latest_pt2_5_x | fts_release_pt2_5_x | fts_latest_pt2_6_x | fts_release_pt2_6_x | fts_latest_pt2_7_x | fts_release_pt2_7_x)
            ${repo_home}/scripts/build_fts_env.sh --repo_home=${repo_home} --target_env_name=$1 ${pip_flags_param} ${no_commit_pin_param}
            ;;
        *)
            echo "no matching environment found, exiting..." >> $coverage_session_log
            exit 1
            ;;
    esac
}

collect_env_coverage(){
    temp_special_log="${tmp_coverage_dir}/special_test_output_$1_${d}.log"
	if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    source ~/.venvs/$1/bin/activate
    printf "Current venv prompt is now ${VIRTUAL_ENV_PROMPT} \n" >> $coverage_session_log
	cd ${repo_home}
    source ./scripts/infra_utils.sh
	case $1 in
	    fts_latest | fts_release | fts_release_pt2_5_x | fts_release_pt2_6_x | fts_release_pt2_7_x)
			python -m coverage erase
			python -m coverage run --append --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v 2>&1 >> $coverage_session_log
            (./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_f' --log_file=${coverage_session_log} 2>&1 >> ${temp_special_log}) > /dev/null
            if [[ $include_experimental -eq 1 ]]; then
                echo "Running tests that require experimental patches using $1" >> $coverage_session_log
                (./tests/special_tests.sh --mark_type=exp_patch --filter_pattern='test_f' --log_file=${coverage_session_log} --experiment_patch_mask="1 0 0 1" 2>&1 >> ${temp_special_log}) > /dev/null
            else
                echo "Skipping tests that require experimental patches." >> $coverage_session_log
            fi
	        ;;
	    # fts_latest_pt2_0_1 | fts_release_pt2_0_1)
		# 	(./tests/standalone_tests.sh -k 'test_fsdp_multi_gpus[cust_awp_mwp_2_0_parity_no_use_orig] or test_fsdp_multi_gpus[cust_awp_noprec_dynamo]' --no-header 2>&1 > $temp_standalone_out) > /dev/null
		# 	;;
	    *)
	        echo "no matching environment found, exiting..."  >> $coverage_session_log
	        exit 1
	        ;;
	esac
}

env_rebuild_collect(){
	if [[ $no_rebuild_base -eq 1 ]]; then
		echo "Skipping rebuild of the base FTS env ${target_env_name}" >> $coverage_session_log
	else
		echo "Beginning FTS env rebuild for $1" >> $coverage_session_log
		env_rebuild "$1"
	fi
	echo "Collecting coverage for the FTS env $1" >> $coverage_session_log
    printf "\n"  >> $coverage_session_log
	collect_env_coverage "$1"
}


## Main coverage collection logic
start_time=$(date +%s)
echo "FTS coverage collection executing at ${d} PT" > $coverage_session_log
echo "Generating base coverage for the FTS env ${target_env_name}" >> $coverage_session_log
env_rebuild_collect "${target_env_name}"
case ${target_env_name} in
    fts_latest)
        echo "No env-specific additional coverage currently required for ${target_env_name}" >> $coverage_session_log
        ;;
    fts_release | fts_release_pt2_5_x | fts_release_pt2_5_x | fts_release_pt2_6_x | fts_release_pt2_7_x)
        echo "No env-specific additional coverage currently required for ${target_env_name}" >> $coverage_session_log
        ;;
    # fts_release_ptx_y_z)  # special path to be used when releasing a previous patch version after a new minor version available
    #     echo "Generating env-specific coverage for the FTS env fts_release_pta_b_c" >> $coverage_session_log
    #     env_rebuild_collect "fts_release_pta_b_c"
    #     echo "Generating env-specific coverage for the FTS env fts_release_ptd_e_f" >> $coverage_session_log
    #     env_rebuild_collect "fts_release_ptd_e_f"
    #     ;;
    *)
        echo "no matching environment found, exiting..."  >> $coverage_session_log
        exit 1
        ;;
esac
echo "Writing collected coverage stats for FTS env ${target_env_name}" >> $coverage_session_log
python -m coverage report -m >> $coverage_session_log
show_elapsed_time $coverage_session_log "FTS coverage collection"
