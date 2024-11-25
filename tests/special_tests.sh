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

set -eo pipefail  # only disable this when debugging to allow more context

unset mark_type
unset log_file
unset filter_pattern
unset experiments_list
unset experiment_patch_mask
unset collect_dir
unset PL_RUN_STANDALONE_TESTS
unset FTS_RUN_STANDALONE_TESTS
unset FTS_EXPERIMENTAL_PATCH_TESTS

source $(dirname "$0")/infra_utils.sh

usage(){
>&2 cat << EOF
Usage: $0
   [ --mark_type input]
   [ --log_file input]
   [ --filter_pattern input]
   [ --experiments_list input]
   [ --experiment_patch_mask input]
   [ --collect_dir input]
   [ --help ]
   Examples:
	# run all standalone tests (but not experimental ones, note --mark_type defaults to 'standalone' ):
	#   ./tests/special_tests.sh
	# run all standalone tests following a pattern:
	#   ./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_f'
	# run all standalone tests following a pattern using a non-default test collection directory:
	#   ./tests/special_tests.sh --mark_type=standalone --collect_dir='src/fts_examples' --filter_pattern='model_parallel_examples'
  # run all standalone tests passing a parent process log file to use:
  #   ./tests/special_tests.sh --mark_type=standalone --log_file=/tmp/some_parent_process_file_to_append_to.log
  # run all experimental tests following a pattern that are supported by a given experimental patch mask using the
  # default `tests/.experiments` experiments definition location:
	#   ./tests/special_tests.sh --mark_type=exp_patch --filter_pattern='test_f' --experiment_patch_mask="1 0 0"
  # same as above, but use a custom experiments definition location:
  #   ./tests/special_tests.sh --mark_type=exp_patch --filter_pattern='model_parallel' --experiments_list=tests/.my_experiments --experiment_patch_mask="1 0 0"
EOF
exit 1
}

args=$(getopt -o '' --long mark_type:,log_file:,filter_pattern:,experiments_list:,experiment_patch_mask:,collect_dir:,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi


eval set -- ${args}
while :
do
  case $1 in
    --mark_type)  mark_type=$2    ; shift 2  ;;
    --log_file)  log_file=$2    ; shift 2  ;;
    --filter_pattern)  filter_pattern=$2    ; shift 2  ;;
    --experiments_list)  experiments_list=$2    ; shift 2  ;;
    --experiment_patch_mask) experiment_patch_mask+=($2) ; shift 2  ;;
    --collect_dir)  collect_dir=$2    ; shift 2  ;;
    --help)    usage      ; shift   ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

d=`date +%Y%m%d%H%M%S`
tmp_log_dir="/tmp"
mark_type=${mark_type:-"standalone"}
experiments_list=${experiments_list:-$(dirname "$0")/.experiments}
if [ -s "${experiments_list}" ]; then
  if [ -z "${experiment_patch_mask:-}" ]; then
    experiment_patch_mask=($(cat tests/.experiments | awk '{for(i=1;i<=NF;i++) print "0"}'))
  fi
fi
collect_dir=${collect_dir:-"tests"}
special_test_session_log=${log_file:-"${tmp_log_dir}/special_tests_${mark_type}_${d}.log"}
test_session_tmp_log="${tmp_log_dir}/special_tests_raw_${mark_type}_${d}.log"

# default python coverage arguments
exec_defaults='-m coverage run --source src/finetuning_scheduler --append -m pytest --capture=no --no-header -v -s -rA'
collect_defaults="-m pytest ${collect_dir} -q --collect-only --pythonwarnings ignore"
start_time=$(date +%s)
echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $special_test_session_log
printf "FTS special tests beginning execution at ${d} PT \n" | tee -a $special_test_session_log
echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $special_test_session_log
printf "\n" | tee -a $special_test_session_log

define_configuration(){
  echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $special_test_session_log
  printf "Run configuration: \n" | tee -a $special_test_session_log
  echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $special_test_session_log
  case ${mark_type} in
    standalone)
      echo "Collecting and running standalone tests." | tee -a $special_test_session_log
      export PL_RUN_STANDALONE_TESTS=1
      ;;
    exp_patch)
      echo "Collecting and running only experimental patch tests that currently require the provided patch mask (${experiment_patch_mask[@]})." | tee -a $special_test_session_log
      export FTS_EXPERIMENTAL_PATCH_TESTS=1
      ;;
    *)
      echo "no matching `mark_type` found, exiting..." | tee -a $special_test_session_log
      exit 1
      ;;
  esac

  if [ -s "${experiments_list}" ]; then
    # toggle optional experimental patches if requested
    toggle_experimental_patches ${experiments_list} "${experiment_patch_mask[@]}"
  else
    echo "No experimental patches were found in the environment." | tee -a $special_test_session_log
  fi
  printf "${patch_report}" | tee -a $special_test_session_log

  if [[ -n ${filter_pattern} ]]; then
    echo "Using filter pattern: ${filter_pattern}" | tee -a $special_test_session_log
    exec_defaults+=" -k ${filter_pattern}"
    collect_defaults+=" -k ${filter_pattern}"
  fi
  printf '\n' | tee -a  $special_test_session_log
}

trap 'show_test_results "$special_test_session_log" "$test_session_tmp_log"' EXIT  # show the output on exit

## Special coverage collection flow
define_configuration
collect_tests "$collect_defaults" "$special_test_session_log"
execute_tests "$exec_defaults" "$special_test_session_log" "$test_session_tmp_log"
