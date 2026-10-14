#!/bin/bash
# Test infra utility functions
# Note we use local variables for many of these to allow more usage flexibility in different contexts


# Optional GPU-lease serialization (no-op unless GPU_LEASE_CMD is set). See scripts/gpu_lease_wrap.sh.
_gpu_lease_wrap="$(dirname "${BASH_SOURCE[0]}")/../scripts/gpu_lease_wrap.sh"
# shellcheck source=../scripts/gpu_lease_wrap.sh
[[ -r "${_gpu_lease_wrap}" ]] && source "${_gpu_lease_wrap}"
type maybe_gpu_lease &>/dev/null || maybe_gpu_lease() { "$@"; }
source $(dirname "$0")/../scripts/infra_utils.sh

toggle_experimental_patches() {
    # Function to encapsulate toggling of the current FTS experimental patch flags on and off. Usage example:
    #    toggle_experimental_patches /path/to/.experiments 1 0 1
    export patch_report=''
    filepath="$1"
    shift

    declare -a exp_patch_flags=($(cat "$filepath"))
    declare -a patch_mask=("$@")

    if [[ ${#exp_patch_flags[@]} -ne ${#patch_mask[@]} ]]; then
        echo "Error: There are currently ${#exp_patch_flags[@]} defined experiments, provided mask should have that length." >&2
        return 1
    fi

    for i in "${!exp_patch_flags[@]}"; do
        let arg_idx=i+1
        if [[ ${patch_mask[$i]} -eq 1 ]]; then
            export "${exp_patch_flags[$i]}"=1
            patch_report+="${exp_patch_flags[$i]} value is now: ${!exp_patch_flags[$i]}\n"
        else
            unset "${exp_patch_flags[$i]}"
        fi
    done
}

collect_tests(){
  local collect_def="$1"
  local collect_log="$2"
  local collect_rc=0
  special_tests=$(python ${collect_def}) || collect_rc=$?
  if [[ $collect_rc -eq 0 ]]; then
    # match only lines with tests
    declare -a -g parameterizations=($(grep -oP '\S+::test_\S+' <<< "$special_tests"))
    echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $collect_log
    printf "Collected the following tests: \n" | tee -a  $collect_log
    echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $collect_log
    printf '%s\n' "${parameterizations[@]}" | tee -a  $collect_log
    num_collected_tests="${#parameterizations[@]}"
    echo "Total number of tests: ${#parameterizations[@]}" | tee -a  $collect_log
    printf '\n' | tee -a  $collect_log
  elif [[ $collect_rc -eq 5 ]]; then
    # pytest's exit code 5 means nothing was selected, which is legitimate (e.g. an empty experimental-patch partition)
    printf "No tests were selected by the following collection command: python ${collect_def} \n" | tee -a $collect_log
    printf "Exiting without running tests. \n" | tee -a $collect_log
    export no_tests_collected=1
    exit 0
  else
    # any other non-zero exit is a collection error: fail, rather than report a phase that ran nothing as green
    printf "Test collection failed (pytest exit code ${collect_rc}): python ${collect_def} \n" | tee -a $collect_log
    printf '%s\n' "$special_tests" | tail -n 40 | tee -a $collect_log
    export collection_failed=1
    exit 1
  fi
}

select_tests(){
  # Keep only the collected tests listed in "$1", one node id per line (blank lines and `#` comments are ignored), in
  # the file's order. A listed test that was not collected fails the run, so the list cannot fall out of date silently.
  local select_file="$1"
  local select_log="$2"
  if [[ ! -r "$select_file" ]]; then
    printf "Cannot read the test list ${select_file}. \n" | tee -a $select_log
    export collection_failed=1
    exit 1
  fi
  local -a wanted=($(grep -vE '^[[:space:]]*(#|$)' "$select_file" || true))
  if [[ ${#wanted[@]} -eq 0 ]]; then
    printf "The test list ${select_file} names no tests. \n" | tee -a $select_log
    export collection_failed=1
    exit 1
  fi
  local -a kept=() missing=()
  for w in "${wanted[@]}"; do
    if printf '%s\n' "${parameterizations[@]}" | grep -qxF -- "$w"; then kept+=("$w"); else missing+=("$w"); fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    printf "These tests listed in ${select_file} were not collected: \n" | tee -a $select_log
    printf '%s\n' "${missing[@]}" | tee -a $select_log
    export collection_failed=1
    exit 1
  fi
  parameterizations=("${kept[@]}")
  num_collected_tests="${#parameterizations[@]}"
  printf "Selected the ${num_collected_tests} tests listed in ${select_file}: \n" | tee -a $select_log
  printf '%s\n' "${parameterizations[@]}" | tee -a $select_log
  printf '\n' | tee -a $select_log
}

execute_tests(){
  ensure_tests
  local execute_def="$1"
  local execute_log="$2"
  local tmp_raw_log="$3"
  local allow_failures=${4:-0}  # Default to 0 (fail on first error)
  # hardcoded tests to skip - space separated
  blocklist=''
  export report=''
  echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $execute_log
  printf "Running the collected tests: \n" | tee -a  $execute_log
  echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $execute_log

  for i in "${!parameterizations[@]}"; do
    parameterization=${parameterizations[$i]}

    # check blocklist
    if echo $blocklist | grep -F "${parameterization}"; then
      report+="Skipped\t$parameterization\n"
      continue
    fi

    # run the test
    echo "Running ${parameterization}" | tee -a $execute_log
    # capture the exit code rather than letting the outer `set -eo pipefail` abort here, otherwise the
    # result parsing below (and --allow-failures) is unreachable for any failing parameterization
    test_rc=0
    # each test's output is parsed on its own, then appended to the session's raw log
    test_raw_log=$(mktemp)
    (set -o pipefail; python ${execute_def} ${parameterization} 2>&1 | sed "s,\x1b\[[0-9;]*[a-zA-Z],,g" > "$test_raw_log") > /dev/null || test_rc=$?
    cat "$test_raw_log" >> $tmp_raw_log
    test_to_find=`echo ${parameterization} | sed 's/\[/\\\[/g; s/\]/\\\]/g'`
    # Under Lightning's `ddp` launcher every other rank re-runs this test in its own pytest session, writing to the same
    # output, and only rank zero's exit code reaches this shell. So a FAILED or ERROR line from any rank fails the test,
    # as does a non-zero exit, whichever rank's verdict is printed first, and so does a session that never finished (a
    # rank that died, or was aborted in a collective, prints no verdict at all).
    sessions=$(grep -c "test session starts" "$test_raw_log" || true)
    finished=$(grep -cE "^=+ .+ in [0-9.]+s.* =+$" "$test_raw_log" || true)
    if grep -qE "(FAILED|ERROR) .*${test_to_find}" "$test_raw_log"; then
      parameterization_result="${parameterization}: FAILED"
    elif [[ $test_rc -ne 0 ]]; then
      parameterization_result="${parameterization}: FAILED (exit code ${test_rc})"
    elif [[ $finished -lt $sessions ]]; then
      parameterization_result="${parameterization}: FAILED (${sessions} pytest sessions started, ${finished} finished)"
    elif pass_or_fail=$(grep -E "(PASSED|XPASS|XFAIL) .*${test_to_find}" "$test_raw_log"); then
      parameterization_result=`echo $pass_or_fail | awk 'NR==1 {print $2 ": "  $1}'`;
    elif skipped=$(grep -E "${test_to_find}.*SKIPPED" "$test_raw_log"); then
      parameterization_result=`echo $skipped | awk 'NR==1 {print $1 ": "  $2}'`;
    elif grep -qE "SKIPPED \[[0-9]+\] $(echo "${parameterization%%::*}" | sed 's/[][]/\\&/g'):" "$test_raw_log"; then
      # pytest's short-summary SKIPPED lines cite file:LINE (not ::testname), and the progress
      # line's SKIPPED tag can be separated from the test id by interleaved live-log output
      # (e.g. a skip raised mid-test after log capture), so fall back to the file-scoped summary.
      # Only reached on exit 0, so a crashed test can never be misclassified as a skip.
      parameterization_result="${parameterization}: SKIPPED"
    else
      echo "Could not parse result!" | tee -a $execute_log
      parameterization_result="UNKNOWN: see $tmp_raw_log"
    fi
    rm -f "$test_raw_log"
    report+="Ran\t${parameterization_result}\n"
    if [[ "$parameterization_result" == *FAILED* && $allow_failures -eq 0 ]]; then
      echo "Test failed: ${parameterization_result}" | tee -a $execute_log
      echo "Stopping execution (use --allow-failures to continue after failures)" | tee -a $execute_log
      break
    fi
  done
}

show_test_counts(){
  local test_log="$1"
  export num_failed=0
  export num_other=0
  if grep_succ=($(printf "$report" | grep -c "PASSED\|XPASSED\|XFAIL")); then num_succ=$grep_succ; else num_succ=0; fi
  if grep_failed=($(printf "$report" | grep -c "FAILED")); then num_failed=$grep_failed; fi
  if grep_skipped=($(printf "$report" | grep -c "SKIPPED")); then num_skipped=$grep_skipped; else num_skipped=0; fi
  printf "\n" | tee -a $test_log
  echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
  printf "Test count summary: \n" | tee -a  $test_log
  echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
  printf "Collected: $num_collected_tests \n" | tee -a $test_log
  printf "Succeeded (passed+xpassed+xfail): $num_succ \n" | tee -a $test_log
  printf "Intentionally skipped: $num_skipped \n" | tee -a $test_log
  printf "Failed: $num_failed \n" | tee -a $test_log
  num_other=$(($num_collected_tests - $num_succ - $num_failed - $num_skipped))
  if [ $num_other -gt 0 ]; then
    printf "Other (usually tests skipped due to prior test failure): $num_other \n" | tee -a $test_log
  fi
  printf '\n' | tee -a  $test_log
}

show_summary(){
  local test_log="$1"
  # summarize test report
  echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
  printf "Finished Tests: \n" | tee -a  $test_log
  echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
  printf "$report" | tee -a $test_log
}

show_final_summary(){
  local test_log="$1"
  local tmp_raw_log="${2:-}"
  show_summary "$test_log"
  show_test_counts "$test_log"
  show_elapsed_time "$test_log"
  exit_with_status "$test_log"
}

exit_with_status(){
  local test_log="$1"
  exit_code=0
  if [ $num_failed -gt 0 ] || [ $num_other -gt 0 ]; then
    exit_code=1
    printf "**Failure (${num_failed}) or other (${num_other}) test counts were greater than 0**! \n" | tee -a $test_log
  else
    printf "Failure (${num_failed}) and other (${num_other}) test counts were not greater than 0. \n" | tee -a $test_log
  fi
  printf "Exiting with status code ${exit_code}. \n" | tee -a $test_log
  exit $exit_code
}

ensure_tests(){
  if [ -n "$no_tests_collected" ]; then
    exit 0
  fi
}

show_test_results(){
  if [ -n "$collection_failed" ]; then
    exit 1
  fi
  ensure_tests
  local test_log="$1"
  local tmp_raw_log="$2"
  if [ -f ${tmp_raw_log} ]; then
    raw_filename=$(basename "$tmp_raw_log")
    # a fresh checkout has no ./lightning_logs, and a failed cp here aborts this EXIT trap under `set -e`
    # before any results are reported
    mkdir -p ./lightning_logs
    cp "$tmp_raw_log" "./lightning_logs/$raw_filename"
    echo -n "##vso[task.uploadfile]"
    echo "${PWD}/lightning_logs/$raw_filename"
    # `failed` as a word, so pytest's "xfailed" count is not one; the lines printed are the ones detected, and printing
    # none must not end this EXIT trap under `set -eo pipefail` before the summary and exit status are reported
    local error_pattern='error|exception|traceback|\bfailed\b'
    if grep --quiet --ignore-case --extended-regexp "$error_pattern" ${tmp_raw_log}; then
      echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
      printf "Potential errors detected. Uploading ${tmp_raw_log} and grepping exception/error lines below: \n" | tee -a $test_log
      echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
      printf ": \n" | tee -a $test_log
      grep --ignore-case --extended-regexp "$error_pattern" ${tmp_raw_log} | tee -a $test_log || true
      printf "\n" | tee -a $test_log
      show_final_summary "$test_log"
    else
      printf "No detected errors. \n" | tee -a $test_log
      printf "\n" | tee -a $test_log
      show_final_summary "$test_log"
    fi
  elif [ -f ${test_log} ]; then  # if the summary log but not the raw test log exists, check for collection errors
    if grep --ignore-case --extended-regexp 'traceback|failed' ${test_log} ; then
      echo "Potential collection error!" | tee -a $test_log
      show_final_summary "$test_log"
      exit 1
    fi
  fi
}
