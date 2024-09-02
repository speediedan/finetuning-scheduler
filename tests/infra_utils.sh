#!/bin/bash
# Test infra utility functions
# Note we use local variables for many of these to allow more usage flexibility in different contexts

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
  if special_tests=$(python3 ${collect_def}); then
    # match only lines with tests
    declare -a -g parameterizations=($(grep -oP '\S+::test_\S+' <<< "$special_tests"))
    echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $collect_log
    printf "Collected the following tests: \n" | tee -a  $collect_log
    echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $collect_log
    printf '%s\n' "${parameterizations[@]}" | tee -a  $collect_log
    num_collected_tests="${#parameterizations[@]}"
    echo "Total number of tests: ${#parameterizations[@]}" | tee -a  $collect_log
    printf '\n' | tee -a  $collect_log
  else
    printf "No tests were found with the following collection command: python3 ${collect_def} \n" | tee -a $collect_log
    printf "Exiting without running tests. \n" | tee -a $collect_log
    export no_tests_collected=1
    exit 0
  fi
}

execute_tests(){
  ensure_tests
  local execute_def="$1"
  local execute_log="$2"
  local tmp_out="$3"
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
    (python ${execute_def} ${parameterization} 2>&1 | sed "s,\x1b\[[0-9;]*[a-zA-Z],,g" >> $tmp_out) > /dev/null
    test_to_find=`echo ${parameterization} | sed 's/\[/\\\[/g; s/\]/\\\]/g'`
    if pass_or_fail=$(grep -E "(PASSED|FAILED|XPASS|XFAIL) .*${test_to_find}" $tmp_out); then
      parameterization_result=`echo $pass_or_fail | awk 'NR==1 {print $2 ": "  $1}'`;
    elif skipped=$(grep -E "${test_to_find}.*SKIPPED" $tmp_out); then
      parameterization_result=`echo $skipped | awk 'NR==1 {print $1 ": "  $2}'`;
    else
      echo "Could not parse result!" | tee -a $execute_log
      parameterization_result="UNKNOWN: see $tmp_out"
    fi
    report+="Ran\t${parameterization_result}\n"
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
  local tmp_out="${2:-}"
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
  ensure_tests
  local test_log="$1"
  local tmp_out="$2"
  if [ -f ${tmp_out} ]; then
    if grep_errors=($(grep --ignore-case --extended-regexp 'error|exception|traceback|failed' ${tmp_out})); then
      echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
      printf "Potential errors detected. See ${tmp_out} for details. Exception/error lines to follow. \n" | tee -a $test_log
      echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
      printf "\n" | tee -a $test_log
      show_final_summary "$test_log"
      echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
      printf "Grepped exception/error lines: \n" | tee -a $test_log
      echo `printf "%0.s-" {1..120} && printf "\n"` | tee -a $test_log
      grep --ignore-case --extended-regexp 'error|exception' ${tmp_out} | tee -a $test_log
      printf "\n" | tee -a $test_log
    else
      printf "No detected errors. \n" | tee -a $test_log
      printf "\n" | tee -a $test_log
      show_final_summary "$test_log"
    fi
  elif [ -f ${test_log} ]; then  # if the log but not the out exists, check for collection errors
    if grep --ignore-case --extended-regexp 'traceback|failed' ${test_log} ; then
      echo "Potential collection error!" | tee -a $test_log
      show_final_summary "$test_log"
      exit 1
    fi
  fi
}

show_elapsed_time(){
  local test_log="$1"
  script_name=${2:-$(basename "$0")}
  ## write elapsed time in user-friendly fashion
  end_time=$(date +%s)
  elapsed_seconds=$(($end_time-$start_time))
  if (( $elapsed_seconds/60 == 0 )); then
      printf "${script_name} completed in $elapsed_seconds seconds \n" | tee -a $test_log
  elif (( $elapsed_seconds%60 == 0 )); then
      printf  "${script_name} completed in $(($elapsed_seconds/60)) minutes \n" | tee -a $test_log
  else
      printf "${script_name} completed in $(($elapsed_seconds/60)) minutes and $(($elapsed_seconds%60)) seconds \n" | tee -a $test_log
  fi
  printf "\n" | tee -a $test_log
}
