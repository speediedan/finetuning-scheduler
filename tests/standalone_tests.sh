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
# Originally based on https://bit.ly/3AZGVVT
set -e

# this environment variable allows special tests to run
export PL_RUN_STANDALONE_TESTS=1
# python arguments
defaults='-m coverage run --source src/finetuning_scheduler --append -m pytest --capture=no --no-header -v -s'

# find tests marked as `@RunIf(standalone=True)`. done manually instead of with pytest because it is faster
grep_output=$(grep --recursive --word-regexp 'tests' --regexp 'standalone=True' --include '*.py' --exclude 'tests/conftest.py')

# file paths, remove duplicates
files=$(echo "$grep_output" | cut -f1 -d: | sort | uniq)

# get the list of parametrizations. we need to call them separately. the last two lines are removed.
# note: if there's a syntax error, this will fail with some garbled output
if [[ "$OSTYPE" == "darwin"* ]]; then
  parametrizations=$(pytest $files --collect-only --quiet "$@" | tail -r | sed -e '1,3d' | tail -r)
else
  parametrizations=$(pytest $files --collect-only --quiet "$@" | head -n -2)
fi
parametrizations_arr=($parametrizations)

# tests to skip - space separated
blocklist=''
report=''

rm -f standalone_test_output.txt  # in case it exists, remove it
function show_output {
  if [ -f standalone_test_output.txt ]; then  # if exists
    cat standalone_test_output.txt
    # heuristic: stop if there's mentions of errors. this can prevent false negatives when only some of the ranks fail
    if grep --quiet --ignore-case --extended-regexp 'error|exception|traceback|failed' standalone_test_output.txt; then
      echo "Potential error! Stopping."
      rm standalone_test_output.txt
      exit 1
    fi
    rm standalone_test_output.txt
  fi
}
trap show_output EXIT  # show the output on exit

for i in "${!parametrizations_arr[@]}"; do
  parametrization=${parametrizations_arr[$i]}

  # check blocklist
  if echo $blocklist | grep -F "${parametrization}"; then
    report+="Skipped\t$parametrization\n"
    continue
  fi

  # run the test
  echo "Running ${parametrization}"
  python ${defaults} "${parametrization}"

  report+="Ran\t$parametrization\n"
done

show_output

# echo test report
printf '=%.s' {1..80}
printf "\n$report"
printf '=%.s' {1..80}
printf '\n'
