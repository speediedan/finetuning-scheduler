#!/usr/bin/env bash
# Print the standalone scope this build runs: `smoke` (tests/.standalone_smoke) or `full` (every standalone test).
#
#   standalone_scope.sh
#
# A pull request into `main` runs the smoke set unless it carries the FULL_LABEL label, which is read when the build
# runs, so a label added before the gate is approved still counts. Everything else runs the full suite: release
# branches and tags, builds of main, manual runs (including one queued on a pull request's merge ref), and pull
# requests into a release branch.
#
# A label lookup that fails runs the full suite: a missed smoke run costs minutes, a missed full run costs coverage.
set -uo pipefail

FULL_LABEL=${FULL_LABEL:-full-gpu-suite}

scope() { echo "$1"; echo "standalone scope: $1 ($2)" >&2; exit 0; }

[[ "${BUILD_REASON:-}" == "PullRequest" ]] || scope full "build reason ${BUILD_REASON:-unknown}"
target=${SYSTEM_PULLREQUEST_TARGETBRANCH:-}
[[ "${target#refs/heads/}" == "main" ]] || scope full "a pull request into ${target:-an unknown branch}"

url="https://api.github.com/repos/${BUILD_REPOSITORY_NAME:-}/issues/${SYSTEM_PULLREQUEST_PULLREQUESTNUMBER:-}/labels?per_page=100"
if ! labels=$(curl -fsS --max-time 30 "$url"); then
    scope full "could not read the pull request's labels from ${url}"
fi
if ! has_label=$(python3 -c 'import json, sys; print(sys.argv[1] in {l["name"] for l in json.load(sys.stdin)})' \
        "$FULL_LABEL" <<< "$labels"); then
    scope full "could not parse the pull request's labels"
fi
[[ "$has_label" == "True" ]] && scope full "the pull request carries the ${FULL_LABEL} label"
scope smoke "a pull request into main without the ${FULL_LABEL} label"
