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
"""CPU tests for the GPU job's lease wait (``.azure-pipelines/scripts/gpu_job_container.sh lease``), run against a
fake host lease tool that times out (exit 75) a given number of times.

A build that is not a scheduled one waits again while its lease budget lasts, logging who holds the leases, and fails
only once the budget is spent; a scheduled run keeps its single wait.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".azure-pipelines" / "scripts" / "gpu_job_container.sh"


def _bash_major() -> int:
    bash = shutil.which("bash")
    if not bash:
        return 0
    out = subprocess.run([bash, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


pytestmark = pytest.mark.skipif(sys.platform == "win32" or _bash_major() < 4, reason="needs bash 4 (the GPU job's)")

FAKE_TOOL = """#!/usr/bin/env bash
# a host lease tool that times out FAKE_FAILS times, then grants
case "$1" in
  --status) echo "holder: local-suite-1 since 10:00 (pytest)"; exit 0 ;;
  --hold)
    n=$(( $(cat "$FAKE_COUNT" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAKE_COUNT"
    while [[ $# -gt 0 ]]; do [[ "$1" == --timeout ]] && echo "$2" >> "$FAKE_COUNT.timeouts"; shift; done
    (( n > FAKE_FAILS )) || exit 75
    exit 0 ;;
esac
"""


def _lease(tmp_path: Path, fails: int, **env) -> tuple[subprocess.CompletedProcess, int, list[int]]:
    (tmp_path / "leases").mkdir()
    tool = tmp_path / "tool"
    tool.write_text(FAKE_TOOL)
    tool.chmod(0o755)
    count = tmp_path / "count"
    run_env = {
        k: v for k, v in os.environ.items() if not k.startswith(("FTS_LEASE", "BUILD_"))
    } | {
        "FTS_LEASE_DIR": str(tmp_path / "leases"),
        "FTS_LEASE_TOOL": str(tool),
        "FTS_LEASE_PIDFILE": str(tmp_path / "lease.pid"),
        "FAKE_COUNT": str(count),
        "FAKE_FAILS": str(fails),
        "BUILD_BUILDID": "1",
        **env,
    }
    r = subprocess.run(["bash", str(SCRIPT), "lease"], cwd=ROOT, env=run_env, capture_output=True, text=True,
                       timeout=60)
    holds = int(count.read_text()) if count.exists() else 0
    timeouts = [int(x) for x in (tmp_path / "count.timeouts").read_text().split()] if holds else []
    return r, holds, timeouts


def test_a_timed_out_wait_is_retried_while_budget_is_left(tmp_path):
    r, holds, timeouts = _lease(tmp_path, fails=2, FTS_LEASE_WAIT="1", FTS_LEASE_BUDGET="3",
                                BUILD_REASON="PullRequest")
    assert r.returncode == 0, r.stdout + r.stderr
    assert holds == 3 and timeouts == [1, 1, 1]
    assert r.stdout.count("waiting again") == 2 and "holder: local-suite-1" in r.stdout


def test_a_spent_budget_fails_naming_the_holders(tmp_path):
    r, holds, _ = _lease(tmp_path, fails=99, FTS_LEASE_WAIT="1", FTS_LEASE_BUDGET="2", BUILD_REASON="IndividualCI")
    assert r.returncode == 75 and holds == 2
    assert "whole lease budget (2s)" in r.stdout and "holder: local-suite-1" in r.stdout


def test_the_default_budget_is_two_waits(tmp_path):
    r, holds, _ = _lease(tmp_path, fails=99, FTS_LEASE_WAIT="1", BUILD_REASON="PullRequest")
    assert r.returncode == 75 and holds == 2


def test_a_scheduled_run_keeps_its_single_wait(tmp_path):
    r, holds, timeouts = _lease(tmp_path, fails=99, FTS_LEASE_WAIT="1", BUILD_REASON="Schedule")
    assert r.returncode == 75 and holds == 1 and timeouts == [1]
    assert "waiting again" not in r.stdout


def test_a_free_host_is_held_at_once(tmp_path):
    r, holds, timeouts = _lease(tmp_path, fails=0, BUILD_REASON="PullRequest")
    assert r.returncode == 0 and holds == 1 and timeouts == [3600]


def test_the_job_timeout_fits_the_budget():
    text = (ROOT / ".azure-pipelines" / "templates" / "fts-gpu-job.yml").read_text()
    assert "timeoutInMinutes: 195" in text and "timeoutInMinutes: 150" in text
    # two 60-minute waits and the longest measured run (53 minutes) fit with room to spare
    assert 2 * 60 + 53 < 195
