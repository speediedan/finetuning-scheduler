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
"""A pytest plugin recording what a GPU CI phase collected and how each node ended, per rank.

Load it with ``-p tests.gpu_ci.phase_plugin`` (or through ``PYTEST_ADDOPTS``, which every rank's re-launched pytest
session inherits, so the special-test runner needs no change). It does nothing unless ``FTS_PHASE_RECORD_DIR`` names
a directory; then each pytest session (each rank's, under Lightning's ``ddp`` launcher) appends JSON lines to its own
file there:

- ``{"event": "start"}`` when the session starts;
- ``{"event": "collect_error", "nodeid": ...}`` for each collector that failed (an import error, say);
- ``{"event": "collected", "nodeids": [...]}`` after collection, so after ``tests/conftest.py`` has selected the phase's
  tests (the node ids the phase runs, in the order it runs them);
- ``{"event": "outcome", "nodeid": ..., "outcome": ...}`` for each test stage that decides a node's outcome;
- ``{"event": "finish", "exitstatus": ...}`` when the session finishes. A rank that died or hung in a collective never
  writes it, which ``phase_manifest.py`` counts as a missing rank completion.

Every line carries the session's ``rank`` (``LOCAL_RANK``, else ``RANK``, else 0 for the launching process),
``world_size`` when the launcher set it, and ``pid``.
"""
import json
import os
from pathlib import Path

RECORD_DIR_ENV = "FTS_PHASE_RECORD_DIR"


def _identity() -> dict:
    world_size = os.environ.get("WORLD_SIZE", "")
    return {
        "rank": int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))),
        "world_size": int(world_size) if world_size.isdigit() else None,
        "pid": os.getpid(),
    }


def node_outcome(report) -> str | None:
    """The phase-report outcome one ``pytest_runtest_logreport`` report decides, or ``None`` when it decides
    nothing.

    Mirrors pytest's own terminal categories: a failed setup or teardown is an error, a failed call a failure, an
    expected failure ``xfailed`` (a strict XPASS already arrives as a failed call), a skip at any stage ``skipped``.
    """
    if report.failed:
        return "failed" if report.when == "call" else "errors"
    if report.skipped:
        return "xfailed" if hasattr(report, "wasxfail") else "skipped"
    if report.when == "call":
        return "xpassed" if hasattr(report, "wasxfail") else "passed"
    return None


class _PhaseRecorder:
    """Appends one session's records to its own JSON-lines file, so ranks never interleave writes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**record, **_identity()}) + "\n")

    def pytest_sessionstart(self, session):
        self._write({"event": "start"})

    def pytest_collectreport(self, report):
        if report.failed:
            self._write({"event": "collect_error", "nodeid": report.nodeid})

    def pytest_collection_finish(self, session):
        self._write({"event": "collected", "nodeids": [item.nodeid for item in session.items]})

    def pytest_runtest_logreport(self, report):
        outcome = node_outcome(report)
        if outcome is not None:
            self._write({"event": "outcome", "nodeid": report.nodeid, "when": report.when, "outcome": outcome})

    def pytest_sessionfinish(self, session, exitstatus):
        self._write({"event": "finish", "exitstatus": int(exitstatus)})


def pytest_configure(config):
    record_dir = os.environ.get(RECORD_DIR_ENV)
    if not record_dir:
        return
    Path(record_dir).mkdir(parents=True, exist_ok=True)
    ident = _identity()
    path = Path(record_dir) / f"records-rank{ident['rank']}-pid{ident['pid']}.jsonl"
    config.pluginmanager.register(_PhaseRecorder(path), "fts_phase_recorder")
