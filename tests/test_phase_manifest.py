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
"""CPU tests for the GPU CI phase manifests and reports (``tests/gpu_ci``)."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.gpu_ci import phase_manifest as pm
from tests.gpu_ci.phase_plugin import node_outcome

# The release checker's synthetic manifest and the digest it records for it: the rule here must give the same bytes.
CHECKER_EXAMPLE = {
    "example_only": True,
    "repository": "speediedan/finetuning-scheduler",
    "revision": "synthetic-fixture-not-execution-evidence",
    "phase": "standalone",
    "scope": "full",
    "selector": "PL_RUN_STANDALONE_TESTS=1; synthetic standalone selection",
    "raw_node_ids": [f"synthetic/test_fixture.py::test_case_{i}" for i in range(12)],
    "selected_node_ids": [f"synthetic/test_fixture.py::test_case_{i}" for i in range(6)],
}
CHECKER_EXAMPLE_SHA256 = "334d2f4bde964b57544331554f1867c9d679218b2785813821b7e56e7a05a174"

SYNTHETIC_TESTS = """
import os
import pytest

def test_pass():
    pass

def test_fail():
    assert False

@pytest.mark.skip(reason="synthetic skip")
def test_skip():
    pass

@pytest.mark.xfail(reason="synthetic xfail", strict=True)
def test_xfail():
    assert False

@pytest.mark.xfail(reason="synthetic xpass", strict=False)
def test_xpass():
    pass

@pytest.fixture
def broken():
    raise RuntimeError("setup error")

def test_setup_error(broken):
    pass
"""
# Selects on an environment variable, as tests/conftest.py does, so raw and selected collections differ.
SYNTHETIC_CONFTEST = """
import os

def pytest_collection_modifyitems(items):
    if os.getenv("FTS_CUDA_TESTS_ONLY") == "1":
        items[:] = [i for i in items if not i.name.startswith("test_pass_only_raw")]
"""


def _session(rank, events, finish=True, collected=("t.py::test_a",)):
    records = [{"event": "start", "rank": rank}, {"event": "collected", "nodeids": list(collected), "rank": rank}]
    records += [{"event": "outcome", "nodeid": n, "when": "call", "outcome": o, "rank": rank} for n, o in events]
    if finish:
        records.append({"event": "finish", "exitstatus": 0, "rank": rank})
    return records


def _manifest(selected, raw=None):
    return {
        "repository": pm.REPOSITORY,
        "revision": "r",
        "phase": "standalone",
        "scope": "full",
        "selector": "s",
        "raw_node_ids": list(raw or selected),
        "selected_node_ids": list(selected),
        "collection_errors": [],
    }


def _report(manifest, sessions, **kwargs):
    return pm.build_report(manifest, sessions, vendor="amd", evidence_class="parity-pre-agent", **kwargs)


def test_digest_matches_release_checker_rule():
    assert pm.manifest_digest(CHECKER_EXAMPLE) == CHECKER_EXAMPLE_SHA256
    reordered = dict(reversed(list(CHECKER_EXAMPLE.items())))
    assert pm.manifest_digest(reordered) == CHECKER_EXAMPLE_SHA256


@pytest.mark.parametrize(
    "when, outcome, wasxfail, expected",
    [
        ("call", "failed", False, "failed"),
        ("setup", "failed", False, "errors"),
        ("teardown", "failed", False, "errors"),
        ("setup", "skipped", False, "skipped"),
        ("call", "skipped", True, "xfailed"),
        ("call", "passed", False, "passed"),
        ("call", "passed", True, "xpassed"),
        ("setup", "passed", False, None),
        ("teardown", "passed", False, None),
    ],
)
def test_node_outcome(when, outcome, wasxfail, expected):
    report = SimpleNamespace(
        when=when, failed=outcome == "failed", skipped=outcome == "skipped", passed=outcome == "passed"
    )
    if wasxfail:
        report.wasxfail = "reason"
    assert node_outcome(report) == expected


def test_phase_table_matches_pipeline_and_release_scope():
    assert list(pm.PHASES) == ["gpu-nonstandalone", "standalone", "examples", "examples-multigpu", "pool-reverse"]
    assert [n for n, p in pm.PHASES.items() if p.release] == list(pm.PHASES)[:4]
    modules = pm.pool_modules()
    assert modules and modules == tuple(sorted(modules))
    assert all(pm.POOL_FIXTURE in (pm.ROOT / m).read_text(encoding="utf-8") for m in modules)
    # this module must not join the reverse-order leg by naming the fixture itself
    assert "tests/test_phase_manifest.py" not in modules
    assert [n for n, p in pm.PHASES.items() if not p.measured] == ["examples", "pool-reverse"]
    # a measured phase's command must write coverage (the special-test runner runs every test under ``coverage run``)
    for name, phase in pm.PHASES.items():
        measures = "coverage run" in phase.command or "gpu_epsilon.special_tests" in phase.command
        assert measures == phase.measured, name
    selector = pm.selector_string(pm.PHASES["pool-reverse"], "full")
    assert selector.startswith("FTS_REVERSE_ORDER=1 PL_RUN_CUDA_TESTS=1; pytest ")
    assert all(m in selector for m in modules)
    assert pm.selector_string(pm.PHASES["standalone"], "smoke").endswith(f"; select_file={pm.SMOKE_FILE}")
    with pytest.raises(ValueError, match="scope 'smoke' is not defined"):
        pm.build_manifest("examples", "smoke")


def test_reviewed_exclusions_name_existing_tests():
    # every reviewed skip or xfail names a test that still exists (its module, function and parameter id) and gives a
    # reason, so a renamed test cannot leave a stale entry that makes some other skip look reviewed
    reviewed = json.loads((pm.ROOT / "tests/gpu_ci/reviewed_exclusions.json").read_text(encoding="utf-8"))
    assert reviewed
    for nodeid, reason in reviewed.items():
        assert reason.strip(), nodeid
        module, _, test = nodeid.partition("::")
        name, _, param = test.partition("[")
        source = (pm.ROOT / module).read_text(encoding="utf-8")
        assert f"def {name}(" in source, nodeid
        assert not param or f'"{param.rstrip("]")}"' in source, nodeid


def test_run_env_records_every_rank(tmp_path):
    env = pm.run_env("standalone", tmp_path)
    assert env["PL_RUN_STANDALONE_TESTS"] == "1" and env["PL_RUN_CUDA_TESTS"] == "1"
    assert env[pm.RECORD_DIR_ENV] == str(tmp_path.resolve() / "records" / "standalone")
    assert env["PYTEST_ADDOPTS"].endswith(f"-p {pm.PLUGIN}")
    # a measured phase gets its own coverage data file; an unmeasured one none
    assert env["COVERAGE_FILE"] == str(pm.coverage_file(tmp_path, "standalone"))
    assert "COVERAGE_FILE" not in pm.run_env("examples", tmp_path)


def _measure(tmp_path, out_dir, phase, body):
    """Run a tiny module under coverage into ``phase``'s data file, as a measured phase's command would."""
    (tmp_path / "mod.py").write_text(body)
    data = pm.coverage_file(out_dir, phase)
    data.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "COVERAGE_FILE": str(data)}
    subprocess.run([sys.executable, "-m", "coverage", "run", "mod.py"], cwd=tmp_path, env=env, check=True)


def test_coverage_summary_and_diff(tmp_path):
    ref, cand = tmp_path / "ref", tmp_path / "cand"
    _measure(tmp_path, ref, "standalone", "x = 1\nif x:\n    y = 2\nelse:\n    y = 3\n")
    _measure(tmp_path, cand, "standalone", "x = 0\nif x:\n    y = 2\nelse:\n    y = 3\n")
    summary = pm.coverage_summary("standalone", ref, root=tmp_path)
    assert summary["measured"] and not summary["missing"] and summary["files"] == 1
    assert summary["covered_lines"] == 3 and summary["json"] == "coverage/standalone.json"
    assert pm.coverage_summary("examples", ref, root=tmp_path) == {"measured": False}
    assert pm.coverage_summary("gpu-nonstandalone", ref, root=tmp_path)["missing"]
    pm.coverage_summary("standalone", cand, root=tmp_path)
    diff = pm.coverage_diff(ref, cand, ["standalone"])
    # the reference executed line 3, the candidate line 5 instead: one line lost, one gained
    assert diff["standalone"]["lost"] == {"mod.py": [3]} and diff["standalone"]["gained"] == {"mod.py": [5]}
    assert diff["union"]["kept_fraction"] == round(2 / 3, 4)
    assert pm.main(["coverage-diff", str(ref), str(ref), "--phase", "standalone"]) == 0
    assert pm.main(["coverage-diff", str(ref), str(cand), "--phase", "standalone"]) == 1
    # a measured phase without coverage on one side is an evidence gap, never a pass
    assert pm.main(["coverage-diff", str(ref), str(cand), "--phase", "gpu-nonstandalone"]) == 1


def test_collect_run_and_report_end_to_end(tmp_path):
    """Collect a synthetic tree raw and selected through the plugin, run it recorded, and report it."""
    (tmp_path / "test_synthetic.py").write_text(SYNTHETIC_TESTS + "\ndef test_pass_only_raw():\n    pass\n")
    (tmp_path / "conftest.py").write_text(SYNTHETIC_CONFTEST)
    raw, raw_errors, _ = pm._collect(("test_synthetic.py",), (), {}, tmp_path)
    selected, sel_errors, _ = pm._collect(("test_synthetic.py",), (), {"FTS_CUDA_TESTS_ONLY": "1"}, tmp_path)
    assert not raw_errors and not sel_errors
    assert len(raw) == 7 and len(selected) == 6 and "test_synthetic.py::test_pass_only_raw" not in selected
    record_dir = tmp_path / "records"
    env = {k: v for k, v in os.environ.items() if k not in pm.SELECTION_VARS}
    env.update(FTS_CUDA_TESTS_ONLY="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(pm.ROOT))
    env[pm.RECORD_DIR_ENV] = str(record_dir)
    cmd = [sys.executable, "-m", "pytest", "test_synthetic.py", "-q", "-p", "no:cacheprovider", "-p", pm.PLUGIN]
    subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    report = _report(_manifest(selected, raw), pm.read_records(record_dir))
    o = report["outcomes"]
    assert (o["passed"], o["failed"], o["errors"], o["skipped"], o["xfailed"], o["xpassed"]) == (1, 1, 1, 1, 1, 1)
    assert o["attempted"] == 6 and o["unexecuted"] == 0
    assert report["collection"] == {"raw": 7, "deselected": 1, "selected": 6, "errors": 0}
    assert report["classification"] == "failing" and report["evidence_class"] == "parity-pre-agent"
    assert report["nodes"]["test_synthetic.py::test_setup_error"]["outcome"] == "errors"


def test_report_counts_ranks_completions_and_drift():
    a, b, c = "t.py::test_a", "t.py::test_b", "t.py::test_c"
    clean = _report(_manifest([a]), [_session(0, [(a, "passed")]), _session(1, [(a, "passed")])])
    assert clean["classification"] == "clean" and clean["nodes"][a]["ranks"] == {"0": "passed", "1": "passed"}
    assert clean["manifest_sha256"] == pm.manifest_digest(_manifest([a]))
    # rank 0 passed, but another rank failed: the node fails
    rank_fail = _report(_manifest([a]), [_session(0, [(a, "passed")]), _session(1, [(a, "failed")])])
    assert rank_fail["outcomes"]["failed"] == 1 and rank_fail["rank_failures"] == 1
    # a rank that started and never finished fails the node it gave no verdict for
    hung = _report(_manifest([a]), [_session(0, [(a, "passed")]), _session(1, [], finish=False)])
    assert hung["missing_rank_completions"] == 1 and hung["outcomes"]["failed"] == 1
    assert hung["nodes"][a]["ranks"]["1"] == "incomplete" and hung["classification"] == "failing"
    # a selected node with no rank-0 verdict is unexecuted; a node outside the manifest is a collection error
    drift = _report(_manifest([a, b]), [_session(0, [(a, "passed"), (c, "passed")], collected=(a, b, c))])
    assert drift["outcomes"]["unexecuted"] == 1 and drift["nodes"][b]["outcome"] == "unexecuted"
    assert drift["unmanifested_nodes"] == [c] and drift["collection"]["errors"] == 1
    # skips need a reviewed reason, and a phase with nothing passed is never clean
    skips = _report(_manifest([a, b]), [_session(0, [(a, "passed"), (b, "skipped")], collected=(a, b))])
    assert skips["classification"] == "unreviewed-exclusions" and not skips["exclusions_reviewed"]
    reviewed = _report(
        _manifest([a, b]),
        [_session(0, [(a, "passed"), (b, "skipped")], collected=(a, b))],
        reviewed_exclusions={b: "DeepSpeed not installed"},
    )
    assert reviewed["classification"] == "clean" and reviewed["exclusions"][0]["reason"] == "DeepSpeed not installed"
    empty = _report(_manifest([b]), [_session(0, [(b, "skipped")], collected=(b,))], reviewed_exclusions={b: "x"})
    assert empty["classification"] == "empty"


def test_report_refuses_unknown_vendor_or_evidence_class():
    with pytest.raises(ValueError, match="vendor"):
        pm.build_report(_manifest([]), [], vendor="intel", evidence_class="pipeline")
    with pytest.raises(ValueError, match="evidence class"):
        pm.build_report(_manifest([]), [], vendor="nvidia", evidence_class="release")


def test_cli_report_round_trip(tmp_path, monkeypatch):
    # device names would initialize CUDA in the test process; the identity is checked on its own below
    monkeypatch.setattr(pm, "device_identity", lambda: {"device_count": 0, "devices": []})
    a = "t.py::test_a"
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "examples.json").write_text(json.dumps({**_manifest([a]), "phase": "examples"}))
    records = tmp_path / "records" / "examples"
    records.mkdir(parents=True)
    (records / "records-rank0-pid1.jsonl").write_text("\n".join(json.dumps(r) for r in _session(0, [(a, "passed")])))
    args = ["report", "--phase", "examples", "--out-dir", str(tmp_path), "--vendor", "nvidia"]
    assert pm.main([*args, "--evidence-class", "pipeline", "--lease-mode", "tool"]) == 0
    report = json.loads((tmp_path / "reports" / "examples.json").read_text(encoding="utf-8"))
    assert report["backend"] == "cuda" and report["lease_mode"] == "tool" and "python" in report["versions"]
    assert {"lightning_commit", "torch_record_sha256", "device_count"} <= set(report["versions"])
    assert report["coverage"] == {"measured": False} and report["nodes"][a]["world_size"] is None


def test_device_identity_records_visible_device_variables(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    ident = pm.device_identity()
    assert ident["ROCR_VISIBLE_DEVICES"] == "GPU-a,GPU-b" and ident["device_count"] == 0 and ident["devices"] == []
    assert set(pm.DEVICE_VARS) <= set(ident)


def test_build_identity_reads_installed_distributions():
    assert pm._record_sha256("torch") is not None and len(pm._record_sha256("torch")) == 64
    assert pm._record_sha256("not-an-installed-distribution") is None
    assert pm._vcs_commit("not-an-installed-distribution") is None
