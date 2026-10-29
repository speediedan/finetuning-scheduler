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
"""CPU tests for the release eligibility check: the shared checker (``gpu_epsilon.release_eligibility``) over this
project's own evidence, as ``tests/gpu_ci/phase_manifest.py`` writes it, with this project's configuration
(``tests/gpu_ci/release.json``).

The shared checker's rules are tested in gpu-epsilon. These cases are the contract: they run it over reports built by
this project's report writer, and check that the configuration agrees with the phases and the pipelines.
"""
import hashlib
import json
from pathlib import Path

import pytest
from gpu_epsilon import release_eligibility as re_

from tests.gpu_ci import phase_manifest as pm

CONFIG_PATH = Path(__file__).parent / "gpu_ci" / "release.json"
CONFIG = re_.ReleaseConfig.load(CONFIG_PATH)

REVISION = "0" * 40
IMAGES = {"nvidia": "fts:cuda@sha256:" + "a" * 64, "amd": "fts:rocm@sha256:" + "b" * 64}
TORCH = {"nvidia": "2.15.0+cu132", "amd": "2.15.0+rocm10.0"}
NODES = [f"tests/test_x.py::test_{i}" for i in range(4)]
COVERAGE_EXPORT = b'{"totals": {"covered_lines": 10}}'


def _manifest(phase):
    return {
        "repository": pm.REPOSITORY,
        "revision": REVISION,
        "phase": phase,
        "scope": "full",
        "selector": f"selector for {phase}",
        "tree_dirty_paths": [],
        "raw_node_ids": NODES + ["tests/test_x.py::test_deselected"],
        "selected_node_ids": list(NODES),
        "collection_errors": [],
        "collect_exit_codes": {"raw": 0, "selected": 0},
    }


def _session(outcomes):
    records = [{"event": "start", "rank": 0}, {"event": "collected", "nodeids": list(NODES), "rank": 0}]
    records += [{"event": "outcome", "nodeid": n, "when": "call", "outcome": o, "rank": 0} for n, o in outcomes.items()]
    return records + [{"event": "finish", "exitstatus": 0, "rank": 0}]


def _coverage(phase):
    if not pm.PHASES[phase].measured:
        return {"measured": False}
    sha = hashlib.sha256(COVERAGE_EXPORT).hexdigest()
    return {
        "measured": True,
        "missing": False,
        "json": f"coverage/{phase}.json",
        "json_sha256": sha,
        "covered_lines": 10,
    }


def _leg(vendor, outcomes=None, reviewed=None):
    versions = {
        "python": "3.13.16",
        "torch": TORCH[vendor],
        "torch_git_version": "c" * 40,
        "lightning": "2.6.1",
        "lightning_commit": None,
        "finetuning_scheduler": "2.15.0",
    }
    leg = {}
    for phase in CONFIG.phases:
        manifest = _manifest(phase)
        report = pm.build_report(
            manifest,
            [_session(outcomes or dict.fromkeys(NODES, "passed"))],
            vendor=vendor,
            evidence_class="pipeline",
            lease_mode="tool",
            image=IMAGES[vendor],
            versions=versions,
            reviewed_exclusions=reviewed,
            coverage=_coverage(phase),
        )
        coverage_sha = report["coverage"].get("json_sha256")
        leg[phase] = {"report": report, "manifest": manifest, "coverage_sha256": coverage_sha}
    return leg


def _legs():
    return {vendor: _leg(vendor) for vendor in pm.VENDORS}


def _check(legs, results=None):
    return re_.check_release(CONFIG, legs, results or {"nvidia": "Succeeded", "amd": "Succeeded"}, REVISION)


def _write(root, vendor, leg, attempt=1):
    phases = root / f"fts-release-{vendor}-{attempt}" / "fts-phases"
    for kind in ("reports", "manifests"):
        (phases / kind).mkdir(parents=True, exist_ok=True)
    for phase, docs in leg.items():
        (phases / "reports" / f"{phase}.json").write_text(json.dumps(docs["report"]))
        (phases / "manifests" / f"{phase}.json").write_text(json.dumps(docs["manifest"]))
        if docs["report"]["coverage"].get("json"):
            (phases / "coverage").mkdir(exist_ok=True)
            (phases / docs["report"]["coverage"]["json"]).write_bytes(COVERAGE_EXPORT)


def test_required_phases_are_the_release_phases():
    assert CONFIG.phases == ("gpu-nonstandalone", "standalone", "examples", "examples-multigpu")
    assert CONFIG.phases == tuple(name for name, phase in pm.PHASES.items() if phase.release)


def test_config_agrees_with_the_phase_manifest():
    assert CONFIG.repository == pm.REPOSITORY
    assert CONFIG.vendors == pm.VENDORS
    assert set(CONFIG.measured_phases) == {p for p in CONFIG.phases if pm.PHASES[p].measured}
    # the counts phase_manifest.classify fails a phase on, beyond the outcome counts
    report = pm.build_report(_manifest("standalone"), [], vendor="nvidia", evidence_class="pipeline")
    assert set(CONFIG.failure_counts) == {"rank_failures", "missing_rank_completions"} <= set(report)


def test_config_agrees_with_the_pipelines():
    root = Path(__file__).resolve().parents[1]
    job = (root / ".azure-pipelines" / "templates" / "gpu-job.yml").read_text()
    release = (root / ".azure-pipelines" / "gpu-release.yml").read_text()
    container = (root / ".azure-pipelines" / "scripts" / "gpu_job_container.sh").read_text()
    assert f"artifact: {CONFIG.artifact_prefix}-${{{{ parameters.vendor }}}}-$(System.JobAttempt)" in job
    assert f"out=/evidence/{CONFIG.phases_dir}" in container
    assert "--config tests/gpu_ci/release.json" in release
    assert all(f"--{vendor}-result" in release for vendor in CONFIG.vendors)


def test_two_clean_full_legs_are_eligible():
    verdict = _check(_legs())
    assert verdict["eligible"] and verdict["revision"] == REVISION
    assert {v: e["image"] for v, e in verdict["evidence"].items()} == IMAGES
    assert set(verdict["evidence"]["amd"]["phases"]) == set(CONFIG.phases)


def _set(vendor, phase, key, value, doc="report"):
    def mutate(legs):
        docs = legs[vendor][phase]
        docs[doc][key] = value
        if doc == "manifest":
            # a run that really collected this way: its report carries the changed manifest's digest
            docs["report"]["manifest_sha256"] = pm.manifest_digest(docs["manifest"])

    return mutate


def _drop_phase(legs):
    del legs["amd"]["examples"]


def _drop_vendor(legs):
    del legs["nvidia"]


def _tamper_manifest(legs):
    legs["nvidia"]["standalone"]["manifest"]["selected_node_ids"].append("tests/test_x.py::test_deselected")


def _inflate_passes(legs):
    report = legs["amd"]["standalone"]["report"]
    report["outcomes"]["passed"] += 1
    report["outcomes"]["attempted"] += 1


def _other_selection(legs):
    # a vendor that selected a different node set, consistently with its own manifest and report
    manifest = legs["amd"]["standalone"]["manifest"]
    manifest["raw_node_ids"].append("tests/test_x.py::test_amd_only")
    report = legs["amd"]["standalone"]["report"]
    report["collection"].update(raw=len(manifest["raw_node_ids"]), deselected=report["collection"]["deselected"] + 1)
    report["manifest_sha256"] = pm.manifest_digest(manifest)


def _flip_node(legs):
    # the reviewed failure mode: a node's recorded outcome is a failure while the summary counts still read clean
    legs["amd"]["standalone"]["report"]["nodes"][NODES[0]]["outcome"] = "failed"


def _drop_node(legs):
    del legs["amd"]["standalone"]["report"]["nodes"][NODES[0]]


def _unmanifested(legs):
    legs["nvidia"]["examples"]["report"]["unmanifested_nodes"] = ["tests/test_x.py::test_extra"]


def _skip_without_reason(legs):
    # a skipped node whose exclusion has no reason, in a report that nonetheless claims its exclusions were reviewed
    for vendor in legs:
        legs[vendor] = _leg(vendor, {**dict.fromkeys(NODES, "passed"), NODES[0]: "skipped"}, {NODES[0]: "reviewed"})
    legs["amd"]["examples"]["report"]["exclusions"][0]["reason"] = None


def _drop_exclusion(legs):
    for vendor in legs:
        legs[vendor] = _leg(vendor, {**dict.fromkeys(NODES, "passed"), NODES[0]: "skipped"}, {NODES[0]: "reviewed"})
    legs["nvidia"]["examples"]["report"]["exclusions"] = []


def _coverage_missing(legs):
    legs["amd"]["standalone"]["report"]["coverage"] = {"measured": True, "missing": True}


def _coverage_tampered(legs):
    legs["nvidia"]["gpu-nonstandalone"]["coverage_sha256"] = "d" * 64


def _versions(key, value, vendor="amd"):
    def mutate(legs):
        for docs in legs[vendor].values():
            docs["report"]["versions"] = {**docs["report"]["versions"], key: value}

    return mutate


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (_drop_vendor, "nvidia: no release evidence"),
        (_drop_phase, r"amd: no evidence for required phase\(s\) examples"),
        (_set("nvidia", "standalone", "revision", "1" * 40), "revision"),
        (_set("amd", "examples", "scope", "smoke"), "scope"),
        (_set("amd", "standalone", "evidence_class", "parity-pre-agent"), "evidence_class"),
        (_set("nvidia", "standalone", "lease_mode", "plain"), "lease_mode"),
        (_set("nvidia", "standalone", "backend", "hip"), "backend"),
        (_set("amd", "standalone", "image", "fts:rocm-latest"), "not digest-pinned"),
        (_set("amd", "examples", "image", IMAGES["nvidia"]), "image differs"),
        (_set("nvidia", "standalone", "tree_dirty_paths", ["src/x.py"], doc="manifest"), "dirty tree"),
        (_set("nvidia", "standalone", "collect_exit_codes", {"raw": 0, "selected": 2}, doc="manifest"), "exited"),
        (_tamper_manifest, "manifest digest mismatch"),
        (_inflate_passes, "unaccounted nodes"),
        (_set("amd", "standalone", "classification", "failing"), "report claims"),
        (_other_selection, "the vendors collected or selected different nodes"),
        (_flip_node, "outcome counts disagree with the node records"),
        (_drop_node, "node records are not the selection"),
        (_unmanifested, "nodes ran that the manifest did not select"),
        (_skip_without_reason, "exclusions_reviewed disagrees with reasons"),
        (_drop_exclusion, "exclusions do not list exactly the skipped and xfailed nodes"),
        (_coverage_missing, "coverage evidence is missing"),
        (_coverage_tampered, "coverage export missing or not the one the report digested"),
        (_versions("lightning_commit", "abc123"), "installed versions differ across vendors"),
        (_versions("torch_git_version", "e" * 40), "installed versions differ across vendors"),
        (_versions("lightning", "2.7.0"), "installed versions differ across vendors"),
        (_versions("finetuning_scheduler", None), "installed versions missing"),
    ],
)
def test_ineligible_evidence(mutate, reason):
    legs = _legs()
    mutate(legs)
    with pytest.raises(re_.Ineligible, match=reason):
        _check(legs)


@pytest.mark.parametrize(
    "outcome, classification",
    [("failed", "failing"), ("errors", "failing"), ("xpassed", "failing"), ("skipped", "unreviewed-exclusions")],
)
def test_unclean_phase_is_ineligible(outcome, classification):
    legs = _legs()
    legs["amd"] = _leg("amd", {**dict.fromkeys(NODES, "passed"), NODES[0]: outcome})
    with pytest.raises(re_.Ineligible, match=f"amd/gpu-nonstandalone: phase is {classification}"):
        _check(legs)


def test_reviewed_skip_is_eligible():
    outcomes = {**dict.fromkeys(NODES, "passed"), NODES[0]: "skipped"}
    legs = {v: _leg(v, outcomes, {NODES[0]: "reviewed"}) for v in pm.VENDORS}
    assert _check(legs)["eligible"]


def test_vendor_torch_tags_differ_but_releases_agree():
    verdict = _check(_legs())
    assert {v: e["versions"]["torch"] for v, e in verdict["evidence"].items()} == TORCH


def test_leg_result_must_be_succeeded():
    with pytest.raises(re_.Ineligible, match="amd leg result is 'SucceededWithIssues'"):
        _check(_legs(), {"nvidia": "Succeeded", "amd": "SucceededWithIssues"})


def test_cli_writes_verdict_from_latest_attempt(tmp_path, capsys):
    artifacts, output = tmp_path / "artifacts", tmp_path / "out" / "eligibility.json"
    legs = _legs()
    _write(artifacts, "nvidia", legs["nvidia"])
    _write(artifacts, "amd", _leg("amd", {**dict.fromkeys(NODES, "passed"), NODES[0]: "failed"}), attempt=1)
    _write(artifacts, "amd", legs["amd"], attempt=2)
    argv = ["--config", str(CONFIG_PATH), "--artifacts", str(artifacts), "--revision", REVISION]
    argv += ["--output", str(output)]
    assert re_.main(argv + ["--nvidia-result", "Succeeded", "--amd-result", "Succeeded"]) == 0
    assert json.loads(output.read_text())["eligible"] is True
    assert "ELIGIBLE" in capsys.readouterr().out


def test_cli_removes_a_stale_verdict_when_ineligible(tmp_path, capsys):
    artifacts, output = tmp_path / "artifacts", tmp_path / "eligibility.json"
    _write(artifacts, "nvidia", _legs()["nvidia"])
    output.write_text('{"eligible": true}')
    argv = ["--config", str(CONFIG_PATH), "--artifacts", str(artifacts), "--revision", REVISION]
    argv += ["--output", str(output)]
    assert re_.main(argv + ["--nvidia-result", "Succeeded", "--amd-result", "Failed"]) == 1
    assert not output.exists()
    assert "INELIGIBLE: amd leg result is 'Failed'" in capsys.readouterr().err


def test_cli_reports_a_report_of_the_wrong_shape(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    _write(artifacts, "nvidia", _legs()["nvidia"])
    _write(artifacts, "amd", _legs()["amd"])
    (artifacts / "fts-release-amd-1" / "fts-phases" / "reports" / "examples.json").write_text("[]")
    argv = ["--config", str(CONFIG_PATH), "--artifacts", str(artifacts), "--revision", REVISION]
    argv += ["--output", str(tmp_path / "e.json")]
    assert re_.main(argv + ["--nvidia-result", "Succeeded", "--amd-result", "Succeeded"]) == 1
    assert "INELIGIBLE" in capsys.readouterr().err


def test_cli_reports_unreadable_evidence(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    _write(artifacts, "nvidia", _legs()["nvidia"])
    _write(artifacts, "amd", _legs()["amd"])
    (artifacts / "fts-release-amd-1" / "fts-phases" / "reports" / "standalone.json").write_text("{")
    argv = ["--config", str(CONFIG_PATH), "--artifacts", str(artifacts), "--revision", REVISION]
    argv += ["--output", str(tmp_path / "e.json")]
    assert re_.main(argv + ["--nvidia-result", "Succeeded", "--amd-result", "Succeeded"]) == 1
    assert "INELIGIBLE" in capsys.readouterr().err
