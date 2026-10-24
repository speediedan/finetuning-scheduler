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
"""CPU tests for the release eligibility checker (``tests/gpu_ci/release_eligibility.py``)."""
import json

import pytest

from tests.gpu_ci import phase_manifest as pm
from tests.gpu_ci import release_eligibility as re_

REVISION = "0" * 40
IMAGES = {"nvidia": "fts:cuda@sha256:" + "a" * 64, "amd": "fts:rocm@sha256:" + "b" * 64}
TORCH = {"nvidia": "2.15.0+cu132", "amd": "2.15.0+rocm10.0"}
NODES = [f"tests/test_x.py::test_{i}" for i in range(4)]


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


def _leg(vendor, outcomes=None):
    versions = {"python": "3.13.16", "torch": TORCH[vendor], "lightning": "2.6.1", "finetuning_scheduler": "2.15.0"}
    leg = {}
    for phase in re_.REQUIRED_PHASES:
        manifest = _manifest(phase)
        report = pm.build_report(
            manifest,
            [_session(outcomes or dict.fromkeys(NODES, "passed"))],
            vendor=vendor,
            evidence_class="pipeline",
            lease_mode="tool",
            image=IMAGES[vendor],
            versions=versions,
        )
        leg[phase] = {"report": report, "manifest": manifest}
    return leg


def _legs():
    return {vendor: _leg(vendor) for vendor in pm.VENDORS}


def _check(legs, results=None):
    return re_.check_release(legs, results or {"nvidia": "Succeeded", "amd": "Succeeded"}, REVISION)


def _write(root, vendor, leg, attempt=1):
    phases = root / f"fts-release-{vendor}-{attempt}" / "fts-phases"
    for kind in ("reports", "manifests"):
        (phases / kind).mkdir(parents=True, exist_ok=True)
    for phase, docs in leg.items():
        (phases / "reports" / f"{phase}.json").write_text(json.dumps(docs["report"]))
        (phases / "manifests" / f"{phase}.json").write_text(json.dumps(docs["manifest"]))


def test_required_phases_are_the_release_phases():
    assert re_.REQUIRED_PHASES == ("gpu-nonstandalone", "standalone", "examples", "examples-multigpu")


def test_two_clean_full_legs_are_eligible():
    verdict = _check(_legs())
    assert verdict["eligible"] and verdict["revision"] == REVISION
    assert {v: e["image"] for v, e in verdict["evidence"].items()} == IMAGES
    assert set(verdict["evidence"]["amd"]["phases"]) == set(re_.REQUIRED_PHASES)


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
        (_versions("lightning_commit", "abc123"), "installed versions differ across vendors"),
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
    argv = ["--artifacts", str(artifacts), "--revision", REVISION, "--output", str(output)]
    assert re_.main(argv + ["--nvidia-result", "Succeeded", "--amd-result", "Succeeded"]) == 0
    assert json.loads(output.read_text())["eligible"] is True
    assert "ELIGIBLE" in capsys.readouterr().out


def test_cli_removes_a_stale_verdict_when_ineligible(tmp_path, capsys):
    artifacts, output = tmp_path / "artifacts", tmp_path / "eligibility.json"
    _write(artifacts, "nvidia", _legs()["nvidia"])
    output.write_text('{"eligible": true}')
    argv = ["--artifacts", str(artifacts), "--revision", REVISION, "--output", str(output)]
    assert re_.main(argv + ["--nvidia-result", "Succeeded", "--amd-result", "Failed"]) == 1
    assert not output.exists()
    assert "INELIGIBLE: amd leg result is 'Failed'" in capsys.readouterr().err


def test_cli_reports_unreadable_evidence(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    _write(artifacts, "nvidia", _legs()["nvidia"])
    _write(artifacts, "amd", _legs()["amd"])
    (artifacts / "fts-release-amd-1" / "fts-phases" / "reports" / "standalone.json").write_text("{")
    argv = ["--artifacts", str(artifacts), "--revision", REVISION, "--output", str(tmp_path / "e.json")]
    assert re_.main(argv + ["--nvidia-result", "Succeeded", "--amd-result", "Succeeded"]) == 1
    assert "INELIGIBLE" in capsys.readouterr().err
