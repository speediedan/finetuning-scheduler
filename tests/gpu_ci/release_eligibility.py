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
"""Release eligibility: one full, clean, like-for-like GPU run on each vendor, at the release candidate's revision.

A release needs a demanded green on each vendor (#57). The release pipeline (``.azure-pipelines/gpu-release.yml``) runs
one full leg per vendor through the any-vendor job template, each publishing its phase evidence as
``fts-release-<vendor>-<attempt>``, then runs this checker on a CPU job over both. Its verdict is the only result a
publication step may consume: the legs' own results are necessary, not sufficient.

The checker re-derives what it can rather than trusting a report's own summary:

- each required phase has a manifest and a report, the report's ``manifest_sha256`` is the manifest's digest (the rule
  ``phase_manifest.manifest_digest`` defines), and its counts agree with the manifest's node lists;
- the phase is ``clean`` by ``phase_manifest.classify`` applied to the counts, and the report says so too;
- the run is the candidate's: repository and revision match, the scope is full, the tree it collected from was clean,
  the evidence class is ``pipeline`` (a parity run made outside the pipeline is never release evidence), the lease was
  taken through the host's lease tool, and the image is digest-pinned;
- within a leg, every phase names the same image and installed versions; across the legs, each phase collected and
  selected the same node ids with the same selector, and the installed Fine-Tuning Scheduler, Lightning and torch
  release (without its local build tag) agree, as do the commits torch and Lightning were built from.

Each phase's summary counts are recounted from its per-node records and its exclusions from the nodes it skipped or
expected to fail, and a measured phase's coverage export is checked against the digest its report recorded.

Usage, from the repository root, on the downloaded artifacts (standard library only)::

    python -m tests.gpu_ci.release_eligibility --artifacts "$(Pipeline.Workspace)" \\
        --revision "$(Build.SourceVersion)" --nvidia-result Succeeded --amd-result Succeeded \\
        --output eligibility.json
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from tests.gpu_ci.phase_manifest import OUTCOMES, PHASES, REPOSITORY, VENDORS, classify, manifest_digest

# The phases a release leg runs (pool-reverse is the nightly's alone).
REQUIRED_PHASES = tuple(name for name, phase in PHASES.items() if phase.release)
ARTIFACT = re.compile(r"fts-release-(?P<vendor>[a-z]+)-(?P<attempt>\d+)")
DIGEST_PINNED = re.compile(r".+@sha256:[0-9a-f]{64}")
# Installed versions that must agree across vendors. ``torch`` is compared without its local tag (``+cu132``,
# ``+rocm10.0``), the one part of it that names the vendor.
SHARED_VERSIONS = ("finetuning_scheduler", "lightning", "torch")
# What names the build behind a version string that does not: a test-channel torch release candidate carries the final
# version, and a commit-pinned Lightning one ``.dev`` version for every commit. Both are vendor-neutral commits, so they
# must agree too (``torch_record_sha256`` and ``torch_cuda``/``torch_hip`` differ per wheel and stay out).
SHARED_BUILDS = ("torch_git_version", "lightning_commit")


class Ineligible(ValueError):
    """The evidence does not make the candidate releasable."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Ineligible(message)


def leg_dirs(artifacts: Path) -> dict[str, Path]:
    """Each vendor's evidence: its latest attempt's ``fts-phases`` directory (a re-run leg publishes a new
    attempt)."""
    latest: dict[str, tuple[int, Path]] = {}
    for path in sorted(artifacts.iterdir()) if artifacts.is_dir() else ():
        match = ARTIFACT.fullmatch(path.name)
        if match and path.is_dir():
            vendor, attempt = match["vendor"], int(match["attempt"])
            if vendor not in latest or attempt > latest[vendor][0]:
                latest[vendor] = (attempt, path / "fts-phases")
    return {vendor: root for vendor, (_, root) in latest.items()}


def load_leg(root: Path) -> dict[str, dict]:
    """``{phase: {"report": ..., "manifest": ..., "coverage_sha256": ...}}`` for every required phase under
    ``root``.

    ``coverage_sha256`` is the digest of the coverage export the report names, when that file is present.
    """
    phases = {}
    for phase in REQUIRED_PHASES:
        report, manifest = root / "reports" / f"{phase}.json", root / "manifests" / f"{phase}.json"
        if report.is_file() and manifest.is_file():
            doc = json.loads(report.read_text())
            export = root / str((doc.get("coverage") or {}).get("json") or "")
            phases[phase] = {
                "report": doc,
                "manifest": json.loads(manifest.read_text()),
                "coverage_sha256": hashlib.sha256(export.read_bytes()).hexdigest() if export.is_file() else None,
            }
    return phases


def check_phase(
    vendor: str, phase: str, report: dict, manifest: dict, revision: str, coverage_sha256: str | None = None
) -> None:
    """One phase of one leg: identity, digest, counts against the manifest and the node records, exclusions,
    coverage evidence, and a clean classification."""
    where = f"{vendor}/{phase}"
    for key, expected in (
        ("repository", REPOSITORY),
        ("revision", revision),
        ("phase", phase),
        ("scope", "full"),
        ("evidence_class", "pipeline"),
        ("vendor", vendor),
        ("backend", VENDORS[vendor]),
        ("lease_mode", "tool"),
        ("count_unit", "unique-node-id"),
    ):
        require(report.get(key) == expected, f"{where}: {key} is {report.get(key)!r}, not {expected!r}")
    require(DIGEST_PINNED.fullmatch(report.get("image") or "") is not None, f"{where}: image is not digest-pinned")
    for key in ("repository", "revision", "phase", "scope", "selector"):
        require(report[key] == manifest[key], f"{where}: report and manifest disagree on {key}")
    require(report["manifest_sha256"] == manifest_digest(manifest), f"{where}: manifest digest mismatch")
    require(not manifest.get("tree_dirty_paths"), f"{where}: collected from a dirty tree")
    require(not any(manifest.get("collect_exit_codes", {}).values()), f"{where}: a collection exited non-zero")
    raw, selected = manifest["raw_node_ids"], manifest["selected_node_ids"]
    require(len(raw) == len(set(raw)) and len(selected) == len(set(selected)), f"{where}: duplicate manifest node")
    require(set(selected) <= set(raw), f"{where}: a selected node is absent from the raw collection")
    collection, outcomes = report["collection"], report["outcomes"]
    require(
        all(type(v) is int and v >= 0 for v in (*collection.values(), *outcomes.values())),
        f"{where}: counts must be nonnegative integers",
    )
    require(
        collection["raw"] == len(raw) and collection["selected"] == len(selected),
        f"{where}: counts disagree with the manifest",
    )
    require(
        collection["raw"] == collection["selected"] + collection["deselected"],
        f"{where}: collection counts do not add up",
    )
    require(collection["selected"] == outcomes["attempted"] + outcomes["unexecuted"], f"{where}: unaccounted nodes")
    require(sum(outcomes[k] for k in OUTCOMES) == outcomes["attempted"], f"{where}: outcomes do not add up")
    # The summary counts decide cleanliness, so they are recounted from the per-node records, which must cover exactly
    # the manifest's selection.
    nodes = report.get("nodes")
    require(isinstance(nodes, dict) and set(nodes) == set(selected), f"{where}: node records are not the selection")
    recount = dict.fromkeys((*OUTCOMES, "unexecuted"), 0)
    for nodeid, node in nodes.items():
        require(node.get("outcome") in recount, f"{where}: {nodeid} has outcome {node.get('outcome')!r}")
        recount[node["outcome"]] += 1
    require(
        all(outcomes[k] == n for k, n in recount.items()), f"{where}: outcome counts disagree with the node records"
    )
    require(not report.get("unmanifested_nodes"), f"{where}: nodes ran that the manifest did not select")
    # A skip or expected failure is clean only with a reviewed reason: the exclusions must list exactly those nodes,
    # and "reviewed" must mean every one of them has its reason.
    excluded = {n for n, node in nodes.items() if node["outcome"] in ("skipped", "xfailed")}
    reasons = {e.get("nodeid"): e.get("reason") for e in report.get("exclusions", [])}
    require(set(reasons) == excluded, f"{where}: exclusions do not list exactly the skipped and xfailed nodes")
    require(
        report["exclusions_reviewed"] == all(reasons.values()), f"{where}: exclusions_reviewed disagrees with reasons"
    )
    # Coverage is part of a measured phase's evidence: the export must be present and be the one the report digested.
    # It is not compared across vendors here (``phase_manifest coverage-diff`` does that): a backend may run lines of
    # its own.
    if PHASES[phase].measured:
        coverage = report.get("coverage") or {}
        require(coverage.get("measured") and not coverage.get("missing"), f"{where}: coverage evidence is missing")
        require(
            coverage_sha256 is not None and coverage_sha256 == coverage.get("json_sha256"),
            f"{where}: coverage export missing or not the one the report digested",
        )
        require((coverage.get("covered_lines") or 0) > 0, f"{where}: coverage records no executed lines")
    classification = classify(report)
    require(classification == "clean", f"{where}: phase is {classification}")
    require(report.get("classification") == classification, f"{where}: report claims {report.get('classification')!r}")


def check_release(legs: dict[str, dict[str, dict]], results: dict[str, str], revision: str) -> dict:
    """The verdict over both legs, or :class:`Ineligible` naming the first reason the candidate is not
    releasable."""
    require(bool(revision), "candidate revision is missing")
    for vendor in VENDORS:
        require(results.get(vendor) == "Succeeded", f"{vendor} leg result is {results.get(vendor)!r}, not Succeeded")
        require(vendor in legs, f"{vendor}: no release evidence (fts-release-{vendor}-<attempt>)")
    selections: dict[str, tuple] = {}
    installed_by_vendor: dict[str, dict] = {}
    evidence = {}
    for vendor in VENDORS:
        phases = legs[vendor]
        missing = sorted(set(REQUIRED_PHASES) - set(phases))
        require(not missing, f"{vendor}: no evidence for required phase(s) {', '.join(missing)}")
        first = phases[REQUIRED_PHASES[0]]["report"]
        for phase in REQUIRED_PHASES:
            report, manifest = phases[phase]["report"], phases[phase]["manifest"]
            check_phase(vendor, phase, report, manifest, revision, phases[phase].get("coverage_sha256"))
            for key in ("image", "versions"):
                require(report[key] == first[key], f"{vendor}/{phase}: {key} differs from the leg's other phases")
            selection = (manifest["selector"], sorted(manifest["raw_node_ids"]), sorted(manifest["selected_node_ids"]))
            require(
                selections.setdefault(phase, selection) == selection,
                f"{phase}: the vendors collected or selected different nodes; classify the difference before release",
            )
        versions = first["versions"]
        installed = {k: (versions.get(k) or "").split("+")[0] for k in SHARED_VERSIONS}
        require(all(installed.values()), f"{vendor}: installed versions missing ({', '.join(SHARED_VERSIONS)})")
        installed_by_vendor[vendor] = {**installed, **{k: versions.get(k) for k in SHARED_BUILDS}}
        evidence[vendor] = {
            "image": first["image"],
            "versions": {k: versions.get(k) for k in ("python", *SHARED_VERSIONS, *SHARED_BUILDS)},
            "phases": {p: phases[p]["report"]["manifest_sha256"] for p in REQUIRED_PHASES},
        }
    require(
        len({json.dumps(v, sort_keys=True) for v in installed_by_vendor.values()}) == 1,
        f"installed versions differ across vendors: {installed_by_vendor}",
    )
    return {
        "eligible": True,
        "repository": REPOSITORY,
        "revision": revision,
        "results": dict(results),
        "required_phases": list(REQUIRED_PHASES),
        "evidence": evidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, required=True, help="the directory the legs' artifacts downloaded to")
    parser.add_argument("--revision", required=True, help="the release candidate's commit")
    parser.add_argument("--nvidia-result", required=True, help="the NVIDIA leg's job result (Succeeded, Failed, ...)")
    parser.add_argument("--amd-result", required=True, help="the AMD leg's job result")
    parser.add_argument("--output", type=Path, required=True, help="where to write the verdict when eligible")
    args = parser.parse_args(argv)
    # A verdict left by an earlier invocation must never survive a failed check as current evidence.
    args.output.unlink(missing_ok=True)
    try:
        legs = {vendor: load_leg(root) for vendor, root in leg_dirs(args.artifacts).items()}
        verdict = check_release(legs, {"nvidia": args.nvidia_result, "amd": args.amd_result}, args.revision)
    except (Ineligible, KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError) as exc:
        print(f"INELIGIBLE: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"ELIGIBLE: {', '.join(VENDORS)}, {len(REQUIRED_PHASES)} full phases each, revision {args.revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
