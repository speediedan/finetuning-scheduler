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
"""Phase manifests and phase reports for the GPU CI phases.

A GPU CI run is reported phase by phase. For each phase this module writes two documents:

- a **manifest** (``collect``): the node ids the phase's paths collect (``raw_node_ids``) and the ones its selector
  keeps (``selected_node_ids``), with the repository, revision, phase, scope and selector string. Collection only: it
  runs on the CPU with ``CUDA_VISIBLE_DEVICES`` emptied and executes nothing, so it is evidence about selection, never
  a pass.
- a **report** (``report``): per-node, per-rank outcomes read from the records ``phase_plugin.py`` wrote while the phase
  ran, counted against the manifest, plus the run's identity (vendor, backend, evidence class, image, lease mode,
  installed versions) and the manifest's SHA-256.

The phases and their selectors are the steps of ``.azure-pipelines/gpu-tests.yml``. A phase's ``selector`` is the
selection-relevant part of its step (the pytest paths and arguments and the selection variables), not wrappers that do
not change selection (``coverage run``) or that only stop early (``--maxfail``): a phase is reported node by node, and
a node left unexecuted fails it anyway.

Coverage is part of a phase's evidence: a second host or vendor substitutes for the first only if it executes the same
lines. The phases the pipeline runs under ``coverage run`` (``measured``) get one coverage data file each
(``COVERAGE_FILE``, set by ``run-env``); ``report`` exports it with ``coverage json`` from the repository root, so file
names are relative and two hosts' checkouts compare, and records the export's digest and line counts. ``coverage-diff``
lists, per phase and file, the lines one run executed and the other did not. A phase the pipeline does not measure is
reported as unmeasured, never as zero coverage.

The manifest digest is SHA-256 over the manifest serialized as UTF-8 JSON with sorted keys and compact separators and
no trailing newline, the rule the release eligibility checker applies. A report never claims more than its evidence
class: ``parity-pre-agent`` marks a run made outside the pipeline (no image digest), which the release checker refuses.

Usage, from the repository root (``python -m tests.gpu_ci.phase_manifest --help`` for the details)::

    python -m tests.gpu_ci.phase_manifest list
    python -m tests.gpu_ci.phase_manifest collect --phase standalone --out-dir /tmp/fts-phases
    python -m tests.gpu_ci.phase_manifest run-env --phase standalone --out-dir /tmp/fts-phases
    python -m tests.gpu_ci.phase_manifest report --phase standalone --out-dir /tmp/fts-phases \\
        --vendor amd --evidence-class parity-pre-agent --lease-mode tool
    python -m tests.gpu_ci.phase_manifest coverage-diff /tmp/fts-phases-nvidia /tmp/fts-phases-amd

Keep ``--out-dir`` outside the repository: the manifest records the tree's untracked files, so an out-dir inside the
checkout marks every later collection dirty.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "speediedan/finetuning-scheduler"
PLUGIN = "tests.gpu_ci.phase_plugin"
RECORD_DIR_ENV = "FTS_PHASE_RECORD_DIR"
# The fixture whose modules form the reverse-order leg (``grep -l`` over ``tests/test_*.py`` in the pipeline). Named
# here, not spelled out in a test module, so that module does not join the leg itself.
POOL_FIXTURE = "dist_pool_fixture"
COVERAGE_SOURCE = "src/finetuning_scheduler"
# Visible-device variables a report records: a run's device set is part of what makes two hosts comparable.
DEVICE_VARS = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
# Variables tests/conftest.py selects or orders on; stripped from the environment before every collection.
SELECTION_VARS = ("PL_RUN_STANDALONE_TESTS", "FTS_EXPERIMENTAL_PATCH_TESTS", "FTS_CUDA_TESTS_ONLY", "FTS_REVERSE_ORDER")
# Set for every GPU phase (the pipeline's job variable): it gates skips, never selection.
RUN_ENV = {"PL_RUN_CUDA_TESTS": "1"}
VENDORS = {"nvidia": "cuda", "amd": "hip"}
EVIDENCE_CLASSES = ("pipeline", "parity-pre-agent")
OUTCOMES = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
SMOKE_FILE = "tests/.standalone_smoke"


@dataclass(frozen=True)
class Phase:
    """One GPU CI phase: what it collects, how it selects, and how the pipeline runs it."""

    name: str
    paths: tuple[str, ...]
    selection_env: dict = field(default_factory=dict)
    args: tuple[str, ...] = ()
    command: str = ""
    release: bool = True
    # whether the pipeline runs the phase under ``coverage run`` (``special_tests.sh`` always does)
    measured: bool = True

    def resolved_paths(self, root: Path = ROOT) -> tuple[str, ...]:
        if self.name == "pool-reverse":
            return pool_modules(root)
        return self.paths


PHASES = {
    p.name: p
    for p in (
        Phase(
            "gpu-nonstandalone",
            ("src/finetuning_scheduler", "tests"),
            {"FTS_CUDA_TESTS_ONLY": "1"},
            # as the pipeline runs it: a measured phase's command must write the coverage data ``run-env`` points at
            command=(
                "python -m coverage run --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v"
            ),
        ),
        Phase(
            "standalone",
            ("tests",),
            {"PL_RUN_STANDALONE_TESTS": "1"},
            command="bash ./tests/special_tests.sh --mark_type=standalone",
        ),
        Phase("examples", ("src/fts_examples",), command="python -m pytest src/fts_examples -v", measured=False),
        Phase(
            "examples-multigpu",
            ("src/fts_examples",),
            {"PL_RUN_STANDALONE_TESTS": "1"},
            ("-k", "model_parallel_examples"),
            command=(
                "bash ./tests/special_tests.sh --mark_type=standalone --collect_dir=src/fts_examples"
                " --filter_pattern=model_parallel_examples"
            ),
        ),
        # The scheduled reverse-order leg: in cross-host parity, never in release eligibility.
        Phase(
            "pool-reverse",
            (),
            {"FTS_REVERSE_ORDER": "1"},
            command="python -m pytest <pool modules> -v",
            release=False,
            measured=False,
        ),
    )
}


def pool_modules(root: Path = ROOT) -> tuple[str, ...]:
    """The modules the reverse-order leg runs (the pipeline's ``grep -l`` for :data:`POOL_FIXTURE` over
    ``tests/test_*.py``), resolved and sorted."""
    return tuple(
        p.relative_to(root).as_posix()
        for p in sorted((root / "tests").glob("test_*.py"))
        if POOL_FIXTURE in p.read_text(encoding="utf-8")
    )


def selector_string(phase: Phase, scope: str, root: Path = ROOT) -> str:
    """The phase's selection, as one comparable string: selection variables, then the pytest paths and
    arguments."""
    env = " ".join(f"{k}={v}" for k, v in sorted({**RUN_ENV, **phase.selection_env}.items()))
    parts = [env, "; pytest", *phase.resolved_paths(root), *phase.args]
    if scope == "smoke":
        parts.append(f"; select_file={SMOKE_FILE}")
    return " ".join(parts).replace(" ; ", "; ")


def manifest_digest(manifest: dict) -> str:
    """SHA-256 of the manifest as sorted, compact UTF-8 JSON without a trailing newline (the release checker's
    rule)."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_records(record_dir: Path) -> list[list[dict]]:
    """Each session's records (one list per JSON-lines file ``phase_plugin.py`` wrote), in file-name order."""
    sessions = []
    for path in sorted(Path(record_dir).glob("records-*.jsonl")):
        sessions.append([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    return sessions


def _collect(
    paths: tuple[str, ...], args: tuple[str, ...], env_update: dict, root: Path
) -> tuple[list[str], list, int]:
    """Collect once through the plugin: the selected node ids, the failed collectors and pytest's exit code."""
    env = {k: v for k, v in os.environ.items() if k not in SELECTION_VARS}
    env.update(RUN_ENV)
    env.update(env_update)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *filter(None, [env.get("PYTHONPATH")])])
    record_dir = Path(tempfile.mkdtemp(prefix="fts-phase-collect-"))
    env[RECORD_DIR_ENV] = str(record_dir)
    cmd = [sys.executable, "-m", "pytest", *paths, *args, "--collect-only", "-q"]
    cmd += ["-p", "no:cacheprovider", "-p", PLUGIN]
    try:
        proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        records = [r for session in read_records(record_dir) for r in session]
    finally:
        shutil.rmtree(record_dir, ignore_errors=True)
    collected = [r["nodeids"] for r in records if r["event"] == "collected"]
    errors = sorted(r["nodeid"] for r in records if r["event"] == "collect_error")
    if len(collected) != 1:
        errors.append(f"collection recorded {len(collected)} times (pytest exit {proc.returncode})")
    return (collected[0] if collected else []), errors, proc.returncode


def git_state(root: Path = ROOT) -> tuple[str, list[str] | None]:
    """``HEAD`` and the paths that differ from it (untracked included); ``None`` when git cannot say."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True, text=True
    )
    dirty = None if status.returncode else sorted(line[3:] for line in status.stdout.splitlines() if line.strip())
    return (head.stdout.strip() if head.returncode == 0 else "unknown"), dirty


def build_manifest(phase_name: str, scope: str = "full", root: Path = ROOT) -> dict:
    """Collect ``phase_name`` twice on the CPU (all of its paths, then its selection) and describe both."""
    phase = PHASES[phase_name]
    if scope not in ("full", "smoke") or (scope == "smoke" and phase_name != "standalone"):
        raise ValueError(f"scope {scope!r} is not defined for phase {phase_name!r}")
    paths = phase.resolved_paths(root)
    if not paths:
        raise ValueError(f"phase {phase_name!r} resolved to no paths")
    raw, raw_errors, raw_exit = _collect(paths, (), {}, root)
    selected, sel_errors, sel_exit = _collect(paths, phase.args, phase.selection_env, root)
    if scope == "smoke":
        wanted = [line.strip() for line in (root / SMOKE_FILE).read_text(encoding="utf-8").splitlines()]
        wanted = [w for w in wanted if w and not w.startswith("#")]
        sel_errors += [f"smoke node not collected: {w}" for w in wanted if w not in selected]
        selected = [w for w in wanted if w in selected]
    revision, dirty = git_state(root)
    return {
        "repository": REPOSITORY,
        "revision": revision,
        "phase": phase_name,
        "scope": scope,
        "selector": selector_string(phase, scope, root),
        # A tree that differs from the revision makes the revision a wrong name for what was collected.
        "tree_dirty_paths": dirty,
        "raw_node_ids": raw,
        "selected_node_ids": selected,
        "collection_errors": sorted(set(raw_errors + sel_errors)),
        "collect_exit_codes": {"raw": raw_exit, "selected": sel_exit},
    }


def _session_node_outcomes(records: list[dict]) -> dict[str, str]:
    """A session's verdict per node: a failure or error in any stage wins, else the deciding stage's outcome."""
    per_node: dict[str, list[str]] = {}
    for r in records:
        if r["event"] == "outcome":
            per_node.setdefault(r["nodeid"], []).append(r["outcome"])
    verdicts = {}
    for nodeid, outcomes in per_node.items():
        verdicts[nodeid] = next((o for o in ("failed", "errors") if o in outcomes), outcomes[0])
    return verdicts


def build_report(
    manifest: dict,
    sessions: list[list[dict]],
    *,
    vendor: str,
    evidence_class: str,
    lease_mode: str | None = None,
    image: str | None = None,
    versions: dict | None = None,
    reviewed_exclusions: dict | None = None,
    coverage: dict | None = None,
) -> dict:
    """Count one phase's per-node, per-rank outcomes against its manifest and describe the run that produced them.

    A node's outcome is rank 0's (the launching process) unless any rank failed or errored on it, which fails the node;
    a selected node with no rank-0 verdict is ``unexecuted``. A session that started but never finished is a missing
    rank completion, and fails every node it collected that it gave no verdict for. A node that ran but is not in the
    manifest's selection is a collection error: the run and the manifest disagree about what the phase is.
    """
    if vendor not in VENDORS:
        raise ValueError(f"vendor must be one of {sorted(VENDORS)}")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"evidence class must be one of {EVIDENCE_CLASSES}")
    reviewed = reviewed_exclusions or {}
    phase = PHASES.get(manifest["phase"])
    selected = manifest["selected_node_ids"]
    selected_set = set(selected)
    ranks: dict[str, dict[str, str]] = {}
    world_sizes: dict[str, set] = {}
    missing_completions = 0
    for records in sessions:
        if not records:
            continue
        rank = str(records[0]["rank"])
        verdicts = _session_node_outcomes(records)
        world_size = next((r["world_size"] for r in records if r.get("world_size")), None)
        for nodeid, verdict in verdicts.items():
            ranks.setdefault(nodeid, {})[rank] = verdict
            if world_size:
                world_sizes.setdefault(nodeid, set()).add(world_size)
        finished = any(r["event"] == "finish" for r in records)
        collected = next((r["nodeids"] for r in records if r["event"] == "collected"), [])
        if not finished and any(r["event"] == "start" for r in records):
            missing_completions += 1
            for nodeid in collected:
                if nodeid not in verdicts:
                    ranks.setdefault(nodeid, {})[rank] = "incomplete"
    nodes, counts = {}, dict.fromkeys(OUTCOMES, 0)
    unexecuted, rank_failures = [], 0
    for nodeid in selected:
        per_rank = ranks.get(nodeid, {})
        rank_failures += sum(1 for r, o in per_rank.items() if r != "0" and o in ("failed", "errors", "incomplete"))
        if any(o == "incomplete" for o in per_rank.values()) or any(
            o in ("failed", "errors") for r, o in per_rank.items() if r != "0"
        ):
            outcome = "failed"
        elif "0" in per_rank:
            outcome = per_rank["0"]
        else:
            unexecuted.append(nodeid)
            nodes[nodeid] = {"outcome": "unexecuted", "ranks": per_rank}
            continue
        counts[outcome] += 1
        # the launcher's world size, when one was set: two hosts can agree on a verdict yet run a node on different
        # numbers of ranks
        world_size = max(world_sizes.get(nodeid, ()), default=None)
        nodes[nodeid] = {"outcome": outcome, "ranks": per_rank, "world_size": world_size}
    unmanifested = sorted(set(ranks) - selected_set)
    exclusions = [
        {"nodeid": n, "outcome": v["outcome"], "reason": reviewed.get(n)}
        for n, v in nodes.items()
        if v["outcome"] in ("skipped", "xfailed")
    ]
    errors = len(manifest.get("collection_errors", [])) + len(unmanifested)
    report = {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "phase": manifest["phase"],
        "scope": manifest["scope"],
        "selector": manifest["selector"],
        "evidence_class": evidence_class,
        "vendor": vendor,
        "backend": VENDORS[vendor],
        "lease_mode": lease_mode,
        "image": image,
        "versions": versions or {},
        "count_unit": "unique-node-id",
        "collection": {
            "raw": len(manifest["raw_node_ids"]),
            "deselected": len(manifest["raw_node_ids"]) - len(selected),
            "selected": len(selected),
            "errors": errors,
        },
        "outcomes": {"attempted": len(selected) - len(unexecuted), **counts, "unexecuted": len(unexecuted)},
        "rank_failures": rank_failures,
        "missing_rank_completions": missing_completions,
        "exclusions_reviewed": all(e["reason"] for e in exclusions),
        "exclusions": exclusions,
        "unmanifested_nodes": unmanifested,
        "nodes": nodes,
        "manifest_sha256": manifest_digest(manifest),
        "coverage": coverage or {"measured": phase.measured if phase else None},
    }
    report["classification"] = classify(report)
    return report


def classify(report: dict) -> str:
    """One word for the phase: ``clean``, ``unreviewed-exclusions``, ``empty`` or ``failing``."""
    o, c = report["outcomes"], report["collection"]
    if c["errors"] or o["failed"] or o["errors"] or o["xpassed"] or o["unexecuted"]:
        return "failing"
    if report["rank_failures"] or report["missing_rank_completions"]:
        return "failing"
    if not c["selected"] or not o["passed"]:
        return "empty"
    return "clean" if report["exclusions_reviewed"] else "unreviewed-exclusions"


def installed_versions() -> dict:
    """The versions a phase report records: interpreter, torch (with its commit and backend), Lightning, FTS."""
    import platform

    versions = {"python": platform.python_version()}
    try:
        import torch

        versions.update(
            torch=torch.__version__,
            torch_git_version=torch.version.git_version,
            torch_cuda=torch.version.cuda,
            torch_hip=getattr(torch.version, "hip", None),
        )
    except ImportError as exc:  # a report from a broken environment should say so, not crash
        versions["torch"] = f"unavailable: {exc}"
    for name, module in (("lightning", "lightning"), ("finetuning_scheduler", "finetuning_scheduler")):
        try:
            versions[name] = __import__(module).__version__
        except (ImportError, AttributeError) as exc:
            versions[name] = f"unavailable: {exc}"
    # A commit pin installs Lightning with a ``.dev0`` version string, and PyTorch's test-channel release candidates
    # carry the final version string, so neither version names the build: record what does.
    versions["lightning_commit"] = _vcs_commit("lightning")
    versions["torch_record_sha256"] = _record_sha256("torch")
    return versions


def _distribution(name: str):
    from importlib import metadata

    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def _vcs_commit(name: str) -> str | None:
    """The commit a distribution was installed from (its ``direct_url.json``), or ``None`` for an index install."""
    dist = _distribution(name)
    direct_url = dist.read_text("direct_url.json") if dist else None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id") if direct_url else None


def _record_sha256(name: str) -> str | None:
    """SHA-256 of a distribution's installed ``RECORD``, which hashes every file it installed."""
    dist = _distribution(name)
    record = dist.read_text("RECORD") if dist else None
    return hashlib.sha256(record.encode()).hexdigest() if record else None


def device_identity() -> dict:
    """The devices the run saw: the visible-device variables, the device count and each device's name."""
    ident = {var: os.environ.get(var) for var in DEVICE_VARS}
    try:
        import torch

        count = torch.cuda.device_count()
        ident.update(device_count=count, devices=[torch.cuda.get_device_name(i) for i in range(count)])
    except (ImportError, RuntimeError, AssertionError) as exc:
        ident.update(device_count=None, devices=f"unavailable: {exc}")
    return ident


def coverage_file(out_dir: Path, phase_name: str) -> Path:
    """The phase's coverage data file (``special_tests.sh`` appends every test's run to it)."""
    return Path(out_dir).resolve() / "coverage" / phase_name / ".coverage"


def coverage_summary(phase_name: str, out_dir: Path, root: Path = ROOT) -> dict:
    """Export a measured phase's coverage data as JSON (file names relative to ``root``) and summarize it.

    A measured phase without a data file is reported as missing, which a comparison must treat as an evidence gap, not
    as zero lines.
    """
    if not PHASES[phase_name].measured:
        return {"measured": False}
    data = coverage_file(out_dir, phase_name)
    if not data.exists():
        return {"measured": True, "missing": True, "data_file": str(data)}
    export = Path(out_dir).resolve() / "coverage" / f"{phase_name}.json"
    cmd = [sys.executable, "-m", "coverage", "json", f"--data-file={data}", "-o", str(export), "--ignore-errors"]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if proc.returncode or not export.exists():
        return {"measured": True, "missing": True, "data_file": str(data), "error": proc.stderr.strip()[-500:]}
    doc = json.loads(export.read_text(encoding="utf-8"))
    return {
        "measured": True,
        "missing": False,
        "data_file_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "json": export.relative_to(Path(out_dir).resolve()).as_posix(),
        "json_sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
        "files": len(doc["files"]),
        "num_statements": doc["totals"]["num_statements"],
        "covered_lines": doc["totals"]["covered_lines"],
        "percent_covered": round(doc["totals"]["percent_covered"], 2),
    }


def executed_lines(out_dir: Path, phase_name: str) -> dict[str, set[int]] | None:
    """The lines a phase's coverage export records as executed, per relative file name; ``None`` if absent."""
    export = Path(out_dir) / "coverage" / f"{phase_name}.json"
    if not export.exists():
        return None
    doc = json.loads(export.read_text(encoding="utf-8"))
    return {name: set(f["executed_lines"]) for name, f in doc["files"].items()}


def coverage_diff(reference: Path, candidate: Path, phase_names=None) -> dict:
    """Per measured phase, and for their union, the lines the reference run executed that the candidate did not
    (``lost``) and the reverse (``gained``).

    Coverage parity is an empty ``lost`` everywhere.
    """
    names = [n for n in (phase_names or PHASES) if PHASES[n].measured]
    result, union = {}, {"reference": {}, "candidate": {}}
    for name in names:
        ref, cand = executed_lines(reference, name), executed_lines(candidate, name)
        if ref is None or cand is None:
            result[name] = {"missing": [side for side, v in (("reference", ref), ("candidate", cand)) if v is None]}
            continue
        for side, lines in (("reference", ref), ("candidate", cand)):
            for f, ls in lines.items():
                union[side].setdefault(f, set()).update(ls)
        result[name] = _line_delta(ref, cand)
    result["union"] = _line_delta(union["reference"], union["candidate"])
    return result


def _line_delta(ref: dict[str, set[int]], cand: dict[str, set[int]]) -> dict:
    files = sorted(set(ref) | set(cand))
    lost = {f: sorted(ref.get(f, set()) - cand.get(f, set())) for f in files}
    gained = {f: sorted(cand.get(f, set()) - ref.get(f, set())) for f in files}
    n_ref = sum(map(len, ref.values()))
    kept = n_ref - sum(map(len, lost.values()))
    return {
        "reference_lines": n_ref,
        "candidate_lines": sum(map(len, cand.values())),
        "lost": {f: v for f, v in lost.items() if v},
        "gained": {f: v for f, v in gained.items() if v},
        "kept_fraction": round(kept / n_ref, 4) if n_ref else None,
    }


def run_env(phase_name: str, out_dir: Path) -> dict:
    """The environment a phase's pipeline command needs to run selected, recorded and, if the pipeline measures it,
    under its own coverage data file (every rank inherits it)."""
    phase = PHASES[phase_name]
    addopts = " ".join(filter(None, [os.environ.get("PYTEST_ADDOPTS", ""), "-p", PLUGIN]))
    record_dir = Path(out_dir).resolve() / "records" / phase_name
    env = {**RUN_ENV, **phase.selection_env, RECORD_DIR_ENV: str(record_dir), "PYTEST_ADDOPTS": addopts.strip()}
    if phase.measured:
        data = coverage_file(out_dir, phase_name)
        data.parent.mkdir(parents=True, exist_ok=True)
        env["COVERAGE_FILE"] = str(data)
    return env


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the phases, their selectors and the pipeline commands that run them")
    collect = sub.add_parser("collect", help="write <out-dir>/manifests/<phase>.json (CPU only, executes nothing)")
    collect.add_argument("--phase", action="append", choices=sorted(PHASES), required=True)
    collect.add_argument("--scope", choices=("full", "smoke"), default="full")
    collect.add_argument("--out-dir", type=Path, required=True)
    env = sub.add_parser("run-env", help="print the exports that run a phase selected and recorded")
    env.add_argument("--phase", choices=sorted(PHASES), required=True)
    env.add_argument("--out-dir", type=Path, required=True)
    report = sub.add_parser("report", help="write <out-dir>/reports/<phase>.json from the manifest and the records")
    report.add_argument("--phase", choices=sorted(PHASES), required=True)
    report.add_argument("--out-dir", type=Path, required=True)
    report.add_argument("--vendor", choices=sorted(VENDORS), required=True)
    report.add_argument("--evidence-class", choices=EVIDENCE_CLASSES, required=True)
    report.add_argument("--lease-mode", default=None)
    report.add_argument("--image", default=None)
    report.add_argument("--reviewed-exclusions", type=Path, default=None, help="JSON object: node id -> reason")
    diff = sub.add_parser("coverage-diff", help="lines one run's measured phases executed and the other's did not")
    diff.add_argument("reference", type=Path, help="out-dir of the reference run (for example the NVIDIA host)")
    diff.add_argument("candidate", type=Path, help="out-dir of the run compared with it")
    diff.add_argument("--phase", action="append", choices=sorted(n for n, p in PHASES.items() if p.measured))
    diff.add_argument("--out", type=Path, default=None, help="write the full result as JSON")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name, phase in PHASES.items():
            print(
                f"{name}\trelease={phase.release}\tmeasured={phase.measured}"
                f"\tselector={selector_string(phase, 'full')}\t{phase.command}"
            )
        return 0
    if args.cmd == "collect":
        status = 0
        for name in args.phase:
            manifest = build_manifest(name, args.scope)
            _write_json(args.out_dir / "manifests" / f"{name}.json", manifest)
            # collected beside phase, always: a count without its phase reads as a suite result
            print(
                f"phase={name} scope={args.scope} raw={len(manifest['raw_node_ids'])}"
                f" selected={len(manifest['selected_node_ids'])} errors={len(manifest['collection_errors'])}"
                f" (collection only) sha256={manifest_digest(manifest)}"
            )
            status |= bool(manifest["collection_errors"])
        return status
    if args.cmd == "run-env":
        for key, value in run_env(args.phase, args.out_dir).items():
            print(f"export {key}={json.dumps(value)}")
        return 0
    if args.cmd == "coverage-diff":
        result = coverage_diff(args.reference, args.candidate, args.phase)
        if args.out:
            _write_json(args.out, result)
        lost_any = False
        for name, r in result.items():
            if "missing" in r:
                print(f"{name}: coverage export missing on {', '.join(r['missing'])}")
                lost_any = True
                continue
            lost = sum(map(len, r["lost"].values()))
            lost_any |= bool(lost)
            print(
                f"{name}: reference={r['reference_lines']} candidate={r['candidate_lines']} lost={lost}"
                f" gained={sum(map(len, r['gained'].values()))} kept={r['kept_fraction']}"
            )
            for f, lines in r["lost"].items():
                print(f"  lost {f}: {len(lines)} lines")
        return 1 if lost_any else 0
    manifest = json.loads((args.out_dir / "manifests" / f"{args.phase}.json").read_text(encoding="utf-8"))
    reviewed = json.loads(args.reviewed_exclusions.read_text(encoding="utf-8")) if args.reviewed_exclusions else None
    doc = build_report(
        manifest,
        read_records(args.out_dir / "records" / args.phase),
        vendor=args.vendor,
        evidence_class=args.evidence_class,
        lease_mode=args.lease_mode,
        image=args.image,
        versions={**installed_versions(), **device_identity()},
        reviewed_exclusions=reviewed,
        coverage=coverage_summary(args.phase, args.out_dir),
    )
    _write_json(args.out_dir / "reports" / f"{args.phase}.json", doc)
    o = doc["outcomes"]
    print(
        f"phase={args.phase} {doc['evidence_class']} selected={doc['collection']['selected']}"
        f" passed={o['passed']} failed={o['failed']} errors={o['errors']} skipped={o['skipped']}"
        f" xfailed={o['xfailed']} xpassed={o['xpassed']} unexecuted={o['unexecuted']}"
        f" rank_failures={doc['rank_failures']} missing_rank_completions={doc['missing_rank_completions']}"
        f" -> {doc['classification']}"
    )
    cov = doc["coverage"]
    if cov.get("measured"):
        status = "MISSING" if cov.get("missing") else f"{cov['covered_lines']}/{cov['num_statements']} lines"
        print(f"phase={args.phase} coverage: {status}")
    return 0 if doc["classification"] == "clean" else 1


if __name__ == "__main__":
    sys.exit(main())
