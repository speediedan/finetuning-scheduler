"""The docs-artifact renderer behind ``interpretune-render-notebook-docs``: executed notebook copies for a docs
site.

A repository publishing notebooks with ``notebook_publishing.py`` keeps them in up to three lanes:

1. ``dev_dir``: authoring copies.
2. ``publish_dir``: the notebooks users clone and run, kept stripped of outputs.
3. ``docs_artifact_dir``: what THIS module produces: the same notebooks, executed, **with** outputs, for the docs
   site to render in preference to lane 2. They must be exempt from any output-stripping hook, which would remove
   exactly what makes them useful.

Docs artifacts drop the publisher's ``install-deps`` cell (on a static page it is noise) and the
``injected-parameters`` cell papermill inserts.

The engine is repository-neutral and needs only the standard library plus papermill, which only execution imports.
Another repository vendors this file unmodified beside ``notebook_publishing.py`` (it loads that sibling for the
configuration) and runs it from its own ``scripts/`` directory. Everything repository-specific comes from the
``[tool.notebook-publishing]`` table::

    [tool.notebook-publishing]
    publish_dir = "examples/publish"          # shared with the publisher
    docs_artifact_dir = "docs/notebook_artifacts"   # required by this renderer
    docs_hooks = "scripts/notebook_docs_hooks.py"   # optional repository hooks, loaded by file path
    docs_render_command = "python scripts/notebook_docs.py"   # optional; how drift reports say to re-render
    [tool.notebook-publishing.docs_parameters."lane/notebook"]   # optional papermill parameters per notebook,
    epochs = 1                                                  # keyed by path under publish_dir, no suffix

A hooks file defines ``HOOKS``, a dict naming any of these callables (an unknown name is refused):

- ``stamp(notebook)``: add repository metadata to an artifact on every write.
- ``check(artifact, *, rel, output_text) -> [(kind, message), ...]``: extra ``--check-stale`` checks over an
  artifact whose sources match; ``kind`` is ``"drift"`` (fails the check) or ``"note"`` (reported only).
- ``notes() -> [message, ...]``: notes reported once per ``--check-stale`` run.
- ``restamp(artifact, *, output_text) -> (outcome, detail)``: rewrite the stamp without executing, for
  ``--restamp``; ``outcome`` is ``"unchanged"``, ``"restamped"`` or ``"refused"``. Raise ``RuntimeError`` when
  restamping cannot happen at all.

Usage::

    python notebook_docs.py --no-execute          # strip-only refresh; keeps existing outputs
    python notebook_docs.py --notebook <name>     # execute and capture outputs (needs the full environment)
    python notebook_docs.py --check-stale         # exit 3 on drift, so CI can tell drift from a crash
    python notebook_docs.py --list
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

# Drift gets its own exit code so a caller can tell it from this script FAILING TO RUN, which also exits
# non-zero. A docs build once mapped every non-zero exit to "an artifact has drifted", so when a missing
# import made the script crash it reported drift -- a cause that was not merely unproven but false. Only a
# distinct code lets the caller say WHICH happened.
#
# 3, NOT 2, and the difference is load-bearing: argparse exits 2 on a usage error, which is not something
# this script chooses (an unrecognized flag exits 2 before main() is ever entered). Using 2 for drift would
# report a mistyped or renamed flag as drift. The taken codes are 0 success, 1 uncaught exception or refused
# configuration, 2 argparse usage.
DRIFT_EXIT_CODE = 3

# Cells removed from the docs artifact, matched on the `metadata.id` the publisher sets.
DOCS_EXCLUDED_CELL_IDS = {"install-deps"}

# Cells removed by papermill TAG. `injected-parameters` is the cell papermill inserts to record the
# parameters a run was executed with. On a docs page it is actively misleading: a parameterized run
# renders e.g. `GENERATE_MISSING_LOCAL_EXPLANATIONS = True` directly beneath the notebook's own
# parameters cell documenting it as False, so the page appears to contradict itself.
DOCS_EXCLUDED_CELL_TAGS = {"injected-parameters"}

HOOK_NAMES = ("stamp", "check", "notes", "restamp")
CHECK_KINDS = ("drift", "note")
RESTAMP_OUTCOMES = ("unchanged", "restamped", "refused")
SKIP_MARKER = "[SKIPPED]"


def _load_publisher() -> ModuleType:
    """The sibling publisher module, which owns the shared configuration table.

    Imported through the package when this module was (the installed console script), and loaded from the file
    beside this one otherwise: a vendored copy, or a shim that loads this file by path to avoid importing a package
    whose ``__init__`` pulls in heavy dependencies.
    """
    if __package__:
        from . import notebook_publishing

        return notebook_publishing
    name = "_notebook_publishing_sibling"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("notebook_publishing.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve `from __future__` annotations through sys.modules
    spec.loader.exec_module(module)
    return module


_pub = _load_publisher()
NotebookConfigError = _pub.NotebookConfigError


def load_hooks(path: Path) -> dict[str, Callable[..., Any]]:
    """Load a repository's ``HOOKS`` from a Python file, refusing anything this engine would not call.

    A misspelled hook name would otherwise be silently never called, and its check would pass forever.
    """
    if not path.is_file():
        raise NotebookConfigError(f"docs_hooks names {path}, which is not a file")
    spec = importlib.util.spec_from_file_location("_notebook_docs_hooks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    hooks = getattr(module, "HOOKS", None)
    if not isinstance(hooks, dict):
        raise NotebookConfigError(f"{path} defines no HOOKS dict")
    unknown = sorted(set(hooks) - set(HOOK_NAMES))
    if unknown:
        raise NotebookConfigError(f"{path}: unknown hook(s) {unknown}; known hooks are {list(HOOK_NAMES)}")
    not_callable = sorted(name for name, fn in hooks.items() if not callable(fn))
    if not_callable:
        raise NotebookConfigError(f"{path}: hook(s) {not_callable} are not callable")
    return dict(hooks)


def _default_render_command() -> str:
    invoked = sys.argv[0] if sys.argv and sys.argv[0] else "notebook_docs.py"
    return f"python {invoked}" if invoked.endswith(".py") else Path(invoked).name


@dataclass(frozen=True)
class DocsConfig:
    """Where published notebooks are read from and executed artifacts are written, plus the repository hooks."""

    root: Path
    publish_dir: Path
    artifact_dir: Path
    render_command: str
    hooks: dict[str, Callable[..., Any]] = field(default_factory=dict)
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_pyproject(cls, root: Path) -> DocsConfig:
        """Read the shared notebook table, refusing a missing required key or a malformed one by name."""
        table = _pub.load_table(root)
        _pub.require_keys(table, ("publish_dir", "docs_artifact_dir"), "the docs-artifact renderer")
        parameters = table.get("docs_parameters", {})
        if not isinstance(parameters, dict) or not all(isinstance(v, dict) for v in parameters.values()):
            raise NotebookConfigError(
                "docs_parameters must be a table of tables keyed by notebook path under publish_dir, without suffix"
            )
        return cls(
            root=root,
            publish_dir=root / table["publish_dir"],
            artifact_dir=root / table["docs_artifact_dir"],
            render_command=str(table.get("docs_render_command") or _default_render_command()),
            hooks=load_hooks(root / table["docs_hooks"]) if "docs_hooks" in table else {},
            parameters={str(k): dict(v) for k, v in parameters.items()},
        )

    @property
    def path_markers(self) -> tuple[str, ...]:
        """The in-repository directories an absolute path is trimmed back to (see :func:`normalize_host_paths`)."""
        return tuple(
            d.relative_to(self.root).as_posix().rstrip("/") + "/" for d in (self.publish_dir, self.artifact_dir)
        )


def notebook_key(rel: Path) -> str:
    """How ``docs_parameters`` names a notebook: its path under ``publish_dir``, POSIX form, without suffix."""
    return rel.with_suffix("").as_posix()


def _is_excluded_cell(cell: dict[str, Any]) -> bool:
    meta = cell.get("metadata", {})
    if meta.get("id") in DOCS_EXCLUDED_CELL_IDS:
        return True
    return bool(DOCS_EXCLUDED_CELL_TAGS.intersection(meta.get("tags", []) or []))


def load_notebook(path: Path) -> dict[str, Any]:
    """Parse a notebook file."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(notebook: dict[str, Any], path: Path) -> None:
    """Write a notebook file, creating its directory, newline-terminated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _repo_relative(path_str: str, markers: tuple[str, ...]) -> str:
    """Trim an absolute path down to its in-repository tail, or return it unchanged."""
    for marker in markers:
        i = path_str.find(marker)
        if i != -1:
            return path_str[i:]
    return path_str


def _user_root_pattern(user: str) -> re.Pattern[str]:
    """Match the executing user's root wherever it lives, and ONLY as a filesystem path.

    Their files are not all under `/home`. Measured on real artifacts: worktrees under
    `/mnt/cache/<user>/worktrees/<name>/`, virtualenvs under `/mnt/cache/<user>/.venvs/`, and a Hub
    cache under `/mnt/cache_extended/<user>/.cache/`. A `/home`-only rule silently misses every one,
    which is how the first version of this function passed its own check while leaving 22 paths in
    place: the pattern had been written from the examples that had already been noticed.

    The leading lookbehind is what keeps this safe. `huggingface.co/<user>/rte` names a real published
    Hub repository, and the segment after the host looks exactly like a path component. Requiring the
    match to begin at a `/` that does not follow a word character, a colon OR ANOTHER SLASH means a
    URL cannot match, while a path appearing after a space, a quote or a bracket can. The slash in
    that list is not defensive padding: without it `https://github.com/<user>/<repo>` matched
    from the second slash of `//` and became `https:/~/<repo>`, corrupting a working link.
    """
    return re.compile(r"(?<![\w.:/])/(?:[\w.\-]+/)*?" + re.escape(user) + r"/")


def normalize_host_paths(notebook: dict[str, Any], *, markers: tuple[str, ...], user: str | None = None) -> int:
    """Strip the executing host's identity out of what gets published.

    An artifact records WHERE it was rendered, which is whoever ran the render host that day. Two places
    carry it, and they need different treatment because only one has any value to a reader.

    `metadata.papermill.input_path` / `output_path` are absolute paths into the renderer's own working
    copy, often a scratch worktree, which says nothing about the example. Rewritten to the in-repository
    tail starting at one of ``markers`` (the publish and artifact directories), the part true for every reader.

    Output TEXT carries absolute paths inside warning locations, e.g.
    `.../src/pkg/config.py:222: UserWarning ...`. Only the user's root is replaced, leaving the rest intact,
    because the file and line ARE the useful content of a warning.

    What this deliberately does NOT touch: `owner/repo` identifiers naming real published repositories the
    examples genuinely use, and prose in source cells. **They look alike to a grep and are opposites in
    intent**, which is why this works on parsed notebook structure and on anchored path patterns rather than
    on the file's raw text.
    """
    user = user or getpass.getuser()
    user_root = _user_root_pattern(user)
    changed = 0
    pm_meta = (notebook.get("metadata") or {}).get("papermill") or {}
    for key in ("input_path", "output_path"):
        val = pm_meta.get(key)
        if isinstance(val, str):
            rel = user_root.sub("~/", _repo_relative(val, markers))
            if rel != val:
                pm_meta[key] = rel
                changed += 1
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []) or []:
            for fld in ("text", "traceback"):
                lines = output.get(fld)
                if isinstance(lines, list):
                    swapped = [user_root.sub("~/", ln) if isinstance(ln, str) else ln for ln in lines]
                    if swapped != lines:
                        output[fld] = swapped
                        changed += 1
    return changed


def strip_docs_excluded_cells(notebook: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Drop cells that only make sense in a runnable copy (the commented-out installer, papermill's record)."""
    cells = notebook.get("cells", [])
    kept = [c for c in cells if not _is_excluded_cell(c)]
    return notebook | {"cells": kept}, len(cells) - len(kept)


def has_outputs(notebook: dict[str, Any]) -> bool:
    """Whether any code cell carries captured output."""
    return any(c.get("outputs") for c in notebook.get("cells", []) if c.get("cell_type") == "code")


def captured_output_text(notebook: dict[str, Any]) -> str:
    """All captured output text in one blob, which is the half `artifact_matches_source` cannot see."""
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []) or []:
            chunks.append("".join(output.get("text", []) or []))
            for mime, payload in (output.get("data") or {}).items():
                if mime.startswith("text/"):
                    chunks.append("".join(payload if isinstance(payload, list) else [payload]))
    return "".join(chunks)


def skipped_sections(artifact: dict[str, Any]) -> list[str]:
    """Output lines announcing a section the notebook enabled but the render skipped.

    A notebook section that needs something the render host lacked (a private Hub repo the token could
    not see, a local service) prints ``[SKIPPED] <section>: <reason>`` and moves on, by design: a reader
    without that access still gets the rest. The rendered artifact is the exception. The docs show it as
    the section's output, so a skip there is a page silently missing the thing the section exists to
    show, with no build error anywhere. Only OUTPUTS are read: the marker in a cell's source is the print
    that produces it, not evidence that it fired.
    """
    found: list[str] = []
    for cell in artifact.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = captured_output_text({"cells": [cell]})
        found.extend(line.strip() for line in text.splitlines() if line.strip().startswith(SKIP_MARKER))
    return found


def artifact_matches_source(artifact: dict[str, Any], source: dict[str, Any]) -> bool:
    """Do artifact and publish source agree on everything except outputs?

    The docs site renders the ARTIFACT in preference to the publish copy. So a notebook edit that changes only prose is
    invisible on the site until the artifact is rebuilt -- the page keeps rendering the old text with no error anywhere.
    This compares cell types and sources (ignoring the cells the artifact deliberately drops) so that drift is
    detectable.
    """

    def signature(notebook: dict[str, Any]) -> list[tuple[str, str]]:
        return [
            (c["cell_type"], "".join(c.get("source", [])))
            for c in notebook.get("cells", [])
            if not _is_excluded_cell(c)
        ]

    return signature(artifact) == signature(source)


def discover(cfg: DocsConfig, selector: str | None) -> list[Path]:
    """Published notebooks, optionally narrowed to those whose path under ``publish_dir`` contains ``selector``."""
    paths = sorted(cfg.publish_dir.rglob("*.ipynb"))
    if selector:
        paths = [p for p in paths if selector in str(p.relative_to(cfg.publish_dir))]
    return paths


def unknown_parameter_keys(cfg: DocsConfig) -> list[str]:
    """``docs_parameters`` keys naming no published notebook: a typo would otherwise run with the defaults."""
    known = {notebook_key(p.relative_to(cfg.publish_dir)) for p in discover(cfg, None)}
    return sorted(set(cfg.parameters) - known)


def execute(source: Path, target: Path, timeout: int, parameters: dict[str, Any]) -> None:
    """Execute via papermill, writing the executed copy to ``target``."""
    import papermill as pm

    target.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        str(source),
        str(target),
        parameters=parameters or {},
        kernel_name="python3",
        cwd=str(source.parent),  # notebooks resolve sidecar assets relative to themselves
        progress_bar=False,
        execution_timeout=timeout,
    )


def check_stale(cfg: DocsConfig, notebooks: list[Path]) -> int:
    """Report every drifted, missing or skip-carrying artifact by name; ``DRIFT_EXIT_CODE`` if there is any."""
    missing: list[str] = []
    stale: list[str] = []
    skipped: list[str] = []
    hook_drift: list[str] = []
    notes: list[str] = []
    check = cfg.hooks.get("check")
    for source in notebooks:
        rel = source.relative_to(cfg.publish_dir)
        artifact = cfg.artifact_dir / rel
        if not artifact.exists():
            # A published notebook with NO artifact renders code-only on the site and nothing fails, so an absent
            # artifact is drift of the same kind as a stale one, reported by name with the command that fixes it.
            missing.append(str(rel))
            continue
        artifact_nb = load_notebook(artifact)
        if not artifact_matches_source(artifact_nb, load_notebook(source)):
            stale.append(str(rel))
            continue  # a source-drifted artifact needs a rebuild regardless of its outputs
        skipped.extend(f"{rel}: {line}" for line in skipped_sections(artifact_nb))
        if check is None:
            continue
        for kind, message in check(artifact_nb, rel=str(rel), output_text=captured_output_text(artifact_nb)):
            if kind not in CHECK_KINDS:
                raise NotebookConfigError(f"check hook returned kind {kind!r}; expected one of {list(CHECK_KINDS)}")
            (hook_drift if kind == "drift" else notes).append(message)
    if (global_notes := cfg.hooks.get("notes")) is not None:
        notes.extend(global_notes())

    for rel_str in missing:
        print(
            f"MISSING artifact (render it: {cfg.render_command} --notebook {Path(rel_str).stem}): {rel_str}",
            file=sys.stderr,
        )
    for rel_str in stale:
        print(f"STALE artifact (rebuild it): {rel_str}", file=sys.stderr)
    for entry in skipped:
        print(
            f"SKIPPED section in artifact (re-render it on a host with the access the section needs): {entry}",
            file=sys.stderr,
        )
    for message in hook_drift + notes:
        print(message, file=sys.stderr)
    total = len(missing) + len(stale) + len(skipped) + len(hook_drift)
    print(f"{total} stale or missing artifact(s); {len(notes)} note(s)")
    return DRIFT_EXIT_CODE if total else 0


def restamp(cfg: DocsConfig, notebooks: list[Path], *, dry_run: bool) -> int:
    """Rewrite each existing artifact's repository stamp through the ``restamp`` hook, without executing."""
    hook = cfg.hooks.get("restamp")
    if hook is None:
        print("error: --restamp needs a docs_hooks file whose HOOKS define restamp", file=sys.stderr)
        return 1
    restamped = refused = 0
    for source in notebooks:
        rel = source.relative_to(cfg.publish_dir)
        artifact = cfg.artifact_dir / rel
        if not artifact.exists():
            continue
        artifact_nb = load_notebook(artifact)
        try:
            outcome, detail = hook(artifact_nb, output_text=captured_output_text(artifact_nb))
        except RuntimeError as exc:
            print(f"error: cannot restamp: {exc}", file=sys.stderr)
            return 1
        if outcome not in RESTAMP_OUTCOMES:
            raise NotebookConfigError(f"restamp hook returned {outcome!r}; expected one of {list(RESTAMP_OUTCOMES)}")
        if outcome == "unchanged":
            continue
        if outcome == "refused":
            print(f"REFUSED {rel}: {detail}", file=sys.stderr)
            refused += 1
        elif dry_run:
            print(f"[dry-run] restamp {rel}")
        else:
            save_notebook(artifact_nb, artifact)
            print(f"restamped {rel}")
            restamped += 1
    print(f"{restamped} artifact(s) restamped; {refused} refused")
    return 1 if refused else 0


def render(
    cfg: DocsConfig, notebooks: list[Path], *, execute_cells: bool, timeout: int, parameters: dict[str, Any]
) -> int:
    """Execute (or, without ``execute_cells``, re-strip) each notebook into its docs artifact."""
    failures: list[str] = []
    stamp = cfg.hooks.get("stamp")
    for source in notebooks:
        rel = source.relative_to(cfg.publish_dir)
        artifact = cfg.artifact_dir / rel
        try:
            if execute_cells:
                print(f"executing {rel} ...", flush=True)
                execute(source, artifact, timeout, {**cfg.parameters.get(notebook_key(rel), {}), **parameters})
            elif not artifact.exists():
                # Preserve outputs already captured; only fall back to the stripped copy.
                artifact.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, artifact)

            notebook, removed = strip_docs_excluded_cells(load_notebook(artifact))
            # Stamp and normalize on EVERY write, including a strip-only one: it does not change outputs, so the
            # stamp still describes them, and an artifact lands identically regardless of whose copy rendered it.
            if stamp is not None:
                stamp(notebook)
            normalize_host_paths(notebook, markers=cfg.path_markers)
            save_notebook(notebook, artifact)
            outputs = "with outputs" if has_outputs(notebook) else "NO outputs"
            print(f"  wrote {rel} ({outputs}, {removed} cell(s) removed)")
        except Exception as exc:  # - report and continue to the next notebook
            # Discard whatever papermill managed to write. A half-executed notebook still lands on disk, carrying
            # the traceback as an error output and (because the strip step never ran) the install-deps cell, and
            # the docs site would render that traceback as though it were the example's real output. Removing it
            # makes the page fall back to code-only, which is the honest result.
            if artifact.exists():
                artifact.unlink()
                print(f"  discarded partial artifact for {rel}", file=sys.stderr)
            failures.append(f"{rel}: {type(exc).__name__}: {exc}")
            print(f"  FAILED {rel}: {type(exc).__name__}: {exc}", file=sys.stderr)

    # Sidecar assets (images, helper modules) the notebooks reference relatively.
    for asset in cfg.publish_dir.rglob("*"):
        if asset.is_file() and asset.suffix != ".ipynb" and not asset.name.startswith("."):
            target = cfg.artifact_dir / asset.relative_to(cfg.publish_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset, target)

    if failures:
        print(f"\n{len(failures)} notebook(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    """``interpretune-render-notebook-docs [--root DIR] [--notebook SUBSTR] [--check-stale | --restamp |
    --list]``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None, help="repository root (default: nearest pyproject.toml)")
    parser.add_argument("--notebook", help="Substring selecting which notebooks to process (default: all).")
    parser.add_argument(
        "--no-execute",
        dest="execute",
        action="store_false",
        help="Do not run anything. Re-strips excluded cells and preserves any outputs already present.",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Per-cell execution timeout (default: 1800).")
    parser.add_argument(
        "--parameters",
        default="{}",
        help="JSON dict of papermill parameters, applied over the notebook's docs_parameters from the table.",
    )
    parser.add_argument("--list", action="store_true", help="List candidate notebooks and their artifact status.")
    parser.add_argument(
        "--check-stale",
        action="store_true",
        help=f"Exit {DRIFT_EXIT_CODE} if any artifact has drifted from its publish source, is missing, carries a "
        "skipped section, or fails a repository check. The docs render artifacts, so any of these means the site "
        "is showing something stale with no build error.",
    )
    parser.add_argument(
        "--restamp",
        action="store_true",
        help="Rewrite the repository hook's stamp on existing artifacts WITHOUT executing anything (needs a "
        "restamp hook). The hook refuses an artifact whose outputs need a real re-execute.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen; write nothing.")
    args = parser.parse_args(argv)

    try:
        cfg = DocsConfig.from_pyproject(_pub.find_root(args.root or root))
        if unknown := unknown_parameter_keys(cfg):
            raise NotebookConfigError(
                f"docs_parameters names notebook(s) {unknown} not found under {cfg.publish_dir} "
                "(keys are paths under publish_dir without suffix)"
            )
    except NotebookConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    notebooks = discover(cfg, args.notebook)
    if not notebooks:
        print(f"No notebooks matched under {cfg.publish_dir}", file=sys.stderr)
        return 1

    if args.check_stale:
        return check_stale(cfg, notebooks)
    if args.restamp:
        return restamp(cfg, notebooks, dry_run=args.dry_run)
    if args.list:
        for source in notebooks:
            rel = source.relative_to(cfg.publish_dir)
            artifact = cfg.artifact_dir / rel
            if not artifact.exists():
                state = "MISSING  (docs render code-only)"
            elif has_outputs(load_notebook(artifact)):
                state = "executed (docs render real output)"
            else:
                state = "no outputs (docs render code-only)"
            print(f"  {state:<34} {rel}")
        return 0
    if args.dry_run:
        for source in notebooks:
            print(f"[dry-run] {'execute' if args.execute else 'copy'} {source.relative_to(cfg.publish_dir)}")
        return 0
    return render(
        cfg, notebooks, execute_cells=args.execute, timeout=args.timeout, parameters=json.loads(args.parameters)
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
