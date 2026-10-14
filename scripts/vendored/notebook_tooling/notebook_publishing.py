"""The notebook publishing engine behind ``interpretune-publish-notebooks``.

Copies a repository's dev notebooks to its publish directory, strips ``remove-cell`` tagged cells, rewrites
dev-path imports, and prepends a Colab badge and an install cell, tracking content hashes so only changed files
republish. It is repository-neutral and imports only the standard library, so another repository can vendor this
file unmodified (beside ``notebook_docs.py``, the docs-artifact renderer, which loads it as a sibling) and run it
from its own ``scripts/`` directory without installing interpretune. ``scripts/publish_notebooks.py`` in this
repository is a shim over :func:`main`.

Everything repository-specific comes from the ``[tool.notebook-publishing]`` table of the consuming repository's
``pyproject.toml``; the engine holds no repository paths. ``[tool.interpretune.notebooks]``, the table's earlier
name, is still read, and a pyproject carrying both is refused rather than merged::

    [tool.notebook-publishing]
    dev_dir = "examples/dev"                  # required; relative to the pyproject directory
    publish_dir = "examples/publish"          # required
    colab_repo = "org/repo"                   # required; the GitHub org/repo the badge opens
    install_cell = ["# %pip install my-package[examples]"]   # required; lines of the prepended install cell
    colab_branch = "main"                     # optional
    hash_key_base = ""                        # optional; the directory stored hash keys are relative to
    [tool.notebook-publishing.import_rewrites]   # optional substring rewrites applied to code cells
    "examples.dev." = "examples.publish."

The ``docs_*`` keys of the same table configure the docs-artifact renderer (see ``notebook_docs.py``). A key
neither tool reads is refused by name, so a misspelled key fails instead of silently falling back to a default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HASH_FILE = ".notebook_hashes.json"
TABLE = ("tool", "notebook-publishing")
LEGACY_TABLE = ("tool", "interpretune", "notebooks")
EXCLUDED_PARTS = ("__pycache__",)
EXCLUDED_SUFFIXES = (".pyc",)

PUBLISHER_REQUIRED_KEYS = ("dev_dir", "publish_dir", "colab_repo", "install_cell")
PUBLISHER_OPTIONAL_KEYS = ("colab_branch", "hash_key_base", "import_rewrites")
# The renderer's keys share the table and are listed here, in the module both tools load, so that either tool
# can refuse a key that NEITHER reads.
DOCS_KEYS = ("docs_artifact_dir", "docs_hooks", "docs_parameters", "docs_render_command")
KNOWN_KEYS = frozenset(PUBLISHER_REQUIRED_KEYS + PUBLISHER_OPTIONAL_KEYS + DOCS_KEYS)


class NotebookConfigError(ValueError):
    """The repository's notebook table is missing, ambiguous, or names a key no tool here reads."""


def _table_name(path: tuple[str, ...]) -> str:
    return "[" + ".".join(path) + "]"


def _lookup(data: Any, path: tuple[str, ...]) -> dict[str, Any] | None:
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return dict(data) if isinstance(data, dict) else None


def load_table(root: Path) -> dict[str, Any]:
    """The notebook table from ``root/pyproject.toml``, with every key checked against the ones the tools read.

    Raises :class:`NotebookConfigError` naming what was met: no pyproject, no table, both the current and the legacy
    table (which would win is a guess, so neither does), or a key no tool reads.
    """
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        raise NotebookConfigError(f"no pyproject.toml at {root}; the notebook tools read {_table_name(TABLE)} there")
    data = _load_toml(pyproject)
    current, legacy = _lookup(data, TABLE), _lookup(data, LEGACY_TABLE)
    if current is not None and legacy is not None:
        raise NotebookConfigError(
            f"{pyproject} carries both {_table_name(TABLE)} and the legacy {_table_name(LEGACY_TABLE)}; keep one"
        )
    table = current if current is not None else legacy
    if table is None:
        raise NotebookConfigError(f"{pyproject} has no {_table_name(TABLE)} table")
    unknown = sorted(set(table) - KNOWN_KEYS)
    if unknown:
        raise NotebookConfigError(f"{pyproject}: unknown notebook-table key(s) {unknown}; known: {sorted(KNOWN_KEYS)}")
    return table


def require_keys(table: dict[str, Any], keys: tuple[str, ...], purpose: str) -> None:
    """Refuse, by name, a table missing any of ``keys``; ``purpose`` names the tool that needs them."""
    missing = [k for k in keys if k not in table]
    if missing:
        raise NotebookConfigError(f"{_table_name(TABLE)} is missing {missing}, required by {purpose}")


@dataclass(frozen=True)
class PublisherConfig:
    """Where notebooks come from and go to, and what the published copies get prepended."""

    root: Path
    dev_dir: Path
    publish_dir: Path
    colab_repo: str
    install_cell: tuple[str, ...]
    colab_branch: str = "main"
    hash_key_base: Path | None = None
    import_rewrites: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_pyproject(cls, root: Path) -> PublisherConfig:
        """Read the notebook table from ``root/pyproject.toml``, refusing a missing required key by name."""
        table = load_table(root)
        require_keys(table, PUBLISHER_REQUIRED_KEYS, "the notebook publisher")
        return cls(
            root=root,
            dev_dir=root / table["dev_dir"],
            publish_dir=root / table["publish_dir"],
            colab_repo=str(table["colab_repo"]),
            install_cell=tuple(_as_lines(table["install_cell"])),
            colab_branch=str(table.get("colab_branch", "main")),
            hash_key_base=root / str(table.get("hash_key_base", "")),
            import_rewrites=dict(table.get("import_rewrites", {})),
        )

    def hash_key(self, dev_path: Path) -> str:
        """The stable key a file is tracked under: its path relative to ``hash_key_base`` (default: the root)."""
        return dev_path.relative_to(self.hash_key_base or self.root).as_posix()


def _as_lines(value: Any) -> list[str]:
    lines = [str(v) for v in (value if isinstance(value, (list, tuple)) else str(value).splitlines())]
    return [line if line.endswith("\n") else line + "\n" for line in lines]


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - 3.10 only
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def find_root(start: Path | None = None) -> Path:
    """The nearest ancestor of ``start`` (default: cwd) holding a ``pyproject.toml``, or ``start`` itself."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


# -- hashing ---------------------------------------------------------------------------------------


def compute_file_hash(file_path: Path) -> str:
    """SHA256 of a file's bytes with CRLF folded to LF.

    A checkout with ``core.autocrlf`` (the Windows runners) rewrites every text file's line endings, and a raw
    hash then reports every notebook as out of date on one platform only. Folding applies to text files and is a
    no-op for LF ones, so stored hashes stay valid; binary files hash raw.
    """
    with open(file_path, "rb") as fh:
        data = fh.read()
    if _is_text(data):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _is_text(data: bytes) -> bool:
    """UTF-8 with no NUL byte: the files a CRLF checkout rewrites. A PNG with an incidental 0x0d0a stays raw."""
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def load_notebook_hashes(publish_dir: Path) -> dict[str, str]:
    """The stored hashes, or an empty mapping when nothing has been published yet."""
    hash_file = publish_dir / HASH_FILE
    if hash_file.exists():
        with open(hash_file, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_notebook_hashes(publish_dir: Path, hashes: dict[str, str]) -> None:
    """Write the hashes with a trailing newline (the end-of-file-fixer hook would otherwise rewrite the file on the
    next commit, change its hash, and make this engine republish forever)."""
    publish_dir.mkdir(parents=True, exist_ok=True)
    with open(publish_dir / HASH_FILE, "w", encoding="utf-8") as fh:
        json.dump(hashes, fh, indent=2, sort_keys=True)
        fh.write("\n")


def publishable_files(dev_dir: Path) -> list[Path]:
    """Every file under ``dev_dir`` that is a publication input: bytecode caches are never one."""
    return sorted(
        p
        for p in dev_dir.rglob("*")
        if p.is_file() and not any(part in EXCLUDED_PARTS for part in p.parts) and p.suffix not in EXCLUDED_SUFFIXES
    )


def changed_files(cfg: PublisherConfig, stored: dict[str, str]) -> list[Path]:
    """The publication inputs whose content hash differs from the stored one."""
    return [p for p in publishable_files(cfg.dev_dir) if stored.get(cfg.hash_key(p)) != compute_file_hash(p)]


# -- notebook transforms ---------------------------------------------------------------------------


def load_notebook(path: Path) -> dict[str, Any]:
    """Parse a notebook file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_notebook(notebook: dict[str, Any], path: Path) -> None:
    """Write a notebook file, newline-terminated (see :func:`save_notebook_hashes`)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(notebook, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def strip_remove_cell_tags(notebook: dict[str, Any]) -> dict[str, Any]:
    """Drop every cell tagged ``remove-cell``."""
    notebook["cells"] = [
        c for c in notebook.get("cells", []) if "remove-cell" not in (c.get("metadata", {}).get("tags") or [])
    ]
    return notebook


def rewrite_imports(notebook: dict[str, Any], rewrites: dict[str, str]) -> dict[str, Any]:
    """Apply the configured substring rewrites to every code cell, preserving the list-of-lines form.

    Every code cell is rewritten into the canonical form (the last line carries no newline), whether or not a
    rewrite matched, so republishing an unchanged notebook reproduces its published copy byte for byte. Cell
    metadata is untouched, so a tagged cell such as papermill's ``parameters`` cell keeps its tag and its code.
    """
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else source
        for old, new in rewrites.items():
            text = text.replace(old, new)
        if isinstance(source, list):
            lines = text.split("\n")
            if text.endswith("\n"):
                lines = lines[:-1]
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
        else:
            cell["source"] = text
    return notebook


def colab_url(cfg: PublisherConfig, publish_path: Path) -> str:
    """The Colab link for a published notebook."""
    rel = publish_path.relative_to(cfg.root).as_posix()
    return f"https://colab.research.google.com/github/{cfg.colab_repo}/blob/{cfg.colab_branch}/{rel}"


def add_colab_badge_and_install_cell(
    notebook: dict[str, Any], cfg: PublisherConfig, publish_path: Path
) -> dict[str, Any]:
    """Prepend the badge markdown cell and the install code cell, identified by ``metadata.id`` so downstream
    tooling (the docs artifact renderer) can find and drop them."""
    badge = {
        "cell_type": "markdown",
        "metadata": {"id": "colab-badge"},
        "source": [
            f'<a href="{colab_url(cfg, publish_path)}">\n',
            '  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />\n',
            "</a>",
        ],
    }
    install = {
        "cell_type": "code",
        "metadata": {"id": "install-deps", "language": "python"},
        "source": list(cfg.install_cell),
        "outputs": [],
        "execution_count": None,
    }
    notebook["cells"] = [badge, install] + notebook.get("cells", [])
    return notebook


def publish_file(cfg: PublisherConfig, dev_path: Path, *, dry_run: bool = False) -> Path:
    """Publish one dev file (a notebook is transformed; anything else is copied); returns the publish path."""
    publish_path = cfg.publish_dir / dev_path.relative_to(cfg.dev_dir)
    if dry_run:
        print(f"WOULD {'PROCESS' if dev_path.suffix == '.ipynb' else 'COPY'}: {dev_path} -> {publish_path}")
        return publish_path
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    if dev_path.suffix == ".ipynb":
        notebook = strip_remove_cell_tags(load_notebook(dev_path))
        notebook = rewrite_imports(notebook, cfg.import_rewrites)
        notebook = add_colab_badge_and_install_cell(notebook, cfg, publish_path)
        save_notebook(notebook, publish_path)
        print(f"PROCESSED: {dev_path} -> {publish_path}")
    else:
        shutil.copy2(dev_path, publish_path)
        print(f"COPIED: {dev_path} -> {publish_path}")
    return publish_path


# -- entry point -----------------------------------------------------------------------------------


def publish(cfg: PublisherConfig, *, dry_run: bool = False, check_only: bool = False, force: bool = False) -> int:
    """Run one publication pass; the exit code is ``1`` from ``check_only`` when anything is out of date."""
    if not cfg.dev_dir.is_dir():
        print(f"ERROR: dev directory not found: {cfg.dev_dir}")
        return 1
    stored = load_notebook_hashes(cfg.publish_dir)
    pending = publishable_files(cfg.dev_dir) if force else changed_files(cfg, stored)
    if check_only:
        if pending:
            print(f"Found {len(pending)} file(s) that need publishing:")
            for path in pending:
                print(f"  {cfg.hash_key(path)}")
            return 1
        print("All files are up to date.")
        return 0
    if not pending:
        print("All files are up to date. No files to publish.")
        return 0
    notebooks = [p for p in pending if p.suffix == ".ipynb"]
    print(f"Found {len(pending)} files to publish\nSource: {cfg.dev_dir}\nTarget: {cfg.publish_dir}")
    if notebooks:
        print(f"Notebooks: {len(notebooks)}")
    print()
    updated = dict(stored)
    for path in pending:
        publish_file(cfg, path, dry_run=dry_run)
        updated[cfg.hash_key(path)] = compute_file_hash(path)
    if dry_run:
        print(f"\nDry run complete. Would publish {len(pending)} files.")
    else:
        save_notebook_hashes(cfg.publish_dir, updated)
        print(f"\nPublished {len(pending)} files.")
    return 0


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    """``interpretune-publish-notebooks [--root DIR] [--dry-run] [--check-only] [--force]``."""
    parser = argparse.ArgumentParser(
        prog="interpretune-publish-notebooks",
        description="Publish a repository's dev notebooks to its publish directory ([tool.notebook-publishing]).",
    )
    parser.add_argument("--root", type=Path, default=None, help="repository root (default: nearest pyproject.toml)")
    parser.add_argument("--dry-run", action="store_true", help="show what would be done without writing")
    parser.add_argument("--check-only", action="store_true", help="exit 1 when any notebook needs publishing")
    parser.add_argument("--force", action="store_true", help="republish every file, ignoring stored hashes")
    args = parser.parse_args(argv)
    try:
        cfg = PublisherConfig.from_pyproject(find_root(args.root or root))
    except NotebookConfigError as exc:
        print(f"ERROR: {exc}")
        return 1
    return publish(cfg, dry_run=args.dry_run, check_only=args.check_only, force=args.force)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
