#!/usr/bin/env python3
"""Report whether this repo's vendored skills still match the shared master they were copied from.

``check_vendored_skills.py`` answers "was a vendored copy edited here", offline, on every commit. This answers the
other question, "has the master moved ahead of the copy", which needs the master and so cannot be a pre-commit hook: a
hook that fails on a clean offline clone teaches contributors to pass ``--no-verify``. Run it as its own CI job (or by
hand), where reading the master over the network is expected.

The vendored skills are the ones the manifest names (``.agents/skills/.shared-skills.sha256``, and the compat
``.claude/skills/.shared-skills.sha256`` when present), so repo-local skills beside them are never compared. Each
vendored skill directory is compared file by file with the same skill in the master, at one resolved commit:

- a file whose bytes differ is printed as a unified diff;
- a file only the master has means a refresh is due (the master added it);
- a file only the copy has means the master removed or renamed it, and a stale script beside a current ``SKILL.md``
  still runs, so it counts as drift too;
- a vendored skill the master does not have at all is refused, since there is nothing to compare it with.

Exit status: 0 every vendored skill identical to the master, 1 any drifted (each difference is printed), 2 the
master, a manifest or a vendored directory could not be read.

Fix drift by refreshing the vendored copies (and their manifests) from the master, never by editing them here.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = "speediedan/skills"
MANIFEST = ".shared-skills.sha256"
# The neutral primary first, then the compat tree some repos keep through the agent-neutral migration.
PRIMARY_DIR, COMPAT_DIR = Path(".agents/skills"), Path(".claude/skills")
# Byproducts of running a vendored skill's scripts, never part of the skill.
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc"}


class Unreadable(Exception):
    """The master, a manifest or a vendored directory could not be read; the comparison cannot be made."""


def _ignored(rel: Path) -> bool:
    return bool(IGNORED_PARTS.intersection(rel.parts)) or rel.suffix in IGNORED_SUFFIXES


def vendored_skills(skills_dir: Path) -> list[str]:
    """The skill names a tree's manifest lists, in manifest order."""
    manifest = REPO / skills_dir / MANIFEST
    if not manifest.is_file():
        raise Unreadable(f"{skills_dir / MANIFEST} is missing, so the vendored skills there are unknown")
    names: list[str] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel = Path(line.partition("  ")[2])
        try:
            name = rel.relative_to(skills_dir).parts[0]
        except (ValueError, IndexError):
            raise Unreadable(f"{skills_dir / MANIFEST} lists {rel}, which is not under {skills_dir}") from None
        if name not in names:
            names.append(name)
    return names


def local_files(skill_dir: Path) -> dict[str, bytes]:
    """Every file of one vendored skill, keyed by its path inside the skill directory."""
    if not skill_dir.is_dir():
        raise Unreadable(f"{skill_dir.relative_to(REPO)} is named in the manifest but is not a directory")
    files = {}
    for path in sorted(skill_dir.rglob("*")):
        rel = path.relative_to(skill_dir)
        if path.is_file() and not _ignored(rel):
            files[rel.as_posix()] = path.read_bytes()
    return files


class Master:
    """The master's skills at one commit: from a local clone (``against_dir``) or from GitHub."""

    def __init__(self, repo: str, ref: str, against_dir: str | None) -> None:
        self.repo, self.against_dir = repo, against_dir
        if against_dir:
            root = Path(against_dir)
            self.label = against_dir
            self.index = self._index(p.relative_to(root).as_posix() for p in root.glob("plugins/*/skills/**/*"))
            return
        self.sha = self._api(f"repos/{repo}/commits/{ref}")["sha"]
        tree = self._api(f"repos/{repo}/git/trees/{self.sha}?recursive=1")
        if tree.get("truncated"):
            # A partial listing would report every file it omitted as removed upstream.
            raise Unreadable(f"GitHub truncated the tree listing of {repo}@{self.sha}")
        self.label = f"{repo}@{self.sha[:12]}"
        self.index = self._index(e["path"] for e in tree["tree"] if e["type"] == "blob")

    def _index(self, paths) -> dict[str, dict[str, str]]:
        """``{skill: {path inside the skill: path in the master}}`` over ``plugins/<p>/skills/<skill>/``."""
        index: dict[str, dict[str, str]] = {}
        for path in paths:
            parts = Path(path).parts
            if len(parts) < 5 or parts[0] != "plugins" or parts[2] != "skills":
                continue
            if self.against_dir and not (Path(self.against_dir) / path).is_file():
                continue
            rel = Path(*parts[4:])
            if not _ignored(rel):
                index.setdefault(parts[3], {})[rel.as_posix()] = path
        return index

    def _get(self, url: str, accept: str) -> bytes:
        headers = {"User-Agent": "check-skills-upstream", "Accept": accept}
        token = os.environ.get("GITHUB_TOKEN")
        if token and url.startswith("https://api.github.com/"):
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as resp:
                return resp.read()
        except (OSError, urllib.error.URLError) as exc:
            raise Unreadable(f"could not read {url}: {exc}") from None

    def _api(self, endpoint: str) -> dict:
        return json.loads(self._get(f"https://api.github.com/{endpoint}", "application/vnd.github+json"))

    def read(self, path: str) -> bytes:
        if self.against_dir:
            return (Path(self.against_dir) / path).read_bytes()
        return self._get(f"https://raw.githubusercontent.com/{self.repo}/{self.sha}/{path}", "*/*")


def compare_skill(name: str, skill_dir: Path, master: Master) -> list[str]:
    """Every difference between one vendored skill and the master's, as printable lines; empty when identical."""
    ours = local_files(skill_dir)
    theirs = master.index[name]
    shown = skill_dir.relative_to(REPO).as_posix()
    out: list[str] = []
    for rel in sorted(set(ours) | set(theirs)):
        if rel not in theirs:
            out.append(f"  {shown}/{rel}: not in {master.label} (removed or renamed there; delete it here)\n")
        elif rel not in ours:
            out.append(f"  {shown}/{rel}: missing here (added in {master.label})\n")
        else:
            other = master.read(theirs[rel])
            if other != ours[rel]:
                diff = list(
                    difflib.unified_diff(
                        ours[rel].decode(errors="replace").splitlines(keepends=True),
                        other.decode(errors="replace").splitlines(keepends=True),
                        fromfile=f"{shown}/{rel}",
                        tofile=f"{master.label}:{theirs[rel]}",
                    )
                )
                out.extend(diff or [f"  {shown}/{rel}: differs only in bytes a text diff does not show\n"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=UPSTREAM, help=f"the master's GitHub repository (default {UPSTREAM})")
    ap.add_argument("--ref", default="main", help="the master's branch, tag or commit (default main)")
    ap.add_argument("--against-dir", help="compare with a local clone of the master instead of GitHub")
    ap.add_argument("--root", type=Path, help="the consuming repo to check (default: the repo this script sits in)")
    ap.add_argument(
        "--skills-dir",
        action="append",
        type=Path,
        help="a vendored tree to check (repeatable; default .agents/skills, and .claude/skills if it has a manifest)",
    )
    args = ap.parse_args(argv)
    if args.root:
        global REPO
        REPO = args.root.resolve()
    dirs = args.skills_dir or [PRIMARY_DIR] + ([COMPAT_DIR] if (REPO / COMPAT_DIR / MANIFEST).is_file() else [])
    try:
        master = Master(args.repo, args.ref, args.against_dir)
        status = 0
        for skills_dir in dirs:
            for name in vendored_skills(skills_dir):
                if name not in master.index:
                    print(f"REFUSED: {skills_dir / name} is vendored, but {master.label} has no skill named {name}")
                    status = 2
                    continue
                drift = compare_skill(name, REPO / skills_dir / name, master)
                if drift:
                    print(f"{skills_dir / name} has drifted from {master.label}; refresh it from the master:")
                    sys.stdout.writelines(drift)
                    status = max(status, 1)
                else:
                    print(f"{skills_dir / name}: {len(master.index[name])} file(s) identical to {master.label}")
    except Unreadable as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return status


if __name__ == "__main__":
    sys.exit(main())
