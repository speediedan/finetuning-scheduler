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
"""Keep this repo's own agent skills identical in both skills trees.

A skill written for this repository (as opposed to one vendored from a shared master, which
`check_vendored_skills.py` guards) lives in two places: `.agents/skills/<name>/`, the canonical copy that opencode,
Codex and Claude all read, and `.claude/skills/<name>/`, the compatibility copy Claude reads. Two hand-maintained
copies drift: an edit lands in whichever tree the author's harness reads, and the other harness keeps the old text
with nothing to say it is stale. This check makes the copy mechanical.

- Canonical is `.agents/skills/`. Edit there, then run `python scripts/check_skill_mirrors.py --sync`, which rewrites
  `.claude/skills/<name>/` as a byte-identical copy (removing files the canonical copy no longer has).
- The copies are real files, not symlinks: a symlink dangles for a harness or checkout that does not follow it
  (Windows checkouts without `core.symlinks`, for one).
- Vendored skills (named in `.agents/skills/.shared-skills.sha256`) are skipped: their copies come from the shared
  master.

The check exits non-zero, naming each differing skill and file, when the trees disagree, or when a repo-native skill
exists only in `.claude/skills/` (which has no canonical copy to sync from).
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANONICAL = Path(".agents/skills")
MIRROR = Path(".claude/skills")
MANIFEST = ".shared-skills.sha256"


def _vendored(tree: Path) -> set[str]:
    """Names of the skills a tree's vendored-skills manifest lists."""
    manifest = REPO / tree / MANIFEST
    if not manifest.exists():
        return set()
    names = set()
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rel = Path(line.partition("  ")[2])
            names.add(rel.relative_to(tree).parts[0])
    return names


def _skills(tree: Path, vendored: set[str]) -> set[str]:
    root = REPO / tree
    return {p.name for p in root.iterdir() if p.is_dir() and p.name not in vendored} if root.is_dir() else set()


def _files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def _differences(name: str) -> list[str]:
    canonical, mirror = REPO / CANONICAL / name, REPO / MIRROR / name
    if not mirror.is_dir():
        return [f"{MIRROR / name}/ is missing"]
    ours, theirs = _files(canonical), _files(mirror)
    problems = [f"{MIRROR / name / rel} is missing" for rel in sorted(ours - theirs)]
    problems += [f"{MIRROR / name / rel} has no canonical copy" for rel in sorted(theirs - ours)]
    problems += [f"{MIRROR / name / rel} differs" for rel in sorted(ours & theirs)
                 if not filecmp.cmp(canonical / rel, mirror / rel, shallow=False)]
    return problems


def _sync(name: str) -> None:
    mirror = REPO / MIRROR / name
    if mirror.exists():
        shutil.rmtree(mirror)
    shutil.copytree(REPO / CANONICAL / name, mirror, ignore=shutil.ignore_patterns("__pycache__"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keep this repo's own skills identical in .agents and .claude.")
    parser.add_argument("--sync", action="store_true", help=f"rewrite {MIRROR}/<name>/ from {CANONICAL}/<name>/")
    args = parser.parse_args(argv)

    vendored = _vendored(CANONICAL) | _vendored(MIRROR)
    native = _skills(CANONICAL, vendored)
    orphans = sorted(_skills(MIRROR, vendored) - native)
    if args.sync:
        for name in sorted(native):
            _sync(name)
        print(f"synced {len(native)} skill(s) from {CANONICAL} to {MIRROR}: {', '.join(sorted(native))}")
    problems = {name: diffs for name in sorted(native) if (diffs := _differences(name))}
    if orphans:
        print(f"error: skill(s) only in {MIRROR}, with no canonical copy in {CANONICAL}: {', '.join(orphans)}")
        print(f"       move each to {CANONICAL}/<name>/, then run `python scripts/check_skill_mirrors.py --sync`.")
    if problems:
        print(f"error: {MIRROR} differs from the canonical {CANONICAL} for {len(problems)} skill(s):")
        for name, diffs in problems.items():
            for diff in diffs:
                print(f"       {diff}")
        print(f"       edit {CANONICAL}/<name>/, then run `python scripts/check_skill_mirrors.py --sync`.")
    if orphans or problems:
        return 1
    if not native:
        print(f"error: no repo-native skills found in {CANONICAL}; check the path.")
        return 1
    # announce what was verified: a silent pass is indistinguishable from a hook that did not run
    print(f"skill mirrors identical: {len(native)} skill(s) in {CANONICAL} match {MIRROR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
