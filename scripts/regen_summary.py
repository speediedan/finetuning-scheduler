#!/usr/bin/env python3
"""Compute a short top-5 package-change summary from a git diff patch.

Usage: python scripts/regen_summary.py <patch_path> <out_path>
Writes an empty file if no package-like changes are found.
"""

from pathlib import Path
import re
import sys


def parse_req(line: str):
    # strip leading +/- and whitespace
    s = line[1:].strip()
    # remove inline comments
    s = re.sub(r"\s+#.*$", "", s).strip()
    # If it's a VCS or URL spec, return the whole spec as the name for now
    if s.startswith("git+") or s.startswith("http://") or s.startswith("https://") or "egg=" in s:
        # try to extract egg name
        m = re.search(r"egg=([^&\n]+)", s)
        if m:
            return m.group(1), s
        return s, s

    # Match name[extras] optional comparator
    m = re.match(r"^([A-Za-z0-9._+\-]+)(?:\[[^\]]+\])?\s*(?:([<>!=~]{1,2})\s*(\S+))?", s)
    if m:
        name = m.group(1)
        op = m.group(2) or ""
        ver = m.group(3) or ""
        spec = (op + ver).strip()
        return name, spec
    return s, ""


def main(argv):
    if len(argv) < 3:
        print("usage: regen_summary.py <patch_path> <out_path>")
        return 2
    patch = Path(argv[1])
    out = Path(argv[2])
    if not patch.exists() or patch.stat().st_size == 0:
        out.write_text("")
        print("No patch to summarize")
        return 0

    added = {}
    removed = {}
    for ln in patch.read_text().splitlines():
        if not ln:
            continue
        if ln.startswith(("+++", "---", "diff ", "index ", "@@")):
            continue
        if ln.startswith("+"):
            name, spec = parse_req(ln)
            added[name] = spec
        elif ln.startswith("-"):
            name, spec = parse_req(ln)
            removed[name] = spec

    keys = sorted(set(list(added.keys()) + list(removed.keys())))
    changes = []
    for k in keys:
        a = added.get(k)
        r = removed.get(k)
        if r and a:
            changes.append(f"{k}: {r or '<unspecified>'} → {a or '<unspecified>'}")
        elif a and not r:
            changes.append(f"{k}: added {a or ''}")
        elif r and not a:
            changes.append(f"{k}: removed {r or ''}")

    if changes:
        keep = changes[:5]
        content = "Top changes:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(keep))
        out.write_text(content)
        print(content)
    else:
        out.write_text(
            "No package-like changes found in the req diff patch This is usually an artifact of "
            "pip-compile logging changes and can be ignored but you may inspect the req diff patch in the "
            "`regen-ci-req-report` workflow logs manually if desired."
        )
        print("No package-like changes found in patch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
