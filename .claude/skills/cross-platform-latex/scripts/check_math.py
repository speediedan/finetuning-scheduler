#!/usr/bin/env python3
"""Flag LaTeX-in-Markdown constructs that render differently across GitHub, MyST and MkDocs.

Dependency-free by design (stdlib only): this has to run in a pre-commit hook, in CI, and on a
contributor's machine that has installed nothing.

Three modes:
  check_math.py FILE...                lint math constructs (default)
  check_math.py --config-audit ROOT    report which targets have math enabled
  check_math.py --collision-scan ROOT  find prose that math would swallow if enabled

Exits non-zero when anything is reported, so it gates.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------------
# Masking. Every rule below cares only about prose and math, never about code. Code spans and
# fenced blocks are replaced with a filler character rather than removed so that line and column
# numbers still point at the real source location.
# --------------------------------------------------------------------------------------------

FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})")


def mask_code(text: str) -> str:
    """Blank out fenced blocks and inline code spans, preserving offsets and line structure."""
    lines = text.split("\n")
    out, fence_marker = [], None
    for line in lines:
        m = FENCE.match(line)
        if fence_marker is None and m:
            fence_marker = m.group(2)[0] * 3
            out.append(" " * len(line))
            continue
        if fence_marker is not None:
            out.append(" " * len(line))
            if line.strip().startswith(fence_marker):
                fence_marker = None
            continue
        out.append(line)
    masked = "\n".join(out)
    # Inline code spans, longest runs of backticks first so ``x`` is handled before `x`.
    for n in (3, 2, 1):
        tick = "`" * n
        pat = re.compile(re.escape(tick) + r"(?!`)(.+?)" + re.escape(tick), re.S)
        masked = pat.sub(lambda m: " " * len(m.group(0)), masked)
    return masked


def find_math_spans(masked: str):
    """Yield (start, end, kind, body) for display and inline math in masked text."""
    spans = []
    for m in re.finditer(r"\$\$(.+?)\$\$", masked, re.S):
        spans.append((m.start(), m.end(), "display", m.group(1)))
    taken = [(s, e) for s, e, _, _ in spans]
    for m in re.finditer(r"(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)", masked):
        if any(s <= m.start() < e for s, e in taken):
            continue
        spans.append((m.start(), m.end(), "inline", m.group(1)))
    return sorted(spans)


def line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


# --------------------------------------------------------------------------------------------
# Rules. Each carries the reason, because a finding a reader cannot act on is noise.
# --------------------------------------------------------------------------------------------


def _emphasis_pair(body: str) -> bool:
    """True if an emphasis-opening underscore precedes an emphasis-closing one.

    CommonMark decides what an underscore can do from the characters either side of it. A '_' with
    punctuation before and an alphanumeric after (as in '}_i') can OPEN emphasis; one with an
    alphanumeric before and punctuation after (as in 'W_{') can CLOSE it; one with punctuation on
    both sides (as in '}_{') can do either. An ordinary intraword subscript like 'x_i' can do
    neither, which is why most maths is unaffected.

    Only a matched pair produces an <em>, so a lone opener is harmless - measured, two '}_i' groups
    with no closer render correctly, while a single '}_i' followed by one 'W_{' does not.
    """
    def punct(c: str) -> bool:
        return c != "" and not c.isalnum() and not c.isspace()

    opens, closes = [], []
    for m in re.finditer(r"(?<!\\)_", body):
        i = m.start()
        prev, nxt = (body[i - 1] if i else ""), (body[i + 1] if i + 1 < len(body) else "")
        left = nxt != "" and not nxt.isspace() and (not punct(nxt) or prev == "" or prev.isspace() or punct(prev))
        right = prev != "" and not prev.isspace() and (not punct(prev) or nxt == "" or nxt.isspace() or punct(nxt))
        if left and (not right or punct(prev)):
            opens.append(i)
        if right and (not left or punct(nxt)):
            closes.append(i)
    return any(c > o for o in opens for c in closes)


def lint(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    masked = mask_code(raw)
    findings = []

    def add(idx, code, msg):
        findings.append((line_of(raw, idx), code, msg))

    for start, end, kind, body in find_math_spans(masked):
        whole = masked[start:end]

        if kind == "display" and "\n" not in whole and re.search(r"(?<!\\)\\\\(?!\\)", body):
            add(start, "single-line-block",
                "display math with a '\\\\' line break written on one line; GitHub eats one "
                "backslash. Put the '$$' delimiters on their own lines.")

        if "\\\\\\\\" in body:
            add(start, "quad-backslash",
                "four backslashes in math; correct only for an unconfigured python-markdown, and "
                "wrong once math is properly enabled. Use two.")

        # GitHub strips a backslash before ASCII punctuation, treating it as a CommonMark escape.
        # LaTeX spacing commands are exactly that shape, so '\,' arrives as a stray comma in the
        # formula and '\%' becomes a comment that eats the rest of the line. MyST and MkDocs keep
        # them, so the two targets disagree about what the reader sees.
        for m2 in re.finditer(r"\\([,!;:%&#])", body):
            add(start, "escaped-punctuation-in-math",
                f"'\\{m2.group(1)}' loses its backslash on GitHub and arrives as a literal "
                f"'{m2.group(1)}'. Drop the spacing command, or use a form that needs no backslash.")
            break

        if kind == "display" and _emphasis_pair(body):
            add(start, "emphasis-pair-in-math",
                "an underscore that can OPEN markdown emphasis is followed by one that can CLOSE it "
                "(e.g. '}_i' then 'W_{'). On GitHub the pair becomes an <em>, which triggers a "
                "second escaping pass that double-escapes the alignment ampersands and collapses "
                "the columns. MyST and MkDocs render it fine, so this diverges silently. Write the "
                "subscript intraword ('\\hat x_i', 'L_{...}') so the underscore stays inert.")

        if re.search(r"[<>]", body):
            add(start, "raw-angle-bracket",
                "raw '<' or '>' inside math is HTML-escaped to an entity on some targets. Use "
                "\\lt / \\gt.")

        # The raw slice, not the masked copy: masking blanks a code span to spaces, so only the raw
        # text can distinguish $`x^2`$ from a genuinely loose $ x^2 $.
        raw_slice = raw[start:end]
        if kind == "inline" and re.fullmatch(r"\$`[^`\n]+`\$", raw_slice):
            add(start, "github-only-backticks",
                "the $`...`$ form is GitHub-only; MyST passes the backticks to the math engine and "
                "python-markdown makes a code span. Use plain $...$.")
            continue

        # GitHub refuses an inline span whose opening '$' is pressed against a word character or
        # hyphen, so natural compounds like 'top-$q$' render as plain text there while MyST renders
        # them as math. A TRAILING hyphen is fine; only the opening side matters.
        if kind == "inline":
            before = raw[start - 1] if start else " "
            if before not in " \t\n([{*_`\"'" and not before.isspace():
                add(start, "tight-open-delimiter",
                    f"inline math opens immediately after {before!r}; GitHub will not treat this as "
                    "math (MyST will), so it renders as literal text on one target only. Put a space "
                    "before the '$', or move the prefix inside the math.")

        if kind == "inline" and (body[:1].isspace() or body[-1:].isspace()):
            add(start, "loose-delimiter",
                "space between '$' and the math; some renderers refuse to match. Close it up.")

    for m in re.finditer(r"^\s*\\begin\{(align|equation|gather|multline)\*?\}", masked, re.M):
        # Bare top-level environments need the amsmath extension, which is off by default.
        preceding = masked[:m.start()]
        if preceding.count("$$") % 2 == 0:
            findings.append((line_of(raw, m.start()), "bare-amsmath",
                             f"bare \\begin{{{m.group(1)}}} at top level needs the 'amsmath' "
                             "extension. Nest 'aligned' inside a $$ block instead."))

    return findings


# --------------------------------------------------------------------------------------------
# Collision scan: what enabling math would break in prose that already exists.
# --------------------------------------------------------------------------------------------

# A shell variable is a '$' followed by a name. Single letters are excluded deliberately: '$x' is
# far more likely to be the start of an equation than a variable, and a scanner that cries wolf on
# real math is one people switch off.
SHELLISH = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]{1,}")


def mask_fences_only(text: str) -> str:
    """Blank fenced blocks but NOT inline code spans - backticks do not protect against this."""
    out, fence = [], None
    for line in text.split("\n"):
        m = FENCE.match(line)
        if fence is None and m:
            fence = m.group(2)[0] * 3
            out.append(" " * len(line)); continue
        if fence is not None:
            out.append(" " * len(line))
            if line.strip().startswith(fence):
                fence = None
            continue
        out.append(line)
    return "\n".join(out)


def collision_scan(root: Path, strict: bool = False):
    """Find prose that would be captured as math once '$' becomes a math delimiter.

    Measured rule: math opens at a '$' in PROSE and closes at the next '$' anywhere on the line -
    including one inside a code span, which is why backticks are not the protection they look like.
    A '$' that is itself inside a code span cannot open math, so a code span appearing first on the
    line is harmless. The hazard is therefore: a prose '$' with any later '$' on the same line.
    """
    findings = []
    paths = sorted(root.rglob("*.md")) if root.is_dir() else [root]
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="replace")
        unfenced = mask_fences_only(raw)          # '$' still visible inside inline code spans
        prose_only = mask_code(raw)               # '$' visible only where it can OPEN math
        for i, (all_line, prose_line) in enumerate(
            zip(unfenced.split("\n"), prose_only.split("\n")), 1
        ):
            openers = [m.start() for m in re.finditer(r"(?<!\\)\$", prose_line)]
            all_d = [m.start() for m in re.finditer(r"(?<!\\)\$", all_line)]
            if not openers or not all_d:
                continue
            if not any(q > p for p in openers for q in all_d):
                continue
            names = SHELLISH.findall(prose_line)
            if not names and not strict:
                # No variable-shaped name in prose: most likely intended math, not a collision.
                continue
            hint = ("shell variables " + ", ".join(names[:3])) if names else "literal '$' in prose"
            findings.append((path, i, hint, all_line.strip()[:90]))
    return findings


# --------------------------------------------------------------------------------------------
# Config audit: is math actually switched on for each target present in the repo?
# --------------------------------------------------------------------------------------------


def _strip_comments(text: str) -> str:
    """Remove '#' comments. Without this the audit reads a comment mentioning an extension as proof
    that it is enabled - failing toward false confidence, which is the worst direction for a tool
    whose whole job is to answer 'is math actually on?'."""
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in text.split("\n"))


def config_audit(root: Path):
    """Report per-target math configuration.

    'required' entries are what make math render at all; 'optional' entries buy extra syntax and
    their absence is not a fault. Conflating the two made this tool contradict the skill's own
    recipe, which prefers 'aligned' inside '$$' precisely so amsmath can stay off.
    """
    rows = []
    for conf in sorted(root.rglob("conf.py")):
        text = _strip_comments(conf.read_text(encoding="utf-8", errors="replace"))
        m = re.search(r"myst_enable_extensions\s*=\s*\[(.*?)\]", text, re.S)
        if m is None and "myst" not in text:
            continue
        enabled = re.findall(r"[\"']([a-z_]+)[\"']", m.group(1)) if m else []
        guards = [
            ("myst_dmath_allow_space", "False" in (re.search(
                r"myst_dmath_allow_space\s*=\s*(\w+)", text) or [None, ""])[1]),
            ("myst_dmath_allow_digits", "False" in (re.search(
                r"myst_dmath_allow_digits\s*=\s*(\w+)", text) or [None, ""])[1]),
        ]
        rows.append((conf, "sphinx-myst",
                     {"dollarmath": "dollarmath" in enabled},
                     {"amsmath (only for bare \\begin{align})": "amsmath" in enabled},
                     dict(guards),
                     "myst_enable_extensions in conf.py"))
    for conf in sorted(root.rglob("mkdocs.yml")):
        text = conf.read_text(encoding="utf-8", errors="replace")
        rows.append((conf, "mkdocs",
                     {"pymdownx.arithmatex": "arithmatex" in text,
                      "mathjax/katex asset": bool(re.search(r"mathjax|katex", text, re.I))},
                     {}, {},
                     "markdown_extensions + extra_javascript in mkdocs.yml"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--config-audit", metavar="ROOT", type=Path)
    ap.add_argument("--collision-scan", metavar="ROOT", type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="collision scan: also report pairings with no variable-shaped name")
    args = ap.parse_args()

    rc = 0

    if args.config_audit:
        rows = config_audit(args.config_audit)
        if not rows:
            print("no Sphinx/MyST or MkDocs config found; GitHub needs no configuration")
        for conf, kind, required, optional, guards, where in rows:
            print(f"{conf} [{kind}]")
            for name, ok in required.items():
                print(f"  {'ENABLED ' if ok else 'MISSING '} {name}   (required)")
            for name, ok in optional.items():
                print(f"  {'ENABLED ' if ok else 'off     '} {name}   (optional)")
            for name, ok in guards.items():
                print(f"  {'SET     ' if ok else 'unset   '} {name}   (recommended: reduces prose collisions)")
            if not all(required.values()):
                print(f"  -> math will NOT render; fix {where} (see references/config-recipes.md)")
                rc = 1
            elif not all(guards.values()) and guards:
                print("  -> math renders. Consider the dmath guards, then run --collision-scan.")
        return rc

    if args.collision_scan:
        hits = collision_scan(args.collision_scan, strict=args.strict)
        for path, ln, hint, text in hits:
            print(f"{path}:{ln}: two unescaped '$' would pair as math once enabled ({hint})")
            print(f"    {text}")
        if hits:
            print(f"\n{len(hits)} line(s) at risk. Put EVERY '$' on the line in a code span, or\n  escape as \\$. A partial wrap is worse than none: a prose '$' closes inside the\n  code span. Consider myst_dmath_allow_space=False as well.")
            rc = 1
        else:
            print("no prose dollar-sign collisions found")
        return rc

    if not args.paths:
        ap.error("give files to lint, or use --config-audit / --collision-scan")

    files = []
    for p in args.paths:
        files.extend(sorted(p.rglob("*.md")) if p.is_dir() else [p])
    total = 0
    for path in files:
        for ln, code, msg in lint(path):
            print(f"{path}:{ln}: [{code}] {msg}")
            total += 1
    if total:
        print(f"\n{total} finding(s).")
        rc = 1
    else:
        print(f"{len(files)} file(s) checked, no portability findings")
    return rc


if __name__ == "__main__":
    sys.exit(main())
