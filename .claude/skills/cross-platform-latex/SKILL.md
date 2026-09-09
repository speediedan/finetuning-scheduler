---
name: cross-platform-latex
description: Write LaTeX math in Markdown that renders correctly on GitHub, Sphinx/MyST (Read the Docs), MkDocs Material, and Jupyter notebooks at the same time. Use this skill whenever you are about to write, edit, or review mathematical notation in any Markdown file, README, docs page, docstring, issue, or PR body - equations, matrices, summations, expectations, gradients, loss functions, norms, or Greek symbols - and whenever math renders as literal dollar signs, line breaks collapse into one line, or an equation looks right on GitHub but broken in the docs build. Also use it before enabling math in a docs config, because turning math on can silently corrupt existing prose.
---

# Cross-platform LaTeX in Markdown

Math in Markdown fails in two unrelated places, and almost every piece of folklore on the subject
confuses them:

1. **The math engine never sees the equation.** The Markdown renderer was never configured to treat
   `$` as a math delimiter, so it emits literal dollar signs. No amount of escaping fixes this.
1. **The Markdown parser mutates the equation before the math engine sees it.** Backslashes get
   consumed, `<` becomes `&lt;`, underscores become emphasis. This is where escaping hacks come from.

The two need opposite responses, which is why advice that works in one setting breaks another. The
common counsel "use four backslashes for line breaks" is a real fix for failure 2 in *one*
configuration - and it corrupts the equation in every configuration where that parser is set up
correctly. Diagnose which failure you have before reaching for a remedy.

## Step 1: find out what the target actually is

Do this before writing math, not after it renders wrong. The same `$x^2$` is math on one target and
literal text on another, and the difference is a config file, not the syntax.

Run the checker's config audit, which reads the repo's docs config and reports what is enabled:

```bash
python scripts/check_math.py --config-audit <repo-root>
```

What each target needs:

| Target                                            | Needs                                          | Without it                              |
| ------------------------------------------------- | ---------------------------------------------- | --------------------------------------- |
| GitHub (README, issues, PRs, `.md` on github.com) | nothing, native                                | n/a                                     |
| Sphinx + MyST (Read the Docs)                     | `"dollarmath"` in `myst_enable_extensions`     | `$x^2$` renders as literal `$x^2$`      |
| Sphinx + MyST, bare `\begin{align}`               | `"amsmath"` as well                            | the environment renders as literal text |
| MkDocs Material                                   | `pymdownx.arithmatex` **and** a MathJax config | literal text, and backslashes are eaten |
| Jupyter markdown cells via MyST-NB                | same as Sphinx + MyST                          | same as Sphinx + MyST                   |

Two traps worth naming, because both look like the syntax is wrong when the config is:

- **MyST enables no extensions by default.** Its `enable_extensions` default is an empty set, so a
  Sphinx site that renders prose beautifully can still have math switched off entirely. A config that
  lists `colon_fence` and `deflist` but not `dollarmath` is the common shape.
- **Loading MathJax is not enabling math.** MkDocs sites often load MathJax via `extra_javascript`
  and stop there. MathJax 3's stock configuration does not treat `$...$` as inline math, and without
  `pymdownx.arithmatex` python-markdown mangles the content first anyway. Both halves are required.

`references/config-recipes.md` has the exact stanza to add for each target.

## Step 2: write the form that survives everywhere

These rules are not stylistic. Each one is the resolution of a specific disagreement between
renderers, and the reason is given so you can tell when it stops applying.

### Display math: put `$$` on its own lines

```markdown
$$
\begin{aligned}
a &= b \\
c &= d
\end{aligned}
$$
```

This single structural choice is what makes `\\` line breaks work. Written on one line as
`$$\begin{aligned}a &= b \\ c &= d\end{aligned}$$`, GitHub's parser consumes one of the two
backslashes and the line break silently disappears; given its own lines, the same content passes
through untouched. Keep a blank line before and after the block.

### Use exactly two backslashes for a line break

Never four. Four is the advice you will find most often, and it is a fix for an unconfigured
python-markdown, which eats one backslash from a `$$` block. But that configuration cannot render
math at all, so it is a bug to repair in the config rather than a syntax to adopt. Once each target
is set up as Step 1 describes, four backslashes is wrong everywhere: GitHub turns `\\\\` into `\\\`,
and MyST and arithmatex pass all four through to a renderer that does not want them.

### Inline math: `$` tight against the content

Write `$x^2$`, not `$ x^2 $`. Renderers differ on whether a space after the opening delimiter voids
the match, so the tight form is the one they agree on. Do not worry about subscripts: `$a_1 + b_2$`
is math on every configured target, and the underscores do not become emphasis, because the math span
is claimed before emphasis is processed.

### Prefer `\lt` and `\gt` to `<` and `>`

`$x \lt y$`, not `$x < y$`. GitHub encodes a raw `<` inside math twice, so what reaches the math
engine is the four-character string `&lt;` rather than the operator. MyST and MkDocs pass it through
intact, which makes this a GitHub-specific corruption - and one that spacing around the bracket does
not prevent, contrary to common advice. The macros cost nothing and are correct on every target, so
there is no reason to track which one you are writing for.

### Do not put a backslash before ASCII punctuation

`\,` `\!` `\;` `\:` for spacing, and `\%` `\&` `\#` for literals, all lose their backslash on
GitHub, which reads them as CommonMark escapes. The result is not a missing space but a **wrong
formula**: `\bar J_\ell\, h` arrives as `\bar J_\ell, h` and renders a stray comma, and `\%`
becomes a bare `%`, which starts a LaTeX comment that swallows the rest of the line. MyST and MkDocs
preserve all of them, so a formula can be right in the docs build and quietly wrong on GitHub.

Write the spacing out of the expression instead - juxtaposition is almost always enough - and where a
literal percent or ampersand is genuinely needed, prefer a form that avoids the backslash.

### Do not press an inline `$` against the preceding character

Write `the top $q$ of the coordinates`, not `the top-$q$ of the coordinates`. GitHub refuses an inline
span whose opening `$` is pressed against a word character or hyphen, so the expression renders as
plain text there while MyST renders it as maths. A *trailing* hyphen is fine, which is what makes this
easy to miss: `$\gamma$-marked` works and `permuted-$\gamma$` does not.

Hyphenated compounds are the usual casualty, and they are exactly what technical prose reaches for -
`top-$q$`, `token-$c$`, `layer-$\ell$`. Either put a space before the delimiter, or move the prefix
inside the maths.

### Keep emphasis-capable underscores out of display math

Most subscripts are inert. `$a_1 + b_2$` and `x_i` are math on every target and never become
emphasis, because an underscore surrounded by alphanumerics can neither open nor close it.

The ones that bite have punctuation on one side. Markdown decides what an underscore can do from its
neighbours: `}_i` (punctuation before, letter after) can **open** emphasis, `W_{` (letter before,
punctuation after) can **close** it, and `}_{` can do either. A block is corrupted only when an
opener is followed by a closer - GitHub turns the pair into an `<em>`, which triggers a second
escaping pass that double-escapes the alignment ampersands and collapses the `aligned` columns.

That pairing rule explains cases that otherwise look arbitrary, all measured:

| block contains             | result on GitHub                                |
| -------------------------- | ----------------------------------------------- |
| one `\hat{x}_i`            | fine - an opener with nothing to close it       |
| two `\hat{x}_i`            | fine - two openers, still no closer             |
| `\hat{x}_i` then `W_{enc}` | **broken** - opener, then closer                |
| one `\underbrace{a}_{r}`   | fine                                            |
| two `\underbrace{a}_{r}`   | **broken** - `}_{` is both, so two of them pair |

MyST and MkDocs render every one of these correctly, so this diverges silently: the docs build looks
right while the README is mangled. The fix is to keep the underscore intraword - `\hat x_i` instead
of `\hat{x}_i`, `L_{\mathrm{recon}}` instead of `\mathcal{L}_{\mathrm{recon}}` - which is
identical TeX. Where a block genuinely needs term labels, an extra alignment column carrying
`\text{...}` works everywhere and introduces no underscores at all. `check_math.py` implements this
flanking rule, but a block that is dense with braces is worth rendering once against GitHub directly.

### Keep environments inside `$$`

Write `\begin{aligned}...\end{aligned}` nested inside a `$$` block rather than relying on a bare
`\begin{align}` at the top level. The bare form needs a separate extension that is off by default,
and `aligned` inside `$$` needs nothing beyond what Step 1 already required.

### Do not use the backtick-dollar form outside GitHub-only files

GitHub accepts `` $`x^2`$ `` as inline math. Nothing else does: MyST passes the backticks through to
the math engine as literal characters, and python-markdown turns them into a code span sitting
between two dollar signs. It is the least portable construct in common use, and its usual
justification - that it protects subscripts from becoming italics - describes a problem that does not
occur. Reach for it only in a file that will never be rendered anywhere but github.com, and say so in
a comment when you do.

## Step 3: check before publishing

```bash
python scripts/check_math.py <path>...
```

The checker flags the constructs above, and deliberately does not flag shell variables or currency -
see Hazards. Run it on the files you touched; it exits non-zero on findings so it works in a hook or
in CI.

## Hazards when turning math on in an existing repo

Enabling `dollarmath` is not a safe no-op for pages that already exist. A dollar sign in ordinary
prose becomes a math delimiter the moment the extension is on, and the text after it disappears into
an equation. On MyST, a line reading `Set $HOME and $PATH now.` renders as *Set* followed by a math
span containing `HOME and`, followed by `PATH now.` - the prose is swallowed with no error and no
warning anywhere in the build log. It cannot be caught by reading build output; only by looking at
the rendered page.

The precise rule, which is not the one most people assume: **math opens at a `$` in prose and closes
at the next `$` anywhere on that line, including one inside a code span.** A `$` that is itself
inside a code span cannot open math, so a code span appearing earlier on the line is harmless.

Engines differ in how much of this they will match, and the difference is easy to mis-generalise from
one example. MyST captures both a spaced pair (`$HOME and $PATH`) and an adjacent one
(`$TMPDIR/$SLURM_JOB_ID`). MkDocs declines the spaced pair but **captures the adjacent one**. GitHub
declines both. So this is not a MyST-only hazard: a page that survives one engine can still be
corrupted by another, and the adjacent form is the one that travels furthest.

That has a consequence worth stating on its own, because the obvious remedy gets it backwards:

> **Backticks protect only if every `$` on the line is inside one.** Wrapping some but not all is
> worse than wrapping none. `` Set $HOME then run `echo $PATH`. `` is corrupted precisely *because*
> the second variable is in a code span - the prose `$` opens a span that closes inside the backticks,
> and the code formatting is destroyed along with the words.

Two defences, and they are complementary rather than alternatives:

**Harden the parser.** MyST can refuse the matches that cause this:

```python
myst_dmath_allow_space = False    # '$ x $' no longer matches, so '$HOME and $PATH' is safe
myst_dmath_allow_digits = False   # '2$x$' no longer matches
```

This fixes every case where the paired text has a space against a delimiter, which is most prose. It
does **not** fix an adjacent pair like `$TMPDIR/$SLURM_JOB_ID`, where nothing separates the names, so
it reduces the blast radius rather than closing the hole. The cost is two authoring styles you lose:
`$ E = mc^2 $` and `2$x$`. Both are ones this skill tells you not to write anyway.

**Fix the source.** Scan before you flip the switch:

```bash
python scripts/check_math.py --collision-scan <docs-root>
```

Put *every* dollar sign on a reported line inside a code span, or escape it as `\$`. Fenced blocks
are immune throughout, so only prose and inline code need attention.

This is a pre-enablement instrument. Once a page legitimately contains math, its own equations look
identical to the hazard, and the scan will report them. Run it before the change, act on what it
finds, and treat later runs on math-bearing pages as expected noise rather than a regression.

## The Sphinx-only escape hatch, and why it is a last resort

MyST offers `` {math}`x^2` `` and a ```` ```{math} ```` directive, which need no configuration and
cannot collide with prose dollar signs. On a page that will only ever be a Sphinx build, that is a
genuinely safe option, and it is the right answer when a document is so dense with shell variables
that hardening it would mean rewriting most of its prose.

Measured cost: on GitHub the role renders as the literal text `{math}` followed by a code span, and
the directive renders as a plain code block. So it is not a portable form - it is a Sphinx form that
happens to be immune to the collision. Reach for it only when a page is docs-only *and* the prose
genuinely resists hardening. Everywhere else, prefer `$` syntax and fix the collisions: one syntax
that renders everywhere is worth more than a per-target dialect, and the moment a `{math}` page gets
excerpted into a README the notation is silently broken.

## When you genuinely need per-platform output

Prefer one portable source. If a document truly must render differently, keep the divergence in the
build rather than the prose: MyST substitutions or an include, not two copies of an equation that
will drift. If you must write a platform-specific construct inline, leave a comment saying which
target it is for and what breaks elsewhere, so the next person does not "fix" it by making it worse.

## Keeping this skill honest

The compatibility claims here are measurements, not recollections, and they have an expiry: they
describe the renderer versions installed when they were taken. A parser release can change any of
them. When a claim looks wrong, or after a major version bump of any renderer, re-measure rather than
re-reason:

```bash
python scripts/probe_renderers.py --report
```

It renders a fixed corpus of the constructs above through whichever engines are importable, prints
the resulting matrix with the versions it used, and marks any cell that disagrees with the recorded
table in `references/platform-matrix.md`. Update that file from the output and note the versions. A
disagreement is information, not a failure - it means a renderer moved and the guidance above needs a
line changed.

## Reference material

- `references/platform-matrix.md` - the measured construct-by-renderer table, with the versions each
  row was taken against. Read it when you need to know whether one specific construct is safe.
- `references/config-recipes.md` - exact config stanzas to enable math per target, and how to verify
  the change took effect.
