# Enabling math, per target

Each recipe is the smallest change that makes `$...$` and `$$...$$` render, plus how to confirm it
worked. Verify rather than assume: every one of these failures is silent, and a docs build emits no
warning when math is switched off.

## GitHub

Nothing to configure. Math is native in Markdown files, issues, pull request bodies and discussions.

## Sphinx with MyST (including Read the Docs)

MyST enables **no** syntax extensions by default: its `enable_extensions` default is an empty set.
Add `dollarmath`, and `amsmath` only if you need bare `\begin{align}` at the top level:

```python
# docs/source/conf.py
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",   # $...$ and $$...$$
    "amsmath",      # bare \begin{align} etc. at top level; unnecessary if you nest in $$
]
```

Nesting `aligned` inside `$$` needs only `dollarmath`, so prefer that and leave `amsmath` off.

If the repo has prose containing shell variables or prices, add the delimiter guards at the same
time. They cost two authoring styles this skill already tells you to avoid, and they remove most of
the collision surface:

```python
myst_dmath_allow_space = False    # '$ x $' stops matching, so '$HOME and $PATH' is left alone
myst_dmath_allow_digits = False   # '2$x$' stops matching
```

They are not a complete fix: an adjacent pair such as `$TMPDIR/$SLURM_JOB_ID` has no space against a
delimiter and still parses as math. Run the collision scan as well.

Sphinx renders the result with MathJax through `sphinx.ext.mathjax`, which is on by default. If the
theme strips it or the build is offline, math renders as plain text with no error.

Confirm:

```bash
python scripts/check_math.py --config-audit .
grep -r 'class="math' docs/_build/html | head    # after a build
```

### Notebooks via MyST-NB

Markdown cells go through the same parser with the same config, so the Sphinx recipe covers them.
There is nothing notebook-specific to enable. Note that a notebook's own preview in Jupyter uses a
different math engine than the docs build, so a cell that looks right in Jupyter can still be wrong
in the rendered docs - check the built page, not the notebook.

## MkDocs (Material and others)

Two halves, and loading MathJax alone is the common half-done state:

```yaml
# mkdocs.yml
markdown_extensions:
  - pymdownx.arithmatex:
      generic: true

extra_javascript:
  - javascripts/mathjax.js
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js
```

```javascript
// docs/javascripts/mathjax.js
window.MathJax = {
  tex: { inlineMath: [["$", "$"]], displayMath: [["$$", "$$"]] },
  options: { ignoreHtmlClass: ".*", processHtmlClass: "arithmatex" },
};
```

The JavaScript half is not optional. MathJax 3's stock configuration does **not** treat `$...$` as
inline math, so a site that loads the library and stops there renders inline math as literal text.
Without `arithmatex`, python-markdown also consumes one backslash from every `\\` in a `$$` block
before MathJax ever sees it - which is where the "use four backslashes" folklore comes from.

Confirm:

```bash
python scripts/check_math.py --config-audit .
grep -r 'class="arithmatex"' site/ | head        # after a build
```

## Before enabling math on an existing repo

Turning math on changes how existing pages render. Scan first:

```bash
python scripts/check_math.py --collision-scan docs/
```

See the Hazards section of SKILL.md for what to do with the findings.
