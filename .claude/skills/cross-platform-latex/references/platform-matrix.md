# Measured construct-by-renderer matrix

Every cell here was produced by rendering the construct and reading what reached the math engine, not
by reasoning about parser documentation. Regenerate it with:

```bash
python scripts/probe_renderers.py --report
```

**Measured 2026-09-07** against myst-parser 5.1.0 (`dollarmath` + `amsmath`), python-markdown 3.10.3
with pymdown-extensions 11.0.2 (`arithmatex`), and the live github.com Markdown API. Each engine is
configured as `config-recipes.md` prescribes; an unconfigured engine renders nearly everything as
literal text and is not interesting to tabulate.

## Reading the table

The three verdicts describe **what the renderer did**, which is not the same question as what you
should write:

- **verbatim** - the math engine received exactly what the author typed.
- **altered** - the engine received something different (a consumed backslash, a double-encoded
  entity, a stray backtick), or math was produced where the author wrote prose.
- **no-math** - no math span was produced; the text renders literally.

`verbatim` is fidelity, not endorsement. Arithmatex reproduces four backslashes perfectly, and four
backslashes is still broken LaTeX. Likewise `no-math` is the *correct* outcome for the shell-variable
row. Use SKILL.md for what to write; use this table to understand why.

| construct                  | MyST     | MkDocs   | GitHub   |
| -------------------------- | -------- | -------- | -------- |
| `$x^2$`                    | verbatim | verbatim | verbatim |
| `$a_1 + b_2$`              | verbatim | verbatim | verbatim |
| `$ x^2 $` (loose)          | verbatim | no-math  | no-math  |
| `` $`x^2`$ ``              | altered  | no-math  | verbatim |
| `$x<y$`                    | verbatim | verbatim | altered  |
| `$x \lt y$`                | verbatim | verbatim | verbatim |
| `$$…\\…$$` on one line     | verbatim | verbatim | altered  |
| `$$` own lines, `\\`       | verbatim | verbatim | verbatim |
| `$$` own lines, `\\\\`     | verbatim | verbatim | altered  |
| bare `\begin{align}`       | verbatim | verbatim | no-math  |
| `Set $HOME and $PATH now.` | altered  | no-math  | no-math  |

## What the table implies

Exactly four constructs are safe everywhere: tight inline `$...$`, subscripts inside it, `\lt` / `\gt`
in place of angle brackets, and a `$$` block whose delimiters sit on their own lines using two
backslashes for line breaks. That set is the whole of the portable core in SKILL.md, and it is small
on purpose.

Three rows deserve a note, because each is the origin of a piece of common advice:

- **`$$…\\…$$` on one line, altered on GitHub.** GitHub consumes one of the two backslashes, so the
  line break vanishes and `\c` becomes an undefined control sequence. Giving the delimiters their own
  lines fixes it with no escaping.
- **Four backslashes, altered on GitHub.** GitHub turns `\\\\` into `\\\`; MyST and arithmatex hand
  all four to the engine unchanged. The advice to write four is a fix for an *unconfigured*
  python-markdown, which eats one - a state in which math does not render at all. It is a config bug
  to repair, not a syntax to adopt.
- **Emphasis-capable underscores, altered on GitHub only.** Markdown decides what an underscore can
  do from its neighbours: `}_i` can open emphasis, `W_{` can close it, and `}_{` can do either. A
  block breaks when an opener is followed by a closer - the pair becomes an `<em>`, triggering a
  second escaping pass that double-escapes the alignment ampersands and collapses the columns. This
  is why one `\underbrace{a}_{r}` is fine and two are not, why two `\hat{x}_i` are fine (two openers,
  no closer), and why one `\hat{x}_i` followed by one `W_{enc}` is not. Intraword subscripts such as
  `x_i` can do neither and are always safe. MyST and MkDocs render every variant correctly, so this
  is invisible unless GitHub is checked directly.
- **`Set $HOME and $PATH now.`, altered on MyST.** The two dollar signs pair up and the prose between
  them becomes an equation. GitHub declines the match and MkDocs declines *this* one - but see the
  adjacent-pair row directly above, which MkDocs does capture. The hazard is therefore not
  MyST-specific; the spaced and adjacent forms have different reach, and only GitHub refuses both. Note the adjacent row: a code span does **not** make the line safe. Math opens at a prose `$`
  and closes at the next `$` on the line even when that one is inside backticks, so a partial wrap
  converts a two-variable prose line into a corrupted one. Only fenced blocks are immune outright.

Two further rows are GitHub-only losses, and both were found by converting a real document rather
than by probing constructs someone thought to test:

- **`\,` and friends lose the backslash on GitHub.** Every `\` before ASCII punctuation is read as a
  CommonMark escape, so LaTeX spacing commands arrive as literal punctuation inside the formula. This
  is a wrong formula rather than a missing space, and `\%` is worse still, since a bare `%` comments
  out the rest of the line.
- **An inline `$` pressed against the previous character is not math on GitHub.** `top-$q$` renders as
  text there and as math on MyST. A trailing hyphen is unaffected, so `$\gamma$-marked` is fine.

## Non-portable forms, and why they are still worth knowing

Two constructs render on exactly one target. Neither is a recommendation; both come up often enough
that a reader needs to know what they cost.

| form                    | GitHub | MyST                                      | MkDocs                    |
| ----------------------- | ------ | ----------------------------------------- | ------------------------- |
| `` $`x^2`$ ``           | math   | backticks reach the engine                | code span between two `$` |
| ```` ```math ```` fence | math   | code block, plus a Pygments lexer warning | code block                |

The fence has one measured property worth recording: because fenced content skips inline processing,
it is immune to the emphasis hazard above - two `\underbrace{...}_{...}` groups render correctly
inside a ```` ```math ```` fence and incorrectly inside `$$` on the same page. That makes it
tempting for a GitHub-only file. It also means a document using it cannot move into a docs build
without being rewritten, which is the trade the portable `$$` form exists to avoid.

## When a cell disagrees

A disagreement means a renderer moved, which is information rather than a fault. Re-run the probe,
update this table and the recorded baseline in `scripts/probe_renderers.py`, note the new versions,
and change whichever sentence in SKILL.md depended on the old behaviour.
