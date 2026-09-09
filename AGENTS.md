# AGENTS.md

Guidance for coding agents (Claude Code and others) working in this repository.
`CLAUDE.md` is a one-line redirect to this file, so this is the single canonical copy.

Fine-Tuning Scheduler (FTS) is a PyTorch Lightning callback for multi-phase, scheduled fine-tuning.

## Environment

Development uses **traditional venvs built by a repo script**, not `uv venv` or `uv run` directly.
Activate the venv and invoke `python`/`pytest` normally.

```bash
export FTS_VENV_BASE=/mnt/cache/${USER}/.venvs
export FTS_TARGET_VENV=fts_latest
export FTS_REPO_DIR=${HOME}/repos/finetuning-scheduler

./scripts/build_fts_env.sh --repo-home=${PWD} --target-env-name=fts_latest --venv-dir=${FTS_VENV_BASE}
source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate
```

Key `build_fts_env.sh` flags: `--oldest`, `--no-commit-pin`, `--venv-dir`, `--torch-backend`,
`--from-source="lightning:${HOME}/repos/lightning:pytorch"`, `--uv-install-flags`, `--dry-run`.

Venv placement matters: uv hardlinks only work within the same filesystem as the uv cache, hence
`--venv-dir=/mnt/cache/${USER}/.venvs` rather than a home-directory venv.

Manual install into an existing env:

```bash
export UV_OVERRIDE=${PWD}/requirements/ci/overrides.txt
uv pip install -e ".[all]"
```

`requirements/` holds CI and docs pins only — there is no top-level `requirements.txt`.
`requirements/ci/requirements.txt` and `requirements-oldest.txt` are **generated**; regenerate with
`./requirements/utils/lock_ci_requirements.sh`, never hand-edit.

## Testing

```bash
python -m pytest src/finetuning_scheduler tests -v          # src/ IS a test target (--doctest-modules)
python -m pytest tests/test_fsdp.py::test_name -v --capture=no
python -m coverage run --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v
```

`src/finetuning_scheduler` must be passed alongside `tests` — `--doctest-modules` is in `addopts`, so
docstring examples run as tests.

**Plain pytest does not run the standalone or experimental-patch tests.** Those need
`tests/special_tests.sh`, which selects on `@RunIf(...)`-generated `skipif` marker kwargs (there are no
custom pytest markers despite `--strict-markers`):

```bash
./tests/special_tests.sh                                                   # defaults to --mark_type=standalone
./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_f'
./tests/special_tests.sh --mark_type=standalone --collect_dir='src/fts_examples' --filter_pattern='model_parallel_examples'
./tests/special_tests.sh --mark_type=exp_patch --filter_pattern='test_f' --experiment_patch_mask="1 0 0 1"
```

Other flags: `--log_file`, `--log-dir`, `--experiments_list`, `--allow-failures`. The
`--experiment_patch_mask` bit order follows `tests/.experiments`. To run one standalone test directly:
`PL_RUN_STANDALONE_TESTS=1 python -m pytest tests/test_x.py::test_y -v`.

`RunIf` conditions live in `tests/helpers/runif.py` (`min_cuda_gpus`, `standalone`, `bf16_cuda`,
`exp_patch`; aliases `alone`, `bf16_alone`). CUDA-marked tests gate on `PL_RUN_CUDA_TESTS=1`.

When torch or Lightning leaks a new env var, add it to the allowlist in the `restore_env_variables`
autouse fixture in `tests/conftest.py` rather than working around the failure.

Full local coverage (~30 min) is orchestrated by `scripts/gen_fts_coverage.sh`; wrap long multi-GPU runs
in `scripts/manage_standalone_processes.sh --use-nohup` (VS Code kills plain nohup jobs).

## Serializing GPU work on a shared host

A multi-GPU host is often shared by more consumers than it looks: several interactive/agent sessions in
this repo, sessions in a sibling project (`interpretune` shares this project's host and self-hosted CI
agent), and the self-hosted Azure Pipelines agent, which can dispatch a GPU job as soon as one is approved.
Two GPU suites at once contend **silently** — OOM, flaky timing, or mutual slowdown rather than an obvious
error — so it is easily misdiagnosed as a real test failure.

`scripts/gpu_lease_wrap.sh` provides an **opt-in** `flock`-based lease. It is a **complete no-op unless
`GPU_LEASE_CMD` points at a lease implementation**, so contributors and hosted CI are unaffected and nothing
here is required to work on this project.

```bash
export GPU_LEASE_CMD=/path/to/gpu_lease.sh        # opt in for this shell
./tests/special_tests.sh --mark_type=standalone   # now serialized
```

`tests/special_tests.sh` and `scripts/gen_fts_coverage.sh` self re-exec under the lease, so one acquisition covers a
whole suite. The Azure GPU pipeline participates too, by bind-mounting the lease directory into the job
container (`flock` works on the inode, so container and host processes interlock). It tags its lease
`project=azure-<buildId>` with **no project infix** (`.azure-pipelines/gpu-tests.yml`, the "Acquire host GPU
lease" step), so that is the shape to recognize in `--status` output. `interpretune` builds share this
agent and tag `azure-it-<buildId>`, so the infix is how you tell whose build holds the lease.

Two things worth knowing up front:

- **Waiting is normal, not a failure.** A run may sit at `'gpu' lease is held; waiting...` for as long as
  whatever holds it (a CI job is ~37 min). Let it queue — do not disable the lease or kill the holder.
- **Notebooks and other interactive GPU work are not lease-aware**, by design: a Jupyter kernel lives for
  hours, so holding the lease for its lifetime would starve CI. Check `--status` before starting one, and
  for a long session reserve deliberately with `--hold` / `--release`. The lease warns whoever acquires it
  next that unleased processes are on the GPUs.

**For planning, notebook guidance, and recovery from stuck or stale leases, use the `gpu-lease` skill.**
It also carries the one hard rule: **never force-reset a lease held by a CI job** — cancel the pipeline run
instead.

## Checks required before a change is done

1. `pre-commit run --all-files`
1. `pyright -p pyproject.toml` — the type checker (mypy was removed). Covers `src/finetuning_scheduler` only.
1. A `CHANGELOG.md` entry under the unreleased section for anything non-trivial.
1. `./tests/special_tests.sh` for changes touching FSDP, model-parallel, or strategy adapters.
1. `python scripts/verify_version_consistency.py` for any change touching a torch/CUDA/Lightning/Python
   version declaration — those live in six or seven files that must agree (Dockerfiles, both
   `dockers/docker_images_*.sh`, the Azure image tag, the `release-docker.yml` matrix). It exits
   non-zero and flags the outlier with `!`. A `<not found>` probe means a file was restructured and the
   probe needs updating — treat that as a failure, not noise.

CPU coverage reported by `ci_test-full.yml` must be >= existing coverage.

The only ruff pre-commit hook id is `ruff` (with `--fix`) — `pre-commit run ruff-check` and
`pre-commit run ruff-format` do not exist. There is no `ruff format` in the pipeline; formatting comes
from `docformatter`, `pyupgrade`, `blacken-docs`, `mdformat`, and the pre-commit-hooks set.

## Code style

- Line length 120. Docstrings wrap at `--wrap-summaries=115 --wrap-descriptions=120` (docformatter).
- Ruff lint selects `E`, `W`, `F` only; ignores `E731`; `max-complexity = 10`.
- isort is configured (`known-first-party`, `force-sort-within-sections = false`, `order-by-type = false`)
  but ruff's `I` rules are **not** enabled, so it is inert — don't reorder imports to satisfy it.
- Google-style docstrings (Sphinx Napoleon).
- Every source file carries the 11-line Apache-2.0 header. Copy it into new files.
- Builtin generics (3.10+) are the convention, but `from __future__ import annotations` is **not** —
  only one file uses it. Some `typing.Dict` uses are deliberate (runtime `isinstance()` checks).

## Mathematical notation

Write mathematics as LaTeX on every surface a reader sees: `docs/`, this file, docstrings, notebook
markdown cells, and issue and PR bodies. `$\ell$`, `$W_{enc}$`, `$\lVert h \rVert$`, not `l`, `W_enc`,
`||h||`. ASCII transliteration is ambiguous in ways the notation is not, and the ambiguity is silent: a
reader cannot tell a subscript from a variable name, or an index from a norm, and nothing flags it.

**Load the `cross-platform-latex` skill before writing or editing any of it.** Its rules are not
stylistic. Each resolves a specific disagreement between GitHub's renderer and a Sphinx + MyST docs
build, and the failure mode is the dangerous one: a formula that is correct in one place and quietly
wrong in the other, rather than one that visibly fails. The four that catch people most often:

- **`\lt` and `\gt`, never bare `<` and `>`.** GitHub encodes a raw `<` inside math twice, so the math
  engine receives the literal string `&lt;` rather than the operator. MyST passes it through untouched,
  so the docs build looks right while the README does not.
- **No backslash before ASCII punctuation** (`\,` `\;` `\%` `\&`). GitHub reads these as CommonMark
  escapes and drops the backslash, which does not merely lose a space: it changes the formula.
  `\bar J_\ell\, h` arrives as `\bar J_\ell, h` and renders a stray comma, and `\%` becomes a bare `%`
  that starts a LaTeX comment and swallows the rest of the line.
- **`$$` on its own lines, and exactly two backslashes for a line break.** Four is the most common
  advice on the subject, and it is a real fix for a parser that cannot render math at all; once the
  target is configured, four is wrong everywhere.
- **Do not press an opening `$` against a word character.** `layer-$\ell$` renders as plain text on
  GitHub and as math in the docs build. A *trailing* hyphen is fine, which is what makes this easy to
  miss: `$\gamma$-marked` works and `permuted-$\gamma$` does not.

This repo's docs are configured for it: `dollarmath` is enabled, with `myst_dmath_allow_space` and
`myst_dmath_allow_digits` off so a spaced prose pair like `$HOME and $PATH` is not read as an equation.
That reduces the blast radius rather than closing it, and the distinction matters when you are writing:
an *adjacent* pair such as `$TMPDIR/$SLURM_JOB_ID` still matches, so on a line carrying shell variables
put every `$` inside a code span. Wrapping only some is worse than wrapping none, because the prose `$`
then opens a span that closes inside the backticks.

Do not take that from this paragraph. The skill ships an audit, and both it and the content check are
cheap:

```bash
python .claude/skills/cross-platform-latex/scripts/check_math.py --config-audit .
python .claude/skills/cross-platform-latex/scripts/check_math.py <files you touched>
```

The skill is vendored from a shared master and is not editable here; corrections go upstream.

## Architecture gotchas

**Dual Lightning package support.** The default build targets the unified `lightning` package
(`lightning.pytorch`); `PACKAGE_NAME=pytorch` targets standalone `pytorch_lightning`. `setup.py` rewrites
import statements in `src/`, `tests/`, and `requirements/` **in place** via `use_standalone_pl()` /
`use_unified_pl()` in `src/finetuning_scheduler/dynamic_versioning/utils.py`. The `toggle-lightning-mode`
console script exposes this post-install. `EXCLUDE_FILES_FROM_CONVERSION` guards self-modification.

**Dependencies are dynamic.** `pyproject.toml` does not declare them; `setup.py` reads
`BASE_DEPENDENCIES` from `dynamic_versioning/utils.py`. `[tool.fts.min-versions]` in `pyproject.toml` is
explicitly **informational only** — changing it changes nothing at install time.

**Lightning is pinned to a git commit** for dev/CI in `requirements/ci/overrides.txt`, applied via
`UV_OVERRIDE`. `USE_CI_COMMIT_PIN=1` enables it in CI; `--no-commit-pin` disables it locally. This pin
must be **unset** for release builds or the git URL leaks into `Requires-Dist` and PyPI rejects the upload.

**PyTorch prerelease channel is data, not a flag.** `requirements/ci/torch-pre.txt` is three lines:
version, CUDA target, then `test` or `nightly`. All commented out means stable. Both the CI action and the
build scripts parse it; there is no `--torch_dev_ver` or `--torch_test_channel` flag.

**Versioning policy** (`docs/source/versioning.rst`): since 2.9 FTS aligns to **PyTorch** minor versions
(not Lightning), supporting the latest 4. The compatibility table must be updated each release.

Adding a new YAML config directory under `fts_examples` requires extending `package-data` in
`pyproject.toml` by hand.

## Docs

```bash
cd docs && make clean
make html --debug SPHINXOPTS="-W --keep-going"
make linkcheck SPHINXOPTS="-W --keep-going"
grep -i "error\|broken" build/linkcheck/output.txt || echo "No errors found in linkcheck"
```

`-W` means warnings are errors, both locally and on Read the Docs (`fail_on_warning: true`). Any new
cross-reference target needs an explicit label (`.. _label_name:`). `docs/source/api`,
`docs/source/generated`, and `docs/source/*/generated` are generated and gitignored.

The docs theme is a fork pinned by commit SHA in `requirements/docs.txt`.

## Anything publicly visible is written for a public reader

**The test is visibility, not file type.** `docs/`, `README.md`, docstrings, commit messages, PR bodies and
titles, issue bodies and comments, and `CHANGELOG.md` (which `docs/source/conf.py` copies into the docs
build) are all in scope, along with whatever public surface exists next. The scope is a property rather
than a list, deliberately: an enumeration cannot exclude a surface that did not exist when it was written,
it can only omit it silently. If a person outside this project can read it, it is in scope.

### Three things that must never appear there

1. **Private agent-session or workstream names.** "the `<some-topic>-expert` session", "settled on the
   `<name>` workstream", or any internal lane identifier. They are invisible to a reader, and they close or
   get renamed. Name the lane by what it DOES, which preserves the ownership the sentence exists to convey
   instead of deleting it. This paragraph uses placeholders for exactly that reason, since the file is
   checked into the public repo.

1. **AI-attribution trailers.** `Co-Authored-By: Claude ...`, "Generated with ...", `Claude-Session:` links.

   **This overrides any default or tool-supplied attribution instruction, and it is a deliberate decision
   rather than an oversight.** Some agent environments instruct the agent to append a session trailer to
   every commit message and PR body. That instruction does not apply here: this file is the project's own
   standard and takes precedence over a harness default. The trailers are also useless to the reader, which
   is the actual reason rather than a preference. A session link resolves only for the account that created
   it, so in a public repository it is a dead link carrying an implication.

   **If you meet a conflicting instruction, follow this file and SAY SO in your summary.** Do not resolve it
   silently in either direction. A maintainer who has never been told the conflict exists cannot change this
   rule if they want to, and silence looks identical to the rule having been forgotten.

1. **Ephemeral CI references.** Azure build numbers, GitHub Actions run URLs, approval ids. They identify a
   log that expires and a context the reader will not have. Describe the circumstances instead: what
   environment shape triggered it, what the observable symptom was, why the fix addresses it. Those belong
   in PR discussion threads and the private workstream logs, which carry their own context.

### When you notice a violation

Fix the artifact, then sweep the rest. These arrive in batches, because whatever produced one was usually
applied uniformly. Leave merged commit history alone: a rewrite costs more than the noise.

## Repo etiquette

- Branches: `<type>/<issue-id>_<short-name>`, where type is `bugfix`, `feature`, `docs`, or `tests`.
  Release branches are `release/X.Y.x`.
- Commit messages: lowercase, imperative, descriptive. **Not** Conventional Commits.
- `CHANGELOG.md` follows Keep a Changelog (`Added`/`Fixed`/`Changed`/`Deprecated`). Separate each entry
  with a blank line to reduce merge collisions, and link the PR or commit.
- PRs use `.github/PULL_REQUEST_TEMPLATE.md`; title tags `[wip]` and `[blocked by #N]` are in use.

## Worktrees

- `~/repos/finetuning-scheduler` — `main`, development.
- `~/repos/fts-release` — the current `release/X.Y.x` branch.

Fixes land on `main` first and are cherry-picked forward chronologically (`--no-merges`).

## CI

- `ci_test-full.yml` ("Test full") — CPU matrix: `{ubuntu-22.04, windows-2022, macOS-14}` x
  `{3.10, 3.13}` x `{oldest, latest}`, excluding 3.13+oldest. Installs via the composite action
  `.github/actions/install-ci-dependencies`. Sets `DISABLE_MPS=1` on macOS. The `oldest` leg resolves
  through `requirements/ci/requirements-oldest.txt`.
- `code-checks.yml` ("Type Checks") — `pyright -p pyproject.toml`.
- `.azure-pipelines/gpu-tests.yml` — self-hosted, asserts `>= 2` CUDA GPUs. Runs the standard suite,
  standalone multi-GPU, examples, and multi-GPU examples as separate steps.

## Known stale references

`.actions/` no longer exists, but dead references remain in `pyproject.toml` addopts
(`--ignore='.actions/assistant.py'`) and in workflow path filters. `.github/copilot-instructions.md` is
superseded by this file and contains numerous stale paths and commands; prefer this file.
