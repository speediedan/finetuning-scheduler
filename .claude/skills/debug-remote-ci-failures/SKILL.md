---
name: debug-remote-ci-failures
description: Systematic workflow for debugging finetuning-scheduler GitHub Actions failures across the Ubuntu/Windows/macOS x python x oldest/latest matrix — pulling logs and junit artifacts, categorizing failures, isolating root cause, and running the fix/push/verify loop. Use when "Test full" or "Type Checks" fails on a PR or push.
license: Apache-2.0
---

# Debugging FTS Remote CI Failures

Covers the GitHub Actions legs only. For the self-hosted Azure GPU pipeline, use the
`az-pipelines-debug` skill.

Adapted from the interpretune skill of the same name; the interpretune-specific failure taxonomy
(analysis fixtures, nnsight, HuggingFace dataset fingerprinting) has no FTS analog and was dropped rather
than ported. This version is deliberately thinner — extend it as real FTS incidents accumulate.

## Prerequisites

- `gh` authenticated. Note the current token lacks the `workflow` scope: `gh run list/view/download` work,
  but re-running or dispatching workflows via the API does not — use the web UI for that.
- Push access to the branch under test.

## The matrix

`ci_test-full.yml` ("Test full") is **four-dimensional**, which matters for reading job and artifact names:

```
os        ∈ {ubuntu-22.04, windows-2022, macOS-14}
python    ∈ {3.10, 3.13}
requires  ∈ {oldest, latest}        # 3.13 + oldest is excluded
release   ∈ (matrix dimension; see the workflow)
```

`code-checks.yml` ("Type Checks") is a single ubuntu job running `pyright -p pyproject.toml`.

Concurrency cancels in-progress runs on push **except** on `main` and `release/*`. A "cancelled" job on a
feature branch usually means you pushed again, not that anything broke.

## Step 1: Pull the Logs

**FTS uploads junit XML only, and only on failure.** There is no `output.txt` and no resource-monitor
artifact — recipes that grep `output.txt` are from the interpretune version and will not work here.

```bash
gh run list --branch <branch-name> --limit 10

# Per-job status, the fastest first look
gh run view <run_id> --json jobs --jq '.jobs[] | {name: .name, status: .status, conclusion: .conclusion}'
gh run view <run_id> --json jobs --jq '.jobs[] | select(.conclusion == "failure") | .name'

# Full log
gh run view <run_id> --log > /tmp/ci_run_<run_id>.log 2>&1

# One job's raw log (job names carry all four matrix dims)
job_id=$(gh api "repos/speediedan/finetuning-scheduler/actions/runs/<run_id>/jobs" \
  --jq '.jobs[] | select(.name | contains("windows")) | .id')
gh api "repos/speediedan/finetuning-scheduler/actions/jobs/${job_id}/logs" > /tmp/ci_job_${job_id}.log
```

Artifacts, when the run failed:

```bash
mkdir -p /tmp/ci_artifacts_<run_id>
gh run download <run_id> --dir /tmp/ci_artifacts_<run_id>
# artifact dirs are named pytest-results-<OS>-<pyver>-<oldest|latest>-<release>
# each contains junit/test-results-*.xml
```

Extract failures from the junit XML rather than grepping prose:

```bash
python - <<'PY'
import glob, xml.etree.ElementTree as ET
for f in glob.glob('/tmp/ci_artifacts_*/**/test-results-*.xml', recursive=True):
    for tc in ET.parse(f).iter('testcase'):
        for bad in list(tc.findall('failure')) + list(tc.findall('error')):
            print(f"{f.split('/')[-1]}  {tc.get('classname')}::{tc.get('name')}\n    {(bad.get('message') or '')[:200]}")
PY
```

Falling back to the plain log works too, since pytest's summary is in stdout:

```bash
grep -E "^FAILED|^ERROR" /tmp/ci_job_<job_id>.log | sort -u
grep -A 200 "= FAILURES =" /tmp/ci_job_<job_id>.log
```

## Step 2: Categorize

**A. Windows encoding.** Windows runners default to cp1252; non-ASCII in test output, file writes, or
docstrings raises `UnicodeEncodeError` on Windows only. Fix by writing with an explicit `encoding="utf-8"`
rather than by removing the character.

**B. Dependency / pin drift.** The single most common cross-platform FTS divergence. Check, in order:

- `requirements/ci/overrides.txt` — the Lightning **git commit pin**, applied via `UV_OVERRIDE`. It moves
  independently of released Lightning, so a green local run against a stale venv proves nothing.
- `requirements/ci/torch-pre.txt` — three lines: version, CUDA target, `test`|`nightly`. All commented out
  means stable. A nightly here that the Dockerfiles don't match is a real inconsistency.
- `requirements/ci/requirements.txt` vs `requirements-oldest.txt` — generated. If they look wrong,
  regenerate rather than edit: `./requirements/utils/lock_ci_requirements.sh`.
- `[tool.uv] override-dependencies` in `pyproject.toml` is intentionally **empty**; pinning happens through
  `UV_OVERRIDE`, not there.

An `oldest`-only failure is almost always this category. Reproduce with
`./scripts/build_fts_env.sh --repo-home=${PWD} --target-env-name=fts_oldest --oldest`.

**C. Type-check-only failure.** "Type Checks" red while "Test full" is green means pyright, which covers
`src/finetuning_scheduler` only. Reproduce exactly: `pyright -p pyproject.toml`. Don't silence with
`# type: ignore` before checking whether the relevant `report*` rule is already set to `none` in
`[tool.pyright]` — several are, deliberately.

**D. Test infrastructure.** Env-var leakage from torch/Lightning trips the `restore_env_variables` autouse
fixture in `tests/conftest.py`; add the new variable to its allowlist. Also check whether the test needed
`@RunIf` gating it did not get — plain pytest runs neither standalone nor exp_patch tests.

**E. Runner shutdown / cancellation.** A job that ends without a pytest summary, or with exit 143, was
cancelled or evicted rather than failed. Confirm against the concurrency rule above before investigating.

## Step 3: Prioritize

1. Failures that reproduce locally — fix these first, they need no CI round trip.
1. `oldest`-only failures — reproducible locally via a `fts_oldest` env, so still cheap.
1. Single-platform failures (Windows/macOS only) — need CI to verify, so batch them.
1. Intermittent failures — check git history first; a couple of tests are known-flaky or deliberately
   disabled on newer PyTorch.

Batch everything in tiers 3 and 4 into one push. Each CI round trip costs up to 90 minutes.

## Step 4: Fix, Push, Verify

```bash
pre-commit run --all-files      # will modify files; re-stage after
pyright -p pyproject.toml
python -m pytest src/finetuning_scheduler tests -v

git add -A && git commit -m "<lowercase imperative description>"
git push origin <branch>
```

Pre-commit's `end-of-file-fixer`, `trailing-whitespace`, `mdformat`, and `ruff --fix` all rewrite files in
place, so a commit made without running them first will come back modified. Run them before staging.

Then watch:

```bash
gh run list --branch <branch> --limit 3
gh run view <run_id> --json jobs --jq '.jobs[] | select(.conclusion == "failure") | .name'
```

## Step 5: Compare Across Platforms

Whether a failure set is identical across OSes tells you the category immediately — shared means logic,
divergent means platform.

```bash
for d in /tmp/ci_artifacts_<run_id>/*/; do
  echo "== $d"
  python - "$d" <<'PY'
import sys, glob, xml.etree.ElementTree as ET
names = set()
for f in glob.glob(sys.argv[1] + '/**/test-results-*.xml', recursive=True):
    for tc in ET.parse(f).iter('testcase'):
        if tc.find('failure') is not None or tc.find('error') is not None:
            names.add(f"{tc.get('classname')}::{tc.get('name')}")
print('\n'.join(sorted(names)))
PY
done
```

## Related Files

- `.github/workflows/ci_test-full.yml`, `.github/workflows/code-checks.yml`
- `.github/actions/install-ci-dependencies/action.yml` — how CI resolves `oldest` vs `latest`
- `requirements/ci/overrides.txt`, `torch-pre.txt`, `requirements.txt`, `requirements-oldest.txt`
- `requirements/utils/lock_ci_requirements.sh`
- `tests/conftest.py`, `tests/helpers/runif.py`
