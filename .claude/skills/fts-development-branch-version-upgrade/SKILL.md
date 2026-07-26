---
name: fts-development-branch-version-upgrade
description: Automates Fine-Tuning Scheduler development branch version upgrade process including basic metadata oriented code changes, documentation updates, environment setup, testing, and coverage collection. Use when starting development on a new FTS minor release (e.g., 2.10.0 -> 2.11.0).
license: Apache-2.0
metadata:
  author: speediedan
  version: '1.0'
compatibility: Requires bash, git, uv, Python 3.10+, and access to local FTS repository at ~/repos/finetuning-scheduler
---

# FTS Development Branch Version Upgrade Skill

This skill automates the process of upgrading the Fine-Tuning Scheduler (FTS) development branch to a new version, including all necessary metadata oriented code changes (for now, may have more ambitious goals for code fixes in the future), documentation updates, environment setup, and validation steps.

## When to Use This Skill

Use this skill when:

- Starting development on a new FTS minor or major release
- Need to bump PyTorch minimum/maximum supported versions
- Upgrading CUDA versions for Docker builds
- Need to synchronize version metadata across all FTS files

## Required User Inputs

Before running this skill, gather the following information:

1. **Current FTS version** (e.g., 2.10.0) - typically found in `src/finetuning_scheduler/__about__.py` with `.dev0` suffix
1. **New FTS version** (e.g., 2.11.0) - the target version to upgrade to
1. **New PyTorch minimum version** (e.g., 2.7.0) - oldest PyTorch version to support (support 4-5 minor versions)
1. **New PyTorch maximum version** (e.g., 2.11.0) - latest PyTorch version to test
1. **New PyTorch nightly version** (e.g., dev20260121) - target nightly build for development
1. **New CUDA version** (e.g., 13.0.2) - CUDA toolkit version for Docker images
1. **\[Optional\] Updated TORCH_CUDA_ARCH_LIST** (e.g., "7.5;8.0;8.6;9.0;10.0;12.0+PTX") - mirror upstream PyTorch
1. **\[Optional\] New Lightning min/max versions** - if Lightning compatibility changes
1. **\[Optional\] Report output location** - defaults to `~/repos/distributed-insight/project_admin/finetuning-scheduler/handoff_docs/` or `/tmp/`

## Prerequisites

- FTS repository checked out locally at `~/repos/finetuning-scheduler` on the `main` branch
- Clean working tree (commit or stash uncommitted changes first)
- UV package manager installed
- Python 3.10+ available
- Sufficient disk space for coverage collection (~2GB)
- Active environment variables: `FTS_VENV_BASE`, `FTS_TARGET_VENV`, `FTS_REPO_DIR`

## Step-by-Step Process

### Phase 1: Gather Context and Validate Inputs

1. **Verify working directory**:

   ```bash
   cd ~/repos/finetuning-scheduler
   git status  # Ensure on main branch with clean working tree
   ```

1. **Prompt user for missing inputs** if not provided:

   - Current version (read from `src/finetuning_scheduler/__about__.py`)
   - New version
   - New PyTorch min/max versions
   - New nightly version
   - New CUDA version
   - Optional: CUDA arch list, Lightning versions

1. **Validate inputs**:

   - Version format: semantic versioning (X.Y.Z)
   - PyTorch versions: min \< max
   - CUDA version: major.minor.patch format
   - Nightly format: X.Y.Z.devYYYYMMDD

### Phase 2: Update Version Metadata Files

Update the following files with version changes:

#### Core Version Files

1. **`src/finetuning_scheduler/__about__.py`**:

   ```python
   __version__ = "{new_version}.dev0"
   ```

1. **`CITATION.cff`** — **do NOT bump during a development-branch upgrade.**

   `CITATION.cff` describes the most recent *published* release and is what citation tooling resolves
   against. It must only ever be advanced as part of cutting an actual release, never when moving the
   development branch to a new `.dev0` version. A mismatch between `CITATION.cff` (e.g. `2.13.0`) and
   `__about__.py` (e.g. `2.14.0.dev0`) on `main` is **expected and correct** — do not "fix" it.

   **But do verify it is not STALE.** Phase 2b deliberately excludes `CITATION.cff`, so nothing else
   catches the failure mode where it names a version that was never published (observed: it read `2.11.0`
   for months after that release was skipped). It must equal the newest git tag:

   ```bash
   echo "CITATION.cff: $(grep '^version:' CITATION.cff | awk '{print $2}')"
   echo "newest tag  : $(git tag --sort=-v:refname | head -1 | sed 's/^v//')"
   ```

   If they disagree, `CITATION.cff` is stale — correct it to the newest tag *before* proceeding, as a
   separate concern from this upgrade. Also confirm the release's Zenodo DOI was added under
   `identifiers:` (post-release, per `release_flow.md` step 14).

1. **`CHANGELOG.md`**:

   - Add new version section at top:
     ```markdown
     ## [{new_version}] - 2026-XX-XX

     ### Added

     ### Fixed

     ### Changed

     ### Deprecated
     ```
   - Update previous version release date if not set

   ⚠️ **If you are SKIPPING one or more versions** (e.g. 2.11.0.dev0 → 2.13.0 because PyTorch 2.11 and
   2.12 shipped without matching FTS releases), do **not** add a new section — the existing unreleased
   section already holds the accumulated work. Instead:

   - **Retitle** the existing unreleased heading to the new version.
   - Add a `## [X.Y.Z] and [X.Y+1.Z] - not released` stanza below it recording which versions were skipped
     and why, and noting that the retained min-torch means every version they would have targeted is
     still supported.

   Set the date to `2026-XX-XX` when bumping the dev branch, and to the real date only when the release is
   actually cut (`release_flow.md` step 2).

#### PyTorch Version Files

4. **`src/finetuning_scheduler/dynamic_versioning/utils.py`**:

   ```python
   BASE_DEPENDENCIES = [
       "torch>={new_pytorch_min}",
   ]
   ```

1. **`src/finetuning_scheduler/dynamic_versioning/utils.py`** — the Lightning constraint:

   ```python
   LIGHTNING_VERSION = ">={lightning_min},<{lightning_ceiling}"
   ```

   ⚠️ **Ceiling trap.** This constraint is the real install-time gate for both `lightning` and
   `pytorch-lightning`. A ceiling like `<2.6.1` excludes *every* Lightning patch release in the series
   (2.6.1, 2.6.4, 2.6.5, …), silently pinning users to the `.0`. Unless there is a specific known
   incompatibility, the ceiling should exclude the next **minor** (e.g. `>=2.6.0,<2.7.0`), not the next
   patch. Verify against the actual published versions before choosing:

   ```bash
   pip index versions lightning
   ```

1. **`pyproject.toml`**:

   ```toml
   [tool.fts.min-versions]
   torch = ">={new_pytorch_min}"
   lightning = ">={lightning_min},<{lightning_ceiling}"   # must mirror LIGHTNING_VERSION exactly
   ```

1. **`requirements/ci/torch-pre.txt`**:

   ```
   {new_pytorch_max}.{nightly_version}
   cu{cuda_major}{cuda_minor}0
   nightly
   ```

   ⚠️ **Only if the target is actually a prerelease.** This file is *data* — three lines parsed by both
   the CI action and the build scripts — and all-commented means "use stable torch from PyPI".

   - **Stable/catch-up release** (target torch is already GA): leave the file **fully commented out**.
     Populating it pins CI to a nightly that will be pruned from the index within weeks, at which point
     env builds fail to resolve. Note this also changes lockfile generation: `--no-emit-package torch` is
     passed *only* on the prerelease path, so a stable target makes `requirements.txt` pin `torch`
     and removes `torch-override.txt`. Both are expected.
   - **Prerelease target**: populate it, and pick the nightly deliberately rather than taking the newest.
     Enumerate what exists and confirm the wheels are complete across every target CI exercises
     (`cu*`/`cpu` × min and max Python) — a nightly missing any of them will fail some matrix leg:
     ```bash
     python -c "
     import urllib.request, re
     h = urllib.request.urlopen('https://download.pytorch.org/whl/nightly/cu130/torch/').read().decode()
     print(sorted(set(re.findall(r'torch-(2\.14\.0\.dev\d{8})%2Bcu130-cp313-', h)))[-8:])"
     ```
     Prefer one a couple of days old so obvious upstream regressions have had time to surface. Expect to
     re-pin nearer feature freeze and then switch to the `test` (RC) channel once RC1 lands.

   The **same nightly must also be set in `dockers/base-cuda/Dockerfile`**, where exactly one of the
   stable / nightly / test install lines may be uncommented. A stale hardcoded nightly there broke the
   image build once the old nightly was pruned.

#### Determining the target CUDA version

If the user did not supply `New CUDA version`, derive it rather than guessing — `RELEASE.md` prose has
been observed to be incomplete (it omits CUDA 12.9 for 2.13 despite cu129 Linux wheels existing).

1. **`CUDA_STABLE`** in `.github/scripts/generate_binary_build_matrix.py` **at the target release tag**
   (not `main`) is authoritative: it drives both the primary-tested CI version and the PyPI wheel.
1. **Confirm against published wheels** by enumerating `https://download.pytorch.org/whl/torch/` for
   `torch-<version>%2B(cu\d+)-` variants.
1. **Take the toolkit patch version from torch's own pin** — this is what the Docker base image must
   match:
   ```bash
   python -c "from importlib.metadata import requires; print([r for r in requires('torch') if 'cuda-toolkit' in r])"
   # torch 2.13.0 -> cuda-toolkit[...]==13.0.3  =>  ARG CUDA_VERSION=13.0.3
   ```
1. **Mirror `TORCH_CUDA_ARCH_LIST`** from `TORCH_CUDA_ARCH_LIST_TABLE` in
   `.ci/manywheel/build_env_setup.py` for that CUDA version, `x86_64`. CUDA 13.0/13.2 → `{75, 80, 86, 90, 100, 120}` → `"7.5;8.0;8.6;9.0;10.0;12.0+PTX"`. **Do not drop `7.5`** — the CI host has an
   RTX 2070 SUPER (sm_75).

The per-release decision is announced in a dedicated RFC issue (2.11 → pytorch#172663,
2.12 → pytorch#178665, 2.14 → pytorch#190355). Check the issue for the target release; if it is still
open, surface that to the user rather than assuming.

#### Docker Configuration Files

7. **`dockers/base-cuda/Dockerfile`**:

   ```dockerfile
   ARG CUDA_VERSION={new_cuda_version}
   ARG PYTORCH_VERSION={new_pytorch_max}
   ENV TORCH_CUDA_ARCH_LIST="{new_cuda_arch_list}"
   ```

   - Update nightly installation line (if another test or stable version of the line exists, comment those out and uncomment the nightly line):
     ```dockerfile
     uv pip install --prerelease=allow torch=={new_pytorch_max}.{nightly_version} --index-url https://download.pytorch.org/whl/nightly/cu{cuda_major}{cuda_minor}0
     ```

1. **`dockers/fts-az-base/Dockerfile`**:

   ```dockerfile
   ARG PYTORCH_VERSION={new_pytorch_max}
   ```

1. **`dockers/release/Dockerfile`**:

   ```dockerfile
   ARG PYTORCH_VERSION={new_pytorch_max}
   ```

1. **`dockers/docker_images_main.sh`**:

   ```bash
   declare -A iv=(["cuda"]="{new_cuda_version}" ["python"]="{new_python_version}" \
                  ["pytorch"]="{new_pytorch_max}" ["lightning"]="{lightning_minor}" ["cust_build"]="1")
   ```

1. **`dockers/docker_images_release.sh`**:

   ```bash
   declare -A iv=(["cuda"]="{new_cuda_version}" ["python"]="{new_python_version}" \
                  ["pytorch"]="{new_pytorch_max}" ["lightning"]="{lightning_minor}" ["cust_build"]="0")
   ```

   ⚠️ **Every key must be updated in BOTH scripts.** These two files differ only in `cust_build`; any
   other divergence is drift. Historically the `lightning` key was updated in `docker_images_main.sh`
   but missed in `docker_images_release.sh`, leaving `2.5` against `2.6` everywhere else. Phase 2b
   catches exactly this.

#### GitHub Workflows and CI Files

12. **`.github/workflows/release-docker.yml`**:

    ```yaml
    pytorch_version: ["{new_pytorch_max}"]
    cust_base: ["cu{new_cuda_version}-"]
    ```

01. **`.azure-pipelines/gpu-tests.yml`**:

    ```yaml
    image: "speediedan/finetuning-scheduler:py3.13-pt{new_pytorch_max}-pl2.6-azpl-init"
    ```

#### Documentation Files

14. **`docs/source/versioning.rst`**:

    - Add new version row to compatibility table:
      ```rst
      * - {new_version}.x
        - {new_pytorch_min}
        - {new_pytorch_max}
        - >= {lightning_min}
      ```

    ⚠️ **Check whether the top row is speculative before adding.** The table documents *released*
    versions. If the current top row describes a version that was never published (it will exist if a
    previous dev-branch bump added it), **replace** that row rather than adding a new one — otherwise the
    matrix advertises support for a release users cannot install.

01. **`docs/source/install/dynamic_versioning.rst`**:

    - Update example torch version in comments
    - Update CUDA target examples

01. **`CLAUDE.md`**:

    - Update minimum PyTorch version references
    - Update example installation commands with new versions

    (`.github/copilot-instructions.md` is superseded by `CLAUDE.md`. Update it too only while it still exists.)

01. **`README.md`**:

    - Update example versions in installation instructions
    - Update torch-pre.txt format examples
    - Update build status table if applicable (usually is applicable, we will have new oldest and newest tested versions of pytorch with potentially new CUDA versions)

01. **`tests/README.md`**:

    - Update example installation commands

01. **`.github/CONTRIBUTING.md`**:

    - Update example installation commands

01. **`.github/ISSUE_TEMPLATE/bug_report.md`**:

    - Update example versions in environment section

#### Build Scripts

21. **`scripts/build_fts_env.sh`**:

    - Update default torch-backend (e.g., `cu130`)
    - Update comments with new version examples

01. **`scripts/gen_fts_coverage.sh`**:

    - Update torch-backend comments and defaults

01. **`scripts/infra_utils.sh`**:

    - Update CUDA target comments in `get_torch_index_url()`

01. **`requirements/utils/lock_ci_requirements.sh`**:

    - Update comments with new CUDA targets in manual installation examples

#### Dependency Pins

1. **`requirements/ci/overrides.txt`**:

   - Retarget the Lightning git commit pin (`lightning @ git+https://github.com/Lightning-AI/lightning.git@<sha>`)
     if the upgrade tracks a new Lightning commit. This pin is applied via `UV_OVERRIDE` and is easy to miss
     because it lives outside `pyproject.toml`.

1. **`requirements/ci/torch-override.txt`**:

   - Regenerated by Phase 3, but its header comments hardcode the torch prerelease version and CUDA target.
     Update those so they don't contradict the regenerated content.

1. **`pyproject.toml`** `[tool.fts.min-versions]`:

   - The `lightning` entry lives in the same table as `torch`. If Lightning min/max changed, update it here
     too. Note this table is informational only — the enforced values are in `dynamic_versioning/utils.py`.

### Phase 2b: Verify cross-file version consistency (MANDATORY GATE)

Several of the versions above are declared in six or seven places that must agree. Updating some and
missing others is the single most common defect in this process, and it is invisible until a Docker
build or Azure job pulls a tag that was never built. Run the audit before proceeding:

```bash
python scripts/verify_version_consistency.py
```

It prints every declaration side by side, flags the outlier with `!`, and exits non-zero on any
disagreement. **Do not proceed to Phase 3 until it exits 0.** Example of the failure it catches:

```
lightning: MISMATCH -> ['2.5', '2.6']
    fts-az-base Dockerfile       2.6
    docker_images_main.sh        2.6
  ! docker_images_release.sh     2.5      <- the file that was missed
```

If a probe reports `<not found>`, a file was restructured and the probe in
`scripts/verify_version_consistency.py` needs updating — treat that as a real failure, not noise, since
a stale probe silently stops checking that file.

Note the audit deliberately does **not** cover `CITATION.cff` or `__about__.py`: on a development branch
those are *expected* to disagree (see Phase 2, Core Version Files).

### Phase 2c: Verify the coverage badge actually renders (MANDATORY GATE)

The README coverage badge is flag-filtered (`badge.svg?flag=gpu`) and fails **silently** — it renders the
word "unknown" rather than erroring, so a broken badge can sit on the README indefinitely. Check it
directly rather than trusting the rendered README:

```bash
for f in gpu cpu pytest; do
  printf "flag=%s -> " "$f"
  curl -sS "https://codecov.io/gh/speediedan/finetuning-scheduler/branch/main/graph/badge.svg?flag=${f}" \
    | grep -oE ">[a-z0-9%]+</text>" | tail -1
done
# also the unfiltered total:
curl -sS "https://codecov.io/gh/speediedan/finetuning-scheduler/branch/main/graph/badge.svg" \
  | grep -oE ">[a-z0-9%]+</text>" | tail -1
```

Repeat for the release branch, URL-encoding the slash: `branch/release%2F2.13.x`.

**Expected:** every flag resolves to a percentage, and `gpu` is at or near the coverage the Azure
pipeline reports (100% at the time of writing). **"unknown" for any flag is a failure.**

Interpretation when something is wrong:

| Symptom                                                    | Likely cause                                                                                                                                                                                                                |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gpu` unknown, `cpu` fine                                  | The branch HEAD has no GPU upload and carryforward is off. Codecov's branch badge reflects the **HEAD commit**, and Azure runs less often than the CPU matrix. Fix: `flags: {gpu: {carryforward: true}}` in `.codecov.yml`. |
| All flags unknown                                          | Upload credential or slug problem.                                                                                                                                                                                          |
| Total much lower than the GPU pipeline's reported coverage | The GPU report is not merging into the branch total — same carryforward cause.                                                                                                                                              |

To distinguish a credential problem from a carryforward problem, upload a throwaway flag manually
against the branch HEAD; if it appears, the credential and CLI invocation are fine:

```bash
set -a && . ./.env && set +a       # CODECOV_TOKEN (upload token, NOT an API token)
/tmp/codecov upload-process --slug 'speediedan/finetuning-scheduler' -t "${CODECOV_TOKEN}" \
  --commit-sha "$(git rev-parse origin/main)" --git-service 'github' \
  -n "token-diagnostic" -F tokencheck -f 'coverage.xml'
```

Validate any `.codecov.yml` change before committing — it accepts unknown keys silently otherwise:

```bash
curl --data-binary @.codecov.yml https://codecov.io/validate
```

### Phase 3: Regenerate CI Requirements

> **Skip Phases 3-6 if the dependency work is already done.** A version bump that follows a completed
> currency PR (lockfiles already regenerated, env already rebuilt, coverage already collected, docs
> already verified) only needs the metadata edits plus the Phase 2b/2c gates. Re-running these phases is
> not harmful, just slow — ~40 minutes of rebuild and coverage for no new information. Decide by checking
> whether `git log` since the last release already contains a lockfile regeneration, and whether
> `./requirements/utils/lock_ci_requirements.sh` produces a clean `git diff`.

After updating version files, regenerate locked requirements:

```bash
cd ${FTS_REPO_DIR}
source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate
./requirements/utils/lock_ci_requirements.sh
```

**Expected outputs**:

- `requirements/ci/requirements.txt` (highest resolution)
- `requirements/ci/requirements-oldest.txt` (lowest resolution)
- `requirements/ci/torch-override.txt` (torch prerelease override)

**Validation**:

- Check that torch version in requirements.txt matches expectations
- Verify torch-override.txt contains correct prerelease version
- Ensure no unexpected dependency changes

### Phase 4: Rebuild Development Environment

Rebuild the development environment with new dependencies:

```bash
~/repos/finetuning-scheduler/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/finetuning-scheduler/scripts/build_fts_env.sh \
  --repo-home=${HOME}/repos/finetuning-scheduler \
  --target-env-name=fts_latest \
  --venv-dir=/mnt/cache/${USER}/.venvs
```

**Monitor progress**:

```bash
tail -f $(ls -rt /tmp/build_fts_env_* | tail -1)
```

**Expected duration**: ~5 minutes (mostly to download new torch version)

**Validation**:

- Build completes without errors
- New PyTorch version is installed
- UV doesn't report hardlink warnings (if venv on same filesystem as UV cache)

### Phase 5: Collect Full Coverage

Run comprehensive test coverage collection (runs in background):

```bash
~/repos/finetuning-scheduler/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh \
  --repo-home=${HOME}/repos/finetuning-scheduler \
  --target-env-name=fts_latest \
  --venv-dir=/mnt/cache/${USER}/.venvs \
  --no-rebuild-base \
  --allow-failures
```

**Monitor progress**:

```bash
tail -f $(ls -rt /tmp/gen_fts_coverage_fts_* | tail -1)
```

**Expected duration**: ~30 minutes

#### Handle Test Failures

**New leaked variables**:

- If PyTorch or Lightning leaks new environment variables, add to `tests/conftest.py`:
  ```python
  @pytest.fixture(scope="function", autouse=True)
  def restore_env_variables():
      allowlist = {
          # ... existing entries ...
          "NEW_LEAKED_VAR",  # leaked by torch/lightning {version}
      }
  ```

**Other errors**:

- Document all errors in the upgrade report
- Most errors should be addressed manually post-upgrade
- Include error messages, affected tests, and potential fixes

### Phase 5b: Type Check

A PyTorch or Lightning bump is exactly when new type errors surface, and `code-checks.yml` gates on this:

```bash
cd ${FTS_REPO_DIR} && source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate
pyright -p pyproject.toml
```

Pyright covers `src/finetuning_scheduler` only (`tests`, `docs`, `build`, `dist` are excluded). Record any
new diagnostics in the upgrade report; resolve them before opening the version-bump PR.

Also run the full pre-commit suite once the metadata edits are in:

```bash
pre-commit run --all-files
```

### Phase 6: Rebuild and Validate Documentation

1. **Clean and rebuild docs**:

   ```bash
   export FTS_VENV_BASE=/mnt/cache/${USER}/.venvs
   export FTS_TARGET_VENV=fts_latest
   export FTS_REPO_DIR=${HOME}/repos/finetuning-scheduler

   cd ${FTS_REPO_DIR} && source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate
   cd docs && make clean
   make html --debug SPHINXOPTS="-W --keep-going"
   ```

1. **Run linkcheck**:

   ```bash
   make linkcheck SPHINXOPTS="-W --keep-going"
   grep -i "error\|broken" build/linkcheck/output.txt || echo "No errors found"
   ```

**Expected results**:

- No build warnings or errors
- No broken links
- All cross-references resolve correctly

**Document issues**:

- Any new warnings or errors
- Broken links (may need URL updates)
- Missing cross-references

### Phase 7: Generate Upgrade Report

Create comprehensive report at the determined output location (default: `~/repos/distributed-insight/project_admin/finetuning-scheduler/handoff_docs/fts_version_upgrade_{new_version}_YYYYMMDD.md`):

#### Report Structure

```markdown
# FTS Version Upgrade Report: {old_version} → {new_version}

**Date**: YYYY-MM-DD
**Agent**: [Your name/identifier]
**Duration**: [Total time taken]

## Executive Summary

- FTS version upgraded from {old_version} to {new_version}
- PyTorch support updated: {old_pytorch_min}-{old_pytorch_max} → {new_pytorch_min}-{new_pytorch_max}
- CUDA version updated: {old_cuda} → {new_cuda}
- Coverage: X% (change from baseline)
- Status: [All tests passing / N failures documented below]

## Changes Made

### Version Metadata Updates
- [List all files changed with brief description]

### Dependency Updates
- PyTorch minimum: {old} → {new}
- PyTorch maximum: {old} → {new}
- PyTorch nightly: {old} → {new}
- CUDA version: {old} → {new}
- [Any other dependency changes]

### Docker Configuration
- Base image CUDA: {old} → {new}
- PyTorch version: {old} → {new}
- CUDA arch list: [changes if any]

## Build and Test Results

### Environment Rebuild
- Status: [Success/Failure]
- Duration: [X minutes]
- Issues: [None / list issues]

### Coverage Collection
- Status: [Success/Partial/Failure]
- Duration: [X minutes]
- Total coverage: X%
- Coverage change: [+/- X%]

### Test Failures
[If any tests failed]

#### Leaked Variables Fixed
- `VARIABLE_NAME`: leaked by torch/lightning {version} - [added to allowlist]

#### Outstanding Issues
1. **Test**: test_name
   **Error**: [error message]
   **Proposed fix**: [suggestion]

### Documentation Build
- Status: [Success/Failure]
- Warnings: [count]
- Broken links: [count]
- Issues: [None / list issues]

## Recommendations

### Immediate Actions Required
- [List any critical issues that need manual intervention]

### Follow-up Tasks
- [List non-critical improvements or cleanups]

### Skill Improvements
[Suggestions for improving this skill based on issues encountered]

## Validation Checklist

- [ ] All version files updated
- [ ] CI requirements regenerated
- [ ] Development environment rebuilt successfully
- [ ] Coverage collected (with acceptable failure rate)
- [ ] Documentation builds without errors
- [ ] No new broken links
- [ ] Leaked variables handled
- [ ] Report generated

## Files Modified

[Complete list of files changed during upgrade]

## Next Steps

1. Review and address outstanding test failures
2. Commit changes with message: "Bump version to {new_version}, update PyTorch to {new_pytorch_max}"
3. Create PR for version upgrade
4. Monitor CI runs for any platform-specific issues

## Appendix

### Full Coverage Output
[Attach or reference full coverage report]

### Build Logs
[Reference to build log locations]

### Test Error Details
[Full stack traces for failed tests]
```

### Phase 8: Propose Skill Improvements

If any unexpected issues were encountered, suggest updates to this skill:

**Common improvement areas**:

- New files that need version updates
- Changed file paths or structures
- New validation steps needed
- Improved error handling
- Better progress monitoring
- Additional automation opportunities

**Format for suggestions**:

```markdown
## Suggested Skill Improvements

### Addition: [New step/file to handle]
**Reason**: [Why this is needed]
**Implementation**: [How to add it]

### Clarification: [Existing step that was unclear]
**Issue**: [What was confusing]
**Proposed update**: [Clearer wording]

### Automation: [Manual step that could be automated]
**Current process**: [What's done manually]
**Proposed automation**: [How to automate]
```

## Error Handling

### Common Issues and Solutions

**Issue**: UV hardlink warnings during environment build
**Solution**: Ensure venv is on same filesystem as UV cache (use `--venv-dir` flag)

**Issue**: Torch prerelease not found
**Solution**: Verify nightly version exists at PyTorch download site, may need to use different date

**Issue**: Coverage collection hangs
**Solution**: Check for conflicting pytest processes, kill if >40 minutes old

**Issue**: Documentation build fails with missing references
**Solution**: Check for typos in cross-references, ensure all referenced sections exist

**Issue**: Locked requirements have unexpected versions
**Solution**: Check pyproject.toml constraints, may need to update dependency pins

## Validation Steps

After completing all phases, verify:

1. **Version consistency**: `python scripts/verify_version_consistency.py` exits 0 (see Phase 2b)
1. **Coverage badge**: every flag renders a percentage, not "unknown" (see Phase 2c)
1. **Build success**: Environment builds without errors
1. **Test status**: Coverage collected (failures documented)
1. **Documentation**: Builds cleanly with no warnings
1. **Git status**: All changes tracked, ready to commit

## Notes

- Always run on a clean working tree (commit or stash first)
- Use `--allow-failures` for initial coverage run to capture all issues
- Monitor log files to catch issues early
- Keep old environment as backup until new one validated
- Document all deviations from expected behavior

## Related Documentation

- [FTS Versioning Policy](https://finetuning-scheduler.readthedocs.io/en/latest/versioning.html)
- [Dynamic Versioning System](https://finetuning-scheduler.readthedocs.io/en/latest/install/dynamic_versioning.html)
- [Contributing Guide](https://github.com/speediedan/finetuning-scheduler/blob/main/.github/CONTRIBUTING.md)
- [Agent Skills Specification](https://agentskills.io/specification)
