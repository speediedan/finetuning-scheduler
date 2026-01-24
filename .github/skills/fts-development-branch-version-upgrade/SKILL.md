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
1. **New CUDA version** (e.g., 13.0.0) - CUDA toolkit version for Docker images
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

1. **`CITATION.cff`**:

   ```yaml
   version: {new_version}
   ```

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

#### PyTorch Version Files

4. **`src/finetuning_scheduler/dynamic_versioning/utils.py`**:

   ```python
   BASE_DEPENDENCIES = [
       "torch>={new_pytorch_min}",
   ]
   ```

1. **`pyproject.toml`**:

   ```toml
   [tool.fts.min-versions]
   torch = ">={new_pytorch_min}"
   ```

1. **`requirements/ci/torch-pre.txt`**:

   ```
   {new_pytorch_max}.{nightly_version}
   cu{cuda_major}{cuda_minor}0
   nightly
   ```

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
   declare -A iv=(["cuda"]="{new_cuda_version}" ["pytorch"]="{new_pytorch_max}" ...)
   ```

1. **`dockers/docker_images_release.sh`**:

   ```bash
   declare -A iv=(["cuda"]="{new_cuda_version}" ["pytorch"]="{new_pytorch_max}" ...)
   ```

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

01. **`docs/source/install/dynamic_versioning.rst`**:

    - Update example torch version in comments
    - Update CUDA target examples

01. **`.github/copilot-instructions.md`**:

    - Update minimum PyTorch version in "Key Technologies"
    - Update example installation commands with new versions

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

### Phase 3: Regenerate CI Requirements

After updating version files, regenerate locked requirements:

```bash
cd ~/repos/finetuning-scheduler
source /mnt/cache/${USER}/.venvs/fts_latest/bin/activate
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

1. **Version consistency**: All files reference new version correctly
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
