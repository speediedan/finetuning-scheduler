# Testing Finetuning Scheduler

## Quick Start (Using Build Script)

The build script automatically handles Lightning commit pinning and optional PyTorch nightly:

```bash
git clone https://github.com/speediedan/finetuning-scheduler
cd finetuning-scheduler

# Create development environment (handles Lightning pin automatically)
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest

# Activate and run tests (use your venv base path)
export FTS_VENV_BASE=~/.venvs  # or /mnt/cache/${USER}/.venvs
export FTS_VENV_NAME=fts_latest
source ${FTS_VENV_BASE}/${FTS_VENV_NAME}/bin/activate
pytest -xvs .
```

## Manual Installation

```bash
git clone https://github.com/speediedan/finetuning-scheduler
cd finetuning-scheduler

# Set UV_OVERRIDE to use the pinned Lightning commit
export UV_OVERRIDE=${PWD}/requirements/ci/overrides.txt

# Install dev deps
uv pip install ".[all]"

# Run tests
pytest -xvs .
```

## Manual Installation with PyTorch Nightly

When `torch-nightly.txt` is configured, use a two-step installation approach:

```bash
git clone https://github.com/speediedan/finetuning-scheduler
cd finetuning-scheduler

# Step 1: Install PyTorch nightly (adjust version and CUDA target as needed)
uv pip install --prerelease=if-necessary-or-explicit torch==2.10.0.dev20251124 \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Step 2: Install FTS with Lightning commit pin (torch already installed, will be skipped)
export UV_OVERRIDE=${PWD}/requirements/ci/overrides.txt
uv pip install -e ".[all]"

# Run tests
pytest -xvs .
```

The nightly version is specified in `requirements/ci/torch-nightly.txt` and documented in
`requirements/ci/torch_override.txt` for reference.

## Testing Infrastructure

FTS provides helpful scripts to accelerate building test environments and generating coverage reports.
The `gen_fts_coverage.sh` script handles environment creation/rebuild and coverage collection in a single command.

**Set environment context variables (developer-specific paths):**

```bash
export FTS_VENV_BASE=/mnt/cache/${USER}/.venvs
export FTS_TARGET_VENV=fts_latest
export FTS_REPO_DIR=${HOME}/repos/finetuning-scheduler  # Example: adjust to your local repo path
```

### Generating Coverage Reports (Recommended)

The `gen_fts_coverage.sh` script orchestrates environment building/rebuilding, running tests
(including standalone/special tests if GPU resources are available), and generating coverage reports:

**Simple usage (from repository root):**

```bash
cd ${FTS_REPO_DIR}

# Note: Both --repo-home and --repo_home work (backward compatible)
# Generate coverage with environment rebuild (builds/rebuilds environment automatically)
./scripts/gen_fts_coverage.sh --repo-home=${PWD} --target-env-name=fts_latest

# Generate coverage without rebuilding environment (faster if env already set up)
./scripts/gen_fts_coverage.sh --repo-home=${PWD} --target-env-name=fts_latest --no-rebuild-base

# Skip standalone/special tests for faster iteration
./scripts/gen_fts_coverage.sh --repo-home=${PWD} --target-env-name=fts_latest --no-rebuild-base --no-special

# Run all example tests (comprehensive validation, includes both non-standalone and standalone marked tests)
./scripts/gen_fts_coverage.sh --repo-home=${PWD} --target-env-name=fts_latest --no-rebuild-base --run-all-and-examples

# Allow tests to continue after failures (useful for validation workflows)
./scripts/gen_fts_coverage.sh --repo-home=${PWD} --target-env-name=fts_latest --no-rebuild-base --allow-failures
```

**Using nohup wrapper for long-running coverage collection:**

```bash
# Generate coverage using nohup wrapper (recommended for full coverage runs)
${FTS_REPO_DIR}/scripts/manage_standalone_processes.sh --use-nohup \
  ${FTS_REPO_DIR}/scripts/gen_fts_coverage.sh \
  --repo-home=${FTS_REPO_DIR} \
  --target-env-name=fts_latest \
  --venv-dir=${FTS_VENV_BASE}

# Include experimental patch tests
${FTS_REPO_DIR}/scripts/manage_standalone_processes.sh --use-nohup \
  ${FTS_REPO_DIR}/scripts/gen_fts_coverage.sh \
  --repo-home=${FTS_REPO_DIR} \
  --target-env-name=fts_latest \
  --venv-dir=${FTS_VENV_BASE} \
  --include-experimental

# Run all example tests (comprehensive validation)
${FTS_REPO_DIR}/scripts/manage_standalone_processes.sh --use-nohup \
  ${FTS_REPO_DIR}/scripts/gen_fts_coverage.sh \
  --repo-home=${FTS_REPO_DIR} \
  --target-env-name=fts_latest \
  --venv-dir=${FTS_VENV_BASE} \
  --no-rebuild-base \
  --run-all-and-examples

# Generate coverage with oldest dependencies (Python 3.10, mirrors CI oldest matrix)
${FTS_REPO_DIR}/scripts/manage_standalone_processes.sh --use-nohup \
  ${FTS_REPO_DIR}/scripts/gen_fts_coverage.sh \
  --repo-home=${FTS_REPO_DIR} \
  --target-env-name=fts_oldest \
  --venv-dir=${FTS_VENV_BASE} \
  --oldest

# Generate coverage with oldest deps, skip special tests (faster CI-like run)
${FTS_REPO_DIR}/scripts/manage_standalone_processes.sh --use-nohup \
  ${FTS_REPO_DIR}/scripts/gen_fts_coverage.sh \
  --repo-home=${FTS_REPO_DIR} \
  --target-env-name=fts_oldest \
  --venv-dir=${FTS_VENV_BASE} \
  --oldest \
  --no-special
```

### Monitoring Coverage Generation

Monitor progress by tailing the log files:

```bash
# Main coverage log
tail -f `ls -rt /tmp/gen_fts_coverage_fts_* | tail -1`
```

### Building Test Environments (Standalone)

If you need to build an environment without running coverage (e.g., for manual testing),
the `build_fts_env.sh` script creates virtual environments with proper Lightning commit pinning
and optional PyTorch nightly (if configured in `requirements/ci/torch-nightly.txt`):

```bash
# From the repository root directory
cd ${FTS_REPO_DIR}

# Latest development environment with stable PyTorch
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest

# With explicit venv directory
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest --venv-dir=${FTS_VENV_BASE}

# Latest with PyTorch test/RC channel
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest --torch_test_channel

# Oldest supported dependencies (Python 3.10, mirrors CI oldest matrix)
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_oldest --oldest
```

### Running Special Tests

Execute specific test subsets:

```bash
# Run an individual special test
./tests/special_tests.sh --mark_type=standalone --filter_pattern='model_parallel_integration[fsdp_autocm_tp]'

# Collect new expected state for multiple tests
export FTS_GLOBAL_STATE_LOG_MODE=1 && \
./tests/special_tests.sh --mark_type=exp_patch --filter_pattern='model_parallel' --experiment_patch_mask="1 0 0 1" && \
unset FTS_GLOBAL_STATE_LOG_MODE
```

## Notes

- The examples assume environment variables are configured for your development setup:
  - `FTS_REPO_DIR` - path to your local finetuning-scheduler clone
  - `FTS_VENV_BASE` - base directory for virtual environments
  - `FTS_TARGET_VENV` - name of the active virtual environment
- For release branch testing, create a separate working tree (e.g., `~/repos/fts-release`)
- Update `build_fts_env.sh` when testing with different PyTorch versions
- For CI testing configurations, Lightning is pinned to a specific commit via `UV_OVERRIDE` using the override file at `requirements/ci/overrides.txt`

## Running Basic Coverage Manually

### To generate full-coverage (requires minimum of 2 GPUS)

```bash
cd finetuning-scheduler
python -m coverage erase && \
python -m coverage run --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v && \
(./tests/standalone_tests.sh -k test_f --no-header 2>&1 > /tmp/standalone.out) > /dev/null && \
egrep '(Running|passed|failed|error)' /tmp/standalone.out && \
python -m coverage report -m
```

### To generate cpu-only coverage:

```bash
cd finetuning-scheduler
python -m coverage run --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v
```
