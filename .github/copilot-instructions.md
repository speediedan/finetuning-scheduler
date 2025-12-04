# Copilot Instructions for Fine-Tuning Scheduler

## Repository Overview

**Fine-Tuning Scheduler (FTS)** is a PyTorch Lightning callback that enables flexible, multi-phase fine-tuning schedules. It allows users to define custom parameter unfreezing and optimizer configuration schedules through YAML configuration files.

**Key Technologies:**

- Python 3.9+ (CI tests on 3.9 and 3.12)
- PyTorch 2.6.0+ with PyTorch Lightning ecosystem
- Core deps: pytorch-lightning (standalone) or lightning (unified package), transformers

**Repository Size:** ~100 files, primarily Python, with YAML configs and shell scripts

## Lightning Package Support

FTS supports both standalone and unified Lightning packages:

- **Unified (default):** `lightning` package with `lightning.pytorch`
- **Standalone:** `pytorch_lightning` package

**USE_CI_COMMIT_PIN Environment Variable:**

- When set, installs Lightning from a git commit (specified in `requirements/ci/overrides.txt`)
- Default in dev/CI builds for consistent testing against latest Lightning changes
- Can be disabled with `--no_commit_pin` flag in build scripts

## Code Standards

### Required Before Each Commit

- Run tests in your local environment and ensure all tests are passing:

```bash
cd /home/speediedan/repos/finetuning-scheduler && python -m pytest src/finetuning_scheduler tests -v
```

- Ensure all pre-commit hooks pass.
- If the copilot session is still failing despite trying to get tests and pre-commit hooks passing for some time, it's okay to commit your intermediate work with a comment about the present challenge to be dealt with in a subsequent session.

### Requirement for Each Pull Request

- All pull requests must pass the CI checks.
- Ensure that the code is well-documented, with docstrings for all public functions and classes.
- Write unit tests for new functionality and ensure existing tests pass.
- Ensure the cpu coverage reported by our `ci_test-full.yml` workflow is >= the existing coverage.

## Build and Validation Commands

### Environment Setup

Development environment uses `uv` for fast, reliable dependency management:

Set environment context variables (developer-specific paths):

```bash
export FTS_VENV_BASE=/mnt/cache/${USER}/.venvs
export FTS_TARGET_VENV=fts_latest
export FTS_REPO_DIR=${HOME}/repos/finetuning-scheduler  # Example: adjust to your local repo path
```

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create development environment (creates traditional venv)
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest

# Activate the environment
cd ${FTS_REPO_DIR} && \
source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate

# Run commands directly (no need for 'uv run')
python --version
python -m pytest tests/
```

### Development Environment Scripts

Use the provided build script for automated setup:

```bash
# Standard development build (uses FTS_VENV_BASE or default ~/.venvs)
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest

# Build with explicit venv directory (recommended for hardlink performance)
./scripts/build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --venv-dir=/mnt/cache/${USER}/.venvs

# Build without Lightning commit pinning (use PyPI release)
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest --no_commit_pin

# Build with specific PyTorch nightly version
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest --torch_dev_ver=dev20240201

# Build with PyTorch test channel
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest --torch_test_channel

# Build from Lightning source
./scripts/build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --from-source="lightning:${HOME}/repos/lightning"

# Build with oldest compatible dependencies (Python 3.10, mirrors CI oldest matrix)
./scripts/build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_oldest --oldest
```

**Venv Location Options:**

- **OPTION 1 (Recommended):** Use `--venv-dir` to set the base directory
- **OPTION 2:** Use `FTS_VENV_BASE` environment variable
- **OPTION 3:** Use default (`~/.venvs/<target_env_name>`)

**Why placement matters:** UV uses hardlinks for fast installs, but hardlinks only work within the same filesystem. Placing venv on same filesystem as UV cache ensures fast installs and no warnings.

### Linting and Code Quality

**Always run linting before committing (assumes activated venv):**

```bash
# Activate your environment first
cd ${FTS_REPO_DIR} && \
source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate

# Run pre-commit hooks (includes ruff, docformatter, yaml checks)
pre-commit run --all-files

# Run ruff linting directly (configured in pyproject.toml)
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files
```

### Testing

**Test command (needs activated venv):**

```bash
# Activate your environment first
cd ${FTS_REPO_DIR} && \
source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate

# Basic test run
python -m pytest src/finetuning_scheduler tests -v

# With coverage
python -m coverage run --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v
python -m coverage report

# Test collection only (to check test discovery)
pytest --collect-only
```

**Test timing:** Most tests run quickly (\<30s), but some integration tests may take 1-2 minutes.

### Running Special Tests (Standalone and Experimental)

Some tests require special environment setup and are marked with `@pytest.mark.standalone` or require experimental patches.

**Using the test harness:**

```bash
# Run all standalone tests (default mark_type is 'standalone')
./tests/special_tests.sh

# Run standalone tests following a pattern
./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_f'

# Run experimental patch tests with a patch mask
./tests/special_tests.sh --mark_type=exp_patch --filter_pattern='test_f' --experiment_patch_mask="1 0 0 1"
```

**Manual execution:**

```bash
cd ${FTS_REPO_DIR} && \
source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate
# Run specific standalone test
PL_RUN_STANDALONE_TESTS=1 python -m pytest tests/test_specific.py::test_function -v
```

## Project Layout and Architecture

### Source Code Structure

```
src/finetuning_scheduler/     # Main package
├── __init__.py               # Package init with callback exports
├── _callback.py              # Main FinetuningScheduler callback
├── fts.py                    # Core scheduling logic
├── fts_supporters.py         # Supporting classes and utilities
├── strategy_adapters/        # Strategy-specific adapters
│   ├── fsdp.py               # FSDP strategy adapter
│   └── ddp.py                # DDP strategy adapter
└── setup_tools.py            # Package setup utilities

src/fts_examples/             # Example experiments
├── stable/                   # Stable examples
│   ├── fts_superglue.py      # SuperGLUE fine-tuning example
│   └── config/               # YAML configuration files
└── ipynb/                    # Jupyter notebook examples
```

### Configuration Files

- `pyproject.toml` - Main project config, dependencies, ruff/pytest settings
- `setup.py` - Dynamic dependency handling for Lightning packages
- `.pre-commit-config.yaml` - Code quality hooks
- `requirements/` - Dependency files
  - `base.txt` - Core dependencies
  - `extra.txt` - Extra dependencies
  - `docs.txt` - Documentation dependencies
  - `ci/overrides.txt` - Dependency overrides for dev/CI (Lightning commit pin, etc.)

### Key Entry Points

- Main callback: `finetuning_scheduler.FinetuningScheduler`
- Example script: `src/fts_examples/stable/fts_superglue.py`

## CI and Validation Pipeline

### GitHub Actions Workflow

**File:** `.github/workflows/ci_test-full.yml`

**Triggers:** Push/PR to main, changes to source/test files
**Platforms:** Ubuntu 22.04, Windows 2022, macOS 14 (Python 3.9, 3.12)
**Timeout:** 90 minutes

**Matrix Strategy:**

- `requires: ["oldest", "latest"]` - Tests oldest and newest compatible versions
- Uses `assistant.py replace_oldest_ver` to set minimum versions for oldest tests

**CI Process:**

1. Use composite action `.github/actions/install-ci-dependencies`
1. Install FTS in editable mode with dependencies
1. Run pytest with coverage
1. Upload coverage to Codecov

**CI Installation Flow:**

```bash
# Uses astral-sh/setup-uv@v7 action
# Generates Lightning override file for commit pinning
# Installs FTS with uv pip install -e ".[all]"
```

**Environment Variables for CI:**

- `USE_CI_COMMIT_PIN` - When set, uses Lightning from git commit pin
- `DISABLE_MPS` - Set to "1" on macOS to disable MPS tests

### Manual Validation Steps

Set environment context variables (developer-specific paths):

```bash
export FTS_VENV_BASE=/mnt/cache/${USER}/.venvs
export FTS_TARGET_VENV=fts_latest
export FTS_REPO_DIR=${HOME}/repos/finetuning-scheduler

cd ${FTS_REPO_DIR} && \
source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate

# Run pre-commit hooks
pre-commit run --all-files

# Run tests
python -m pytest src/finetuning_scheduler tests -v
```

### Coverage Collection

Use the `manage_standalone_processes.sh` harness with `--use-nohup` to run coverage collection in an isolated process. Output is written to `/tmp/gen_fts_coverage_<env>_<timestamp>.log`.

**Monitoring progress:**

```bash
# Tail the most recent coverage log
tail -f `ls -rt /tmp/gen_fts_coverage_fts_* | tail -1`
```

**Common coverage commands:**

```bash
# Generate coverage with rebuild (fts_latest with stable PyTorch)
~/repos/finetuning-scheduler/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh \
  --repo_home=${HOME}/repos/finetuning-scheduler \
  --target_env_name=fts_latest \
  --venv-dir=/mnt/cache/${USER}/.venvs

# Generate coverage without rebuild
~/repos/finetuning-scheduler/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh \
  --repo_home=${HOME}/repos/finetuning-scheduler \
  --target_env_name=fts_latest \
  --venv-dir=/mnt/cache/${USER}/.venvs \
  --no_rebuild_base

# Include experimental patch tests
~/repos/finetuning-scheduler/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh \
  --repo_home=${HOME}/repos/finetuning-scheduler \
  --target_env_name=fts_latest \
  --venv-dir=/mnt/cache/${USER}/.venvs \
  --include_experimental

# Generate coverage with oldest dependencies (Python 3.10, mirrors CI oldest matrix)
~/repos/finetuning-scheduler/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh \
  --repo_home=${HOME}/repos/finetuning-scheduler \
  --target_env_name=fts_oldest \
  --venv-dir=/mnt/cache/${USER}/.venvs \
  --oldest

# Generate coverage with oldest deps, skip special tests (faster CI-like run)
~/repos/finetuning-scheduler/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh \
  --repo_home=${HOME}/repos/finetuning-scheduler \
  --target_env_name=fts_oldest \
  --venv-dir=/mnt/cache/${USER}/.venvs \
  --oldest \
  --no-special
```

**Flags:**

- `--oldest`: Uses Python 3.10 and `requirements/ci/requirements-oldest.txt` (mirrors CI oldest matrix)
- `--no-special`: Skips `special_tests.sh` standalone and experimental patch tests (faster iteration)
- `--venv-dir`: Base directory for venvs (recommended: `/mnt/cache/${USER}/.venvs` for hardlink performance)

## Special Dependencies and Known Issues

### Lightning Dependency

**Dynamic Versioning:** FTS supports both unified (`lightning`) and standalone (`pytorch_lightning`) packages. The `setup.py` uses `dynamic_versioning_utils.py` to:

- Determine correct package name based on `PACKAGE_NAME` env var
- Handle commit pinning via `USE_CI_COMMIT_PIN` env var
- Toggle imports between package formats

### Dependency Constraints

- **torch** requires 2.6.0+ (oldest tested)
- **pytorch-lightning** / **lightning** requires compatible version

### Import Dependencies

- Lightning imports are conditional based on installed package
- Use `toggle_lightning_imports()` utility for package switching

## Development Guidelines

### Code Style

- **Line length:** 120 characters (configured in ruff)
- **Import style:** Sort within sections, first-party packages listed
- **Type hints:** Use modern syntax with `from __future__ import annotations`
- **Docstrings:** Google-style format

### Testing Guidelines

- Test files mirror `src/` structure in `tests/`
- Use pytest fixtures from `conftest.py`
- Add coverage for new functionality
- Standalone tests require `PL_RUN_STANDALONE_TESTS=1` env var

### Adding New Features

1. Implement feature in appropriate module
1. Add tests in `tests/`
1. Update docstrings and type hints
1. Run pre-commit and tests
1. Update CHANGELOG.md if applicable

### Configuration

- YAML configs in `src/fts_examples/stable/config/`
- Use LightningCLI for configuration parsing
- Support jsonargparse CLI integration

**Trust these instructions** - only search for additional information if these instructions are incomplete or incorrect for your specific task. The repository structure and build process follow established patterns, but following these guidelines will minimize exploration and failed commands.
