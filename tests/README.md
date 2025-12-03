# Testing Finetuning Scheduler

```bash
git clone https://github.com/speediedan/finetuning-scheduler
cd finetuning-scheduler

# FTS pins Lightning to a specific commit for CI and development (similar to PyTorch's approach with Triton)
# When running tests or developing, it's recommended to use this environment variable when installing
export USE_CI_COMMIT_PIN="1"

# install dev deps
python -m pip install ".[all]"

# run tests
pytest -xvs .
```

## Testing Infrastructure

FTS provides helpful scripts to accelerating building test environments and generation of coverage reports.

### Building Test Environments

The `build_fts_env.sh` script helps create virtual environments for testing (using `venv`):

```bash
# Latest development environment with stable PyTorch
~/repos/finetuning-scheduler/scripts/build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest

# Latest with PyTorch test/RC channel
~/repos/finetuning-scheduler/scripts/build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_test_channel

# Latest with specific PyTorch nightly
~/repos/finetuning-scheduler/scripts/build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_dev_ver=dev20241121 --torchvision_dev_ver=dev20241121

# Latest with oldest supported PyTorch version
~/repos/finetuning-scheduler/scripts/build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest_pt_oldest

# Release environment
~/repos/fts-release/scripts/build_fts_env.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release
```

### Generating Coverage Reports

The `gen_fts_coverage.sh` script runs tests and generates coverage reports (including special tests that will run based on the available hardware):

```bash
# Generate coverage without rebuilding environment
nohup ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --no_rebuild_base > /tmp/fts_latest_coverage_nohup.out 2>&1 &

# Generate coverage with experimental tests
nohup ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --no_rebuild_base --include_experimental > /tmp/fts_latest_coverage_nohup.out 2>&1 &

# Generate coverage with specific PyTorch nightly
nohup ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_dev_ver=dev20250201 --torchvision_dev_ver=dev20250201 > /tmp/fts_latest_coverage_nohup.out 2>&1 &

# Generate coverage with PyTorch test channel and verbose pip install
nohup ~/repos/finetuning-scheduler/scripts/gen_fts_coverage.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --pip_install_flags="-vv" --torch_test_channel > /tmp/fts_latest_coverage_nohup.out 2>&1 &

# Generate release coverage with PyTorch test channel
nohup ~/repos/fts-release/scripts/gen_fts_coverage.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release --torch_test_channel > /tmp/fts_release_coverage_nohup.out 2>&1 &

# Generate coverage for release with specific PyTorch channel
nohup ~/repos/fts-release/scripts/gen_fts_coverage.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release_pt2_6_x --no_rebuild_base > /tmp/fts_release_coverage_nohup.out 2>&1 &
```

### Monitoring Coverage Generation

Monitor progress by tailing the log files:

```bash
# Main coverage log
tail -f `ls -rt /tmp/gen_fts_coverage_fts_* | tail -1`
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

- The examples assume two separate working trees:
  - `~/repos/finetuning-scheduler` for development
  - `~/repos/fts-release` for release branches
- Update `build_fts_env.sh` when testing with different PyTorch versions
- For CI testing configurations, use the `USE_CI_COMMIT_PIN=1` environment variable to install Lightning from the commit specified in `requirements/ci/overrides.txt`

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
