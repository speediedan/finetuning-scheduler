# Automated Warnings Cleanup

This directory contains automation for managing expected warnings in the finetuning-scheduler test suite.

## Overview

The warnings cleanup automation replaces the manual process previously documented in `distributed-insight/project_admin/finetuning-scheduler/admin_docs/warnings_cleanup_process.md`.

**Note:** This automation is designed for **local use only**. CI integration is not currently supported due to multi-GPU requirements for special tests and limited self-hosted runner capacity.

## Architecture

### Components

1. **`automate_warnings_cleanup.py`** (main orchestrator)

   - Python script that coordinates the entire workflow
   - Handles environment setup, test execution, warning analysis, and file updates
   - Supports three modes: full, verify, validate-only

1. **`run_warnings_cleanup.sh`** (convenience wrapper)

   - Shell script providing user-friendly interface
   - Handles argument parsing and logging
   - Colorized output for better UX

### Existing Scripts Reused

- `build_fts_env.sh` - Environment creation
- `gen_fts_coverage.sh` - Test execution framework
- `manage_standalone_processes.sh` - Background process management
- `tests/special_tests.sh` - Standalone test execution

## Usage

### Usage

#### Full cleanup before release

```bash
# This will:
# 1. Rebuild fts_latest and fts_oldest environments
# 2. Run all tests (including standalone/special tests)
# 3. Analyze warnings
# 4. Update expected_warns.py by commenting out unmatched warnings
# 5. Verify all warnings are now matched
# 6. Validates with gen_fts_coverage.sh that all tests pass
# 7. Generate detailed json report
./scripts/run_warnings_cleanup.sh --full
```

#### Dry run (see what would happen)

```bash
./scripts/run_warnings_cleanup.sh --full --dry-run
```

## Modes

### `--verify` (default)

- Checks if current warnings match expected warnings
- No modifications to code
- Fast iteration during development
- Exits with error if mismatches found

### `--full`

- Complete cleanup workflow
- Rebuilds environments (unless `--no-rebuild`)
- Runs all tests
- **Conservative warning removal**: Only comments out warnings that are:
  - Unmatched in **ALL** test contexts (nonexample_latest, nonexample_oldest, example_latest)
  - Consistently absent across multiple test scenarios
- Warnings found in **ANY** context are preserved (they may be intermittent, hardware-specific, or only appear in standalone/distributed tests)
- Verifies updates are correct
- Use before releases or after major dependency updates

### `--validate-only`

- Runs full coverage with all examples and allows failures to validate current expected_warns.py
- Useful after manual edits to expected_warns.py or failed previous cleanup runs

**Important:** The script is intentionally conservative to avoid removing warnings that appear only in specific scenarios (e.g., FSDP multi-GPU tests, timing-dependent warnings). Manual review is recommended if the same warnings persist across multiple cleanup runs.

### `--dry-run` (can be combined with any mode)

- **Intelligent dry-run**: Uses existing test logs if available
- Shows what warnings would be commented out
- Displays complete workflow without making changes
- Generates JSON report with unmatched warnings
- No files are modified
- Perfect for previewing changes before committing

## Workflow Details

### 1. Environment Setup

```bash
# Builds/rebuilds test environments
build_fts_env.sh --repo_home=... --target_env_name=fts_latest
build_fts_env.sh --repo_home=... --target_env_name=fts_oldest --oldest
```

### 2. Test Execution

Tests are run in multiple contexts:

- **nonexample_latest**: Core tests with latest dependencies
- **nonexample_oldest**: Core tests with oldest supported dependencies
- **example_latest**: Example tests with special standalone tests

### 3. Warning Analysis

```bash
# Generate current expected warnings
python -c "from tests.helpers.expected_warns import print_warns; print_warns(example_warns=False)"

# Compare against test logs
# Identifies:
# - Unmatched warnings (expected but not found)
# - New warnings (found but not expected)
```

### 4. Update expected_warns.py (full mode only)

```python
# Comments out unmatched warnings with TODO marker
# "warning text that was not found",  # TODO: Remove after successful CI run
```

### 5. Verification

Re-runs analysis to confirm all warnings now match.

### 6. Coverage Validation

Runs `gen_fts_coverage.sh --run-all-and-examples --allow-failures` to ensure all tests pass with updated expected_warns.py.

### 7. Report Generation

Creates JSON report with:

- Total warnings expected/found per context
- List of unmatched warnings
- List of new warnings
- Timestamp and metadata

## File Structure

```
scripts/
├── automate_warnings_cleanup.py    # Main Python orchestrator
├── run_warnings_cleanup.sh         # Shell wrapper
├── README_warnings_cleanup.md      # This file
├── build_fts_env.sh               # (existing) Environment builder
├── gen_fts_coverage.sh            # (existing) Coverage generator
└── manage_standalone_processes.sh # (existing) Process manager

expected_warns.py              # Warning definitions (modified by automation)

/tmp/ (or custom --working-dir)
├── warnings_cleanup_report_*.json      # Analysis results
├── review_warn_nonexample_latest_*.out # Test logs
├── review_warn_nonexample_oldest_*.out
├── review_warn_example_latest_*.out
├── current_nonexample_warns.out        # Expected warnings list
└── current_example_warns.out
```

## Configuration

### Environment Variables

- `FTS_VENV_BASE`: Base directory for virtual environments (default: `~/.venvs`)
- `REPO_HOME`: Repository root (auto-detected)
- `FTS_CLEANUP_LOG`: Override log file location

### Script Arguments

```bash
# Python script
python scripts/automate_warnings_cleanup.py \
    --mode {full|verify|validate-only} \
    --repo-home /path/to/repo \
    --venv-base /path/to/venvs \
    [--no-rebuild] \
    [--skip-special-tests] \
    [--dry-run] \
    [--interactive] \
    [--working-dir /path/to/workdir]

# Shell wrapper (simpler interface)
./scripts/run_warnings_cleanup.sh \
    {--full|--verify|--validate-only} \
    [--no-rebuild] \
    [--skip-special] \
    [--dry-run] \
    [--venv-base DIR] \
    [--working-dir DIR]
```

## Common Workflows

### Pre-Release Cleanup

```bash
# Option 1: Run directly (blocks terminal)
./scripts/run_warnings_cleanup.sh --full

# Option 2: Run in background (recommended for long operations)
./scripts/manage_standalone_processes.sh --use-nohup ./scripts/run_warnings_cleanup.sh --full

# Monitor progress if running in background
tail -f $(ls -t /tmp/run_warnings_cleanup_*_wrapper.out | head -1)

# After completion, review changes
git diff tests/helpers/expected_warns.py

# 3. Commit with descriptive message
git add tests/helpers/expected_warns.py
git commit -m "chore: Comment out obsolete warnings (pre-v2.5.0)"

# 4. Push and wait for CI
git push

# 5. After successful CI, delete commented warnings
# (Edit expected_warns.py manually or run cleanup again in next release)
```

### Development Iteration

```bash
# Quick check without rebuilding envs or running special tests, for updating scripts/basic logic, won't work with warnings-specific analysis since we don't run special tests
./scripts/run_warnings_cleanup.sh --verify --no-rebuild --skip-special
```

### Understanding What Would Be Pruned

```bash
# Step 1: Run verify to generate fresh test lo (uses existing logs if available)
./scripts/run_warnings_cleanup.sh --full --dry-run

# Check the generated logs
ls -lt /tmp/review_warn_*.out
cat /tmp/warnings_cleanup_report_*.json | jq .

# Review unmatched warnings
jq '.results[].unmatched_warnings[]' /tmp/warnings_cleanup_report_*.json

# Run in background to avoid blocking terminal
./scripts/manage_standalone_processes.sh --use-nohup ./scripts/run_warnings_cleanup.sh --verify
# - Show which warnings would be commented out
# - NOT modify any files
```

### Debugging Warning Issues

```bash
# Run in dry-run mode to see what would happen
./scripts/run_warnings_cleanup.sh --full --dry-run

# Check the generated logs
ls -lt /tmp/review_warn_*.out
cat /tmp/warnings_cleanup_report_*.json | jq .

# Review unmatched warnings
jq '.results[].unmatched_warnings[]' /tmp/warnings_cleanup_report_*.json
```

## Understanding Warning Removal Behavior

### Conservative Approach (IMPORTANT)

The script uses a **conservative strategy** to prevent incorrectly removing warnings:

**Only comments out warnings that are:**

- Absent from **ALL** test contexts (nonexample_latest, nonexample_oldest, example_latest)
- Consistently not appearing across all test scenarios

**Preserves warnings that:**

- Appear in **ANY** test context
- Are intermittent or timing-dependent
- Only appear in specific scenarios (e.g., standalone/distributed tests, FSDP multi-GPU tests)
- Are hardware-specific (e.g., CUDA vs CPU)

**Example:** If a warning appears in `nonexample_latest` but not in `nonexample_oldest`, it will be **kept** because it's needed in at least one context.

### Why This Matters

Warnings can be:

1. **Context-specific**: Only appear in distributed tests (e.g., `torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch`)
1. **Intermittent**: Appear based on timing, initialization order, or environmental factors
1. **Hardware-dependent**: Only appear with certain GPU configurations
1. **Test-specific**: Only triggered by certain test scenarios

**Result:** If you see warnings in the "unmatched in SOME contexts" log message, this is **expected and correct**. Manual review is needed only if the same warnings persist across multiple cleanup runs.

### Manual Review Recommended When

- The same warnings are listed as "unmatched in SOME contexts" across 3+ cleanup runs
- Warnings have comments indicating they were "temporarily required" months/years ago
- PyTorch/Lightning versions have been upgraded significantly since the warning was added

### After Each Minor Release

1. Run `./scripts/run_warnings_cleanup.sh --full`
1. Commit updated `expected_warns.py` (commented warnings)
1. After successful CI run, delete commented warnings:
   ```python
   # In expected_warns.py, delete lines like:
   # "old warning",  # TODO: Remove after successful CI run
   ```

### When Dependencies Update

Major dependency updates (PyTorch, Lightning) may introduce new warnings:

```bash
# 1. Update dependencies
# 2. Run cleanup to detect new warnings
./scripts/run_warnings_cleanup.sh --full

# 3. Review new warnings in report
cat /tmp/warnings_cleanup_report_*.json | jq '.results[].new_warnings'

# 4. Add new expected warnings to expected_warns.py if legitimate
# 5. Re-run cleanup to verify
./scripts/run_warnings_cleanup.sh --verify
```

### Updating the Automation

When modifying the automation scripts:

1. Test locally first with `--dry-run`
1. Test with `--verify` mode
1. Test with `--full` mode on a test branch
1. Update documentation if new flags/features added

## Troubleshooting

### "Environment not found"

```bash
# Rebuild environments manually
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest
./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_oldest --oldest
```

### "Warnings still unmatched after update"

This indicates the warning text in `expected_warns.py` doesn't exactly match the log output:

```bash
# 1. Check the exact warning text in logs
grep -i "warning pattern" /tmp/review_warn_*.out

# 2. Update expected_warns.py with exact text
# 3. Re-run verification
./scripts/run_warnings_cleanup.sh --verify
```
