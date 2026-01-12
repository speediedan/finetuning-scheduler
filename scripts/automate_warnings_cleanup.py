#!/usr/bin/env python3
"""Automated Warnings Cleanup Orchestration Script.

This script automates the warnings cleanup process for finetuning-scheduler.
Run locally during development or before releases to maintain expected warnings inventory.

Usage:
    # Full cleanup (rebuilds envs, runs all tests, updates expected_warns.py)
    python scripts/automate_warnings_cleanup.py --mode full

    # Verification only (no changes to expected_warns.py)
    python scripts/automate_warnings_cleanup.py --mode verify

    # Dry run (show what would be done)
    python scripts/automate_warnings_cleanup.py --mode full --dry-run

    # Interactive mode (prompts before changes)
    python scripts/automate_warnings_cleanup.py --mode full --interactive

Example workflows:
    # Pre-release cleanup
    python scripts/automate_warnings_cleanup.py --mode full

    # Quick verification during development
    python scripts/automate_warnings_cleanup.py --mode verify --no-rebuild --skip-special-tests
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class WarningCleanupConfig:
    """Configuration for warnings cleanup automation."""

    repo_home: Path
    venv_base: Path
    target_env_latest: str = "fts_latest"
    target_env_oldest: str = "fts_oldest"
    mode: str = "verify"  # full, verify
    dry_run: bool = False
    interactive: bool = False
    rebuild_envs: bool = True
    skip_special_tests: bool = False
    working_dir: Path = field(default_factory=lambda: Path("/tmp"))
    log_dir: Path | None = None  # Timestamped directory for all logs

    def __post_init__(self):
        self.repo_home = Path(self.repo_home).resolve()
        self.venv_base = Path(self.venv_base).resolve()
        self.working_dir = Path(self.working_dir).resolve()

        # Create timestamped log directory if not provided
        if self.log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = self.working_dir / f"warnings_cleanup_{timestamp}"
        else:
            self.log_dir = Path(self.log_dir).resolve()

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.scripts_dir = self.repo_home / "scripts"
        self.tests_dir = self.repo_home / "tests"
        self.expected_warns_file = self.tests_dir / "helpers" / "expected_warns.py"


@dataclass
class WarningAnalysisResult:
    """Results from warning analysis."""

    total_expected: int
    total_found: int
    unmatched_warnings: list[str] = field(default_factory=list)
    new_warnings: list[str] = field(default_factory=list)
    test_context: str = "unknown"
    log_file: Path | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def has_issues(self) -> bool:
        # Only new warnings are issues; unmatched warnings are expected with conservative cleanup
        return len(self.new_warnings) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_expected": self.total_expected,
            "total_found": self.total_found,
            "unmatched_warnings": self.unmatched_warnings,
            "new_warnings": self.new_warnings,
            "test_context": self.test_context,
            "log_file": str(self.log_file) if self.log_file else None,
            "timestamp": self.timestamp,
        }


class WarningsCleanupOrchestrator:
    """Orchestrates the automated warnings cleanup process."""

    def __init__(self, config: WarningCleanupConfig):
        self.config = config
        self.results: dict[str, WarningAnalysisResult] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    def run(self) -> int:
        """Execute the warnings cleanup workflow.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        logger.info(f"Starting warnings cleanup in {self.config.mode} mode")
        logger.info(f"Repository: {self.config.repo_home}")
        logger.info(f"Dry run: {self.config.dry_run}")

        try:
            # Special case: validate-only mode skips to coverage validation
            if self.config.mode == "validate-only":
                logger.info("=" * 80)
                logger.info("VALIDATE-ONLY MODE")
                logger.info("=" * 80)
                logger.info("Skipping directly to coverage validation step")
                logger.info("This assumes expected_warns.py has already been updated")
                logger.info("")

                validation_success = self._run_coverage_validation()
                if not validation_success:
                    logger.warning("⚠️  Coverage validation detected failures")
                    logger.warning("Review the warnings and re-run cleanup if needed")
                    return 1
                else:
                    logger.info("✓ Coverage validation passed!")
                    return 0

            # Step 1: Environment setup
            if self.config.rebuild_envs and not self._setup_environments():
                logger.error("Environment setup failed")
                return 1

            # Step 2: Toggle pytest capture mode
            if not self._toggle_pytest_capture(enable=True):
                logger.error("Failed to enable pytest capture mode")
                return 1

            # Step 3: Run tests and collect warnings
            if not self._run_warning_collection():
                logger.error("Warning collection failed")
                return 1

            # Step 4: Analyze warnings
            if not self._analyze_warnings():
                logger.error("Warning analysis failed")
                return 1

            # Step 5: Update expected_warns.py (if mode allows)
            if self.config.mode == "full" and not self.config.dry_run:
                if not self._update_expected_warnings():
                    logger.error("Failed to update expected warnings")
                    return 1

                # Step 6: Verify updates
                if not self._verify_updates():
                    logger.error("Verification of updates failed")
                    return 1

                # Step 7: Run coverage validation (optional but recommended)
                logger.info("=" * 80)
                logger.info("COVERAGE VALIDATION (Optional)")
                logger.info("=" * 80)
                logger.info("This step runs full coverage with all examples to verify")
                logger.info("that commented warnings don't cause test failures.")
                logger.info("This may take 30-40 minutes.")
                logger.info("")

                if self.config.interactive:
                    response = input("Run coverage validation? [y/N]: ")
                    if response.lower() != 'y':
                        logger.info("Skipping coverage validation (can run manually later)")
                        run_validation = False
                    else:
                        run_validation = True
                else:
                    # Auto-run in non-interactive mode
                    run_validation = True

                if run_validation:
                    validation_success = self._run_coverage_validation()
                    if not validation_success:
                        logger.warning("⚠️  Coverage validation detected failures")
                        logger.warning("Review the warnings and re-run cleanup if needed")
                        # Don't fail the overall process, but warn user
                    else:
                        logger.info("✓ Coverage validation passed!")

            # Step 8: Generate report
            self._generate_report()

            # Step 9: Restore pytest config
            self._toggle_pytest_capture(enable=False)

            logger.info("✓ Warnings cleanup completed successfully")
            return 0

        except Exception as e:
            logger.exception(f"Unexpected error during warnings cleanup: {e}")
            return 1
        finally:
            # Always restore pytest config
            self._toggle_pytest_capture(enable=False)

    def _setup_environments(self) -> bool:
        """Build or rebuild test environments."""
        logger.info("Setting up test environments...")

        build_script = self.config.scripts_dir / "build_fts_env.sh"
        if not build_script.exists():
            logger.error(f"Build script not found: {build_script}")
            return False

        envs_to_build = [
            (self.config.target_env_latest, []),
            (self.config.target_env_oldest, ["--oldest"]),
        ]

        for env_name, extra_args in envs_to_build:
            logger.info(f"Building environment: {env_name}")
            cmd = [
                str(build_script),
                f"--repo_home={self.config.repo_home}",
                f"--target_env_name={env_name}",
                f"--venv-dir={self.config.venv_base}",
                *extra_args,
            ]

            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
                continue

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=self.config.repo_home,
                )
                logger.info(f"Environment {env_name} built successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to build {env_name}: {e.stderr}")
                return False

        return True

    def _toggle_pytest_capture(self, enable: bool) -> bool:
        """Toggle pytest capture mode in pyproject.toml."""
        logger.info(f"{'Enabling' if enable else 'Disabling'} pytest capture mode...")

        pyproject_file = self.config.repo_home / "pyproject.toml"
        if not pyproject_file.exists():
            logger.error(f"pyproject.toml not found: {pyproject_file}")
            return False

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would toggle pytest capture: {enable}")
            return True

        # Read current config
        content = pyproject_file.read_text()
        backup_file = pyproject_file.with_suffix(".toml.bak")
        backup_file.write_text(content)

        # Toggle capture mode
        if enable:
            # Add --capture=no
            pattern = r'(addopts = """[^"]*--disable-pytest-warnings)'
            replacement = r'\1 --capture=no'
        else:
            # Remove --capture=no
            pattern = r'(addopts = """[^"]*--disable-pytest-warnings)\s*--capture=no'
            replacement = r'\1'

        new_content = re.sub(pattern, replacement, content)

        if new_content == content:
            logger.warning("No changes made to pytest config (pattern not found)")

        pyproject_file.write_text(new_content)
        logger.info(f"Pytest capture mode {'enabled' if enable else 'disabled'}")
        return True

    def _run_warning_collection(self) -> bool:
        """Run all tests and collect warnings."""
        # In dry-run mode, check if we can use existing logs instead of running tests
        if self.config.dry_run:
            logger.info("Dry-run mode: Checking for existing test logs...")
            if self._can_use_existing_logs():
                logger.info("Found existing logs, will use them for dry-run analysis")
                return True
            else:
                logger.warning("No existing logs found for dry-run mode")
                logger.warning("In dry-run mode, we skip actual test execution")
                logger.warning("To see what warnings would be pruned, run in verify mode first to generate logs")
                return True  # Continue anyway to demonstrate the workflow

        logger.info("Running tests to collect warnings...")

        # Set environment variable for warning details
        env = os.environ.copy()
        env["FTS_WARN_DETAILS"] = "1"

        test_configs = [
            {
                "name": "nonexample_latest",
                "env": self.config.target_env_latest,
                "test_args": ["src/finetuning_scheduler", "tests"],
                "include_special": not self.config.skip_special_tests,
            },
            {
                "name": "nonexample_oldest",
                "env": self.config.target_env_oldest,
                "test_args": ["src/finetuning_scheduler", "tests"],
                "include_special": False,  # Only run special tests once
            },
            {
                "name": "example_latest",
                "env": self.config.target_env_latest,
                "test_args": ["src/fts_examples"],
                "pytest_flags": [
                    "--maxfail=1",
                    "--durations=0",
                    "-W",
                    "ignore:`np.object`:DeprecationWarning",
                    "-W",
                    "ignore:'`np.int` is':DeprecationWarning",
                ],
                "include_special": not self.config.skip_special_tests,
                "special_filter": "test_examples",
                "special_collect_dir": "src/fts_examples",
            },
        ]

        for test_config in test_configs:
            if not self._run_test_suite(test_config, env):
                logger.error(f"Test suite failed: {test_config['name']}")
                return False

        return True

    def _can_use_existing_logs(self) -> bool:
        """Check if existing test logs are available for dry-run analysis."""
        # Look for recent log files in working directory
        log_patterns = [
            "review_warn_nonexample_latest_*.out",
            "review_warn_nonexample_oldest_*.out",
            "review_warn_example_latest_*.out",
        ]

        found_logs = []
        for pattern in log_patterns:
            matches = sorted(self.config.working_dir.glob(pattern))
            if matches:
                latest = matches[-1]
                found_logs.append(latest)
                logger.info(f"  Found existing log: {latest.name}")

        # If we found at least one log file, we can use them
        if found_logs:
            # Populate results with existing log files
            for log_file in found_logs:
                # Parse context name from filename
                name = log_file.stem.replace("review_warn_", "").rsplit("_", 1)[0]
                self.results[name] = WarningAnalysisResult(
                    total_expected=0,
                    total_found=0,
                    test_context=name,
                    log_file=log_file,
                )
            return True

        return False

    def _run_test_suite(self, test_config: dict[str, Any], env: dict[str, str]) -> bool:
        """Run a specific test suite and capture output."""
        name = test_config["name"]
        env_name = test_config["env"]
        log_file = self.config.log_dir / f"review_warn_{name}_{self.timestamp}.out"

        logger.info(f"Running test suite: {name} (env: {env_name})")

        # Activate environment and run pytest
        venv_activate = self.config.venv_base / env_name / "bin" / "activate"
        if not venv_activate.exists() and not self.config.dry_run:
            logger.error(f"Environment activation script not found: {venv_activate}")
            return False
        elif not venv_activate.exists() and self.config.dry_run:
            logger.warning(f"[DRY RUN] Environment not found (would be created): {venv_activate}")

        # Build pytest command
        pytest_cmd = ["python", "-m", "pytest", *test_config["test_args"], "-v"]
        if "pytest_flags" in test_config:
            pytest_cmd.extend(test_config["pytest_flags"])

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would run: source {venv_activate} && {' '.join(pytest_cmd)}")
            # In dry-run mode, we don't create dummy logs - we'll use existing ones if available
            # Store result reference but without log file
            self.results[name] = WarningAnalysisResult(
                total_expected=0,
                total_found=0,
                test_context=name,
                log_file=None,
            )
            return True

        # Execute pytest
        full_cmd = f"source {venv_activate} && {' '.join(pytest_cmd)}"
        try:
            with open(log_file, "w") as f:
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    env=env,
                    cwd=self.config.repo_home,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    executable="/bin/bash",
                )
            logger.info(f"Test suite {name} completed (exit code: {result.returncode})")
        except Exception as e:
            logger.error(f"Failed to run test suite {name}: {e}")
            return False

        # Run special tests if configured
        if test_config.get("include_special"):
            if not self._run_special_tests(name, env_name, log_file, test_config, env):
                logger.warning(f"Special tests failed for {name} (continuing anyway)")

        # Store log file reference
        self.results[name] = WarningAnalysisResult(
            total_expected=0,
            total_found=0,
            test_context=name,
            log_file=log_file,
        )

        return True

    def _run_special_tests(
        self,
        suite_name: str,
        env_name: str,
        log_file: Path,
        test_config: dict[str, Any],
        env: dict[str, str],
    ) -> bool:
        """Run special/standalone tests."""
        logger.info(f"Running special tests for {suite_name}...")

        special_script = self.config.tests_dir / "special_tests.sh"
        if not special_script.exists():
            logger.warning(f"Special tests script not found: {special_script}")
            return False

        venv_activate = self.config.venv_base / env_name / "bin" / "activate"

        cmd_parts = ["source", str(venv_activate), "&&", str(special_script), "--mark_type=standalone",
                     f"--log-dir={self.config.log_dir}"]

        if "special_filter" in test_config:
            cmd_parts.extend(["--filter_pattern", test_config["special_filter"]])
        if "special_collect_dir" in test_config:
            cmd_parts.extend(["--collect_dir", test_config["special_collect_dir"]])

        full_cmd = " ".join(cmd_parts)

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would run special tests: {full_cmd}")
            return True

        try:
            with open(log_file, "a") as f:
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    env=env,
                    cwd=self.config.repo_home,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    executable="/bin/bash",
                )

            # Append special tests raw log from log_dir
            raw_logs = sorted(self.config.log_dir.glob("special_tests_raw_standalone_*.log"))
            if raw_logs:
                latest_raw = raw_logs[-1]
                with open(log_file, "a") as f, open(latest_raw) as raw:
                    f.write("\n--- Special Tests Raw Log ---\n")
                    f.write(raw.read())

            logger.info(f"Special tests completed (exit code: {result.returncode})")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to run special tests: {e}")
            return False

    def _analyze_warnings(self) -> bool:
        """Analyze warnings from test logs."""
        logger.info("Analyzing warnings...")

        # Generate current expected warnings lists
        if not self._generate_expected_warnings_lists():
            return False

        # Analyze each test context
        for context_name, result in self.results.items():
            if result.log_file is None:
                if self.config.dry_run:
                    logger.info(f"[DRY RUN] Skipping analysis for {context_name} (no log file)")
                    continue
                else:
                    logger.warning(f"Skipping {context_name} - no log file available")
                    continue

            is_example = "example" in context_name
            expected_warns_file = (
                self.config.log_dir / f"current_{'example' if is_example else 'nonexample'}_warns.out"
            )

            if not expected_warns_file.exists():
                if self.config.dry_run:
                    logger.warning(f"[DRY RUN] Expected warnings file not found: {expected_warns_file}")
                    continue
                logger.error(f"Expected warnings file not found: {expected_warns_file}")
                return False

            # Run warning comparison
            analysis = self._compare_warnings(expected_warns_file, result.log_file)
            result.total_expected = analysis["total_expected"]
            result.total_found = analysis["total_found"]
            result.unmatched_warnings = analysis["unmatched"]
            result.new_warnings = analysis.get("new", [])

            logger.info(
                f"Context {context_name}: {result.total_found}/{result.total_expected} warnings found, "
                f"{len(result.unmatched_warnings)} unmatched"
            )

        return True

    def _generate_expected_warnings_lists(self) -> bool:
        """Generate current expected warnings lists from expected_warns.py."""
        logger.info("Generating expected warnings lists...")

        venv_activate = self.config.venv_base / self.config.target_env_latest / "bin" / "activate"

        # Check if environment exists (needed even in dry-run mode to generate warnings list)
        if not venv_activate.exists():
            if self.config.dry_run:
                logger.warning(f"[DRY RUN] Environment not found: {venv_activate}")
                logger.warning("[DRY RUN] Cannot generate expected warnings without environment")
                logger.warning("[DRY RUN] Run 'verify' or 'full' mode first to create environment")
                return True  # Don't fail in dry-run
            else:
                logger.error(f"Environment not found: {venv_activate}")
                return False

        for is_example in [False, True]:
            context = "example" if is_example else "nonexample"
            output_file = self.config.log_dir / f"current_{context}_warns.out"

            cmd = (
                f"source {venv_activate} && python -c "
                f'"from tests.helpers.expected_warns import print_warns; '
                f'print_warns(example_warns={is_example})"'
            )

            if self.config.dry_run:
                # In dry-run, actually generate the warnings list if env exists
                # This lets us show what would be pruned
                logger.info(f"[DRY RUN] Generating {context} warnings list for analysis...")

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=self.config.repo_home,
                    capture_output=True,
                    text=True,
                    executable="/bin/bash",
                )
                output_file.write_text(result.stdout)
                logger.info(f"Generated {context} warnings list: {output_file}")
            except Exception as e:
                logger.error(f"Failed to generate {context} warnings list: {e}")
                return False

        return True

    def _compare_warnings(self, expected_file: Path, log_file: Path) -> dict[str, Any]:
        """Compare expected warnings against test log."""
        logger.debug(f"Comparing {expected_file} against {log_file}")

        # Read expected warnings
        expected_warns = []
        with open(expected_file) as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('"') and line.endswith('"'):
                    expected_warns.append(line[1:-1])  # Remove quotes

        # Read log file
        log_content = log_file.read_text()

        # Find unmatched warnings
        unmatched = []
        for warn in expected_warns:
            # Escape special regex characters in warning text
            escaped_warn = re.escape(warn)
            if not re.search(escaped_warn, log_content, re.IGNORECASE):
                unmatched.append(warn)

        # TODO: Detect new warnings not in expected list (requires parsing pytest output)
        new_warnings = []

        return {
            "total_expected": len(expected_warns),
            "total_found": len(expected_warns) - len(unmatched),
            "unmatched": unmatched,
            "new": new_warnings,
        }

    def _update_expected_warnings(self) -> bool:
        """Update expected_warns.py by commenting out unmatched warnings.

        CONSERVATIVE APPROACH: Only comments out warnings that are unmatched in ALL contexts.
        This prevents incorrectly removing warnings that appear only in specific test scenarios
        (e.g., standalone/distributed tests, specific hardware, timing-dependent warnings).
        """
        logger.info("Updating expected_warns.py...")

        if self.config.dry_run:
            logger.info("[DRY RUN] Would update expected_warns.py")
            return True

        # Collect warnings unmatched in ALL contexts (intersection)
        # Start with unmatched from first context
        if not self.results:
            logger.info("No results to process")
            return True

        all_contexts_unmatched = None
        for context_name, result in self.results.items():
            context_unmatched = set(result.unmatched_warnings)
            if all_contexts_unmatched is None:
                all_contexts_unmatched = context_unmatched
            else:
                # Only keep warnings that are unmatched in THIS context too
                all_contexts_unmatched &= context_unmatched

        # Also collect warnings found in at least one context (for logging)
        found_in_some_context = set()
        for result in self.results.values():
            found_in_some_context.update(result.unmatched_warnings)
        found_in_some_context -= all_contexts_unmatched

        if not all_contexts_unmatched:
            logger.info("No warnings are unmatched in ALL contexts")
            logger.info("✓ All warnings are present in at least one context - nothing to comment out")
            if found_in_some_context:
                logger.warning(
                    f"⚠️  {len(found_in_some_context)} warning(s) are unmatched in SOME contexts. "
                    f"These are kept because they may appear in specific test scenarios:"
                )
                for warn in sorted(found_in_some_context):
                    logger.warning(f"  - {warn}")
                logger.warning("Manual review recommended if these persist across multiple cleanup runs.")
            return True

        # Prominently display warnings that will be commented out
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"⚠️  {len(all_contexts_unmatched)} warning(s) will be COMMENTED OUT")
        logger.info("(These warnings were unmatched in ALL test contexts)")
        logger.info("=" * 80)
        for warn in sorted(all_contexts_unmatched):
            logger.info(f"  - {warn}")
        logger.info("=" * 80)
        logger.info("")

        # Read expected_warns.py
        content = self.config.expected_warns_file.read_text()
        lines = content.split("\n")

        # Create backup
        backup_file = self.config.expected_warns_file.with_suffix(".py.bak")
        backup_file.write_text(content)

        # Comment out warnings unmatched in ALL contexts
        modified = False
        new_lines = []
        for line in lines:
            # Check if line contains a warning unmatched in ALL contexts
            should_comment = False
            for warn in all_contexts_unmatched:
                # Match lines like: "warning text",
                if f'"{warn}"' in line and not line.strip().startswith("#"):
                    should_comment = True
                    break

            if should_comment:
                # Comment out the line
                new_lines.append(f"    # {line.strip()}  # TODO: Safe to remove - not found in any test context")
                modified = True
                logger.info(f"Commented out warning (absent in ALL contexts): {line.strip()}")
            else:
                new_lines.append(line)

        if modified:
            self.config.expected_warns_file.write_text("\n".join(new_lines))
            logger.info(f"Updated {self.config.expected_warns_file}")
            logger.info(
                f"✓ Only warnings absent from ALL {len(self.results)} contexts were commented out. "
                f"Warnings appearing in specific scenarios were preserved."
            )
        else:
            logger.info("No changes needed to expected_warns.py")

        return True

    def _verify_updates(self) -> bool:
        """Verify that warnings cleanup was successful.

        With conservative strategy, we expect warnings to be unmatched in SOME contexts (those are the warnings that
        appear only in specific scenarios). This is correct. Only fail if we have warnings that were supposed to be
        removed but weren't.
        """
        logger.info("Verifying updates...")

        # Regenerate expected warnings lists
        if not self._generate_expected_warnings_lists():
            return False

        # Re-analyze against existing logs
        # Track warnings that are still unmatched in ALL contexts (these should have been removed)
        unmatched_in_all = None
        contexts_analyzed = 0

        for context_name, result in self.results.items():
            if result.log_file is None:
                continue

            is_example = "example" in context_name
            expected_warns_file = (
                self.config.log_dir / f"current_{'example' if is_example else 'nonexample'}_warns.out"
            )

            analysis = self._compare_warnings(expected_warns_file, result.log_file)
            contexts_analyzed += 1

            # Track unmatched warnings across all contexts
            context_unmatched = set(analysis["unmatched"])
            if unmatched_in_all is None:
                unmatched_in_all = context_unmatched
            else:
                unmatched_in_all &= context_unmatched

            # Report match status neutrally
            matched_count = analysis["total_found"]
            unmatched_count = len(analysis["unmatched"])
            logger.info(
                f"Context {context_name}: {matched_count} matched, {unmatched_count} unmatched "
                f"(out of {analysis['total_expected']} expected)"
            )
            if analysis["unmatched"]:
                for warn in analysis["unmatched"][:3]:  # Show first 3
                    logger.debug(f"  - {warn}")

        # Check if any warnings are still unmatched in ALL contexts (these should have been removed)
        if unmatched_in_all:
            logger.error(
                f"❌ {len(unmatched_in_all)} warning(s) are unmatched in ALL {contexts_analyzed} contexts. "
                f"These should have been commented out but weren't:"
            )
            for warn in unmatched_in_all:
                logger.error(f"  - {warn}")
            return False

        logger.info(
            "✓ Verification complete: Warnings appearing in specific contexts are preserved as expected. "
            "No warnings are incorrectly uncommented."
        )
        return True

    def _run_coverage_validation(self) -> bool:
        """Run full coverage with all examples to validate warnings cleanup.

        This runs gen_fts_coverage.sh with --run-all-and-examples and --allow-failures to verify that the commented-out
        warnings don't cause test failures.
        """
        logger.info("Running coverage validation to verify warnings cleanup...")
        logger.info("This may take 30-40 minutes...")

        coverage_script = self.config.repo_home / "scripts" / "gen_fts_coverage.sh"
        if not coverage_script.exists():
            logger.error(f"Coverage script not found: {coverage_script}")
            return False

        # Run coverage with all examples and allow failures
        # Note: gen_fts_coverage.sh will use FTS_VENV_BASE if venv-dir not specified
        cmd = [
            str(coverage_script),
            f"--repo-home={self.config.repo_home}",
            f"--target-env-name={self.config.target_env_latest}",
            "--run-all-and-examples",
            "--allow-failures",
            "--no-rebuild-base",
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        # Prepare environment with FTS_VENV_BASE
        import os
        env = os.environ.copy()
        env["FTS_VENV_BASE"] = str(self.config.venv_base)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.repo_home,
                capture_output=True,
                text=True,
                env=env,
            )

            if result.returncode != 0:
                logger.error("Coverage validation failed!")
                logger.error(f"Exit code: {result.returncode}")
                logger.error("Check the coverage log for details")

                # Parse for failed tests and associated warnings
                failed_tests = self._parse_failed_tests_from_coverage(result.stdout + result.stderr)
                if failed_tests:
                    logger.warning("⚠️  MANUAL ACTION REQUIRED:")
                    logger.warning("The following tests failed during coverage validation:")
                    logger.warning("These failures may be due to incorrectly commented warnings.")
                    logger.warning("")
                    for test_name, warning_context in failed_tests:
                        logger.warning(f"  - {test_name}")
                        if warning_context:
                            logger.warning(f"    Related warning: {warning_context}")
                    logger.warning("")
                    logger.warning("Please:")
                    logger.warning("1. Review tests/helpers/expected_warns.py")
                    logger.warning("2. Uncomment warnings associated with failed tests")
                    logger.warning("3. Re-run warnings cleanup verification:")
                    logger.warning("   ./scripts/run_warnings_cleanup.sh --verify")

                return False

            logger.info("✓ Coverage validation passed!")
            return True

        except Exception as e:
            logger.error(f"Error running coverage validation: {e}")
            return False

    def _parse_failed_tests_from_coverage(self, log_output: str) -> list[tuple[str, str]]:
        """Parse coverage log to find failed tests and associated warnings.

        Returns list of (test_name, warning_context) tuples.
        """
        import re

        failed_tests = []
        lines = log_output.split("\n")

        for i, line in enumerate(lines):
            # Look for FAILED test lines
            if "FAILED" in line:
                # Extract test name
                match = re.search(r"FAILED\s+(\S+)", line)
                if match:
                    test_name = match.group(1)

                    # Look for warning-related context in surrounding lines
                    warning_context = ""
                    for j in range(max(0, i-5), min(len(lines), i+5)):
                        if any(keyword in lines[j].lower() for keyword in ["warning", "unexpected", "assert"]):
                            warning_context = lines[j].strip()
                            break

                    failed_tests.append((test_name, warning_context))

        return failed_tests

    def _has_new_warnings(self) -> bool:
        """Check if any new warnings were detected."""
        for result in self.results.values():
            if result.new_warnings:
                return True
        return False

    def _generate_report(self) -> None:
        """Generate summary report of warnings cleanup."""
        logger.info("Generating report...")

        report_file = self.config.log_dir / f"warnings_cleanup_report_{self.timestamp}.json"

        report = {
            "timestamp": self.timestamp,
            "mode": self.config.mode,
            "dry_run": self.config.dry_run,
            "repo_home": str(self.config.repo_home),
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "summary": {
                "total_contexts": len(self.results),
                "contexts_with_issues": sum(1 for r in self.results.values() if r.has_issues),
                "total_unmatched": sum(len(r.unmatched_warnings) for r in self.results.values()),
                "total_new": sum(len(r.new_warnings) for r in self.results.values()),
            },
        }

        report_file.write_text(json.dumps(report, indent=2))
        logger.info(f"Report written to: {report_file}")

        # Print summary to console
        print("\n" + "=" * 80)
        print("WARNINGS CLEANUP SUMMARY")
        print("=" * 80)
        print(f"Mode: {self.config.mode}")
        print(f"Timestamp: {self.timestamp}")
        print(f"\nTotal contexts analyzed: {report['summary']['total_contexts']}")
        print(f"Contexts with issues: {report['summary']['contexts_with_issues']}")
        print(f"Total unmatched warnings: {report['summary']['total_unmatched']}")
        print(f"Total new warnings: {report['summary']['total_new']}")
        print("\nDetails by context:")
        for name, result in self.results.items():
            status = "✓" if not result.has_issues else "✗"
            print(f"  {status} {name}: {result.total_found}/{result.total_expected} matched")
            if result.unmatched_warnings:
                print(f"      Unmatched: {len(result.unmatched_warnings)}")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Automated warnings cleanup for finetuning-scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "verify", "validate-only"],
        default="verify",
        help="Operation mode: full (update files), verify (check only), validate-only (run coverage validation only)",
    )
    parser.add_argument(
        "--repo-home",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory",
    )
    parser.add_argument(
        "--venv-base",
        type=Path,
        default=Path.home() / ".venvs",
        help="Base directory for virtual environments",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip rebuilding environments",
    )
    parser.add_argument(
        "--skip-special-tests",
        action="store_true",
        help="Skip special/standalone tests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for confirmation before making changes",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("/tmp"),
        help="Directory for temporary files and logs",
    )

    args = parser.parse_args()

    # Create config
    config = WarningCleanupConfig(
        repo_home=args.repo_home,
        venv_base=args.venv_base,
        mode=args.mode,
        dry_run=args.dry_run,
        interactive=args.interactive,
        rebuild_envs=not args.no_rebuild,
        skip_special_tests=args.skip_special_tests,
        working_dir=args.working_dir,
    )

    # Run orchestrator
    orchestrator = WarningsCleanupOrchestrator(config)
    exit_code = orchestrator.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
