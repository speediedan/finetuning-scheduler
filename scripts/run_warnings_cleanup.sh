#!/bin/bash
#
# Wrapper script for automated warnings cleanup
# This script provides convenient shortcuts for common warning cleanup operations
#
# Can be run directly or via manage_standalone_processes.sh harness for background execution
#
# Usage examples:
#
#   # Direct execution - Full cleanup with environment rebuild (pre-release)
#   ./scripts/run_warnings_cleanup.sh --full
#
#   # Direct execution - Quick verification without rebuild (development iteration)
#   ./scripts/run_warnings_cleanup.sh --verify --no-rebuild
#
#   # Background execution via harness
#   ./scripts/manage_standalone_processes.sh --use-nohup ./scripts/run_warnings_cleanup.sh --full
#
#   # Dry run to see what would happen
#   ./scripts/run_warnings_cleanup.sh --full --dry-run
#
set -eo pipefail

# Default values
MODE="verify"
REPO_HOME="${REPO_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_BASE="${FTS_VENV_BASE:-${HOME}/.venvs}"
EXTRA_ARGS=()

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated warnings cleanup for finetuning-scheduler (local use)

OPTIONS:
    --full              Full cleanup mode (updates expected_warns.py)
    --verify            Verification mode only (no changes) [default]
    --validate-only     Skip to coverage validation step only
                        (assumes warnings have already been updated)

    --no-rebuild        Skip rebuilding test environments
    --skip-special      Skip standalone/special tests
    --dry-run           Show what would be done without making changes
    --interactive       Prompt before making changes

    --venv-base DIR     Base directory for virtual environments
                        (default: ${VENV_BASE})
    --working-dir DIR   Directory for logs and temp files
                        (default: /tmp)

    --help              Show this help message

EXAMPLES:
    # Pre-release cleanup (rebuilds envs, updates warnings)
    $0 --full

    # Quick verification during development
    $0 --verify --no-rebuild --skip-special

    # See what would happen without making changes
    $0 --full --dry-run

    # Run in background via manage_standalone_processes.sh
    ./scripts/manage_standalone_processes.sh --use-nohup $0 --full

ENVIRONMENT VARIABLES:
    REPO_HOME           Repository root (detected automatically)
    FTS_VENV_BASE       Base directory for virtual environments
    FTS_CLEANUP_LOG     Override log file location

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            MODE="full"
            shift
            ;;
        --verify)
            MODE="verify"
            shift
            ;;
        --validate-only)
            MODE="validate-only"
            shift
            ;;
        --no-rebuild)
            EXTRA_ARGS+=("--no-rebuild")
            shift
            ;;
        --skip-special)
            EXTRA_ARGS+=("--skip-special-tests")
            shift
            ;;
        --dry-run)
            EXTRA_ARGS+=("--dry-run")
            shift
            ;;
        --interactive)
            EXTRA_ARGS+=("--interactive")
            shift
            ;;
        --venv-base)
            VENV_BASE="$2"
            shift 2
            ;;
        --working-dir)
            EXTRA_ARGS+=("--working-dir" "$2")
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate environment
if [[ ! -d "${REPO_HOME}" ]]; then
    echo -e "${RED}Error: Repository not found: ${REPO_HOME}${NC}"
    exit 1
fi

if [[ ! -f "${REPO_HOME}/scripts/automate_warnings_cleanup.py" ]]; then
    echo -e "${RED}Error: Main script not found: ${REPO_HOME}/scripts/automate_warnings_cleanup.py${NC}"
    exit 1
fi

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found in PATH${NC}"
    exit 1
fi

# Print configuration
echo -e "${GREEN}=== Warnings Cleanup Configuration ===${NC}"
echo "Mode: ${MODE}"
echo "Repository: ${REPO_HOME}"
echo "Venv base: ${VENV_BASE}"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

# Create log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${FTS_CLEANUP_LOG:-/tmp/warnings_cleanup_${TIMESTAMP}.log}"

# Check if we're running under manage_standalone_processes.sh
# (it sets a wrapper output file)
if [[ -n "${WRAPPER_OUT}" ]]; then
    echo "Running under manage_standalone_processes.sh wrapper"
    LOG_FILE="${WRAPPER_OUT%.out}_detailed.log"
fi

echo "Log file: ${LOG_FILE}"
echo ""

# Run the Python orchestrator
echo -e "${GREEN}Starting warnings cleanup...${NC}"
python3 "${REPO_HOME}/scripts/automate_warnings_cleanup.py" \
    --mode "${MODE}" \
    --repo-home "${REPO_HOME}" \
    --venv-base "${VENV_BASE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

# Report results
echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo -e "${GREEN}✓ Warnings cleanup completed successfully${NC}"
    echo -e "Log file: ${LOG_FILE}"
else
    echo -e "${RED}✗ Warnings cleanup failed (exit code: ${EXIT_CODE})${NC}"
    echo -e "Check log file for details: ${LOG_FILE}"
fi

exit ${EXIT_CODE}
