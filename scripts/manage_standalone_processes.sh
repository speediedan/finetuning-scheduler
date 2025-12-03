#!/bin/bash
#
# Generic script to manage standalone processes
# Usage examples:
#   ./manage_standalone_processes.sh /path/to/script.sh --arg1 value1 --arg2 value2
#   ./manage_standalone_processes.sh python -m pytest tests/some_test.py
#   ./manage_standalone_processes.sh --use-nohup /path/to/script.sh --arg1 value1
set -eo pipefail

# Directory of this script (so we can find the config file next to it)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Config file containing one regex (or plain string) per line to match against pgrep -f
# Lines beginning with # or blank lines are ignored.
REGEX_CFG_FILE="$SCRIPT_DIR/manage_standalone_regex.cfg"

# Default configuration
USE_NOHUP=false

# Process flags
while [[ "$1" == --* ]]; do
  case "$1" in
    --use-nohup)
      USE_NOHUP=true
      shift
      ;;
    --regex-cfg=*)
      REGEX_CFG_FILE="${1#--regex-cfg=}"
      shift
      ;;
    --regex-cfg)
      shift
      if [ -z "$1" ] || [[ "$1" == --* ]]; then
        echo "Error: --regex-cfg requires a file path argument"
        exit 2
      fi
      REGEX_CFG_FILE="$1"
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Get current date/time for log file naming
d=`date +%Y%m%d%H%M%S`

# Validate no conflicting processes are running. Reads regex patterns from
# $REGEX_CFG_FILE (one pattern per line). Lines starting with '#' or blank
# lines are ignored. The patterns are OR-joined into a single pgrep regex.
validate_process_not_running() {
  current_pid=$$

  if [ ! -f "$REGEX_CFG_FILE" ]; then
    echo "Error: regex config file not found: $REGEX_CFG_FILE"
    echo "Please create the file with one regex or plain string per line (use '#' for comments). Aborting."
    exit 2
  fi

  # Read cfg lines, ignore comments and blank lines
  mapfile -t patterns < <(grep -E -v '^\s*(#|$)' "$REGEX_CFG_FILE" || true)

  if [ ${#patterns[@]} -eq 0 ]; then
    echo "Error: No patterns found in regex config file: $REGEX_CFG_FILE"
    echo "Ensure the file contains at least one non-comment pattern. Aborting."
    exit 2
  fi

  if [ ${#patterns[@]} -eq 0 ]; then
    echo "No patterns found to check for conflicting processes; proceeding..."
    return 0
  fi

  # Join patterns with | for pgrep -f
  IFS='|'; pgrep_pattern="${patterns[*]}"; unset IFS

  if pgrep -f "$pgrep_pattern" | grep -v "^${current_pid}$" > /dev/null; then
    echo "Error: Found running processes that may conflict:"
    pgrep -fa "$pgrep_pattern" | grep -v "^${current_pid}$"
    exit 1
  fi

  echo "No conflicting processes found, proceeding..."
}

if [ -z "$1" ]; then
  echo "Usage: $0 command [arguments]"
  echo "Example: $0 ./gen_fts_coverage.sh --arg1 value1"
  echo ""
  echo "This script will check for potentially conflicting processes before running."
  echo "It reads process-match patterns from the config file located next to this script:"
  echo "  $REGEX_CFG_FILE"
  echo "The config file should contain one regex or plain string per line. Lines starting"
  echo "with '#' or blank lines are ignored. Patterns are OR-joined and used with 'pgrep -f'."
  echo ""
  echo "Options:"
  echo "  --use-nohup           Run the command under nohup (background)."
  echo "  --regex-cfg <path>    Override the default regex config file path."
  exit 1
fi

# Use all arguments, not just the first one
CMD="$@"
SCRIPT_NAME="$1"
BASE_NAME=$(basename "$SCRIPT_NAME" .sh)
WRAPPER_OUT="/tmp/${BASE_NAME}_${d}_wrapper.out"

# Validate no processes are running
validate_process_not_running

echo "Starting command: $CMD"
# Check if we should use nohup (VSCode kills nohup jobs: https://github.com/microsoft/vscode/issues/231216)
if [ "$USE_NOHUP" = true ]; then
    echo "Running in background with nohup..."
    nohup $CMD > "$WRAPPER_OUT" 2>&1 &
    echo "Wrapper process started with PID: $!" | tee -a "$WRAPPER_OUT"
    echo "Wrapper output logged to: $WRAPPER_OUT"
    echo "First 5 lines of output will be displayed in 3 seconds as a sanity check..."
    echo `printf "%0.s-" {1..120} && printf "\n"`
    sleep 3
    cat "$WRAPPER_OUT" | head -n 5
else
    echo "Running in foreground (default for VSCode compatibility)..."
    $CMD > "$WRAPPER_OUT" 2>&1
    echo "Wrapper process completed, wrapper output saved to: $WRAPPER_OUT"
fi
