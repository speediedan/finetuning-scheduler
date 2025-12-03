#!/bin/bash
# Simple wrapper around uv pip compile for CI requirements locking
# This replaces complex requirements file chains with a straightforward uv-based approach
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CI_DIR="${REPO_ROOT}/requirements/ci"

# Ensure output directory exists
mkdir -p "${CI_DIR}"

echo "Generating locked CI requirements from pyproject.toml..."

# uv pip compile can read directly from pyproject.toml
# We include:
# - Base dependencies (always included from [project.dependencies])
# - Optional dependencies: examples, cli, extra, ipynb
# - Dependency groups: test, dev
#
# Note: git-deps group is excluded from locking because git URLs cannot be
# included in universal lock files. It will be installed separately.
#
# The --resolution flag can be used to control version resolution:
# - "highest" (default): Use newest compatible versions
# - "lowest": Use oldest compatible versions (useful for testing minimum version support)

RESOLUTION="${RESOLUTION:-highest}"

uv pip compile \
    "${REPO_ROOT}/pyproject.toml" \
    --extra examples \
    --extra cli \
    --extra extra \
    --extra ipynb \
    --group dev \
    --group test \
    --output-file "${CI_DIR}/requirements.txt" \
    --upgrade \
    --no-strip-extras \
    --resolution "${RESOLUTION}" \
    --universal

echo "✓ Generated ${CI_DIR}/requirements.txt (resolution: ${RESOLUTION})"
echo ""
echo "Note: git-deps group (Lightning git URL dependencies) is installed separately in CI"
echo ""
echo "To generate oldest version requirements for testing, run:"
echo "  RESOLUTION=lowest ${SCRIPT_DIR}/lock_ci_requirements.sh"
