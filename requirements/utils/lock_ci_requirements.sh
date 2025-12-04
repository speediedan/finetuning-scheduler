#!/bin/bash
# Simple wrapper around uv pip compile for CI requirements locking
# This replaces complex requirements file chains with a straightforward uv-based approach
#
# Generates two lock files:
# - requirements.txt: highest resolution (default, for latest tests)
# - requirements-oldest.txt: lowest resolution (for oldest version testing)
#
# Dependencies are defined in pyproject.toml with explicit minimum versions,
# which allows uv to properly resolve oldest compatible versions.
#
# Torch handling:
# - Lock files are backend-agnostic (no --torch-backend flag)
# - At install time:
#   - CI uses --torch-backend=cpu to get CPU variant
#   - Local builds use --torch-backend=auto for GPU auto-detection
#   - When torch-nightly.txt is configured: torch is pre-installed from specific
#     nightly index (cpu or cuda) and filtered from requirements file
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CI_DIR="${REPO_ROOT}/requirements/ci"
TORCH_NIGHTLY_FILE="${CI_DIR}/torch-nightly.txt"

# Ensure output directory exists
mkdir -p "${CI_DIR}"

echo "Generating locked CI requirements from pyproject.toml..."

# Check if torch nightly is configured
# Returns: torch_version if nightly is enabled, empty string otherwise
get_torch_nightly_version() {
    if [[ -f "${TORCH_NIGHTLY_FILE}" ]]; then
        # Read first non-comment, non-empty line as torch version
        local version=$(grep -v '^#' "${TORCH_NIGHTLY_FILE}" | grep -v '^$' | head -1)
        if [[ -n "${version}" ]]; then
            echo "${version}"
            return
        fi
    fi
    echo ""
}

# uv pip compile reads directly from pyproject.toml
# We include:
# - Base dependencies (always included from [project.dependencies])
# - Optional dependencies via 'all' extra
# - Dependency groups: test, dev
#
# Note: git-deps group is excluded from locking because git URLs cannot be
# included in universal lock files. It will be installed separately via UV_OVERRIDE.
#
# Note on Python version compatibility:
# - Lock files are generated for Python 3.10+ (our minimum supported version)
#
# All dependencies in pyproject.toml have explicit minimum versions (>=x.y.z),
# which ensures uv can properly resolve oldest compatible versions with
# --resolution=lowest-direct without needing external constraint files.

# Check if torch nightly is configured (for informational message)
TORCH_NIGHTLY_VERSION=$(get_torch_nightly_version)

generate_lockfile() {
    local resolution=$1
    local output_file=$2
    local python_version=$3
    local filter_torch=$4  # "true" to filter out torch from the output

    echo "Generating ${output_file} with resolution=${resolution}, python=${python_version}..."

    # Lock files are backend-agnostic - torch backend is specified at install time:
    # - CI: --torch-backend=cpu
    # - Local: --torch-backend=auto
    # - Nightly: torch pre-installed, filtered from requirements
    if [[ "${filter_torch}" == "true" ]]; then
        # Generate to temp file, then filter torch
        local temp_file=$(mktemp)
        uv pip compile \
            "${REPO_ROOT}/pyproject.toml" \
            --extra all \
            --group dev \
            --group test \
            --output-file "${temp_file}" \
            --upgrade \
            --no-strip-extras \
            --resolution "${resolution}" \
            --universal \
            --python-version "${python_version}"
        grep -v '^torch==' "${temp_file}" > "${output_file}"
        rm -f "${temp_file}"
        echo "✓ Generated ${output_file} (torch filtered)"
    else
        uv pip compile \
            "${REPO_ROOT}/pyproject.toml" \
            --extra all \
            --group dev \
            --group test \
            --output-file "${output_file}" \
            --upgrade \
            --no-strip-extras \
            --resolution "${resolution}" \
            --universal \
            --python-version "${python_version}"
        echo "✓ Generated ${output_file}"
    fi
}

# Generate both lock files
# - Latest: Python 3.10 (minimum supported) to ensure proper version markers for
#   packages like contourpy that have Python version requirements
# - Oldest: Python 3.10 (minimum supported), lowest resolution
#
# When torch nightly is configured, torch is filtered from requirements.txt
# because it's pre-installed separately from the nightly index
FILTER_TORCH="false"
if [[ -n "${TORCH_NIGHTLY_VERSION}" ]]; then
    FILTER_TORCH="true"
    echo "Torch nightly mode: ${TORCH_NIGHTLY_VERSION} - torch will be filtered from requirements.txt"
fi

generate_lockfile "highest" "${CI_DIR}/requirements.txt" "3.10" "${FILTER_TORCH}"
generate_lockfile "lowest-direct" "${CI_DIR}/requirements-oldest.txt" "3.10" "false"

echo ""
echo "Generated lock files:"
echo "  - ${CI_DIR}/requirements.txt (highest resolution, for latest tests)"
echo "  - ${CI_DIR}/requirements-oldest.txt (lowest resolution, for oldest tests)"
echo ""
if [[ -n "${TORCH_NIGHTLY_VERSION}" ]]; then
    echo "⚠️  Torch nightly mode: ${TORCH_NIGHTLY_VERSION}"
    echo "   requirements.txt has torch filtered out (pre-installed separately)"
    echo ""
    echo "Installation (CI - CPU):"
    echo "  1. Pre-install torch:  uv pip install --prerelease=allow torch==${TORCH_NIGHTLY_VERSION}+cpu --index-url https://download.pytorch.org/whl/nightly/cpu"
    echo "  2. Install deps:       UV_OVERRIDE=requirements/ci/overrides.txt uv pip install -e . -r requirements/ci/requirements.txt"
    echo ""
    echo "Installation (local - CUDA, e.g., cu128):"
    echo "  1. Pre-install torch:  uv pip install --prerelease=allow torch==${TORCH_NIGHTLY_VERSION} --index-url https://download.pytorch.org/whl/nightly/cu128"
    echo "  2. Install FTS:        UV_OVERRIDE=requirements/ci/overrides.txt uv pip install -e . -r requirements/ci/requirements.txt"
else
    echo "Standard installation (torch included in requirements.txt):"
    echo "  CI:    UV_OVERRIDE=requirements/ci/overrides.txt uv pip install -e . -r requirements/ci/requirements.txt --torch-backend=cpu"
    echo "  Local: UV_OVERRIDE=requirements/ci/overrides.txt uv pip install -e . -r requirements/ci/requirements.txt --torch-backend=auto"
fi
echo ""
echo "Note: Lightning commit pin is applied via UV_OVERRIDE=requirements/ci/overrides.txt"
