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
# - When torch-nightly.txt is configured:
#   - Lock file is generated with torch pinned to the nightly version
#   - Uses PyTorch nightly index for resolution
#   - Docker image and CI both use the same nightly version
# - Without torch-nightly.txt:
#   - Uses stable torch from PyPI
#   - CI uses --torch-backend=cpu for CPU variant
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

# Check if torch nightly is configured
TORCH_NIGHTLY_VERSION=$(get_torch_nightly_version)

generate_lockfile() {
    local resolution=$1
    local output_file=$2
    local python_version=$3
    local use_nightly=$4  # "true" to use torch nightly

    echo "Generating ${output_file} with resolution=${resolution}, python=${python_version}..."

    # Build the base compile command
    local compile_cmd=(
        uv pip compile
        "${REPO_ROOT}/pyproject.toml"
        --extra all
        --group dev
        --group test
        --output-file "${output_file}"
        --upgrade
        --no-strip-extras
        --resolution "${resolution}"
        --universal
        --python-version "${python_version}"
    )

    # When using torch nightly:
    # 1. Create a temporary override file to pin torch to the nightly version for dependency resolution
    # 2. Use --prerelease=if-necessary-or-explicit to only allow prereleases for explicitly specified packages (torch)
    #    or where all versions of the package are pre-release
    # 3. Use --extra-index-url with nightly CPU index for torch resolution
    # 4. Use --index-strategy=unsafe-best-match to prefer PyPI for non-torch packages
    # 5. Use --no-emit-package=torch to exclude torch from output (installed separately with backend)
    if [[ "${use_nightly}" == "true" && -n "${TORCH_NIGHTLY_VERSION}" ]]; then
        local torch_override_file=$(mktemp)
        echo "torch==${TORCH_NIGHTLY_VERSION}" > "${torch_override_file}"

        compile_cmd+=(
            --prerelease=if-necessary-or-explicit
            --override "${torch_override_file}"
            --extra-index-url "https://download.pytorch.org/whl/nightly/cpu"
            --index-strategy unsafe-best-match
            --no-emit-package torch
        )

        echo "  Using torch nightly: ${TORCH_NIGHTLY_VERSION} (excluded from output, dependencies resolved)"
        "${compile_cmd[@]}"

        rm -f "${torch_override_file}"
        echo "✓ Generated ${output_file} (torch ${TORCH_NIGHTLY_VERSION} excluded, install separately)"
    else
        "${compile_cmd[@]}"
        echo "✓ Generated ${output_file}"
    fi
}

# Generate both lock files
# - Latest: Python 3.10 (minimum supported) to ensure proper version markers for
#   packages like contourpy that have Python version requirements
# - Oldest: Python 3.10 (minimum supported), lowest resolution
#
# When torch nightly is configured:
# - requirements.txt excludes torch (installed separately with appropriate backend)
# - torch dependencies are still resolved against the nightly version
# - requirements-oldest.txt uses stable torch (for minimum version testing)

USE_NIGHTLY="false"
if [[ -n "${TORCH_NIGHTLY_VERSION}" ]]; then
    USE_NIGHTLY="true"
    echo "Torch nightly mode: ${TORCH_NIGHTLY_VERSION}"
fi

generate_lockfile "highest" "${CI_DIR}/requirements.txt" "3.10" "${USE_NIGHTLY}"
generate_lockfile "lowest-direct" "${CI_DIR}/requirements-oldest.txt" "3.10" "false"

echo ""
echo "Generated lock files:"
echo "  - ${CI_DIR}/requirements.txt (highest resolution, for latest tests)"
echo "  - ${CI_DIR}/requirements-oldest.txt (lowest resolution, for oldest tests)"
echo ""
if [[ -n "${TORCH_NIGHTLY_VERSION}" ]]; then
    echo "⚠️  Torch nightly mode: ${TORCH_NIGHTLY_VERSION}"
    echo "   requirements.txt excludes torch (dependencies resolved against nightly)"
    echo "   torch must be installed separately before other dependencies"
    echo ""
    echo "Docker image installation (CUDA):"
    echo "  Ensure Dockerfile installs: torch==${TORCH_NIGHTLY_VERSION} from nightly/cu128 index"
    echo ""
    echo "Azure Pipelines (Docker with pre-installed torch):"
    echo "  UV_OVERRIDE=requirements/ci/overrides.txt uv pip install -e . -r requirements/ci/requirements.txt"
    echo ""
    echo "GitHub Actions (CPU):"
    echo "  1. uv pip install --prerelease=allow torch==${TORCH_NIGHTLY_VERSION}+cpu --index-url https://download.pytorch.org/whl/nightly/cpu"
    echo "  2. UV_OVERRIDE=requirements/ci/overrides.txt uv pip install -e . -r requirements/ci/requirements.txt"
else
    echo "Standard installation (stable torch):"
    echo "  CI:    UV_OVERRIDE=requirements/ci/overrides.txt uv pip install -e . -r requirements/ci/requirements.txt --torch-backend=cpu"
    echo "  Local: UV_OVERRIDE=requirements/ci/overrides.txt uv pip install -e . -r requirements/ci/requirements.txt --torch-backend=auto"
fi
echo ""
echo "Note: Lightning commit pin is applied via UV_OVERRIDE=requirements/ci/overrides.txt"
