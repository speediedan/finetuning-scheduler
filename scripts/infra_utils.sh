#!/bin/bash
# infra utility functions

show_elapsed_time(){
  local test_log="$1"
  script_name=${2:-$(basename "$0")}
  ## write elapsed time in user-friendly fashion
  end_time=$(date +%s)
  elapsed_seconds=$(($end_time-$start_time))
  if (( $elapsed_seconds/60 == 0 )); then
      printf "${script_name} completed in $elapsed_seconds seconds \n" | tee -a $test_log
  elif (( $elapsed_seconds%60 == 0 )); then
      printf  "${script_name} completed in $(($elapsed_seconds/60)) minutes \n" | tee -a $test_log
  else
      printf "${script_name} completed in $(($elapsed_seconds/60)) minutes and $(($elapsed_seconds%60)) seconds \n" | tee -a $test_log
  fi
  printf "\n" | tee -a $test_log
}

# Function to safely deactivate a virtual environment if one is active
maybe_deactivate(){
    deactivate 2>/dev/null || true
}

# Function to strip leading and trailing quotes from a string
# Usage: cleaned=$(strip_quotes "$variable")
strip_quotes(){
    local val="$1"
    # Remove both single and double quotes from start/end
    val="${val%\' }"; val="${val#\' }"
    val="${val%\"}"; val="${val#\"}"
    echo "$val"
}

# Function to expand tilde in paths
# Usage: expanded_path=$(expand_tilde "~/some/path")
expand_tilde(){
    local p="$1"
    if [[ -n "$p" ]] && [[ "$p" == ~* ]]; then
        # Use eval to expand ~ reliably
        eval echo "$p"
    else
        echo "$p"
    fi
}

# Determine venv path based on priority: --venv-dir > FTS_VENV_BASE > default ~/.venvs
# Usage: venv_path=$(determine_venv_path "$venv_dir" "$target_env_name")
# Arguments:
#   $1: venv_dir (optional) - explicit base directory from --venv-dir flag
#   $2: target_env_name (required) - environment name to append to base
# Returns: Full venv path (e.g., /mnt/cache/user/.venvs/fts_latest)
determine_venv_path(){
    local venv_dir="$1"
    local target_env_name="$2"
    local venv_base

    if [[ -z "${target_env_name}" ]]; then
        echo "Error: target_env_name is required" >&2
        return 1
    fi

    if [[ -n "${venv_dir}" ]]; then
        # Explicit --venv-dir provided (base directory)
        venv_base="${venv_dir}"
    elif [[ -n "${FTS_VENV_BASE}" ]]; then
        # Use FTS_VENV_BASE environment variable
        venv_base="${FTS_VENV_BASE}"
    else
        # Use default
        venv_base="~/.venvs"
    fi

    # Expand tilde if present
    venv_base=$(expand_tilde "${venv_base}")

    # Return full path
    echo "${venv_base}/${target_env_name}"
}

# Split a colon-delimited string while respecting quoted segments.
# Colons that appear inside single or double quotes are treated as literal characters.
# Usage:
#   local fields=()
#   split_colon_fields "pkg:path:extra:VAR='json:with:colons'" fields
split_colon_fields(){
    local input="$1"
    local -n result=$2
    local current=""
    local in_single=0
    local in_double=0
    local char

    while IFS= read -r -n1 char || [[ -n $char ]]; do
        if [[ $char == "'" && $in_double -eq 0 ]]; then
            ((in_single ^= 1))
            current+="$char"
            continue
        fi
        if [[ $char == '"' && $in_single -eq 0 ]]; then
            ((in_double ^= 1))
            current+="$char"
            continue
        fi
        if [[ $char == ':' && $in_single -eq 0 && $in_double -eq 0 ]]; then
            result+=("$current")
            current=""
        else
            current+="$char"
        fi
    done <<< "$input"

    result+=("$current")
}

# Parse from-source specifications into an associative array
# Format: package:path[:extras][:env_var=value...]
# Usage: parse_from_source_specs "spec1;spec2;..." from_source_packages_array_name
# Example:
#   declare -A from_source_packages
#   parse_from_source_specs "$from_source_spec" from_source_packages
parse_from_source_specs(){
    local from_source_spec="$1"
    local -n pkg_array=$2  # nameref to associative array

    if [[ -z ${from_source_spec} ]]; then
        return 0
    fi

    IFS=';' read -ra PAIRS <<< "${from_source_spec}"
    for pair in "${PAIRS[@]}"; do
        # Split on colons to get all fields, but respect quoted values
        local FIELDS=()
        split_colon_fields "${pair}" FIELDS

        if [[ ${#FIELDS[@]} -lt 2 ]]; then
            echo "Error: Invalid from-source format: '$pair'" >&2
            echo "Expected format: package:path[:extras][:env_var=value...]" >&2
            return 1
        fi

        local pkg_name="$(strip_quotes "${FIELDS[0]}")"
        local pkg_path="$(strip_quotes "${FIELDS[1]}")"
        local pkg_extras=""
        local pkg_env_vars=""

        # Process remaining fields - first non-env field is extras, rest are env vars
        for ((i=2; i<${#FIELDS[@]}; i++)); do
            local field="${FIELDS[i]}"
            local stripped_field="$(strip_quotes "${field}")"
            if [[ $stripped_field =~ ^[A-Z_][A-Z0-9_]*=.*$ ]]; then
                # This is an env var (contains =)
                if [[ -n ${pkg_env_vars} ]]; then
                    pkg_env_vars="${pkg_env_vars}|${field}"
                else
                    pkg_env_vars="${field}"
                fi
            elif [[ -z ${pkg_extras} && -n ${stripped_field} ]]; then
                # First non-env, non-empty field is extras
                pkg_extras="${stripped_field}"
            fi
        done

        # Normalize package name (convert underscores to hyphens for consistency)
        pkg_name="${pkg_name//_/-}"
        # Store as "path|extras|env_vars" so we can split later
        pkg_array[$pkg_name]="${pkg_path}|${pkg_extras}|${pkg_env_vars}"
    done
}

# Install packages from source with optional extras and environment variables
# Installs packages with all their dependencies - UV's git dependency caching ensures
# that commit-pinned dependencies (e.g., Lightning pinned by finetuning-scheduler) will
# be respected by subsequent installations.
# Usage: install_from_source_packages from_source_packages_array_name venv_path [uv_install_flags]
# Example:
#   declare -A from_source_packages
#   parse_from_source_specs "$specs" from_source_packages
#   install_from_source_packages from_source_packages "/path/to/venv" "--no-cache"
# Note: The venv_path parameter should be the full path to the venv (not just the name)
install_from_source_packages(){
    local -n pkg_array=$1  # nameref to associative array
    local venv_path="$2"
    local uv_install_flags="${3:-}"

    if [[ ${#pkg_array[@]} -eq 0 ]]; then
        return 0
    fi

    source "${venv_path}/bin/activate"

    # Install packages from source if requested (do this before main package to avoid conflicts)
    for pkg in "${!pkg_array[@]}"; do
        local pkg_spec="${pkg_array[$pkg]}"
        # Split on pipe delimiter to get path, extras, and env vars
        IFS='|' read -r pkg_path pkg_extras pkg_env_vars <<< "${pkg_spec}"

        # Uninstall any existing installations (try both package name formats)
        local pkg_underscore="${pkg//-/_}"
        uv pip uninstall -y "${pkg}" "${pkg_underscore}" 2>/dev/null || true

        cd "${pkg_path}"

        # Build install target with optional extras
        if [[ -n ${pkg_extras} ]]; then
            local install_target=".[${pkg_extras}]"
            echo "Installing ${pkg} from source at ${pkg_path} with extras: [${pkg_extras}]"
        else
            local install_target="."
            echo "Installing ${pkg} from source at ${pkg_path} (no extras)"
        fi

        # Set environment variables if specified
        local env_vars_set=()
        if [[ -n ${pkg_env_vars} ]]; then
            echo "Setting environment variables for ${pkg} installation:"
            IFS='|' read -ra ENV_VARS <<< "${pkg_env_vars}"
            for env_var in "${ENV_VARS[@]}"; do
                if [[ $env_var =~ ^([^=]+)=(.*)$ ]]; then
                    local var_name="${BASH_REMATCH[1]}"
                    local var_value="${BASH_REMATCH[2]}"
                    echo "  export ${var_name}=${var_value}"
                    export "${var_name}=${var_value}"
                    env_vars_set+=("${var_name}")
                fi
            done
        fi

        uv pip install ${uv_install_flags} -e "${install_target}"

        # Unset environment variables after installation
        if [[ ${#env_vars_set[@]} -gt 0 ]]; then
            echo "Unsetting temporary environment variables for ${pkg}:"
            for var_name in "${env_vars_set[@]}"; do
                echo "  unset ${var_name}"
                unset "${var_name}"
            done
        fi

        # Verify editable installation
        echo "Verifying ${pkg} installation..."
        if uv pip show "${pkg_underscore}" 2>/dev/null | grep -q "Editable project location:"; then
            echo "✓ ${pkg} is installed in editable mode"
        elif uv pip show "${pkg}" 2>/dev/null | grep -q "Editable project location:"; then
            echo "✓ ${pkg} is installed in editable mode"
        else
            echo "⚠ Warning: ${pkg} may not be installed in editable mode"
        fi
    done
}

# Read torch prerelease configuration from requirements/ci/torch-pre.txt
# Returns values via global variables:
#   TORCH_PRE_VERSION - torch version to install
#   TORCH_PRE_CUDA    - CUDA target for local builds
#   TORCH_PRE_CHANNEL - channel type: "test" or "nightly"
# Note: Requires REPO_ROOT to be set (usually via SCRIPT_DIR)
read_torch_pre_config() {
    TORCH_PRE_VERSION=""
    TORCH_PRE_CUDA=""
    TORCH_PRE_CHANNEL=""

    # Determine repo root from SCRIPT_DIR if not already set
    local repo_root="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
    local pre_file="${repo_root}/requirements/ci/torch-pre.txt"

    if [[ ! -f "${pre_file}" ]]; then
        return
    fi

    # Read non-comment, non-empty lines
    local lines=($(grep -v '^#' "${pre_file}" | grep -v '^$' || true))

    if [[ ${#lines[@]} -ge 3 ]]; then
        TORCH_PRE_VERSION="${lines[0]}"
        TORCH_PRE_CUDA="${lines[1]}"
        TORCH_PRE_CHANNEL="${lines[2]}"

        # Validate channel
        if [[ "${TORCH_PRE_CHANNEL}" != "test" && "${TORCH_PRE_CHANNEL}" != "nightly" ]]; then
            echo "ERROR: Invalid channel '${TORCH_PRE_CHANNEL}' in ${pre_file}" >&2
            echo "Must be 'test' or 'nightly'" >&2
            return 1
        fi
    fi
}

# Get torch index URL based on channel and CUDA target
# Args: $1 = channel ("test" or "nightly"), $2 = cuda_target (e.g., "cu128" or "cpu")
# Returns: PyTorch wheel index URL
get_torch_index_url() {
    local channel="$1"
    local cuda_target="${2:-cpu}"

    if [[ "${channel}" == "test" ]]; then
        echo "https://download.pytorch.org/whl/test/${cuda_target}"
    elif [[ "${channel}" == "nightly" ]]; then
        echo "https://download.pytorch.org/whl/nightly/${cuda_target}"
    else
        echo "ERROR: Invalid channel: ${channel}" >&2
        return 1
    fi
}
