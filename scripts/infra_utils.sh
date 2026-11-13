#!/usr/bin/env bash
# infra utility functions
#
# Portability note: most helpers here are POSIX/BSD-friendly (macOS bash 3.2 included),
# but `split_colon_fields`, `parse_from_source_specs`, and `install_from_source_packages`
# use namerefs/associative arrays and therefore require bash >= 4.3 (callers such as
# build_it_env.sh enforce this with an early version check).
#
# Kept byte-identical between interpretune and finetuning-scheduler (interpretune's
# scripts/check_shared_scripts_sync.py fails on drift). What differs per project lives in
# infra_utils_project.sh beside this file, sourced at the end: it names the venv-base variable
# (INFRA_VENV_BASE_VAR) and defines any helper only that project uses.


# Optional GPU-lease serialization (no-op unless GPU_LEASE_CMD is set). See scripts/gpu_lease_wrap.sh.
_gpu_lease_wrap="$(dirname "${BASH_SOURCE[0]}")/gpu_lease_wrap.sh"
# shellcheck source=gpu_lease_wrap.sh
[[ -r "${_gpu_lease_wrap}" ]] && source "${_gpu_lease_wrap}"
type maybe_gpu_lease &>/dev/null || maybe_gpu_lease() { "$@"; }
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

# Expand unquoted ~/ segments that appear at the start of a value or after
# whitespace/= so env vars like FLAGS="-r ~/requirements.txt" resolve correctly.
# Usage: expanded_value=$(expand_tilde_segments "$value")
expand_tilde_segments(){
    local value="$1"
    local home_dir="${HOME}"

    value="${value/#~\//${home_dir}/}"
    value="${value//" ~/"/" ${home_dir}/"}"
    value="${value//"=~/"/"=${home_dir}/"}"

    echo "$value"
}

# Read torch prerelease configuration from torch-pre.txt
# This function reads the torch-pre.txt file and sets global variables:
#   TORCH_PRE_VERSION: The torch version to install (e.g., "2.10.0.dev20250122")
#   TORCH_PRE_CUDA: The CUDA target (e.g., "cu128")
#   TORCH_PRE_CHANNEL: The channel type (e.g., "nightly" or "test")
# If torch-pre.txt doesn't exist or is empty, these variables will be empty. A channel other
# than "nightly", "test" or "stable" is refused (return 1), naming the file.
# Usage:
#   read_torch_pre_config ["/path/to/torch-pre.txt"]   # default: requirements/ci/torch-pre.txt
#   if [[ -n "${TORCH_PRE_VERSION}" ]]; then
#       echo "Using torch prerelease: ${TORCH_PRE_VERSION}"
#   fi
read_torch_pre_config(){
    local torch_pre_file="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/requirements/ci/torch-pre.txt}"
    TORCH_PRE_VERSION=""
    TORCH_PRE_CUDA=""
    TORCH_PRE_CHANNEL=""

    if [[ -f "${torch_pre_file}" ]]; then
        # Read 3-line config: version, CUDA target, channel type
        # (while-read loop instead of readarray/mapfile for macOS bash 3.x compatibility)
        local _line _idx=0
        while IFS= read -r _line || [[ -n "${_line}" ]]; do
            case ${_idx} in
                0) TORCH_PRE_VERSION="${_line}" ;;
                1) TORCH_PRE_CUDA="${_line}" ;;
                2) TORCH_PRE_CHANNEL="${_line}" ;;
            esac
            _idx=$((_idx + 1))
        done < <(grep -v '^#' "${torch_pre_file}" | grep -v '^$')
    fi

    case "${TORCH_PRE_CHANNEL}" in
        nightly|test|stable|"") ;;
        *)
            echo "Error: Invalid torch channel '${TORCH_PRE_CHANNEL}' in ${torch_pre_file}" >&2
            echo "Must be 'nightly', 'test' or 'stable'" >&2
            return 1
            ;;
    esac
}

# Get the PyTorch index URL for a given channel and CUDA target
# Arguments:
#   $1: channel - "nightly", "test", or "stable"
#   $2: cuda_target - e.g., "cu128", "cu126", "cpu" (default: "cpu")
# Returns: The index URL (e.g., "https://download.pytorch.org/whl/nightly/cu128")
# Usage:
#   index_url=$(get_torch_index_url "nightly" "cu128")
get_torch_index_url(){
    local channel="$1"
    local cuda_target="${2:-cpu}"

    case "${channel}" in
        nightly)
            echo "https://download.pytorch.org/whl/nightly/${cuda_target}"
            ;;
        test)
            echo "https://download.pytorch.org/whl/test/${cuda_target}"
            ;;
        stable|"")
            # Stable channel uses different URL format
            echo "https://download.pytorch.org/whl/${cuda_target}"
            ;;
        *)
            echo "Error: Unknown torch channel: ${channel}" >&2
            return 1
            ;;
    esac
}

# Determine venv path based on priority: --venv-dir > the project's venv-base variable > default ~/.venvs
# The project's variable is the one INFRA_VENV_BASE_VAR names (infra_utils_project.sh sets it, e.g.
# IT_VENV_BASE or FTS_VENV_BASE).
# Usage: venv_path=$(determine_venv_path "$venv_dir" "$target_env_name")
# Arguments:
#   $1: venv_dir (optional) - explicit base directory from --venv-dir flag
#   $2: target_env_name (required) - environment name to append to base
# Returns: Full venv path (e.g., /mnt/cache/user/.venvs/it_latest)
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
    elif [[ -n "${INFRA_VENV_BASE_VAR:-}" && -n "${!INFRA_VENV_BASE_VAR:-}" ]]; then
        # Use the project's venv-base environment variable
        venv_base="${!INFRA_VENV_BASE_VAR}"
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
        # Note: uv pip uninstall does not support -y flag (no confirmation prompt needed)
        local pkg_underscore="${pkg//-/_}"
        uv pip uninstall "${pkg}" "${pkg_underscore}" 2>/dev/null || true

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
        local pkg_extra_flags=""
        if [[ -n ${pkg_env_vars} ]]; then
            echo "Setting environment variables for ${pkg} installation:"
            IFS='|' read -ra ENV_VARS <<< "${pkg_env_vars}"
            for env_var in "${ENV_VARS[@]}"; do
                if [[ $env_var =~ ^([^=]+)=(.*)$ ]]; then
                    local var_name="${BASH_REMATCH[1]}"
                    local var_value="${BASH_REMATCH[2]}"
                    var_value=$(expand_tilde_segments "${var_value}")
                    echo "  export ${var_name}=${var_value}"
                    export "${var_name}=${var_value}"
                    env_vars_set+=("${var_name}")
                    if [[ ${var_name} == "FLAGS" ]]; then
                        # The FLAGS keyword carries additional `uv pip install` flags for this
                        # package's editable install (e.g., "FLAGS=-r /path/to/exported_reqs.txt"
                        # for Poetry-legacy packages such as SAELens whose dev/test dependencies
                        # are not expressible as extras).
                        pkg_extra_flags="$(strip_quotes "${var_value}")"
                    fi
                fi
            done
        fi

        # If UV_OVERRIDE is set and contains the package being installed, filter
        # that entry out to a temp file. uv uses the override to resolve the
        # package from PyPI instead of the local "-e ." source when the package
        # name appears in the override file, preventing editable installs.
        local _orig_uv_override=""
        if [[ -n "${UV_OVERRIDE:-}" && -f "${UV_OVERRIDE}" ]]; then
            if grep -qiE "^${pkg}([^a-zA-Z0-9_-]|$)|^${pkg_underscore}([^a-zA-Z0-9_-]|$)" "${UV_OVERRIDE}"; then
                _orig_uv_override="${UV_OVERRIDE}"
                local _tmp_override
                _tmp_override=$(mktemp)
                grep -viE "^${pkg}([^a-zA-Z0-9_-]|$)|^${pkg_underscore}([^a-zA-Z0-9_-]|$)" "${UV_OVERRIDE}" > "${_tmp_override}"
                export UV_OVERRIDE="${_tmp_override}"
                echo "  (filtered ${pkg} from UV_OVERRIDE for editable install)"
            fi
        fi

        uv pip install ${uv_install_flags} ${pkg_extra_flags} -e "${install_target}"

        # Restore original UV_OVERRIDE if we created a temp filtered version
        if [[ -n "${_orig_uv_override}" ]]; then
            rm -f "${UV_OVERRIDE}"
            export UV_OVERRIDE="${_orig_uv_override}"
        fi

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

# The project's own part: its venv-base variable and any helper only it uses.
_infra_utils_project="$(dirname "${BASH_SOURCE[0]}")/infra_utils_project.sh"
if [[ -r "${_infra_utils_project}" ]]; then
    # shellcheck source=infra_utils_project.sh
    source "${_infra_utils_project}"
fi
