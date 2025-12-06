#!/bin/bash
#
# Utility script to build FTS environments using uv
# Usage examples:
# build latest (uses FTS_VENV_BASE or default ~/.venvs):
#   ./build_fts_env.sh --repo_home=~/repos/finetuning-scheduler --target_env_name=fts_latest
# build latest with explicit venv directory (recommended for hardlink performance):
#   ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --venv-dir=/mnt/cache/${USER}/.venvs
# build oldest (CI oldest build simulation with Python 3.10 and oldest deps):
#   ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_oldest --oldest
# build release:
#   ./build_fts_env.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release
# build latest with torch test channel:
#   ./build_fts_env.sh --repo_home=~/repos/finetuning-scheduler --target_env_name=fts_latest --torch_test_channel
# build latest from a package from source:
#   ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --from-source="lightning:${HOME}/repos/lightning:pytorch"
#
# To use a specific PyTorch nightly, edit requirements/ci/torch-nightly.txt
set -eo pipefail

unset repo_home
unset target_env_name
unset torch_test_channel
unset uv_install_flags
unset no_commit_pin
unset venv_dir
unset oldest
declare -a from_source_specs=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/infra_utils.sh"

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo_home input]
   [ --target_env_name input ]
   [ --oldest ]                # Use oldest CI requirements (Python 3.10, requirements-oldest.txt)
   [ --torch_test_channel ]    # Use PyTorch test/RC channel
   [ --uv_install_flags "flags" ]
   [ --no_commit_pin ]
   [ --venv-dir input ]
   [ --from-source "package:path[:extras][:ENV_VAR=value]" ]
   [ --help ]
   Examples:
    # build latest (uses FTS_VENV_BASE or default ~/.venvs):
    #   ./build_fts_env.sh --repo_home=\${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest
    # build latest with explicit venv directory (recommended for hardlink performance):
    #   ./build_fts_env.sh --repo_home=\${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --venv-dir=/mnt/cache/\${USER}/.venvs
    # build oldest (CI oldest build simulation):
    #   ./build_fts_env.sh --repo_home=\${HOME}/repos/finetuning-scheduler --target_env_name=fts_oldest --oldest --venv-dir=/mnt/cache/\${USER}/.venvs
    # build release:
    #   ./build_fts_env.sh --repo_home=\${HOME}/repos/fts-release --target_env_name=fts_release
    # build latest with torch test channel:
    #   ./build_fts_env.sh --repo_home=\${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_test_channel
    # build latest with no cache:
    #   ./build_fts_env.sh --repo_home=\${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --uv_install_flags="--no-cache"
    # build latest without using CI commit pinning:
    #   ./build_fts_env.sh --repo_home=\${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --no_commit_pin
    # build latest from Lightning source:
    #   ./build_fts_env.sh --repo_home=\${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --from-source="lightning:\${HOME}/repos/lightning:pytorch"

    # To use a specific PyTorch nightly, edit requirements/ci/torch-nightly.txt:
    #   Line 1: torch version (e.g., 2.10.0.dev20251124)
    #   Line 2: CUDA target (e.g., cu128)
EOF
exit 1
}

args=$(getopt -o '' --long repo_home:,target_env_name:,oldest,torch_test_channel,uv_install_flags:,no_commit_pin,venv-dir:,from-source:,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo_home)  repo_home=$2    ; shift 2  ;;
    --target_env_name)  target_env_name=$2  ; shift 2 ;;
    --oldest)   oldest=1 ; shift  ;;
    --torch_test_channel)   torch_test_channel=1 ; shift  ;;
    --uv_install_flags)   uv_install_flags=$2 ; shift 2 ;;
    --no_commit_pin)   no_commit_pin=1 ; shift  ;;
    --venv-dir)   venv_dir=$2 ; shift 2 ;;
    --from-source)
        # Accumulate multiple --from-source flags into array, join with semicolon
        from_source_specs+=("$2")
        shift 2
        ;;
    --help)    usage      ; shift   ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

# Use uv_install_flags in uv pip commands
uv_install_flags=${uv_install_flags:-""}

# Determine venv path using infra_utils function
venv_path=$(determine_venv_path "${venv_dir}" "${target_env_name}")
echo "Target venv path: ${venv_path}"

# Join from-source specs with semicolons
from_source_spec=""
if [[ ${#from_source_specs[@]} -gt 0 ]]; then
    from_source_spec=$(IFS=';'; echo "${from_source_specs[*]}")
fi

# Read torch nightly configuration from requirements/ci/torch-nightly.txt
# Returns two values via global variables: TORCH_NIGHTLY_VERSION and TORCH_NIGHTLY_CUDA
read_torch_nightly_config() {
    local nightly_file="${repo_home}/requirements/ci/torch-nightly.txt"
    TORCH_NIGHTLY_VERSION=""
    TORCH_NIGHTLY_CUDA=""

    if [[ -f "${nightly_file}" ]]; then
        # Read non-comment, non-empty lines
        local lines=()
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Skip comments and empty lines
            [[ "$line" =~ ^# ]] && continue
            [[ -z "$line" ]] && continue
            lines+=("$line")
        done < "${nightly_file}"

        # First line is torch version, second is CUDA target
        if [[ ${#lines[@]} -ge 1 ]]; then
            TORCH_NIGHTLY_VERSION="${lines[0]}"
        fi
        if [[ ${#lines[@]} -ge 2 ]]; then
            TORCH_NIGHTLY_CUDA="${lines[1]}"
        fi
    fi
}

clear_activate_env(){
    local python_version=$1
    echo "Creating venv at ${venv_path} with ${python_version}..."
    uv venv --clear --python ${python_version} "${venv_path}"
    source "${venv_path}/bin/activate"
    echo "Current venv prompt is now ${VIRTUAL_ENV_PROMPT}"
}

log_torch_version(){
    local context=$1
    echo "[VERSION CHECK - ${context}] torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
}

base_env_build(){
    # Use Python 3.10 for oldest builds, 3.12 for latest
    local python_version="python3.12"
    if [[ -n ${oldest} ]]; then
        python_version="python3.10"
        echo "Using Python 3.10 for oldest build"
    fi

    clear_activate_env ${python_version}

    # Check for torch nightly configuration (skip for oldest builds)
    if [[ -z ${oldest} ]]; then
        read_torch_nightly_config
    fi

    # Handle PyTorch version selection (pre-install before FTS dependencies)
    # Priority: oldest (stable from lock) > torch nightly from config > torch test channel > stable (via --torch-backend in fts_install)
    if [[ -n ${oldest} ]]; then
        # For oldest builds, torch is installed from requirements-oldest.txt (stable version)
        echo "Using torch stable from requirements-oldest.txt for oldest build"
    elif [[ -n "${TORCH_NIGHTLY_VERSION}" ]]; then
        # Nightly version from torch-nightly.txt with specified CUDA backend
        local cuda_target="${TORCH_NIGHTLY_CUDA:-cu128}"  # Default to cu128 if not specified
        local torch_pkg="torch==${TORCH_NIGHTLY_VERSION}"
        local torch_index_url="https://download.pytorch.org/whl/nightly/${cuda_target}"
        echo "Pre-installing PyTorch nightly from torch-nightly.txt: ${torch_pkg}"
        echo "  CUDA target: ${cuda_target}"
        echo "  Index URL: ${torch_index_url}"
        uv pip install ${uv_install_flags} --prerelease=allow "${torch_pkg}" --index-url "${torch_index_url}"
        log_torch_version "after PyTorch nightly pre-install"
    elif [[ -n ${torch_test_channel} ]]; then
        # Test/RC channel - pre-install torch with test index and auto backend for GPU detection
        local torch_index_url="https://download.pytorch.org/whl/test"
        echo "Pre-installing PyTorch from test channel: ${torch_index_url}"
        echo "  Using --torch-backend=auto for GPU auto-detection"
        uv pip install ${uv_install_flags} --prerelease=allow torch --index-url ${torch_index_url} --torch-backend=auto
        log_torch_version "after PyTorch test channel pre-install"
    fi
    # For stable builds, torch will be installed via FTS dependencies with --torch-backend=auto
}

fts_install(){
    source "${venv_path}/bin/activate"
    unset PACKAGE_NAME
    cd ${repo_home}

    # Set UV_OVERRIDE for Lightning commit pin, unless --no_commit_pin is specified
    local override_file="${repo_home}/requirements/ci/overrides.txt"
    if [[ -z ${no_commit_pin} ]]; then
        export UV_OVERRIDE="${override_file}"
        echo "Using Lightning from commit pin via UV_OVERRIDE"
        echo "Override file: ${UV_OVERRIDE}"
    else
        unset UV_OVERRIDE
        echo "Using Lightning from PyPI (no override)"
    fi

    # Install FTS using locked requirements file
    # UV_OVERRIDE env var handles Lightning commit pinning automatically
    # When torch nightly is configured, requirements.txt already has torch filtered
    local req_file="${repo_home}/requirements/ci/requirements.txt"
    local torch_backend_flag=""

    # For oldest builds, use requirements-oldest.txt
    if [[ -n ${oldest} ]]; then
        req_file="${repo_home}/requirements/ci/requirements-oldest.txt"
        echo "Using oldest requirements file: ${req_file}"
        # Oldest builds use torch stable from lock file, need --torch-backend=auto
        torch_backend_flag="--torch-backend=auto"
    elif [[ -n "${TORCH_NIGHTLY_VERSION}" || -n ${torch_test_channel} ]]; then
        # Torch already pre-installed (nightly or test channel)
        # When nightly: requirements.txt already has torch filtered during lock generation
        # When test channel: filter at runtime
        if [[ -n ${torch_test_channel} ]]; then
            echo "Torch test channel pre-installed, filtering torch from requirements..."
            grep -v '^torch==' "${req_file}" > /tmp/requirements_no_torch.txt
            req_file="/tmp/requirements_no_torch.txt"
        fi
        echo "Using requirements without torch (pre-installed)"
    else
        # Use auto torch backend for GPU detection
        torch_backend_flag="--torch-backend=auto"
        echo "Using torch backend: auto (GPU auto-detection)"
    fi

    uv pip install ${uv_install_flags} -e . -r "${req_file}" ${torch_backend_flag}
    log_torch_version "after FTS install"

    # Install from source packages if specified
    if [[ -n ${from_source_spec} ]]; then
        echo "Installing from-source packages..."
        declare -A from_source_packages
        parse_from_source_specs "${from_source_spec}" from_source_packages
        install_from_source_packages from_source_packages "${venv_path}" "${uv_install_flags}"
        cd ${repo_home}
    fi

    # Install docs requirements - UV_OVERRIDE env var is still set if applicable
    uv pip install ${uv_install_flags} -r requirements/docs.txt ${torch_backend_flag}
    log_torch_version "after docs requirements install"

    # Install pip for pre-commit (it uses pip internally)
    uv pip install pip

    # Development setup
    pyright -p pyproject.toml || echo "⚠ pyright check had issues, continuing..."
    pre-commit install
    git lfs install

    # Verify installation
    python -c "import importlib.metadata; import torch; import lightning.pytorch; import finetuning_scheduler;
for package in ['torch', 'lightning', 'finetuning_scheduler']:
    print(f'{package} version: {importlib.metadata.distribution(package).version}');"
}

d=`date +%Y%m%d%H%M%S`
echo "FTS env build executing at ${d} PT"
echo "Beginning env removal/update for ${target_env_name}"
maybe_deactivate
echo "Beginning FTS base env install for ${target_env_name}"
base_env_build
echo "Beginning FTS dev install for ${target_env_name}"
fts_install
echo "FTS env successfully built for ${target_env_name}!"
