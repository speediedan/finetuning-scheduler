#!/bin/bash
#
# Utility script to build FTS environments using uv
# Usage examples:
# build latest (uses FTS_VENV_BASE or default ~/.venvs):
#   ./build_fts_env.sh --repo-home=~/repos/finetuning-scheduler --target-env-name=fts_latest
# build latest with explicit venv directory (recommended for hardlink performance):
#   ./build_fts_env.sh --repo-home=${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --venv-dir=/mnt/cache/${USER}/.venvs
# build oldest (CI oldest build simulation with Python 3.10 and oldest deps):
#   ./build_fts_env.sh --repo-home=${HOME}/repos/finetuning-scheduler --target-env-name=fts_oldest --oldest
# build release:
#   ./build_fts_env.sh --repo-home=${HOME}/repos/fts-release --target-env-name=fts_release
# build latest from a package from source:
#   ./build_fts_env.sh --repo-home=${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --from-source="lightning:${HOME}/repos/lightning:pytorch"
#
# To configure PyTorch version (nightly/test/stable), edit requirements/ci/torch-pre.txt
set -eo pipefail

unset repo_home
unset target_env_name
unset uv_install_flags
unset no_commit_pin
unset venv_dir
unset torch_backend
unset oldest
declare -a from_source_specs=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/infra_utils.sh"

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo-home input]
   [ --target-env-name input ]
   [ --oldest ]                # Use oldest CI requirements (Python 3.10, requirements-oldest.txt)
   [ --uv-install-flags "flags" ]
   [ --no-commit-pin ]
   [ --venv-dir input ]
   [ --torch-backend input ]  (cpu, cu130, auto; default: cu130 for CUDA 12.8)
   [ --from-source "package:path[:extras][:ENV_VAR=value]" ]
   [ --help ]
   Examples:
    # build latest (uses FTS_VENV_BASE or default ~/.venvs):
    #   ./build_fts_env.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest
    # build latest with explicit venv directory (recommended for hardlink performance):
    #   ./build_fts_env.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --venv-dir=/mnt/cache/\${USER}/.venvs
    # build oldest (CI oldest build simulation):
    #   ./build_fts_env.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_oldest --oldest --venv-dir=/mnt/cache/\${USER}/.venvs
    # build release:
    #   ./build_fts_env.sh --repo-home=\${HOME}/repos/fts-release --target-env-name=fts_release
    # build latest with no cache:
    #   ./build_fts_env.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --uv-install-flags="--no-cache"
    # build latest without using CI commit pinning:
    #   ./build_fts_env.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --no-commit-pin
    # build latest from Lightning source:
    #   ./build_fts_env.sh --repo-home=\${HOME}/repos/finetuning-scheduler --target-env-name=fts_latest --from-source="lightning:\${HOME}/repos/lightning:pytorch"

    # To configure PyTorch version, edit requirements/ci/torch-pre.txt:
    #   Line 1: torch version (e.g., 2.11.0 for test, 2.11.0.dev20260121 for nightly)
    #   Line 2: CUDA target (e.g., cu130)
    #   Line 3: channel type (test or nightly)
EOF
exit 1
}

args=$(getopt -o '' --long repo-home:,repo_home:,target-env-name:,target_env_name:,oldest,uv-install-flags:,uv_install_flags:,no-commit-pin,no_commit_pin,venv-dir:,torch-backend:,from-source:,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo-home|--repo_home)  repo_home=$2    ; shift 2  ;;
    --target-env-name|--target_env_name)  target_env_name=$2  ; shift 2 ;;
    --oldest)   oldest=1 ; shift  ;;
    --uv-install-flags|--uv_install_flags)   uv_install_flags=$2 ; shift 2 ;;
    --no-commit-pin|--no_commit_pin)   no_commit_pin=1 ; shift  ;;
    --venv-dir)   venv_dir=$2 ; shift 2 ;;
    --torch-backend)   torch_backend=$2   ; shift 2 ;;
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

# Set default torch backend if not specified (auto = auto-detect CUDA/CPU)
torch_backend=${torch_backend:-"cu130"}

# Join from-source specs with semicolons
from_source_spec=""
if [[ ${#from_source_specs[@]} -gt 0 ]]; then
    from_source_spec=$(IFS=';'; echo "${from_source_specs[*]}")
fi

# Read torch prerelease configuration (now handled by infra_utils.sh::read_torch_pre_config)

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
    # Use Python 3.10 for oldest builds, 3.13 for latest
    local python_version="python3.13"
    if [[ -n ${oldest} ]]; then
        python_version="python3.10"
        echo "Using Python 3.10 for oldest build"
    fi

    clear_activate_env ${python_version}

    # Check for torch prerelease configuration (skip for oldest builds)
    if [[ -z ${oldest} ]]; then
        read_torch_pre_config
    fi

    # Handle PyTorch version selection (pre-install before FTS dependencies)
    # Priority: oldest (stable from lock) > torch prerelease from config > stable (via --torch-backend in fts_install)
    if [[ -n ${oldest} ]]; then
        # For oldest builds, torch is installed from requirements-oldest.txt (stable version)
        echo "Using torch stable from requirements-oldest.txt for oldest build"
    elif [[ -n "${TORCH_PRE_VERSION}" ]]; then
        # Prerelease (nightly or test) configured in torch-pre.txt
        local cuda_target="${TORCH_PRE_CUDA:-cu130}"  # Default to cu130 if not specified
        local torch_pkg="torch==${TORCH_PRE_VERSION}"
        local torch_index_url=$(get_torch_index_url "${TORCH_PRE_CHANNEL}" "${cuda_target}")

        echo "Pre-installing PyTorch ${TORCH_PRE_CHANNEL} from torch-pre.txt: ${torch_pkg}"
        echo "  Channel: ${TORCH_PRE_CHANNEL}"
        echo "  CUDA target: ${cuda_target}"
        echo "  Index URL: ${torch_index_url}"

        uv pip install ${uv_install_flags} --prerelease=allow "${torch_pkg}" --index-url "${torch_index_url}"
        log_torch_version "after PyTorch ${TORCH_PRE_CHANNEL} pre-install"
    fi
    # For stable builds, torch will be installed via FTS dependencies with --torch-backend=auto
}

fts_install(){
    source "${venv_path}/bin/activate"
    unset PACKAGE_NAME
    cd ${repo_home}

    # Set UV_OVERRIDE for Lightning commit pin, unless --no-commit-pin is specified
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
    elif [[ -n "${TORCH_PRE_VERSION}" ]]; then
        # Torch prerelease already pre-installed (nightly or test channel)
        # requirements.txt already has torch filtered during lock generation
        echo "Using requirements without torch (pre-installed ${TORCH_PRE_CHANNEL})"
    else
        # Install stable torch - use --torch-backend for automatic backend selection
        echo "Installing stable torch with --torch-backend=${torch_backend}..."
        uv pip install ${uv_install_flags} torch --torch-backend=${torch_backend}
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
