#!/bin/bash
#
# Utility script to build FTS environments using uv
# Usage examples:
# build latest (uses FTS_VENV_BASE or default ~/.venvs):
#   ./build_fts_env.sh --repo_home=~/repos/finetuning-scheduler --target_env_name=fts_latest
# build latest with explicit venv directory (recommended for hardlink performance):
#   ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --venv-dir=/mnt/cache/${USER}/.venvs
# build release:
#   ./build_fts_env.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release
# build latest with specific pytorch nightly:
#   ./build_fts_env.sh --repo_home=~/repos/finetuning-scheduler --target_env_name=fts_latest --torch_dev_ver=dev20230820
# build latest with torch test channel:
#    ./build_fts_env.sh --repo_home=~/repos/finetuning-scheduler --target_env_name=fts_latest --torch_test_channel
# build latest from a package from source:
#    ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --from-source="lightning:${HOME}/repos/lightning:pytorch"
set -eo pipefail

unset repo_home
unset target_env_name
unset torch_dev_ver
unset torch_test_channel
unset uv_install_flags
unset no_commit_pin
unset venv_dir
declare -a from_source_specs=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/infra_utils.sh"

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo_home input]
   [ --target_env_name input ]
   [ --torch_dev_ver input ]
   [ --torch_test_channel ]
   [ --uv_install_flags "flags" ]
   [ --no_commit_pin ]
   [ --venv-dir input ]
   [ --from-source "package:path[:extras][:ENV_VAR=value]" ]
   [ --help ]
   Examples:
    # build latest (uses FTS_VENV_BASE or default ~/.venvs):
    #   ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest
    # build latest with explicit venv directory (recommended for hardlink performance):
    #   ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --venv-dir=/mnt/cache/\${USER}/.venvs
    # build release:
    #   ./build_fts_env.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release
    # build release from torch test channel:
    #   ./build_fts_env.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release --torch_test_channel
    # build latest with specific pytorch nightly:
    #   ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_dev_ver=dev20231014
    # build latest with torch test channel:
    #    ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_test_channel
    # build latest with no cache:
    #    ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --uv_install_flags="--no-cache"
    # build latest without using CI commit pinning:
    #    ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --no_commit_pin
    # build latest from Lightning source:
    #    ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --from-source="lightning:\${HOME}/repos/lightning:pytorch"
EOF
exit 1
}

args=$(getopt -o '' --long repo_home:,target_env_name:,torch_dev_ver:,torch_test_channel,uv_install_flags:,no_commit_pin,venv-dir:,from-source:,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo_home)  repo_home=$2    ; shift 2  ;;
    --target_env_name)  target_env_name=$2  ; shift 2 ;;
    --torch_dev_ver)   torch_dev_ver=$2   ; shift 2 ;;
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
    local torch_index_url=""
    local torch_pkg=""
    local python_version="python3.12"

    case ${target_env_name} in
        fts_latest)
            if [[ -n ${torch_dev_ver} ]]; then
                # For nightly torch, only install torch (no torchvision needed)
                torch_pkg="torch==2.10.0.${torch_dev_ver}"
                torch_index_url="https://download.pytorch.org/whl/nightly/cu128"
            elif [[ $torch_test_channel -eq 1 ]]; then
                torch_pkg="torch==2.10.0"
                torch_index_url="https://download.pytorch.org/whl/test/cu128"
            else
                torch_pkg="torch"
                torch_index_url="https://download.pytorch.org/whl/cu128"
            fi
            ;;
        fts_release)
            if [[ $torch_test_channel -eq 1 ]]; then
                torch_pkg="torch torchvision"
                torch_index_url="https://download.pytorch.org/whl/test/cu128"
            else
                torch_pkg="torch torchvision"
                torch_index_url="https://download.pytorch.org/whl/cu128"
            fi
            ;;
        fts_latest_pt_oldest | fts_latest_pt2_6_x | fts_release_pt2_6_x)
            torch_pkg="torch==2.6.0 torchvision"
            torch_index_url="https://download.pytorch.org/whl/cu126"
            ;;
        fts_latest_pt2_7_x | fts_release_pt2_7_x)
            torch_pkg="torch==2.7.1 torchvision"
            torch_index_url="https://download.pytorch.org/whl/cu128"
            ;;
        fts_latest_pt2_8_x | fts_release_pt2_8_x)
            torch_pkg="torch==2.8.0 torchvision"
            torch_index_url="https://download.pytorch.org/whl/cu128"
            ;;
        fts_latest_pt2_9_x | fts_release_pt2_9_x)
            torch_pkg="torch==2.9.0 torchvision"
            torch_index_url="https://download.pytorch.org/whl/cu128"
            ;;
        fts_latest_pt2_10_x | fts_release_pt2_10_x)
            torch_pkg="torch==2.10.0 torchvision"
            torch_index_url="https://download.pytorch.org/whl/cu128"
            ;;
        *)
            echo "no matching environment found, exiting..."
            exit 1
            ;;
    esac

    clear_activate_env ${python_version}

    # Install PyTorch with specified index URL
    echo "Installing PyTorch: ${torch_pkg} from ${torch_index_url}"
    uv pip install ${uv_install_flags} --prerelease=allow ${torch_pkg} --index-url ${torch_index_url}
    log_torch_version "after PyTorch install"
}

fts_install(){
    source "${venv_path}/bin/activate"
    unset PACKAGE_NAME
    cd ${repo_home}

    # Use static override file for Lightning commit pin when USE_CI_COMMIT_PIN is set
    local override_file="${repo_home}/requirements/ci/overrides.txt"
    local use_override=0

    # Set USE_CI_COMMIT_PIN by default, unless --no_commit_pin is specified
    if [[ -z ${no_commit_pin} ]]; then
        export USE_CI_COMMIT_PIN="1"
        echo "Using Lightning from commit pin (dev/ci mode)"
        echo "Override file: ${override_file}"
        use_override=1
    else
        unset USE_CI_COMMIT_PIN
        echo "Using Lightning from PyPI"
    fi

    # Install FTS with or without override
    if [[ ${use_override} -eq 1 ]]; then
        uv pip install ${uv_install_flags} -e ".[all]" --override "${override_file}"
        log_torch_version "after FTS install with override"
    else
        uv pip install ${uv_install_flags} -e ".[all]"
        log_torch_version "after FTS install (no override)"
    fi

    # Install from source packages if specified
    if [[ -n ${from_source_spec} ]]; then
        echo "Installing from-source packages..."
        declare -A from_source_packages
        parse_from_source_specs "${from_source_spec}" from_source_packages
        install_from_source_packages from_source_packages "${venv_path}" "${uv_install_flags}"
        cd ${repo_home}
    fi

    # Install docs requirements with override to maintain consistency
    if [[ ${use_override} -eq 1 ]]; then
        uv pip install ${uv_install_flags} -r requirements/docs.txt --override "${override_file}"
    else
        uv pip install ${uv_install_flags} -r requirements/docs.txt
    fi
    log_torch_version "after docs requirements install"

    # Install pip for mypy and pre-commit (they use pip internally)
    uv pip install pip

    # Development setup
    rm -rf .mypy_cache
    mypy --install-types --non-interactive
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
