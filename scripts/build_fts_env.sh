#!/bin/bash
#
# Utility script to build FTS environments
# Usage examples:
# build latest:
#   ./build_fts_env.sh --repo_home=~/repos/finetuning-scheduler --target_env_name=fts_latest
# build release:
#   ./build_fts_env.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release
# build latest with specific pytorch nightly:
#   ./build_fts_env.sh --repo_home=~/repos/finetuning-scheduler --target_env_name=fts_latest --torch_dev_ver=dev20230820 --torchvision_dev_ver=dev20230821
# build latest with torch test channel:
#    ./build_fts_env.sh --repo_home=~/repos/finetuning-scheduler --target_env_name=fts_latest --torch_test_channel
set -eo pipefail

unset repo_home
unset target_env_name
unset torch_dev_ver
unset torchvision_dev_ver
unset torch_test_channel
unset pip_install_flags
unset no_commit_pin

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo_home input]
   [ --target_env_name input ]
   [ --torch_dev_ver input ]
   [ --torchvision_dev_ver input ]
   [ --torch_test_channel ]
   [ --pip_install_flags "flags" ]
   [ --no_commit_pin ]
   [ --help ]
   Examples:
    # build latest:
    #   ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest
    # build release:
    #   ./build_fts_env.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release
    # build release from torch test channel:
    #   ./build_fts_env.sh --repo_home=${HOME}/repos/fts-release --target_env_name=fts_release --torch_test_channel
    # build latest with specific pytorch nightly:
    #   ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_dev_ver=dev20231014 --torchvision_dev_ver=dev20231014
    # build latest with torch test channel:
    #    ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --torch_test_channel
    # build latest with no cache directory:
    #    ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --pip_install_flags="--no-cache-dir"
    # build latest without using CI commit pinning:
    #    ./build_fts_env.sh --repo_home=${HOME}/repos/finetuning-scheduler --target_env_name=fts_latest --no_commit_pin
EOF
exit 1
}

args=$(getopt -o '' --long repo_home:,target_env_name:,torch_dev_ver:,torchvision_dev_ver:,torch_test_channel,pip_install_flags:,no_commit_pin,help -- "$@")
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
    --torchvision_dev_ver)   torchvision_dev_ver=$2   ; shift 2 ;;
    --torch_test_channel)   torch_test_channel=1 ; shift  ;;
    --pip_install_flags)   pip_install_flags=$2 ; shift 2 ;;
    --no_commit_pin)   no_commit_pin=1 ; shift  ;;
    --help)    usage      ; shift   ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

# Use pip_install_flags in pip commands
pip_install_flags=${pip_install_flags:-""}

maybe_deactivate(){
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
}

clear_activate_env(){
    $1 -m venv --clear ~/.venvs/${target_env_name}
    source ~/.venvs/${target_env_name}/bin/activate
    echo "Current venv prompt is now ${VIRTUAL_ENV_PROMPT}"
    pip install ${pip_install_flags} --upgrade pip
}

base_env_build(){
    case ${target_env_name} in
        fts_latest)
            clear_activate_env python3.12
            if [[ -n ${torch_dev_ver} ]]; then
                if [[ -n ${torchvision_dev_ver} ]]; then
                    torchvision_dev_ver=${torch_dev_ver}
                fi
                pip install ${pip_install_flags} --pre torch==2.10.0.${torch_dev_ver} --index-url https://download.pytorch.org/whl/nightly/cu128
            elif [[ $torch_test_channel -eq 1 ]]; then
                pip install ${pip_install_flags} --pre torch==2.10.0 --index-url https://download.pytorch.org/whl/test/cu128
            else
                pip install ${pip_install_flags} torch torchvision --index-url https://download.pytorch.org/whl/cu128
            fi
            ;;
        fts_release)
            clear_activate_env python3.12
            if [[ $torch_test_channel -eq 1 ]]; then
                pip install ${pip_install_flags} --pre torch torchvision --index-url https://download.pytorch.org/whl/test/cu128
            else
                pip install ${pip_install_flags} torch torchvision --index-url https://download.pytorch.org/whl/cu128
            fi
            ;;
        fts_latest_pt_oldest | fts_latest_pt2_6_x | fts_release_pt2_6_x)
            clear_activate_env python3.12
            pip install ${pip_install_flags} torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126
            ;;
        fts_latest_pt2_7_x | fts_release_pt2_7_x)
            clear_activate_env python3.12
            pip install ${pip_install_flags} torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
            ;;
        fts_latest_pt2_8_x | fts_release_pt2_8_x)
            clear_activate_env python3.12
            pip install ${pip_install_flags} torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
            ;;
        fts_latest_pt2_9_x | fts_release_pt2_9_x)
            clear_activate_env python3.12
            pip install ${pip_install_flags} torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
            ;;
        fts_latest_pt2_10_x | fts_release_pt2_10_x)
            clear_activate_env python3.12
            pip install ${pip_install_flags} torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
            ;;
        *)
            echo "no matching environment found, exiting..."
            exit 1
            ;;
    esac

}

fts_install(){
    source ~/.venvs/${target_env_name}/bin/activate
    unset PACKAGE_NAME
    cd ${repo_home}

    # Set USE_CI_COMMIT_PIN by default, unless --no_commit_pin is specified
    if [[ -z ${no_commit_pin} ]]; then
        export USE_CI_COMMIT_PIN="1"
    else
        unset USE_CI_COMMIT_PIN
    fi

    python -m pip install ${pip_install_flags} -e ".[all]" -r requirements/docs.txt
    rm -rf .mypy_cache
    mypy --install-types --non-interactive
    pre-commit install
    git lfs install
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
