#!/bin/bash
#
# Utility script to build and upload the docker images required to execute the CI for the main FTS dev branch
# N.B. this script assumes access to the relevant docker registry has already been granted via either a manual
# `docker login` command or more likely, a local wrapper script that invokes this script

set -eo pipefail

repo_home=$1
registry_name=$2
build_new="${3:-1}"
push_remote="${4:-1}"

eval "$(conda shell.bash hook)"  # setup shell functions for conda, uses conda's .bashrc resident defined hook to execute conda init setup to enable subsequent conda command usage
conda deactivate

d=`date +%Y%m%d%H%M%S`
tmp_docker_build_log_dir="/tmp"
docker_build_log="${tmp_docker_build_log_dir}/fts_build_docker_main_images_${d}.log"


maybe_push(){
    if [[ -n $push_remote ]]; then
        echo "Beginning upload of built images..." >> $docker_build_log
        docker push ${registry_name}:${latest_pt} >> $docker_build_log
		docker push ${registry_name}:${latest_azpl_init} >> $docker_build_log
		docker push ${registry_name}:${pt_2_0_1} >> $docker_build_log
		docker push ${registry_name}:${pt_2_0_1_azpl_init} >> $docker_build_log
    else
        echo "Directed to skip push of built images." >> $docker_build_log
    fi
}

maybe_build(){
	if [[ $build_new -ne 0 ]]; then
		# main latest supported PyTorch image
		latest_pt="base-cu12.1-py3.11-pt2.2.0-pl2.2"
		echo "Building latest pt: ${latest_pt}" >> $docker_build_log
		docker image build -t ${latest_pt} -f dockers/base-cuda/Dockerfile --build-arg PYTHON_VERSION=3.11 \
		  --build-arg PYTORCH_VERSION=2.2.0 --build-arg CUST_BUILD=1 --no-cache . >> $docker_build_log
		docker tag ${latest_pt} ${registry_name}:${latest_pt} >> $docker_build_log
		# associated azpl-init image
		latest_azpl_init="py3.11-pt2.2.0-pl2.2-azpl-init"
		docker image build -t ${latest_azpl_init} -f dockers/fts-az-base/Dockerfile --build-arg PYTHON_VERSION=3.11 \
		  --build-arg CUST_BASE=cu12.1- --no-cache . >> $docker_build_log
		docker tag ${latest_azpl_init} ${registry_name}:${latest_azpl_init} >> $docker_build_log
		################################################################################################################
		# other previous PyTorch image(s)
		# only PyTorch 2.0.1 currently required
		pt_2_0_1="base-cu11.8-py3.10-pt2.0.1-pl2.2"
		echo "Building images for pt 2.0.1: ${pt_2_0_1}" >> $docker_build_log
		docker image build -t ${pt_2_0_1} -f dockers/base-cuda/Dockerfile --build-arg PYTHON_VERSION=3.10 \
		  --build-arg PYTORCH_VERSION=2.0.1 --build-arg CUST_BUILD=0 --no-cache . >> $docker_build_log
		docker tag ${pt_2_0_1} ${registry_name}:${pt_2_0_1} >> $docker_build_log
		# associated azpl-init image
		pt_2_0_1_azpl_init="py3.10-pt2.0.1-pl2.2-azpl-init"
		docker image build -t ${pt_2_0_1_azpl_init} -f dockers/fts-az-base/Dockerfile --build-arg PYTHON_VERSION=3.10
		  --build-arg CUST_BASE=cu11.8- --build-arg PYTORCH_VERSION=2.0.1 --no-cache . >> $docker_build_log
		docker tag ${pt_2_0_1_azpl_init} ${registry_name}:${pt_2_0_1_azpl_init} >> $docker_build_log
		echo "Image build successful" >> $docker_build_log
	else
        echo "Directed to skip building of new images, now checking whether to push..."
    fi
}


cd ${repo_home}
maybe_build
maybe_push
