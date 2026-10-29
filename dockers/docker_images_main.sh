#!/bin/bash
#
# Utility script to build and upload the docker images required to execute the CI for the main FTS dev branch
# N.B. this script assumes access to the relevant docker registry has already been granted via either a manual
# `docker login` command or more likely, a local wrapper script that invokes this script

set -eo pipefail
source $(dirname "$0")/build_image_version.sh

repo_home=$1
registry_name=$2
build_new="${3:-1}"
push_remote="${4:-1}"
# 0 skips the ROCm images (the AMD agent's twin of the CUDA pair, #57)
build_rocm="${5:-1}"

maybe_deactivate

d=`date +%Y%m%d%H%M%S`
tmp_docker_build_log_dir="/tmp"
docker_build_log="${tmp_docker_build_log_dir}/fts_update_docker_main_images_${d}.log"


maybe_push(){
    if [[ $push_remote -ne 0 ]]; then
        echo "Beginning upload of built images..." >> $docker_build_log
		echo "Pushing ${registry_name}:${latest_pt} ..." >> $docker_build_log
        docker push ${registry_name}:${latest_pt} >> $docker_build_log
		echo "Finished pushing ${registry_name}:${latest_pt} ..." >> $docker_build_log
		echo "Pushing ${registry_name}:${latest_azpl} ..." >> $docker_build_log
		docker push ${registry_name}:${latest_azpl} >> $docker_build_log
		echo "Finished pushing ${registry_name}:${latest_azpl} ..." >> $docker_build_log
		if [[ $build_rocm -ne 0 ]]; then
			for tag in ${latest_rocm_pt} ${latest_rocm_azpl}; do
				echo "Pushing ${registry_name}:${tag} ..." >> $docker_build_log
				docker push ${registry_name}:${tag} >> $docker_build_log
				echo "Finished pushing ${registry_name}:${tag} ..." >> $docker_build_log
			done
		fi
    else
        echo "Directed to skip push of built images." >> $docker_build_log
    fi
}

maybe_build(){
	if [[ $build_new -ne 0 ]]; then
		echo "Building images base: $2 and azpl-init: $3" >> $docker_build_log
		build_version "$1" "$2" "$3"
	fi
}

build_eval(){
	# latest PyTorch image supported by release
	# see CUDA_ARCHES_FULL_VERSION for the full version of the pytorch-provided toolkit
	declare -A iv=(["cuda"]="13.2.1" ["python"]="3.13" ["pytorch"]="2.15.0" ["lightning"]="2.6" ["cust_build"]="1")
	export latest_pt="base-cu${iv["cuda"]}-py${iv["python"]}-pt${iv["pytorch"]}-pl${iv["lightning"]}"
	export latest_azpl="py${iv["python"]}-pt${iv["pytorch"]}-pl${iv["lightning"]}-azpl-init"
	maybe_build iv "${latest_pt}" "${latest_azpl}"
	# the AMD agent's images: the same Python, PyTorch and Lightning, with PyTorch's own ROCm wheel (dockers/base-rocm)
	declare -A rv=(["rocm"]="10.0" ["python"]="${iv["python"]}" ["pytorch"]="${iv["pytorch"]}" ["lightning"]="${iv["lightning"]}" ["channel"]="")
	export latest_rocm_pt="base-rocm${rv["rocm"]}-py${rv["python"]}-pt${rv["pytorch"]}-pl${rv["lightning"]}"
	export latest_rocm_azpl="py${rv["python"]}-pt${rv["pytorch"]}-pl${rv["lightning"]}-rocm${rv["rocm"]}-azpl-init"
	if [[ $build_new -ne 0 && $build_rocm -ne 0 ]]; then
		echo "Building ROCm images base: ${latest_rocm_pt} and azpl-init: ${latest_rocm_azpl}" >> $docker_build_log
		build_rocm_version rv "${latest_rocm_pt}" "${latest_rocm_azpl}"
	fi
	if [[ $build_new -ne 0 ]]; then
		echo "All image builds successful." >> $docker_build_log
	else
        echo "Directed to skip building of new images, now checking whether to push..."  >> $docker_build_log
    fi
}


cd ${repo_home}
echo "Building and/or pushing images for repository home: ${repo_home}" >> $docker_build_log
build_eval
maybe_push
echo "Finished building and/or updating images for repository home: ${repo_home}" >> $docker_build_log
