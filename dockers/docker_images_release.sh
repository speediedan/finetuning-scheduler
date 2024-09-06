#!/bin/bash
#
# Utility script to build and upload the docker images required to execute the CI for the latest FTS release branch
# N.B. this script assumes access to the relevant docker registry has already been granted via either a manual
# `docker login` command or more likely, a local wrapper script that invokes this script

set -eo pipefail
source $(dirname "$0")/build_image_version.sh

repo_home=$1
registry_name=$2
build_new="${3:-1}"
push_remote="${4:-1}"

# setup shell functions for conda, uses conda's .bashrc resident defined hook to execute conda init setup to enable
#   subsequent conda command usage
eval "$(conda shell.bash hook)"
conda deactivate

d=`date +%Y%m%d%H%M%S`
tmp_docker_build_log_dir="/tmp"
docker_build_log="${tmp_docker_build_log_dir}/fts_build_docker_release_images_${d}.log"

maybe_push(){
    if [[ $push_remote -ne 0 ]]; then
        echo "Beginning upload of built images..." >> $docker_build_log
		echo "Pushing ${registry_name}:${latest_pt} ..." >> $docker_build_log
        docker push ${registry_name}:${latest_pt} >> $docker_build_log
		echo "Finished pushing ${registry_name}:${latest_pt} ..." >> $docker_build_log
		echo "Pushing ${registry_name}:${latest_azpl} ..." >> $docker_build_log
		docker push ${registry_name}:${latest_azpl} >> $docker_build_log
		echo "Finished pushing ${registry_name}:${latest_azpl} ..." >> $docker_build_log
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
	declare -A iv=(["cuda"]="12.4.0" ["python"]="3.12" ["pytorch"]="2.4.1" ["lightning"]="2.4" ["cust_build"]="0")
	export latest_pt="base-cu${iv["cuda"]}-py${iv["python"]}-pt${iv["pytorch"]}-pl${iv["lightning"]}"
	export latest_azpl="py${iv["python"]}-pt${iv["pytorch"]}-pl${iv["lightning"]}-azpl-init"
	maybe_build iv "${latest_pt}" "${latest_azpl}"
	if [[ $build_new -ne 0 ]]; then
		echo "All image builds successful." >> $docker_build_log
	else
        echo "Directed to skip building of new images, now checking whether to push..." >> $docker_build_log
    fi
}


cd ${repo_home}
echo "Building and/or pushing images for repository home: ${repo_home}" >> $docker_build_log
build_eval
maybe_push
echo "Finished building and/or updating images for repository home: ${repo_home}" >> $docker_build_log
