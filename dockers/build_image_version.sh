#!/bin/bash
#
# Configurable base/azpl_init image building function shared by both main and release FTS CI

build_version(){
	local -n iv_ref="$1"
	base_name=$2
	azpl_name=$3
	# build base image
	echo "Building base image name: ${base_name}" >> $docker_build_log
	docker image build -t ${base_name} -f dockers/base-cuda/Dockerfile --build-arg CUDA_VERSION=${iv_ref['cuda']} \
	  --build-arg PYTHON_VERSION=${iv_ref['python']} --build-arg PYTORCH_VERSION=${iv_ref['pytorch']} \
	  --build-arg CUST_BUILD=${iv_ref['cust_build']} --no-cache . >> $docker_build_log
	docker tag ${base_name} ${registry_name}:${base_name} >> $docker_build_log
	# build associated azpl-init image
	echo "Building azpl-init image name: ${azpl_name}" >> $docker_build_log
	docker image build -t ${azpl_name} -f dockers/fts-az-base/Dockerfile \
	  --build-arg PYTHON_VERSION=${iv_ref["python"]} --build-arg CUST_BASE=cu${iv_ref["cuda"]}- \
	  --build-arg PYTORCH_VERSION=${iv_ref["pytorch"]} --no-cache . >> $docker_build_log
	docker tag ${azpl_name} ${registry_name}:${azpl_name} >> $docker_build_log
}

maybe_deactivate(){
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
}
