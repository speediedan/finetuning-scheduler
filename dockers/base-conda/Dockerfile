# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VERSION=11.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=1.10
ARG CONDA_VERSION=4.11.0

SHELL ["/bin/bash", "-c"]
ENV \
    PATH="$PATH:/root/.local/bin" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=US/Pacific

RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
    && \
    curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_${CONDA_VERSION}-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh && \
# Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*
ENV \
    PATH="/root/miniconda3/bin:$PATH" \
    LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MAKEFLAGS="-j$(nproc)" \
    TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5" \
    CONDA_ENV=finetuning-scheduler

# conda init
RUN conda update -n base -c defaults conda && \
    conda create -y --name $CONDA_ENV -c pytorch -c nvidia python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION} cudatoolkit=${CUDA_VERSION} && \
    conda init bash && \
    conda clean -ya

ENV \
    PATH=/root/miniconda3/envs/${CONDA_ENV}/bin:$PATH \
    LD_LIBRARY_PATH="/root/miniconda3/envs/${CONDA_ENV}/lib:$LD_LIBRARY_PATH" \
    CONDA_DEFAULT_ENV=${CONDA_ENV}

COPY ./requirements/extra.txt requirements-extra.txt
COPY ./requirements/examples.txt requirements-examples.txt
COPY ./requirements/test.txt requirements-test.txt
COPY ./requirements/pl_adjust_versions.py requirements_adjust_versions.py

RUN \
    pip list | grep torch && \
    python -c "import torch; print(torch.__version__)" && \
    python requirements_adjust_versions.py requirements-extra.txt && \
    python requirements_adjust_versions.py requirements-examples.txt && \
    # Install remaining requirements
    pip install -r requirements-extra.txt --no-cache-dir && \
    pip install -r requirements-examples.txt --no-cache-dir && \
    pip install -r requirements-test.txt --no-cache-dir && \
    rm requirements*

RUN \
    CUDA_VERSION_MAJOR=$(python -c "import torch ; print(torch.version.cuda.split('.')[0])") && \
    py_ver=$(python -c "print(int('$PYTHON_VERSION'.split('.') >= '3.9'.split('.')))")

RUN \
    pip --version && \
    conda info && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__"
