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

ARG CUDA_VERSION=11.6

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ARG CUDATOOLKIT_VERSION=11.6
ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=1.12
ARG CONDA_VERSION=4.13.0

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
    # LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MAKEFLAGS="-j2" \
    TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6" \
    CONDA_ENV=finetuning-scheduler \
    CONDA_DEFAULT_ENV=${CONDA_ENV}

LABEL maintainer="Dan Dale <https://github.com/speediedan>"

WORKDIR /home/finetuning-scheduler
COPY ./tests ./tests
COPY ./requirements ./requirements
COPY ./src ./src
COPY ./requirements.txt ./setup.py ./setup.cfg ./pyproject.toml ./README.md ./

ENV PATH=/root/miniconda3/envs/${CONDA_ENV}/bin:$PATH \
    CONDA_DEFAULT_ENV=${CONDA_ENV}
# conda init
RUN conda update -n base -c defaults conda && \
    conda create -y --name $CONDA_ENV && \
    conda init bash

SHELL ["conda", "run", "--no-capture-output", "-n", "finetuning-scheduler", "/bin/bash", "-c"]

RUN conda install -c pytorch -c nvidia python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION} cudatoolkit=${CUDATOOLKIT_VERSION} && \
    pip install ".[all]" && \
    conda clean -ya && \
    rm -rf requirements.* requirements/

COPY ./dockers/release-conda/conda_entrypoint.sh ./conda_entrypoint.sh
RUN echo "conda activate ${CONDA_ENV}" >> ~/.bashrc
RUN pip --version && \
    conda info && \
    pip list && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)" && \
    python -c "import finetuning_scheduler as fts; print(fts.__version__)"
ENTRYPOINT ["./conda_entrypoint.sh"]
