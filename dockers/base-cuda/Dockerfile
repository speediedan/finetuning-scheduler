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
# initially based on https://bit.ly/3pdAf1G

ARG CUDA_VERSION=12.1.0
ARG OS_VER=ubuntu20.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS_VER}

ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION=2.1.0
ARG CUST_BUILD=0
ARG MKL_THREADING_LAYER=GNU

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    CUST_BUILD=${CUST_BUILD} \
    DEBIAN_FRONTEND=noninteractive \
    TZ=US/Pacific \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6" \
    MKL_THREADING_LAYER=${MKL_THREADING_LAYER} \
    MAKEFLAGS="-j2"
# temporary fix for https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update -qq --fix-missing  && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        cmake \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        software-properties-common \
        libopenmpi-dev \
    && \
# Install python
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
        python3-pip \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt requirements.txt
COPY ./requirements/ ./requirements/

ENV PYTHONPATH=/usr/lib/python${PYTHON_VERSION}/site-packages

RUN \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate | python${PYTHON_VERSION} && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    python${PYTHON_VERSION} -m pip install --upgrade pip  && \
    # Disable cache
    pip config set global.cache-dir false && \
    pip install virtualenv && \
    mkdir /tmp/venvs && \
    python -m virtualenv -p python${PYTHON_VERSION} /tmp/venvs/fts_dev && \
    . /tmp/venvs/fts_dev/bin/activate && \
    # set particular PyTorch version by default
    if [[ "${CUST_BUILD}" -eq 0 ]]; then \
        python ./requirements/pl_adjust_versions.py requirements/base.txt ${PYTORCH_VERSION}; \
        python ./requirements/pl_adjust_versions.py requirements/extra.txt ${PYTORCH_VERSION}; \
        python ./requirements/pl_adjust_versions.py requirements/examples.txt ${PYTORCH_VERSION}; \
    else \
        # or target a specific cuda build of latest stable packages
        # pip install torch==1.11.1+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html; \
        # pip install --pre torch==2.1.0.dev20230807  torchvision==0.16.0.dev20230807 -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html; \
        #pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118; \
        #pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html; \
        pip install --pre torch torchvision -f https://download.pytorch.org/whl/test/cu121/torch_test.html; \
    fi && \
    # Install all requirements
    pip install -r requirements/devel.txt --no-cache-dir && \
    # Update six to avoid unnecessary warnings associated with 3.10: https://github.com/benjaminp/six/pull/343
    pip install -U six --no-cache-dir && \
    chmod -R 777 /tmp/venvs/fts_dev && \
    rm -rf requirements.* requirements/

RUN \
    . /tmp/venvs/fts_dev/bin/activate && \
    CUDA_VERSION_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])") && \
    py_ver=$(python -c "print(int('$PYTHON_VERSION'.split('.') >= '3.10'.split('.')))")

RUN \
    # Show what we have
    . /tmp/venvs/fts_dev/bin/activate && \
    pip --version && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__"
