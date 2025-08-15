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

ARG CUDA_VERSION=12.8.1
ARG OS_VER=ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS_VER}

ARG PYTHON_VERSION=3.12
ARG PYTORCH_VERSION=2.9.0
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
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0" \
    MKL_THREADING_LAYER=${MKL_THREADING_LAYER} \
    MAKEFLAGS="-j2"
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
ENV USE_CI_COMMIT_PIN="1"

RUN \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate | python${PYTHON_VERSION} && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    python${PYTHON_VERSION} -m pip install --upgrade pip setuptools && \
    # Disable cache
    pip config set global.cache-dir false && \
    pip install virtualenv && \
    mkdir /tmp/venvs && \
    python -m virtualenv -p python${PYTHON_VERSION} /tmp/venvs/fts_dev && \
    . /tmp/venvs/fts_dev/bin/activate && \
    # set particular PyTorch version by default
    if [[ "${CUST_BUILD}" -eq 0 ]]; then \
        CUDA_VERSION_MM=${CUDA_VERSION%.*}; \
        pip install torch --no-cache-dir \
        --find-links="https://download.pytorch.org/whl/cu${CUDA_VERSION_MM//'.'/''}/torch_stable.html" \
        --find-links="https://download.pytorch.org/whl/test/cu${CUDA_VERSION_MM//'.'/''}/torch" \
        --find-links="https://download.pytorch.org/whl/test/cu${CUDA_VERSION_MM//'.'/''}/pytorch-triton"; \
    else \
        # or target a specific cuda build, by specifying a particular index url w/...
        # ... default channel
        #pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128; \
        # ... pytorch patch version
        # pip install torch==1.11.1+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html; \
        # ... pytorch nightly dev version
        pip install --pre torch==2.9.0.dev20250811 --index-url https://download.pytorch.org/whl/nightly/cu128; \
        # ... test channel
        # pip install --pre torch==2.9.0 --index-url https://download.pytorch.org/whl/test/cu128; \
    fi && \
    # We avoid installing Lightning and other dependencies here as they are usually upgraded anyway later in
    # CI but we may re-enable in the future.
    # LIGHTNING_COMMIT=$(cat ./requirements/lightning_pin.txt) && \
    # pip install "lightning @ git+https://github.com/Lightning-AI/lightning.git@${LIGHTNING_COMMIT}#egg=lightning" --no-cache-dir && \
    # Install all requirements
    # pip install -r requirements/devel.txt --no-cache-dir && \
    # Update six
    pip install -U six --no-cache-dir && \
    chmod -R 777 /tmp/venvs/fts_dev && \
    rm -rf requirements.* requirements/

RUN \
    set -x && \
    . /tmp/venvs/fts_dev/bin/activate && \
    echo "Checking CUDA version:" && \
    CUDA_VERSION_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])") && \
    echo "CUDA Version Major: ${CUDA_VERSION_MAJOR}" && \
    echo "Checking Python version compatibility:" && \
    py_ver=$(python -c "print(int('$PYTHON_VERSION'.split('.') >= '3.12'.split('.')))") && \
    echo "Python version check result: ${py_ver}"

RUN \
    set -x && \
    echo "============= Environment Information =============" && \
    . /tmp/venvs/fts_dev/bin/activate && \
    echo "Pip version:" && \
    pip --version && \
    echo "Installed packages:" && \
    pip list && \
    echo "Verifying Python version:" && \
    python -c "import sys; ver = sys.version_info; print(f'Python {ver.major}.{ver.minor} detected'); assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    echo "Verifying PyTorch version:" && \
    python -c "import torch; print(f'PyTorch {torch.__version__} detected'); assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__" && \
    echo "============= Environment Verification Complete ============="
