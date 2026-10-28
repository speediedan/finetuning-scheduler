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
#
# ROCm twin of dockers/base-cuda (#57): the same venv path, torch exclusion file and permissions contract, so
# dockers/fts-az-base builds on it unchanged with CUST_BASE=rocm<version>-. The torch wheel is PyTorch's own ROCm build
# of the exact PYTORCH_VERSION, from download.pytorch.org/whl/[<channel>/]rocm<version>, never a default-index torch
# (which would silently be a CUDA build).
#
# No ROCm base image: the ROCm torch wheels depend on the rocm-sdk wheels, which carry the HIP runtime, the math
# libraries and amd-smi, and torch loads those rather than any /opt/rocm. FTS builds no extensions from source, so a
# second ROCm userspace (rocm/dev, about 15 GB) would add nothing. The host supplies only the kernel driver and the
# device nodes (/dev/kfd, /dev/dri).

ARG OS_VER=24.04

FROM ubuntu:${OS_VER}

# Names the wheel index (rocm10.0 for torch 2.15, read from download.pytorch.org/whl).
ARG ROCM_VERSION=10.0
ARG PYTHON_VERSION=3.13
ARG PYTORCH_VERSION=2.15.0
# "" for a released torch; "test" for a release candidate, whose wheels are published only on that index.
ARG TORCH_CHANNEL=""
ARG MKL_THREADING_LAYER=GNU

SHELL ["/bin/bash", "-c"]
ENV \
    DEBIAN_FRONTEND=noninteractive \
    TZ=US/Pacific \
    PATH="$PATH:/root/.local/bin" \
    MKL_THREADING_LAYER=${MKL_THREADING_LAYER} \
    MAKEFLAGS="-j2" \
    UV_HTTP_TIMEOUT=120
RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
        gnupg \
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
        libnuma1 \
        libdrm2 \
    && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements/ ./requirements/

# One layer for the whole venv, permissions included: a chmod in a later layer would copy every file of the venv again,
# and the venv is most of the image.
#   - rocm_amdsmi.pth: torch counts and names devices through the amdsmi Python module before HIP initializes, which is
#     what lets ddp_fork start its workers without an initialized parent. The rocm-sdk core wheel ships that module
#     beside its own libamd_smi but does not put it on sys.path, so the .pth does (the host venvs point theirs at
#     /opt/rocm instead).
#   - torch-excludes.txt keeps the image's ROCm torch from being replaced by a PyPI (CUDA) wheel when CI installs
#     requirements.txt, exactly as in dockers/base-cuda.
#   - chmod lets userns-remapped rootless Docker users update the venv, as in dockers/base-cuda.
RUN \
    uv venv --python python${PYTHON_VERSION} /tmp/venvs/fts_dev && \
    . /tmp/venvs/fts_dev/bin/activate && \
    uv pip install --prerelease=allow "torch==${PYTORCH_VERSION}+rocm${ROCM_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL:+${TORCH_CHANNEL}/}rocm${ROCM_VERSION}" && \
    uv pip install -U six && \
    site=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    test -f "${site}/_rocm_sdk_core/share/amd_smi/amdsmi/__init__.py" && \
    echo "${site}/_rocm_sdk_core/share/amd_smi" > "${site}/rocm_amdsmi.pth" && \
    python -c "import amdsmi; print('amdsmi', amdsmi.__file__)" && \
    rm -rf requirements.* requirements/ /root/.cache && \
    chmod -R 777 /tmp/venvs/fts_dev && \
    echo "torch" > /tmp/venvs/fts_dev/torch-excludes.txt && \
    chmod 644 /tmp/venvs/fts_dev/torch-excludes.txt

# Backend truth is torch.version.hip, not the device API (the HIP build keeps torch.cuda): a CUDA wheel slipping in
# fails the build here rather than a test run on the AMD agent. No device is needed for this check.
RUN \
    set -x && \
    . /tmp/venvs/fts_dev/bin/activate && \
    uv --version && \
    uv pip list && \
    python -c "import sys; ver = sys.version_info; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; v = torch.__version__; assert v.startswith('$PYTORCH_VERSION'), v; \
assert torch.version.hip and torch.version.cuda is None, (torch.version.hip, torch.version.cuda); \
print('torch', v, 'hip', torch.version.hip, 'git', torch.version.git_version)"
