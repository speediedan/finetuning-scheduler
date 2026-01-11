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

ARG PYTHON_VERSION=3.13
ARG PYTORCH_VERSION=2.10.0
ARG LIGHTNING_VERSION=2.6
ARG CUST_BASE

FROM speediedan/finetuning-scheduler:base-${CUST_BASE}py${PYTHON_VERSION}-pt${PYTORCH_VERSION}-pl${LIGHTNING_VERSION}

LABEL maintainer="Dan Dale <https://github.com/speediedan>"

RUN apt-get update \
 && apt-get install -y --no-install-recommends sudo \
 && rm -rf /var/lib/apt/lists/*
RUN useradd -c 'azpl_init user' -d /home/azpl_init -m -k /etc/skel -s /bin/bash -ou 0 -g 0 azpl_init

COPY dockers/fts-az-base/.bashrc /home/azpl_init/.bashrc
USER azpl_init
