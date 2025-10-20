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

ARG PYTHON_VERSION=3.12
ARG PYTORCH_VERSION=2.9.0
ARG LIGHTNING_VERSION=2.5
ARG CUST_BASE

FROM speediedan/finetuning-scheduler:base-${CUST_BASE}py${PYTHON_VERSION}-pt${PYTORCH_VERSION}-pl${LIGHTNING_VERSION}

LABEL maintainer="Dan Dale <https://github.com/speediedan>"

WORKDIR /home/finetuning-scheduler
COPY ./tests ./tests
COPY ./requirements ./requirements
COPY ./src ./src
COPY ./.actions ./.actions
COPY ./requirements.txt ./setup.py ./pyproject.toml ./README.md ./

# install dependencies
RUN \
    pip install -U pip && \
    pip install ".[all]" --no-cache-dir && \
    rm -rf requirements.* requirements/

RUN python --version && \
    pip --version && \
    pip list && \
    python -c "import lightning as L; print(L.__version__)" && \
    python -c "import finetuning_scheduler as fts; print(fts.__version__)"
