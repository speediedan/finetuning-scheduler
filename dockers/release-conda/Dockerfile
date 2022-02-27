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

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=1.10

FROM base-conda-py3.9-pt1.10-pl1.6

LABEL maintainer="Dan Dale <https://github.com/speediedan>"

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
WORKDIR /home/finetuning-scheduler
COPY ./tests ./tests
COPY ./requirements ./requirements
COPY ./fts_examples ./fts_examples
COPY ./finetuning_scheduler ./finetuning_scheduler
COPY ./requirements.txt ./setup.py ./setup.cfg ./pyproject.toml ./README.md ./

SHELL ["conda", "run", "--no-capture-output", "-n", "finetuning-scheduler", "/bin/bash", "-c"]

# upgrade cudatoolkit due to 11.1 issues
# RUN conda install -c pytorch -c nvidia python=3.9 pytorch=1.10 cudatoolkit=11.3 -y
RUN conda install -c pytorch -c nvidia mkl numpy python=3.9 pytorch=1.10 cudatoolkit=11.3 -y
# install dependencies
RUN \
    pip install ".[all]" && \
    conda clean -ya && \
    rm -rf requirements.* requirements/

RUN python --version && \
    pip --version && \
    pip list && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)" && \
    python -c "import finetuning_scheduler as fts; print(fts.__version__)"

ENV CONDA_DEFAULT_ENV=${CONDA_ENV}
COPY ./dockers/release-conda/conda_entrypoint.sh ./conda_entrypoint.sh
RUN echo "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64" >> ~/.bashrc && \
    echo "conda activate ${CONDA_ENV}" >> ~/.bashrc
ENTRYPOINT ["./conda_entrypoint.sh"]
