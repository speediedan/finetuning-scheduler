<div align="center">

<img src="docs/source/_static/images/logos/logo_fts.png" width="400px">

**A PyTorch Lightning extension that enhances model experimentation with flexible finetuning schedules.**

______________________________________________________________________

<p align="center">
  <a href="https://finetuning-scheduler.readthedocs.io/en/latest/">Docs</a> •
  <a href="#Setup">Setup</a> •
  <a href="#examples">Examples</a> •
  <a href="#community">Community</a>
</p>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/finetuning-scheduler)](https://pypi.org/project/finetuning-scheduler/)
[![PyPI Status](https://badge.fury.io/py/finetuning-scheduler.svg)](https://badge.fury.io/py/finetuning-scheduler)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/finetuning-scheduler?color=%23000080)
[![Docker Hub](https://img.shields.io/badge/docker-latest-000080.svg?logo=data:image/png;base64,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)](https://hub.docker.com/repository/docker/speediedan/finetuning-scheduler)\
[![codecov](https://codecov.io/gh/speediedan/finetuning-scheduler/branch/main/graph/badge.svg)](https://codecov.io/gh/speediedan/finetuning-scheduler)
[![ReadTheDocs](https://readthedocs.org/projects/finetuning-scheduler/badge/?version=latest)](https://finetuning-scheduler.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/455666112.svg)](https://zenodo.org/badge/latestdoi/455666112)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/speediedan/finetuning-scheduler/blob/master/LICENSE)

</div>

______________________________________________________________________

<img width="300px" src="docs/source/_static/images/fts/fts_explicit_loss_anim.gif" alt="FinetuningScheduler explicit loss animation" align="right"/>

[FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) is simple to use yet powerful, offering a number of features that facilitate model research and exploration:

- easy specification of flexible finetuning schedules with explicit or regex-based parameter selection
  - implicit schedules for initial/naive model exploration
  - explicit schedules for performance tuning, fine-grained behavioral experimentation and computational efficiency
- automatic restoration of best per-phase checkpoints driven by iterative application of early-stopping criteria to each finetuning phase
- composition of early-stopping and manually-set epoch-driven finetuning phase transitions

______________________________________________________________________

## Setup

### Step 0: Install from PyPI

```bash
pip install finetuning-scheduler
```

<!-- following section will be skipped from PyPI description -->

<details>
  <summary>Additional installation options</summary>
    <!-- following section will be skipped from PyPI description -->

#### *Install Optional Packages*

#### To install additional packages required for examples:

```bash
pip install finetuning-scheduler['examples']
```

#### or to include packages for examples, development and testing:

```bash
pip install finetuning-scheduler['all']
```

#### *Conda Installation*

**Note:** prefer latest tested pytorch and cuda toolkit by including official pytorch and nvidia channels:

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch cudatoolkit=11.3 finetuning-scheduler
```

#### *Source Installation*

#### To install from source (editable) using a custom version of pytorch-lightning (includes docs as well)

#### **Note**: minimum supported pytorch-lightning release is 1.6.0

```bash
# update the url below with the desired release #, e.g. for finetuning-scheduler release 0.1.2:
git clone https://github.com/speediedan/finetuning-scheduler.git@release/0.1.2
cd finetuning-scheduler
python -m pip install -e ".[all]" -r requirements/docs.txt
```

</details>

<!-- end skipping PyPI description -->

### Step 1: Import the FinetuningScheduler callback and start finetuning!

```python
from pytorch_lightning import Trainer
from finetuning_scheduler import FinetuningScheduler

trainer = Trainer(callbacks=[FinetuningScheduler()])
```

Get started by following [the Finetuning Scheduler introduction](https://finetuning-scheduler.readthedocs.io/en/latest/index.html) which includes a [CLI-based example](https://finetuning-scheduler.readthedocs.io/en/latest/index.html#scheduled-finetuning-superglue) or by following the notebook-based Finetuning Scheduler tutorial (link will be added as soon as it is released on the PyTorch Lightning production site).

______________________________________________________________________

## Examples

### Scheduled Finetuning For SuperGLUE

- Notebook-based Tutorial (link will be added as soon as it is released on the PyTorch Lightning production site)
- [CLI-based Tutorial](https://finetuning-scheduler.readthedocs.io/en/latest/#scheduled-finetuning-superglue)

______________________________________________________________________

## Continuous Integration

Finetuning Scheduler is rigorously tested across multiple CPUs, GPUs and against major Python and PyTorch versions. Each Finetuning Scheduler minor release (major.minor.patch) is paired with a PyTorch Lightning minor release (e.g. Finetuning Scheduler 0.1 depends upon PyTorch Lightning 1.6).

To ensure maximum stability, the latest PyTorch Lightning patch release fully tested with Finetuning Scheduler is set as a maximum dependency in Finetuning Scheduler's requirements.txt (e.g. \<= 1.6.1). If you'd like to test a specific PyTorch Lightning patch version greater than that currently in Finetuning Scheduler's requirements.txt, it will likely work but you should install Finetuning Scheduler from source and update the requirements.txt as desired.

<details>
  <summary>Current build statuses for Finetuning Scheduler </summary>

|   System / PyTorch ver   |                                                                                                                 1.8 (LTS, min. req.)                                                                                                                  |                                                                                                      1.11 (latest)                                                                                                       |
| :----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Linux py3.9 \[GPUs\*\*\] | [![Build Status](https://dev.azure.com//speediedan/finetuning-scheduler/_apis/build/status/Multi-GPU%20&%20Example%20Tests?branchName=main)](https://dev.azure.com/PytorchLightning/pytorch-lightning/_build/latest?definitionId=6&branchName=master) |                                                                                                            -                                                                                                             |
|     Linux py3.{7,9}      |               [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml)                | [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml) |
|      OSX py3.{7,9}       |               [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml)                | [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml) |
|    Windows py3.{7,9}     |               [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml)                | [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml) |

- \*\* tests run on two RTX 2070s

</details>

## Community

Finetuning Scheduler is developed and maintained by the community in close communication with the [PyTorch Lightning team](https://pytorch-lightning.readthedocs.io/en/latest/governance.html#leads). Thanks to everyone in the community for their tireless effort building and improving the immensely useful core PyTorch Lightning project.

PR's welcome! Please see the [contributing guidelines](https://finetuning-scheduler.readthedocs.io/en/latest/generated/CONTRIBUTING.html) (which are essentially the same as PyTorch Lightning's).

______________________________________________________________________

## Citing Finetuning Scheduler

Please cite:

```tex
@misc{Dan_Dale_2022_6463952,
    author       = {Dan Dale},
    title        = {{Finetuning Scheduler}},
    month        = Feb,
    year         = 2022,
    doi          = {10.5281/zenodo.6463952},
    publisher    = {Zenodo},
    url          = {https://zenodo.org/record/6463952}
    }
```

Feel free to star the repo as well if you find it useful or interesting. Thanks 😊!
