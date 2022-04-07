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

<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/finetuning-scheduler)](https://pypi.org/project/finetuning-scheduler/)
[![PyPI Status](https://badge.fury.io/py/finetuning-scheduler.svg)](https://badge.fury.io/py/finetuning-scheduler) -->

[![codecov](https://codecov.io/gh/speediedan/finetuning-scheduler/branch/main/graph/badge.svg)](https://codecov.io/gh/speediedan/finetuning-scheduler)
[![ReadTheDocs](https://readthedocs.org/projects/finetuning-scheduler/badge/?version=latest)](https://finetuning-scheduler.readthedocs.io/en/latest/)
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

#### To install additional packages required for examples:

```bash
pip install finetuning-scheduler['examples']
```

#### or to include packages for Examples, Development and Testing:

```bash
pip install finetuning-scheduler['all']
```

#### To install from source (editable) using a custom version of pytorch-lightning (includes docs as well)

#### **Note**: minimum supported pytorch-lightning release is 1.6.0

```bash
git clone https://github.com/speediedan/finetuning-scheduler.git@release/0.1.0
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

## Community

Finetuning Scheduler is developed and maintained by the community in close communication with the [PyTorch Lightning team](https://pytorch-lightning.readthedocs.io/en/latest/governance.html#leads). Thanks to everyone in the community for their tireless effort building and improving the immensely useful core PyTorch Lightning project.

PR's welcome! Please see the [contributing guidelines](https://finetuning-scheduler.readthedocs.io/en/latest/generated/CONTRIBUTING.html) (which are essentially the same as PyTorch Lightning's).
