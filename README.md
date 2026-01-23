<div align="center">

<img src="docs/source/_static/images/logos/logo_fts.png" width="401px">

**A PyTorch Lightning extension that enhances model experimentation with flexible fine-tuning schedules.**

______________________________________________________________________

<p align="center">
  <a href="https://finetuning-scheduler.readthedocs.io/en/stable/">Docs</a> •
  <a href="#Setup">Setup</a> •
  <a href="#examples">Examples</a> •
  <a href="#community">Community</a>
</p>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/finetuning-scheduler)](https://pypi.org/project/finetuning-scheduler/)
[![PyPI Status](https://badge.fury.io/py/finetuning-scheduler.svg)](https://badge.fury.io/py/finetuning-scheduler)\
[![codecov](https://codecov.io/gh/speediedan/finetuning-scheduler/branch/main/graph/badge.svg?flag=gpu)](https://codecov.io/gh/speediedan/finetuning-scheduler)
[![ReadTheDocs](https://readthedocs.org/projects/finetuning-scheduler/badge/?version=latest)](https://finetuning-scheduler.readthedocs.io/en/stable/)
[![DOI](https://zenodo.org/badge/455666112.svg)](https://zenodo.org/badge/latestdoi/455666112)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/speediedan/finetuning-scheduler/blob/master/LICENSE)

</div>

______________________________________________________________________

<img width="300px" src="docs/source/_static/images/fts/fts_explicit_loss_anim.gif" alt="FinetuningScheduler explicit loss animation" align="right"/>

[FinetuningScheduler](https://finetuning-scheduler.readthedocs.io/en/stable/api/finetuning_scheduler.fts.html#finetuning_scheduler.fts.FinetuningScheduler) is simple to use yet powerful, offering a number of features that facilitate model research and exploration:

- easy specification of flexible fine-tuning schedules with explicit or regex-based parameter selection
  - implicit schedules for initial/naive model exploration
  - explicit schedules for performance tuning, fine-grained behavioral experimentation and computational efficiency
- automatic restoration of best per-phase checkpoints driven by iterative application of early-stopping criteria to each fine-tuning phase
- composition of early-stopping and manually-set epoch-driven fine-tuning phase transitions

______________________________________________________________________

## Setup

### Step 0: Install from PyPI

Starting with version 2.10, [uv](https://docs.astral.sh/uv/) is the preferred installation approach for Fine-Tuning Scheduler.

```bash
# Install uv if you haven't already (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Fine-Tuning Scheduler
uv pip install finetuning-scheduler
```

<!-- following section will be skipped from PyPI description -->

<details>
  <summary>Additional installation options</summary>
    <!-- following section will be skipped from PyPI description -->

#### *Install Optional Packages*

#### To install additional packages required for examples:

```bash
uv pip install finetuning-scheduler['examples']
```

#### or to include packages for examples, development and testing:

```bash
uv pip install finetuning-scheduler['all']
```

#### *Source Installation Examples*

#### Using the build script (recommended):

The `build_fts_env.sh` script automatically handles Lightning commit pinning and optional PyTorch nightly installation:

```bash
git clone https://github.com/speediedan/finetuning-scheduler.git
cd finetuning-scheduler

# Standard development build (handles Lightning pin automatically and installs an optional PyTorch prerelease if configured in `requirements/ci/torch-pre.txt`)
./scripts/build_fts_env.sh --repo-home=${PWD} --target-env-name=fts_latest --venv-dir=/path/to/.venvs

# To configure PyTorch prerelease used by the build scripts, edit `requirements/ci/torch-pre.txt`:
#   Line 1: torch version (e.g., 2.10.0 for test/RC or 2.10.0.dev20251203 for nightly)
#   Line 2: CUDA target (e.g., cu128) — CI uses cpu
#   Line 3: channel type: "test" or "nightly"

# Note: `manage_standalone_processes.sh` is an optional wrapper to run long-running coverage/build scripts with checks for other concurrent conflicting processes, and run options (--use-nohup etc) — you can also run `build_fts_env.sh` and `gen_fts_coverage.sh` directly.

# Activate (use your venv base path)
export FTS_VENV_BASE=~/.venvs  # or /mnt/cache/${USER}/.venvs
export FTS_VENV_NAME=fts_latest
source ${FTS_VENV_BASE}/${FTS_VENV_NAME}/bin/activate
```

#### Manual installation with Lightning commit pin:

```bash
git clone https://github.com/speediedan/finetuning-scheduler.git
cd finetuning-scheduler

# Set UV_OVERRIDE to use the pinned Lightning commit
export UV_OVERRIDE=${PWD}/requirements/ci/overrides.txt
uv pip install -e ".[all]"
```

## Manual installation with a PyTorch prerelease (nightly/test)

When using a PyTorch prerelease (nightly or test), use a two-step installation approach:

```bash
git clone https://github.com/speediedan/finetuning-scheduler.git
cd finetuning-scheduler

# Step 1: Install a PyTorch prerelease (adjust version and CUDA target as needed; see configuration in requirements/ci/torch-pre.txt)
# Example (nightly):
uv pip install --prerelease=if-necessary-or-explicit torch==2.10.0.dev20251124 --index-url https://download.pytorch.org/whl/nightly/cu128

# Step 2: Install FTS with Lightning commit pin (torch already installed, will be skipped)
export UV_OVERRIDE=${PWD}/requirements/ci/overrides.txt
uv pip install -e ".[all]"
```

The prerelease version is configured via `requirements/ci/torch-pre.txt`. The `lock_ci_requirements.sh` and `build_fts_env.sh` scripts will read this file to determine whether to pre-install a prerelease PyTorch during builds.

#### Install a specific FTS version from source using the standalone `pytorch-lighting` package:

```bash
export FTS_VERSION=2.6.0
export PACKAGE_NAME=pytorch
git clone -b v${FTS_VERSION} https://github.com/speediedan/finetuning-scheduler
cd finetuning-scheduler
export UV_OVERRIDE=${PWD}/requirements/ci/overrides.txt
uv pip install -e ".[all]"
```

#### *Latest Docker Image*

Note, publishing of new `finetuning-scheduler` version-specific docker images was paused after the `2.0.2` patch release. If new version-specific images are required, please raise an issue.

![Docker Image Version (tag latest semver)](https://img.shields.io/docker/v/speediedan/finetuning-scheduler/latest?color=%23000080&label=docker)

</details>

<!-- end skipping PyPI description -->

### Step 1: Import the FinetuningScheduler callback and start fine-tuning!

```python
import lightning as L
from finetuning_scheduler import FinetuningScheduler

trainer = L.Trainer(callbacks=[FinetuningScheduler()])
```

Get started by following [the Fine-Tuning Scheduler introduction](https://finetuning-scheduler.readthedocs.io/en/stable/index.html) which includes a [CLI-based example](https://finetuning-scheduler.readthedocs.io/en/stable/index.html#example-scheduled-fine-tuning-for-superglue) or by following the [notebook-based](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/finetuning-scheduler.html) Fine-Tuning Scheduler tutorial.

______________________________________________________________________

### Installation Using the Standalone `pytorch-lightning` Package

*applicable to versions >= `2.0.0`*

Now that the core Lightning package is `lightning` rather than `pytorch-lightning`, Fine-Tuning Scheduler (FTS) by default depends upon the `lightning` package rather than the standalone `pytorch-lightning`. If you would like to continue to use FTS with the standalone `pytorch-lightning` package instead, you can still do so as follows:

Install a given FTS release (for example v2.0.0) using standalone `pytorch-lightning`:

```bash
export FTS_VERSION=2.0.0
export PACKAGE_NAME=pytorch
wget https://github.com/speediedan/finetuning-scheduler/releases/download/v${FTS_VERSION}/finetuning-scheduler-${FTS_VERSION}.tar.gz
uv pip install finetuning-scheduler-${FTS_VERSION}.tar.gz
```

### Dynamic Versioning

FTS (as of version `2.6.0`) now enables dynamic versioning both at installation time and via CLI post-installation. Initially, the dynamic versioning system allows toggling between Lightning unified and standalone imports. The two conversion operations are individually idempotent and mutually reversible.

#### Toggling Between Unified and Standalone Lightning Imports

FTS provides a simple CLI tool to easily toggle between unified and standalone import installation versions post-installation:

```bash
# Toggle from unified to standalone Lightning imports
toggle-lightning-mode --mode standalone

# Toggle from standalone to unified Lightning imports (default)
toggle-lightning-mode --mode unified
```

> **Note:** If you have the standalone package (`pytorch-lightning`) installed but not the unified package (`lightning`), toggling to unified mode will be prevented. You must install the `lightning` package first before toggling.

This can be useful when:

- You need to adapt existing code to work with a different Lightning package
- You're switching between projects using different Lightning import styles
- You want to test compatibility with both import styles

______________________________________________________________________

## Examples

### Scheduled Fine-Tuning For SuperGLUE

- [Notebook-based Tutorial](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/finetuning-scheduler.html)
- [CLI-based Tutorial](https://finetuning-scheduler.readthedocs.io/en/stable/#example-scheduled-fine-tuning-for-superglue)
- [FSDP Scheduled Fine-Tuning](https://finetuning-scheduler.readthedocs.io/en/stable/advanced/fsdp_scheduled_fine_tuning.html)
- [LR Scheduler Reinitialization](https://finetuning-scheduler.readthedocs.io/en/stable/advanced/lr_scheduler_reinitialization.html) (advanced)
- [Optimizer Reinitialization](https://finetuning-scheduler.readthedocs.io/en/stable/advanced/optimizer_reinitialization.html) (advanced)

______________________________________________________________________

## Continuous Integration

Fine-Tuning Scheduler is rigorously tested across multiple CPUs, GPUs and against major Python and PyTorch versions.

**Versioning Policy (Updated in 2.9)**: Starting with the 2.9 minor release, Fine-Tuning Scheduler is pivoting from tight Lightning version alignment to **core PyTorch version alignment**. This change:

- Provides greater flexibility to integrate the latest PyTorch functionality increasingly important in research
- Reduces maintenance burden while continuing to support the stable Lightning API and robust integration
- Officially supports **at least the latest 4 PyTorch minor releases** (e.g., when PyTorch 2.9 is released, FTS supports >= 2.6)

This versioning approach is motivated by Lightning's evolving release cadence (see [Lightning Issue #21073](https://github.com/Lightning-AI/pytorch-lightning/issues/21073) and [PR #21107](https://github.com/Lightning-AI/pytorch-lightning/pull/21107)) and allows FTS to adopt new PyTorch capabilities more rapidly while maintaining clear deprecation policies.

See the [versioning documentation](https://finetuning-scheduler.readthedocs.io/en/stable/versioning.html) for complete details on compatibility policies and migration guidance.

**Prior Versioning (\< 2.9)**: Each Fine-Tuning Scheduler minor release (major.minor.patch) was paired with a Lightning minor release (e.g., Fine-Tuning Scheduler 2.0 depends upon Lightning 2.0). To ensure maximum stability, the latest Lightning patch release fully tested with Fine-Tuning Scheduler was set as a maximum dependency in Fine-Tuning Scheduler's requirements.txt (e.g., \<= 1.7.1).

<details>
  <summary>Current build statuses for Fine-Tuning Scheduler </summary>

| System / (PyTorch/Python ver) |                                                                                                        2.6.0/3.10                                                                                                        |                                                                                                             2.10.0/3.10, 2.10.0/3.13                                                                                                             |
| :---------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      Linux \[GPUs\*\*\]       |                                                                                                            -                                                                                                             | [![Build Status](https://dev.azure.com//speediedan/finetuning-scheduler/_apis/build/status/Multi-GPU%20&%20Example%20Tests?branchName=main)](https://dev.azure.com/speediedan/finetuning-scheduler/_build/latest?definitionId=1&branchName=main) |
|     Linux (Ubuntu 22.04)      | [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml) |             [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml)             |
|           OSX (14)            | [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml) |             [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml)             |
|        Windows (2022)         | [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml) |             [![Test](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml/badge.svg?branch=main&event=push)](https://github.com/speediedan/finetuning-scheduler/actions/workflows/ci_test-full.yml)             |

- \*\* tests run on one RTX 4090 and one RTX 2070

</details>

## Community

Fine-Tuning Scheduler is developed and maintained by the community in close communication with the [Lightning team](https://pytorch-lightning.readthedocs.io/en/stable/governance.html). Thanks to everyone in the community for their tireless effort building and improving the immensely useful core Lightning project.

PR's welcome! Please see the [contributing guidelines](https://finetuning-scheduler.readthedocs.io/en/stable/generated/CONTRIBUTING.html) (which are essentially the same as Lightning's).

______________________________________________________________________

## Citing Fine-Tuning Scheduler

Please cite:

```tex
@misc{Dan_Dale_2022_6463952,
    author       = {Dan Dale},
    title        = {{Fine-Tuning Scheduler}},
    month        = Feb,
    year         = 2022,
    doi          = {10.5281/zenodo.6463952},
    publisher    = {Zenodo},
    url          = {https://zenodo.org/record/6463952}
    }
```

Feel free to star the repo as well if you find it useful or interesting. Thanks 😊!
