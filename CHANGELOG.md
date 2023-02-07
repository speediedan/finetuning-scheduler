# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [0.4.0] - 2023-01-25

### Added

- **FSDP Scheduled Fine-Tuning** is now supported! [See the tutorial here.](https://finetuning-scheduler.readthedocs.io/en/latest/advanced/fsdp_scheduled_fine_tuning.html)
- Introduced [``StrategyAdapter``](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.strategy_adapters.html#finetuning_scheduler.strategy_adapters.StrategyAdapter)s. If you want to extend Fine-Tuning Scheduler (FTS) to use a custom, currently unsupported strategy or override current FTS behavior in the context of a given training strategy, subclassing ``StrategyAdapter`` is now a way to do so. See [``FSDPStrategyAdapter``](https://finetuning-scheduler.readthedocs.io/en/latest/api/finetuning_scheduler.strategy_adapters.html#finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter) for an example implementation.
- support for `pytorch-lightning` 1.9.0

### Changed

- decomposed ``add_optimizer_groups`` to accommodate the corner case where FTS is being used without an lr scheduler configuration, also cleanup unrequired example testing warning exceptions
- updated the fts repo issue template


### Fixed

- removed PATH adjustments that are no longer necessary due to https://github.com/Lightning-AI/lightning/pull/15485

### Removed

- removed references to the ``finetuning-scheduler`` conda-forge package (at least temporarily) due to the current unavailability of upstream dependencies (i.e. the [pytorch-lightning conda-forge package](https://anaconda.org/conda-forge/pytorch-lightning/files) ). Installation of FTS via pip within a conda env is the recommended installation approach (both in the interim and in general).


## [0.3.4] - 2023-01-24

### Added

- support for `pytorch-lightning` 1.8.6
- Notify the user when ``max_depth`` is reached and provide the current training session stopping conditions. Resolves [#7](https://github.com/speediedan/finetuning-scheduler/issues/7).


### Changed

- set package version ceilings for the examples requirements along with a note regarding their introduction for stability
- promoted PL CLI references to top-level package

### Fixed

- replaced deprecated ``Batch`` object reference with ``LazyDict``


## [0.3.3] - 2022-12-09

### Added

- support for `pytorch-lightning` 1.8.4

### Changed

- pinned `jsonargparse` dependency to <4.18.0 until [#205](https://github.com/omni-us/jsonargparse/issues/205) is fixed

## [0.3.2] - 2022-11-18

### Added

- support for `pytorch-lightning` 1.8.2

## [0.3.1] - 2022-11-10

### Added

- support for `pytorch-lightning` 1.8.1
- augmented `standalone_tests.sh` to be more robust to false negatives

### Changed

- added temporary expected `distutils` warning until fixed upstream in PL
- updated `depth` type hint to accommodate updated mypy default config
- bumped full test timeout to be more conservative given a dependent package that is currently slow to install in some contexts (i.e. `grpcio` on MacOS 11 with python `3.10`)

## [0.3.0] - 2022-11-04

### Added

- support for pytorch-lightning 1.8.0
- support for python 3.10
- support for PyTorch 1.13
- support for `ZeroRedundancyOptimizer`

### Fixed

- call to PL `BaseFinetuning.freeze` did not properly hand control of `BatchNorm` module thawing to FTS schedule. Resolves [#5](https://github.com/speediedan/finetuning-scheduler/issues/5).
- fixed codecov config for azure pipeline gpu-based coverage

### Changed

- Refactored unexpected and expected multi-warning checks to use a single test helper function
- Adjusted multiple FTS imports to adapt to reorganized PL/Lite imports
- Refactored fts-torch collect_env interface to allow for (slow) collect_env evolution on a per-torch version basis
- Bumped required jsonargparse version
- adapted to PL protection of `_distributed_available`
- made callback setup stage arg mandatory
- updated mypy config to align with PL `Trainer` handling
- updated dockerfile defs for PyTorch 1.13 and python 3.10
- updated github actions versions to current versions
- excluded python 3.10 from torch 1.9 testing due to incompatibility

### Deprecated

- removed use of deprecated `LightningCLI` `save_config_overwrite` in PL 1.8


## [0.2.3] - 2022-10-01

### Added

- support for pytorch-lightning 1.7.7
- add new temporary HF expected warning to examples
- added HF `evaluate` dependency for examples

### Changed

- Use HF `evaluate.load()` instead of `datasets.load_metric()`

## [0.2.2] - 2022-09-17

### Added

- support for pytorch-lightning 1.7.6
- added detection of multiple instances of a given callback dependency parent
- add new expected warning to examples

### Fixed

- import fts to workaround pl TypeError via sphinx import, switch to non-TLS pytorch inv object connection due to current certificate issues

### Changed

- bumped pytorch dependency in docker image to 1.12.1

## [0.2.1] - 2022-08-13

### Added

- support for pytorch-lightning 1.7.1
- added support for ReduceLROnPlateau lr schedulers
- improved user experience with additional lr scheduler configuration inspection (using an allowlist approach) and
  enhanced documentation. Expanded use of ``allow_untested`` to allow use of unsupported/untested lr schedulers
- added initial user-configured optimizer state inspection prior to phase ``0`` execution, issuing warnings to the user
  if appropriate. Added associated documentation [#4](https://github.com/speediedan/finetuning-scheduler/issues/4)

### Fixed

- pruned test_examples.py from wheel

### Changed

- removed a few unused internal conditions relating to lr reinitialization and parameter group addition

## [0.2.0] - 2022-08-06

### Added

- support for pytorch-lightning 1.7.0
- switched to [src-layout project structure](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)
- increased flexibility of internal package management
- added a patch to examples to allow them to work with torch 1.12.0 despite issue #80809
- added sync for test log calls for multi-gpu testing

### Fixed

- adjusted runif condition for examples tests
- minor type annotation stylistic correction to avoid jsonargparse issue fixed in
  [#148](https://github.com/omni-us/jsonargparse/pull/148)

### Changed

- streamlined MANIFEST.in directives
- updated docker image dependencies
- disable mypy unused ignore warnings due to variable behavior depending on ptl installation method
  (e.g. pytorch-lightning vs full lightning package)
- changed full ci testing on mac to use macOS-11 instead of macOS-10.15
- several type-hint mypy directive updates
- unpinned protobuf in requirements as no longer necessary
- updated cuda docker images to use pytorch-lightning 1.7.0, torch 1.12.0 and cuda-11.6
- refactored mock strategy test to use a different mock strategy
- updated pyproject.toml with jupytext metadata bypass configuration for nb test cleanup
- updated ptl external class references for ptl 1.7.0
- narrowed scope of runif test helper module to only used conditions
- updated nb tutorial links to point to stable branch of docs
- unpinned jsonargparse and bumped min version to 4.9.0
- moved core requirements.txt to requirements/base.txt and update load_requirements and setup to reference lightning
  meta package
- update azure pipelines ci to use torch 1.12.0
- renamed instantiate_registered_class meth to instantiate_class due to ptl 1.7 deprecation of cli registry
  functionality

### Deprecated

- removed ddp2 support
- removed use of ptl cli registries in examples due to its deprecation

## [0.1.8] - 2022-07-13

### Added

- enhanced support and testing for lr schedulers with lr_lambdas attributes
- accept and automatically convert schedules with non-integer phase keys (that are convertible to integers) to integers

### Fixed

- pinned jsonargparse to be <= 4.10.1 due to regression with PTL cli with 4.10.2
### Changed

- updated PL links for new lightning-ai github urls
- added a minimum hydra requirement for cli usage (due to omegaconf version incompatibility)
- separated cli requirements
- replace closed compound instances of `finetuning` with the hyphenated compound version `fine-tuning` in textual
  contexts. (The way language evolves, `fine-tuning` will eventually become `finetuning` but it seems like the research
  community prefers the hyphenated form for now.)
- update fine-tuning scheduler logo for hyphenation
- update strategy resolution in test helper module runif

### Deprecated

## [0.1.7] - 2022-06-10
### Fixed

- bump omegaconf version requirement in examples reqs (in addition to extra reqs) due to omegaconf bug

### Added

### Changed

### Deprecated

## [0.1.6] - 2022-06-10

### Added

- Enable use of untested strategies with new flag and user warning
- Update various dependency minimum versions
- Minor example logging update

### Fixed
- minor privacy policy link update
- bump omegaconf version requirement due to omegaconf bug

### Changed

### Deprecated

## [0.1.5] - 2022-06-02

### Added

- Bumped latest tested PL patch version to 1.6.4
- Added basic notebook-based example tests a new ipynb-specific extra
- Updated docker definitions
- Extended multi-gpu testing to include both oldest and latest supported PyTorch versions
- Enhanced requirements parsing functionality
### Fixed
- cleaned up acknowledged warnings in multi-gpu example testing
### Changed

### Deprecated

## [0.1.4] - 2022-05-24

### Added

- Added LR scheduler reinitialization functionality ([#2](https://github.com/speediedan/finetuning-scheduler/pull/2))
- Added advanced usage documentation
- Added advanced scheduling examples
- added notebook-based tutorial link
- enhanced cli-based example hparam logging among other code clarifications

### Changed

### Fixed

- addressed URI length limit for custom badge
- allow new deberta fast tokenizer conversion warning for transformers >= 4.19
### Deprecated

## [0.1.3] - 2022-05-04

### Added

-

### Changed

- bumped latest tested PL patch version to 1.6.3
### Fixed

-
### Deprecated

-

## [0.1.2] - 2022-04-27

### Added

- added multiple badges (docker, conda, zenodo)
- added build status matrix to readme

### Changed

- bumped latest tested PL patch version to 1.6.2
- updated citation cff configuration to include all version metadata
- removed tag-based trigger for azure-pipelines multi-gpu job

### Fixed

-
### Deprecated

-

## [0.1.1] - 2022-04-15

### Added

- added conda-forge package
- added docker release and pypi workflows
- additional badges for readme, testing enhancements for oldest/newest pl patch versions

### Changed

- bumped latest tested PL patch version to 1.6.1, CLI example depends on PL logger fix ([#12609](https://github.com/Lightning-AI/lightning/pull/12609))

### Deprecated

-

### Fixed

- Addressed version prefix issue with readme transformation for pypi


## [0.1.0] - 2022-04-07

### Added

- None (initial release)

### Changed

- None (initial release)

### Deprecated

- None (initial release)

### Fixed

- None (initial release)
