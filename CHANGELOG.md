# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

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
