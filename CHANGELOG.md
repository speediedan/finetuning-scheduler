# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [2.15.0] - 2026-XX-XX

### Added

- `MemProfiler` (`fts_examples.profiling`) no longer assumes CUDA: it collects accelerator memory statistics through `torch.accelerator.memory_stats` (adding any statistic only the backend's own `memory_stats` reports, such as `num_oom_rejections` on CUDA) and allocator history through the current backend's memory module (`torch.cuda.memory` on CUDA and ROCm, `torch.xpu.memory` on XPU), and skips a metric the backend does not provide with a warning naming the backend rather than failing. The extended PyTorch Profiler records the accelerator's activity likewise. The profiling guide documents which metrics each backend provides. The `rank_<n>_memory_stats.yaml` dump is now a plain mapping that `yaml.safe_load` reads [#104](https://github.com/speediedan/finetuning-scheduler/issues/104).

- Added an FSDP1-to-FSDP2 migration guide: how an FSDP1 strategy configuration (wrap and activation checkpointing policies, `awp_overrides`, sharding strategy, mixed precision, CPU offload, state dict type) maps to `ModelParallelStrategy` with an `fsdp_plan`, why FSDP2 makes most phase-aligned wrapping unnecessary (a unit spanning phases costs all-gathers only when it reshards after forward, and the outermost unit does not), and how FSDP1 checkpoints resume under FSDP2, with the FSDP1 DeBERTa example translated [#58](https://github.com/speediedan/finetuning-scheduler/issues/58).

- Added self-hosted tutorial notebooks, rendered with their outputs in the docs and published with a Colab badge: the Lightning tutorial (formerly hosted in the Lightning tutorials, whose link had broken) and a new PyTorch API version that runs the same explicit, implicit and baseline scenarios in a plain training loop driven by `FTSController`. Both take papermill parameters for shortened runs. They are published and rendered by notebook tooling vendored unmodified from interpretune (`scripts/vendored/notebook_tooling`, checked by `scripts/check_vendored_notebook_tooling.py`), and the docs render notebooks with `myst-nb` in place of `nbsphinx` [#105](https://github.com/speediedan/finetuning-scheduler/issues/105).

- `ModelParallelStrategyAdapter.fsdp_plan` keys may select modules by class: `class:TransformerBlock` (or `class:my_pkg.blocks.TransformerBlock`) selects every instance of that class, subclasses included, as `ModuleWrapPolicy` does. An `fsdp_plan` entry's `ignored_params` may name parameters by name or regex (or a list of them), resolved when the plan is applied, so a YAML plan can use it [#55](https://github.com/speediedan/finetuning-scheduler/issues/55).

- Added the `pg_overrides` and `phase_config` schedule keys to the Lightning callback: a phase after the first applies its `pg_overrides` (optimizer options other than `params` and `lr`) to the parameter groups it adds (phase 0's groups come from `configure_optimizers`, so phase 0's `pg_overrides` is refused), and `FinetuningScheduler.add_phase_hook(fn, when="start"|"end")` (experimental) calls `fn` with each phase's `PhaseEvent`, whose `phase_config` is the phase's key: start hooks for phase 0 at fit start (not when a fit resumes) and for each later phase at its transition, end hooks when a phase ends (after the best checkpoint is restored) and for the final phase at train end. Strategy adapters gain an `on_fts_teardown` hook, called when a fit ends or fails [#85](https://github.com/speediedan/finetuning-scheduler/issues/85).

- Added an experimental externally driven mode to `FTSController`, for a training host that owns the loop and its own phase execution: `attach(executor, schedule)` adopts the host's optimizer and lr schedulers instead of `setup()`, `request_advance(reason)` sequences a transition the host decided on (the best-checkpoint restore, then end hooks, gates and start hooks, as for a controller-driven controller), and `resume_at(depth)` puts a resumed run at its depth; the loop methods, `save`/`load`/`load_state_dict` and gates are refused there. A `PhaseExecutor` (`finetuning_scheduler.core.executor`) carries out each phase; `ControllerExecutor` is the controller's own. A checkpoint policy may define `restore()` to restore the best checkpoint itself. `finetuning_scheduler.lightning.controller_bridge` provides the Lightning side (`LightningPhaseExecutor`, which executes phases through the callback's own `step_pg`, `thaw_to_depth` and the strategy adapter's `on_after_phase_transition`; `FTSCheckpointPolicy`, which restores through `restore_best_ckpt`; and `attach_controller`), and transitions sequenced through it dispatch to strategy adapters exactly as the callback's own. This is the groundwork for running the Lightning callback on the controller [#85](https://github.com/speediedan/finetuning-scheduler/issues/85).

- Added `finetuning_scheduler.strategies.plan.shard_model(model, mesh, plan, *, default_kwargs=None, shard_root=True)`, which applies an FSDP2 plan in the `ModelParallelStrategyAdapter.fsdp_plan` format (module names, regexes and `class:` keys; `ignored_params` by name; the `composable` `act_ckpt` mode; `cpu_offload_policy`, which works here, unlike under Lightning's `ModelParallelStrategy`) to a plain PyTorch model and then shards the root module. The Lightning adapter resolves its plan with the same module, and `shard_model` refuses a plan that lists a module after one of its ancestors. `FSDP2Adapter` validation, and so both `FTSController` and the Lightning `ModelParallelStrategyAdapter`, now warns about each FSDP unit that reshards after forward and holds parameters thawing in different phases, naming the phases: from the earliest of them on, its backward pass all-gathers the parameters not yet trained too. The Lightning FSDP2 auto-plan example gives `model.norm` its own unit accordingly. Tests now cover resuming an FSDP1 checkpoint under `ModelParallelStrategy` (a full checkpoint directly, a sharded one after Lightning's `consolidate_checkpoint`), and the model-parallel guide documents that path [#55](https://github.com/speediedan/finetuning-scheduler/issues/55).

- Added `FTSController.export_best(fn=None, *, roles=("adapter",), path=None)`, exporting the best checkpoint's tensors whose role matches `roles` (for example only the adapters), from the checkpoint rather than the live model (keyed by wrapper-free names, for a compiled or activation-checkpointed model too) and, for a sharded model, in full on global rank zero. It writes `{tensors, roles, metadata}` with `torch.save`, or calls `fn(tensors, metadata, path)` to write another format. `FTSController(artifacts=...)` records the external artifacts a run depends on (`{name: {path, sha256, revision}}`, validated at construction) in every checkpoint's and export's metadata [#52](https://github.com/speediedan/finetuning-scheduler/issues/52).

- Added `finetuning_scheduler.strategies`, experimental torch-native distributed adapters (`DistributedAdapter`): `UnshardedAdapter`, `FSDP2Adapter` and `DTensorAdapter` recognize a model's layout, check a schedule's physical parameters against it, and read and write the training state's tensors (one file, or a `torch.distributed.checkpoint` directory written by every rank). `FTSController` takes `distributed=` (detected by default) and delegates its sharded checkpoint IO to the adapter; the package imports only PyTorch and the scheduling core [#54](https://github.com/speediedan/finetuning-scheduler/issues/54).

- Added `TunableView` support to the Lightning callback: a view attached to the `LightningModule` as `fts_tunable_view` names the schedule's parameters, as it does for `FTSController`. The schedule compiles through the view (its logical names, coupled tensors and roles), keeps its logical names, and thaws and optimizes the physical parameters they stand for; `role:` selectors are accepted (also without a view, through the module's own names, but not under an adapter that renames parameters through the parameter-naming methods, which refuses them by name); a view declaring `trainable=False` is refused with its reason, as is one not rooted at the `LightningModule`; and schedule generation uses the view's logical names and leaves out tensors with frozen roles. A strategy adapter that overrides the parameter-naming methods keeps naming parameters itself and takes precedence over an attached view, with a `FutureWarning` naming both. The FSDP (FSDP1) strategy accepts an attached view whose naming scheme is `"physical"`, applying its roles and coupled tensors (without `use_orig_params`, its wrap validation refuses an FSDP-flattened parameter that mixes a `fitted` or `instrument` tensor with scheduled parameters, even of the same module), and refuses one with logical names [#107](https://github.com/speediedan/finetuning-scheduler/issues/107); `ModelParallelStrategy` validates the compiled physical parameters through the distributed adapters, so it works under a view [#54](https://github.com/speediedan/finetuning-scheduler/issues/54).

- Added promotion to the experimental PyTorch API: a phase's `promote` key, `{selector_or_name: "latent:<name>"}`, promotes `instrument` tensors to a `latent` role when the phase starts (after its start hooks, before its thaw), so that phase or a later one can schedule them through `role:latent:<name>`. A promoted parameter that shares storage with another tensor (a tied weight) gets its own copy, and a view can own the promotion through its `promote` member. Promotions are recorded in checkpoint metadata and in the controller's state, replayed before a saved state loads, and checkpointed from then on. Restoring a checkpoint saved before a promotion (a best checkpoint, or the rollback point of a vetoed phase) puts every checkpoint entry the promotion added (the promoted tensors and any alias it untied) back to its value at promotion; those values are kept on the CPU for the rest of the run and saved with `save()`. Promotion refuses on a sharded model, a view that overrides its checkpoint state must define `promote`, and the Lightning callback refuses the key. The conformance suite gains a `promotion` case [#53](https://github.com/speediedan/finetuning-scheduler/issues/53).

- Added top-k checkpointing to the experimental PyTorch API: `BestCheckpoint(save_top_k=...)` keeps the `save_top_k` best checkpoints (`-1` keeps every one) with Lightning's `ModelCheckpoint` admission and eviction rules, and lists them in `best_k_models`. The default, `save_top_k=1`, keeps today's behavior. Every checkpoint now gets a unique name and is never rewritten, and a phase's rollback point is pinned by reference rather than copied [#52](https://github.com/speediedan/finetuning-scheduler/issues/52).

- Added a warning when an FSDP2 data-parallel run under `FTSController` leaves scheduled parameters outside every FSDP2 module (a plan that does not shard the root module), including tensor-parallel `DTensor` parameters, since nothing synchronizes their gradients across data-parallel ranks [#87](https://github.com/speediedan/finetuning-scheduler/issues/87).

- Added an FSDP2 and tensor-parallel example for the experimental PyTorch API (`fts_examples/model_parallel/torch_mp_example.py`), the plain-loop counterpart of the Lightning model-parallel example. A `torchrun` script with no Lightning import, it builds the same Llama model on the meta device, applies the tensor-parallel plan and then an FSDP2 plan over a data-parallel by tensor-parallel device mesh, and runs the same schedule with `FTSController`, `PlateauTransition`, `max_transition_epoch` and `BestCheckpoint`, using loss parallelism under tensor parallelism. The plan helpers and the random-token dataset the two examples share moved to a Lightning-free module, `fts_examples/model_parallel/parallel_plans.py` (still importable from their old locations). The example shards the model with `shard_model` (below), which applies the Lightning example's `fsdp_plan` format and shards the root module last, so no parameter is left unsharded [#88](https://github.com/speediedan/finetuning-scheduler/issues/88).

- Added `TunableView`, an experimental protocol that lets a schedule address a model through logical parameter names (for example the names an interpretability library gives a wrapped model, where one logical name can stand for several physical tensors). A view needs four members (`root`, `support()`, `to_physical()`, `to_logical()`); optional members (`coupled()`, `depth_groups()`, `roles()`, checkpoint state and others) default to Fine-Tuning Scheduler's implementations, so a view need not inherit from any Fine-Tuning Scheduler class. `CanonicalView` is the identity view every plain model gets, and resolves a parametrized tensor by its unparametrized name. Parameter and buffer roles (`base`, `adapter[:name]`, `intervention[:name]`, `latent:<name>`, `fitted`, `instrument`) and `role:` selectors are defined in `finetuning_scheduler.core.roles`. `finetuning_scheduler.core.compile.compile_schedule` compiles a schedule's logical names once into the physical parameters each phase thaws, refusing schedules whose phases share storage or that schedule `fitted` or `instrument` tensors. `finetuning_scheduler.testing.conformance.TunableViewConformance` checks a view implementation against the contract (it needs neither pytest nor Lightning, and no case applying is a failure), and `finetuning_scheduler.testing.fixtures` provides four known-good fixture views. `FTSController` accepts a `view` (or finds one attached to the model as `fts_tunable_view`), compiles the schedule through it, thaws and optimizes physical parameters, never thaws or optimizes `fitted` or `instrument` tensors, leaves `instrument` tensors out of its checkpoints, and refuses to resume from state compiled through a different view [#53](https://github.com/speediedan/finetuning-scheduler/issues/53).

- Added checkpointing to the experimental PyTorch API: a `BestCheckpoint` policy (top-1 by a monitored metric, optionally saving only scheduled parameters), restoring the best checkpoint's model, optimizer and lr scheduler state at phase boundaries (`restore_best`, as the Lightning callback does), phase gates that can veto a phase and roll back to the best as of its start, `register_stateful` for data-loader positions and similar state, and `save()`/`load()` for exact resume (random streams included). Sharded (FSDP2/DTensor) models are checkpointed through `torch.distributed.checkpoint` [#51](https://github.com/speediedan/finetuning-scheduler/issues/51).

- Added the `finetuning-scheduler[lightning]` and `finetuning-scheduler[pytorch-lightning]` extras. While Lightning remains a base dependency they install nothing more; adopting them now makes the planned move of Lightning behind the extra (no earlier than 2.17) a no-op [#51](https://github.com/speediedan/finetuning-scheduler/issues/51).

- Added `FTSController`, an experimental PyTorch API for scheduled fine-tuning in a plain training loop. The loop stays the user's: `setup()`, `epochs()` (or `epoch_start()`/`epoch_end()`) and `step_end()` drive phase transitions. It covers schedules loaded, generated or given as mappings; optimizer and lr scheduler factories or instances (with phase-0 enforcement); `max_transition_epoch` and the new `max_transition_step` transitions; pluggable transition policies (`PlateauTransition`, `EpochTransition`, `StepTransition`, `AnyOf`, `CallableTransition`) agreed across ranks; optimizer and lr scheduler reinitialization; per-phase `phase_config` payloads and `pg_overrides`; phase hooks run with the training random streams isolated; batch fingerprints; resumable state; and a configuration hash. A `TrainingHost` protocol (`TorchHost`; `LightningHost` for a Lightning `Trainer`) is the seam between the scheduling core and the loop [#51](https://github.com/speediedan/finetuning-scheduler/issues/51).

- Added `finetuning_scheduler.core`, a scheduling core that does not depend on Lightning. It holds schedule generation, loading, saving and validation; optimizer and lr scheduler reinitialization checks; parameter group growth; lr state restoration; and the scheduler state. It imports only PyTorch, PyYAML, fsspec and the standard library, and the Lightning integration delegates to it. The core raises configuration errors as `FTSConfigurationError`, a `ValueError`. Under Lightning they are raised as `FTSMisconfigurationException`, which is both an `FTSConfigurationError` and a Lightning `MisconfigurationException`, so existing `except MisconfigurationException` handlers keep catching them, and `except ValueError` now catches them too [#50](https://github.com/speediedan/finetuning-scheduler/issues/50).

- Added `scripts/rocm_smoke.py`, a self-contained distributed smoke check (PyTorch only) for vetting a multi-GPU host, ROCm in particular, before relying on it: checked collectives and bandwidth, a collective soak, a two-phase FSDP2 schedule against a single-process run, bf16 mixed precision, a distributed-checkpoint restore and compilation at a phase transition, each run as configured and with `NCCL_PROTO=Simple` [#74](https://github.com/speediedan/finetuning-scheduler/issues/74).

- Added a persistent distributed test pool (`tests/helpers/dist_pool.py`) that serves many distributed test cases from one process group per test module instead of launching processes per test. Validity rules enforced around every case keep cases from inheriting each other's state, a failure on any rank fails the case, and an error, timeout or violation respawns the pool [#56](https://github.com/speediedan/finetuning-scheduler/issues/56). Under `coverage run`, the pool measures its workers with the same settings and merges their data when it closes, so code that only distributed cases exercise counts toward coverage [#57](https://github.com/speediedan/finetuning-scheduler/issues/57).

- Added phase manifests and phase reports for the GPU CI phases (`tests/gpu_ci/phase_manifest.py`, with the pytest plugin `tests/gpu_ci/phase_plugin.py`). `collect` writes, per phase (`gpu-nonstandalone`, `standalone`, `examples`, `examples-multigpu` and the scheduled `pool-reverse`), the node ids its paths collect and the ones its selector keeps, on the CPU and executing nothing. `report` counts the per-node, per-rank outcomes the plugin recorded while the phase ran against that manifest: a failure on any rank, or a rank that started and never finished, fails the node, and a test that ran but is not in the manifest is a collection error. Each report names its evidence class (`pipeline` or `parity-pre-agent`, for runs made outside the pipeline), and manifests are hashed with the rule the release checker applies. Coverage is part of a phase's evidence: each phase the pipeline measures gets its own coverage data file, the report records its JSON export (file names relative to the repository) with the device set and the exact torch and Lightning builds, and `coverage-diff` lists the lines one run executed and another did not, so a second GPU vendor can be held to the first's coverage. They serve both the cross-host parity runs that precede a second, AMD GPU agent and that agent's pipeline phases [#57](https://github.com/speediedan/finetuning-scheduler/issues/57).

- Added a ROCm CI image beside the CUDA one (`dockers/base-rocm`, built with the azpl-init layer by `dockers/docker_images_main.sh` unless its fifth argument is `0`): the same Python, venv path and torch exclusion file, with PyTorch's own ROCm wheel and no ROCm base image, since that wheel brings the ROCm runtime it loads. Added an any-vendor GPU job template (`.azure-pipelines/templates/gpu-job.yml`), a host job that picks its digest-pinned image and device options from the vendor the agent declares, runs the GPU phases of the multi-GPU pipeline in that container under the host leases, and fails when any phase's report is not clean, with a manual pipeline (`.azure-pipelines/gpu-proof.yml`) that admits an agent to it. The skips and expected failures the GPU phases have on every vendor are listed with their reasons in `tests/gpu_ci/reviewed_exclusions.json`, which both pipelines' phase reports apply, so a phase is reported clean only when each of its exclusions is reviewed [#57](https://github.com/speediedan/finetuning-scheduler/issues/57).

- Added a GPU release-validation pipeline (`.azure-pipelines/gpu-release.yml`), queued by hand on a release candidate: one full leg per vendor, each demanding its vendor's agent, then a CPU job that checks both legs' phase evidence with `tests/gpu_ci/release_eligibility.py`. The candidate is eligible only when every release phase on each vendor is clean, at the candidate's revision, from a pipeline run on a digest-pinned image under the host's lease tool, and the two vendors collected and selected the same tests on the same Fine-Tuning Scheduler, Lightning and torch release, built from the same torch and Lightning commits. Outcome counts are recounted from each phase's per-node records, exclusions must each carry a reviewed reason, and a measured phase's coverage export must match its report's digest; the verdict is published as the `fts-release-eligibility` artifact. `scripts/verify_version_consistency.py` now also checks that every GPU pipeline pins the same image digests [#57](https://github.com/speediedan/finetuning-scheduler/issues/57).

### Fixed

- Fixed `TunableViewConformance`'s `thaw_is_exact_and_deterministic` case building a fresh view (through `make_view`) for every logical name it samples. A view over a heavyweight wrapper is a whole model each, so a GPT-2 target with about 150 logical names took 25 GB of memory and failed on Windows; the case now reuses one view, which checks the same thing [#124](https://github.com/speediedan/finetuning-scheduler/issues/124).

- Fixed the check that a configured optimizer matches phase 0 calling the strategy adapter's name inspection once per module parameter instead of once: the FSDP adapter's inspection walks the module, so with `use_orig_params=True` the check was quadratic in the parameter count (about 12 s at 4,000 parameters). Phase thawing and parameter-group construction also test names against sets rather than lists now [#122](https://github.com/speediedan/finetuning-scheduler/issues/122).

- Fixed `tests/special_tests.sh` exiting 1 without its summary when every test passed but the log mentioned an `xfailed` test: its error scan detected with one pattern and printed with another, and the print finding nothing ended the summary trap under `set -eo pipefail` [#116](https://github.com/speediedan/finetuning-scheduler/issues/116).

- Fixed resuming an FSDP1 (`FSDPStrategy`) checkpoint under `ModelParallelStrategy` (FSDP2): restoring the best checkpoint at the next phase transition failed with a `KeyError`, since the checkpoint named its parameter groups' parameters with FSDP1's wrapper segments (`_fsdp_wrapped_module.`). Saved and current parameters are now matched by wrapper-free name. A best checkpoint that is an FSDP1 sharded directory, which `ModelParallelStrategy` cannot load, is restored from its consolidation beside it (`<directory>.consolidated`, the default output of Lightning's `consolidate_checkpoint` utility), and without one a resume that will restore it is refused up front with the command to run [#55](https://github.com/speediedan/finetuning-scheduler/issues/55).

- Fixed resuming a multi-GPU fit deadlocking. A distributed checkpoint's callback state is restored onto rank zero's device on every rank, so on every other rank `FTSEarlyStopping`'s improvement message (subtracting `best_score` from the current metric) and, with an unchanged checkpoint directory, `FTSCheckpoint`'s top-k comparisons raised at the first validation, and rank zero waited in the next collective. Restored scores are now moved to the CPU on load [#112](https://github.com/speediedan/finetuning-scheduler/issues/112).

- Fixed a resumed fit whose best checkpoint was saved at an earlier phase losing, at its next phase transition, the optimizer parameter groups of the phases in between, so their parameters were no longer optimized: on resume, the callback (and `FTSCheckpoint`, when the checkpoint directory changed) took the resumed phase as the best checkpoint's, while the checkpoint restored was the earlier one. The phases replayed onto the restored optimizer state now start after the best checkpoint's own phase [#110](https://github.com/speediedan/finetuning-scheduler/issues/110).

- Fixed the FSDP (FSDP1) strategy adapter disabling Lightning's optimizer restoration for every later fit in the process: it patched Lightning's `Strategy` class itself. It now patches the running strategy's own class, and restores it when the fit ends or fails [#111](https://github.com/speediedan/finetuning-scheduler/issues/111).

- Fixed the callback restoring, at a phase transition, a best checkpoint saved before an explicit optimizer reinitialization into the reinitialized optimizer: the old optimizer's state and parameter-group hyperparameters replaced the new optimizer's, and an optimizer of a different class failed at its next step (for example `KeyError: 'betas'` after a switch from `SGD` to `Adam`). When a reinitialization since the best checkpoint changed the optimizer's class, the checkpoint now restores the model state only, and the optimizer is rebuilt as the latest reinitialization configures it, as `FTSController` decides by class; a checkpoint whose optimizer has the class of the one in use restores as before [#108](https://github.com/speediedan/finetuning-scheduler/issues/108).

- Fixed the callback skipping the best-checkpoint restore at the first phase transition after resuming a fit with `trainer.fit(model, ckpt_path=...)`. A resume was detected in `setup`, before Lightning sets `trainer.ckpt_path` for a path passed to `fit()`, so only the `trainer.ckpt_path = ...` form resumed correctly; it is now detected when Lightning restores the callbacks, whichever form is used. On resume, `FTSCheckpoint` also restores `best_model_score`, drops top-k entries whose files no longer exist with a warning, and raises at once when the best checkpoint it would restore at the next transition has been removed, rather than failing at that transition [#86](https://github.com/speediedan/finetuning-scheduler/issues/86).

- Fixed `FTSController` restoring the wrong best checkpoint, or the wrong rollback point after a gate's veto, when resuming from a `save()` taken before the run wrote a newer best and was interrupted: the newer files had replaced the ones the save referred to. Checkpoints now have unique names, the checkpoints the latest save refers to are kept until the next save, and `load()` raises `FileNotFoundError` when a resumed state's best checkpoint or rollback point has been removed [#52](https://github.com/speediedan/finetuning-scheduler/issues/52).

- Fixed every rank of a distributed `FTSController` run acting as rank zero (each warning, logging and writing the saved schedule) once the Lightning integration had been imported, under a launcher that passes ranks to `init_process_group` directly and sets no rank environment variable. The Lightning integration's rank-zero backend now takes the rank from the process group whenever one is initialized, as the core's default backend already did [#97](https://github.com/speediedan/finetuning-scheduler/issues/97).

- Fixed the callback thawing unsharded copies of the root FSDP2 module's parameters after a validation pass, so they never trained, when `restore_best` is off and no checkpoint is written at that validation. The callback now reshards every FSDP2 module before it restores and thaws parameters at a phase transition [#92](https://github.com/speediedan/finetuning-scheduler/issues/92).

- Fixed `FTSController` thawing FSDP2's unsharded parameters, which then never trained: FSDP2 can leave a module's unsharded parameters registered between steps (a root module's after a forward-only evaluation pass, or after a training forward when all of its parameters were frozen). The controller now reshards every FSDP2 module before it freezes or thaws parameters, restores a checkpoint or writes one [#90](https://github.com/speediedan/finetuning-scheduler/issues/90).

- Fixed `FTSController` accepting a `DistributedDataParallel` wrapper on a submodule, or a module under torch's composable `replicate()`, neither of which synchronizes parameters thawed after phase 0. Both are now refused, as a `DistributedDataParallel`-wrapped model already was [#89](https://github.com/speediedan/finetuning-scheduler/issues/89).

- Fixed the gradients of parameters thawed after phase 0 never being synchronized across processes under the DDP strategies (`ddp`, `ddp_spawn`, `ddp_fork`, `ddp_notebook` and the `ddp_find_unused_parameters_*` variants). `DistributedDataParallel` registers with its gradient reducer only the parameters that require grad when it wraps the module, so each process trained the later phases' parameters on its own data alone and the replicas diverged silently. Fine-Tuning Scheduler now rebuilds the `DistributedDataParallel` wrapper whenever a phase transition (or resuming at a later depth) changes the trainable parameters, through a new `StrategyAdapter.on_after_phase_transition` hook. Custom strategy adapters inherit the fix; an adapter that overrides the new hook should call `super()` [#72](https://github.com/speediedan/finetuning-scheduler/issues/72).

- Fixed two configuration error messages missing a space between sentences ("which is aninvalid configuration" and "schedulecould not be cast"). [#51](https://github.com/speediedan/finetuning-scheduler/issues/51)

- Fixed the examples' environment logging (`collect_env_info`) failing under PyTorch `2.15`, which added a `rocm_compiled_version` field to `torch.utils.collect_env.SystemEnv`. The examples build that tuple themselves to avoid initializing CUDA; they now pass only the fields the installed PyTorch defines and mark any they do not inspect, so a future field no longer breaks every example.

- Fixed a reinitialization `class_path` without a module path (for example `class_path: SGD`) raising an uncaught `ValueError` from tuple unpacking. It now raises a `MisconfigurationException` that names the `class_path` and asks for a fully qualified path [#61](https://github.com/speediedan/finetuning-scheduler/issues/61).

- Fixed custom strategy adapter resolution accepting targets that are not `StrategyAdapter` subclasses: the result of its `issubclass` check was discarded, so a wrong target failed later and far from its configuration. Targets resolved by entry-point name or by fully qualified name are now refused at resolution [#62](https://github.com/speediedan/finetuning-scheduler/issues/62).

- Fixed `ScheduleImplMixin.sync` creating attributes named with a dotted path (for example `epoch_progress.current.completed`) on the Lightning fit loop. Dotted paths are now read-only views. The old write never reached the nested counter, so scheduling behavior is unchanged [#63](https://github.com/speediedan/finetuning-scheduler/issues/63).

- Fixed `lock_ci_requirements.sh` ignoring the CUDA target configured in `requirements/ci/torch-pre.txt`. The generated `torch-override.txt` header and the printed installation commands hardcoded the `nightly/cu130` index, so a `test` channel or `cu132` configuration documented the wrong index.

- Fixed `scripts/gpu_lease_wrap.sh` having drifted from the copy it shares with interpretune. `ci_gpu_lease_acquire` and `ci_gpu_lease_release` now take an optional lease name (default `gpu`), so a CI job can also take the host's `cpu-heavy` lease. Existing callers are unchanged [#66](https://github.com/speediedan/finetuning-scheduler/issues/66).

- Fixed `scripts/gen_fts_coverage.sh` sending its commands' stderr to the terminal instead of the coverage session log, and its `--dry-run` listing no standalone tests: it selected with pytest markers this repository does not define, and set an environment variable nothing reads [#67](https://github.com/speediedan/finetuning-scheduler/issues/67).

- Fixed `ci_gpu_lease_release` freeing the GPU lease before removing its holder record, so a local job that acquired the lease at that moment could not replace the record and ran unattributed. The record is now removed first, and an existing record is overwritten without a creating open, which `fs.protected_regular` forbids for another user's file [#73](https://github.com/speediedan/finetuning-scheduler/issues/73).

- Fixed `tests/special_tests.sh` scoring a test by the first verdict printed in its shared log and ignoring the exit code. Under Lightning's `ddp` launcher every rank prints a verdict, so a passing rank could hide a failing one. A test now fails when any rank reports it failed or the run exits non-zero, and each test's output is parsed on its own [#57](https://github.com/speediedan/finetuning-scheduler/issues/57).

- Fixed `tests/special_tests.sh` reporting success when test collection failed, building its default experiment mask from `tests/.experiments` whatever `--experiments_list` named, and failing when `--log-dir` did not exist. It and `scripts/gen_fts_coverage.sh` now reject a bad flag before waiting for the GPU lease [#68](https://github.com/speediedan/finetuning-scheduler/issues/68).

### Changed

- The documentation now leads with the PyTorch API, with Lightning as a fully supported interface: the introduction presents both interfaces, and each step of its walkthrough (Basic Usage, the default schedule, specifying a schedule, transitions and resuming) shows the PyTorch and Lightning code in synced tabs (`sphinx-design`), so one choice switches every example on the page. The Lightning callback's usage details moved to their own page in a new Lightning section, beside the Lightning tutorials, and the PyTorch API tutorial notebook moved into the PyTorch API section. That notebook now logs its runs to TensorBoard under the Lightning tutorial's metric names, and reviews the training curves as that tutorial does. The plain-loop guide's minimal example reads its schedule from YAML [#58](https://github.com/speediedan/finetuning-scheduler/issues/58).

- The Azure multi-GPU pipeline now records its phases: before the leases it writes each phase's manifest (collection only), each test step runs with its phase's recording plugin and, for a measured phase, its own coverage data file (the GPU coverage upload combines the CUDA and standalone phases, as before), and under the leases it writes one report per phase and publishes the manifests and reports as a build artifact. The reports are advisory for now: the test steps remain the gate. The job also demands an agent declaring `GPU_VENDOR` `nvidia`, which keeps its CUDA image off a second, AMD agent in the same pool [#57](https://github.com/speediedan/finetuning-scheduler/issues/57).

- The Azure multi-GPU pipeline now runs on either GPU agent, NVIDIA or AMD, through the any-vendor job template, so pull requests, builds of `main` and release branches, and the nightly run take whichever agent is free. Its phase reports now gate the build (a skip or expected failure must be listed in `tests/gpu_ci/reviewed_exclusions.json`), the GPU coverage upload is named for the vendor that produced it, and a manual run can pin one vendor's agent with the `vendor` parameter [#57](https://github.com/speediedan/finetuning-scheduler/issues/57).

- The SuperGLUE example and its tutorial notebook tokenize through a module-level `convert_to_features` function with explicit arguments, in place of a `LightningDataModule` method. `datasets` now fingerprints the transform from those arguments alone, so its cache is reused across runs [#105](https://github.com/speediedan/finetuning-scheduler/issues/105).

- A `ModelParallelStrategyAdapter.fsdp_plan` that asks for CPU offload (`cpu_offload_policy`, or an `offload_policy` that offloads to the CPU) is now refused with an explanation. Lightning's `ModelParallelStrategy` moves the model to the GPU after `configure_model`, which FSDP2's CPU-offloaded shards cannot follow, so such a plan failed during setup with `_apply(): Couldn't swap ...`. CPU offload works with the PyTorch API (`FTSController`), where it is now tested on NCCL, best-checkpoint restores included [#55](https://github.com/speediedan/finetuning-scheduler/issues/55).

- The FSDP2 auto-plan example (`fts_fsdp_auto_plan.yaml`) shards the transformer itself last, so its embeddings are sharded too, and runs in the example tests again; the model-parallel adapter imports `fully_shard` from the public `torch.distributed.fsdp` [#55](https://github.com/speediedan/finetuning-scheduler/issues/55).

- The Lightning callback now sequences its phase transitions through an externally driven `FTSController` (attached at fit start): the callback still decides when a phase ends, and the controller carries the transition out through the callback's `restore_best_ckpt`, `step_pg` and `thaw_to_depth` and the strategy adapter's hooks, so strategy adapters and callback subclasses see the same calls, in the same order, as before. The callback's public API, schedule format and checkpoints are unchanged [#85](https://github.com/speediedan/finetuning-scheduler/issues/85).

- A reinitialization `class_path` may now be the bare name of a `torch.optim` optimizer or lr scheduler (for example `SGD` or `StepLR`); other classes still need a fully qualified path [#51](https://github.com/speediedan/finetuning-scheduler/issues/51).

- The Lightning callback now refuses the schedule keys only the PyTorch API interprets (`max_transition_step`, `promote`) instead of silently ignoring them [#51](https://github.com/speediedan/finetuning-scheduler/issues/51); it interprets `phase_config` and `pg_overrides` itself [#85](https://github.com/speediedan/finetuning-scheduler/issues/85).

- The Lightning callback's phase-transition, thawing and optimizer logic now lives in the Lightning-free core (`finetuning_scheduler.core.transitions`, `.phase`, `.schedule` and `.optim`), shared with the PyTorch API; behavior is unchanged [#51](https://github.com/speediedan/finetuning-scheduler/issues/51).

- Raised the minimum supported PyTorch version to `2.12.0`, maintaining the "latest 4 PyTorch minor releases" support window documented in `docs/source/versioning.rst`.

- Raised the minimum supported Python version to `3.11`, in step with PyTorch `2.15`, which drops Python `3.10`. Deprecated in `2.14.1`. The CPU CI matrix now tests Python `3.11` and `3.13`.

- The development branch now tracks the PyTorch `2.15` release candidates. The CI Docker images and the Azure multi-GPU pipeline use CUDA `13.2` (toolkit `13.2.1`), matching PyTorch `2.15`'s default CUDA version.

- Raised dependency floors that predate Python `3.11` support, which the `oldest` CI leg exposed once Python `3.11` became the minimum. In the optional extras: `hydra-core` `>=1.3.0` and `omegaconf` `>=2.3.0` (earlier releases define dataclass defaults Python `3.11` rejects), `tensorboardX` `>=2.6.2` (earlier releases call `float()` on a one-element array, an error under NumPy `2.4`) and `psutil` `>=5.9.4` (earlier releases ship no Python `3.11` wheels). In the test dependencies: `mlflow` `>=3.9.0` (earlier releases import a pool class removed in SQLAlchemy `2.1`) and `coverage` `>=7.0.0` (earlier releases ship no Python `3.11` wheel for macOS arm64).

- Raised the `jsonargparse` floor to `4.40.0` and added its `jsonnet` extra in the optional extras. Lightning `2.7` requires `jsonargparse[jsonnet,signatures]>=4.39` for `LightningCLI`, and `4.40.0` is the first release whose `jsonnet` extra uses `jsonnet` releases that ship wheels for every platform.

- Capped `jsonargparse` below `5.0` in the `cli`, `extra`, `examples` and `all` extras. jsonargparse `5.0` removes `ArgumentParser.add_instantiator`, which `LightningCLI` still calls, and Lightning's own `<5.0` cap applies only to Lightning's extras, which FTS does not install. The examples now call `add_class_arguments` in place of `add_dataclass_arguments`, which `5.0` also removes [#60](https://github.com/speediedan/finetuning-scheduler/issues/60).

- Declared `PyYAML>=6.0.2` as a base dependency. FTS imports `yaml` directly but previously received it only through Lightning. The floor is the first release with wheels for every supported Python version [#64](https://github.com/speediedan/finetuning-scheduler/issues/64).

- Declared `fsspec>=2023.5.0` as a base dependency. The scheduling core reads and writes schedules through it directly; it previously arrived only through Lightning [#50](https://github.com/speediedan/finetuning-scheduler/issues/50).

- Declared `numpy>=1.23.2` as a base dependency. A distributed `FTSController` uses PyTorch's object collectives, which need NumPy although PyTorch does not declare it, so without NumPy `setup()` failed with "Numpy is not available"; it previously arrived only through Lightning. The floor is the first release with Python `3.11` wheels [#96](https://github.com/speediedan/finetuning-scheduler/issues/96).

- Importing `finetuning_scheduler` no longer imports Lightning. The Lightning integration (`FinetuningScheduler`, `FTSCheckpoint`, `FTSEarlyStopping`, the scheduling mixins, the `fts`, `fts_supporters` and `strategy_adapters` modules, and `types.BaseCallbackDepType`) loads on first access, and raises an `ImportError` naming the missing package when Lightning is not installed. Lightning remains a base dependency [#50](https://github.com/speediedan/finetuning-scheduler/issues/50).

- Moved the Lightning integration modules to `finetuning_scheduler.lightning.fts` and `finetuning_scheduler.lightning.fts_supporters`. `finetuning_scheduler.fts` and `finetuning_scheduler.fts_supporters` remain importable and keep their API documentation pages and anchors. The classes' `__module__` (and so their `repr`) now names the new modules; logger names are unchanged [#50](https://github.com/speediedan/finetuning-scheduler/issues/50).

- CUDA tests (`RunIf(min_cuda_gpus=...)` and `RunIf(bf16_cuda=True)`) now run only with `PL_RUN_CUDA_TESTS=1`, as the `min_cuda_gpus` documentation already stated, so a plain `pytest` run never uses a host's GPUs. `tests/special_tests.sh`, `scripts/gen_fts_coverage.sh` and the GPU pipeline set it [#68](https://github.com/speediedan/finetuning-scheduler/issues/68).

- The GPU pipeline holds its host leases through `ci_host_lease_hold` in `scripts/gpu_lease_wrap.sh`, which goes through the host's lease tool when an operator has mounted it into the job container (fair queueing, and waiting for `cpu-heavy` before taking the GPUs) and through the plain locks otherwise. The same function serves every repository that shares the wrapper, in place of a lease adapter per pipeline. The gate waits up to an hour (was 40 minutes) to acquire them, since a local GPU suite or another repository's build on the same host can hold them that long [#57](https://github.com/speediedan/finetuning-scheduler/issues/57).

- The GPU pipeline now runs only the CUDA tests that are not standalone, in place of the whole suite, which the hosted CPU matrix already runs on CPU. Single-GPU cases of the phase-transition, optimizer reinitialization and resumption tests join that phase, so those paths still run on a GPU. A pull request into `main` runs a smoke set of six standalone tests (`tests/.standalone_smoke`), one per strategy family. Release branches and tags, builds of `main`, manual runs, and a pull request labeled `full-gpu-suite` run every standalone test, now including those in modules not matching `test_f`, which had left out a two-GPU DDP test. The pipeline also takes the host's `gpu` and `cpu-heavy` (host memory) leases after installing rather than before [#57](https://github.com/speediedan/finetuning-scheduler/issues/57).

- The documentation now uses [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/) in place of a fork of the Lightning Sphinx theme, with a light and dark mode. The docs build moves to Sphinx `8.2`, which the theme requires, and no longer caps `jinja2` below `3.1` [#101](https://github.com/speediedan/finetuning-scheduler/issues/101).

- Tests now cover every line of the package that the CPU and GPU suites together left unexercised, most of them configuration refusals and edge paths of the PyTorch API (`FTSController`, its checkpoint policy and export, schedule compilation and the conformance suite) and of the Lightning bridge and host. Coverage measurement now leaves out bare `...` bodies (protocol members) and `if TYPE_CHECKING:` imports, and the controller's unreachable fallback for reading a model's sharding before `setup()` is removed [#49](https://github.com/speediedan/finetuning-scheduler/issues/49).

### Deprecated

- The CUDA-specific `MemProfiler` configuration keys `cuda_allocator_history`, `collect_funcs.cuda` and `collect_funcs.cuda_allocator_history` are deprecated in favor of `allocator_history`, `collect_funcs.accelerator` and `collect_funcs.allocator_history`, as are `MemProfiler.cuda_allocator_history_snap` and `ProfilerHooksMixin.cuda_allocator_history` (now `allocator_history_snap` and `allocator_history`). The old names keep working, with a `FutureWarning`, and will be unsupported no earlier than 2.17 [#104](https://github.com/speediedan/finetuning-scheduler/issues/104).

- Renaming parameters by overriding `StrategyAdapter`'s parameter-naming methods (`logical_param_translation`, `get_named_params_for_schedule_validation` or `gen_ft_schedule`, usually with `fts_optim_transform` and `validate_ft_sched`) is deprecated in favor of a `TunableView` attached to the module as `fts_tunable_view`, and will be unsupported no earlier than 2.17. An adapter class that does so issues a one-time `FutureWarning`, attributed to the adapter's definition, when it is resolved. Overrides are measured against the nearest built-in adapter, so FSDP's flat-parameter transform is not one. Sharding customization through `StrategyAdapter` subclasses is unaffected, including an adapter that overrides only `fts_optim_transform` (a sharding transform) or only adds checks in `validate_ft_sched` [#54](https://github.com/speediedan/finetuning-scheduler/issues/54).

- Lightning will become an optional dependency, no earlier than 2.17. The first `FinetuningScheduler` constructed in a process issues a one-time `FutureWarning` naming the `finetuning-scheduler[lightning]` extra; the versioning policy now states when moving a dependency behind an extra is a packaging change rather than an API change [#51](https://github.com/speediedan/finetuning-scheduler/issues/51).

## [2.14.1] - 2026-10-09

### Added

- Added `docs-linkcheck.yml`, a weekly scheduled Sphinx `linkcheck`. No workflow ran `linkcheck` previously, which is how 18 broken links accumulated unnoticed. A docs *build* failure fails the job (matching Read the Docs' `fail_on_warning`), while broken external links instead open or update a single long-lived tracking issue and leave the job green, so third-party link rot cannot mask a genuine build regression.

- Vendored the `cross-platform-latex` skill from [speediedan/skills](https://github.com/speediedan/skills) and added `scripts/check_vendored_skills.py`, a dependency-free pre-commit guard asserting the vendored copies still match `.claude/skills/.shared-skills.sha256`. A repo-level `exclude` stops the formatting hooks rewriting those files, since a rewrite would desync the very manifest that asserts they match the master.

- Added an opt-in, `flock`-based GPU lease (`scripts/gpu_lease_wrap.sh`) that serializes GPU test suites on a shared multi-GPU host. `tests/special_tests.sh` and `scripts/gen_fts_coverage.sh` re-exec themselves under the lease, so one acquisition covers a whole suite. It is a complete no-op unless `GPU_LEASE_CMD` points at a lease implementation, so contributors and hosted CI are unaffected [#33](https://github.com/speediedan/finetuning-scheduler/pull/33).

### Fixed

- Fixed `make linkcheck`, which reported 18 broken links. 17 were false positives: PyTorch's documentation site renders API anchors client-side, so `linkcheck` cannot resolve fragments like `#torch.nn.Module` even though the pages resolve. Added `linkcheck_anchors_ignore_for_url` for `pytorch.org` in `docs/source/conf.py`. The remaining failure was a genuine 404: a historical changelog entry linked `jsonargparse` PR #205 as an issue, and that repository has issues disabled.

- Fixed the scheduled `linkcheck` reporting 14 broken links. 13 were anchors on the Lightning documentation, which is now a client-rendered application that returns the same HTML shell for every path, so no fragment on it can be resolved; `lightning.ai/docs` joins `linkcheck_anchors_ignore_for_url`. The remaining one was Stack Overflow rejecting `linkcheck`'s requests with `403 Forbidden` while redirecting normally for a browser, so it is now listed in `linkcheck_ignore`. Resolves [#34](https://github.com/speediedan/finetuning-scheduler/issues/34).

### Changed

- Enabled MyST's `dollarmath` extension in the docs build. `myst_update_mathjax = False` was already set, so the build was configured as though it rendered math while `myst_enable_extensions` was absent entirely, meaning `$x^2$` in a Markdown page rendered as the literal characters. Also set `myst_dmath_allow_space = False` and `myst_dmath_allow_digits = False`, which stop a spaced dollar pair in prose being captured as an equation.

- Raised the minimum supported PyTorch version to `2.11.0`, restoring the strict "latest 4 PyTorch minor releases" support window documented in `docs/source/versioning.rst`. This was announced as advance notice in `2.13.0`.

- Validated support for PyTorch `2.14.1` (the latest tested version) and Lightning `2.6.6`. The CI Docker images and the Azure multi-GPU pipeline now use stable PyTorch `2.14.1` with CUDA `13.0` (toolkit `13.0.3`), and the pinned CI lockfiles were regenerated against stable PyTorch, so `requirements/ci/torch-override.txt` is removed until the next prerelease cycle.

### Deprecated

- Removed the static `ScheduleImplMixin.gen_ft_schedule()` method, deprecated since `2.10.0`. Use the `StrategyAdapter.gen_ft_schedule()` instance method instead, which strategy adapters can override to customize schedule generation and which FTS itself has used since `2.10.0`.

- Deprecated Python `3.10` support. FTS `2.15.0` will require Python `>=3.11`, in step with PyTorch `2.15`, which drops Python `3.10` (itself end-of-life in October 2026). FTS `2.14.x` continues to support Python `3.10`.

## [2.14.0] - not released

No FTS `2.14.0` release was published. FTS `2.14.1` is the first release of the `2.14` line, numbered to match
PyTorch `2.14.1`, the latest PyTorch release it is validated against. It includes everything planned for `2.14.0`,
including the removal of `ScheduleImplMixin.gen_ft_schedule()` (announced for `2.14.0`) and the minimum supported
PyTorch of `2.11.0`.

## [2.13.0] - 2026-07-26

### Added

- Added `scripts/verify_version_consistency.py`, a dependency-free audit of the torch/CUDA/Lightning/Python version declarations that are duplicated across the Dockerfiles, both `dockers/docker_images_*.sh` scripts, the Azure pipeline image tag and the `release-docker.yml` matrix. Exits non-zero on any disagreement and flags the outlier.

- Added CI requirement regeneration drift detection: a report-only workflow (`regen-ci-req-report.yml`) that comments on PRs when the pinned lockfiles are stale, and a scheduled workflow (`regen-ci-req-check.yml`) that can open a PR with refreshed pins. Includes the `regen-ci-reqs` and `check-file-diffs` composite actions and `scripts/regen_summary.py`.

### Fixed

- Fixed `ScheduleImplMixin._add_groups` adding duplicate parameter groups on resume, which raised `ValueError: some parameters appear in more than one parameter group`. Parameters already owned by an existing optimizer parameter group are now excluded, and `0` is returned when no candidates remain so the caller does not extend `base_lrs`/`min_lrs`/`lr_lambdas` for groups that were never created. Resolves [#30](https://github.com/speediedan/finetuning-scheduler/issues/30).

- Fixed `FTSCheckpoint.load_state_dict` unconditionally asserting that an `FTSEarlyStopping` callback is attached, which made checkpoints unloadable when configured with `epoch_transitions_only=True` (a mode in which FTS deliberately does not attach one). The callback is now resolved only within the branch that dereferences it. Resolves [#29](https://github.com/speediedan/finetuning-scheduler/issues/29).

- Relaxed (`LIGHTNING_VERSION`) `<2.6.1` to allow patch-level upgrades, now excluding only the next minor (`<2.7.0`). Validated against Lightning `2.6.5`.

- Fixed five `pytest.mark.parametrize` decorators that declared a single argname with a stray trailing comma (e.g. `"ckpt,"`). `pytest` 9.1 treats the trailing comma as the tuple form and attempts to unpack each parameter value, producing a collection error.

- Fixed the generated CI lockfiles embedding the absolute path of the generating checkout in their uv header, which made them differ on every other checkout even when the resolved pins were identical. The lockfile trailing newline is now normalized for the same reason, so regeneration is diff-stable.

- Raised the `pytest-rerunfailures` floor to `11.0`. `10.x` imports `pkg_resources` at module scope, which `setuptools` `81+` no longer provides, breaking collection on the `oldest` CI leg. Test-only dependency, so the supported install surface for users is unaffected.

### Changed

- Modernized type hints for Python `3.10+` by bumping the `pyupgrade` pre-commit hook to `--py310-plus`.

- Removed dead `.actions/` references from the pytest `addopts` and the CI workflow path filters.

- Enabled Codecov carryforward for the `gpu`, `cpu` and `pytest` flags. GPU coverage is uploaded by the Azure pipeline, which runs less often than the GitHub Actions CPU matrix, so any branch HEAD without a GPU upload previously had no `gpu` flag data at all — rendering the README coverage badge as "unknown" and reporting the branch total as CPU-only coverage rather than the full 100%.

- Regenerated the pinned CI lockfiles after approximately six months without a refresh. Notable movement: `lightning`/`pytorch-lightning` `2.6.0` → `2.6.5`, `transformers` `5.0.0` → `5.14.1`, `torchmetrics` `1.8.2` → `1.9.0`, `pytest` `9.0.2` → `9.1.1`. `requirements/ci/requirements.txt` now pins `torch` again (it is only excluded on the prerelease code path), and `requirements/ci/torch-override.txt` is removed since it only exists while a prerelease channel is active.

### Deprecated

- The removal target for direct calls to `ScheduleImplMixin.gen_ft_schedule()` has been restated from `2.12.0` to `2.14.0`. No `2.11.0` or `2.12.0` release was published (see below), so the original target was never reachable; `2.14.0` restores a real migration window for downstream consumers.

- **Advance notice:** the minimum supported PyTorch version will be raised to `2.11.0` in the `2.14.0` release, restoring the strict "latest 4 PyTorch minor releases" support window documented in `docs/source/versioning.rst`. `2.13.0` retains a minimum of `2.7.0`.

### Contributors

Thanks to the following users/contributors for their feedback and/or contributions in this release:
@DimitrisMantas

## [2.11.0] and [2.12.0] - not released

No FTS `2.11.0` or `2.12.0` release was published. PyTorch `2.11.0` (2026-03-18) and `2.12.0`
(2026-05-13) shipped without corresponding FTS releases; rather than back-fill releases that no user
could retroactively depend on, FTS resumed its PyTorch-aligned cadence with a single catch-up release.
Because that release retains a minimum supported PyTorch of `2.7.0`, every PyTorch version a `2.11.0` or
`2.12.0` release would have targeted remains supported. See `docs/source/versioning.rst`.

## [2.10.0] - 2026-01-26

### Added

- Add CI testing for Python `3.13` and update CI validation to include Python `3.13`. [#28](https://github.com/speediedan/finetuning-scheduler/pull/28/)
- Entry-point plugin discovery for strategy adapters and a new `StrategyAdapter.before_restore_model()` hook — allows installing/registering adapter plugins via standard Python entry points (short-name aliases supported); includes tests and docs. [#26](https://github.com/speediedan/finetuning-scheduler/pull/26/)
- Add `requirements/utils/prune_torch_deps.py` utility to prune torch-only transitive dependencies when generating CI lockfiles and simplify lockfile pruning. [#23](https://github.com/speediedan/finetuning-scheduler/pull/23/)

### Fixed

- Bumped gh-action-pypi-publish to version that fully supports PEP 639 (this is the change that required the post0 release)
- Fix Pyright issues surfaced during the type-hints migration and address related CI/test issues. [#26](https://github.com/speediedan/finetuning-scheduler/pull/26/)
- Fix edge-case LR lambda handling surfaced with PyTorch 2.10. [c043815](https://github.com/speediedan/finetuning-scheduler/commit/c043815)
- Bump citation metadata and fix Codecov flag syntax. [a26d026](https://github.com/speediedan/finetuning-scheduler/commit/a26d026)

### Changed

- Modernized type hints for Python `3.10+` and completed related Pyright migration steps; automated test warnings cleanup and improved CI/build scripts and docs requirements. [#28](https://github.com/speediedan/finetuning-scheduler/pull/28/)
- Migrate CI and build infrastructure to use `uv` (locked requirements, `requirements/ci/overrides.txt` for Lightning commit pins, `manage_standalone_processes.sh` process wrapper, improved torch nightly workflow). [#22](https://github.com/speediedan/finetuning-scheduler/pull/22/)
- Update import semantics and adapter configuration to accept both `module.Class` and `module:Class` forms and support plural `custom_strategy_adapters` mappings. [#26](https://github.com/speediedan/finetuning-scheduler/pull/26/)
- Simplify and secure CI lockfile generation by pruning torch-only dependencies and documenting security rationale. [#23](https://github.com/speediedan/finetuning-scheduler/pull/23/)
- Update minimum Python for builds where applicable: drop support for Python `3.9` (minimum Python `3.10`) and add validation for `3.13`. [#22](https://github.com/speediedan/finetuning-scheduler/pull/22/) [#28](https://github.com/speediedan/finetuning-scheduler/pull/28/)

### Deprecated

- Direct calls to `ScheduleImplMixin.gen_ft_schedule()` are deprecated in favor of `StrategyAdapter.gen_ft_schedule()` instance overrides to allow strategy-specific schedule generation and customization; direct calls will be removed in a future release (2.12.0). (see `src/finetuning_scheduler/fts_supporters.py`)


## [2.9.0] - 2025-10-20

### Added

- Added YAML (multi)representer for ``PretrainedConfig`` object types
- Introduced a `log_dir` parameter to allow specifying a custom directory for artifacts, defaulting to `trainer.log_dir` or `trainer.default_root_dir` [#17](https://github.com/speediedan/finetuning-scheduler/pull/17/).
- Added trainer convenience reference to FTS for a cleaner interface and enable future improved encapsulation
- Improved testing infrastructure with centralized test warnings and coverage/environment build management scripts
- Added dynamic versioning system with CLI tool `toggle-lightning-mode` for switching between unified (`lightning.pytorch`) and standalone (`pytorch_lightning`) imports. Resolves [#10](https://github.com/speediedan/finetuning-scheduler/issues/10).
- Added support for Lightning CI commit pinning via `USE_CI_COMMIT_PIN` environment variable
- Modernized build system using pyproject.toml with setuptools PEP 639 support

### Fixed

- Addressed upstream issue [#20615](https://github.com/Lightning-AI/pytorch-lightning/issues/20615) where `trainer.log_dir` resolution logic does not accommodate artifact persistence via `trainer.log_dir` for loggers that do not have a `save_dir` set (using `trainer.default_root_dir`) by refactoring and enhancing FTS `log_dir` handling in [#17](https://github.com/speediedan/finetuning-scheduler/pull/17/).

### Changed

- Official versioning policy to align with PyTorch minor releases; see the new Versioning documentation and compatibility matrix
  (docs/versioning.rst) for details on supported PyTorch/Lightning ranges: [Versioning docs](https://finetuning-scheduler.readthedocs.io/en/stable/versioning.html)

- Updated documentation and improved type annotations
- einsum patch no longer required for PyTorch >= 2.6 for FTS to leverage 2D mesh parallelism
- Improved CI configuration with automatic Lightning commit pinning

### Deprecated

- removed support for PyTorch `2.2`, `2.3`, and `2.4`
- removed use of conda builds (aligning with upstream PyTorch)

## [2.5.3] - 2025-08-14

### Added

- Verified support for Lightning ``2.5.2`` and ``2.5.3``

### Fixed

- Updated explicit pytorch version mapping matrix to include recent PyTorch release
- Fixed newly failing test dependent on deprecated Lightning class attribute. Resolved [#19](https://github.com/speediedan/finetuning-scheduler/issues/19).

### Changed

- For the examples extra, updated minimum `datasets` version to `4.0.0` to ensure the new API (especially important removal of `trust_remote_code`) is used.

## [2.5.1] - 2025-03-27

### Added

- Support for Lightning ``2.5.1``
- Added (multi)representer for ``PretrainedConfig`` object types

## [2.5.0] - 2024-12-20

### Added

- Support for Lightning and PyTorch ``2.5.0``
- FTS support for PyTorch's composable distributed (e.g. ``fully_shard``, ``checkpoint``) and Tensor Parallelism (TP) APIs
- Support for Lightning's ``ModelParallelStrategy``
- Experimental 'Auto' FSDP2 Plan Configuration feature, allowing application of the ``fully_shard`` API using module
name/pattern-based configuration instead of manually inspecting modules and applying the API in ``LightningModule.configure_model``
- FSDP2 'Auto' Plan Convenience Aliases, simplifying use of both composable and non-composable activation checkpointing APIs
- Flexible orchestration of advanced profiling combining multiple complementary PyTorch profilers with FTS ``MemProfiler``

### Fixed

- Added logic to more robustly condition depth-aligned checkpoint metadata updates to address edge-cases where `current_score` precisely equaled the `best_model_score` at multiple different depths. Resolved [#15](https://github.com/speediedan/finetuning-scheduler/issues/15).

### Deprecated

- As upstream PyTorch [has deprecated](https://github.com/pytorch/pytorch/issues/138506) official Anaconda channel builds, `finetuning-scheduler` will no longer be releasing conda builds. Installation of FTS via pip (irrespective of the virtual environment used) is the recommended installation approach.
- removed support for PyTorch `2.1`

## [2.4.0] - 2024-08-15

### Added

- Support for Lightning and PyTorch ``2.4.0``
- Support for Python ``3.12``

### Changed

- Changed default value of the ``frozen_bn_track_running_stats`` option to the FTS callback constructor to ``True``.

### Deprecated

- removed support for PyTorch `2.0`
- removed support for Python `3.8`

## [2.3.3] - 2024-07-09

- Support for Lightning <= ``2.3.3`` (includes critical security fixes) and PyTorch <= ``2.3.1``

## [2.3.2] - 2024-07-08

- Support for Lightning <= ``2.3.2`` and PyTorch <= ``2.3.1``

## [2.3.0] - 2024-05-17

### Added

- Support for Lightning and PyTorch ``2.3.0``
- Introduced the ``frozen_bn_track_running_stats`` option to the FTS callback constructor, allowing the user to override the default Lightning behavior that disables ``track_running_stats`` when freezing BatchNorm layers. Resolves[#13](https://github.com/speediedan/finetuning-scheduler/issues/13).

### Deprecated

- removed support for PyTorch `1.13`

## [2.2.4] - 2024-05-04

### Added

- Support for Lightning ``2.2.4`` and PyTorch ``2.2.2``

## [2.2.1] - 2024-03-04

### Added

- Support for Lightning ``2.2.1``

## [2.2.0] - 2024-02-08

### Added

- Support for Lightning and PyTorch ``2.2.0``
- FTS now inspects any base `EarlyStopping` or `ModelCheckpoint` configuration passed in by the user and applies that configuration when instantiating the required FTS callback dependencies (i.e., `FTSEarlyStopping` or `FTSCheckpoint`). Part of the resolution to [#12](https://github.com/speediedan/finetuning-scheduler/issues/12).

### Changed

- updated reference to renamed `FSDPPrecision`
- increased `jsonargparse` minimum supported version to `4.26.1`

### Fixed

- Explicitly `rank_zero_only`-guarded `ScheduleImplMixin.save_schedule` and `ScheduleImplMixin.gen_ft_schedule`. Some codepaths were incorrectly invoking them from non-`rank_zero_only` guarded contexts. Resolved [#11](https://github.com/speediedan/finetuning-scheduler/issues/11).
- Added a note in the documentation indicating more clearly the behavior of FTS when no monitor metric configuration is provided. Part of the resolution to [#12](https://github.com/speediedan/finetuning-scheduler/issues/12).

### Deprecated

- removed support for PyTorch `1.12`
- removed legacy FTS examples


## [2.1.4] - 2024-02-02

### Added

- Support for Lightning ``2.1.4``

### Changed

- bumped `sphinx` requirement to `>5.0,<6.0`

### Deprecated

- removed deprecated lr `verbose` init param usage
- removed deprecated `tensorboard.dev` references


## [2.1.3] - 2023-12-21

### Added

- Support for Lightning ``2.1.3``


## [2.1.2] - 2023-12-20

### Added

- Support for Lightning ``2.1.2``

### Fixed

- Explicitly `rank_zero_only`-guarded `ScheduleImplMixin.save_schedule` and `ScheduleImplMixin.gen_ft_schedule`. Some codepaths were incorrectly invoking them from non-`rank_zero_only` guarded contexts. Resolves [#11](https://github.com/speediedan/finetuning-scheduler/issues/11).


## [2.1.1] - 2023-11-08

### Added

- Support for Lightning ``2.1.1``


## [2.1.0] - 2023-10-12

### Added

- Support for Lightning and PyTorch ``2.1.0``
- Support for Python ``3.11``
- Support for simplified scheduled FSDP training with PyTorch >= ``2.1.0`` and ``use_orig_params`` set to ``True``
- Unified different FSDP `use_orig_params` mode code-paths to support saving/restoring full, consolidated OSD (PyTorch versions >= ``2.0.0``)
- added support for FSDP `activation_checkpointing_policy` and updated FSDP profiling examples accordingly
- added support for `CustomPolicy` and new implementation of `ModuleWrapPolicy` with FSDP `2.1.0`

### Changed

- FSDP profiling examples now use a patched version of ``FSDPStrategy`` to avoid https://github.com/omni-us/jsonargparse/issues/337 with ``jsonargparse`` < ``4.23.1``

### Fixed

- updated ``validate_min_wrap_condition`` to avoid overly restrictive validation in some ``use_orig_params`` contexts
- for PyTorch versions < 2.0, when using the FSDP strategy, disabled optimizer state saving/restoration per https://github.com/Lightning-AI/lightning/pull/18296
- improved fsdp strategy adapter `no_decay` attribute handling

### Deprecated

- ``FSDPStrategyAdapter`` now uses the ``configure_model`` hook rather than the deprecated ``configure_sharded_model`` hook to apply the relevant model wrapping. See https://github.com/Lightning-AI/lightning/pull/18004 for more context regarding ``configure_sharded_model`` deprecation.
- Dropped support for PyTorch ``1.11.x``.

## [2.0.9] - 2023-10-02

- Support for Lightning 2.0.8 and 2.0.9

## [2.0.7] - 2023-08-16

- Support for Lightning 2.0.7

## [2.0.6] - 2023-08-15

- Support for Lightning 2.0.5 and 2.0.6

## [2.0.4] - 2023-06-22

- Support for PyTorch Lightning 2.0.3 and 2.0.4
- adjusted default example log name
- disabled fsdp 1.x mixed precision tests temporarily until https://github.com/Lightning-AI/lightning/pull/17807 is merged


## [2.0.2] - 2023-04-06

### Added

- Beta support for [optimizer reinitialization](https://finetuning-scheduler.readthedocs.io/en/stable/advanced/optimizer_reinitialization.html). Resolves [#6](https://github.com/speediedan/finetuning-scheduler/issues/6)
- Use structural typing for Fine-Tuning Scheduler supported optimizers with ``ParamGroupAddable``
- Support for ``jsonargparse`` version ``4.20.1``

### Changed

- During schedule phase transitions, the latest LR state will be restored before proceeding with the next phase configuration and execution (mostly relevant to lr scheduler and optimizer reinitialization but also improves configuration when restoring best checkpoints across multiple depths)

### Fixed

- Allow sharded optimizers ``ZeroRedundancyOptimizer`` to be properly reconfigured if necessary in the context of ``enforce_phase0_params`` set to ``True``.


## [2.0.1] - 2023-04-05

### Added

- Support for PyTorch Lightning 2.0.1
- Lightning support for ``use_orig_params`` via ([#16733](https://github.com/Lightning-AI/lightning/pull/16733))

## [2.0.0] - 2023-03-15

### Added

- Support for PyTorch and PyTorch Lightning 2.0.0!
- New ``enforce_phase0_params`` feature.  FTS ensures the optimizer configured in ``configure_optimizers`` will optimize the parameters (and only those parameters) scheduled to be optimized in phase ``0`` of the current fine-tuning schedule. ([#9](https://github.com/speediedan/finetuning-scheduler/pull/9))
- Support for ``torch.compile``
- Support for numerous new FSDP options including preview support for some FSDP options coming soon to Lightning (e.g. ``use_orig_params``)
- When using FTS with FSDP, support the use of ``_FSDPPolicy`` ``auto_wrap_policy`` wrappers (new in PyTorch 2.0.0)
- Extensive testing for FSDP in many newly supported 2.x contexts (including 1.x FSDP compatibility multi-gpu tests)
- Support for strategies that do not have a canonical `strategy_name` but use `_strategy_flag`

### Changed

- Now that the core Lightning package is `lightning` rather than `pytorch-lightning`, Fine-Tuning Scheduler (FTS) by default depends upon the `lightning` package rather than the standalone `pytorch-lightning`. If you would like to continue to use FTS with the standalone `pytorch-lightning` package instead, you can still do so (see [README](https://github.com/speediedan/finetuning-scheduler/blob/main/README.md)). Resolves ([#8](https://github.com/speediedan/finetuning-scheduler/issues/8)).
- Fine-Tuning Scheduler (FTS) major version numbers will align with the rest of the PyTorch ecosystem (e.g. FTS 2.x supports PyTorch and Lightning >= 2.0)
- Switched to use ``ruff`` instead of ``flake8`` for linting
- Replaced `fsdp_optim_view` with either `fsdp_optim_transform` or `fsdp_optim_inspect` depending on usage context because the transformation is now not always read-only
- Moved Lightning 1.x examples to `legacy` subfolder and created new FTS/Lightning 2.x examples in `stable` subfolder


### Removed

- Removed ``training_epoch_end`` and ``validation_epoch_end`` in accord with Lightning
- Removed `DP` strategy support in accord with Lightning
- Removed support for Python `3.7` and PyTorch `1.10` in accord with Lightning

### Fixed

- Adapted loop synchronization during training resume to upstream Lightning changes

## [0.4.1] - 2023-03-14

### Added

- Support for ``pytorch-lightning`` 1.9.4 (which may be the final Lightning 1.x release as PyTorch 2.0 will be released tomorrow)

## [0.4.0] - 2023-01-25

### Added

- **FSDP Scheduled Fine-Tuning** is now supported! [See the tutorial here.](https://finetuning-scheduler.readthedocs.io/en/stable/distributed/fsdp_scheduled_fine_tuning.html)
- Introduced [``StrategyAdapter``](https://finetuning-scheduler.readthedocs.io/en/stable/api/finetuning_scheduler.strategy_adapters.html#finetuning_scheduler.strategy_adapters.StrategyAdapter)s. If you want to extend Fine-Tuning Scheduler (FTS) to use a custom, currently unsupported strategy or override current FTS behavior in the context of a given training strategy, subclassing ``StrategyAdapter`` is now a way to do so. See [``FSDPStrategyAdapter``](https://finetuning-scheduler.readthedocs.io/en/stable/api/finetuning_scheduler.strategy_adapters.html#finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter) for an example implementation.
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

- pinned `jsonargparse` dependency to <4.18.0 until [#205](https://github.com/omni-us/jsonargparse/pull/205) is fixed

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

- removed a few unused internal conditions relating to lr scheduler reinitialization and parameter group addition

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
