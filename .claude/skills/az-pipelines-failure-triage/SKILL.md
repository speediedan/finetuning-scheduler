---
name: az-pipelines-failure-triage
description: Diagnose a finetuning-scheduler Azure GPU build that started and went wrong - attribute the failure to runner infra, a genuine test failure, or expected-path drift, reproduce a GPU-only failure locally, and narrow FSDP/model-parallel assertions. Use when an FTS build is red. For a build that has not started (gated, unauthorized, or never dispatched), use `az-pipelines-ops` instead.
license: Apache-2.0
---

# Triaging FTS Azure GPU Pipeline Failures

This skill covers **what goes wrong in this repo once a job is running**. The failure taxonomy is
FTS-specific, which is why it lives here rather than in a shared skill.

| You need                                                                                     | Use                                   |
| -------------------------------------------------------------------------------------------- | ------------------------------------- |
| A build that has **not started**: gated, waiting on resource authorization, never dispatched | `az-pipelines-ops` (shared, vendored) |
| GPU serialization, a lease wait, a lease that looks stuck                                    | `gpu-lease` (shared, vendored)        |
| A build that **started and failed**                                                          | this skill                            |

**This half is self-contained and its steps start at 1.** Driving the pipeline, from authorization
through gates to dispatch, is its own procedure in `az-pipelines-ops`, also numbered from 1. Step
numbers are local to each half; the pointers above and below are what connect them.

> ⛔ **Do not diagnose a `notStarted` build here.** Start at `az-pipelines-ops` Step 1, which reads the
> build timeline first. The approvals API cannot see a build blocked on resource authorization, so an
> empty approvals response reads as an all-clear and routes you to inspect the agent. That is a
> documented path from a permissions problem to restarting a healthy runner this project shares with
> `interpretune`.

## When to Use This Skill

- A step dies with exit `137` or a shutdown signal (OOM / agent restart)
- A step fails on the agent that passes under plain local `pytest`
- An FSDP or model-parallel expected-path assertion fails and you need to tell a real behavioral change
  from benign upstream drift
- A GPU-only failure needs to be reproduced locally

## Constraints and Ground Truth

- Pipeline definition: `.azure-pipelines/gpu-tests.yml`, org `https://dev.azure.com/speediedan`,
  project `finetuning-scheduler`, definition id **1**, name **"Multi-GPU & Example Tests"**. These are
  the values to substitute for `${ORG}` and `${PROJECT}` when following `az-pipelines-ops`.
- **The pool is shared with interpretune.** Org pool id 1 (`Default`, self-hosted) serves both projects, so
  a queued interpretune build will block FTS. Always check the pool, not just the FTS project. Note that
  the pool id is not the queue id; see `az-pipelines-ops` Step 1 for why that distinction bites.
- PR-triggered runs are approval-gated and will sit pending until released. `drafts: false`, so draft PRs
  do not trigger at all.
- Auth is `AZURE_DEVOPS_EXT_PAT` in the environment.
- Approvals can be driven with the shared
  `project_admin/shared_admin_scripts/az_pipeline_agent_scripts/manage-approvals.sh` in the private admin
  repo (`-o speediedan -p finetuning-scheduler`), or with the REST calls in `az-pipelines-ops` Step 2.
- Runner: a self-hosted agent on a GPU host, running rootless Docker on cgroups v2. The systemd unit sets
  `OOMScoreAdjust=-900`; it does **not** set `MemoryMax`/`MemoryHigh` (an earlier version of this file
  claimed it did — verified absent 2026-07-29, no drop-in exists).

> **Host-specific values live in `CLAUDE.local.md`, not here.** This skill is deliberately
> host-independent. Agent hostname, RAM/swap, GPU models, the agent install directory and the agent's uid
> all vary by machine, so the commands below use `$AGENT_HOME` and illustrative example values. Substitute
> from `CLAUDE.local.md` for the machine you are on:
>
> ```bash
> AGENT_HOME=${AGENT_HOME:-/opt/az_pipeline_agent}   # example default
> AGENT_UID=${AGENT_UID:-998}                        # example; the uid the agent runs as
> ```

- The pipeline hard-asserts `>= 2` CUDA GPUs (`gpu-tests.yml`, the "Env details" step). A single-GPU agent
  fails there, not in the tests.
- Container image `speediedan/finetuning-scheduler:py3.13-pt2.14.0-pl2.6-azpl-init`, in-container venv
  `/tmp/venvs/fts_dev`, `--gpus all --shm-size=512m`.
- Only `CODECOV_TOKEN` is mapped. There are no HuggingFace or gated-model secrets in this pipeline.

### The step split

FTS does **not** split CPU-only from CUDA-marked the way interpretune does — GPUs are live in every test
step. The steps, in order:

| Step                              | What it runs                                                                                                                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Install dependencies`            | `uv pip install -e . -r requirements/ci/requirements.txt --excludes /tmp/venvs/fts_dev/torch-excludes.txt`                                                                      |
| `Env details`                     | `collect_env_details.py` + the `>= 2` GPU assertion                                                                                                                             |
| `Testing: standard`               | `coverage run --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v`                                                                                    |
| `Testing: standalone multi-gpu`   | `bash ./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_f'`                                                                                                |
| `Testing: Experimental Multi-GPU` | **currently commented out** (`--mark_type=exp_patch --experiment_patch_mask="1 0 0 1"`)                                                                                         |
| `Statistics`                      | codecov CLI upload, flags `gpu,pytest`                                                                                                                                          |
| `Testing: Examples`               | `python -m pytest src/fts_examples -v --maxfail=1 --durations=0`                                                                                                                |
| `Testing: Multi-GPU Examples`     | `special_tests.sh --mark_type=standalone --collect_dir='src/fts_examples' --filter_pattern='model_parallel_examples'`, with `PYTORCH_KERNEL_CACHE_PATH=/__w/_temp/kernel_cache` |
| `Cleaning up agent workspace`     | `sudo chmod -R 775 /__w/1/s`, `condition: always()`                                                                                                                             |

`timeoutInMinutes: 100`. Marks are `standalone` and `exp_patch` only — there is no `profile_ci` mark, and
no `--reruns` flags, so a flaky test fails the build on first occurrence.

## GPU lease

This pipeline participates in the host GPU lease: it bind-mounts the lease directory into the job
container, acquires in the first step for the life of the job, releases in `Cleaning up agent workspace`,
and fails open when the directory is absent. The mechanics are commented inline in
`.azure-pipelines/gpu-tests.yml`, and AGENTS.md carries the repo-level summary and the CI lease tag shape.

**For everything else about the lease, including the hard rule that a CI-held lease is never
force-reset, use the `gpu-lease` skill.**

## Step 1: Triage the Failure Class

**Queue / approval** — the build never left `notStarted`. This is not a triage case; go to
`az-pipelines-ops` Step 1 and read the timeline. Do not conclude anything from the approvals endpoint
alone.

**Infra / runner** — exit `137`, "shutdown signal", agent log shows a restart, or Docker socket errors.
Memory pressure is the usual cause, especially on a host with little swap relative to RAM (the current
host is an example: roughly 62 GiB RAM against 2 GiB swap). Multi-GPU standalone tests each spawn a fresh
process, so peak usage scales with parallelism. Read the agent's own logs before concluding:

```bash
tail -f "$AGENT_HOME"/_diag/Agent_*.log
ls -1t "$AGENT_HOME"/_diag/Worker_*.log | head
```

Recovery, **only once `az-pipelines-ops` Step 1 has ruled out approval and authorization**:

```bash
sudo "$AGENT_HOME"/restart-stack.sh
```

Check for orphaned processes from an interrupted standalone run before re-queuing —
`scripts/manage_standalone_processes.sh` is the tool for reaping them.

**Test failure** — reproduce locally (Step 2). Note which step failed: a `Testing: standard` failure is
usually reproducible on any machine; a `standalone multi-gpu` or `Multi-GPU Examples` failure usually is not.

## Step 2: Reproduce Locally

Reproduce the pipeline's exact environment by running the same image. Full walkthrough in the private
`fts_azure_pipeline_local.md`; the essentials:

```bash
docker network create --label test_net local_test_net
CONTAINER_NAME=$(/usr/bin/docker create -t --name test_ci_container --gpus all \
  --label test_net --network local_test_net --shm-size=512m \
  -v "/var/run/user/$(id -u)/docker.sock":"/var/run/docker.sock" \
  -v "/usr/bin/docker":"/tmp/docker:ro" \
  speediedan/finetuning-scheduler:py3.13-pt2.14.0-pl2.6-azpl-init)
docker start $CONTAINER_NAME && docker exec -i -t $CONTAINER_NAME bash
```

Inside the container, replay the pipeline's install verbatim — the `--excludes` is what preserves the
image's prebuilt CUDA torch:

```bash
source /tmp/venvs/fts_dev/bin/activate
export UV_OVERRIDE="${PWD}/requirements/ci/overrides.txt"
uv pip install -e . -r requirements/ci/requirements.txt --excludes /tmp/venvs/fts_dev/torch-excludes.txt
```

Note the socket path differs from the pipeline's (`/var/run/user/$AGENT_UID/docker.sock`, the agent's own
uid — e.g. `998`). Using your own uid locally is intentional — do not "fix" it to match the pipeline.

Always tear down:

```bash
docker container stop $CONTAINER_NAME && docker container rm $CONTAINER_NAME
docker network prune --force --filter "label=test_net"
```

Faster iteration without the container, when the failure is not image-specific:

```bash
source ${FTS_VENV_BASE}/${FTS_TARGET_VENV}/bin/activate
python -m pytest src/finetuning_scheduler tests -v
PL_RUN_STANDALONE_TESTS=1 python -m pytest tests/test_model_parallel.py::test_name -v
bash ./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_f'
```

`special_tests.sh` runs `set -e`; to see every failure in one pass rather than stopping at the first:

```bash
sed -i 's/set -e/set +e/g' ./tests/special_tests.sh   # revert before committing
```

## Step 3: Narrowing Multi-GPU Failures

FTS's heavy GPU surface is FSDP and model-parallel, and both assert against recorded expected states in
`tests/fsdp_expected_paths.py` and `tests/model_parallel_expected_paths.py`. When one of those fails:

- Distinguish a *behavioral* change (wrong wrapping / scheduling) from an *expected-path* drift (upstream
  changed something benign and the recorded expectation needs updating). Only the latter justifies editing
  the expected-paths module.
- Run the single parametrization rather than the file — these tests are heavily parametrized and each case
  spawns its own process.
- Check whether the failure is oldest-vs-latest Lightning specific before assuming an FTS bug; the git
  commit pin in `requirements/ci/overrides.txt` moves independently of the released Lightning version.
- Some CPU-offload FSDP2 configs are deliberately disabled on newer PyTorch pending upstream fixes, and a
  2D model-parallel test was disabled for ~10% flakiness. Check git history before chasing a "new" flake.

## What Not to Do

- Don't re-queue a build to "see if it passes" before checking the agent log — if the agent is wedged,
  every re-queue burns a 100-minute timeout slot on a pool interpretune also needs.
- Don't restart the agent on the strength of an empty approvals response. See the warning at the top.
- Don't edit expected-path modules to make a test green without establishing which side actually changed.
- Don't commit the `set +e` patch to `special_tests.sh`.

## Expected Outcome

- The failure is attributed to one of: runner infra, or a genuine test failure. A build that never
  started is not attributed here at all; it belongs to `az-pipelines-ops`.
- Genuine test failures are reproduced locally, in-container if image-specific.
- Infra failures end with a stack restart and a re-queued build, not a code change.
