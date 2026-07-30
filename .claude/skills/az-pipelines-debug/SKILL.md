---
name: az-pipelines-debug
description: Debug and operate the finetuning-scheduler self-hosted Azure GPU pipeline — PAT-backed approval release, queue triage, worker dispatch checks, per-step failure diagnosis, and local reproduction of a failing GPU step. Use when an FTS Azure build is stuck, unapproved, or failing in a way that plain pytest does not reproduce.
license: Apache-2.0
---

# Debugging the FTS Azure GPU Pipeline

Adapted from the interpretune skill of the same name. Both projects share one self-hosted agent and one
Azure DevOps organization, so the infrastructure half transfers verbatim; the pipeline shape does not.

## When to Use This Skill

- An FTS build sits in `notStarted` and you need to find out whether it is queue-blocked or approval-gated
- The agent is online but no `Worker_*.log` appears
- A step dies with exit `137` or a shutdown signal (OOM / agent restart)
- You want to approve, reject, or monitor a run from the shell instead of the web UI
- A GPU-only failure needs to be reproduced locally

## Constraints and Ground Truth

- Pipeline definition: `.azure-pipelines/gpu-tests.yml`, org `https://dev.azure.com/speediedan`,
  project `finetuning-scheduler`, definition id **1**, name **"Multi-GPU & Example Tests"**.
- **The pool is shared with interpretune.** Org pool id 1 (`Default`, self-hosted) serves both projects, so
  a queued interpretune build will block FTS. Always check the pool, not just the FTS project.
- PR-triggered runs are approval-gated and will sit pending until released (see Step 2). `drafts: false`,
  so draft PRs do not trigger at all.
- Auth is `AZURE_DEVOPS_EXT_PAT` in the environment.
- Runner: a self-hosted agent on a GPU host, running rootless Docker on cgroups v2. The systemd unit sets
  `OOMScoreAdjust=-900`; it does **not** set `MemoryMax`/`MemoryHigh` (an earlier version of this file
  claimed it did — verified absent 2026-07-29, no drop-in exists).

> **Host-specific values live in `CLAUDE.local.md`, not here.** This skill is deliberately
> host-independent. Agent hostname, RAM/swap, GPU models, the agent install directory and the agent's uid
> all vary by machine, so the commands below use `$AGENT_HOME` and illustrative example values. Substitute
> from `CLAUDE.local.md` (or `.github/copilot-instructions.md`'s successor) for the machine you are on:
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

## GPU lease: how this pipeline interacts with local GPU work

The self-hosted agent runs **one Azure job at a time**, so two pipeline runs never collide with each
other. The real collision risk is a pipeline job landing on top of a **local** multi-GPU run (this host is
shared with interpretune, and both are worked on interactively).

Since 2.14 the job participates in the host GPU lease:

- `container.volumes` bind-mounts `/tmp/di_leases:/gpu_leases`. `flock` operates on the inode, so the
  lock file interlocks between the container and host processes. **Nothing about the agent installation
  changes** — no hooks, no systemd edits, no wrapper around the agent.
- The first step (`Acquire host GPU lease`) parks a detached holder for the life of the job, waiting up to
  **2400s**. A step-scoped `flock` would not work: each step is a separate shell, so the lease would be
  dropped as soon as the acquiring step ended.
- The `Cleaning up agent workspace` step (`condition: always()`) releases it. This is belt-and-braces
  only: if the job is cancelled or the container is torn down, every process inside dies and the kernel
  frees the lease automatically. **There is no stale-lock path.**
- It **fails open**. If `/gpu_leases` is not mounted, the step logs and continues unserialized — a missing
  convenience must never break the build.

Debugging:

```bash
gpu_lease.sh --status     # a CI holder shows project=azure-<buildId> and a [container] tag
gpu_lease.sh --doctor     # flags stale metadata, dead holders, and GPU users holding no lease
gpu_lease.sh --reset      # clears stale metadata for FREE leases only
gpu_lease.sh --reset --force   # kills the holder of a genuinely held lease
```

### ⛔ Never reset a lease held by CI — either project's CI

`gpu_lease.sh --reset --force` **kills the holder process**. That is the right escape hatch for a wedged
*local* run and the wrong tool for a pipeline job, for two reasons:

1. **The holder pid is meaningless on the host.** A CI holder lives in the job container's PID namespace,
   so `--force` either fails to kill it or, worse, kills an unrelated host process that happens to share
   that pid number. `--status` marks these holders with a `[container]` tag and `project=azure-<buildId>`
   (interpretune: `azure-it-<buildId>`) — treat either as read-only.
1. **The lease is already self-healing for CI.** Container teardown kills every process inside the job, and
   the kernel releases the lease. There is no stale-lock path to clean up.

**The host and pool are shared between finetuning-scheduler and interpretune**, so a lease you did not
expect may legitimately belong to the *other* project's pipeline job or local suite. Check `project=`
before assuming it is stale.

Correct responses:

| Situation                                       | Do this                                                           |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| Lease held by a CI job you want to stop         | **Cancel the pipeline run.** Teardown frees the lease.            |
| Lease held by the other project                 | Leave it. Wait, or coordinate — do not reset.                     |
| CI job timed out waiting for the lease          | A genuine conflict. Let the local run finish and re-queue.        |
| Lease looks stale (`--status` flags an anomaly) | `gpu_lease.sh --doctor`, then plain `--reset` (free leases only). |
| Genuinely wedged **local** run                  | `--reset --force` is appropriate here.                            |

**Never kill or restart the agent to free a lease.** `restart-stack.sh` is for a wedged agent, not for lease
recovery, and restarting it mid-job strands the run without releasing anything the kernel would not have
released anyway.

If a job times out waiting, the log names the current holder. That is a genuine local/CI conflict: let the
local run finish and re-queue, rather than disabling the lease.

## Step 1: Verify Auth and Build State

```bash
printenv AZURE_DEVOPS_EXT_PAT | wc -c

az pipelines build show --id <build_id> \
  --organization https://dev.azure.com/speediedan --project finetuning-scheduler -o table

curl -sS -u ":${AZURE_DEVOPS_EXT_PAT}" \
  "https://dev.azure.com/speediedan/finetuning-scheduler/_apis/pipelines/approvals?state=pending&api-version=7.1-preview.1"
```

An empty `{"count":0,"value":[]}` means nothing is awaiting approval — the build is queue-blocked or
already dispatched, so go to Step 3.

## Step 2: Release or Reject a Gated Run

Prefer the shared script (it is already `-o`/`-p` parameterized, so pass the FTS project):

```bash
cd ~/repos/distributed-insight/project_admin/shared_admin_scripts/az_pipeline_agent_scripts
./manage-approvals.sh -o speediedan -p finetuning-scheduler -m list
./manage-approvals.sh -o speediedan -p finetuning-scheduler -m approve -i "<approval_id>" -c "Approved via CLI for self-hosted GPU validation."
./manage-approvals.sh -o speediedan -p finetuning-scheduler -m reject -i "<approval_id>"   # terminates the gated build
./manage-approvals.sh -o speediedan -p finetuning-scheduler -m reject-all                  # dispose all stale pending gates
```

REST fallback:

```bash
curl -sS -X PATCH -u ":${AZURE_DEVOPS_EXT_PAT}" \
  -H "Content-Type: application/json" \
  -d '[{"approvalId":"<approval_id>","status":"approved","comment":"Approved via CLI for self-hosted GPU validation."}]' \
  "https://dev.azure.com/speediedan/finetuning-scheduler/_apis/pipelines/approvals?api-version=7.1-preview.1"
```

## Step 3: Monitor Dispatch and Runner Activity

```bash
watch -n 30 'az pipelines build show --id <build_id> \
  --organization https://dev.azure.com/speediedan --project finetuning-scheduler \
  --query "{status:status,result:result,startTime:startTime,finishTime:finishTime}" -o json'

tail -f "$AGENT_HOME"/_diag/Agent_*.log
ls -1t "$AGENT_HOME"/_diag/Worker_*.log | head

# pool is shared — this lists agents serving BOTH projects
az pipelines agent list --organization https://dev.azure.com/speediedan --pool-id 1 -o table
```

Approved but no `Worker_*.log` within a minute or two means dispatch failed, not the tests. Check whether
an interpretune build holds the agent before touching anything.

## Step 4: Triage the Failure Class

**Queue / approval** — build never leaves `notStarted`, no worker log. Confirm the pool is idle and the
approval was actually released. Re-check Step 1's approvals endpoint.

**Infra / runner** — exit `137`, "shutdown signal", agent log shows a restart, or Docker socket errors.
Memory pressure is the usual cause, especially on a host with little swap relative to RAM (the current
host is an example: roughly 62 GiB RAM against 2 GiB swap). Multi-GPU standalone tests each spawn a fresh
process, so peak usage scales with parallelism. Recovery:

```bash
sudo "$AGENT_HOME"/restart-stack.sh
```

Check for orphaned processes from an interrupted standalone run before re-queuing —
`scripts/manage_standalone_processes.sh` is the tool for reaping them.

**Test failure** — reproduce locally (Step 5). Note which step failed: a `Testing: standard` failure is
usually reproducible on any machine; a `standalone multi-gpu` or `Multi-GPU Examples` failure usually is not.

## Step 5: Reproduce Locally

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

## Step 6: Narrowing Multi-GPU Failures

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
- Don't approve stale pending gates in bulk without listing them first; `reject-all` disposes of real
  pending runs too.
- Don't edit expected-path modules to make a test green without establishing which side actually changed.
- Don't commit the `set +e` patch to `special_tests.sh`.

## Expected Outcome

- The failure is attributed to one of: queue/approval, runner infra, or a genuine test failure.
- Genuine test failures are reproduced locally, in-container if image-specific.
- Infra failures end with a stack restart and a re-queued build, not a code change.
