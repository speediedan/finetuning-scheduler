---
name: gpu-lease
description: Serialize GPU work on a shared multi-GPU host - plan around queuing, reserve GPUs for notebook/interactive sessions, and recover from stuck or stale leases. Use when GPU work must wait, when `--status` shows an unexpected holder, when GPU tests behave flakily for no code reason, or before starting a long notebook run on a shared host.
---

# GPU lease: sharing a multi-GPU host

## When to use this skill

- A GPU run is sitting at `'gpu' lease is held; waiting...` and you need to decide whether to wait.
- `--status` shows a holder you did not start, and you need to know whether it is safe to clear.
- GPU tests are flaky, OOM, or unusually slow with no code change to explain it: suspect contention.
- You are about to start a **notebook / interactive GPU session** on a shared host.
- A lease looks stuck and you need the recovery path.

## The problem this solves

A multi-GPU host is typically shared by more consumers than it looks: several interactive/agent sessions in
this repo, sessions in sibling projects that share this host and its self-hosted CI agent, and the
self-hosted Azure Pipelines agent, which can dispatch a GPU job the moment one is approved.

Two GPU suites running at once contend **silently**. It surfaces as OOM, flaky timing, or mutual slowdown,
never as "you have a contention problem", so it is routinely misdiagnosed as a real test failure and
"fixed" by rerunning or by editing tests. That misdiagnosis is the actual cost.

## Opting in

The lease is **opt-in and a complete no-op unless `GPU_LEASE_CMD` is set**, so contributors and hosted CI
are unaffected and nothing here is required to work on this project. The implementation is
host/operator infrastructure and is deliberately not vendored into this repo.

```bash
export GPU_LEASE_CMD=/path/to/gpu_lease.sh
```

Some repos carry a test harness that **self re-execs** under the lease, so one acquisition covers a whole
suite. Check this repo's AGENTS.md for whether it has one. Where there is none, and for anything else that
touches the GPU, wrap it explicitly:

```bash
$GPU_LEASE_CMD -- python -m pytest tests/... -v     # block until free, then run
$GPU_LEASE_CMD --cpu-heavy -- <cmd>                 # also take the CPU lease (profiling/benchmarks)
$GPU_LEASE_CMD --timeout 3600 -- <cmd>              # bound the wait; exit 75 (EX_TEMPFAIL) on expiry
$GPU_LEASE_CMD --status                             # who holds each lease, since when, running what
$GPU_LEASE_CMD --doctor                             # full diagnosis (see Recovery)
```

Use `--cpu-heavy` for profiling and benchmark legs: their numbers are only comparable on an otherwise-quiet
machine, so they should exclude CPU-only work too, even when it would not affect correctness.

## Planning around it

**Waiting is normal, not a failure.** Do not "fix" a queued run by disabling the lease or killing the
holder. That reintroduces exactly the contention it prevents.

- **Assume you may wait.** Worst realistic case is a CI job (~37 min) behind which a local suite (~26 min)
  is also queued. Budget for it, or use `--timeout` and do something else.
- **Check `--status` before committing to a GPU run** in a plan or a promise. If something else holds the
  lease, say so rather than starting work that will silently block.
- **Prefer one long acquisition over many short ones.** Releasing between tests lets another consumer
  interleave mid-suite.
- **Do non-GPU work while queued**: docs, plans, lockfiles, lint and type checks all proceed fine.
- Azure runs need no coordination *between themselves*: the agent runs one job at a time. The lease exists
  to stop a pipeline job colliding with **local** work.

## Notebooks and other interactive GPU work

**This is the known gap, and it is handled by convention rather than enforcement.** Notebook and one-off
interactive runs generally will *not* be lease-aware: they are launched by hand, often outlive any single
command, and are the normal way to do exploratory GPU work.

**Wrapping a Jupyter kernel in the lease is the wrong fix.** A kernel lives for hours, so it would hold the
lease for hours and starve CI and every test suite behind it, strictly worse than the contention it would
prevent. So the lease deliberately does not try.

What to do instead, in order of effort:

1. **Check before you start.** `$GPU_LEASE_CMD --status`. If a suite or CI job holds the lease, your
   notebook will contend with it. Wait, or use the other GPU if your work fits on one device.

1. **Reserve deliberately for long sessions.** If a notebook session will occupy the GPUs for a while and
   you do not want a suite or CI job landing on top of it, take the lease explicitly and release it when
   done:

   ```bash
   $GPU_LEASE_CMD --hold --pidfile ~/.gpu_notebook.pid --project notebook-<topic> --timeout 1800
   # ... run the notebook ...
   $GPU_LEASE_CMD --release --pidfile ~/.gpu_notebook.pid
   ```

   This is the same mechanism CI uses to hold a lease across multiple steps. **Release it when you finish.**
   An unreleased hold blocks CI. It is safe if you forget in a crash (the kernel frees the lease when the
   holding process dies) but not if you simply walk away.

1. **Let the detector catch you.** When a lease is acquired while GPUs are already busy, the lease prints an
   advisory warning naming the unleased processes. `--doctor` reports the same thing on demand. So if a
   notebook is quietly using the GPUs, the *next* suite to start will say so instead of just being slow.

**When you see that warning, believe it.** GPU processes running while the lease is free almost always mean
a notebook or an unwrapped run, not a bug in the lease.

## Recovery: stale, stuck, and confusing states

```bash
$GPU_LEASE_CMD --doctor          # stale metadata, dead holders, over-long holds, unleased GPU users,
                                 # and a STALE GPU_LEASE_HELD export
$GPU_LEASE_CMD --reset           # clears metadata for FREE leases only; refuses held ones (exit 3)
$GPU_LEASE_CMD --reset --force   # kills the holder of a genuinely held lease
```

**There is no stale-lock class of failure.** The lock is a `flock`, so the kernel releases it when the
holder dies: a killed suite, a crashed session, or a torn-down CI container all free it automatically.
What *can* go stale is the metadata sidecar (cosmetic) and the `GPU_LEASE_HELD` environment variable.

| Symptom                                               | Meaning                                           | Action                                                          |
| ----------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------- |
| `HELD` with `[container]` and a `project=azure-*` tag | a CI job holds it                                 | **Cancel the pipeline run**, see the warning below              |
| `HELD` by a sibling project                           | legitimately in use                               | Wait or coordinate; do not reset                                |
| `⚠ stale metadata present for a free lease`           | cosmetic leftover                                 | `--reset`                                                       |
| `⚠ held for N min`                                    | possibly a hung job                               | Investigate the pid; `--reset --force` only if genuinely wedged |
| `⚠ GPU_LEASE_HELD ... pid is dead`                    | **stale export**, silently disables serialization | `unset GPU_LEASE_HELD`                                          |
| `⚠ GPU compute apps ... while the gpu lease is FREE`  | notebook or unwrapped run                         | See the notebook section                                        |
| `HELD (just acquired; ...)`                           | normal handoff window                             | Nothing; re-check in a second                                   |

### ⛔ Never force-reset a lease held by CI, and never restart the agent

`--reset --force` **kills the holder**. For a CI job that is both useless and dangerous: the holder pid
lives in the job container's PID namespace, so forcing either fails or kills an unrelated host process that
happens to share that pid number. It is also unnecessary: container teardown already frees the lease.

**Cancel the pipeline run instead.** And never kill or restart the self-hosted agent to recover a lease:
restarting it strands the run without releasing anything the kernel would not have released anyway.

### The one failure mode that was silent

Under rootless docker with userns remapping, the CI container's uid is a host subuid that does not own the
lock file, so opening it **read-write** failed with `EACCES`, and an over-broad fail-open then let the job
proceed *unserialized while reporting success*. It ran GPU tests alongside a local suite and still passed.

Fixed by opening the lock **read-only** (`flock(2)` places a lock regardless of the fd's open mode) and by
narrowing the fail-open so that only a *missing* lease directory means "this host does not use leases".
Recorded here because the lesson generalizes: **a lease that fails open silently is worse than no lease**,
since it manufactures false confidence.
