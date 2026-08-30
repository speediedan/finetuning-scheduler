---
name: az-pipelines-ops
description: Operate a self-hosted Azure DevOps pipeline from the shell - check whether a queued build is gated, waiting on resource authorization, or genuinely stuck; release or reject an approval; and confirm the agent actually picked the job up. Use when a build sits in notStarted, when an approval needs granting or disposing, or before concluding that a runner is broken.
license: Apache-2.0
---

# Operating a self-hosted Azure Pipelines run

This skill covers **driving the pipeline**: authorization, gates, dispatch. It does not cover
diagnosing test failures once a job is running. Check the repository for a local failure-triage skill
before diagnosing a red build, because the failure taxonomy is repo-specific and lives there.

GPU serialization is a separate concern with its own skill. If this host uses a lease, see `gpu-lease`
rather than anything here.

## When to use this skill

- A build stays `notStarted` after a pull request becomes ready for review.
- The agent is online, but no worker log appears for the queued build.
- You need to approve, reject, or dispose of a gated run from the shell rather than the web UI.
- You are about to conclude that a runner is broken and want to rule out permissions first.

## Step 1: Find out why it has not started, and read the timeline FIRST

**A `notStarted` build has three quite different causes, and the approvals API distinguishes only
one of them.** Read the build's timeline before anything else:

```bash
curl -sS -u ":${AZ_PAT}" \
  "${ORG}/${PROJECT}/_apis/build/builds/<id>/timeline?api-version=7.1"
```

| Timeline shows                        | Meaning                                         | Action                             |
| ------------------------------------- | ----------------------------------------------- | ---------------------------------- |
| `Checkpoint.Approval` pending         | a human gate                                    | approve it, see Step 2             |
| `Checkpoint.Authorization` inProgress | the definition is not authorized for a resource | needs a permit, see below          |
| no checkpoint records at all          | genuine queue or dispatch problem               | only now look at the agent, Step 3 |

**Do not start from the approvals API.** For a new definition's first run it returns nothing while the
build is very much blocked, so "no approvals pending" reads as an all-clear and sends you to inspect
queue backlog, worker dispatch, and agent availability. Those steps end at restarting the agent stack,
which is a documented path from a permissions problem to interfering with a healthy shared runner.

**A definition created fresh hits the authorization case on its first run, every time.** A definition
cloned from an existing one inherits its resource authorizations; a new one does not. Check what it is
missing:

```bash
curl -sS -u ":${AZ_PAT}" "${ORG}/${PROJECT}/_apis/pipelines/pipelinePermissions/queue/<queueId>?api-version=7.1-preview.1"
curl -sS -u ":${AZ_PAT}" "${ORG}/${PROJECT}/_apis/pipelines/pipelinePermissions/endpoint/<connectionId>?api-version=7.1-preview.1"
```

An empty `pipelines` list with no `allPipelines` key means this definition has no grant. Note that
**queue ids are project-scoped references to organization-level pools**, so the queue id is usually not
the pool id and querying the pool id returns an unrelated empty result that looks confirmatory.

Granting a pipeline access to a service connection lets it use those credentials, so it is a
trust-boundary change and normally an owner's decision rather than an automatic fix. The safest route
is the waiting build's own "this pipeline needs permission to access a resource" prompt, which names
exactly what it wants.

## Step 2: Release or reject a gated run

Once the timeline shows `Checkpoint.Approval` pending, the approvals API becomes the right tool:

```bash
curl -sS -u ":${AZ_PAT}" \
  "${ORG}/${PROJECT}/_apis/pipelines/approvals?state=pending&api-version=7.1-preview.1"

curl -sS -X PATCH -u ":${AZ_PAT}" -H "Content-Type: application/json" \
  -d '[{"approvalId":"<id>","status":"approved","comment":"<why>"}]' \
  "${ORG}/${PROJECT}/_apis/pipelines/approvals?api-version=7.1-preview.1"
```

Rejecting a gate completes the build as `failed`. That is the normal terminal state for a rejected
approval and not a sign anything went wrong.

**Before approving, check that the resources the job needs are free.** On a shared host, approving
dispatches a heavy job immediately, so an approval granted while a local suite is running is how two
memory-hungry workloads end up killing each other.

**The approval check is usually attached to the agent POOL rather than to a definition**, which means a
new pipeline targeting that pool inherits the gate automatically and cannot ship ungated. Verify with:

```bash
curl -sS -u ":${AZ_PAT}" \
  "${ORG}/${PROJECT}/_apis/pipelines/checks/configurations?resourceType=queue&resourceId=<queueId>&api-version=7.1-preview.1"
```

## Path filters on a `pr:` trigger evaluate the PR's CUMULATIVE diff

A `pr:`-triggered pipeline with `paths:` filters does not evaluate the diff of the push you just made.
It evaluates the pull request's diff as a whole. Three consequences that all surprise people:

- **A docs-only push re-queues a gated build** as soon as the PR overall touches a filtered path. The
  push looks exempt in isolation and is not, because the filter is not being applied to it.
- **`[skip ci]` in the commit message does not rescue it.** That convention is honoured by some hosted
  CI systems and is not what path filters consult.
- **Approving and then pushing wastes the gate.** The approval applies to the queued run; the new push
  supersedes it and queues another run needing another approval. Push first, then approve.

The practical rule on a gated pipeline is to batch pushes and approve last, rather than approving as
soon as a gate appears.

## Step 3: Confirm the agent actually picked it up

Only reach this step when the timeline showed no checkpoint records.

```bash
curl -sS -u ":${AZ_PAT}" "${ORG}/${PROJECT}/_apis/build/builds/<id>?api-version=7.1"
```

- **`az pipelines build list` defaults to COMPLETED builds**, so the scan you would reflexively run to
  find a queued build cannot see it. Pass `--status notStarted`, or query the build directly.
- **Check `sourceBranch` and `sourceVersion`, not just the build number.** Mistaking an earlier
  successful build for the one you are waiting on is easy and makes a commit look validated when it
  was never built.
- **The build-level queue can display a hosted pool even when the YAML job targets a self-hosted one.**
  The YAML `pool:` wins at run time, so treat approval state and actual dispatch as ground truth rather
  than the definition's default queue.
- **Several test phases may be Tasks inside ONE Job.** A job-level view then shows a single record, so
  "the build succeeded" is not evidence that each phase did. Verify at task level when a specific phase
  is the one you care about.

## What not to do

- **Do not restart the agent to clear a stuck build** until the timeline has ruled out approval and
  authorization. Restarting strands the run without fixing a permission. Once the timeline shows no
  checkpoint records, a genuinely wedged agent IS the remaining cause and restarting it is correct;
  the rule is about the order, not a prohibition.
- **Do not treat a long wait as a hang.** A queued job behind a lease or another run is the system
  working, and cancelling loses the queue position.
- **Do not grant "access to all pipelines" on a resource** to unblock one definition. It removes the
  per-pipeline gate permanently, including for definitions created later by mistake.

## Registering a new definition

`az pipelines create` builds the payload itself:

```bash
az pipelines create --name <name> --repository <owner>/<repo> --repository-type github \
  --service-connection <connectionId> --yml-path <path> --skip-first-run
```

Prefer it over cloning an existing definition through the REST API when the target is a **different**
repository. Cloning copies the template's repository block wholesale, and tooling built around that
assumption cannot retarget it: the new definition silently points at the template's repository, takes
the name you asked for, finds a same-named YAML file there, and reports green while testing something
else. Use `--skip-first-run`, then verify the definition's repository **referentially** rather than by
its name.
