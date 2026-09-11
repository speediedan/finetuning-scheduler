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

**Most of what goes wrong here is order, not ignorance.** Each step below is a correct action that
produces a wrong answer when taken before the one that scopes it: querying approvals before reading
the timeline, or approving a gate before identifying which build it belongs to. Both were real
incidents. Treat the sequence as the content.

**Step references:** a bare "Step N" means a step in THIS skill. A reference to a step in a
repository's own companion skill is always qualified by that skill's name. Where a skill is split into
a shared half and a repo-local half, both halves number from 1, so an unqualified number is ambiguous
unless this rule is followed.

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

Once the timeline shows `Checkpoint.Approval` pending, the approvals API becomes the right tool. But
**identify each gate before releasing it.**

**A pending-approvals query is not scoped to your work.** Approvals are per PROJECT, and several
pipelines commonly share one project, so the result set includes gates belonging to other people and
other repositories. The obvious loop, `for id in $(...); do approve $id; done`, releases all of them.
That has happened, twice, and neither instance caused harm by design.

It matters beyond tidiness because an approval is often held deliberately. On a shared host the person
holding it may be waiting for a GPU lease to free or for the machine to go quiet, since approving
dispatches a heavy job immediately. That precondition protects the HOST, not just their run, so
releasing their gate bypasses a safety check that was never yours to waive.

The payload carries the build id under `$expand=steps`, so identification is cheap:

```bash
curl -sS -u ":${AZ_PAT}" \
  "${ORG}/${PROJECT}/_apis/pipelines/approvals?state=pending&\$expand=steps&api-version=7.1-preview.1"
# value[].pipeline.owner._links.web.href ends in buildId=<n>

az pipelines build show --id <n> --organization "${ORG}" --project "${PROJECT}" \
  --query "{def:definition.name,branch:sourceBranch}"
```

Then release only the gates you identified as yours:

```bash
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

### When the queued build should not run at all, CANCEL it

The three points above explain why a build you did not want gets queued. They do not say what to do
about it, and leaving it is not neutral. A queued build on a self-hosted pool holds a place in a queue
that other work is waiting in, and on a single-agent pool it will eventually occupy the agent and, if
the job takes one, the host GPU lease. **Declining to approve is not the same as cancelling**: an
unapproved build sits in the queue indefinitely, and its gate is then a trap for anyone running a
batch approval.

So when a pull request or push queues a build whose diff cannot affect the result -- a comment-only
change, a documentation edit, a rename with no behavioural content -- cancel the build rather than
approving it or ignoring it:

```bash
curl -sS -X PATCH -u ":${AZ_PAT}" -H "Content-Type: application/json" \
  -d '{"status":"cancelling"}' \
  "${ORG}/${PROJECT}/_apis/build/builds/<id>?api-version=7.1"
```

Two things worth knowing:

- **A merge queues its own build.** Cancelling the PR's build and then merging leaves a second build,
  on the target branch, for the same content. Check again after merging.
- **Cancel, then dispose of any gate.** If the build had already reached a pending approval, cancelling
  the build may leave the approval pending; reject it with a reason so a later batch approve cannot
  release a run nobody wants.

**This is a judgement about CONTENT, and path filters cannot make it.** Filters match paths, so a
one-word comment fix inside a pipeline definition is indistinguishable to them from a rewrite of the
same file. Adding excludes to compensate is the wrong lever: it would suppress builds for real changes
to those paths too. The filter is doing its job; the human or agent merging is the one who knows the
change is inert.

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

## A build that fails at `Checkout` in seconds, and then every build after it

Distinct from the three `notStarted` causes above, because this build DID dispatch. It fails in the
checkout task, before any step of the pipeline runs, on a path the previous run left behind:

```
System.UnauthorizedAccessException: Access to the path '.../s/.pytest_cache/v/cache/nodeids' is denied
warning: Unable to run "git clean -ffdx" and "git reset --hard HEAD" successfully,
         delete source folder instead
```

**Mechanism.** A containerised job under rootless docker with userns remapping writes into the mapped
workspace as the container's host **subuid**, not as the agent's uid. The agent begins the next build
with `git clean -ffdx` and `git reset --hard` **as itself**, cannot remove those files, and falls back
to deleting the source folder, which fails the same way.

**It is self-perpetuating, and that is the part that surprises people.** Every later build on that
definition fails identically, and none can fix it, because the cleanup would have to run inside a build
that cannot start. A GREEN run is what creates the condition, so the failure appears immediately after
a success and looks unrelated to it.

**After a manual clear, the FIRST build you run must carry the prevention step.** This is the trap,
and the natural instinct is the wrong order: revalidating the default branch first feels like the safe
confirmation, and if that branch does not yet contain the cleanup step, a green run re-arms the
failure and recreates by hand exactly what was just cleared by hand. Approve the branch carrying the
fix first, precisely because it is the one that cleans up after itself. Only once the fix is on the
default branch is revalidating it safe.

Two things are needed and they are not interchangeable:

1. **Clear the existing leftover once, with host privileges.** Only someone who can act as root on the
   agent host can remove files owned by a container subuid. Nothing inside a pipeline can do it, and no
   amount of re-running helps.

1. **Stop it recurring**, with a step that runs INSIDE the container, where that uid still owns what it
   wrote, under `condition: always()` so a failed run cleans up too:

   ```yaml
   - bash: |
       sudo chmod -R 775 <workspace sources dir> || true
       sudo rm -rf <workspace sources dir>/.pytest_cache || true
     displayName: "Normalize workspace ownership"
     condition: always()
   ```

   The sources dir is the agent's per-definition work path and differs by definition, so take it from
   the job rather than hardcoding one repo's. `.pytest_cache` is the usual culprit; the `chmod` covers
   whatever else a run leaves behind.

**Any containerised job on a self-hosted agent wants step 2 from the day it is written.** Adding it
after the first occurrence still needs a privileged human to clear the backlog first.

## What not to do

- **Do not omit the workspace-ownership cleanup from a new containerised job.** It costs four lines up
  front. Without it the first green run arms a failure that blocks every later build on that definition
  and can only be cleared by someone with root on the agent host.
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
