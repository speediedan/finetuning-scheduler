---
name: fts-release
description: Prepare, publish, verify and close out a Fine-Tuning Scheduler release from its long-lived release/X.Y.x branch - release commit, release-hygiene build, the maintainer authorization gate, GitHub Release creation, PyPI and Read the Docs verification, the release-tag CI run, and the post-release Zenodo DOI / CITATION.cff update on both branches. Use when cutting a release branch, preparing or publishing an FTS release, or doing post-release chores.
license: Apache-2.0
metadata:
  author: speediedan
  version: '1.0'
---

# FTS release: prepare, publish, verify, close out

Fixes land on `main` first and are cherry-picked forward. Releases are always cut from the long-lived
`release/X.Y.x` branch (the `~/repos/fts-release` worktree), never from `main`. Opening the next development cycle
on `main` is a separate job: use the `fts-development-branch-version-upgrade` skill for that.

## The authorization gate (read first)

**Creating the tag and GitHub Release requires explicit, per-release authorization from the maintainer.** Publishing
is irreversible: a published tag must never be deleted or force-pushed (downstream packagers pin tags), so a mistake
can only be corrected with a `.postN` (metadata or docs) or a new patch release (code). Everything up to Phase 3 may
proceed autonomously. Then stop, report readiness with the evidence table from Phase 2, and wait for the
maintainer's go-ahead naming the version. Approval for one release does not carry over to the next.

## Phase 1: Prepare the release branch

1. **Lightning pin.** Before cutting, `main`'s `requirements/ci/overrides.txt` pin should be the latest Lightning
   *release tag* commit, so the release branch inherits a released Lightning and needs no release-only pin.
1. **Cut** (first release of a minor):
   ```bash
   cd ~/repos/fts-release && git fetch origin && git checkout -b release/X.Y.x origin/main
   ```
1. **Release commit: one commit, everything in it.** The `__about__.py` change pulls a gated GPU build on push, so
   every further push to the release branch costs another long CI slot. Bundle:
   - `src/finetuning_scheduler/__about__.py`: drop `.dev0`
   - `CITATION.cff` `version:`
   - the CHANGELOG section title and date (`## [X.Y.Z] - YYYY-MM-DD`)
   - README's `export FTS_VERSION=` source-install example (it clones tag `v${FTS_VERSION}`, which must exist)
   - `.github/ISSUE_TEMPLATE/bug_report.md` version examples
   - `docs/source/versioning.rst` compatibility row
1. **Version numbering may match the PyTorch patch.** 2.14.1 skipped 2.14.0 to match PyTorch 2.14.1. When a
   planned version is skipped, add a `## [X.Y.Z] - not released` CHANGELOG stanza saying what superseded it, and
   make the same CHANGELOG edits on `main`.
1. **Doc links:** run the main-to-stable link sync (`readme_rtd_branch_release_sync.sh` in the maintainer's admin
   scripts) and grep for stray `readthedocs.io/en/latest` links in the files it covers.

## Phase 2: Validate (all before asking for authorization)

Run on the exact tree to be released:

```bash
source ${FTS_VENV_BASE}/fts_release/bin/activate     # built from ~/repos/fts-release
pre-commit run --all-files
pyright -p pyproject.toml
python scripts/verify_version_consistency.py
CUDA_VISIBLE_DEVICES="" python -m pytest src/finetuning_scheduler tests -q
(cd docs && make clean && make html SPHINXOPTS="-W --keep-going")
```

**Release-hygiene build.** The CI commit-pin variables must be unset, or a `@ git+` Lightning URL lands in
`Requires-Dist` and PyPI rejects the upload. Do not upload to TestPyPI by hand: the release workflow does it, and a
manual upload would consume the version slot.

```bash
rm -rf dist build && env -u USE_CI_COMMIT_PIN -u PACKAGE_NAME -u UV_OVERRIDE bash -c \
  'uv venv -q --python 3.13 /tmp/relbuild && . /tmp/relbuild/bin/activate && uv pip install -q build twine && python -m build && twine check dist/*'
unzip -p dist/*.whl '*/METADATA' | grep -E '^(Metadata-Version|Version|Requires-Python|Requires-Dist: (torch|lightning))'
unzip -p dist/*.whl '*/METADATA' | grep -c 'git+'      # must be 0
```

Then install the wheel into another clean venv (`uv pip install --torch-backend=cpu dist/*.whl`) and import the
public API. Run `python -c` checks from a private directory, not `/tmp`, where a stray module can shadow the
standard library.

**CI on the release branch head:** GitHub `Test full` green on every leg, and the Azure GPU pipeline green on every
task (`Testing: CUDA`, named `Testing: standard` on older branches, then `standalone multi-gpu`, `Examples` and
`Multi-GPU Examples`). A push to `release/*` does auto-trigger Azure; find its build by id or
`statusFilter=notStarted` before queueing a manual run, or you create a duplicate. Release the gate per the
`az-pipelines-ops` skill.

**GPU release validation on both vendors:** queue `.azure-pipelines/gpu-release.yml` by hand on the release branch
head, after an operator has switched the AMD agent to its release mode (a vendor-demanded job queued while no agent
matches fails at once). Its `release_eligibility` job must succeed and publish `fts-release-eligibility` with
`"eligible": true` for that head's revision; a green leg alone is not enough. Switch the AMD agent back once both legs
have been assigned.

**Coverage badges** for the branch must render percentages (URL-encode the slash):

```bash
for f in gpu cpu pytest; do curl -sS "https://codecov.io/gh/speediedan/finetuning-scheduler/branch/release%2FX.Y.x/graph/badge.svg?flag=$f" | grep -oE ">[a-z0-9%]+</text>" | tail -1; done
```

Report readiness with: branch head SHA, CI results, coverage, build metadata, and the proposed release notes.
**Stop here until authorized.**

## Phase 3: Publish (authorized only)

1. **Release notes** are the CHANGELOG section for the version, plus any `not released` stanza it supersedes:
   ```bash
   python - > /tmp/notes.md <<'EOF'
   import pathlib; s = pathlib.Path('CHANGELOG.md').read_text()
   print(s[s.index("## [X.Y.Z]"):s.index("## [<previous released version>]")].rstrip())
   EOF
   ```
   Thank external contributors if there were any (`### Contributors` in the CHANGELOG).
1. **Re-confirm the head** is the validated SHA and the tag does not exist:
   ```bash
   git fetch origin --tags && git rev-parse origin/release/X.Y.x && git ls-remote --tags origin 'vX.Y*'
   ```
1. **Create the tag and release in one step, pinned to the full validated SHA** (not the branch name, which can
   move):
   ```bash
   gh release create vX.Y.Z --target <full sha> --title "Fine-Tuning Scheduler Release X.Y.Z" --notes-file /tmp/notes.md
   ```
   Title forms in use: `Fine-Tuning Scheduler Release X.Y.Z`, `Fine-Tuning Scheduler Patch Release X.Y.Z`.

## Phase 4: Verify publication

1. **`release-pypi.yml`** fires on `release: published`. All five jobs must succeed: `build-package`,
   `upload-package`, `check-pypi-environment`, `publish-package-testpypi`, `publish-package-pypi` (trusted
   publishing; no local credential). `check-pypi-environment` stops the release before anything is published unless
   the `pypi` environment has a required reviewer and a deployment policy admitting only release tags (the one-time
   setup below). The TestPyPI job uploads to TestPyPI (environment `testpypi`) and the PyPI job needs it, so a missing
   TestPyPI trusted publisher stops the release before PyPI: register it, then re-run the failed jobs.
   ```bash
   gh run list --workflow release-pypi.yml --limit 1 && gh run watch <id> --exit-status
   ```
1. **Verify the TestPyPI upload, then the maintainer approves PyPI.** The PyPI job waits for an approval of its
   `pypi` deployment (GitHub cancels a run still waiting after 30 days). Install the TestPyPI build in a clean
   environment and import it before asking:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
     finetuning-scheduler==X.Y.Z
   python -c "import finetuning_scheduler as fts; print(fts.__version__)"
   ```
   Report the result and ask the maintainer to approve the deployment (the run's "Review deployments" button). The
   approval is the maintainer's own decision for each release, like the tag: never approve it on their behalf, even
   with their credentials. Their command-line equivalent, after reading the run's pending deployment:
   ```bash
   gh api repos/speediedan/finetuning-scheduler/actions/runs/<id>/pending_deployments
   gh api -X POST repos/speediedan/finetuning-scheduler/actions/runs/<id>/pending_deployments \
     -F 'environment_ids[]=<pypi environment id>' -f state=approved -f comment='TestPyPI X.Y.Z verified'
   ```
1. **PyPI metadata:** latest version, `requires_python`, `torch`/`lightning` requirements, zero `git+`, wheel and
   sdist both present. The JSON API can update before the simple index pip reads; if `pip install` cannot find the
   version, query `https://pypi.org/simple/finetuning-scheduler/` and retry with `--no-cache` rather than treating it
   as a publish failure.
1. **Clean install from PyPI** and import the public API, confirming the resolved torch/Lightning versions.
1. **Read the Docs:** `stable` and the new `vX.Y.Z` version must build at the tag commit, and `latest` must track
   `main`. Check both the API and the rendered pages (the versioning matrix shows the new row):
   ```bash
   curl -s "https://readthedocs.org/api/v3/projects/finetuning-scheduler/builds/?limit=6"
   curl -sL https://finetuning-scheduler.readthedocs.io/en/stable/versioning.html | grep -c '<p>X.Y.x</p>'
   ```
   If `stable` points at an old version, trigger a stable build, and if that fails, toggle the old version
   hidden and unhidden.
1. **Release-tag CI:** the Azure pipeline includes `refs/tags/*` and usually queues a tag build (it has not always
   fired). Release its gate when the agent is quiet and confirm it green. Its tree equals the validated branch head,
   so a missing tag build is not a validation gap.
1. The Docker release workflow (`release-docker.yml`) is disabled by design; CI images are built and pushed manually.

### One-time setup: the `pypi` environment

The PyPI job deploys to the `pypi` environment, and `check-pypi-environment` refuses a release until that
environment requires a maintainer's approval and admits only release tags. A repository administrator creates it
once (the maintainer, never an agent: it is the release's approval gate):

```bash
gh api -X PUT repos/speediedan/finetuning-scheduler/environments/pypi --input - <<'JSON'
{"reviewers": [{"type": "User", "id": <maintainer's user id>}], "prevent_self_review": false,
 "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
JSON
gh api -X POST repos/speediedan/finetuning-scheduler/environments/pypi/deployment-branch-policies \
  -f name='v*' -f type=tag
```

`prevent_self_review` stays off, since the maintainer approves their own release. Optionally scope the PyPI trusted
publisher for finetuning-scheduler to the `pypi` environment, so that only an approved job can publish.

## Phase 5: Close out

1. **Zenodo DOI.** The GitHub integration mints a version DOI under the concept DOI `10.5281/zenodo.6463952` within
   seconds of publication:
   ```bash
   curl -s "https://zenodo.org/api/records?q=conceptrecid:6463952&sort=mostrecent&size=1&all_versions=true" \
     | python -c "import json,sys; h=json.load(sys.stdin)['hits']['hits'][0]; print(h['metadata']['version'], h['doi'])"
   ```
   Append it to `CITATION.cff` `identifiers:` on **both** `main` and `release/X.Y.x`, and on `main` advance
   `version:` to the released version (`main` otherwise keeps the last *published* version, never a `.dev0`). Direct
   pushes are fine: `CITATION.cff` is outside the gated pipeline's paths. The README badge uses the concept DOI, so
   it needs no change.
1. **Issues:** confirm every issue the release fixes is closed, and comment on any reporter-facing ones.
1. **Main's badges:** if a post-merge `main` build was skipped or rejected during the cycle, the README Azure badge
   reads `failed` and the codecov `gpu` flag `unknown` until the next `main` build. Approve one when the agent is
   quiet.

## What not to do

- Do not tag or publish without the maintainer's authorization for this specific version.
- Never delete or force-push a published tag. Use `.postN` for metadata/docs fixes and a patch release for code.
- Do not create the release from `main` or from a branch name instead of the validated SHA.
- Do not push to the release branch piecemeal: each push touching a gated path costs a full GPU CI run.
