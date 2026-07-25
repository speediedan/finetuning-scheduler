#!/usr/bin/env python3
# Copyright The Finetuning-Scheduler authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audit version declarations that are duplicated across the repo and must agree.

FTS declares the same torch/CUDA/Lightning/Python versions in several places (Dockerfiles, the docker
image build scripts, the Azure pipeline image tag, the release-docker workflow matrix). Those are updated
by hand during a version upgrade and have historically drifted apart. This script reports every
declaration side by side and exits non-zero when a group disagrees, so the upgrade skill can gate on it.

It is intentionally dependency-free and read-only so it can run anywhere, including a bare CI step::

    python scripts/verify_version_consistency.py            # audit, exit 1 on mismatch
    python scripts/verify_version_consistency.py --report   # audit, always exit 0
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str | None:
    """Return the text of ``rel`` relative to the repo root, or ``None`` if it does not exist."""
    path = REPO_ROOT / rel
    return path.read_text() if path.is_file() else None


def _search(rel: str, pattern: str) -> str | None:
    """Return the first capture group of ``pattern`` in ``rel``, or ``None`` if unmatched/missing."""
    text = _read(rel)
    if text is None:
        return None
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1) if match else None


# Each group is a set of (label, file, regex) probes whose captured values must all agree.
GROUPS: dict[str, list[tuple[str, str, str]]] = {
    "pytorch": [
        ("base-cuda Dockerfile", "dockers/base-cuda/Dockerfile", r"^ARG PYTORCH_VERSION=([\d.]+)"),
        ("fts-az-base Dockerfile", "dockers/fts-az-base/Dockerfile", r"^ARG PYTORCH_VERSION=([\d.]+)"),
        ("release Dockerfile", "dockers/release/Dockerfile", r"^ARG PYTORCH_VERSION=([\d.]+)"),
        ("docker_images_main.sh", "dockers/docker_images_main.sh", r'\["pytorch"\]="([\d.]+)"'),
        ("docker_images_release.sh", "dockers/docker_images_release.sh", r'\["pytorch"\]="([\d.]+)"'),
        ("azure gpu-tests image", ".azure-pipelines/gpu-tests.yml", r"finetuning-scheduler:py[\d.]+-pt([\d.]+)-"),
        ("release-docker matrix", ".github/workflows/release-docker.yml", r'pytorch_version: \["([\d.]+)"\]'),
    ],
    "lightning": [
        ("fts-az-base Dockerfile", "dockers/fts-az-base/Dockerfile", r"^ARG LIGHTNING_VERSION=([\d.]+)"),
        ("release Dockerfile", "dockers/release/Dockerfile", r"^ARG LIGHTNING_VERSION=([\d.]+)"),
        ("docker_images_main.sh", "dockers/docker_images_main.sh", r'\["lightning"\]="([\d.]+)"'),
        ("docker_images_release.sh", "dockers/docker_images_release.sh", r'\["lightning"\]="([\d.]+)"'),
        ("azure gpu-tests image", ".azure-pipelines/gpu-tests.yml", r"-pl([\d.]+)-azpl-init"),
        ("release-docker matrix", ".github/workflows/release-docker.yml", r'pl_version: \["([\d.]+)"\]'),
    ],
    "python": [
        ("base-cuda Dockerfile", "dockers/base-cuda/Dockerfile", r"^ARG PYTHON_VERSION=([\d.]+)"),
        ("fts-az-base Dockerfile", "dockers/fts-az-base/Dockerfile", r"^ARG PYTHON_VERSION=([\d.]+)"),
        ("release Dockerfile", "dockers/release/Dockerfile", r"^ARG PYTHON_VERSION=([\d.]+)"),
        ("docker_images_main.sh", "dockers/docker_images_main.sh", r'\["python"\]="([\d.]+)"'),
        ("docker_images_release.sh", "dockers/docker_images_release.sh", r'\["python"\]="([\d.]+)"'),
        ("azure gpu-tests image", ".azure-pipelines/gpu-tests.yml", r"finetuning-scheduler:py([\d.]+)-pt"),
        ("release-docker matrix", ".github/workflows/release-docker.yml", r'python_version: \["([\d.]+)"\]'),
    ],
    "cuda": [
        ("base-cuda Dockerfile", "dockers/base-cuda/Dockerfile", r"^ARG CUDA_VERSION=([\d.]+)"),
        ("docker_images_main.sh", "dockers/docker_images_main.sh", r'\["cuda"\]="([\d.]+)"'),
        ("docker_images_release.sh", "dockers/docker_images_release.sh", r'\["cuda"\]="([\d.]+)"'),
        ("release-docker matrix", ".github/workflows/release-docker.yml", r'cust_base: \["cu([\d.]+)-"\]'),
    ],
    "torch-min": [
        (
            "BASE_DEPENDENCIES",
            "src/finetuning_scheduler/dynamic_versioning/utils.py",
            r'"torch>=([\d.]+)"',
        ),
        ("pyproject min-versions", "pyproject.toml", r'^torch = ">=([\d.]+)"'),
    ],
    "lightning-constraint": [
        (
            "LIGHTNING_VERSION",
            "src/finetuning_scheduler/dynamic_versioning/utils.py",
            r'^LIGHTNING_VERSION = "([^"]+)"',
        ),
        ("pyproject min-versions", "pyproject.toml", r'^lightning = "([^"]+)"'),
    ],
}


def audit() -> int:
    """Print every version-declaration group and return the number of groups that disagree."""
    mismatches = 0
    for group, probes in GROUPS.items():
        found = [(label, _search(rel, pat)) for label, rel, pat in probes]
        present = [(label, val) for label, val in found if val is not None]
        distinct = {val for _, val in present}

        if not present:
            print(f"{group}: NO DECLARATIONS FOUND (probes may be stale)")
            mismatches += 1
            continue

        agree = len(distinct) == 1
        status = "OK" if agree else "MISMATCH"
        # when a group disagrees the majority value is almost always the intended one and the outlier is
        # the file that was missed during an upgrade, so flag only the minority values
        majority = max(distinct, key=lambda v: sum(1 for _, other in present if other == v))
        print(f"\n{group}: {status} -> {sorted(distinct)}")
        for label, val in found:
            if val is None:
                marker = "?"
            elif agree or val == majority:
                marker = " "
            else:
                marker = "!"
            print(f"  {marker} {label:<28} {val if val is not None else '<not found>'}")
        if not agree:
            mismatches += 1
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="always exit 0, even when groups disagree")
    args = parser.parse_args()

    mismatches = audit()
    print(f"\n{mismatches} group(s) inconsistent.")
    if mismatches and not args.report:
        print("Re-run with --report to audit without failing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
