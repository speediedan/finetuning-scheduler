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
"""Report pins in a CI lockfile that have no wheel for a given Python version on a CI operating system.

``--resolution lowest-direct`` can select dependency floors that predate a newly raised minimum Python. Such a pin
falls back to an sdist build on the ``oldest`` CI leg, which fails wherever Python headers or a compiler are missing.
This checks each ``name==version`` pin against PyPI for a compatible wheel on Linux x86_64, Windows x86_64 and macOS
arm64 (the CI matrix), honoring environment markers. Pure-Python sdist-only packages are reported too; those build
anywhere and can be ignored.

Example::

    python check_wheel_coverage.py requirements/ci/requirements-oldest.txt --python 3.11
"""
import argparse
import concurrent.futures as cf
import json
import re
import urllib.request

from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

PLATFORMS = {
    "linux": ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64", "manylinux_2_5_x86_64",
              "manylinux1_x86_64", "linux_x86_64"],
    "win": ["win_amd64"],
    "mac": ["macosx_14_0_arm64", "macosx_13_0_arm64", "macosx_12_0_arm64", "macosx_11_0_arm64",
            "macosx_10_9_universal2", "macosx_11_0_universal2", "macosx_10_15_universal2"],
}
MARKER_ENVS = {
    "linux": {"sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64"},
    "win": {"sys_platform": "win32", "platform_system": "Windows", "platform_machine": "AMD64"},
    "mac": {"sys_platform": "darwin", "platform_system": "Darwin", "platform_machine": "arm64"},
}
PIN_RE = re.compile(r"^([A-Za-z0-9_.\-\[\]]+)==([^ ;]+)\s*(?:;\s*(.*))?$")
SKIP = {"torch"}  # installed from the PyTorch index, not PyPI


def parse_pins(lockfile: str) -> list[tuple[str, str, str | None]]:
    pins = []
    with open(lockfile) as f:
        for line in f:
            if m := PIN_RE.match(line.strip()):
                pins.append((re.sub(r"\[.*\]", "", m.group(1)), m.group(2), m.group(3)))
    return pins


def check_pin(pin: tuple[str, str, str | None], py: tuple[int, int]) -> str | None:
    name, version, marker = pin
    if name in SKIP:
        return None
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json") as resp:
            files = [u["filename"] for u in json.load(resp)["urls"]]
    except Exception as err:  # report rather than abort: one bad lookup should not hide the rest
        return f"{name}=={version}: lookup failed ({err})"
    wheels = [f for f in files if f.endswith(".whl")]
    missing = []
    for os_name, platforms in PLATFORMS.items():
        env = dict(MARKER_ENVS[os_name], python_version=f"{py[0]}.{py[1]}", python_full_version=f"{py[0]}.{py[1]}.0",
                   implementation_name="cpython")
        if marker and not Marker(marker).evaluate(env):
            continue
        tags = set(cpython_tags(py, platforms=platforms)) | set(compatible_tags(py, platforms=platforms))
        if not any(set(parse_wheel_filename(w)[3]) & tags for w in wheels):
            missing.append(os_name)
    return f"{name}=={version}: no compatible wheel on {', '.join(missing)}" if missing else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lockfile")
    parser.add_argument("--python", default="3.11", help="target Python version, e.g. 3.11")
    args = parser.parse_args()
    py = tuple(int(x) for x in args.python.split("."))[:2]
    with cf.ThreadPoolExecutor(16) as ex:
        problems = [r for r in ex.map(lambda p: check_pin(p, py), parse_pins(args.lockfile)) if r]
    print("\n".join(problems) if problems else f"all pins have wheels for Python {args.python} on every CI OS")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
