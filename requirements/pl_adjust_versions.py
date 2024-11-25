import os
import re
import sys
from typing import Dict, Optional

# IMPORTANT: this list needs to be sorted in reverse
VERSIONS = [
    dict(torch="2.6.0", torchvision="0.20.1"),  # nightly torchvision nightly not yet bumped as of 20241124
    dict(torch="2.5.1", torchvision="0.20.1"),  # stable
    dict(torch="2.5.0", torchvision="0.20.0"),
    dict(torch="2.4.0", torchvision="0.19.0"),
    dict(torch="2.3.1", torchvision="0.18.1"),
    dict(torch="2.3.0", torchvision="0.18.0"),
    dict(torch="2.2.2", torchvision="0.17.2"),
    dict(torch="2.2.1", torchvision="0.17.1"),
    dict(torch="2.2.0", torchvision="0.17.0"),
]


def find_latest(ver: str) -> Dict[str, str]:
    # drop all except semantic version
    ver = re.search(r"([\.\d]+)", ver).groups()[0]  # type: ignore[union-attr]
    # in case there remaining dot at the end - e.g "1.9.0.dev20210504"
    ver = ver[:-1] if ver[-1] == "." else ver
    print(f"finding ecosystem versions for: {ver}")

    # find first match
    for option in VERSIONS:
        if option["torch"].startswith(ver):
            return option

    raise ValueError(f"Missing {ver} in {VERSIONS}")


def main(req: str, torch_version: Optional[str] = None) -> str:
    if not torch_version:
        import torch

        torch_version = torch.__version__
    assert torch_version, f"invalid torch: {torch_version}"

    # remove comments and strip whitespace
    req = re.sub(rf"\s*#.*{os.linesep}", os.linesep, req).strip()

    latest = find_latest(torch_version)
    for lib, version in latest.items():
        replace = f"{lib}=={version}" if version else ""
        req = re.sub(rf"\b{lib}(?!\w).*", replace, req)

    return req


if __name__ == "__main__":
    if len(sys.argv) == 3:
        requirements_path, torch_version = sys.argv[1:]
    else:
        requirements_path, torch_version = sys.argv[1], None  # type: ignore[assignment]

    with open(requirements_path) as fp:
        requirements = fp.read()
    requirements = main(requirements, torch_version)
    print(requirements)  # on purpose - to debug
    with open(requirements_path, "w") as fp:
        fp.write(requirements)
