# originally based on https://bit.ly/3qVl12n
import datetime
import os
import re
from pprint import pprint
from typing import Sequence

REQUIREMENT_FILES = [
    "requirements/base.txt",
    "requirements/extra.txt",
    "requirements/examples.txt",
]

class AssistantCLI:
    _PATH_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    @staticmethod
    def requirements_prune_pkgs(packages: Sequence[str], req_files: Sequence[str] = REQUIREMENT_FILES) -> None:
        """Remove some packages from given requirement files."""
        if isinstance(req_files, str):
            req_files = [req_files]
        for req in req_files:
            AssistantCLI._prune_packages(req, packages)

    @staticmethod
    def _prune_packages(req_file: str, packages: Sequence[str]) -> None:
        """Remove some packages from given requirement files."""
        with open(req_file) as fp:
            lines = fp.readlines()

        if isinstance(packages, str):
            packages = [packages]
        for pkg in packages:
            lines = [ln for ln in lines if not ln.startswith(pkg)]
        pprint(lines)

        with open(req_file, "w") as fp:
            fp.writelines(lines)

    @staticmethod
    def _replace_min(fname: str) -> None:
        req = open(fname, encoding="utf-8").read().replace(">=", "==")
        open(fname, "w", encoding="utf-8").write(req)

    @staticmethod
    def replace_oldest_ver(requirement_fnames: Sequence[str] = REQUIREMENT_FILES) -> None:
        """Replace the min package version by fixed one."""
        for fname in requirement_fnames:
            AssistantCLI._replace_min(fname)

    @staticmethod
    def prepare_nightly_version(proj_root: str = _PATH_ROOT) -> None:
        """Replace semantic version by date."""
        path_info = os.path.join(proj_root, "finetuning_scheduler", "__about__.py")
        # get today date
        now = datetime.datetime.now()
        now_date = now.strftime("%Y%m%d")

        print(f"prepare init '{path_info}' - replace version by {now_date}")
        with open(path_info) as fp:
            init = fp.read()
        init = re.sub(r'__version__ = [\d\.\w\'"]+', f'__version__ = "{now_date}"', init)
        with open(path_info, "w") as fp:
            fp.write(init)


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(AssistantCLI, as_positional=False)
