# originally based on https://bit.ly/3qVl12n
import datetime
import os
import re
import shutil
from collections.abc import ValuesView
from os.path import dirname, isfile
from pprint import pprint
from typing import List, Optional, Sequence, Tuple

REQUIREMENT_FILES = [
    "requirements/base.txt",
    "requirements/extra.txt",
    "requirements/examples.txt",
]


def _retrieve_files(directory: str, *ext: str) -> List[str]:
    all_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if not ext or any(os.path.split(fname)[1].lower().endswith(e) for e in ext):
                all_files.append(os.path.join(root, fname))

    return all_files


def _replace_standalone_imports(lines: List[str], mapping: List[Tuple[str, str]], lightning_by: str = "") -> List[str]:
    """Replace imports of standalone package to lightning.

    Adapted from `assistant._replace_imports`
    """
    out = lines[:]
    for source_import, target_import in mapping:
        for i, ln in enumerate(out):
            out[i] = re.sub(
                rf"([^_/@]|^){source_import}([^_\w/]|$)",
                rf"\1{target_import}\2",
                ln,
            )
            if lightning_by:  # in addition, replace base package
                out[i] = out[i].replace("from lightning import ", f"from {lightning_by} import ")
                out[i] = out[i].replace("import lightning ", f"import {lightning_by} ")
    return out


def _replace_unified_imports(lines: List[str], mapping: List[Tuple[str, str]], lightning_by: str = "") -> List[str]:
    """Replace imports of standalone package to unified lightning.

    Adapted from `assistant._replace_imports`
    """
    out = lines[:]
    for source_import, target_import in mapping:
        for i, ln in enumerate(out):
            out[i] = re.sub(
                rf"([^_/@]|^){source_import}([^_\w/]|$)",
                rf"\1{target_import}\2",
                ln,
            )
            if lightning_by:  # in addition, replace base package
                out[i] = out[i].replace("from lightning import ", f"from {lightning_by} import ")
                out[i] = out[i].replace("import lightning ", f"import {lightning_by} ")
    return out


def copy_replace_imports(
    source_dir: str,
    source_imports: Sequence[str],
    target_imports: Sequence[str],
    target_dir: Optional[str] = None,
    lightning_by: str = "",
) -> None:
    """Replace package content with import adjustments.

    Adapted from `assistant.copy_replace_imports`
    """
    print(f"Replacing imports: {locals()}")
    assert len(source_imports) == len(target_imports), (
        "source and target imports must have the same length, "
        f"source: {len(source_imports)}, target: {len(target_imports)}"
    )
    if target_dir is None:
        target_dir = source_dir

    ls = _retrieve_files(source_dir)
    for fp in ls:
        fp_new = fp.replace(source_dir, target_dir)
        _, ext = os.path.splitext(fp)
        if ext in (".png", ".jpg", ".ico"):
            os.makedirs(dirname(fp_new), exist_ok=True)
            if not isfile(fp_new):
                shutil.copy(fp, fp_new)
            continue
        elif ext in (".pyc",):
            continue
        # Try to parse everything else
        with open(fp, encoding="utf-8") as fo:
            try:
                lines = fo.readlines()
            except UnicodeDecodeError:
                # a binary file, skip
                print(f"Skipped replacing imports for {fp}")
                continue
        lines = _replace_standalone_imports(lines, list(zip(source_imports, target_imports)), lightning_by=lightning_by)
        os.makedirs(os.path.dirname(fp_new), exist_ok=True)
        with open(fp_new, "w", encoding="utf-8") as fo:
            fo.writelines(lines)


def use_standalone_pl(mapping, src_dirs: ValuesView) -> None:
    for in_place_path in src_dirs:
        for _, _ in mapping.items():
            copy_replace_imports(
                source_dir=str(in_place_path),
                source_imports=mapping.keys(),
                target_imports=mapping.values(),
                target_dir=str(in_place_path),
            )


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

    @staticmethod
    def copy_replace_imports(
        source_dir: str,
        source_import: str,
        target_import: str,
        target_dir: Optional[str] = None,
        lightning_by: str = "",
    ) -> None:
        """Copy package content in-place with import adjustments.

        Adapated from Lightning version of `assistant`.
        """
        source_imports = source_import.strip().split(",")
        target_imports = target_import.strip().split(",")
        copy_replace_imports(
            source_dir, source_imports, target_imports, target_dir=target_dir, lightning_by=lightning_by
        )


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(AssistantCLI, as_positional=False)
