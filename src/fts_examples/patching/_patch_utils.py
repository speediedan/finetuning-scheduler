import operator
import sys
from typing import Callable
import importlib.metadata
from packaging.version import Version
from functools import lru_cache


@lru_cache
def lwt_compare_version(package: str, op: Callable, version: str, use_base_version: bool = True,
                        local_version: str = None) -> bool:
    try:
        pkg_version = Version(importlib.metadata.version(package))
    except (importlib.metadata.PackageNotFoundError):
        return False
    except TypeError:
        # possibly mocked by Sphinx so needs to return True to generate summaries
        return True
    if local_version:
        if not operator.eq(local_version, pkg_version.local):
            return False
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version))


def _prepare_module_ctx(module_path, orig_globals):
    _orig_file = orig_globals.pop('__file__')
    orig_globals.update(vars(sys.modules.get(module_path)))
    orig_globals['__file__'] = _orig_file
    return orig_globals
