import os
from enum import Enum
from typing import NamedTuple, Callable

class OSEnvToggle(NamedTuple):
    env_var_name: str
    default: str

class DependencyPatch(NamedTuple):
    """Ephemeral dependency patches to conditionally apply to the environment.

    To activate a given patch, all defined `condition` callables must return truthy and the `env_flag` must be set (or
    must default) to '1'
    """
    condition: tuple[Callable]  # typically a tuple of `lwt_compare_version` to define version dependency
    env_flag: OSEnvToggle  # a tuple defining the environment variable based condition and its default if not set
    function: Callable
    patched_package: str
    description: str


def _dep_patch_repr(self):
    return f'Patch of {self.patched_package}: {self.description})'

DependencyPatch.__repr__ = _dep_patch_repr

# N.B. One needs to ensure they patch all relevant _calling module_ references to patch targets since we usually patch
# after those calling modules have already secured the original (unpatched) references.

# Example patch definition (not applied, for enum inspection purposes)
example_patch = DependencyPatch(
    condition=(lambda: False,),  # Never applied, this is just an example
    env_flag=OSEnvToggle("ENABLE_FTS_EXAMPLE_PATCH", default="0"),
    function=lambda: None,  # No-op
    patched_package='some_package',
    description='Example patch for ExpPatch enum inspection'
)

class ExpPatch(Enum):
    EXAMPLE_PATCH = example_patch

_DEFINED_PATCHES = set(ExpPatch)
_ACTIVE_PATCHES = set()

for defined_patch in _DEFINED_PATCHES:
    if all(defined_patch.value.condition) and os.environ.get(defined_patch.value.env_flag.env_var_name,
                                                             defined_patch.value.env_flag.default) == "1":
        defined_patch.value.function()
        _ACTIVE_PATCHES.add(defined_patch)
