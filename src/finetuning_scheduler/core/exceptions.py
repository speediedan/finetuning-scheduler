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
r"""
Fine-Tuning Scheduler Core Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exceptions raised by the framework-neutral Fine-Tuning Scheduler core.

"""
from typing import Any


class FTSConfigurationError(ValueError):
    """Raised when a fine-tuning schedule or Fine-Tuning Scheduler configuration is invalid.

    A training host may register a subclass with :func:`register_configuration_error`. Once registered, raising
    ``FTSConfigurationError`` produces an instance of that subclass, so core code raises one type while each host
    receives the exception type its users already catch. The Lightning integration registers a subclass that is also a
    ``MisconfigurationException``. Raising a more specific subclass (such as
    :class:`~finetuning_scheduler.core.views.Unresolvable`) produces an instance of both it and the registered subclass.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "FTSConfigurationError":
        host = _HOST_ERROR_CLS
        if host is None or issubclass(cls, host):
            target: type = cls
        elif cls is FTSConfigurationError:
            target = host
        else:
            target = _with_host(cls, host)
        return ValueError.__new__(target, *args)


_HOST_ERROR_CLS: type[FTSConfigurationError] | None = None
# a specific error class combined with the registered host class, created once per pair
_WITH_HOST: dict[tuple[type, type], type] = {}


def _with_host(cls: type, host: type) -> type:
    """``cls`` combined with the registered host error class, named as ``cls`` is."""
    if (cls, host) not in _WITH_HOST:
        combined = type(cls.__name__, (cls, host), {"__module__": cls.__module__, "__qualname__": cls.__qualname__})
        _WITH_HOST[(cls, host)] = combined
    return _WITH_HOST[(cls, host)]


def register_configuration_error(error_cls: type[FTSConfigurationError]) -> None:
    """Register the :class:`FTSConfigurationError` subclass raised in place of the base class.

    Registration is idempotent; registering a different subclass replaces the previous one.

    Args:
        error_cls: A subclass of :class:`FTSConfigurationError`.

    Raises:
        TypeError: If ``error_cls`` is not a subclass of :class:`FTSConfigurationError`.
    """
    if not (isinstance(error_cls, type) and issubclass(error_cls, FTSConfigurationError)):
        raise TypeError(f"{error_cls!r} is not a subclass of `FTSConfigurationError`.")
    global _HOST_ERROR_CLS
    _HOST_ERROR_CLS = error_cls


def configuration_error_cls() -> type[FTSConfigurationError]:
    """Return the exception type currently raised for :class:`FTSConfigurationError`."""
    return _HOST_ERROR_CLS or FTSConfigurationError
