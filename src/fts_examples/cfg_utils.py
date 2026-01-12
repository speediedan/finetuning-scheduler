import importlib
import yaml
from typing import Any
from dataclasses import dataclass, field, asdict, fields

from transformers import PretrainedConfig
from lightning.pytorch.utilities.exceptions import MisconfigurationException


@dataclass
class ExperimentCfg:
    loss_parallel: bool = True
    experiment_tag: str = 'default'
    log_env_details: bool = True
    batch_size: int = 4
    num_workers: int = 2
    dataset_length: int = 32

@dataclass
class OptimizerCfg:
    class_fqn: str | None = None
    args: dict = field(default_factory=dict)

@dataclass
class LRSchedulerCfg:
    class_fqn: str | None = None
    args: dict = field(default_factory=dict)

@dataclass
class LightningLRSCfg:
    name: str = 'default'
    interval: str = 'epoch'
    frequency: int = 1

    def __post_init__(self):
        self._overridden = _is_overridden(self)

def optimizer_cfg_mapping_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', asdict(data))

def lr_scheduler_cfg_mapping_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', asdict(data))

def pretrained_cfg_mapping_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.to_dict())

# Register all custom representers to both base dumper classes
representers = {
    PretrainedConfig: pretrained_cfg_mapping_representer,
    OptimizerCfg: optimizer_cfg_mapping_representer,
    LRSchedulerCfg: lr_scheduler_cfg_mapping_representer
}

for dumper_cls in [yaml.Dumper, yaml.SafeDumper]:
    for cls, representer in representers.items():
        dumper_cls.add_multi_representer(cls, representer)

def _is_overridden(dataclass_instance) -> bool:
    is_overridden = False
    for f in fields(dataclass_instance.__class__):
        if getattr(dataclass_instance, f.name) != f.default:
            is_overridden = True
            break
    return is_overridden

def resolve_funcs(cfg_obj: Any, func_type: str) -> list:
    resolved_funcs = []
    funcs_to_resolve = getattr(cfg_obj, func_type)
    if not isinstance(funcs_to_resolve, list):
        funcs_to_resolve = [funcs_to_resolve]
    for func_or_qualname in funcs_to_resolve:
        if callable(func_or_qualname):
            resolved_funcs.append(func_or_qualname)  # TODO: inspect if signature is appropriate for custom hooks
        elif func_or_qualname == 'identity_lambda':  # special case often used e.g. w/ saved_tensors_hooks
            resolved_funcs.append(lambda x: x)
        else:
            try:
                module, func = func_or_qualname.rsplit(".", 1)
                mod = importlib.import_module(module)
                resolved_func = getattr(mod, func, None)
                if callable(resolved_func):
                    resolved_funcs.append(resolved_func)
                else:
                    raise MisconfigurationException(f"Custom function {func} from module {module} is not callable!")
            except (AttributeError, ImportError) as e:
                err_msg = f"Unable to import and resolve specified function {func} from module {module}: {e}"
                raise MisconfigurationException(err_msg)
    return resolved_funcs
