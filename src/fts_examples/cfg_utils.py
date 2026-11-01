import yaml
from dataclasses import dataclass, field, asdict, fields

from transformers import PretrainedConfig


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
