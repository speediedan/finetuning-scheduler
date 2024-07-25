from enum import Enum, auto


class AutoStrEnum(Enum):
    def _generate_next_value_(name, start, count, last_values) -> str:  # type: ignore
        return name

class ResultEnum(AutoStrEnum):
    """Characterization of an expected result value based on a test sample transformation or approximation."""
    default = auto()
    nondefault = auto()

# expected training path aliases
path_default = {0: (2, 4), 1: (6, 12), 2: (7, 14)}
path_default_orig = {0: (4, 4), 1: (12, 12), 2: (14, 14)}
path_default_orig_eo_dyn = {0: (4, 4), 1: (12, 12), 2: (14, 14), 3: (14, 14)}
path_ignore_p_uo = {0: (4, 4), 1: (12, 12), 2: (14, 14)}
path_8_14 = {0: (2, 4), 1: (7, 12), 2: (8, 14)}
path_8_16 = {0: (4, 8), 1: (7, 14), 2: (8, 16)}
path_5_10 = {0: (2, 4), 1: (3, 6), 2: (5, 10)}
path_ext_7_14 = {0: (2, 4), 1: (2, 4), 2: (6, 12), 3: (6, 12), 4: (7, 14)}
path_ext_8_16 = {0: (3, 6), 1: (7, 14), 2: (8, 16)}
path_optimlr_reinit = {0: (2, 4, "SGD", 0, 0.1), 1: (6, 12, "Adam", 32, 0.00021), 2: (7, 14, "SGD", 64, 0.002)}
lrs_path_default = {0: (0.1,), 1: (0.07, 1e-06), 2: (0.049, 7e-07, 1e-05)}
lrs_path_optimlr_reinit = {0: (0.1,), 1: (0.00021, 1e-06), 2: (0.002, 1e-06, 3e-06)}

# current_epoch: (bn_layer_state, len(curr_thawed_params), len(logical_param_translation(curr_thawed_params)))
path_bn_track_false = {
    0: (
        {
            8: {
                "layer_fqn": "model._fsdp_wrapped_module.2._fsdp_wrapped_module",
                "track_running_stats": False,
                "training": True,
                "running_mean": ResultEnum.default,
                "running_var": ResultEnum.default,
                "num_batches_tracked": 0,
                "requires_grad": False,
            },
            16: {
                "layer_fqn": "model._fsdp_wrapped_module.6._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.default,
                "running_var": ResultEnum.default,
                "num_batches_tracked": 0,
                "requires_grad": True,
            },
        },
        4,
        8,
    ),
    1: (
        {
            8: {
                "layer_fqn": "model._fsdp_wrapped_module.2._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.default,
                "running_var": ResultEnum.default,
                "num_batches_tracked": 0,
                "requires_grad": True,
            },
            16: {
                "layer_fqn": "model._fsdp_wrapped_module.6._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 16,
                "requires_grad": True,
            },
        },
        8,
        16,
    ),
    2: (
        {
            8: {
                "layer_fqn": "model._fsdp_wrapped_module.2._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.default,
                "running_var": ResultEnum.default,
                "num_batches_tracked": 0,
                "requires_grad": True,
            },
            16: {
                "layer_fqn": "model._fsdp_wrapped_module.6._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 16,
                "requires_grad": True,
            },
        },
        9,
        18,
    ),
    3: (
        {
            8: {
                "layer_fqn": "model._fsdp_wrapped_module.2._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 16,
                "requires_grad": True,
            },
            16: {
                "layer_fqn": "model._fsdp_wrapped_module.6._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 32,
                "requires_grad": True,
            },
        },
        9,
        18,
    ),
}

path_bn_track_true = {
    0: (
        {
            8: {
                "layer_fqn": "model._fsdp_wrapped_module.2._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.default,
                "running_var": ResultEnum.default,
                "num_batches_tracked": 0,
                "requires_grad": False,
            },
            16: {
                "layer_fqn": "model._fsdp_wrapped_module.6._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.default,
                "running_var": ResultEnum.default,
                "num_batches_tracked": 0,
                "requires_grad": True,
            },
        },
        4,
        8,
    ),
    1: (
        {
            8: {
                "layer_fqn": "model._fsdp_wrapped_module.2._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 16,
                "requires_grad": True,
            },
            16: {
                "layer_fqn": "model._fsdp_wrapped_module.6._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 16,
                "requires_grad": True,
            },
        },
        8,
        16,
    ),
    2: (
        {
            8: {
                "layer_fqn": "model._fsdp_wrapped_module.2._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 16,
                "requires_grad": True,
            },
            16: {
                "layer_fqn": "model._fsdp_wrapped_module.6._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 16,
                "requires_grad": True,
            },
        },
        9,
        18,
    ),
    3: (
        {
            8: {
                "layer_fqn": "model._fsdp_wrapped_module.2._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 32,
                "requires_grad": True,
            },
            16: {
                "layer_fqn": "model._fsdp_wrapped_module.6._fsdp_wrapped_module",
                "track_running_stats": True,
                "training": True,
                "running_mean": ResultEnum.nondefault,
                "running_var": ResultEnum.nondefault,
                "num_batches_tracked": 32,
                "requires_grad": True,
            },
        },
        9,
        18,
    ),
}
