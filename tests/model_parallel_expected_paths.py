import torch

from tests.helpers.common import DeviceMeshSummary

## Expected Test Result Configuration Aliases

## example template result, providing TP weight and FSDP module states you want a test to validate
# state_key: ({p_states, fsdp_mod_states}, len(self._fts_state._curr_thawed_params))
# basic_template_result = {
#     0: (
#         {"p_states": {
#             "model.layers.0.feed_forward.w2.weight": {},
#             "model.layers.0.feed_forward.w2.bias": {},
#             "model.layers.1.feed_forward.w2.weight": {},
#             "model.layers.1.feed_forward.w2.bias": {},
#             "model.norm.weight": {},
#             "model.norm.bias": {},
#             "model.output.weight": {},
#         }},
#         3,
#     ),
#     1: (
#         {"p_states": {
#             "model.layers.0.feed_forward.w2.weight": {},
#             "model.layers.0.feed_forward.w2.bias": {},
#             "model.layers.1.feed_forward.w2.weight": {},
#             "model.layers.1.feed_forward.w2.bias": {},
#             "model.norm.weight": {},
#             "model.norm.bias": {},
#             "model.output.weight": {},
#         }},
#         27,
#     ),
#     2: (
#         {"p_states": {
#             "model.layers.0.feed_forward.w2.weight": {},
#             "model.layers.0.feed_forward.w2.bias": {},
#             "model.layers.1.feed_forward.w2.weight": {},
#             "model.layers.1.feed_forward.w2.bias": {},
#             "model.norm.weight": {},
#             "model.norm.bias": {},
#             "model.output.weight": {},
#         }},
#         29,
#     ),
# }

# extended_fsdp_template_result = {
#     0: (
#         {"p_states": {
#             "model.layers.0.feed_forward.w2.weight": {},
#             "model.layers.0.feed_forward.w2.bias": {},
#             "model.layers.1.feed_forward.w2.weight": {},
#             "model.layers.1.feed_forward.w2.bias": {},
#             "model.norm.weight": {},
#             "model.norm.bias": {},
#             "model.output.weight": {},
#         },
#         "fsdp_mod_states": {
#             "model.layers.0": {},
#             "model.layers.1": {},
#             "model.norm": {},
#             "model.output": {},
#         },
#         },
#         3,
#     ),
#     1: (
#         {"p_states": {
#             "model.layers.0.feed_forward.w2.weight": {},
#             "model.layers.0.feed_forward.w2.bias": {},
#             "model.layers.1.feed_forward.w2.weight": {},
#             "model.layers.1.feed_forward.w2.bias": {},
#             "model.norm.weight": {},
#             "model.norm.bias": {},
#             "model.output.weight": {},
#         },
#         "fsdp_mod_states": {
#             "model.layers.0": {},
#             "model.layers.1": {},
#             "model.norm": {},
#             "model.output": {},
#         },
#         },
#         27,
#     ),
#     2: (
#         {"p_states": {
#             "model.layers.0.feed_forward.w2.weight": {},
#             "model.layers.0.feed_forward.w2.bias": {},
#             "model.layers.1.feed_forward.w2.weight": {},
#             "model.layers.1.feed_forward.w2.bias": {},
#             "model.norm.weight": {},
#             "model.norm.bias": {},
#             "model.output.weight": {},
#         },
#         "fsdp_mod_states": {
#             "model.layers.0": {},
#             "model.layers.1": {},
#             "model.norm": {},
#             "model.output": {},
#         },
#         },
#         29,
#     ),
# }

path_tt_tp_no_fsdp = {
    0: (
        {"p_states": {
            "model.layers.0.feed_forward.w2.weight": {},
            "model.layers.0.feed_forward.w2.bias": {},
            "model.layers.1.feed_forward.w2.weight": {},
            "model.layers.1.feed_forward.w2.bias": {},
            "model.norm.weight": {},
            "model.norm.bias": {},
            "model.output.weight": {},
        }},
        3,
    ),
    1: (
        {"p_states": {
            "model.layers.0.feed_forward.w2.weight": {},
            "model.layers.0.feed_forward.w2.bias": {},
            "model.layers.1.feed_forward.w2.weight": {},
            "model.layers.1.feed_forward.w2.bias": {},
            "model.norm.weight": {},
            "model.norm.bias": {},
            "model.output.weight": {},
        }},
        27,
    ),
    2: (
        {"p_states": {
            "model.layers.0.feed_forward.w2.weight": {},
            "model.layers.0.feed_forward.w2.bias": {},
            "model.layers.1.feed_forward.w2.weight": {},
            "model.layers.1.feed_forward.w2.bias": {},
            "model.norm.weight": {},
            "model.norm.bias": {},
            "model.output.weight": {},
        }},
        29,
    ),
}

path_tt_fsdp_no_tp = {
    0: (
        {"p_states": {
            "model.layers.0.feed_forward.w2.weight": {"requires_grad": False, "is_DTensor": True},
            "model.layers.0.feed_forward.w2.bias": {"requires_grad": False, "is_DTensor": True},
            "model.layers.1.feed_forward.w2.weight": {"requires_grad": False, "is_DTensor": True},
            "model.layers.1.feed_forward.w2.bias": {"requires_grad": False, "is_DTensor": True},
            "model.norm.weight": {"requires_grad": True, "is_DTensor": True},
            "model.norm.bias": {"requires_grad": True, "is_DTensor": True},
            "model.output.weight": {"requires_grad": True, "is_DTensor": True},
        }},
        3,
    ),
    1: (
        {"p_states": {
            "model.layers.0.feed_forward.w2.weight": {"requires_grad": True, "is_DTensor": True},
            "model.layers.0.feed_forward.w2.bias": {"requires_grad": True, "is_DTensor": True},
            "model.layers.1.feed_forward.w2.weight": {"requires_grad": True, "is_DTensor": True},
            "model.layers.1.feed_forward.w2.bias": {"requires_grad": True, "is_DTensor": True},
            "model.norm.weight": {"requires_grad": True, "is_DTensor": True},
            "model.norm.bias": {"requires_grad": True, "is_DTensor": True},
            "model.output.weight": {"requires_grad": True, "is_DTensor": True},
        }},
        27,
    ),
    2: (
        {"p_states": {
            "model.layers.0.feed_forward.w2.weight": {"requires_grad": True, "is_DTensor": True},
            "model.layers.0.feed_forward.w2.bias": {"requires_grad": True, "is_DTensor": True},
            "model.layers.1.feed_forward.w2.weight": {"requires_grad": True, "is_DTensor": True},
            "model.layers.1.feed_forward.w2.bias": {"requires_grad": True, "is_DTensor": True},
            "model.norm.weight": {"requires_grad": True, "is_DTensor": True},
            "model.norm.bias": {"requires_grad": True, "is_DTensor": True},
            "model.output.weight": {"requires_grad": True, "is_DTensor": True},
        }},
        29,
    ),
}

path_ff_fsdp_tp = {
    0: (
        {
            "p_states": {
                "model.w2.weight": {
                    "is_DTensor": True,
                    "requires_grad": False,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([64, 32]),
                    "local_shape": torch.Size([32, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2, mesh_ndim=2, mesh_shape=(2, 1),
                        mesh_dim_names=("data_parallel", "tensor_parallel"),
                        placement_summary=["shard(dim=0)", "shard(dim=0)"],
                    ),
                },
                "model.w3.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2, 64]),
                    "local_shape": torch.Size([1, 64]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2, mesh_ndim=2, mesh_shape=(2, 1),
                        mesh_dim_names=("data_parallel", "tensor_parallel"),
                        placement_summary=["shard(dim=0)", "shard(dim=1)"],
                    ),
                },
                "model.w3.bias": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2]),
                    "local_shape": torch.Size([1]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=1, mesh_ndim=2, mesh_shape=(2, 1),
                        mesh_dim_names=("data_parallel", "tensor_parallel"),
                        placement_summary=["shard(dim=0)", "replica"],
                    ),
                },
            },
            "fsdp_mod_states": {
                "model.w2": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w2.weight", torch.Size([64, 32]), torch.Size([32, 32])),
                        ("w2.bias", torch.Size([64]), torch.Size([32])),
                    ],
                },
                "model.w3": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w3.weight", torch.Size([2, 64]), torch.Size([1, 64])),
                        ("w3.bias", torch.Size([2]), torch.Size([1])),
                    ],
                },
            },
        },
        2,
    ),
    1: (
        {
            "p_states": {
                "model.w2.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([64, 32]),
                    "local_shape": torch.Size([32, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2, mesh_ndim=2, mesh_shape=(2, 1),
                        mesh_dim_names=("data_parallel", "tensor_parallel"),
                        placement_summary=["shard(dim=0)", "shard(dim=0)"],
                    ),
                },
                "model.w3.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2, 64]),
                    "local_shape": torch.Size([1, 64]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2, mesh_ndim=2, mesh_shape=(2, 1),
                        mesh_dim_names=("data_parallel", "tensor_parallel"),
                        placement_summary=["shard(dim=0)", "shard(dim=1)"],
                    ),
                },
                "model.w3.bias": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2]),
                    "local_shape": torch.Size([1]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=1, mesh_ndim=2, mesh_shape=(2, 1),
                        mesh_dim_names=("data_parallel", "tensor_parallel"),
                        placement_summary=["shard(dim=0)", "replica"],
                    ),
                },
            },
            "fsdp_mod_states": {
                "model.w2": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w2.weight", torch.Size([64, 32]), torch.Size([32, 32])),
                        ("w2.bias", torch.Size([64]), torch.Size([32])),
                    ],
                },
                "model.w3": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w3.weight", torch.Size([2, 64]), torch.Size([1, 64])),
                        ("w3.bias", torch.Size([2]), torch.Size([1])),
                    ],
                },
            },
        },
        4,
    ),
    2: (
        {
            "p_states": {
                "model.w2.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([64, 32]),
                    "local_shape": torch.Size([32, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2, mesh_ndim=2, mesh_shape=(2, 1),
                        mesh_dim_names=("data_parallel", "tensor_parallel"),
                        placement_summary=["shard(dim=0)", "shard(dim=0)"],
                    ),
                },
                "model.w3.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2, 64]),
                    "local_shape": torch.Size([1, 64]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2, mesh_ndim=2, mesh_shape=(2, 1),
                        mesh_dim_names=("data_parallel", "tensor_parallel"),
                        placement_summary=["shard(dim=0)", "shard(dim=1)"],
                    ),
                },
                "model.w3.bias": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2]),
                    "local_shape": torch.Size([1]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=1, mesh_ndim=2, mesh_shape=(2, 1),
                        mesh_dim_names=("data_parallel", "tensor_parallel"),
                        placement_summary=["shard(dim=0)", "replica"],
                    ),
                },
            },
            "fsdp_mod_states": {
                "model.w2": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w2.weight", torch.Size([64, 32]), torch.Size([32, 32])),
                        ("w2.bias", torch.Size([64]), torch.Size([32])),
                    ],
                },
                "model.w3": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w3.weight", torch.Size([2, 64]), torch.Size([1, 64])),
                        ("w3.bias", torch.Size([2]), torch.Size([1])),
                    ],
                },
            },
        },
        6,
    ),
}

path_ff_fsdp_no_tp = {
    0: (
        {
            "p_states": {
                "model.w2.weight": {
                    "is_DTensor": True,
                    "requires_grad": False,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([64, 32]),
                    "local_shape": torch.Size([32, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("data_parallel",),
                        placement_summary=["shard(dim=0)"],
                    ),
                },
                "model.w3.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2, 64]),
                    "local_shape": torch.Size([1, 64]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("data_parallel",),
                        placement_summary=["shard(dim=0)"],
                    ),
                },
            },
            "fsdp_mod_states": {
                "model.w2": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w2.weight", torch.Size([64, 32]), torch.Size([32, 32])),
                        ("w2.bias", torch.Size([64]), torch.Size([32])),
                    ],
                },
                "model.w3": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w3.weight", torch.Size([2, 64]), torch.Size([1, 64])),
                        ("w3.bias", torch.Size([2]), torch.Size([1])),
                    ],
                },
            },
        },
        2,
    ),
    1: (
        {
            "p_states": {
                "model.w2.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([64, 32]),
                    "local_shape": torch.Size([32, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("data_parallel",),
                        placement_summary=["shard(dim=0)"],
                    ),
                },
                "model.w3.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2, 64]),
                    "local_shape": torch.Size([1, 64]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("data_parallel",),
                        placement_summary=["shard(dim=0)"],
                    ),
                },
            },
            "fsdp_mod_states": {
                "model.w2": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w2.weight", torch.Size([64, 32]), torch.Size([32, 32])),
                        ("w2.bias", torch.Size([64]), torch.Size([32])),
                    ],
                },
                "model.w3": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w3.weight", torch.Size([2, 64]), torch.Size([1, 64])),
                        ("w3.bias", torch.Size([2]), torch.Size([1])),
                    ],
                },
            },
        },
        4,
    ),
    2: (
        {
            "p_states": {
                "model.w2.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([64, 32]),
                    "local_shape": torch.Size([32, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("data_parallel",),
                        placement_summary=["shard(dim=0)"],
                    ),
                },
                "model.w3.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2, 64]),
                    "local_shape": torch.Size([1, 64]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("data_parallel",),
                        placement_summary=["shard(dim=0)"],
                    ),
                },
            },
            "fsdp_mod_states": {
                "model.w2": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w2.weight", torch.Size([64, 32]), torch.Size([32, 32])),
                        ("w2.bias", torch.Size([64]), torch.Size([32])),
                    ],
                },
                "model.w3": {
                    "is_fsdp_managed": True,
                    "prec_policy_summ": (None, None, None, True),
                    "param_group_summ": [
                        ("w3.weight", torch.Size([2, 64]), torch.Size([1, 64])),
                        ("w3.bias", torch.Size([2]), torch.Size([1])),
                    ],
                },
            },
        },
        6,
    ),
}

path_ff_tp_no_fsdp = {
    0: (
        {
            "p_states": {
                "model.w2.weight": {
                    "is_DTensor": True,
                    "requires_grad": False,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([64, 32]),
                    "local_shape": torch.Size([32, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("tensor_parallel",),
                        placement_summary=["shard(dim=0)"],
                    ),
                },
                "model.w3.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2, 64]),
                    "local_shape": torch.Size([2, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("tensor_parallel",),
                        placement_summary=["shard(dim=1)"],
                    ),
                },
                "model.w3.bias": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2]),
                    "local_shape": torch.Size([2]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=1,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("tensor_parallel",),
                        placement_summary=["replica"],
                    ),
                },
            },
            "fsdp_mod_states": {},
        },
        2,
    ),
    1: (
        {
            "p_states": {
                "model.w2.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([64, 32]),
                    "local_shape": torch.Size([32, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("tensor_parallel",),
                        placement_summary=["shard(dim=0)"],
                    ),
                },
                "model.w3.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2, 64]),
                    "local_shape": torch.Size([2, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("tensor_parallel",),
                        placement_summary=["shard(dim=1)"],
                    ),
                },
                "model.w3.bias": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2]),
                    "local_shape": torch.Size([2]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=1,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("tensor_parallel",),
                        placement_summary=["replica"],
                    ),
                },
            },
            "fsdp_mod_states": {},
        },
        4,
    ),
    2: (
        {
            "p_states": {
                "model.w2.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([64, 32]),
                    "local_shape": torch.Size([32, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("tensor_parallel",),
                        placement_summary=["shard(dim=0)"],
                    ),
                },
                "model.w3.weight": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2, 64]),
                    "local_shape": torch.Size([2, 32]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=2,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("tensor_parallel",),
                        placement_summary=["shard(dim=1)"],
                    ),
                },
                "model.w3.bias": {
                    "is_DTensor": True,
                    "requires_grad": True,
                    "dtype": torch.float32,
                    "orig_shape": torch.Size([2]),
                    "local_shape": torch.Size([2]),
                    "device_mesh": DeviceMeshSummary(
                        tensor_ndim=1,
                        mesh_ndim=1,
                        mesh_shape=(2,),
                        mesh_dim_names=("tensor_parallel",),
                        placement_summary=["replica"],
                    ),
                },
            },
            "fsdp_mod_states": {},
        },
        6,
    ),
}
