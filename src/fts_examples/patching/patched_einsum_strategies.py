import operator
from fts_examples.patching._patch_utils import _prepare_module_ctx, lwt_compare_version

# we ignore these for the entire file since we're using our global namespace trickeration to patch
# ruff: noqa: F821
# pyright: reportUndefinedVariable=false

if lwt_compare_version("torch", operator.ge, "2.5.0") and lwt_compare_version("torch", operator.le, "2.5.2"):
    globals().update(_prepare_module_ctx('torch.distributed.tensor._ops._einsum_strategy', globals()))


    def gen_einsum_strategies(
        equation: str,
        mesh: DeviceMesh,
        *,
        linearity: bool = False,
    ) -> OpStrategy:
        """Generate a strategy list for the ops that follow einsum style notation."""
        # parse einop equation and extract dims
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        all_mesh_dim_strategies = []

        # generate strategies for each mesh dim
        for mesh_dim in range(mesh.ndim):
            mesh_dim_strategies = []

            # placement list stores placements of [output, input1, input2, ...]
            # first we always have replicate all for inputs and output
            placement_list: List[Placement] = [Replicate()] * (len(input_dims) + 1)
            mesh_dim_strategies.append(placement_list)

            # if mesh.size(mesh_dim) <= 1:
            #     # only replicate strategy for mesh dim with size 1
            #     # TODO: see if this is valid for the submesh case
            #     continue

            # split batch dim
            for batch_dim in edims.batch_dims:
                output_batch_dim = output_dim.index(batch_dim)
                placement_list = [Shard(output_batch_dim)]
                for input_dim in input_dims:
                    input_batch_dim = input_dim.index(batch_dim)
                    placement_list.append(Shard(input_batch_dim))

                mesh_dim_strategies.append(placement_list)

            # split contracting dim
            for contracting_dim in edims.contracting_dims:
                placement_list = [Partial()]
                for input_dim in input_dims:
                    input_contracting_dim = input_dim.index(contracting_dim)
                    placement_list.append(Shard(input_contracting_dim))

                mesh_dim_strategies.append(placement_list)

            # split lhs free dim
            for lhs_dim in edims.lhs_out_only_dims:
                lhs_free_dim = output_dim.index(lhs_dim)
                # this means split the lhs input and output
                # i.e. S(0), R -> S(0)
                lhs_placement_list: List[Placement] = [
                    Shard(lhs_free_dim),
                    Shard(lhs_free_dim),
                    Replicate(),
                ]
                mesh_dim_strategies.append(lhs_placement_list)

            # split rhs free dim
            for rhs_dim in edims.rhs_out_only_dims:
                rhs_free_dim = output_dim.index(rhs_dim)
                rhs_placement_list: List[Placement] = [
                    Shard(rhs_free_dim),
                    Replicate(),
                    Shard(rhs_free_dim),
                ]
                mesh_dim_strategies.append(rhs_placement_list)

            # linearity strategy
            if linearity:
                linearity_placement_list: List[Placement] = [Partial()]
                for input_dim in input_dims:
                    linearity_placement_list.append(Partial())
                mesh_dim_strategies.append(linearity_placement_list)

            all_mesh_dim_strategies.append(mesh_dim_strategies)

        # generate strategies for entire mesh
        strategy_combs = itertools.product(*all_mesh_dim_strategies)

        # TODO: filter out invalid strategies, at this point we generate
        # all possible strategies without considering the whether the tensor
        # dim could be sharded or not, we would need to filter out invalid
        # strategies base on the actual tensor shape
        # (i.e. for Shard, tensor dim size must > mesh size)
        all_strategies = []
        for strategy_comb in strategy_combs:
            spec_list = []
            for specs in zip(*strategy_comb):
                spec_list.append(DTensorSpec(mesh, tuple(specs)))
            strat = PlacementStrategy(output_specs=spec_list[0], input_specs=spec_list[1:])
            all_strategies.append(strat)

        return OpStrategy(all_strategies)
