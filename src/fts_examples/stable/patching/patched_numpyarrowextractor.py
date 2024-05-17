from fts_examples.stable.patching._patch_utils import _prepare_module_ctx


globals().update(_prepare_module_ctx('datasets.formatting.formatting', globals()))


# we ignore these for the entire file since we're using our global namespace trickeration to patch
# ruff: noqa: F821
# pyright: reportUndefinedVariable=false

class NumpyArrowExtractor(BaseArrowExtractor[dict, np.ndarray, dict]):

    def __init__(self, **np_array_kwargs):
        self.np_array_kwargs = np_array_kwargs

    def extract_row(self, pa_table: pa.Table) -> dict:
        return _unnest(self.extract_batch(pa_table))

    def extract_column(self, pa_table: pa.Table) -> np.ndarray:
        return self._arrow_array_to_numpy(pa_table[pa_table.column_names[0]])

    def extract_batch(self, pa_table: pa.Table) -> dict:
        return {col: self._arrow_array_to_numpy(pa_table[col]) for col in pa_table.column_names}

    def _arrow_array_to_numpy(self, pa_array: pa.Array) -> np.ndarray:
        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                array: List = [
                    row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                ]
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and all(
                    not _is_array_with_nulls(chunk) for chunk in pa_array.chunks
                )
                array: List = [
                    row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                ]
        else:
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only)
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and not _is_array_with_nulls(pa_array)
                array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only).tolist()
        if len(array) > 0:
            if any(
                (isinstance(x, np.ndarray) and (x.dtype == object or x.shape != array[0].shape))
                or (isinstance(x, float) and np.isnan(x))
                for x in array
            ):
                return np.array(array, copy=False, dtype=object)
        # return np.array(array, copy=False)
        return np.asarray(array)
