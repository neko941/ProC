import pandas as pd
import polars as pl
import numpy as np

class MemoryReducer:
    def __init__(self):
        self.numeric_int_types = [np.int8, np.int16, np.int32, np.int64]
        self.numeric_float_types = [np.float16, np.float32, np.float64]
        self.numeric_uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]

    def _memory_unit_conversion(self, before, after):
        units = ["bytes", "KB", "MB", "GB", "TB", "PB"]
        unit_index = 0

        # Loop until the appropriate unit is found
        while before >= 1024 and unit_index < len(units) - 1:
            before /= 1024
            after /= 1024
            unit_index += 1

        return before, after, units[unit_index]

    def _print_info(self, before, after, data_type):
        reduction = 100.0 * (before - after) / before
        reduction_str = f"{reduction:.2f}% reduction"
        before, after, unit = self._memory_unit_conversion(before=before, after=after)

        print(f"Reduced {data_type} memory usage from {before:.4f} {unit} to {after:.4f} {unit} ({reduction_str})")

    def _cast_to_optimal_type(self, c_min, c_max, current_type):
        """Helper function to cast to the optimal numeric type based on min and max values."""
        for dtype in self.numeric_int_types:
            if np.can_cast(c_min, dtype) and np.can_cast(c_max, dtype):
                return dtype if current_type != 'float' else None
        for dtype in self.numeric_float_types:
            if np.can_cast(c_min, dtype) and np.can_cast(c_max, dtype):
                return dtype
        return None  # Return None if no suitable type is found

    def reduce_pandas_df(self, df, info=False):
        before = df.memory_usage().sum()

        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                optimal_type = self._cast_to_optimal_type(c_min, c_max, 'float' if 'float' in str(col_type) else 'int')
                if optimal_type:
                    df[col] = df[col].astype(optimal_type)
            else:
                df[col] = df[col].astype('category')

        if info: self._print_info(before, df.memory_usage().sum(), "Pandas DataFrame")
        return df

    def reduce_polars_df(self, df, info=False):
        before = df.estimated_size('b')

        # Mapping from numpy dtypes to Polars dtypes
        np_to_pl_type_mapping = {
            np.int8: pl.Int8,
            np.int16: pl.Int16,
            np.int32: pl.Int32,
            np.int64: pl.Int64,
            np.uint8: pl.UInt8,
            np.uint16: pl.UInt16,
            np.uint32: pl.UInt32,
            np.uint64: pl.UInt64,
            np.float16: pl.Float32,
            np.float32: pl.Float32,
            np.float64: pl.Float64,
        }

        for col in df.columns:
            col_type = df[col].dtype
            if col_type == pl.Utf8:
                df = df.with_columns(df[col].cast(pl.Categorical))
            elif col_type in list(np_to_pl_type_mapping.values()):
                c_min = df[col].min()
                c_max = df[col].max()
                optimal_type = self._cast_to_optimal_type(c_min, c_max, 'float' if 'float' in str(col_type) else 'int')
                # Convert numpy dtype to Polars dtype using the mapping
                optimal_type = np_to_pl_type_mapping.get(optimal_type)
                if optimal_type:
                    df = df.with_columns(df[col].cast(optimal_type))

        if info: self._print_info(before, df.estimated_size('b'), "Polars DataFrame")
        return df

    def reduce_numpy_array(self, arr, info=False):
        before = arr.nbytes
        arr_max = arr.max()
        arr_min = arr.min()
        optimal_type = self._cast_to_optimal_type(arr_min, arr_max, 'float' if 'float' in str(arr.dtype) else 'int')
        if optimal_type:
            arr = arr.astype(optimal_type)

        if info: self._print_info(before, arr.nbytes, "NumPy array")
        return arr

    def __call__(self, data, info=False):
        if isinstance(data, pd.DataFrame):
            return self.reduce_pandas_df(data, info)
        elif isinstance(data, pl.DataFrame):
            return self.reduce_polars_df(data, info)
        elif isinstance(data, np.ndarray):
            return self.reduce_numpy_array(data, info)
        else:
            raise TypeError("Unsupported data type")
