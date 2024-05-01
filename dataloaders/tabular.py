import os
import itertools
import numpy as np
import polars as pl
from typing import List, Tuple
from utils.memory import MemoryReducer
from .base import DatasetController

from utils.scaler import scaler_dict

class TabularDataset(DatasetController):
    def __init__(self,
                 source,
                 target_columns,
                 used_columns,
                 scaler=None,
                 info=True,
                 ratio=None,
                 indices=None,
                 ranges=None,
                 try_parse_dates=True,
                 low_memory=False,
                 split_first=False):
        self.source = source
        self.target_columns = target_columns
        self.used_columns = used_columns
        self.scaler = scaler
        if self.scaler: self.scaler = scaler_dict[scaler.lower()]
        self.info = info
        self.ratio = ratio
        self.indices = indices
        self.ranges = ranges
        self.low_memory = low_memory
        self.try_parse_dates = try_parse_dates
        self.mem_reducer = MemoryReducer()

        self.preprocessing()
        if split_first and ratio is not None:
            self._split_first()
        else:
            self._split_later()

        if info:
            print(f'{self.train_x.shape = }')
            print(f'{self.train_y.shape = }')
            print(f'{self.val_x.shape = }')
            print(f'{self.val_y.shape = }')
            print(f'{self.test_x.shape = }')
            print(f'{self.test_y.shape = }')

    def _split_array_by_ratio(self, arr, ratio):
        """Splits a NumPy array into segments based on the given ratio."""
        assert sum(ratio) == 1, "Ratios must sum to 1"
        sizes = (np.array(ratio) * len(arr)).astype(int)
        start_indices = np.cumsum([0] + list(sizes[:-1]))
        return [arr[start:start + size] for start, size in zip(start_indices, sizes)]

    def _split_first(self):
        self.x, self.y = self.split(
            df=self.df,
            used_columns=self.used_columns,
            target_columns=self.target_columns
        )
        self.train_x, self.val_x, self.test_x = self._split_array_by_ratio(self.x, self.ratio)
        self.train_y, self.val_y, self.test_y = self._split_array_by_ratio(self.y, self.ratio)
        # TODO: normalization
        if self.scaler:
            raise NotImplemented

    def _split_later(self):
        # Split into train, validation, and test sub-DataFrames
        self.train_raw = self.df.slice(*self.ranges[0])
        self.val_raw = self.df.slice(*self.ranges[1])
        self.test_raw = self.df.slice(*self.ranges[2])

        if self.scaler:
            # print(list(set(self.used_columns) | set(self.target_columns)))
            self.scaler.get_statistics(df=self.train_raw, columns=list(set(self.used_columns) | set(self.target_columns)))
            self.train = self.scaler.transform(self.train_raw)
            self.val = self.scaler.transform(self.val_raw)
            self.test = self.scaler.transform(self.test_raw)
        else:
            self.train = self.train_raw
            self.val = self.val_raw
            self.test = self.test_raw

        self.train_x, self.train_y = self.split(
            df=self.train,
            used_columns=self.used_columns,
            target_columns=self.target_columns
        )
        self.val_x, self.val_y = self.split(
            df=self.val,
            used_columns=self.used_columns,
            target_columns=self.target_columns
        )
        self.test_x, self.test_y = self.split(
            df=self.test,
            used_columns=self.used_columns,
            target_columns=self.target_columns
        )

    def preprocessing(self):
        self.df = self.read(source=self.source)
        self.df = self.df.shrink_to_fit()
        self.df = self.mem_reducer(
            data=self.df,
            info=self.info
        )
        self.target_columns, self.used_columns = self.fix_features(
            df=self.df,
            target_columns=self.target_columns,
            used_columns=self.used_columns
        )
        self.ranges = self.fix_ranges(
            df=self.df,
            ratio=self.ratio,
            indices=self.indices,
            ranges=self.ranges
        )

    def read(self, source):
        if os.path.isfile(source):
            return self._read_file(source)
        elif os.path.isdir(source):
            return self._read_dir(source)
        else:
            raise ValueError("Invalid path: must be a file or directory")

    def _read_file(self, file_path):
        if file_path.endswith('.csv'):
            return pl.read_csv(source=file_path, try_parse_dates=self.try_parse_dates, low_memory=self.low_memory)
        else:
            raise ValueError("Only CSV files are supported")

    def _read_dir(self, dir_path):
        dfs = []
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_name.endswith('.csv'):
                df = self._read_file(file_path=file_path)
                dfs.append(df)
        if not dfs:
            raise ValueError("No CSV files found in the directory")
        return dfs

    def fix_features(self, df, target_columns=None, used_columns=None):
        # TODO: change df to column names
        # TODO: check not list => convert to list
        # TODO: if len(columns)==1 and columns[0]='all'
        # Determine used_columns if not provided
        if used_columns is None: used_columns = [col for col in df.columns if df[col].dtype != pl.Datetime]
        if target_columns is None: target_columns = [col for col in df.columns if df[col].dtype != pl.Datetime]

        # TODO: check if exists features

        self.target_columns = target_columns
        self.used_columns = used_columns
        return target_columns, used_columns

    def fix_ranges(
        self,
        df: pl.DataFrame,
        ratio: Tuple[float, ...] = None,
        indices: List[Tuple[int, int]] = None,
        ranges: List[Tuple[int, int]] = None
    ) -> List[Tuple[int, int]]:
        """
        Determines the ranges for splitting a Polars DataFrame based on provided ratios, indices, or direct ranges.

        Args:
            df: The Polars DataFrame to be split.
            ratio: A tuple of ratios representing the proportions of data for each split (e.g., (0.7, 0.1, 0.2) for 70%, 10%, 20% splits). Defaults to None.
            indices: A list of tuples, where each tuple represents the start and end indices for a split. Defaults to None.
            ranges: A list of tuples, where each tuple represents the start index and length of a split. Defaults to None.

        Raises:
            ValueError: If more than one of `ratio`, `indices`, or `ranges` is provided.
            ValueError: If the provided ratios do not sum to 1.
            ValueError: If the provided indices are out of bounds or not in ascending order.
            ValueError: If the provided ranges are out of bounds.

        Returns:
            A list of tuples, where each tuple represents the start index and length of a split.
        """
        total_rows = len(df)

        provided_args = sum(arg is not None for arg in [ratio, indices, ranges])

        if provided_args == 0: ratio = (0.7, 0.1, 0.2)  # Default ratio if none are provided
        elif provided_args > 1: raise ValueError("Only one of 'ratio', 'indices', or 'ranges' should be provided.")

        if ratio is not None:
            ranges = self._fix_ranges_from_ratio(ratio, total_rows)
        elif indices is not None:
            ranges = self._fix_ranges_from_indices(indices, total_rows)
        elif ranges is not None:
            self._validate_ranges(ranges, total_rows)  # Validate directly provided ranges

        return ranges

    def _fix_ranges_from_ratio(
        self,
        ratio: Tuple[float, ...],
        total_rows: int
    ) -> List[Tuple[int, int]]:
        """
        Calculates ranges for data splitting based on provided ratios.

        Args:
            ratio: A tuple of ratios representing the proportions of data for each split.
            total_rows: The total number of rows in the DataFrame.

        Returns:
            A list of tuples, where each tuple represents the start index and length of a split.
        """
        if sum(ratio) > 1.0: raise ValueError("Ratios must sum at most 1.")
        segments = [round(r * total_rows) for r in ratio]
        ranges = [
            (start, length)
            for start, length in zip(itertools.accumulate([0] + segments[:-1]), segments)
        ]
        return ranges

    def _fix_ranges_from_indices(
        self,
        indices: List[Tuple[int, int]],
        total_rows: int
    ) -> List[Tuple[int, int]]:
        """
        Calculates ranges for data splitting based on provided start and end indices.

        Args:
            indices: A list of tuples, where each tuple represents the start and end indices for a split.
            total_rows: The total number of rows in the DataFrame.

        Raises:
            ValueError: If the provided indices are out of bounds or not in ascending order.

        Returns:
            A list of tuples, where each tuple represents the start index and length of a split.
        """
        ranges = []
        for start, end in indices:
            if start < 0 or end > total_rows or start >= end:
                """
                Start Index Out of Bounds: start < 0 checks if the starting index is negative, which would be invalid.
                End Index Out of Bounds: end > total_rows checks if the ending index is greater than the total number of rows in the DataFrame, which would also be invalid.
                Indices in Wrong Order: start >= end checks if the starting index is greater than or equal to the ending index, which would indicate an invalid range (the start should always be less than the end).
                """
                raise ValueError("Invalid index provided. Indices must be within DataFrame bounds and in ascending order.")
            ranges.append((start, end - start))
        return ranges

    def _validate_ranges(
        self,
        ranges: List[Tuple[int, int]],
        total_rows: int
    ) -> None:
        """
        Validates directly provided ranges to ensure they are within DataFrame bounds.

        Args:
            ranges: A list of tuples, where each tuple represents the start index and length of a split.
            total_rows: The total number of rows in the DataFrame.

        Raises:
            ValueError: If any provided range is out of bounds.
        """
        for start, length in ranges:
            if start < 0 or start + length > total_rows:
                raise ValueError("Invalid range provided. Ranges must be within DataFrame bounds.")

    def split(self, df, used_columns, target_columns):
        """
        Splits a Polars DataFrame into features (X) and targets (y) based on column names.

        Args:
            df (pl.DataFrame): The Polars DataFrame to split.
            used_columns (list[str]): A list of column names to include in the features (X).
            target_columns (list[str]): A list of column names to include in the targets (y).

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
                - x: The features DataFrame with the selected columns.
                - y: The targets DataFrame with the selected columns.
        """
        x = df[used_columns].to_numpy()
        y = df[target_columns].to_numpy()
        return x, y

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
