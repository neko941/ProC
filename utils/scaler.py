import polars as pl
import numpy as np
from abc import ABC, abstractmethod

class ScalerBase(ABC):
    def __init__(self):
        self.statistics = None

    def get_statistics(self, df: pl.DataFrame, columns: list[str]) -> dict:
        """Calculates scaling statistics and returns them as a nested dictionary."""
        stats_dict = {}
        for col in columns:
            col_stats = {}
            col_stats["min"] = df.select(pl.col(col).min()).item()  # Get the minimum value
            col_stats["max"] = df.select(pl.col(col).max()).item()  # Get the maximum value
            col_stats["mean"] = df.select(pl.col(col).mean()).item()  # Get the mean
            col_stats["std"] = df.select(pl.col(col).std()).item()  # Get the standard deviation
            col_stats["q1"] = df.select(pl.col(col).quantile(0.25)).item()  # First quartile
            col_stats["q3"] = df.select(pl.col(col).quantile(0.75)).item()  # Third quartile
            col_stats["range"] = col_stats["max"] - col_stats["min"]
            col_stats["iqr"] = col_stats["q3"] - col_stats["q1"]

            stats_dict[col] = col_stats

        self.statistics = stats_dict
        return stats_dict

    @abstractmethod
    def transform(self, df: pl.DataFrame, columns: list[str] = None) -> pl.DataFrame:
        """Abstract method for scaling transformation."""
        pass

    @abstractmethod
    def inverse_transform(self, df: pl.DataFrame, columns: list[str] = None) -> pl.DataFrame:
        """Abstract method for inverse scaling transformation."""
        pass

class MinMaxScaler(ScalerBase):
    def transform(self, df: pl.DataFrame, columns: list[str] = None) -> pl.DataFrame:
        """Min-max scaling."""
        if columns is None:
            columns = list(self.statistics.keys())
        return df.with_columns(
            [
                (pl.col(col) - self.statistics[col]["min"])
                / (self.statistics[col]["max"] - self.statistics[col]["min"])
                for col in columns
            ]
        )

    def inverse_transform(self, df: pl.DataFrame, columns: list[str] = None) -> pl.DataFrame:
        """Inverse min-max scaling."""
        if columns is None:
            columns = list(self.statistics.keys())
        return df.with_columns(
            [
                pl.col(col) * (self.statistics[col]["max"] - self.statistics[col]["min"])
                + self.statistics[col]["min"]
                for col in columns
            ]
        )

class StandardScaler(ScalerBase):
    def transform(self, df: pl.DataFrame, columns: list[str] = None) -> pl.DataFrame:
        """Standard scaling."""
        if columns is None:
            columns = list(self.statistics.keys())
        return df.with_columns(
            [
                (pl.col(col) - self.statistics[col]["mean"]) / self.statistics[col]["std"]
                for col in columns
            ]
        )

    def inverse_transform(self, df: pl.DataFrame, columns: list[str] = None) -> pl.DataFrame:
        """Inverse standard scaling."""
        if columns is None:
            columns = list(self.statistics.keys())
        return df.with_columns(
            [
                pl.col(col) * self.statistics[col]["std"] + self.statistics[col]["mean"]
                for col in columns
            ]
        )

class RobustScaler(ScalerBase):
    def transform(self, df: pl.DataFrame, columns: list[str] = None) -> pl.DataFrame:
        """Robust scaling."""
        if columns is None:
            columns = list(self.statistics.keys())
        return df.with_columns(
            [
                (pl.col(col) - self.statistics[col]["q2"]) / self.statistics[col]["iqr"]
                for col in columns
            ]
        )

    def inverse_transform(self, df: pl.DataFrame, columns: list[str] = None) -> pl.DataFrame:
        """Inverse robust scaling."""
        if columns is None:
            columns = list(self.statistics.keys())
        return df.with_columns(
            [
                pl.col(col) * self.statistics[col]["iqr"] + self.statistics[col]["q2"]
                for col in columns
            ]
        )

scaler_dict = {
    'minmax' : MinMaxScaler(),
    'standard' : StandardScaler(),
    'robust' : RobustScaler()
}
