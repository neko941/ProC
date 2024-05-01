import polars as pl
import datetime
import numpy as np
from .tabular import TabularDataset
from tqdm import tqdm

class TimeSeriesDataset(TabularDataset):
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
                 seq_len=96,
                 pred_len=96,
                 offset=None,
                 granularity=1,
                 unit='hour',
                 date_column='date',
                 split_first=False
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.granularity = granularity
        self.unit = unit
        self.date_column = date_column
        self.offset = offset if offset is not None else self.pred_len

        super().__init__(
            source=source,
            target_columns=target_columns,
            used_columns=used_columns,
            scaler=scaler,
            info=info,
            ratio=ratio,
            indices=indices,
            ranges=ranges,
            try_parse_dates=try_parse_dates,
            low_memory=low_memory,
            split_first=split_first
        )

    def preprocessing(self):
        super().preprocessing()
        self.df = self.fill_timestamps(
            df=self.df,
            date_column=self.date_column,
            granularity=self.granularity,
            unit=self.unit
        )
        self.df = self.df.sort(by=self.date_column)

    def fill_timestamps(
        self,
        df: pl.DataFrame,
        date_column: str = "date",
        granularity: int = 1,
        unit: str = "hour",
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
    ) -> pl.DataFrame:
        """
        Fills missing timestamps in a Polars DataFrame with null values.

        Args:
            df: The Polars DataFrame containing the data.
            date_column: The name of the column containing the timestamps.
            granularity: The interval between timestamps (e.g., 1 for hourly).
            unit: The unit of time for the granularity (e.g., "hour", "minute").
            start: The lower bound for the timestamps (inclusive). If None, use the earliest date.
            end: The upper bound for the timestamps (inclusive). If None, use the latest date.

        Returns:
            A new Polars DataFrame with missing timestamps filled with null values.
        """

        # Dictionary for unit auto-correction (full name to abbreviation)
        unit_corrections = {
            "nanosecond": "ns",
            "microsecond": "us",
            "millisecond": "ms",
            "second": "s",
            "minute": "m",
            "hour": "h",
            "day": "d",
            "week": "w",
            "month": "mo",
            "quarter": "q",
            "year": "y",
        }

        # Create a range of timestamps, convert to DataFrame, and cast in one go
        df = (
            pl.datetime_range(
                start=df[date_column].min() if start is None else start,
                end=df[date_column].max() if end is None else end,
                interval=f"{granularity}{unit_corrections.get(unit.lower(), unit)}",
                closed="both",
                eager=True,
            )
            .cast(df[date_column].dtype)
            .to_frame(name=date_column)
        ).join(df, on=date_column, how="left")

        return df
    # def fill_timestamps(
    #     self,
    #     df: pl.DataFrame,
    #     date_column: str = "date",
    #     granularity: int = 1,
    #     unit: str = "hour",
    #     start: datetime.datetime | None = None,
    #     end: datetime.datetime | None = None,
    # ) -> pl.DataFrame:
    #     """
    #     Fills missing timestamps in a Polars DataFrame with null values.

    #     Args:
    #         df: The Polars DataFrame containing the data.
    #         date_column: The name of the column containing the timestamps.
    #         granularity: The interval between timestamps (e.g., 1 for hourly).
    #         unit: The unit of time for the granularity (e.g., "hour", "minute").
    #         start: The lower bound for the timestamps (inclusive). If None, use the earliest date.
    #         end: The upper bound for the timestamps (inclusive). If None, use the latest date.

    #     Returns:
    #         A new Polars DataFrame with missing timestamps filled with null values.
    #     """

    #     # Determine start and start bounds
    #     if start is None:
    #         start = df[date_column].min()
    #     if end is None:
    #         end = df[date_column].max()

    #     # Dictionary for unit auto-correction (full name to abbreviation)
    #     unit_corrections = {
    #         "nanosecond": "ns",
    #         "microsecond": "us",
    #         "millisecond": "ms",
    #         "second": "s",
    #         "minute": "m",
    #         "hour": "h",
    #         "calendar day": "d",
    #         "calendar week": "w",
    #         "calendar month": "mo",
    #         "calendar quarter": "q",
    #         "calendar year": "y",
    #     }

    #     # Auto-correct the unit if necessary
    #     unit = unit_corrections.get(unit.lower(), unit)

    #     # Create a range of timestamps with the specified granularity
    #     timestamps = pl.datetime_range(
    #         start=start,
    #         end=end,
    #         interval=f"{granularity}{unit}",
    #         closed="both",
    #         eager=True,
    #     ).to_frame(name=date_column)  # Convert to DataFrame with named column

    #     # Left join the original DataFrame with the timestamps to introduce missing rows
    #     df = timestamps.join(df, on=date_column, how="left")

    #     return df

    def split(self, df, used_columns, target_columns):
        """Splits a Polars DataFrame into sequences, excluding those with null values."""
        x, y = [], []

        for i in range(len(df) - self.seq_len - self.pred_len - self.offset + 1):
            # Extract sequences
            seq_x = df[i : i + self.seq_len][used_columns]
            seq_y = df[i + self.seq_len + self.offset : i + self.seq_len + self.pred_len + self.offset][target_columns]

            # Check for null values
            if sum(seq_x.null_count().to_numpy()[0])!=0 or sum(seq_y.null_count().to_numpy()[0])!=0:
                continue  # Skip sequences with null values

            # Convert to NumPy arrays
            seq_x = seq_x.to_numpy()
            seq_y = seq_y.to_numpy()

            x.append(seq_x)
            y.append(seq_y)

        return np.array(x), np.array(y)

class TimeSeriesTransformerDataset(TimeSeriesDataset):
    def __init__(self, source, target_columns, used_columns, scaler=None, info=True, ratio=None, indices=None,
                 ranges=None, try_parse_dates=True, low_memory=False, seq_len=96, pred_len=96, offset=None,
                 granularity=1, unit='hour', date_column='date',split_first=False, label_len=46,timeenc=0):
        self.label_len = label_len
        self.timeenc = timeenc
        super().__init__(
            source,
            target_columns,
            used_columns,
            scaler,
            info,
            ratio,
            indices,
            ranges,
            try_parse_dates,
            low_memory,
            seq_len,
            pred_len,
            offset, granularity, unit, date_column,split_first=split_first)

    def split(self, df, used_columns, target_columns):
        """Splits a Polars DataFrame into sequences, excluding those with null values."""
        df_stamp = df.select('date')
        df_stamp = df_stamp.with_column('date', pl.col('date').cast(pl.Date64))
        if self.timeenc == 0:
            df_stamp = (
                df_stamp
                .with_column('month', pl.date_format(pl.col('date'), "%m").cast(pl.UInt32))
                .with_column('day', pl.date_format(pl.col('date'), "%d").cast(pl.UInt32))
                .with_column('weekday', pl.date_format(pl.col('date'), "%w").cast(pl.UInt32))
                .with_column('hour', pl.date_format(pl.col('date'), "%H").cast(pl.UInt32))
                .drop('date')
            )
            data_stamp = df_stamp.to_numpy()
        elif self.timeenc == 1:
            data_stamp = time_features(pl.col('date'), freq=self.freq).to_numpy().transpose()

        x, y = [], []

        for i in range(len(df) - self.seq_len - self.pred_len + 1):
            # Extract sequences
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = df[s_begin:s_end]
            seq_y = df[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark

            seq_x = df[i : i + self.seq_len][used_columns]
            seq_y = df[i + self.seq_len + self.offset : i + self.seq_len + self.pred_len + self.offset][target_columns]

            # Check for null values
            if sum(seq_x.null_count().to_numpy()[0])!=0 or sum(seq_y.null_count().to_numpy()[0])!=0:
                continue  # Skip sequences with null values

            # Convert to NumPy arrays
            seq_x = seq_x.to_numpy()
            seq_y = seq_y.to_numpy()

            x.append(seq_x)
            y.append(seq_y)

        return np.array(x), np.array(y)

class MultipleTimeSeriesDataset(TimeSeriesDataset):
    def __init__(self, source, target_columns, used_columns, scaler=None, info=True, ratio=None, indices=None,
                 ranges=None, try_parse_dates=True, low_memory=False, seq_len=96, pred_len=96, offset=None,
                 granularity=1, unit='hour', date_column='date', segment_column='segment',split_first=False):
        self.segment_column = segment_column
        super().__init__(
            source,
            target_columns,
            used_columns,
            scaler,
            info,
            ratio,
            indices,
            ranges,
            try_parse_dates,
            low_memory,
            seq_len,
            pred_len,
            offset, granularity, unit, date_column,split_first=split_first)

    def preprocessing(self):
        # TODO: fix range by segmentation
        super(TimeSeriesDataset, self).preprocessing()
        self.df = self.fill_timestamps(
            df=self.df,
            date_column=self.date_column,
            granularity=self.granularity,
            unit=self.unit,
            segment_column=self.segment_column,
            use_global_range=False
        )
        self.df = self.df.sort(by=[self.segment_column, self.date_column])

    def split(self, df, used_columns, target_columns):
        x, y = [], []

        for segment_value in tqdm(df[self.segment_column].unique()):
            segment_df = df.filter(pl.col(self.segment_column) == segment_value)
            segment_x, segment_y = super().split(segment_df, used_columns, target_columns)
            x.extend(segment_x)
            y.extend(segment_y)

        return np.array(x), np.array(y)

    def fill_timestamps(
        self,
        df: pl.DataFrame,
        date_column: str = "date",
        segment_column: str = "station",
        granularity: int = 1,
        unit: str = "hour",
        use_global_range: bool = False,
    ) -> pl.DataFrame:
        """
        Fills missing timestamps for each station, optionally using a global min/max date range.

        Args:
            df: The Polars DataFrame containing the data.
            date_column: The name of the column containing the timestamps.
            segment_column: The name of the column containing the station IDs.
            granularity: The interval between timestamps (e.g., 1 for hourly).
            unit: The unit of time for the granularity (e.g., "hour", "minute").
            use_global_range: If True, use the overall min/max dates for filling.

        Returns:
            A new Polars DataFrame with missing timestamps filled with null values for each station.
        """
    #     return (
    #         df.lazy()
    #         .group_by(segment_column)
    #         .map_groups(
    #             lambda group: super().fill_timestamps(
    #                 group,
    #                 date_column=date_column,
    #                 granularity=granularity,
    #                 unit=unit,
    #                 start=df[date_column].min() if use_global_range else None,
    #                 end=df[date_column].max() if use_global_range else None,
    #             ).with_columns(pl.lit(group[segment_column][0]).alias(segment_column)),  # Fill Station column
    #             schema=df.schema,
    #         )
    #         .collect()
    #     )
        filled_dfs = []
        for segment_value in tqdm(df[self.segment_column].unique()):
            segment_df = df.filter(pl.col(self.segment_column) == segment_value)
            filled_segment_df = super().fill_timestamps(segment_df,
                                                        date_column,
                                                        granularity,
                                                        unit,
                                                        df[date_column].min() if use_global_range else None,
                                                        df[date_column].max() if use_global_range else None)
            # Fill the segment_column with the corresponding value
            filled_segment_df = filled_segment_df.with_columns(
                pl.lit(segment_value).alias(self.segment_column)
            )
            filled_dfs.append(filled_segment_df)

        return pl.concat(filled_dfs)

class ETTh1(TimeSeriesDataset):
    def __init__(self, seq_len, pred_len):
        indices = ([
            (0, 12 * 30 * 24),
            (12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24),
            (12 * 30 * 24 + 4 * 30 * 24 - seq_len, 12 * 30 * 24 + 8 * 30 * 24)]
        )

        super().__init__(
            source="/content/DeLUR/datasets/ETDataset/ETTh1.csv",
            target_columns=None, # None to get all columns
            used_columns=None, # None to get all columns
            scaler='standard',
            info=True,
            ratio=None,
            indices=indices,
            ranges=None,
            try_parse_dates=True,
            low_memory=True,
            seq_len=seq_len,
            pred_len=pred_len,
            offset=None,
            granularity=1,
            unit='hour',
            date_column='date'
        )

class ArangedRainfall(MultipleTimeSeriesDataset):
    def __init__(self, seq_len, pred_len):
        super().__init__(
            source="/content/DeLUR/datasets/Aranged_rainfall_processed_final.csv",
            target_columns=['Value'], # None to get all columns
            used_columns=['Station', 'Value'], # None to get all columns
            segment_column='Station',
            date_column='Date',
            scaler=None,
            info=True,
            ratio=(0.7, 0.1, 0.2),
            indices=None,
            ranges=None,
            try_parse_dates=True,
            low_memory=True,
            seq_len=seq_len,
            pred_len=pred_len,
            offset=None,
            granularity=1,
            unit='day',
            split_first=True
        )
