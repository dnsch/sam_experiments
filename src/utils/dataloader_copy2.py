import os
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# CIFAR10 datasets

import torchvision
import torchvision.transforms as transforms

from pathlib import Path
import random


from typing import Optional, Iterator, Tuple, Dict, List

# Statsforecast Dataloader

# from code.src.utils.paths import get_samformer_dataset_path
from src.utils.model_utils import statsforecast_to_tensor

# For pandas offset parsing (matching original Autoformer)

from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

SCRIPT_DIR = Path(__file__).resolve().parent
from src.utils.paths import get_samformer_dataset_path


# Taken from:

# https://github.com/romilbert/samformer/blob/main/samformer_pytorch/run_demo.py

# https://github.com/romilbert/samformer/blob/main/samformer_pytorch/samformer/utils/dataset.py

# slightly modified


class LabeledDataset(Dataset):
    def __init__(self, x, y, x_mark=None, y_mark=None):
        """
        Converts numpy data to a torch dataset
        Args:
            x (np.array): data matrix
            y (np.array): class labels
            x_mark (np.array, optional): time features for x
            y_mark (np.array, optional): time features for y
        """
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        # Extra data handling needed for Autoformer, TSMixerExt
        self.x_mark = torch.FloatTensor(x_mark) if x_mark is not None else None
        self.y_mark = torch.FloatTensor(y_mark) if y_mark is not None else None
        self.has_time_features = x_mark is not None

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        examples = self.x[idx]
        labels = self.y[idx]

        # Extra data handling needed for Autoformer, TSMixerExt
        if self.has_time_features:
            x_time_features = self.x_mark[idx]
            y_time_features = self.y_mark[idx]
            return examples, labels, x_time_features, y_time_features
        else:
            return examples, labels


# Time Features

# Based on: https://github.com/thuml/Autoformer/blob/main/utils/timefeatures.py

# needed for autoformer


class TimeFeature:
    """Base class for time features."""

    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second of minute encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Day of week encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5  # Note: 52.0 to match original


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq="h"):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


def extract_time_features_raw(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    """
    Extract raw (non-normalized) time features for timeenc=0 mode.

    For hourly data: returns [month, day, weekday, hour]
    For minute data: returns [month, day, weekday, hour, minute]
    where minute is binned into 15-minute intervals.

    Args:
        dates: DatetimeIndex
        freq: Frequency string (e.g., 'h' for hourly, 't' for minutely)

    Returns:
        Array of shape [len(dates), num_features] with raw time features
    """
    df_stamp = pd.DataFrame({"date": dates})
    df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
    df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
    df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
    df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)

    # Determine if we need minute-level features
    # 't' = minutely in pandas, 's' = secondly, also handle patterns like '15min'
    freq_lower = freq.lower()
    needs_minute = freq_lower in ["t", "s"] or "min" in freq_lower

    if needs_minute:
        df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute)
        # Bin minutes into 15-minute intervals
        df_stamp["minute"] = df_stamp["minute"].map(lambda x: x // 15)

    return df_stamp.drop(["date"], axis=1).values


def extract_time_features(dates: pd.DatetimeIndex, freq: str = "h", timeenc: int = 1) -> np.ndarray:
    """
    Extract time features from datetime index.
    Matches the original Autoformer data_loader.py implementation.

    Args:
        dates: DatetimeIndex
        freq: Frequency string (e.g., 'h' for hourly, 't' for minutely)
        timeenc: Time encoding mode:
            0 = raw integer features (month, day, weekday, hour, [minute])
            1 = normalized features between [-0.5, 0.5]

    Returns:
        Array of shape [len(dates), num_features] with time features
    """
    if timeenc == 0:
        return extract_time_features_raw(dates, freq)
    else:  # timeenc == 1
        # time_features returns [num_features, len(dates)], need to transpose
        return time_features(dates, freq).transpose(1, 0)


def construct_sliding_window_data(
    data,
    seq_len,
    pred_len,
    time_increment=1,
    time_features=None,
    label_len=0,  # NEW PARAMETER
):
    """
    Construct sliding window data with optional time features.

    Args:
        data: Data array
        seq_len: Input sequence length
        pred_len: Prediction length
        time_increment: Step size for sliding window
        time_features: Optional time features array (same length as data)
        label_len: Decoder start token length (for Informer-style models)

    Returns:
        If label_len > 0: x, y includes label_len overlap
        If time_features is None: (x, y)
        If time_features is provided: (x, y, x_mark, y_mark)
    """
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    range_ = np.arange(0, n_samples, time_increment)
    x, y = list(), list()

    if time_features is not None:
        x_mark, y_mark = list(), list()

        for i in range_:
            x.append(data[i : (i + seq_len)].T)

            # y includes label_len overlap + pred_len future
            if label_len > 0:
                y_start = i + seq_len - label_len
                y_end = i + seq_len + pred_len
                y.append(data[y_start:y_end].T)
                y_mark.append(time_features[y_start:y_end])
            else:
                y.append(data[(i + seq_len) : (i + seq_len + pred_len)].T)
                y_mark.append(time_features[(i + seq_len) : (i + seq_len + pred_len)])

            x_mark.append(time_features[i : (i + seq_len)])

        return np.array(x), np.array(y), np.array(x_mark), np.array(y_mark)
    else:
        for i in range_:
            x.append(data[i : (i + seq_len)].T)

            if label_len > 0:
                y_start = i + seq_len - label_len
                y_end = i + seq_len + pred_len
                y.append(data[y_start:y_end].T)
            else:
                y.append(data[(i + seq_len) : (i + seq_len + pred_len)].T)

        return np.array(x), np.array(y)


# Extra data handling needed for TSMixerExt
class TSMixerExtDataset(Dataset):
    """
    Dataset for TSMixerExt that provides all required inputs:
    - x_hist: Historical time series
    - x_extra_hist: Extra historical features (e.g., time features)
    - x_extra_future: Future known features (e.g., time features for prediction horizon)
    - x_static: Static features (e.g., channel embeddings or zeros)
    - y: Target values

    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_mark: Optional[np.ndarray] = None,
        y_mark: Optional[np.ndarray] = None,
        static_features: Optional[np.ndarray] = None,
        num_static_features: int = 1,
    ):
        """
        Args:
            x: Historical data (n_samples, num_channels, seq_len)
            y: Target data (n_samples, num_channels, horizon)
            x_mark: Time features for historical period (n_samples, seq_len, num_time_features)
            y_mark: Time features for prediction period (n_samples, horizon, num_time_features)
            static_features: Static features (n_samples, static_channels) or None for zeros
            num_static_features: Number of static features if generating zeros
        """
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

        n_samples = x.shape[0]

        # Handle extra historical features (use time features or zeros)
        if x_mark is not None:
            self.x_extra_hist = torch.FloatTensor(x_mark)
        else:
            # Create placeholder zeros: (n_samples, seq_len, 1)
            seq_len = x.shape[1] if x.ndim == 2 else x.shape[2]
            self.x_extra_hist = torch.zeros(n_samples, seq_len, 1)

        # Handle extra future features (use time features or zeros)
        if y_mark is not None:
            self.x_extra_future = torch.FloatTensor(y_mark)
        else:
            # Create placeholder zeros: (n_samples, horizon, 1)
            horizon = y.shape[1] if y.ndim == 2 else y.shape[2]
            self.x_extra_future = torch.zeros(n_samples, horizon, 1)

        # Handle static features
        if static_features is not None:
            self.x_static = torch.FloatTensor(static_features)
        else:
            # Create placeholder zeros: (n_samples, num_static_features)
            self.x_static = torch.zeros(n_samples, num_static_features)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Transpose x and y from (num_channels, seq_len) to (seq_len, num_channels)
        x_hist = self.x[idx].T if self.x[idx].dim() == 2 else self.x[idx]
        y = self.y[idx].T if self.y[idx].dim() == 2 else self.y[idx]

        return (
            x_hist,  # (seq_len, num_channels)
            self.x_extra_hist[idx],  # (seq_len, extra_channels)
            self.x_extra_future[idx],  # (horizon, extra_channels)
            self.x_static[idx],  # (static_channels,)
            y,  # (horizon, num_channels)
        )


class SamformerDataloader:
    def __init__(
        self,
        dataset=None,
        seq_len=96,
        pred_len=96,
        seed=1,
        time_increment=1,
        train_ratio=0.7,
        val_ratio=0.2,
        batch_size=32,
        sequential_comparison=False,
        # New parameters for Autoformer
        use_time_features=False,  # New parameter
        freq="h",  # New parameter for frequency
        timeenc=1,  # New parameter: 0=raw integers, 1=normalized (matches Autoformer)
        embed="timeF",  # New parameter: timeF uses timeenc=1, others use timeenc=0
        # New parameters for TSMixerExt
        model_type="standard",  # "standard" or "tsmixer_ext"
        num_static_features=1,  # Number of static features for TSMixerExt
    ):
        # file_path = (
        #     SCRIPT_DIR.parents[2] / "data" / "samformer_datasets" / f"{dataset}.csv"
        # )
        file_path = get_samformer_dataset_path(dataset=dataset)
        df_raw = pd.read_csv(file_path, index_col=0)

        # Convert index to datetime if not already
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            df_raw.index = pd.to_datetime(df_raw.index)

        n = len(df_raw)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.time_increment = time_increment
        self.seed = seed
        self.batch_size = batch_size
        # Autoformer init
        self.use_time_features = use_time_features
        self.freq = freq
        # Determine timeenc based on embed parameter (matching Autoformer's data_factory.py)
        self.timeenc = 0 if embed != "timeF" else 1
        # Allow explicit override
        if timeenc is not None and embed == "timeF":
            self.timeenc = timeenc

        # TSMixerExt init
        self.model_type = model_type
        self.num_static_features = num_static_features

        # Force time features for TSMixerExt (used as extra channels)
        if model_type == "tsmixer_ext":
            self.use_time_features = True

        if dataset.startswith("ETTm"):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif dataset.startswith("ETTh"):
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * train_ratio)
            val_end = n - int(n * val_ratio)
            test_end = n

        self.train_end = train_end
        self.val_end = val_end
        self.test_end = test_end

        # Store datetime index for time feature extraction
        self.df_raw = df_raw

        self.dataloader = self.create_dataloader(
            df_raw,
            train_end,
            val_end,
            test_end,
            sequential_comparison=sequential_comparison,
        )

    def _extract_time_features_for_split(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract time features for a dataframe split if enabled."""
        if not self.use_time_features:
            return None
        return extract_time_features(df.index, self.freq, self.timeenc)

    def create_dataloader(
        self,
        df_raw,
        train_end,
        val_end,
        test_end,
        sequential_comparison=False,
    ):
        if sequential_comparison:
            # Get multiple train/val/test splits
            train_val_test_splits = self.construct_multiple_train_test_split_data(
                df_raw,
                splits_num=10,
                train_end=train_end,
                val_end=val_end,
                seed=self.seed,
            )

            # Process each split
            dataloaders_list = []
            scalers_list = []

            for split_data in train_val_test_splits:
                if self.use_time_features:
                    train_data, val_data, test_data, train_time, val_time, test_time = split_data
                else:
                    train_data, val_data, test_data = split_data
                    train_time = val_time = test_time = None

                # Create scaler for this split and fit on training data
                scaler = StandardScaler()
                scaler.fit(train_data.values)

                # Transform data
                train_arr = scaler.transform(train_data.values)
                val_arr = scaler.transform(val_data.values)
                test_arr = scaler.transform(test_data.values)

                # Apply sliding window
                if self.use_time_features:
                    x_train, y_train, x_train_mark, y_train_mark = construct_sliding_window_data(
                        train_arr,
                        self.seq_len,
                        self.pred_len,
                        self.time_increment,
                        train_time,
                    )
                    x_val, y_val, x_val_mark, y_val_mark = construct_sliding_window_data(
                        val_arr,
                        self.seq_len,
                        self.pred_len,
                        self.time_increment,
                        val_time,
                    )
                    x_test, y_test, x_test_mark, y_test_mark = construct_sliding_window_data(
                        test_arr,
                        self.seq_len,
                        self.pred_len,
                        self.time_increment,
                        test_time,
                    )
                else:
                    x_train, y_train = construct_sliding_window_data(
                        train_arr, self.seq_len, self.pred_len, self.time_increment
                    )
                    x_val, y_val = construct_sliding_window_data(
                        val_arr, self.seq_len, self.pred_len, self.time_increment
                    )
                    x_test, y_test = construct_sliding_window_data(
                        test_arr, self.seq_len, self.pred_len, self.time_increment
                    )
                    x_train_mark = y_train_mark = None
                    x_val_mark = y_val_mark = None
                    x_test_mark = y_test_mark = None

                # Create datasets and dataloaders
                #
                # special handling needed for tsmixerex
                if self.model_type == "tsmixer_ext":
                    train_dataset = TSMixerExtDataset(
                        x_train,
                        y_train,
                        x_mark=x_train_mark,
                        y_mark=y_train_mark,
                        num_static_features=self.num_static_features,
                    )
                    val_dataset = TSMixerExtDataset(
                        x_val,
                        y_val,
                        x_mark=x_val_mark,
                        y_mark=y_val_mark,
                        num_static_features=self.num_static_features,
                    )
                    test_dataset = TSMixerExtDataset(
                        x_test,
                        y_test,
                        x_mark=x_test_mark,
                        y_mark=y_test_mark,
                        num_static_features=self.num_static_features,
                    )
                else:
                    train_dataset = LabeledDataset(x_train, y_train, x_train_mark, y_train_mark)
                    val_dataset = LabeledDataset(x_val, y_val, x_val_mark, y_val_mark)
                    test_dataset = LabeledDataset(x_test, y_test, x_test_mark, y_test_mark)

                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=True
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=self.batch_size, shuffle=False
                )

                split_dataloader = {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "test_loader": test_loader,
                }

                dataloaders_list.append(split_dataloader)
                scalers_list.append(scaler)

            # Store the first scaler for compatibility with existing methods
            self.scaler = scalers_list[0]
            self.scalers_list = scalers_list

            return dataloaders_list

        else:
            # Regular processing
            train_df = df_raw[:train_end]
            val_df = df_raw[train_end - self.seq_len : val_end]
            test_df = df_raw[val_end - self.seq_len : test_end]

            # Extract time features if enabled
            train_time = self._extract_time_features_for_split(train_df)
            val_time = self._extract_time_features_for_split(val_df)
            test_time = self._extract_time_features_for_split(test_df)

            # standardize by training set
            self.scaler = StandardScaler()
            self.scaler.fit(train_df.values)
            self.train_arr = self.scaler.transform(train_df.values)
            self.val_arr = self.scaler.transform(val_df.values)
            self.test_arr = self.scaler.transform(test_df.values)

            # apply sliding window
            if self.use_time_features:
                x_train, y_train, x_train_mark, y_train_mark = construct_sliding_window_data(
                    self.train_arr,
                    self.seq_len,
                    self.pred_len,
                    self.time_increment,
                    train_time,
                )
                x_val, y_val, x_val_mark, y_val_mark = construct_sliding_window_data(
                    self.val_arr,
                    self.seq_len,
                    self.pred_len,
                    self.time_increment,
                    val_time,
                )
                x_test, y_test, x_test_mark, y_test_mark = construct_sliding_window_data(
                    self.test_arr,
                    self.seq_len,
                    self.pred_len,
                    self.time_increment,
                    test_time,
                )
            else:
                x_train, y_train = construct_sliding_window_data(
                    self.train_arr, self.seq_len, self.pred_len, self.time_increment
                )
                x_val, y_val = construct_sliding_window_data(
                    self.val_arr, self.seq_len, self.pred_len, self.time_increment
                )
                x_test, y_test = construct_sliding_window_data(
                    self.test_arr, self.seq_len, self.pred_len, self.time_increment
                )
                x_train_mark = y_train_mark = None
                x_val_mark = y_val_mark = None
                x_test_mark = y_test_mark = None

            # Create datasets and dataloaders
            #
            # special handling needed for tsmixerext
            if self.model_type == "tsmixer_ext":
                train_dataset = TSMixerExtDataset(
                    x_train,
                    y_train,
                    x_mark=x_train_mark,
                    y_mark=y_train_mark,
                    num_static_features=self.num_static_features,
                )
                val_dataset = TSMixerExtDataset(
                    x_val,
                    y_val,
                    x_mark=x_val_mark,
                    y_mark=y_val_mark,
                    num_static_features=self.num_static_features,
                )
                test_dataset = TSMixerExtDataset(
                    x_test,
                    y_test,
                    x_mark=x_test_mark,
                    y_mark=y_test_mark,
                    num_static_features=self.num_static_features,
                )
            else:
                train_dataset = LabeledDataset(x_train, y_train, x_train_mark, y_train_mark)
                val_dataset = LabeledDataset(x_val, y_val, x_val_mark, y_val_mark)
                test_dataset = LabeledDataset(x_test, y_test, x_test_mark, y_test_mark)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )

            dataloader = {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
            }
            return dataloader

    def get_dataloader(self):
        return self.dataloader

    def get_scaler(self):
        return self.scaler

    def get_scaler_list(self):
        return self.scalers_list

    def get_test_sliding_loader(
        self,
        stride: Optional[int] = None,
        batch_size: int = 1,
        flatten_y: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Build sliding-window loader over the (scaled) test segment with configurable stride.
        Align stride with Darts rolling windows; default stride = seq_len for non-overlapping.
        """
        stride = stride if stride is not None else self.seq_len

        # Extract time features for test data if enabled
        test_df = self.df_raw[self.val_end - self.seq_len : self.test_end]
        test_time = (
            self._extract_time_features_for_split(test_df) if self.use_time_features else None
        )

        # Rebuild x/y with the chosen stride from scaled test segment
        if self.use_time_features:
            x_test_sw, y_test_sw, x_test_mark, y_test_mark = construct_sliding_window_data(
                self.test_arr, self.seq_len, self.pred_len, stride, test_time
            )
        else:
            x_test_sw, y_test_sw = construct_sliding_window_data(
                self.test_arr, self.seq_len, self.pred_len, stride
            )
            x_test_mark = y_test_mark = None

        if flatten_y:
            y_test_sw = y_test_sw.reshape((y_test_sw.shape[0], -1))

        ds = LabeledDataset(x_test_sw, y_test_sw, x_test_mark, y_test_mark)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    def rolling_test_windows(
        self,
        step: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Optional generator yielding test windows (scaled) with given step.
        Returns: (x, y) if no time features, (x, y, x_mark, y_mark) if time features enabled
        """
        step = step if step is not None else self.seq_len

        # Extract time features for test data if enabled
        test_df = self.df_raw[self.val_end - self.seq_len : self.test_end]
        test_time = (
            self._extract_time_features_for_split(test_df) if self.use_time_features else None
        )

        if self.use_time_features:
            x, y, x_mark, y_mark = construct_sliding_window_data(
                self.test_arr, self.seq_len, self.pred_len, step, test_time
            )
            for i in range(x.shape[0]):
                yield (
                    torch.FloatTensor(x[i]),
                    torch.FloatTensor(y[i]),
                    torch.FloatTensor(x_mark[i]),
                    torch.FloatTensor(y_mark[i]),
                )
        else:
            x, y = construct_sliding_window_data(self.test_arr, self.seq_len, self.pred_len, step)
            for i in range(x.shape[0]):
                yield torch.FloatTensor(x[i]), torch.FloatTensor(y[i])

    def construct_multiple_train_test_split_data(
        self,
        data,
        splits_num=1,
        train_end=None,
        val_end=None,
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)

        splits = []

        # Check if we have enough data for the requested number of splits
        min_data_per_split = self.seq_len + self.pred_len
        available_data_after_first = len(data) - (val_end + self.pred_len)
        max_possible_additional_splits = available_data_after_first // min_data_per_split

        if splits_num - 1 > max_possible_additional_splits:
            raise ValueError(
                f"Cannot create {splits_num} splits. Maximum possible splits: {max_possible_additional_splits + 1}"
            )

        # First split
        first_train_data = data[:train_end]
        first_val_data = data[train_end - self.seq_len : val_end]
        first_test_data = data[val_end - self.seq_len : val_end + self.pred_len]

        if self.use_time_features:
            first_train_time = self._extract_time_features_for_split(first_train_data)
            first_val_time = self._extract_time_features_for_split(first_val_data)
            first_test_time = self._extract_time_features_for_split(first_test_data)
            first_data_split = [
                first_train_data,
                first_val_data,
                first_test_data,
                first_train_time,
                first_val_time,
                first_test_time,
            ]
        else:
            first_data_split = [first_train_data, first_val_data, first_test_data]

        splits.append(first_data_split)

        # Pre-generate evenly distributed random val_ends for remaining splits
        if splits_num > 1:
            start_point = val_end + self.pred_len
            end_point = len(data) - self.pred_len

            total_range = end_point - start_point
            segment_size = total_range // (splits_num - 1)

            random_val_ends = []
            for i in range(splits_num - 1):
                segment_start = start_point + i * segment_size
                segment_end = min(start_point + (i + 1) * segment_size, end_point)

                if segment_end > segment_start:
                    random_val_end = random.randint(segment_start, segment_end)
                    random_val_ends.append(random_val_end)

            # Create splits using the pre-generated val_ends
            for random_val_end in random_val_ends:
                train_ratio = 0.8
                train_end = int(random_val_end * train_ratio)
                train_data = data[:train_end]
                val_data = data[train_end - self.seq_len : random_val_end]
                test_data = data[random_val_end - self.seq_len : random_val_end + self.pred_len]

                if self.use_time_features:
                    train_time = self._extract_time_features_for_split(train_data)
                    val_time = self._extract_time_features_for_split(val_data)
                    test_time = self._extract_time_features_for_split(test_data)
                    splits.append(
                        [
                            train_data,
                            val_data,
                            test_data,
                            train_time,
                            val_time,
                            test_time,
                        ]
                    )
                else:
                    splits.append([train_data, val_data, test_data])

        return splits


class CIFAR10Dataloader:
    def __init__(
        self,
        args,
        raw_data=False,
        data_split=1,
        split_idx=0,
        trainloader_path="",
        testloader_path="",
    ):
        """
        Based on: https://github.com/tomgoldstein/loss-landscape/blob/master/dataloader.py

        Setup dataloader. The data is not randomly cropped as in training because of
        we want to esimate the loss value with a fixed dataset.

        Args:
            raw_data: raw images, no data preprocessing
            data_split: the number of splits for the training dataloader
            split_idx: the index for the split of the dataloader, starting at 0

        Returns:
            dataloader array consisting of trainloader, testloader

        """
        # use specific dataloaders
        if trainloader_path and testloader_path:
            assert os.path.exists(trainloader_path), "trainloader does not exist"
            assert os.path.exists(testloader_path), "testloader does not exist"
            train_loader = torch.load(trainloader_path)
            test_loader = torch.load(testloader_path)
            return train_loader, test_loader

        assert (
            split_idx < data_split
        ), "the index of data partition should be smaller than the total number of split"

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        data_folder = "../data/cifar10_datasets"
        if raw_data:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        trainset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=True, transform=transform
        )
        # If data_split>1, then randomly select a subset of the data. E.g., if datasplit=3, then
        # randomly choose 1/3 of the data.
        if data_split > 1:
            indices = torch.tensor(np.arange(len(trainset)))
            data_num = len(trainset) // data_split  # the number of data in a chunk of the split

            # Randomly sample indices. Use seed=0 in the generator to make this reproducible
            state = np.random.get_state()
            np.random.seed(0)
            indices = np.random.choice(indices, data_num, replace=False)
            np.random.set_state(state)

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                shuffle=False,
                num_workers=args.threads,
            )
        else:
            kwargs = {"num_workers": 2, "pin_memory": True}
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=False, **kwargs
            )
        testset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=False, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads
        )

        self.dataloader = {
            "train_loader": train_loader,
            "test_loader": test_loader,
        }

    def get_dataloader(self):
        return self.dataloader


class StatsforecastDataloader:
    def __init__(
        self,
        dataset,
        args,
        logger,
        train_ratio=0.7,
        val_ratio=0.2,
        apply_scaling=False,
        merge_train_val=False,  # New parameter
    ):
        file_path = get_samformer_dataset_path(dataset=dataset)
        df_raw = pd.read_csv(file_path, index_col=0)

        # Convert index to datetime if it's not already
        df_raw.index = pd.to_datetime(df_raw.index)

        n = len(df_raw)
        self.data_raw = df_raw
        self.seq_len = args.seq_len
        self.pred_len = args.horizon
        self.args = args
        self.apply_scaling = apply_scaling
        self.merge_train_val = merge_train_val

        # Use same split logic as SamformerDataloader
        if dataset.startswith("ETTm"):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif dataset.startswith("ETTh"):
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * train_ratio)
            val_end = n - int(n * val_ratio)
            test_end = n

        # Store split indices
        self.train_end = train_end
        self.val_end = val_end
        self.test_end = test_end

        # Try to create the same train/val/test sets as with the other
        # dataloaders to ensure comparability
        train_df = df_raw[:train_end]
        val_df = df_raw[train_end - self.seq_len : val_end]
        test_df = df_raw[val_end - self.seq_len : test_end]

        # for the arima model, the data should be sequential,
        # I think a validation dataset does not make sense here.
        if merge_train_val:
            # Merge train and val into single train split
            train_df_wide = df_raw[:val_end]  # Includes original train + val
            test_df_wide = df_raw[val_end - self.seq_len : test_end]
            val_df_wide = None  # No separate validation set

            # The other dataloaders are for deep learning architectures that
            # use batching. The model is trained on (shuffled) batches of
            # training data, validated on (shuffled) batches of validation data
            # and then tested on (sequential) batches of test data.
            # The batches in the test data are not connected however, hence for
            # each test batch the model only uses the training data, the
            # validation data and the input section of the current test batch.
            #
            # For the arima model, however, the input data used for training
            # has to be sequential and, due to the autoregressive nature of the
            # model, it can only predict the next x timesteps that come after
            # the training data. Because of this, we can only use the training
            # and validation data, and the input section of the first test
            # batch for fitting the model in order to predict the output
            # section of the first test batch.
            #
            # As a consequence, we can only use the first test batch to ensure
            # a fair comparison (that is, that they use the same input data)
            # between the two models.
            #
            # This approach is handled by the sequential_comparison parameter:

            # TODO: add this as an args parameter
            sequential_comparison = True
            if sequential_comparison:
                # Since we do not give the model input after training and it
                # predicts the next steps of the training input, we include
                # the seq_len data in the training set
                # train_df_wide = df_raw[:val_end]
                # test_df_wide = df_raw[val_end - self.seq_len : test_end]
                #
                # We will create multiple random train/test splits for a better
                # comparison

                # train_df_wide = df_raw[:val_end]
                # # We don't subtract seq_len here as in the samformer model,
                # # bc. we don't need that as input in the Statsforecast models
                # test_df_wide = df_raw[val_end:test_end]

                # Add the scalers s.t. we can have a better comparison with the
                # other results/forecasts
                # TODO: change this to parameter
                scale_data = True
                if scale_data:
                    self.dataloader = self.construct_multiple_train_test_split_data(
                        df_raw,
                        self.seq_len,
                        self.pred_len,
                        splits_num=10,
                        scale_data=scale_data,
                        train_end=train_end,
                        val_end=val_end,
                        seed=self.args.seed,
                    )
                else:
                    self.dataloader = self.construct_multiple_train_test_split_data(
                        df_raw,
                        self.seq_len,
                        self.pred_len,
                        splits_num=10,
                        val_end=val_end,
                        seed=self.args.seed,
                    )

            #     pass

        # But let's keep this functionality
        else:
            # Keep separate train, val, test splits
            train_df_wide = df_raw[:train_end]
            val_df_wide = df_raw[train_end - self.seq_len : val_end]
            test_df_wide = df_raw[val_end - self.seq_len : test_end]

            # Convert to long format
            self.train_df = self._wide_to_long(train_df_wide)
            self.val_df = self._wide_to_long(val_df_wide) if val_df_wide is not None else None
            self.test_df = self._wide_to_long(test_df_wide)

            # Create full dataset
            dfs_to_concat = [self.train_df]
            if self.val_df is not None:
                dfs_to_concat.append(self.val_df)
            dfs_to_concat.append(self.test_df)
            self.full_df = pd.concat(dfs_to_concat, ignore_index=True)

            # Create dataloader dictionary (similar to SamformerDataloader)
            self.dataloader = {
                "train_loader": self.train_df,
                "val_loader": self.val_df,
                "test_loader": self.test_df,
            }

    def get_dataloader(self) -> Dict[str, Optional[pd.DataFrame]]:
        """Get dataloader dictionary containing train, val, and test DataFrames."""
        return self.dataloader

    def _wide_to_long(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        """Convert wide format to long format expected by statsforecast."""
        df_long = df_wide.reset_index().melt(id_vars=["date"], var_name="unique_id", value_name="y")
        df_long = df_long.rename(columns={"date": "ds"})
        # Ensure proper column order
        df_long = df_long[["unique_id", "ds", "y"]]
        return df_long.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def get_train_data(self) -> pd.DataFrame:
        """Get training data in statsforecast format."""
        return self.train_df.copy()

    def get_val_data(self) -> Optional[pd.DataFrame]:
        """Get validation data in statsforecast format. Returns None if merged."""
        if self.val_df is None:
            if self.merge_train_val:
                print("Warning: No validation data available because merge_train_val=True")
            return None
        return self.val_df.copy()

    def get_test_data(self) -> pd.DataFrame:
        """Get test data in statsforecast format."""
        return self.test_df.copy()

    def get_full_data(self) -> pd.DataFrame:
        """Get full dataset in statsforecast format."""
        return self.full_df.copy()

    def get_train_val_data(self) -> pd.DataFrame:
        """Get combined train+val data for final model training."""
        if self.merge_train_val:
            # Already merged in train_df
            return self.train_df.copy()
        else:
            # Concatenate separate train and val
            return pd.concat([self.train_df, self.val_df], ignore_index=True)

    def get_scaler(self):
        """Get the fitted scaler if scaling was applied."""
        return self.scaler

    def get_scaler_list(self):
        return self.scalers_list

    def apply_scaler_to_dataframe(self, df, scaler, value_col="y"):
        """
        Apply scaler transformation and return in long format.
        """
        # Pivot to wide format
        df_wide = df.pivot(index="ds", columns="unique_id", values=value_col)

        # Transform
        values = df_wide.values
        transformed_values = scaler.transform(values)

        # Put transformed values back
        df_wide_transformed = pd.DataFrame(
            transformed_values, index=df_wide.index, columns=df_wide.columns
        )

        # Convert back to long format
        df_long = df_wide_transformed.reset_index().melt(
            id_vars="ds", var_name="unique_id", value_name=value_col
        )

        # Sort to match original order
        df_long = df_long.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        return df_long

    def construct_multiple_train_test_split_data(
        self,
        data,
        seq_len,
        pred_len,
        splits_num=1,
        scale_data=False,
        train_end=None,
        val_end=None,
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)

        splits = []

        # Check if we have enough data for the requested number of splits
        min_data_per_split = seq_len + pred_len
        available_data_after_first = len(data) - (val_end + pred_len)
        max_possible_additional_splits = available_data_after_first // min_data_per_split

        if splits_num - 1 > max_possible_additional_splits:
            raise ValueError(
                f"Cannot create {splits_num} splits. Maximum possible splits: {max_possible_additional_splits + 1}"
            )

        first_train_data = self._wide_to_long(data[:val_end])
        first_test_data = self._wide_to_long(data[val_end : val_end + self.pred_len])
        # Scale the data and return the scaler if provided
        if scale_data:
            self.scalers_list = []
            scaler_data = self._wide_to_long(data[:train_end])
            # TODO: that doesn't look so nice, maybe can be written more
            # elegantly
            scaler_data = torch.permute(statsforecast_to_tensor(scaler_data, "y", False), (1, 0))
            scaler = StandardScaler()
            scaler.fit(scaler_data)
            self.scalers_list.append(scaler)
            # Transform data
            first_train_data = self.apply_scaler_to_dataframe(
                first_train_data, scaler, value_col="y"
            )
            first_test_data = self.apply_scaler_to_dataframe(first_test_data, scaler, value_col="y")

        first_data_split = [first_train_data, first_test_data]
        splits.append(first_data_split)

        # Pre-generate evenly distributed random val_ends for remaining splits
        if splits_num > 1:
            start_point = val_end + self.pred_len  # Earliest possible start for next split
            end_point = len(data) - pred_len  # Latest possible end

            # Divide the available range into roughly equal segments
            total_range = end_point - start_point
            segment_size = total_range // (splits_num - 1)

            random_val_ends = []
            for i in range(splits_num - 1):
                segment_start = start_point + i * segment_size
                segment_end = min(start_point + (i + 1) * segment_size, end_point)

                # Ensure we have at least some range in each segment
                if segment_end > segment_start:
                    random_val_end = random.randint(segment_start, segment_end)
                    random_val_ends.append(random_val_end)

            # Create splits using the pre-generated val_ends
            # val_ends mark the end of the training set
            for random_val_end in random_val_ends:
                if scale_data:
                    # TODO: hard coded train ratio, maybe change this to a parameter
                    # we take 80% of the training set for training the model
                    # the other 20% of that set for validating it and selecting the
                    # best one.
                    train_ratio = 0.8
                    train_end = int(random_val_end * train_ratio)
                    scaler_data = self._wide_to_long(data[:train_end])
                    scaler_data = torch.permute(
                        statsforecast_to_tensor(scaler_data, "y", False), (1, 0)
                    )
                    scaler = StandardScaler()
                    scaler.fit(scaler_data)
                    self.scalers_list.append(scaler)

                train_data = self._wide_to_long(data[:random_val_end])
                test_data = self._wide_to_long(data[random_val_end : random_val_end + pred_len])

                if scale_data:
                    # Transform data
                    train_data = self.apply_scaler_to_dataframe(train_data, scaler, value_col="y")
                    test_data = self.apply_scaler_to_dataframe(test_data, scaler, value_col="y")
                splits.append([train_data, test_data])
        if scale_data:
            self.scaler = self.scalers_list[0]

        return splits
