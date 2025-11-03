import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# CIFAR10 datasets
import torchvision
import torchvision.transforms as transforms

# DartsDataloader
from darts import TimeSeries

from pathlib import Path
import random


from typing import Optional, Iterator, Tuple, Dict
import pdb

SCRIPT_DIR = Path(__file__).resolve().parent


# Taken from:
# https://github.com/romilbert/samformer/blob/main/samformer_pytorch/run_demo.py
# https://github.com/romilbert/samformer/blob/main/samformer_pytorch/samformer/utils/dataset.py
# slightly modified
class LabeledDataset(Dataset):
    def __init__(self, x, y):
        """
        Converts numpy data to a torch dataset
        Args:
            x (np.array): data matrix
            y (np.array): class labels
        """
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def transform(self, x):
        return torch.FloatTensor(x)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        examples = self.x[idx]
        labels = self.y[idx]
        return examples, labels


def construct_sliding_window_data(data, seq_len, pred_len, time_increment=1):
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    range_ = np.arange(0, n_samples, time_increment)
    x, y = list(), list()
    for i in range_:
        x.append(data[i : (i + seq_len)].T)
        y.append(data[(i + seq_len) : (i + seq_len + pred_len)].T)
    return np.array(x), np.array(y)


# TODO: rename this to tsmixer dataloader or wherever the original ETTh1 etc.
# dataset originated from
class SamformerDataloader:
    def __init__(
        self,
        dataset,
        args,
        logger,
        time_increment=1,
        train_ratio=0.7,
        val_ratio=0.2,
    ):
        file_path = (
            SCRIPT_DIR.parents[2] / "data" / "samformer_datasets" / f"{dataset}.csv"
        )
        df_raw = pd.read_csv(file_path, index_col=0)

        n = len(df_raw)
        seq_len = args.seq_len
        pred_len = args.horizon
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.args = args
        self.time_increment = time_increment

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
        sequential_comparison = True

        self.dataloader = self.create_dataloader(
            df_raw,
            train_end,
            val_end,
            test_end,
            sequential_comparison=sequential_comparison,
        )
        # if sequential_comparison:
        #     self.dataloader_list = self.create_dataloader(
        #         df_raw,
        #         train_end,
        #         val_end,
        #         test_end,
        #         sequential_comparison=sequential_comparison,
        #     )
        # else:
        #     self.dataloader = self.create_dataloader(
        #         df_raw,
        #         train_end,
        #         val_end,
        #         test_end,
        #         sequential_comparison=sequential_comparison,
        #     )

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
                seed=self.args.seed,
            )

            # Process each split
            dataloaders_list = []
            scalers_list = []

            for train_data, val_data, test_data in train_val_test_splits:
                # Create scaler for this split and fit on training data
                scaler = StandardScaler()
                scaler.fit(train_data.values)

                # Transform data
                train_arr = scaler.transform(train_data.values)
                val_arr = scaler.transform(val_data.values)
                test_arr = scaler.transform(test_data.values)
                # train_arr = train_data.values
                # val_arr = val_data.values
                # test_arr = test_data.values

                # Apply sliding window
                x_train, y_train = construct_sliding_window_data(
                    train_arr, self.seq_len, self.pred_len, self.time_increment
                )
                x_val, y_val = construct_sliding_window_data(
                    val_arr, self.seq_len, self.pred_len, self.time_increment
                )
                x_test, y_test = construct_sliding_window_data(
                    test_arr, self.seq_len, self.pred_len, self.time_increment
                )

                # flatten target matrices
                flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
                y_train, y_val, y_test = (
                    flatten(y_train),
                    flatten(y_val),
                    flatten(y_test),
                )
                # Create datasets and dataloaders
                train_dataset = LabeledDataset(x_train, y_train)
                val_dataset = LabeledDataset(x_val, y_val)
                test_dataset = LabeledDataset(x_test, y_test)

                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.args.batch_size, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=self.args.batch_size, shuffle=True
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=self.args.batch_size, shuffle=False
                )
                # dataloaders_list[0]['train_loader'].dataset.x[0,0,:12]

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

            # standardize by training set
            self.scaler = StandardScaler()
            self.scaler.fit(train_df.values)
            self.train_arr = self.scaler.transform(train_df.values)
            self.val_arr = self.scaler.transform(val_df.values)
            self.test_arr = self.scaler.transform(test_df.values)

            # apply sliding window
            x_train, y_train = construct_sliding_window_data(
                self.train_arr, self.seq_len, self.pred_len, self.time_increment
            )
            x_val, y_val = construct_sliding_window_data(
                self.val_arr, self.seq_len, self.pred_len, self.time_increment
            )
            x_test, y_test = construct_sliding_window_data(
                self.test_arr, self.seq_len, self.pred_len, self.time_increment
            )

            # flatten target matrices
            flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
            y_train, y_val, y_test = flatten(y_train), flatten(y_val), flatten(y_test)
            # Create datasets and dataloaders
            train_dataset = LabeledDataset(x_train, y_train)
            val_dataset = LabeledDataset(x_val, y_val)
            test_dataset = LabeledDataset(x_test, y_test)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.args.batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.args.batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.args.batch_size, shuffle=False
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
        # Rebuild x/y with the chosen stride from scaled test segment
        x_test_sw, y_test_sw = construct_sliding_window_data(
            self.test_arr, self.seq_len, self.pred_len, stride
        )
        if flatten_y:
            y_test_sw = y_test_sw.reshape((y_test_sw.shape[0], -1))
        ds = LabeledDataset(x_test_sw, y_test_sw)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    def rolling_test_windows(
        self,
        step: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Optional generator yielding (x, y) test windows (scaled) with given step.
        """
        step = step if step is not None else self.seq_len
        x, y = construct_sliding_window_data(
            self.test_arr, self.seq_len, self.pred_len, step
        )
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
        max_possible_additional_splits = (
            available_data_after_first // min_data_per_split
        )

        if splits_num - 1 > max_possible_additional_splits:
            raise ValueError(
                f"Cannot create {splits_num} splits. Maximum possible splits: {max_possible_additional_splits + 1}"
            )

        # first_train_data = data[: val_end + seq_len]
        # first_test_data = data[val_end : val_end + pred_len]
        # first_data_split = [first_train_data, first_test_data]

        first_train_data = data[:train_end]
        first_val_data = data[train_end - self.seq_len : val_end]
        first_test_data = data[val_end - self.seq_len : val_end + self.pred_len]

        first_data_split = [first_train_data, first_val_data, first_test_data]

        splits.append(first_data_split)

        # Pre-generate evenly distributed random val_ends for remaining splits
        if splits_num > 1:
            start_point = (
                val_end + self.pred_len
            )  # Earliest possible start for next split
            end_point = len(data) - self.pred_len  # Latest possible end

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
                # TODO: hard coded train ratio, maybe change this to a parameter
                # we take 80% of the training set for training the model
                # the other 20% of that set for validating it and selecting the
                # best one.
                train_ratio = 0.8
                train_end = int(random_val_end * train_ratio)
                train_data = data[:train_end]
                val_data = data[train_end - self.seq_len : random_val_end]
                test_data = data[
                    random_val_end - self.seq_len : random_val_end + self.pred_len
                ]
                splits.append([train_data, val_data, test_data])

        return splits


# class SamformerDataloader:
#     def __init__(
#         self,
#         dataset,
#         args,
#         logger,
#         time_increment=1,
#         train_ratio=0.7,
#         val_ratio=0.2,
#     ):
#         file_path = (
#             SCRIPT_DIR.parents[2] / "data" / "samformer_datasets" / f"{dataset}.csv"
#         )
#         df_raw = pd.read_csv(file_path, index_col=0)
#
#         n = len(df_raw)
#         seq_len = args.seq_len
#         pred_len = args.horizon
#
#         # train-validation-test split for ETT* datasets
#         if dataset.startswith("ETTm"):
#             train_end = 12 * 30 * 24 * 4
#             val_end = train_end + 4 * 30 * 24 * 4
#             test_end = val_end + 4 * 30 * 24 * 4
#         elif dataset.startswith("ETTh"):
#             train_end = 12 * 30 * 24
#             val_end = train_end + 4 * 30 * 24
#             test_end = val_end + 4 * 30 * 24
#         else:
#             # train_end = round(train_ratio * n)
#             # val_end = round(val_ratio * n)
#             # test_end = n
#             train_end = int(n * train_ratio)
#             val_end = n - int(n * val_ratio)
#             test_end = n
#         train_df = df_raw[:train_end]
#         val_df = df_raw[train_end - seq_len : val_end]
#         # -seq_len, cause that part is used as input
#         test_df = df_raw[val_end - seq_len : test_end]
#         # standardize by training set
#         self.scaler = StandardScaler()
#         self.scaler.fit(train_df.values)
#         train_df, val_df, test_df = [
#             self.scaler.transform(df.values) for df in [train_df, val_df, test_df]
#         ]
#
#         # apply sliding window
#         x_train, y_train = construct_sliding_window_data(
#             train_df, seq_len, pred_len, time_increment
#         )
#         x_val, y_val = construct_sliding_window_data(
#             val_df, seq_len, pred_len, time_increment
#         )
#         x_test, y_test = construct_sliding_window_data(
#             test_df, seq_len, pred_len, time_increment
#         )
#
#         # flatten target matrices
#         flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
#         y_train, y_val, y_test = flatten(y_train), flatten(y_val), flatten(y_test)
#         train_dataset = LabeledDataset(x_train, y_train)
#         val_dataset = LabeledDataset(x_val, y_val)
#         test_dataset = LabeledDataset(x_test, y_test)
#         pdb.set_trace()
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=True
#         )
#         val_loader = torch.utils.data.DataLoader(
#             val_dataset, batch_size=args.batch_size, shuffle=True
#         )
#         test_loader = torch.utils.data.DataLoader(
#             test_dataset, batch_size=args.batch_size, shuffle=False
#         )
#
#         self.dataloader = {
#             "train_loader": train_loader,
#             "val_loader": val_loader,
#             "test_loader": test_loader,
#         }
#
#     def get_dataloader(self):
#         return self.dataloader
#
#     def get_scaler(self):
#         return self.scaler


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

        assert split_idx < data_split, (
            "the index of data partition should be smaller than the total number of split"
        )

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
            data_num = (
                len(trainset) // data_split
            )  # the number of data in a chunk of the split

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

    def get_scaler(self):
        return self.scaler


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
        file_path = (
            SCRIPT_DIR.parents[2] / "data" / "samformer_datasets" / f"{dataset}.csv"
        )
        df_raw = pd.read_csv(file_path, index_col=0)

        # Convert index to datetime if it's not already
        df_raw.index = pd.to_datetime(df_raw.index)

        n = len(df_raw)
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

                train_test_splits = self.construct_multiple_train_test_split_data(
                    df_raw,
                    self.seq_len,
                    self.pred_len,
                    splits_num=10,
                    val_end=val_end,
                    seed=self.args.seed,
                )
                self.dataloader = train_test_splits

            #     pass

        # But let's keep this functionality
        else:
            # Keep separate train, val, test splits
            train_df_wide = df_raw[:train_end]
            val_df_wide = df_raw[train_end - self.seq_len : val_end]
            test_df_wide = df_raw[val_end - self.seq_len : test_end]

            # Convert to long format
            self.train_df = self._wide_to_long(train_df_wide)
            self.val_df = (
                self._wide_to_long(val_df_wide) if val_df_wide is not None else None
            )
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
        df_long = df_wide.reset_index().melt(
            id_vars=["date"], var_name="unique_id", value_name="y"
        )
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
                print(
                    "Warning: No validation data available because merge_train_val=True"
                )
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

    def get_unique_ids(self) -> list:
        """Get list of unique series identifiers."""
        return self.train_df["unique_id"].unique().tolist()

    # def inverse_transform_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Inverse transform predictions if scaling was applied.
    #     """
    #     if not self.apply_scaling or self.scaler is None:
    #         return predictions
    #
    #     pred_copy = predictions.copy()
    #
    #     if "y" in pred_copy.columns:
    #         # Handle long format predictions
    #         unique_ids = self.get_unique_ids()
    #         for i, uid in enumerate(unique_ids):
    #             mask = pred_copy["unique_id"] == uid
    #             if mask.any():
    #                 values = pred_copy.loc[mask, "y"].values.reshape(-1, 1)
    #                 # Create a temporary array with zeros for other features
    #                 temp_array = np.zeros((len(values), len(unique_ids)))
    #                 temp_array[:, i] = values.flatten()
    #                 # Inverse transform and extract the relevant column
    #                 inv_transformed = self.scaler.inverse_transform(temp_array)
    #                 pred_copy.loc[mask, "y"] = inv_transformed[:, i]
    #
    #     return pred_copy

    def get_data_info(self) -> Dict:
        """Get information about the dataset splits."""
        info = {
            "train_size": len(self.train_df) // len(self.get_unique_ids()),
            "test_size": len(self.test_df) // len(self.get_unique_ids()),
            "n_series": len(self.get_unique_ids()),
            "unique_ids": self.get_unique_ids(),
            "merge_train_val": self.merge_train_val,
            "date_range": {
                "train": (self.train_df["ds"].min(), self.train_df["ds"].max()),
                "test": (self.test_df["ds"].min(), self.test_df["ds"].max()),
            },
            "split_indices": {
                "train_end": self.train_end,
                "val_end": self.val_end,
                "test_end": self.test_end,
            },
        }

        if not self.merge_train_val and self.val_df is not None:
            info["val_size"] = len(self.val_df) // len(self.get_unique_ids())
            info["date_range"]["val"] = (
                self.val_df["ds"].min(),
                self.val_df["ds"].max(),
            )

        return info

    def construct_multiple_train_test_split_data(
        self, data, seq_len, pred_len, splits_num=1, val_end=None, seed=None
    ):
        if seed is not None:
            random.seed(seed)

        splits = []

        # Check if we have enough data for the requested number of splits
        min_data_per_split = seq_len + pred_len
        available_data_after_first = len(data) - (val_end + pred_len)
        max_possible_additional_splits = (
            available_data_after_first // min_data_per_split
        )

        if splits_num - 1 > max_possible_additional_splits:
            raise ValueError(
                f"Cannot create {splits_num} splits. Maximum possible splits: {max_possible_additional_splits + 1}"
            )

        first_train_data = self._wide_to_long(data[:val_end])
        first_test_data = self._wide_to_long(data[val_end : val_end + self.pred_len])

        first_data_split = [first_train_data, first_test_data]
        splits.append(first_data_split)

        # Pre-generate evenly distributed random val_ends for remaining splits
        if splits_num > 1:
            start_point = (
                val_end + self.pred_len
            )  # Earliest possible start for next split
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
                train_data = self._wide_to_long(data[:random_val_end])
                test_data = self._wide_to_long(
                    data[random_val_end : random_val_end + pred_len]
                )
                splits.append([train_data, test_data])

        return splits

    # def construct_multiple_train_test_split_data(data, seq_len, pred_len, time_increment=1, splits_num, val_end, test_end):
    #     splits = []
    #     # First Train/Test split should correspond to the original samformer
    #     # Train/Val/Test split
    #
    #     first_statsfcst_train_data = data[: val_end + self.seq_len]
    #     first_statsfcst_test_data = data[val_end:test_end]
    #
    #     first_statsfcst_data_split = [first_statsfcst_train_data,
    #                                   first_statsfcst_test_data]
    #
    #     for split in (splits_num-1):
    #         a
    #
    #
    #
    #     n_samples = data.shape[0] - (seq_len - 1) - pred_len
    #     range_ = np.arange(0, n_samples, time_increment)
    #     x, y = list(), list()
    #     for i in range_:
    #         x.append(data[i : (i + seq_len)].T)
    #         y.append(data[(i + seq_len) : (i + seq_len + pred_len)].T)
    #     return np.array(x), np.array(y)
    #
    #


# Add these imports if not present
from typing import Iterator, Dict, Optional, Tuple


class DartsDataloader:
    def __init__(
        self,
        dataset,
        args,
        logger,
        time_increment=1,
        train_ratio=0.7,
        val_ratio=0.2,
    ):
        self.dataset = dataset
        self.args = args
        self.logger = logger

        file_path = (
            SCRIPT_DIR.parents[2] / "data" / "samformer_datasets" / f"{dataset}.csv"
        )
        df_raw = pd.read_csv(file_path, index_col=0)

        n = len(df_raw)
        seq_len = args.seq_len
        pred_len = args.horizon
        self.seq_len = seq_len
        self.pred_len = pred_len

        # train-validation-test split for ETT* datasets
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

        # Store for rolling windows
        self.df_raw = df_raw
        self.train_end = train_end
        self.val_end = val_end
        self.test_end = test_end

        train_df = df_raw[:train_end]
        val_df = df_raw[train_end:val_end]
        test_df = df_raw[val_end:test_end]
        feature_names = (
            df_raw.columns.tolist()
            if df_raw.columns is not None
            else [f"feature_{i}" for i in range(df_raw.shape[1])]
        )
        self.feature_names = feature_names

        # sliding window data for direct comparison on test
        val_df_sw = df_raw[train_end - seq_len : val_end]
        test_df_sw = df_raw[val_end - seq_len : test_end]
        x_test_sw, y_test_sw = construct_sliding_window_data(
            test_df_sw.values, seq_len, pred_len, time_increment
        )
        test_dataset_sw = LabeledDataset(x_test_sw, y_test_sw)
        test_loader_sw = torch.utils.data.DataLoader(
            test_dataset_sw, batch_size=1, shuffle=False
        )
        self.sw_dataloader = {
            "feature_names": feature_names,
            "test_loader": test_loader_sw,
        }

        # TimeSeries objects (multivariate)
        self.train_series = TimeSeries.from_values(
            train_df.values, columns=feature_names
        )
        self.val_series = TimeSeries.from_values(val_df.values, columns=feature_names)
        self.test_series = TimeSeries.from_values(test_df.values, columns=feature_names)

        self.dataloader = {
            "train_loader": self.train_series,
            "val_loader": self.val_series,
            "test_loader": self.test_series,
        }

        # Log infos
        self.logger.info(f"Dataset: {dataset}")
        self.logger.info(f"Total length: {n}")
        self.logger.info(f"Train: {len(self.train_series)} ({train_end})")
        self.logger.info(f"Val: {len(self.val_series)} ({val_end - train_end})")
        self.logger.info(f"Test: {len(self.test_series)} ({test_end - val_end})")
        self.logger.info(f"Features: {len(feature_names)} - {feature_names}")

    def get_dataloader(self):
        return self.dataloader

    def get_sliding_window_dataloader(self):
        """Return sliding-window test loader (stride = constructor time_increment)"""
        return self.sw_dataloader

    def get_timeseries(self):
        """Return train, val, test TimeSeries objects"""
        return self.train_series, self.val_series, self.test_series

    def get_combined_series(self, splits=["train", "val", "test"]):
        """Get combined TimeSeries for different combinations of splits"""
        series_list = []
        if "train" in splits:
            series_list.append(self.train_series)
        if "val" in splits:
            series_list.append(self.val_series)
        if "test" in splits:
            series_list.append(self.test_series)
        if len(series_list) == 1:
            return series_list[0]
        combined = series_list[0]
        for series in series_list[1:]:
            combined = combined.append(series)
        return combined

    def get_sliding_window_dataset(
        self, split: str = "test", stride: Optional[int] = None
    ) -> torch.utils.data.DataLoader:
        """
        Build sliding-window LabeledDataset/DataLoader (unscaled) for a given split.
        x: (n_windows, D, seq_len), y: (n_windows, D, pred_len)
        """
        assert split in {"train", "val", "test"}
        stride = stride if stride is not None else 1
        s, e = None, None
        if split == "train":
            s, e = 0, self.train_end
        elif split == "val":
            s, e = self.train_end - self.seq_len, self.val_end
        else:
            s, e = self.val_end - self.seq_len, self.test_end

        seg = self.df_raw.iloc[s:e].values
        x, y = construct_sliding_window_data(seg, self.seq_len, self.pred_len, stride)
        ds = LabeledDataset(x, y)  # unflattened y for convenience
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    def rolling_eval_iterator(
        self,
        step: Optional[int] = None,
        merge_val_into_train: bool = True,
        expand_train: bool = True,
    ) -> Iterator[Dict[str, object]]:
        """
        Yields rolling/evolving splits for ARIMA:
          - train_series_k: TimeSeries from start to current cutoff
          - x_window: np.ndarray (D, seq_len)
          - y_window: np.ndarray (D, pred_len)

        Windows start at (val_end - seq_len) and move forward by 'step'.
        At step k, training end is:
          - val_end at k=0 if merge_val_into_train else train_end
          - in_start (expanding) for k>0 if expand_train
        """
        step = step if step is not None else self.seq_len

        # Define first input window start
        in_start0 = self.val_end - self.seq_len
        k = 0
        while True:
            in_start = in_start0 + k * step
            in_end = in_start + self.seq_len
            out_end = in_end + self.pred_len
            if out_end > self.test_end:
                break

            if k == 0:
                train_end_k = self.val_end if merge_val_into_train else self.train_end
            else:
                train_end_k = (
                    in_start
                    if expand_train
                    else (self.val_end if merge_val_into_train else self.train_end)
                )

            train_series_k = TimeSeries.from_values(
                self.df_raw.iloc[:train_end_k].values, columns=self.feature_names
            )
            x_window = self.df_raw.iloc[in_start:in_end].values.T  # (D, seq_len)
            y_window = self.df_raw.iloc[in_end:out_end].values.T  # (D, pred_len)

            yield {
                "k": k,
                "train_series": train_series_k,
                "x": x_window,
                "y": y_window,
                "in_start": in_start,
                "in_end": in_end,
                "out_end": out_end,
            }
            k += 1


# class DartsDataloader:
#     def __init__(
#         self,
#         dataset,
#         args,
#         logger,
#         time_increment=1,
#         train_ratio=0.7,
#         val_ratio=0.2,
#     ):
#         self.dataset = dataset
#         self.args = args
#         self.logger = logger
#
#         file_path = (
#             SCRIPT_DIR.parents[2] / "data" / "samformer_datasets" / f"{dataset}.csv"
#         )
#         df_raw = pd.read_csv(file_path, index_col=0)
#
#         n = len(df_raw)
#         seq_len = args.seq_len
#         pred_len = args.horizon
#
#         # train-validation-test split for ETT* datasets
#         if dataset.startswith("ETTm"):
#             train_end = 12 * 30 * 24 * 4
#             val_end = train_end + 4 * 30 * 24 * 4
#             test_end = val_end + 4 * 30 * 24 * 4
#         elif dataset.startswith("ETTh"):
#             train_end = 12 * 30 * 24
#             val_end = train_end + 4 * 30 * 24
#             test_end = val_end + 4 * 30 * 24
#         else:
#             train_end = int(n * train_ratio)
#             val_end = n - int(n * val_ratio)
#             test_end = n
#
#         train_df = df_raw[:train_end]
#         val_df = df_raw[train_end:val_end]
#         test_df = df_raw[val_end:test_end]
#         feature_names = (
#             df_raw.columns.tolist()
#             if df_raw.columns is not None
#             else [f"feature_{i}" for i in range(df_raw.shape[1])]
#         )
#         # sliding window data for direct comparison
#         val_df_sw = df_raw[train_end - seq_len : val_end]
#         test_df_sw = df_raw[val_end - seq_len : test_end]
#         x_test_sw, y_test_sw = construct_sliding_window_data(
#             test_df_sw, seq_len, pred_len, time_increment
#         )
#         test_dataset_sw = LabeledDataset(x_test_sw, y_test_sw)
#         test_loader_sw = torch.utils.data.DataLoader(
#             test_dataset_sw, batch_size=1, shuffle=False
#         )
#         self.sw_dataloader = {
#             "feature_names": feature_names,
#             "test_loader": test_loader_sw,
#         }
#
#         self.train_series = TimeSeries.from_values(
#             train_df.values, columns=feature_names
#         )
#
#         self.val_series = TimeSeries.from_values(val_df.values, columns=feature_names)
#
#         self.test_series = TimeSeries.from_values(test_df.values, columns=feature_names)
#
#         self.dataloader = {
#             "train_loader": self.train_series,
#             "val_loader": self.val_series,
#             "test_loader": self.test_series,
#         }
#         train_df = df_raw[:train_end]
#         val_df = df_raw[train_end:val_end]
#         test_df = df_raw[val_end:test_end]
#         feature_names = (
#             df_raw.columns.tolist()
#             if df_raw.columns is not None
#             else [f"feature_{i}" for i in range(df_raw.shape[1])]
#         )
#         # self.train_series = TimeSeries.from_values(
#         #     train_df.values, columns=feature_names
#         # )
#         #
#         # self.val_series = TimeSeries.from_values(val_df.values, columns=feature_names)
#         #
#         # self.test_series = TimeSeries.from_values(test_df.values, columns=feature_names)
#
#         # Log infos
#         self.logger.info(f"Dataset: {dataset}")
#         self.logger.info(f"Total length: {n}")
#         self.logger.info(f"Train: {len(self.train_series)} ({train_end})")
#         self.logger.info(f"Val: {len(self.val_series)} ({val_end - train_end})")
#         self.logger.info(f"Test: {len(self.test_series)} ({test_end - val_end})")
#         self.logger.info(f"Features: {len(feature_names)} - {feature_names}")
#
#     def get_dataloader(self):
#         return self.dataloader
#
#     def get_sliding_window_dataloader(self):
#         """Return train, val, test TimeSeries objects"""
#         return self.sw_dataloader
#
#     def get_timeseries(self):
#         """Return train, val, test TimeSeries objects"""
#         return self.train_series, self.val_series, self.test_series
#
#     def get_combined_series(self, splits=["train", "val", "test"]):
#         """Get combined TimeSeries for different combinations of splits"""
#         series_list = []
#         if "train" in splits:
#             series_list.append(self.train_series)
#         if "val" in splits:
#             series_list.append(self.val_series)
#         if "test" in splits:
#             series_list.append(self.test_series)
#
#         if len(series_list) == 1:
#             return series_list[0]
#
#         combined = series_list[0]
#         for series in series_list[1:]:
#             combined = combined.append(series)
#
#         return combined


if __name__ == "__main__":
    # seq_length_x, seq_length_y, y_start=1, stride = 1, transform = None, add_time_in_day=True, add_time_in_week=False
    dataset = "ETTh1"
    dataset = SAMFormerDatasets(dataset, query_api, 12, 12, 1, 1, None, False, False)

    # Get train/val/test splits
    train_dataset, val_dataset, test_dataset = dataset.get_split_datasets(
        train_ratio=0.6, val_ratio=0.2
    )

    print(train_dataset, val_dataset, test_dataset)
