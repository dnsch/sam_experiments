import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# CIFAR10 datasets
import torchvision
import torchvision.transforms as transforms

from pathlib import Path

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
            # train_end = round(train_ratio * n)
            # val_end = round(val_ratio * n)
            # test_end = n
            train_end = int(n * train_ratio)
            val_end = n - int(n * val_ratio)
            test_end = n
        train_df = df_raw[:train_end]
        val_df = df_raw[train_end - seq_len : val_end]
        test_df = df_raw[val_end - seq_len : test_end]
        # standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)
        train_df, val_df, test_df = [
            self.scaler.transform(df.values) for df in [train_df, val_df, test_df]
        ]

        # apply sliding window
        x_train, y_train = construct_sliding_window_data(
            train_df, seq_len, pred_len, time_increment
        )
        x_val, y_val = construct_sliding_window_data(
            val_df, seq_len, pred_len, time_increment
        )
        x_test, y_test = construct_sliding_window_data(
            test_df, seq_len, pred_len, time_increment
        )
        # flatten target matrices
        flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
        y_train, y_val, y_test = flatten(y_train), flatten(y_val), flatten(y_test)
        train_dataset = LabeledDataset(x_train, y_train)
        val_dataset = LabeledDataset(x_val, y_val)
        test_dataset = LabeledDataset(x_test, y_test)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        self.dataloader = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }

    def get_dataloader(self):
        return self.dataloader

    def get_scaler(self):
        return self.scaler


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


if __name__ == "__main__":
    # seq_length_x, seq_length_y, y_start=1, stride = 1, transform = None, add_time_in_day=True, add_time_in_week=False
    dataset = "ETTh1"
    dataset = SAMFormerDatasets(dataset, query_api, 12, 12, 1, 1, None, False, False)

    # Get train/val/test splits
    train_dataset, val_dataset, test_dataset = dataset.get_split_datasets(
        train_ratio=0.6, val_ratio=0.2
    )

    print(train_dataset, val_dataset, test_dataset)
