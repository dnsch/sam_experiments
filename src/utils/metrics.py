import torch
from collections import defaultdict
from typing import Dict, List

from torchmetrics.regression import (
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanAbsoluteError,
)


def get_metric_objects(metric_names: List[str]) -> List:
    """
    Initialize metric calculators from metric name strings.

    Args:
        metric_names: List of metric name strings, e.g., ['mse', 'mape', 'rmse']

    Returns:
        List of initialized metric objects

    Raises:
        ValueError: If unknown metric name is provided
    """
    try:
        from torchmetrics.regression import (
            MeanAbsolutePercentageError,
            MeanSquaredError,
            MeanAbsoluteError,
        )
    except ImportError:
        # TODO: change string to make it compatible with uv maybe
        raise ImportError(
            "torchmetrics not installed. Please install it with: pip install torchmetrics"
        )

    metric_map = {
        "mse": lambda: MeanSquaredError(squared=True),
        "rmse": lambda: MeanSquaredError(squared=False),
        "mae": lambda: MeanAbsoluteError(),
        "mape": lambda: MeanAbsolutePercentageError(),
    }

    metric_objects = []
    for name in metric_names:
        name_lower = name.lower()
        if name_lower in metric_map:
            metric_objects.append(metric_map[name_lower]())
        else:
            raise ValueError(
                f"Unknown metric: '{name}'. Available metrics: {list(metric_map.keys())}.\n"
                f"You can implement new metrics in utils/metrics.py"
            )

    return metric_objects


def get_metric_name_from_object(metric_obj) -> str:
    """
    Extract metric name from metric object.

    Args:
        metric_obj: Metric object instance

    Returns:
        Lowercase metric name string
    """
    class_name = metric_obj.__class__.__name__

    # Common mappings
    name_map = {
        "MeanSquaredError": "mse",
        "MeanAbsoluteError": "mae",
        "MeanAbsolutePercentageError": "mape",
    }

    # Check for RMSE (MSE with squared=False)
    if class_name == "MeanSquaredError" and not getattr(metric_obj, "squared", True):
        return "rmse"

    # Return mapped name or lowercase class name
    return name_map.get(class_name, class_name.lower())


class TrainingMetrics:
    """Generic container for training metrics that can handle any metric names."""

    def __init__(self):
        self.train = defaultdict(list)
        self.val = defaultdict(list)

    def add_train_metrics(self, metrics: Dict[str, float]):
        """Add training metrics for current epoch."""
        for name, value in metrics.items():
            self.train[name].append(value)

    def add_val_metrics(self, metrics: Dict[str, float]):
        """Add validation metrics for current epoch."""
        for name, value in metrics.items():
            self.val[name].append(value)

    def get_train_metric(self, name: str) -> List[float]:
        """Get training metric history by name."""
        return self.train[name]

    def get_val_metric(self, name: str) -> List[float]:
        """Get validation metric history by name."""
        return self.val[name]

    def get_all_metric_names(self) -> List[str]:
        """Get all tracked metric names."""
        return list(set(self.train.keys()) | set(self.val.keys()))


# def get_metric_objects(metric_names: list) -> list:
#     """
#     Map metric names to metric objects.
#
#     Args:
#         metric_names: List of metric name strings, e.g., ['mse', 'mape', 'rmse']
#
#     Returns:
#         List of initialized metric objects
#     """
#     metric_map = {
#         "mse": lambda: MeanSquaredError(squared=True),
#         "rmse": lambda: MeanSquaredError(squared=False),
#         "mae": lambda: MeanAbsoluteError(),
#         "mape": lambda: MeanAbsolutePercentageError(),
#     }
#
#     metric_objects = []
#     for name in metric_names:
#         name_lower = name.lower()
#         if name_lower in metric_map:
#             metric_objects.append(metric_map[name_lower]())
#         else:
#             raise ValueError(
#                 f"Unknown metric: {name}. \n You can implement new metrics in utils/metrics.py"
#             )


# TODO: comment or delete the following functions if not needed anymore
# these are from the original codebase for METRLA etc. datasets, probably won't
# be needed for our experiments
def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse
