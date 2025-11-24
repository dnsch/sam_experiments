import torch
import torch.nn as nn

# ============================================================

# Loss

# ============================================================


def get_loss_function(loss_name):
    """Return the appropriate loss function based on the name."""
    loss_functions = {
        "mse": nn.MSELoss(),
        "crossentropy": nn.CrossEntropyLoss(),
        "mae": nn.L1Loss(),
        "mape": MAPELoss(),
    }

    if loss_name.lower() not in loss_functions:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available options: {list(loss_functions.keys())}"
        )

    return loss_functions[loss_name.lower()]


# MAPE Loss
class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + self.epsilon))) * 100
