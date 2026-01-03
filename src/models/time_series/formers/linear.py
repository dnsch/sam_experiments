import torch
import torch.nn as nn

from src.base.model import BaseModel


class Linear(BaseModel):
    """
    Just one Linear layer for time series forecasting
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        **kwargs,
    ):
        super().__init__(seq_len=seq_len, pred_len=pred_len)

        self.channels = enc_in
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # [Batch, Output length, Channel]

