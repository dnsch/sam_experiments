import torch
import torch.nn as nn

from src.base.model import BaseModel


class NLinear(BaseModel):
    """
    Normalization-Linear for time series forecasting.

    Subtracts the last value of the input sequence before the linear
    transformation and adds it back after, helping handle distribution shift.
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
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]

