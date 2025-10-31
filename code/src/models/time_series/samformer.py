from torch import nn

from src.utils.samformer_utils.attention import scaled_dot_product_attention
from src.utils.samformer_utils.revin import RevIN

from src.base.model import BaseModel


class SAMFormerArchitecture(BaseModel):
    def __init__(
        self,
        num_channels=7,
        seq_len=512,
        hid_dim=16,
        horizon=720,
        use_revin=True,
        **args,
    ):
        super(SAMFormerArchitecture, self).__init__(horizon=horizon, **args)
        self.revin = RevIN(num_features=num_channels)
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)
        self.linear_forecaster = nn.Linear(seq_len, horizon)
        self.use_revin = use_revin

    def forward(self, x, flatten_output=True):
        # RevIN Normalization
        if self.use_revin:
            x_norm = self.revin(x.transpose(1, 2), mode="norm").transpose(
                1, 2
            )  # (n, D, L)
        else:
            x_norm = x
        # Channel-Wise Attention
        queries = self.compute_queries(x_norm)  # (n, D, hid_dim)
        keys = self.compute_keys(x_norm)  # (n, D, hid_dim)
        values = self.compute_values(x_norm)  # (n, D, L)
        if hasattr(nn.functional, "scaled_dot_product_attention"):
            att_score = nn.functional.scaled_dot_product_attention(
                queries, keys, values
            )  # (n, D, L)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values)  # (n, D, L)
        out = x_norm + att_score  # (n, D, L)
        # Linear Forecasting
        out = self.linear_forecaster(out)  # (n, D, H)
        # RevIN Denormalization
        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode="denorm").transpose(
                1, 2
            )  # (n, D, H)
        if flatten_output:
            return out.reshape([out.shape[0], out.shape[1] * out.shape[2]])
        else:
            return out
