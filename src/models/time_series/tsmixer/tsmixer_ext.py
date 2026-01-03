from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel

from .layers import (
    ConditionalFeatureMixing,
    ConditionalMixerLayer,
    TimeBatchNorm2d,
    feature_to_time,
    time_to_feature,
)


class TSMixerExt(BaseModel):
    """TSMixerExt model for time series forecasting.

    This model forecasts time series data by integrating historical time series data,
    future known inputs, and static contextual information.

    Args:
        num_channels: Number of channels in the historical time series data.
        seq_len: Length of the input time series sequences.
        pred_len: Length of the output prediction sequences.
        extra_channels: Number of channels in the extra (future known) inputs.
        hidden_channels: Number of hidden channels used in the mixer layers.
        static_channels: Number of channels in the static feature inputs.
        ff_dim: Inner dimension of the feedforward network in the mixer layers.
        activation_fn: Name of the activation function to be used.
        num_blocks: Number of mixer blocks in the model.
        dropout_rate: Dropout rate used in the mixer layers.
        normalize_before: Whether to apply normalization before mixer layer.
        norm_type: Type of normalization to use. "batch" or "layer".
    """

    def __init__(
        self,
        num_channels: int,
        seq_len: int = 96,
        pred_len: int = 96,
        extra_channels: int = 1,
        hidden_channels: int = 64,
        static_channels: int = 1,
        ff_dim: int = 64,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: str = "layer",
        **kwargs,
    ):
        assert static_channels > 0, "static_channels must be greater than 0"
        # Pass seq_len and pred_len to BaseModel
        super().__init__(seq_len=seq_len, pred_len=pred_len)

        self.num_channels = num_channels

        # Transform activation_fn to callable
        activation_fn_callable = getattr(F, activation_fn)

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type_cls = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Linear projections
        self.fc_hist = nn.Linear(seq_len, pred_len)
        self.fc_out = nn.Linear(hidden_channels, num_channels)

        # Feature mixing layers
        self.feature_mixing_hist = ConditionalFeatureMixing(
            sequence_length=pred_len,
            input_channels=num_channels + extra_channels,
            output_channels=hidden_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn_callable,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type_cls,
        )
        self.feature_mixing_future = ConditionalFeatureMixing(
            sequence_length=pred_len,
            input_channels=extra_channels,
            output_channels=hidden_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn_callable,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type_cls,
        )

        # Conditional mixer layers
        self.conditional_mixer = self._build_mixer(
            num_blocks,
            hidden_channels,
            pred_len,
            ff_dim=ff_dim,
            static_channels=static_channels,
            activation_fn=activation_fn_callable,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type_cls,
        )

        # Initialize weights
        self._init_weights()

    @staticmethod
    def _build_mixer(num_blocks: int, hidden_channels: int, pred_len: int, **kwargs):
        """Build the conditional mixer blocks for the model."""
        channels = [2 * hidden_channels] + [hidden_channels] * (num_blocks - 1)

        return nn.ModuleList(
            [
                ConditionalMixerLayer(
                    input_channels=in_ch,
                    output_channels=out_ch,
                    sequence_length=pred_len,
                    **kwargs,
                )
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        x_extra_hist: torch.Tensor,
        x_extra_future: torch.Tensor,
        x_static: torch.Tensor,
        flatten_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the TSMixerExt model.

        Args:
            x_hist: Historical time series (batch_size, seq_len, num_channels).
            x_extra_hist: Extra historical features (batch_size, seq_len, extra_channels).
            x_extra_future: Future known features (batch_size, pred_len, extra_channels).
            x_static: Static features (batch_size, static_channels).
            flatten_output: Whether to flatten the output.

        Returns:
            Output tensor (batch_size, pred_len, num_channels) or flattened.
        """
        # Note: If use_revin, then x is already revin normalized

        # Concatenate historical data with extra historical features
        x_hist = torch.cat([x_hist, x_extra_hist], dim=-1)

        # Temporal projection
        x_hist = time_to_feature(self.fc_hist(feature_to_time(x_hist)))

        # Feature mixing
        x_hist, _ = self.feature_mixing_hist(x_hist, x_static=x_static)
        x_future, _ = self.feature_mixing_future(x_extra_future, x_static=x_static)

        # Concatenate and process through mixer layers
        x = torch.cat([x_hist, x_future], dim=-1)
        for mixing_layer in self.conditional_mixer:
            x = mixing_layer(x, x_static=x_static)

        # Output projection
        x = self.fc_out(x)

        # Flatten output if requested
        if flatten_output:
            return x.reshape([x.shape[0], x.shape[1] * x.shape[2]])
        else:
            return x


if __name__ == "__main__":
    seq_len = 10
    pred_len = 5
    num_channels = 2
    extra_channels = 3
    hidden_channels = 8
    static_channels = 4

    m = TSMixerExt(
        num_channels=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        extra_channels=extra_channels,
        hidden_channels=hidden_channels,
        static_channels=static_channels,
    )

    x_hist = torch.randn(3, seq_len, num_channels)
    x_extra_hist = torch.randn(3, seq_len, extra_channels)
    x_extra_future = torch.randn(3, pred_len, extra_channels)
    x_static = torch.randn(3, static_channels)

    y = m(x_hist, x_extra_hist, x_extra_future, x_static)
    print(f"Input shape: {x_hist.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {m.param_num():,}")
