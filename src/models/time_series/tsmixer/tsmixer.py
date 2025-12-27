from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel
from src.utils.revin import RevIN

from .layers import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature


class TSMixer(BaseModel):
    """TSMixer model for time series forecasting.

    This model uses a series of mixer layers to process time series data,
    followed by a linear transformation to project the output to the desired
    prediction length.

    Args:
        num_channels: Number of input channels/features.
        seq_len: Length of the input time series sequence.
        horizon: Desired length of the output prediction sequence.
        output_channels: Number of output channels. Defaults to num_channels.
        activation_fn: Activation function to use. Defaults to "relu".
        num_blocks: Number of mixer blocks. Defaults to 2.
        dropout_rate: Dropout rate for regularization. Defaults to 0.1.
        ff_dim: Dimension of feedforward network inside mixer layer. Defaults to 64.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: Type of normalization to use. "batch" or "layer". Defaults to "batch".
        use_revin: Whether to use RevIN normalization. Defaults to False.
    """

    def __init__(
        self,
        num_channels: int,
        seq_len: int = 96,
        horizon: int = 96,
        output_channels: int = None,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        ff_dim: int = 64,
        normalize_before: bool = True,
        norm_type: str = "batch",
        use_revin: bool = False,
        **kwargs,
    ):
        super().__init__(seq_len=seq_len, horizon=horizon)

        self.num_channels = num_channels
        self.output_channels = output_channels if output_channels is not None else num_channels
        self.use_revin = use_revin

        # Transform activation_fn to callable
        activation_fn_callable = getattr(F, activation_fn)

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type_cls = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks,
            num_channels,
            self.output_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn_callable,
            dropout_rate=dropout_rate,
            sequence_length=seq_len,
            normalize_before=normalize_before,
            norm_type=norm_type_cls,
        )

        # Temporal projection layer
        self.temporal_projection = nn.Linear(seq_len, horizon)

        # RevIN normalization
        self.revin = RevIN(num_features=num_channels)

        # Initialize weights
        self._init_weights()

    def _build_mixer(self, num_blocks: int, input_channels: int, output_channels: int, **kwargs):
        """Build the mixer blocks for the model.

        Args:
            num_blocks: Number of mixer blocks to be built.
            input_channels: Number of input channels for the first block.
            output_channels: Number of output channels for the last block.
            **kwargs: Additional keyword arguments for mixer layer configuration.

        Returns:
            nn.Sequential: Sequential container of mixer layers.
        """
        channels = [input_channels] * (num_blocks - 1) + [output_channels]

        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x_hist: torch.Tensor, flatten_output: bool = False) -> torch.Tensor:
        """Forward pass of the TSMixer model.

        Args:
            x_hist: Input time series tensor (batch_size, seq_len, num_channels).
            flatten_output: Whether to flatten the output. Defaults to False.

        Returns:
            Output tensor (batch_size, horizon, output_channels) or flattened.
        """
        # RevIN Normalization
        if self.use_revin:
            x_norm = self.revin(x_hist, mode="norm")
        else:
            x_norm = x_hist

        # Mixer layers
        x = self.mixer_layers(x_norm)

        # Temporal projection
        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)

        # RevIN Denormalization
        if self.use_revin:
            x = self.revin(x, mode="denorm")

        # Flatten output if requested
        if flatten_output:
            return x.reshape([x.shape[0], x.shape[1] * x.shape[2]])
        else:
            return x


if __name__ == "__main__":
    m = TSMixer(num_channels=2, seq_len=10, horizon=5, output_channels=4)
    x = torch.randn(3, 10, 2)
    y = m(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {m.param_num():,}")
