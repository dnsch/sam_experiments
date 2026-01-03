from src.base.torch_engine import TorchEngine

import torch
from typing import Tuple, Dict


class TSMixerExt_Engine(TorchEngine):
    """
    TSMixerExt pytorch trainer implemented in the sklearn fashion.
    """

    def __init__(
        self,
        batch_size=1,
        num_channels=1,
        pred_len=1,
        extra_channels=1,
        static_channels=1,
        **args,
    ):
        super().__init__(**args)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.extra_channels = extra_channels
        self.static_channels = static_channels

    # RevIN overrides
    # TSMixerExt uses [batch, seq_len, channels] format which matches RevIN's expected input
    # So we override to NOT transpose (unlike the default which assumes [batch, channels, seq_len])

    def _revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN normalization to input tensor.

        TSMixerExt x_hist shape: [batch, seq_len, channels]
        RevIN expects: [batch, seq_len, channels]
        No transpose needed.

        Args:
            x: Input tensor of shape [batch, seq_len, channels]

        Returns:
            Normalized tensor of same shape
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="norm")

    def _revin_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN denormalization to output tensor.

        TSMixerExt output shape: [batch, pred_len, channels]
        RevIN expects: [batch, pred_len, channels]
        No transpose needed.

        Args:
            x: Output tensor of shape [batch, pred_len, channels]

        Returns:
            Denormalized tensor of same shape
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="denorm")

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        """
        Prepare TSMixerExt batch with 5 elements:
        - x_hist: (batch, seq_len, num_channels)
        - x_extra_hist: (batch, seq_len, extra_channels)
        - x_extra_future: (batch, pred_len, extra_channels)
        - x_static: (batch, static_channels)
        - y: (batch, pred_len, num_channels)

        """
        x_hist, x_extra_hist, x_extra_future, x_static, y = batch

        # Flatten y for loss calculation
        # y_flat = y.reshape(y.shape[0], -1)

        return {
            "x": x_hist,
            "x_extra_hist": x_extra_hist,
            "x_extra_future": x_extra_future,
            "x_static": x_static,
            "y": y,
        }

    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape to [batch, channels, pred_len] for per-horizon evaluation
        preds = preds.reshape(preds.shape[0], self.num_channels, self.pred_len)
        labels = labels.reshape(labels.shape[0], self.num_channels, self.pred_len)
        return preds, labels

    def _prepare_predictions(self, preds) -> torch.Tensor:
        # Reshape to [batch, channels, pred_len] for per-horizon evaluation
        prepared_preds = preds.permute(0, 2, 1).contiguous()
        return prepared_preds

    def _prepare_ground_truths(self, y_batch) -> torch.Tensor:
        # Reshape to [batch, channels, pred_len] for per-horizon evaluation
        prepared_y_batch = y_batch.permute(0, 2, 1).contiguous()
        return prepared_y_batch

    def _forward_pass(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        TSMixerExt forward pass with all required inputs.
        """
        # Note: If use_revin, then x is already revin normalized
        batch_dict = self._get_batch_inputs()

        out_batch = self.model(
            x_hist=batch_dict["x"],
            x_extra_hist=batch_dict["x_extra_hist"],
            x_extra_future=batch_dict["x_extra_future"],
            x_static=batch_dict["x_static"],
            flatten_output=False,
        )
        return out_batch
