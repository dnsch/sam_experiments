from src.base.torch_engine import TorchEngine

import torch
from typing import Tuple, Dict


class TSMixer_Engine(TorchEngine):
    """
    TSMixer pytorch trainer implemented in the sklearn fashion.
    """

    def __init__(
        self,
        batch_size=1,
        num_channels=1,
        pred_len=1,
        **args,
    ):
        super().__init__(**args)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.pred_len = pred_len

    # RevIN overrides
    # TSMixer uses [batch, seq_len, channels] format which matches RevIN's expected input
    # So we override to NOT transpose (unlike the default which assumes [batch, channels, seq_len])

    def _revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN normalization to input tensor.

        TSMixer input shape: [batch, seq_len, channels]
        RevIN expects: [batch, seq_len, channels]
        No transpose needed (as in the base revin norm)

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

        TSMixer output shape: [batch, pred_len, channels]
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
        batch_dict = super()._prepare_batch(batch)

        # TSMixer expects [batch, seq_len, channels]
        batch_dict["x"] = batch_dict["x"].permute(0, 2, 1).contiguous()

        return batch_dict

    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape to [batch, channels, pred_len] for per-horizon evaluation

        preds = preds.reshape(preds.shape[0], self.num_channels, self.pred_len)
        labels = labels.reshape(labels.shape[0], self.num_channels, self.pred_len)
        return preds, labels

    def _prepare_predictions(self, preds) -> torch.Tensor:
        # Reshape to [batch, channels, pred_len] for per-horizon evaluation
        prepared_preds = preds.permute(0, 2, 1).contiguous()
        return prepared_preds

    def _forward_pass(self, x_batch: torch.Tensor) -> torch.Tensor:
        # Note: If use_revin, then x is already revin normalized
        out_batch = self.model(x_batch, False)
        return out_batch
