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
        use_revin=False,
        **args,
    ):
        super().__init__(**args)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.use_revin = use_revin

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        batch_dict = super()._prepare_batch(batch)

        # TSMixer expects [batch, seq_len, channels]
        batch_dict["x"] = batch_dict["x"].permute(0, 2, 1).contiguous()

        return batch_dict

    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape to [batch, channels, horizon] for per-horizon evaluation
        preds = preds.reshape(preds.shape[0], self.num_channels, self.pred_len)
        labels = labels.reshape(labels.shape[0], self.num_channels, self.pred_len)
        return preds, labels

    def _prepare_predictions(self, preds) -> torch.Tensor:
        # Reshape to [batch, channels, horizon] for per-horizon evaluation
        prepared_preds = preds.permute(0, 2, 1).contiguous()
        return prepared_preds

    def _forward_pass(self, x_batch: torch.Tensor) -> torch.Tensor:
        # TSMixer forward with flatten_output=True
        # out_batch = self.model(x_batch, True)
        out_batch = self.model(x_batch, False)
        return out_batch
