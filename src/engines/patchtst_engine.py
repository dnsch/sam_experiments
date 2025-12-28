from src.base.torch_engine import TorchEngine

import torch
from typing import Tuple, Dict


class PatchTST_Engine(TorchEngine):
    """
    PatchTST pytorch trainer implemented in the sklearn fashion
    """

    def __init__(
        self,
        **args,
    ):
        super().__init__(**args)

    # RevIN override
    # PatchTST receives data as [batch, seq_len, channels] after _prepare_batch
    # which is the expected output for revin, no transpose needded

    def _revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN normalization.

        PatchTST input shape (after _prepare_batch): [batch, seq_len, channels]
        RevIN expects: [batch, seq_len, channels]
        No transpose needed.
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="norm")

    def _revin_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN denormalization.

        PatchTST output shape: [batch, horizon, channels]
        RevIN expects: [batch, horizon, channels]
        No transpose needed.
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="denorm")

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        batch_dict = super()._prepare_batch(batch)

        # Permute to [batch, channels, seq] for PatchTST
        batch_dict["x"] = batch_dict["x"].permute(0, 2, 1).contiguous()
        batch_dict["y"] = batch_dict["y"].permute(0, 2, 1).contiguous()

        return batch_dict

    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = preds.permute(0, 2, 1).contiguous()
        labels = labels.permute(0, 2, 1).contiguous()
        return preds, labels
