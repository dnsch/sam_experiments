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
