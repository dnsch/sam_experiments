from src.base.torch_engine import TorchEngine

import torch
from typing import Tuple, Dict


class DLinear_Engine(TorchEngine):
    """
    DLinear PyTorch trainer
    """

    def __init__(self, **args):
        super().__init__(**args)

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        batch_dict = super()._prepare_batch(batch)

        # DLinear expects [batch, seq_len, channels]
        # Dataloader provides [batch, channels, seq_len]
        batch_dict["x"] = batch_dict["x"].permute(0, 2, 1).contiguous()
        batch_dict["y"] = batch_dict["y"].permute(0, 2, 1).contiguous()

        return batch_dict

    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert back to [batch, channels, timesteps] for evaluation
        preds = preds.permute(0, 2, 1).contiguous()
        labels = labels.permute(0, 2, 1).contiguous()
        return preds, labels

    def _forward_pass(self, x_batch: torch.Tensor) -> torch.Tensor:
        batch_dict = self._get_batch_inputs()
        x = batch_dict["x"]
        out_batch = self.model(x)
        return out_batch
