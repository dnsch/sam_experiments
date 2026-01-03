from src.base.torch_engine import TorchEngine

import torch
from typing import Tuple, Dict


class FEDformer_Engine(TorchEngine):
    """
    FEDformer PyTorch trainer.
    Handles encoder-decoder architecture with time features.
    """

    def __init__(self, label_len: int, **args):
        self._label_len = label_len
        super().__init__(**args)

    def _revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN normalization.

        FEDformer input shape (after _prepare_batch): [batch, seq_len, channels]
        RevIN expects: [batch, seq_len, channels]
        No transpose needed.
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="norm")

    def _revin_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN denormalization.

        FEDformer output shape: [batch, pred_len, channels]
        RevIN expects: [batch, pred_len, channels]
        No transpose needed.
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="denorm")

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        batch_dict = super()._prepare_batch(batch)

        # FEDformer expects [batch, seq_len, channels]
        # Dataloader provides [batch, channels, seq_len]
        batch_dict["x"] = batch_dict["x"].permute(0, 2, 1).contiguous()
        batch_dict["y"] = batch_dict["y"].permute(0, 2, 1).contiguous()

        return batch_dict

    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert back to [batch, channels, timesteps] for evaluation
        preds = preds.permute(0, 2, 1).contiguous()
        labels = labels.permute(0, 2, 1).contiguous()
        return preds, labels

    def _prepare_ground_truths(self, y_batch: torch.Tensor) -> torch.Tensor:
        """
        Prepare ground truths for encoder-decoder models.

        The dataloader returns y with shape [batch, label_len + pred_len, channels]
        but we only compute loss on the last pred_len timesteps.

        Args:
            y_batch: Ground truth tensor of shape [batch, label_len + pred_len, channels]

        Returns:
            Sliced tensor of shape [batch, pred_len, channels]
        """
        # Only use the last pred_len values for loss computation
        return y_batch[:, -self.model.pred_len :, :].contiguous()

    def _forward_pass(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass for FEDformer.
        Prepares decoder input and handles time features.
        """
        batch_dict = self._get_batch_inputs()

        x_enc = batch_dict["x"]
        y_batch = batch_dict["y"]
        x_mark_enc = batch_dict["x_mark"]
        x_mark_dec = batch_dict["y_mark"]

        # Prepare decoder input
        # dec_inp: [batch, label_len + pred_len, channels]
        # Uses last label_len values from encoder input + zeros for prediction
        dec_inp = torch.zeros_like(y_batch[:, -self.model.pred_len :, :]).float()
        dec_inp = (
            torch.cat([x_enc[:, -self._label_len :, :], dec_inp], dim=1).float().to(x_enc.device)
        )

        # Forward pass
        if self.model.output_attention:
            out_batch, attns = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec)
        else:
            out_batch = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec)

        return out_batch
