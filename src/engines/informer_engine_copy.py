from src.base.torch_engine import TorchEngine

import torch
from typing import Tuple, Dict


class Informer_Engine(TorchEngine):
    """
    Informer pytorch trainer implemented in the sklearn fashion.
    """

    def __init__(
        self,
        **args,
    ):
        super().__init__(**args)

    def _revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN normalization.

        Informer input shape (after _prepare_batch): [batch, seq_len, channels]
        RevIN expects: [batch, seq_len, channels]
        No transpose needed.
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="norm")

    def _revin_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN denormalization.

        Informer output shape: [batch, pred_len, channels]
        RevIN expects: [batch, pred_len, channels]
        No transpose needed.
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="denorm")

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        batch_dict = super()._prepare_batch(batch)

        # Informer expects [batch, seq_len, channels]
        batch_dict["x"] = batch_dict["x"].permute(0, 2, 1).contiguous()
        batch_dict["y"] = batch_dict["y"].permute(0, 2, 1).contiguous()

        # x_mark and y_mark are already [batch, seq_len, n_features]
        return batch_dict

    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = preds.permute(0, 2, 1).contiguous()
        labels = labels.permute(0, 2, 1).contiguous()
        return preds, labels

    def _create_decoder_input(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Create decoder input for Informer.
        Uses last label_len steps from encoder input + zeros for pred_len.
        """
        batch_size = x_enc.shape[0]
        channels = x_enc.shape[2]

        # Get last label_len steps from encoder input
        dec_input = x_enc[:, -self.model.label_len :, :].clone()

        # Append pred_len zeros
        zeros = torch.zeros(
            [batch_size, self.model.pred_len, channels],
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        x_dec = torch.cat([dec_input, zeros], dim=1)

        return x_dec

    def _create_decoder_time_features(self, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """
        Create decoder time features for Informer.
        Uses last label_len time features from encoder + zeros for future.
        """
        batch_size = x_mark_enc.shape[0]
        n_features = x_mark_enc.shape[2]

        # Get last label_len time features from encoder
        dec_time_features = x_mark_enc[:, -self.model.label_len :, :].clone()

        # For future pred_len steps, create placeholder time features
        future_time_features = torch.zeros(
            [batch_size, self.model.pred_len, n_features],
            device=x_mark_enc.device,
            dtype=x_mark_enc.dtype,
        )

        x_mark_dec = torch.cat([dec_time_features, future_time_features], dim=1)

        return x_mark_dec

    def _forward_pass(
        self,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        # Get batch data
        batch_dict = self._get_batch_inputs()

        x_enc = batch_dict["x"]
        x_mark_enc = batch_dict.get("x_mark")

        # Create decoder input
        x_dec = self._create_decoder_input(x_enc)

        # Handle time features
        if x_mark_enc is not None:
            x_mark_dec = self._create_decoder_time_features(x_mark_enc)
        else:
            # Create dummy time features if not provided
            batch_size = x_enc.shape[0]
            seq_len = x_enc.shape[1]

            x_mark_enc = torch.zeros(
                [batch_size, seq_len, 4], device=x_enc.device, dtype=x_enc.dtype
            )
            x_mark_dec = self._create_decoder_time_features(x_mark_enc)

        # Forward pass
        out_batch = self.model(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None,
            flatten_output=False,
        )

        return out_batch
