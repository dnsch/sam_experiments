from src.base.torch_engine import TorchEngine

import torch
from typing import Tuple, Dict


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
        self._nan_batch_count = 0  # Track NaN batches for monitoring

    def _revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RevIN normalization."""
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="norm")

    def _revin_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RevIN denormalization."""
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="denorm")

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        batch_dict = super()._prepare_batch(batch)
        batch_dict["x"] = batch_dict["x"].permute(0, 2, 1).contiguous()
        batch_dict["y"] = batch_dict["y"].permute(0, 2, 1).contiguous()
        return batch_dict

    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = preds.permute(0, 2, 1).contiguous()
        labels = labels.permute(0, 2, 1).contiguous()
        return preds, labels

    def _prepare_ground_truths(self, y_batch: torch.Tensor) -> torch.Tensor:
        return y_batch[:, -self.model.pred_len :, :].contiguous()

    def _forward_pass(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass for FEDformer.

        Args:
            x_batch: Input tensor (potentially RevIN normalized)
        """
        batch_dict = self._get_batch_inputs()

        x_enc = x_batch
        y_batch = batch_dict["y"]
        x_mark_enc = batch_dict["x_mark"]
        x_mark_dec = batch_dict["y_mark"]

        # Prepare decoder input
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

    def _has_nan_gradients(self) -> bool:
        """Check if any parameter has NaN gradients."""
        for param in self.model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                return True
        return False

    def _optimizer_step(self, loss: torch.Tensor, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Execute optimizer step for FEDformer.
        Includes NaN-safe handling for SAM optimizer.
        """
        if self._sam:
            # Check if loss is already NaN
            if torch.isnan(loss):
                self._logger.warning("NaN loss detected, skipping batch")
                self._nan_batch_count += 1
                self._optimizer.zero_grad()
                return

            loss.backward()

            # Check for NaN gradients after backward pass
            if self._has_nan_gradients():
                self._logger.warning(
                    f"NaN gradients detected in backward pass, skipping batch "
                    f"(total skipped: {self._nan_batch_count + 1})"
                )
                self._nan_batch_count += 1
                self._optimizer.zero_grad()
                return

            # Gradient clipping BEFORE first_step
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)

            self._optimizer.first_step(zero_grad=True)

            # Second forward pass for SAM
            x_batch_norm = self._revin_norm(x_batch)
            out_batch = self._forward_pass(x_batch_norm)
            out_batch = self._revin_denorm(out_batch)

            # Check for NaN in output after perturbation
            if torch.isnan(out_batch).any():
                self._logger.warning(
                    f"NaN detected after SAM perturbation, reverting and skipping batch "
                    f"(total skipped: {self._nan_batch_count + 1})"
                )
                self._nan_batch_count += 1
                # Revert the perturbation by doing second_step with zero gradients
                self._optimizer.zero_grad()
                self._optimizer.second_step(zero_grad=True)
                return

            prepared_y_batch = self._prepare_ground_truths(y_batch)
            loss = self._compute_loss(out_batch, prepared_y_batch)

            # Check if second loss is NaN
            if torch.isnan(loss):
                self._logger.warning(
                    f"NaN loss after SAM perturbation, reverting and skipping batch "
                    f"(total skipped: {self._nan_batch_count + 1})"
                )
                self._nan_batch_count += 1
                self._optimizer.zero_grad()
                self._optimizer.second_step(zero_grad=True)
                return

            loss.backward()

            # Check for NaN gradients in second backward
            if self._has_nan_gradients():
                self._logger.warning(
                    f"NaN gradients in second backward, skipping second step "
                    f"(total skipped: {self._nan_batch_count + 1})"
                )
                self._nan_batch_count += 1
                self._optimizer.zero_grad()
                self._optimizer.second_step(zero_grad=True)
                return

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.second_step(zero_grad=True)

        elif self._gsam:
            loss.backward()
            self._optimizer.set_closure(
                self._loss_fn, x_batch, y_batch, self._revin_closure_wrapper
            )
            out_batch, loss = self._optimizer.step()
            self._lr_scheduler.step()
            self._optimizer.update_rho_t()
        else:
            # Default (no SAM)
            self._optimizer.zero_grad()
            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

    def _revin_closure_wrapper(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Wrapper for GSAM closure that applies RevIN and uses proper forward pass."""
        x_norm = self._revin_norm(x_batch)
        out = self._forward_pass(x_norm)
        return self._revin_denorm(out)

    def _on_epoch_end(self, epoch: int):
        """Log NaN batch statistics at end of epoch."""
        if self._nan_batch_count > 0:
            self._logger.info(
                f"Epoch {epoch + 1}: Skipped {self._nan_batch_count} batches due to NaN"
            )
            self._nan_batch_count = 0  # Reset counter for next epoch
