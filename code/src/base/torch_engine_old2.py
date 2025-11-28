from pathlib import Path
import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics


@dataclass
class TrainingMetrics:
    """Container for training metrics"""

    train_loss: List[float]
    train_mape: List[float]
    train_rmse: List[float]
    val_loss: List[float]
    val_mape: List[float]
    val_rmse: List[float]


class TorchEngine(ABC):
    def __init__(
        self,
        device,
        model,
        dataloader,
        scaler,
        loss_fn,
        lrate,
        optimizer,
        scheduler,
        clip_grad_value,
        max_epochs,
        patience,
        log_dir,
        logger,
        seed,
        timeout_hours=6,
        enable_plotting=True,
    ):
        self._device = device
        self.model = model.to(self._device)
        self._dataloader = dataloader
        self._scaler = scaler
        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self._timeout_hours = timeout_hours
        self._enable_plotting = enable_plotting

        # Initialize tracking variables
        self._epochs = 0

        # Init paths and create if necessary
        self._plot_path = self._save_path / "plots"
        self._plot_path.mkdir(parents=True, exist_ok=True)

        self._logger.info(f"The number of parameters: {self.model.param_num()}")

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        return tensors.to(self._device)

    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        return tensors.detach().cpu().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        return torch.tensor(nparray, dtype=torch.float32)

    def _inverse_transform(self, tensors):
        inv = lambda tensor: self._scaler.inverse_transform(tensor)
        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        return inv(tensors)

    # ==========================================================================
    # Model Saving and Loading
    # ==========================================================================

    def save_current_model(self, save_path, epoch):
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f"model_{epoch}.pt"
        torch.save(self.model.state_dict(), filename)

    def save_model(self, save_path):
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f"final_model_s{self._seed}.pt"
        torch.save(self.model.state_dict(), filename)

    def load_model(self, save_path):
        filename = save_path / f"final_model_s{self._seed}.pt"
        self.model.load_state_dict(torch.load(filename))

    # ==========================================================================
    # Abstract Methods
    # (these should be overridden in the specific model enignes)
    # ==========================================================================

    @abstractmethod
    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for the model, as models might expect different
        batch dimension formats
        Returns: (input_tensor, target_tensor)
        """
        pass

    @abstractmethod
    def _forward_pass(
        self, x_batch: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """
        Execute forward pass through the model.
        """
        pass

    # ==========================================================================
    # Optional Hook Methods
    # ==========================================================================

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss. Override for custom loss computation."""
        return self._loss_fn(pred, target)

    def _optimizer_step(
        self, loss: torch.Tensor, x_batch: torch.Tensor, y_batch: torch.Tensor
    ):
        """
        Execute optimizer step. Override for custom optimizers (e.g., SAM).
        Default: standard gradient descent.
        """
        self._optimizer.zero_grad()
        loss.backward()
        if self._clip_grad_value != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self._clip_grad_value
            )
        self._optimizer.step()

    # TODO: check this
    def _compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute metrics. Override for custom metrics.
        Returns: dict with 'mape' and 'rmse' keys
        """
        return {
            "mape": 0.0,  # Implement in child if needed
            "rmse": 0.0,  # Implement in child if needed
        }

    def _on_epoch_start(self, epoch: int):
        """Hook called at the start of each epoch."""
        pass

    def _on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Hook called at the end of each epoch. Override for custom behavior (e.g., attention plotting)."""
        pass

    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        if self._lr_scheduler is None:
            return self._lrate
        return self._lr_scheduler.get_last_lr()[0]

    def _step_scheduler(self):
        """Step the learning rate scheduler."""
        if self._lr_scheduler:
            self._lr_scheduler.step()

    def _should_save_epoch_model(self, epoch: int) -> bool:
        """Determine if model should be saved this epoch. Override for custom logic."""
        return True  # Default: save every epoch

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    def train_batch(self) -> Tuple[float, float, float]:
        """
        Train for one epoch. Template method that uses hooks.
        Returns: (mean_loss, mean_mape, mean_rmse)
        """
        self.model.train()
        losses, mapes, rmses = [], [], []

        for batch_idx, batch in enumerate(self._get_train_loader()):
            # Prepare data
            x_batch, y_batch = self._prepare_batch(batch)
            x_batch = self._to_device(x_batch)
            y_batch = self._to_device(y_batch)

            # Forward pass
            # TODO: check if we really need the training bool here
            pred = self._forward_pass(x_batch, training=True)

            # Compute loss
            loss = self._compute_loss(pred, y_batch)

            # Optimizer step (may be overridden for SAM, etc.)
            self._optimizer_step(loss, x_batch, y_batch)

            # Compute metrics
            metrics = self._compute_metrics(pred, y_batch)

            losses.append(loss.item())
            mapes.append(metrics.get("mape", 0.0))
            rmses.append(metrics.get("rmse", 0.0))

        return np.mean(losses), np.mean(mapes), np.mean(rmses)

    def train(self) -> Optional[float]:
        """
        Main training loop. Template method.
        Returns: test loss (if evaluation is performed)
        """
        self._logger.info("Start training!")

        wait, min_loss = 0, np.inf
        all_metrics = TrainingMetrics([], [], [], [], [], [])

        start_time = time.time()

        for epoch in range(self._max_epochs):
            self._on_epoch_start(epoch)

            # Training
            t1 = time.time()
            train_loss, train_mape, train_rmse = self.train_batch()
            t2 = time.time()

            # Validation
            v1 = time.time()
            val_loss, val_mape, val_rmse = self.evaluate("val")
            v2 = time.time()

            # Update learning rate
            current_lr = self._get_current_lr()
            self._step_scheduler()

            # Log metrics
            self._log_epoch_metrics(
                epoch,
                train_loss,
                train_mape,
                train_rmse,
                val_loss,
                val_mape,
                val_rmse,
                t2 - t1,
                v2 - v1,
                current_lr,
            )

            # Track metrics
            all_metrics.train_loss.append(train_loss)
            all_metrics.train_mape.append(train_mape)
            all_metrics.train_rmse.append(train_rmse)
            all_metrics.val_loss.append(val_loss)
            all_metrics.val_mape.append(val_mape)
            all_metrics.val_rmse.append(val_rmse)

            # Save periodic checkpoint
            if self._should_save_epoch_model(epoch):
                model_list_save_path = self._save_path / "saved_models"
                self.save_current_model(model_list_save_path, epoch)

            # Early stopping
            if val_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info(
                    f"Val loss decrease from {min_loss:.4f} to {val_loss:.4f}"
                )
                min_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        f"Early stop at epoch {epoch + 1}, loss = {min_loss:.6f}"
                    )
                    self._epochs = epoch + 1
                    break

            # Timeout check
            if self._check_timeout(start_time):
                self._logger.info(f"Timeout reached at epoch {epoch + 1}")
                self._epochs = epoch + 1
                break

            self._epochs = epoch + 1

            # Custom epoch-end hook
            self._on_epoch_end(
                epoch,
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_mape": train_mape,
                    "val_mape": val_mape,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                },
            )

        # Plot training curves
        if self._enable_plotting:
            self._plot_training_curves(all_metrics)

        # Final evaluation
        test_result = self.evaluate("test")
        return test_result

    def _log_epoch_metrics(
        self,
        epoch,
        train_loss,
        train_mape,
        train_rmse,
        val_loss,
        val_mape,
        val_rmse,
        train_time,
        val_time,
        lr,
    ):
        """Log metrics for an epoch."""
        message = (
            f"Epoch: {epoch + 1:03d}, "
            f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train MAPE: {train_mape:.4f}, "
            f"Valid Loss: {val_loss:.4f}, Valid RMSE: {val_rmse:.4f}, Valid MAPE: {val_mape:.4f}, "
            f"Train Time: {train_time:.4f}s/epoch, Valid Time: {val_time:.4f}s, LR: {lr:.4e}"
        )
        self._logger.info(message)

    def _check_timeout(self, start_time: float) -> bool:
        """Check if training timeout has been reached."""
        elapsed_hours = (time.time() - start_time) / 3600
        return elapsed_hours > self._timeout_hours

    def _plot_training_curves(self, metrics: TrainingMetrics):
        """Plot training curves. Override for custom plotting."""
        try:
            from src.utils.plotting import plot_stats

            plot_stats(
                metrics.train_loss,
                metrics.train_mape,
                metrics.train_rmse,
                metrics.val_loss,
                metrics.val_mape,
                metrics.val_rmse,
                self._epochs,
                self._timeout_hours > 0,
                self._plot_path,
            )
        except Exception as e:
            self._logger.warning(f"Plotting failed: {e}")

    # ==================== Evaluation ====================

    def evaluate(self, mode: str) -> Tuple[float, float, float]:
        """
        Evaluate the model.
        Args:
            mode: 'val' or 'test'
        Returns: (loss, mape, rmse) for val mode, or detailed metrics for test mode
        """
        if mode == "test":
            self.load_model(self._save_path)

        self.model.eval()

        preds, labels = [], []

        loader_key = f"{mode}_loader"
        with torch.no_grad():
            for batch in self._get_dataloader(loader_key):
                x_batch, y_batch = self._prepare_batch(batch)
                x_batch = self._to_device(x_batch)
                y_batch = self._to_device(y_batch)

                pred = self._forward_pass(x_batch, training=False)

                preds.append(pred.squeeze(-1).cpu())
                labels.append(y_batch.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        if mode == "val":
            return self._compute_validation_metrics(preds, labels)
        elif mode == "test":
            return self._compute_test_metrics(preds, labels)

    def _compute_validation_metrics(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Compute validation metrics."""
        loss = self._compute_loss(preds, labels).item()
        metrics = self._compute_metrics(preds, labels)
        return loss, metrics.get("mape", 0.0), metrics.get("rmse", 0.0)

    def _compute_test_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute test metrics and log detailed results.
        Override this for custom test evaluation.
        """
        loss = self._compute_loss(preds, labels).item()
        self._logger.info(f"Test Loss: {loss:.4f}")
        return loss

    def _get_train_loader(self):
        """Get training data loader. Override if dataloader interface differs."""
        return self._dataloader["train_loader"]

    def _get_dataloader(self, key: str):
        """Get data loader by key. Override if dataloader interface differs."""
        return self._dataloader[key]
