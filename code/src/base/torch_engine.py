from pathlib import Path
import sys
import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))

from src.utils.metrics import (
    TrainingMetrics,
    get_metric_objects,
    get_metric_name_from_object,
)


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
        primary_metric="loss",  # Metric to use for early stopping
        metrics=None,
    ):
        # Initialize vars
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
        self._primary_metric = primary_metric  # For early stopping

        # Init metrics
        # metric_names are the strings that are passed to the function
        self._metric_names = metrics if metrics is not None else []
        # _metric_calculators is a dict that maps those names to the metric 
        # objects
        self._metric_calculators = {}
        self._initialize_metrics()


        # Initialize tracking variables
        self._epochs = 0
        # Init paths and create if necessary
        self._plot_path = self._save_path / "plots"
        self._plot_path.mkdir(parents=True, exist_ok=True)

        self._logger.info(f"The number of parameters: {self.model.param_num()}")
        self._logger.info(f"Loss function: {self._loss_fn.__class__.__name__}")
        # TODO: maybe change this, as we already have our loss which is the prim
        # metric
        self._logger.info(f"Primary metric for early stopping: {self._primary_metric}")

    # ==========================================================================
    # Initialization methods
    # ==========================================================================

    def _initialize_metrics(self):
        """
        Initialize metric calculators from metric name strings using the
        get_metric_objects function from utils/metrics.py.
        """
        if not self._metric_names:
            #TODO: handle empty _metric_names
            return

        try:
            # Get metric objects from utils/metrics.py
            metric_objects = get_metric_objects(self._metric_names)

            # Move to device and store in dictionary
            for metric_obj in metric_objects:
                metric_name = get_metric_name_from_object(metric_obj)

                # Move to device if possible
                if hasattr(metric_obj, "to"):
                    metric_obj = metric_obj.to(self._device)

                self._metric_calculators[metric_name] = metric_obj

        except ValueError as e:
            self._logger.error(str(e))
            raise
        except ImportError as e:
            self._logger.warning(str(e))
            self._logger.warning("Additional metrics will not be available.")

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
    # Batch preparation
    # ==========================================================================

    # @abstractmethod
    # def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Prepare batch data for the model, as models might expect different
    #     batch dimension formats
    #     Default dataloader returns batches as batchsize x channels x seq_len
    #     Returns: (input_tensor, target_tensor)
    #     """
    #     pass
    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for the model
        Default dataloader returns batches as batchsize x channels x seq_len
        If model expects different input, override this function with necessary
        permutations
        Returns: (input_tensor, target_tensor)
        """
        x_batch, y_batch = batch
        return x_batch, y_batch

    # TODO: I think we won't need that, can't think of any future use case
    # currently:
    # EDIT: keep s.t. it's clearer what's happening in the train loop

    # @abstractmethod
    def _forward_pass(
        self,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute forward pass through the model.
        """
        out_batch = self.model(x_batch, False)
        return out_batch

    # ==========================================================================
    # Metrics
    # ==========================================================================


    def _get_metric_names(self) -> List[str]:
        """
        Return list of metric names to compute.
        Includes loss and any additional metrics.

        Returns:
            List of metric names, e.g., ['mse', 'mape', 'rmse', 'mae']
        """
        metric_names = [self._get_loss_name()]  # Always include loss
        metric_names.extend(self._metric_calculators.keys())  # Add additional metrics
        return metric_names

    def _get_loss_name(self) -> str:
        """
        Return the name of the loss metric.
        Override if you want a different name than 'loss'.
        """
        # Try to infer from loss function class name
        loss_class_name = self._loss_fn.__class__.__name__.lower()
        if "mse" in loss_class_name:
            return "mse"
        elif "mae" in loss_class_name or "l1" in loss_class_name:
            return "mae"
        elif "huber" in loss_class_name:
            return "huber"
        return "loss"  # Default fallback

    # ==========================================================================
    # Optional Hook Methods
    # ==========================================================================

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss. Override for custom loss computation."""
        return self._loss_fn(pred, target)

    def _optimizer_step(
        self,
        loss: torch.Tensor,
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

    def _compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all metrics: loss + additional metrics from the metrics list.
        Override for custom metric computation.

        Returns:
            Dictionary with metric_name: value pairs
        """
        metrics = {}
        loss_name = self._get_loss_name()

        # Always compute the loss
        loss = self._compute_loss(pred, target)
        metrics[loss_name] = loss.item()

        # Compute additional metrics
        for metric_name, metric_calculator in self._metric_calculators.items():
            try:
                metric_value = metric_calculator(pred, target)
                # TODO: test if this works
                # Handle both tensor and scalar returns
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                metrics[metric_name] = metric_value
            except Exception as e:
                self._logger.warning(f"Failed to compute metric '{metric_name}': {e}")
                metrics[metric_name] = float("nan")

        return metrics

    def _on_epoch_start(self, epoch: int):
        """
        Hook called at the start of each epoch.
        If needed, put calls in here that should be executed at the start of each
        epoch
        """
        pass

    def _on_epoch_end(self, epoch: int):
        """
        Hook called at the end of each epoch.
        If needed, put calls in here that should be executed at the end of each
        epoch
        """
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

    # TODO: change name
    def _should_save_epoch_model(self, epoch: int) -> bool:
        """Determine if model should be saved this epoch."""
        return True

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    def train_batch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        Returns: Dictionary of metric_name: mean_value pairs
        """
        self.model.train()
        batch_metrics = defaultdict(list)

        for batch_idx, batch in enumerate(self._get_train_loader()):
            # Prepare data
            x_batch, y_batch = self._prepare_batch(batch)
            x_batch = self._to_device(x_batch)
            y_batch = self._to_device(y_batch)

            # Forward pass
            pred = self._forward_pass(x_batch)

            # Compute loss
            loss = self._compute_loss(pred, y_batch)

            # Optimizer step
            self._optimizer_step(loss, x_batch, y_batch)

            # Compute all metrics
            metrics = self._compute_metrics(pred, y_batch)

            # Store all metrics
            for metric_name, metric_value in metrics.items():
                batch_metrics[metric_name].append(metric_value)

        # Return mean of all metrics
        return {name: np.mean(values) for name, values in batch_metrics.items()}

    def train(self) -> Optional[float]:
        """
        Main training loop.
        Returns: primary metric value on test set
        """
        self._logger.info("Start training!")
        self._logger.info(f"Tracking metrics: {self._get_metric_names()}")

        wait = 0
        min_metric = np.inf
        all_metrics = TrainingMetrics()

        start_time = time.time()

        for epoch in range(self._max_epochs):
            self._on_epoch_start(epoch)

            # Training
            t1 = time.time()
            train_metrics = self.train_batch()
            t2 = time.time()

            # Validation
            v1 = time.time()
            val_metrics = self.evaluate("val")
            v2 = time.time()

            # Update learning rate
            current_lr = self._get_current_lr()
            self._step_scheduler()

            # Store metrics
            all_metrics.add_train_metrics(train_metrics)
            all_metrics.add_val_metrics(val_metrics)

            # Log metrics
            self._log_epoch_metrics(
                epoch, train_metrics, val_metrics, t2 - t1, v2 - v1, current_lr
            )

            # Save periodic checkpoint
            if self._should_save_epoch_model(epoch):
                model_list_save_path = self._save_path / "saved_models"
                self.save_current_model(model_list_save_path, epoch)

            # TODO: maybe put this in own function -> better readability
            # Early stopping based on primary metric
            # something like
            # check_early_stopping()
            # from here:
            current_metric = val_metrics.get(
                self._primary_metric, val_metrics.get(self._get_loss_name(), np.inf)
            )

            if current_metric < min_metric:
                self.save_model(self._save_path)
                self._logger.info(
                    f"Val {self._primary_metric} decreased from {min_metric:.4f} to {current_metric:.4f} \n"
                    f"----------"
                )
                min_metric = current_metric
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        f"Early stop at epoch {epoch + 1}, {self._primary_metric} = {min_metric:.6f}"
                    )
                    self._epochs = epoch + 1
                    break
            # to here.

            # TODO: check_timeout()
            # Timeout check
            if self._check_timeout(start_time):
                self._logger.info(f"Timeout reached at epoch {epoch + 1}")
                self._epochs = epoch + 1
                break

            self._epochs = epoch + 1

            # Custom epoch-end hook
            self._on_epoch_end(epoch)

        # Plot training curves
        if self._enable_plotting:
            self._plot_training_curves(all_metrics)

        # Final evaluation
        test_result = self.evaluate("test")
        return test_result

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        train_time: float,
        val_time: float,
        lr: float,
    ):
        """Log metrics for an epoch in a generic way."""
        # Build metric strings
        train_str = ", ".join(
            [f"Train {k.upper()}: {v:.4f}" for k, v in train_metrics.items()]
        )
        val_str = ", ".join(
            [f"Val {k.upper()}: {v:.4f}" for k, v in val_metrics.items()]
        )

        message = (
            f"Epoch: {epoch + 1:03d}, \n"
            f"{train_str}, \n"
            f"{val_str}, \n"
            f"Train Time: {train_time:.4f}s, "
            f"Val Time: {val_time:.4f}s, \n"
            f"LR: {lr:.4e} \n"
            "----------"
        )
        self._logger.info(message)

    def _check_timeout(self, start_time: float) -> bool:
        """Check if training timeout has been reached."""
        elapsed_hours = (time.time() - start_time) / 3600
        return elapsed_hours > self._timeout_hours

    def _plot_training_curves(self, metrics: TrainingMetrics):
        """Plot training curves for all tracked metrics."""
        try:
            from src.utils.plotting import plot_stats

            # Individual plots
            plot_stats(
                metrics=metrics,
                last_epoch=self._epochs,
                plot_path=self._plot_path,
            )

            self._logger.info("Training curves plotted successfully")
        except Exception as e:
            self._logger.warning(f"Plotting failed: {e}")

    # ==========================================================================
    # Evaluation
    # ==========================================================================

    def evaluate(self, mode: str) -> Dict[str, float]:
        """
        Evaluate the model.
        Args:
            mode: 'val' or 'test'
        Returns: Dictionary of metric_name: value pairs
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

                pred = self._forward_pass(x_batch)

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
    ) -> Dict[str, float]:
        """Compute validation metrics."""
        return self._compute_metrics(preds, labels)

    def _compute_test_metrics(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute test metrics and log detailed results.
        Override for custom test evaluation.
        """
        metrics = self._compute_metrics(preds, labels)

        # Log all metrics
        for name, value in metrics.items():
            self._logger.info(f"Test {name.upper()}: {value:.4f}")

        return metrics

    def _get_train_loader(self):
        """Get training data loader."""
        return self._dataloader["train_loader"]

    def _get_dataloader(self, key: str):
        """Get data loader by key."""
        return self._dataloader[key]
