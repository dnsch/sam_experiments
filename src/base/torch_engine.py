from pathlib import Path
import sys
import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))

from src.utils.metrics import (
    TrainingMetrics,
    get_metric_objects,
    get_metric_name_from_object,
)

# TODO: change labels and preds to true and pred everywhere for consistency
# e.g. y_true, y_pred etc.
#
# Training set
#
# X_train, y_train = ...
# y_train_pred = model.predict(X_train)
#
# Validation set
#
# X_val, y_val = ...
# y_val_pred = model.predict(X_val)
#
# # Test set
#
# X_test, y_test = ...
# y_test_pred = model.predict(X_test)


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
        # RevIN parameters
        use_revin,
        revin_affine,
        revin_num_features,
        revin_subtract_last,
        sam,
        gsam,
        scheduler,
        clip_grad_value,
        max_epochs,
        patience,
        log_dir,
        logger,
        seed,
        timeout_hours=6,
        enable_plotting=True,
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
        # RevIN parameters
        self._use_revin = use_revin
        self._revin = None
        if self._use_revin:
            from src.utils.revin import RevIN

            # TODO: num_channels is always provided I think
            # Try to get num_features from model if not provided
            num_features = revin_num_features
            if num_features is None:
                num_features = getattr(model, "num_channels", None)
            if num_features is None:
                raise ValueError(
                    "revin_num_features required when use_revin=True. "
                    "Provide it explicitly or ensure model has 'num_channels' attribute."
                )

            # self._revin = RevIN(num_features=num_features, affine=revin_affine).to(self._device)
            #
            self._revin = RevIN(
                num_features=num_features, affine=revin_affine, subtract_last=revin_subtract_last
            ).to(self._device)
            # self._logger.info(f"RevIN enabled with {num_features} features, affine={revin_affine}")
        # Sharpness Aware Minimization
        self._sam = sam
        self._gsam = gsam
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        # TODO:
        # add as argument to args
        self._timeout_hours = timeout_hours
        self._enable_plotting = enable_plotting

        # Storage for current batch data (accessible to subclasses)
        self._current_batch = None

        # Init metrics
        # _metrics is a dict that maps metric names to the metric
        # objects
        self._metrics = self._initialize_metrics(metrics)

        # Initialize tracking variables
        self._epochs = 0
        # Init paths and create if necessary
        self._plot_path = self._save_path / "plots"
        self._plot_path.mkdir(parents=True, exist_ok=True)

        self._logger.info(f"The number of parameters: {self.model.param_num()}")
        self._logger.info(f"Loss function: {self._loss_fn.__class__.__name__}")

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def _to_device(self, tensors):
        """
        Move tensors to device. Handles lists, tuples, dicts, and individual tensors.

        Args:
            tensors: Tensor(s) to move to device. Can be:
                - Single tensor
                - List of tensors
                - Tuple of tensors
                - Dict of tensors
                - None

        Returns:
            Tensor(s) on the specified device, maintaining the same structure
        """
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        elif isinstance(tensors, tuple):
            return tuple(
                tensor.to(self._device) if tensor is not None else None for tensor in tensors
            )
        elif isinstance(tensors, dict):
            return {k: v.to(self._device) if v is not None else None for k, v in tensors.items()}
        elif tensors is None:
            return None
        return tensors.to(self._device)

    def _to_numpy(self, tensors):
        """
        Convert tensor(s) to numpy array(s).

        Args:
            tensors: Tensor or list of tensors to convert

        Returns:
            Numpy array(s) corresponding to input tensor(s)
        """
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        return tensors.detach().cpu().numpy()

    def _to_tensor(self, nparray):
        """
        Convert numpy array(s) to tensor(s).

        Args:
            nparray: Numpy array or list of numpy arrays to convert

        Returns:
            Tensor(s) corresponding to input array(s)
        """
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        return torch.tensor(nparray, dtype=torch.float32)

    def _inverse_transform(self, tensors):
        """
        Apply inverse transformation using the scaler.

        Args:
            tensors: Tensor or list of tensors to inverse transform

        Returns:
            Inverse transformed tensor(s)
        """
        inv = lambda tensor: self._scaler.inverse_transform(tensor)
        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        return inv(tensors)

    # ==========================================================================
    # Model Saving and Loading
    # ==========================================================================

    def save_current_model(self, save_path, epoch):
        """
        Save model checkpoint for a specific epoch.

        Args:
            save_path: Path to save directory
            epoch: Current epoch number
        """
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f"model_{epoch}.pt"
        torch.save(self.model.state_dict(), filename)

    def save_model(self, save_path):
        """
        Save final model.

        Args:
            save_path: Path to save directory
        """
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f"final_model_s{self._seed}.pt"
        torch.save(self.model.state_dict(), filename)

    def load_model(self, save_path):
        """
        Load saved model.

        Args:
            save_path: Path to save directory
        """
        filename = save_path / f"final_model_s{self._seed}.pt"
        self.model.load_state_dict(torch.load(filename))

    # ==========================================================================
    # Batch preparation
    # ==========================================================================

    def _get_train_loader(self):
        """Get training data loader."""
        return self._dataloader["train_loader"]

    def _get_dataloader(self, key: str):
        """
        Get data loader by key.

        Args:
            key: Key for the dataloader (e.g., 'train_loader', 'val_loader', 'test_loader')

        Returns:
            Requested dataloader
        """
        return self._dataloader[key]

    def _has_time_features(self, batch) -> bool:
        """
        Check if batch contains time features.

        Args:
            batch: Raw batch from dataloader

        Returns:
            True if batch contains time features (length 4), False otherwise
        """
        return len(batch) == 4

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        """
        Prepare batch data for the model.
        Default dataloader returns batches as batchsize x channels x seq_len
        If model expects different input, override this function with necessary
        permutations

        Args:
            batch: Raw batch from dataloader. Can be:
                - (x_batch, y_batch) without time features
                - (x_batch, y_batch, x_mark, y_mark) with time features

        Returns:
            Dictionary with keys:
                - 'x': input tensor
                - 'y': target tensor
                - 'x_mark': input time features (None if not available)
                - 'y_mark': target time features (None if not available)

        """
        if self._has_time_features(batch):
            x_batch, y_batch, x_mark, y_mark = batch
            return {
                "x": x_batch,
                "y": y_batch,
                "x_mark": x_mark,
                "y_mark": y_mark,
            }
        else:
            x_batch, y_batch = batch
            return {
                "x": x_batch,
                "y": y_batch,
                "x_mark": None,
                "y_mark": None,
            }

    # TODO: finish this
    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare test data for evaluation
        Default functionality assumes data to be structured as batch x channels
        x timesteps.
        If predictions and labels were prepared differently,
        override this function with necessary permutations, such that the data
        is reshaped as BxCxT (batch x channels x timesteps)

        Args:
            preds: Model predictions
            labels: Ground truth labels

        Returns:
            Tuple of (preds, labels) in the correct format for evaluation
        """
        preds, labels = preds, labels
        return preds, labels

    def _get_batch_inputs(self) -> Dict[str, torch.Tensor]:
        """
        Get the prepared batch inputs.
        Subclasses can call this in _forward_pass to access batch data.

        Returns:
            Dictionary with 'x', 'y', 'x_mark', 'y_mark' keys
        """
        return self._current_batch

    # prepare shape of predictions
    def _prepare_predictions(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Prepare predictions

        Override this method in subclasses to reshape predictions to match
        target dimensions before computing loss/metrics.

        Default implementation: returns pred unchanged.

        Args:
            pred: Model predictions

        Returns:
            prepared_pred with matching shape
        """
        return pred

    def _prepare_ground_truths(self, y_batch: torch.Tensor) -> torch.Tensor:
        """
        Prepare ground truths

        Override this method in subclasses to reshape ground truths to match
        target dimensions before computing loss/metrics.

        Default implementation: returns y_batch unchanged.

        Args:
            y_batch: ground truths

        Returns:
            prepared_y_batch with matching shape
        """
        return y_batch

    # ==========================================================================
    # Metrics
    # ==========================================================================

    def _initialize_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Initialize metric calculators from metric name strings using the
        get_metric_objects function from utils/metrics.py.

        Args:
            metric_names: List of metric names to initialize (e.g., ['mse', 'mae', 'mape'])

        Returns:
            Dict[str, Any]: Dictionary mapping metric names to metric objects
        """
        metric_names = metric_names if metric_names is not None else []

        if not metric_names:
            return {}

        try:
            # Get metric objects from utils/metrics.py
            metric_objects = get_metric_objects(metric_names)

            metric_calculators = {}
            # Move to device and store in dictionary
            for metric_obj in metric_objects:
                metric_name = get_metric_name_from_object(metric_obj)

                # Move to device if possible
                if hasattr(metric_obj, "to"):
                    metric_obj = metric_obj.to(self._device)

                metric_calculators[metric_name] = metric_obj

            return metric_calculators

        except ValueError as e:
            self._logger.error(str(e))
            raise
        except ImportError as e:
            self._logger.warning(str(e))
            self._logger.warning("Additional metrics will not be available.")
            return {}

    def _get_metric_names(self) -> List[str]:
        """
        Return list of metric names to compute.
        Includes loss and any additional metrics.

        Returns:
            List of metric names, e.g., ['mse', 'mape', 'rmse', 'mae']
        """
        metric_names = [self._get_loss_name()]  # Always include loss
        metric_names.extend(self._metrics.keys())  # Add additional metrics
        return metric_names

    def _get_loss_name(self) -> str:
        """
        Return the name of the loss metric.
        Override if you want a different name than 'loss'.

        Returns:
            String name of the loss metric
        """
        # Try to infer from loss function class name
        loss_class_name = self._loss_fn.__class__.__name__.lower()
        # TODO: extend to other losses
        # bad design, will return mse for rmse?
        if "mse" in loss_class_name:
            return "mse"
        elif "mae" in loss_class_name or "l1" in loss_class_name:
            return "mae"
        elif "huber" in loss_class_name:
            return "huber"
        return "loss"  # Default fallback

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss

        Args:
            pred: Model predictions
            target: Ground truth targets

        Returns:
            Loss value as tensor
        """

        return self._loss_fn(pred, target)

    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute all metrics: loss + additional metrics from the metrics list.

        Args:
            pred: Model predictions
            target: Ground truth targets

        Returns:
            Dictionary with metric_name: value pairs
        """
        metrics = {}
        loss_name = self._get_loss_name()

        # Always compute the loss
        loss = self._compute_loss(pred, target)
        metrics[loss_name] = loss.item()

        # Compute additional metrics
        for metric_name, metric_object in self._metrics.items():
            try:
                metric_value = metric_object(pred, target)

                # TODO: test if this works
                # Handle both tensor and scalar returns
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                metrics[metric_name] = metric_value
            except Exception as e:
                self._logger.warning(f"Failed to compute metric '{metric_name}': {e}")
                metrics[metric_name] = float("nan")

        return metrics

    def _compute_validation_metrics(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute validation metrics.

        Args:
            preds: Model predictions
            labels: Ground truth labels

        Returns:
            Dictionary of metric_name: value pairs
        """
        return self._compute_metrics(preds, labels)

    def _compute_test_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute test metrics and log detailed results.
        Override for custom test evaluation.

        Args:
            preds: Model predictions
            labels: Ground truth labels

        Returns:
            Dictionary of metric_name: value pairs
        """
        preds, labels = self._prepare_test_data(preds, labels)
        metrics = self._compute_metrics(preds, labels)

        # Log all metrics
        for name, value in metrics.items():
            self._logger.info(f"Test {name.upper()}: {value:.4f}")

        # Per-horizon metrics - include ALL metric names (loss + additional metrics)
        all_metric_names = self._get_metric_names()
        horizon_metrics = {metric: [] for metric in all_metric_names}

        for i in range(self.model.pred_len):
            horizon_pred = preds[:, :, i].contiguous()
            horizon_true = labels[:, :, i].contiguous()

            # Compute metrics for this horizon
            horizon_result = self._compute_metrics(horizon_pred, horizon_true)

            # Log horizon metrics
            metric_str = ", ".join([f"{k.upper()}: {v:.4f}" for k, v in horizon_result.items()])
            self._logger.info(f"Horizon {i + 1}, {metric_str}")

            # Store for averaging
            for metric_name, value in horizon_result.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                horizon_metrics[metric_name].append(value)

        # Log average across horizons
        avg_metrics = {f"avg_{k}": np.mean(v) for k, v in horizon_metrics.items()}
        avg_str = ", ".join([f"{k.upper()}: {v:.4f}" for k, v in avg_metrics.items()])
        self._logger.info(f"Average per horizon: {avg_str}")

        if self._enable_plotting:
            self._plot_test_results(preds, labels)

        return metrics

    # Logging
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        train_time: float,
        val_time: float,
        lr: float,
    ):
        """
        Log metrics for an epoch in a generic way.

        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            train_time: Training time in seconds
            val_time: Validation time in seconds
            lr: Current learning rate
        """
        # Build metric strings
        train_str = ", ".join([f"Train {k.upper()}: {v:.4f}" for k, v in train_metrics.items()])
        val_str = ", ".join([f"Val {k.upper()}: {v:.4f}" for k, v in val_metrics.items()])

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

    # ==========================================================================
    # Plotting functions
    # ==========================================================================

    def _plot_training_curves(self, metrics: TrainingMetrics):
        """
        Plot training curves for all tracked metrics.

        Args:
            metrics: TrainingMetrics object containing all tracked metrics
        """
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

    def _plot_test_results(self, preds, labels):
        """
        Generate test result plots.

        Args:
            preds: Model predictions
            labels: Ground truth labels
        """
        try:
            from src.utils.plotting import (
                plot_mean_per_day,
                mean_branch_plot,
                branch_plot,
            )

            # Mean predictions per horizon
            per_day_preds = [preds[:, :, i].mean() for i in range(self.model.pred_len)]
            per_day_labels = [labels[:, :, i].mean() for i in range(self.model.pred_len)]

            plot_mean_per_day(
                per_day_preds,
                per_day_labels,
                self._plot_path,
                "mean_per_day_performance_plot.png",
            )

            # Branch plots for different sample sizes
            for n_samples, label in [(5, "first_5"), (100, "first_100")]:
                mean_branch_plot(
                    preds[:n_samples, :, :],
                    labels[:n_samples, :, :],
                    self._plot_path,
                    f"mean_performance_plot_{label}",
                )

                var_index = 0
                branch_plot(
                    preds[:n_samples, :, :],
                    labels[:n_samples, :, :],
                    var_index,
                    self._plot_path,
                    f"sensor_{var_index}_branch_plot_{label}.png",
                )

        except Exception as e:
            self._logger.warning(f"Test plotting failed: {e}")

    # ==========================================================================
    # Training Functions
    # ==========================================================================

    # Added function for better readibility.
    def _forward_pass(
        self,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute forward pass through the model.

        Default implementation: simple forward with input tensor.
        Override this for models requiring additional inputs (like time features).
        Subclasses can access full batch data via self._get_batch_inputs()

        Args:
            x_batch: Input batch tensor

        Returns:
            Model output tensor
        """
        out_batch = self.model(x_batch, False)
        return out_batch

    def _revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN normalization to input tensor.

        Override this method in subclasses if your model requires
        different transpose/permutation logic.

        Default assumes input shape: [batch, channels, seq_len]
        RevIN expects: [batch, seq_len, channels]

        Args:
            x: Input tensor of shape [batch, channels, seq_len]

        Returns:
            Normalized tensor of same shape
        """
        if not self._use_revin or self._revin is None:
            return x
        # Transpose: [B, C, L] -> [B, L, C] for RevIN, then back
        return self._revin(x.transpose(1, 2), mode="norm").transpose(1, 2)

    def _revin_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN denormalization to output tensor.

        Override this method in subclasses if your model requires
        different transpose/permutation logic.

        Default assumes output shape: [batch, channels, pred_len]
        RevIN expects: [batch, pred_len, channels]

        Args:
            x: Output tensor of shape [batch, channels, pred_len]

        Returns:
            Denormalized tensor of same shape
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x.transpose(1, 2), mode="denorm").transpose(1, 2)

    def _optimizer_step(self, loss: torch.Tensor, x_batch, y_batch):
        # TODO: change the var names s.t. they're consistent with the ones in the
        # batch loop
        """
        Execute optimizer step.
        Default: standard gradient descent.

        Args:
            loss: Computed loss tensor
            x_batch: Input batch (used for SAM/GSAM)
            y_batch: Target batch (used for SAM/GSAM)
        """

        if self._sam:
            # SAM optimizer
            loss.backward()
            self._optimizer.first_step(zero_grad=True)

            # Second forward pass for SAM - apply RevIN again
            x_batch_norm = self._revin_norm(x_batch)
            # TODO: maybe remove flatten output
            out_batch = self._forward_pass(x_batch_norm)
            out_batch = self._revin_denorm(out_batch)

            prepared_y_batch = self._prepare_ground_truths(y_batch)
            loss = self._compute_loss(out_batch, prepared_y_batch)

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.second_step(zero_grad=True)

        elif self._gsam:
            # GSAM optimizer - needs custom closure with RevIN
            loss.backward()
            # self._optimizer.set_closure(self._loss_fn, x_batch, y_batch)
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

    # TODO: check if this works
    def _revin_closure_wrapper(self, x_batch):
        """Wrapper for GSAM closure that applies RevIN."""
        x_norm = self._revin_norm(x_batch)
        out = self.model(x_norm, False)
        return self._revin_denorm(out)

    # TODO: test gsam
    def _get_current_lr(self) -> float:
        """
        Get current LR (handles GSAM special case).

        Returns:
            Current learning rate
        """
        if self._gsam:
            return self._lr_scheduler._last_lr[0]
        # No _lr_scheduler
        # if self._lr_scheduler is None:
        else:
            return self._lrate

    def _step_scheduler(self):
        """Step the learning rate scheduler."""
        if self._lr_scheduler:
            self._lr_scheduler.step()

    # ==========================================================================
    # Training Loop checks
    # ==========================================================================

    # TODO: change name
    def _should_save_epoch_model(self, epoch: int) -> bool:
        """
        Determine if model should be saved this epoch.

        Args:
            epoch: Current epoch number

        Returns:
            True if model should be saved
        """
        return True

    def _check_early_stopping(
        self, val_metrics: Dict[str, float], min_metric: float, wait: int, epoch: int
    ) -> Tuple[bool, float, int]:
        """
        Check early stopping criteria based on validation metrics.

        Args:
            val_metrics: Dictionary of validation metrics
            min_metric: Current best metric value
            wait: Current patience counter
            epoch: Current epoch number

        Returns:
            Tuple of (should_stop, updated_min_metric, updated_wait)

            - should_stop: True if early stopping criteria met
            - updated_min_metric: New best metric value

            - updated_wait: New patience counter

        """
        loss_name = self._get_loss_name()
        current_metric = val_metrics.get(loss_name, np.inf)

        if current_metric < min_metric:
            self.save_model(self._save_path)
            self._logger.info(
                f"Val {loss_name} decreased from {min_metric:.4f} to {current_metric:.4f} \n"
                f"----------"
            )
            min_metric = current_metric
            wait = 0
            should_stop = False
        else:
            wait += 1
            if wait >= self._patience:
                self._logger.info(
                    f"Early stop at epoch {epoch + 1}, {loss_name} = {min_metric:.6f}"
                )
                self._epochs = epoch + 1
                should_stop = True
            else:
                should_stop = False

        return should_stop, min_metric, wait

    def _check_timeout(self, start_time: float, epoch: int) -> bool:
        """
        Check if training timeout has been reached.

        Args:
            start_time: Training start timestamp
            epoch: Current epoch number

        Returns:
            True if timeout reached and training should stop
        """
        elapsed_hours = (time.time() - start_time) / 3600

        if elapsed_hours > self._timeout_hours:
            self._logger.info(f"Timeout reached at epoch {epoch + 1}")
            self._epochs = epoch + 1
            return True

        return False

    # ==========================================================================
    # Optional Hook Methods
    # ==========================================================================
    # Override these if you need custom behaviour.

    def _on_epoch_start(self, epoch: int):
        """
        Hook called at the start of each epoch.
        If needed, put calls in here that should be executed at the start of each
        epoch

        Args:
            epoch: Current epoch number
        """
        pass

    def _on_epoch_end(self, epoch: int):
        """
        Hook called at the end of each epoch.
        If needed, put calls in here that should be executed at the end of each
        epoch

        Args:
            epoch: Current epoch number
        """
        pass

    # TODO: change path if it changes
    def _on_forward_pass(self):
        """
        Hook called after the forward pass during training.

        Override in subclasses to capture model internals such as attention patterns
        (first, you need a way to capture these, see the models/time_series/samformer
        model for reference)
        or other diagnostic information for visualization or analysis purposes.
        """
        pass

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    def train_batch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of metric_name: mean_value pairs
        """
        self.model.train()
        batch_metrics = defaultdict(list)

        for batch_idx, batch in enumerate(self._get_train_loader()):
            # Prepare data
            batch_dict = self._prepare_batch(batch)
            self._current_batch = batch_dict

            batch_dict = self._to_device(batch_dict)
            self._current_batch = batch_dict

            x_batch = batch_dict["x"]
            y_batch = batch_dict["y"]

            # RevIN Normalization
            x_batch_norm = self._revin_norm(x_batch)

            # Forward pass
            pred = self._forward_pass(x_batch_norm).contiguous()

            # Hook that allows to capture model internals
            # in our case, we use it to capture the attention patterns after
            # the forward pass
            self._on_forward_pass()

            # RevIN Denormalization
            pred = self._revin_denorm(pred)

            # prepare predictions first (put them in the right shape)
            prepared_pred = self._prepare_predictions(pred)
            prepared_y_batch = self._prepare_ground_truths(y_batch)
            # Compute loss

            loss = self._compute_loss(prepared_pred, prepared_y_batch)

            # Optimizer step
            # TODO: check if this works with prepared_y_batch
            self._optimizer_step(loss, x_batch, y_batch)

            # Compute all metrics
            metrics = self._compute_metrics(prepared_pred, prepared_y_batch)

            # Store all metrics
            for metric_name, metric_value in metrics.items():
                batch_metrics[metric_name].append(metric_value)

        # Return mean of all metrics
        return {name: np.mean(values) for name, values in batch_metrics.items()}

    def train(self) -> Optional[float]:
        """
        Main training loop.

        Returns:
            Loss value on test set
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
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, t2 - t1, v2 - v1, current_lr)

            # Save periodic checkpoint
            if self._should_save_epoch_model(epoch):
                model_list_save_path = self._save_path / "saved_models"
                self.save_current_model(model_list_save_path, epoch)

            # Check early stopping
            should_stop, min_metric, wait = self._check_early_stopping(
                val_metrics, min_metric, wait, epoch
            )
            if should_stop:
                break

            # Check timeout
            if self._check_timeout(start_time, epoch):
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

    # ==========================================================================
    # Evaluation
    # ==========================================================================

    def evaluate(self, mode: str) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            mode: 'val' or 'test'

        Returns:
            Dictionary of metric_name: value pairs
        """
        if mode == "test":
            self.load_model(self._save_path)

        self.model.eval()

        preds, labels = [], []

        loader_key = f"{mode}_loader"
        with torch.no_grad():
            for batch in self._get_dataloader(loader_key):
                batch_dict = self._prepare_batch(batch)
                self._current_batch = batch_dict

                # Move to device
                batch_dict = self._to_device(batch_dict)
                self._current_batch = batch_dict

                x_batch = batch_dict["x"]
                y_batch = batch_dict["y"]

                # RevIN Normalization
                x_batch_norm = self._revin_norm(x_batch)

                pred = self._forward_pass(x_batch_norm)

                # RevIN Denormalization
                pred = self._revin_denorm(pred)

                # TODO: remove squeeze(-1)?
                preds.append(pred.squeeze(-1).cpu())
                labels.append(y_batch.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        prepared_preds = self._prepare_predictions(preds)
        prepared_labels = self._prepare_ground_truths(labels)

        if mode == "val":
            return self._compute_validation_metrics(prepared_preds, prepared_labels)
        elif mode == "test":
            return self._compute_test_metrics(prepared_preds, prepared_labels)
