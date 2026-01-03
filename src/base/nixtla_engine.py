from pathlib import Path
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from src.utils.metrics import (
    get_metric_objects,
    get_metric_name_from_object,
)
from src.utils.model_utils import statsforecast_to_tensor


class NixtlaEngine:
    """
    Engine for Nixtla-based models (statsforecast, neuralforecast, mlforecast).

    This engine provides a unified interface similar to TorchEngine for training
    and evaluating statistical and ML-based forecasting models from the Nixtla ecosystem.
    """

    # TODO: scaler None here, that means we can't scale the data back
    # but as we compare it to other scaled data and preds anyway, maybe
    # this option is not needed
    def __init__(
        self,
        model,
        dataloader,
        scaler=None,
        loss_fn=None,
        backend=None,
        num_channels=1,
        pred_len=1,
        freq=None,
        n_jobs=-1,
        log_dir=None,
        logger=None,
        seed=1,
        enable_plotting=True,
        metrics=None,
        args=None,
        **kwargs,
    ):
        # Initialize core components
        self.model = model
        self._dataloader = dataloader
        self._scaler = scaler
        self._loss_fn = loss_fn
        self._backend = backend
        self._num_channels = num_channels
        self._pred_len = pred_len
        self._freq = freq
        self._n_jobs = n_jobs
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self._enable_plotting = enable_plotting
        self._alias = args.alias if args else None

        # Initialize metrics
        self._metrics = self._initialize_metrics(metrics)

        # Initialize paths
        self._plot_path = self._save_path / "plots"
        self._plot_path.mkdir(parents=True, exist_ok=True)

        # Initialize backend-specific components
        self._initialize_backend(backend, freq, n_jobs)

        # Log initialization info
        self._logger.info(f"Backend: {self._backend}")
        self._logger.info(f"Loss function: {self._loss_fn.__class__.__name__}")
        self._logger.info(f"Prediction length: {self._pred_len}")
        self._logger.info(f"Number of channels: {self._num_channels}")

    # ==========================================================================
    # Backend Initialization
    # ==========================================================================

    def _initialize_backend(self, backend: str, freq: Optional[str], n_jobs: int):
        """
        Initialize backend-specific components.

        Args:
            backend: Backend type ('statsforecast', 'neuralforecast', 'mlforecast')
            freq: Frequency string for time series
            n_jobs: Number of parallel jobs
        """
        if backend == "statsforecast":
            # Statsforecast uses the model directly
            self._obj = self.model
            self._fit_fn = lambda df, X: self.model.fit(df=df)
            self._forecast_fn = lambda h, X, level=None: self.model.predict(h=h)

        elif backend == "neuralforecast":
            from neuralforecast import NeuralForecast

            models = self.model if isinstance(self.model, (list, tuple)) else [self.model]
            self._obj = NeuralForecast(models=list(models), freq=freq)
            self._fit_fn = lambda df, X: self._obj.fit(df=df, val_size=0)
            self._forecast_fn = lambda h, X, level=None: self._obj.predict(h=h, futr_df=X)

        elif backend == "mlforecast":
            from mlforecast import MLForecast

            self._obj = (
                self.model
                if isinstance(self.model, MLForecast)
                else MLForecast(models=self.model, freq=freq)
            )
            self._fit_fn = lambda df, X: self._obj.fit(
                df=df, id_col="unique_id", time_col="ds", target_col="y"
            )
            self._forecast_fn = lambda h, X, level=None: self._obj.predict(h=h, new_df=X)

        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

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
        if isinstance(tensors, torch.Tensor):
            return tensors.detach().cpu().numpy()
        return tensors

    def _to_tensor(self, data):
        """
        Convert data to tensor(s).

        Args:
            data: Data to convert (numpy array, list, or pandas DataFrame)

        Returns:
            Tensor(s) corresponding to input data
        """
        if isinstance(data, list):
            return [torch.tensor(item, dtype=torch.float32) for item in data]
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32)
        return data

    def _inverse_transform(self, tensors):
        """
        Apply inverse transformation using the scaler.

        Args:
            tensors: Tensor or list of tensors to inverse transform

        Returns:
            Inverse transformed tensor(s)
        """
        if self._scaler is None:
            return tensors

        inv = lambda tensor: self._scaler.inverse_transform(tensor)
        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        return inv(tensors)

    # ==========================================================================
    # Model Saving and Loading
    # ==========================================================================

    def save_model(self, save_path: Path):
        """
        Save final model.

        Args:
            save_path: Path to save directory
        """
        save_path.mkdir(parents=True, exist_ok=True)
        saved_models_path = save_path / "saved_models"
        saved_models_path.mkdir(parents=True, exist_ok=True)

        filename = saved_models_path / f"final_model_s{self._seed}.pkl"

        if self._backend == "statsforecast":
            self.model.save(path=filename, max_size=None, trim=None)
        else:
            import pickle

            with open(filename, "wb") as f:
                pickle.dump(self._obj, f)

        self._logger.info(f"Model saved to {filename}")

    def load_model(self, save_path: Path):
        """
        Load saved model.

        Args:
            save_path: Path to save directory
        """
        saved_models_path = save_path / "saved_models"
        filename = saved_models_path / f"final_model_s{self._seed}.pkl"

        if self._backend == "statsforecast":
            self.model.load(filename)
        else:
            import pickle

            with open(filename, "rb") as f:
                self._obj = pickle.load(f)

        self._logger.info(f"Model loaded from {filename}")

    # ==========================================================================
    # Data Preparation
    # ==========================================================================

    def _get_train_data(self):
        """
        Get training data.

        Returns:
            Training DataFrame
        """
        return self._dataloader[0]

    def _get_data(self, mode: str):
        """
        Get data by mode.

        Args:
            mode: 'train', 'val', or 'test'

        Returns:
            Requested data
        """
        if mode == "train":
            return self._dataloader[0]
        elif mode in ["val", "test"]:
            return self._dataloader[1]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _prepare_test_data(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare test data for evaluation.
        Reshapes data to format: (batch, channels, pred_len) to match TorchEngine.

        Args:
            preds: Raw predictions tensor
            labels: Raw labels tensor

        Returns:
            Tuple of (preds, labels) in standardized format (BxCxT)
        """
        # Reshape to (batch, channels, pred_len)
        preds = preds.reshape(preds.shape[0], self._num_channels, self._pred_len)
        labels = labels.reshape(labels.shape[0], self._num_channels, self._pred_len)

        # # Permute to (batch, channels, timesteps) - same as TorchEngine format
        # preds = torch.permute(preds, (0, 2, 1))
        # labels = torch.permute(labels, (0, 2, 1))

        return preds, labels

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
            metric_objects = get_metric_objects(metric_names)
            metric_calculators = {}

            for metric_obj in metric_objects:
                metric_name = get_metric_name_from_object(metric_obj)
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
        metric_names = [self._get_loss_name()]
        metric_names.extend(self._metrics.keys())
        return metric_names

    def _get_loss_name(self) -> str:
        """
        Return the name of the loss metric.
        Override if you want a different name than 'loss'.

        Returns:
            String name of the loss metric
        """
        if self._loss_fn is None:
            return "loss"

        loss_class_name = self._loss_fn.__class__.__name__.lower()

        if "mse" in loss_class_name:
            return "mse"
        elif "mae" in loss_class_name or "l1" in loss_class_name:
            return "mae"
        elif "huber" in loss_class_name:
            return "huber"
        return "loss"

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.

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
        if self._loss_fn is not None:
            loss = self._compute_loss(pred, target)
            metrics[loss_name] = loss.item() if isinstance(loss, torch.Tensor) else loss

        # Compute additional metrics
        for metric_name, metric_object in self._metrics.items():
            try:
                metric_value = metric_object(pred, target)

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

        # Per-horizon metrics
        horizon_metrics = {metric: [] for metric in self._get_metric_names()}

        for i in range(self._pred_len):
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

    # ==========================================================================
    # Logging
    # ==========================================================================

    def _log_training_info(self, fit_time: float):
        """
        Log training information.

        Args:
            fit_time: Time taken for fitting in seconds
        """
        self._logger.info(f"Fitting completed in {fit_time:.4f}s\n----------")

    # ==========================================================================
    # Plotting Functions
    # ==========================================================================

    def _plot_test_results(self, preds: torch.Tensor, labels: torch.Tensor):
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
            per_day_preds = [preds[:, :, i].mean() for i in range(self._pred_len)]
            per_day_labels = [labels[:, :, i].mean() for i in range(self._pred_len)]

            plot_mean_per_day(
                per_day_preds,
                per_day_labels,
                self._plot_path,
                "mean_per_day_performance_plot.png",
            )

            import pdb

            pdb.set_trace()
            # Branch plots for different sample sizes
            # for n_samples, label in [(5, "first_5"), (100, "first_100")]:
            #     mean_branch_plot(
            #         preds[:n_samples, :, :],
            #         labels[:n_samples, :, :],
            #         self._plot_path,
            #         f"mean_performance_plot_{label}",
            #     )
            #
            #     var_index = 0
            #     branch_plot(
            #         preds[:n_samples, :, :],
            #         labels[:n_samples, :, :],
            #         var_index,
            #         self._plot_path,
            #         f"sensor_{var_index}_branch_plot_{label}.png",
            #     )
            #
            mean_branch_plot(
                preds[:, :, :],
                labels[:, :, :],
                self._plot_path,
                f"mean_performance_plot.png",
            )

            var_index = 0
            branch_plot(
                preds[:, :, :],
                labels[:, :, :],
                var_index,
                self._plot_path,
                f"sensor_{var_index}_branch_plot.png",
            )

            self._logger.info("Test plots generated successfully")

        except Exception as e:
            self._logger.warning(f"Test plotting failed: {e}")

    # ==========================================================================
    # Training Functions
    # ==========================================================================

    def _forward_pass(self, h: int, X_df=None) -> Any:
        """
        Execute forward pass (prediction) through the model.

        Args:
            h: Prediction horizon
            X_df: Optional exogenous features DataFrame

        Returns:
            Model predictions
        """
        return self._forecast_fn(h, X_df, None)

    def _fit_model(self, train_data, X_df=None):
        """
        Fit the model on training data.

        Args:
            train_data: Training DataFrame
            X_df: Optional exogenous features DataFrame
        """
        if self._backend == "statsforecast":
            self.model.fit(train_data)
        else:
            self._fit_fn(train_data, X_df)

    # ==========================================================================
    # Optional Hook Methods
    # ==========================================================================

    def _on_train_start(self):
        """
        Hook called at the start of training.
        Override in subclasses for custom behavior.
        """
        pass

    def _on_train_end(self):
        """
        Hook called at the end of training.
        Override in subclasses for custom behavior.
        """
        pass

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    def train(self) -> Optional[Dict[str, float]]:
        """
        Main training method.

        Returns:
            Test metrics dictionary
        """
        self._logger.info("Start training!")
        self._logger.info(f"Tracking metrics: {self._get_metric_names()}")

        self._on_train_start()

        train_data = self._get_train_data()

        # Fit the model
        t1 = time.time()
        self._fit_model(train_data)
        t2 = time.time()

        self._log_training_info(t2 - t1)

        # Save model
        self.save_model(self._save_path)

        self._on_train_end()

        # Evaluate on test set
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

        preds = []
        labels = []

        test_data = self._get_data(mode)

        # Generate predictions
        predictions = self._forward_pass(self._pred_len)

        # Convert predictions to tensor
        out_batch = statsforecast_to_tensor(predictions, self._alias, True)
        label = statsforecast_to_tensor(test_data, "y", True)

        preds.append(out_batch.cpu())
        labels.append(label.cpu())

        # Concatenate
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        if mode == "val":
            preds, labels = self._prepare_test_data(preds, labels)
            return self._compute_validation_metrics(preds, labels)
        elif mode == "test":
            return self._compute_test_metrics(preds, labels)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ==========================================================================
    # Direct API Methods (for compatibility)
    # ==========================================================================

    def fit(self, train_df, X_df=None):
        """
        Fit the model (direct API).

        Args:
            train_df: Training DataFrame
            X_df: Optional exogenous features DataFrame

        Returns:
            Self
        """
        self._fit_fn(train_df, X_df)
        return self

    def forecast(self, h: int, X_df=None, level=None):
        """
        Generate forecasts (direct API).

        Args:
            h: Prediction horizon
            X_df: Optional exogenous features DataFrame
            level: Confidence level(s) for prediction intervals

        Returns:
            Forecast DataFrame
        """
        return self._forecast_fn(h, X_df, level)
