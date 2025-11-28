import torch
import numpy as np
from typing import Dict, List
from src.base.torch_engine import TorchEngine
from torchmetrics.regression import (
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanAbsoluteError,
)


class SAMFormer_Engine(TorchEngine):
    """
    SAMFormer pytorch trainer implemented in the sklearn fashion
    """

    def __init__(
        self,
        batch_size=1,
        num_channels=1,
        pred_len=1,
        no_sam=True,
        use_revin=True,
        gsam=False,
        plot_attention=False,
        **args,
    ):
        super().__init__(**args)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.no_sam = no_sam
        self.use_revin = use_revin
        self.gsam = gsam
        self.plot_attention = plot_attention

        # Attention tracking
        self._attention_patterns = []

    # ==================== Metric Configuration ====================

    # def _get_metric_names(self) -> List[str]:
    #     """Specify which metrics to track."""
    #     return self._tracked_metrics

    # def _get_loss_name(self) -> str:
    #     """Return the name of the loss being used."""
    #     # Infer from loss function
    #     loss_name = self._loss_fn.__class__.__name__.lower()
    #     if "mse" in loss_name:
    #         return "mse"
    #     elif "mae" in loss_name or "l1" in loss_name:
    #         return "mae"
    #     return super()._get_loss_name()

    # ==================== Required Overrides ====================
    #
    # TODO: keep this and maybe add it to an example model config later
    # to  show how new models with necessary overrides could be added

    # def _prepare_batch(self, batch):
    #     """SAMFormer-specific batch preparation."""
    #     x_batch, y_batch = batch
    #     return x_batch, y_batch
    #
    # def _forward_pass(self, x_batch, training=True):
    #     """SAMFormer forward pass with attention tracking."""
    #     # TODO: don't know if that training bool will work like that
    #     # TODO: maybe remove flatten output
    #     out = self.model(x_batch, False)
    #
    #     if self.plot_attention and training:
    #         self._attention_patterns.append(self.model.attention_pattern)
    #
    #     return out

    # def _compute_metrics(
    #     self, pred: torch.Tensor, target: torch.Tensor
    # ) -> Dict[str, float]:
    #     """Compute all requested metrics."""
    #     metrics = {}
    #
    #     # Compute loss
    #     loss_name = self._get_loss_name()
    #     if loss_name in self._tracked_metrics:
    #         metrics[loss_name] = self._compute_loss(pred, target).item()
    #
    #     # Compute additional metrics
    #     for metric_name, calculator in self._metric_calculators.items():
    #         if metric_name in self._tracked_metrics:
    #             metrics[metric_name] = calculator(pred, target).item()
    #
    #     return metrics

    # ==================== Custom Optimizer Step ====================

    def _optimizer_step(self, loss, x_batch, y_batch):
        """Custom optimizer step for SAM/GSAM."""
        if self.no_sam:
            super()._optimizer_step(loss, x_batch, y_batch)
        elif self.gsam:
            loss.backward()
            self._optimizer.set_closure(self._loss_fn, x_batch, y_batch)
            out_batch, loss = self._optimizer.step()
            self._lr_scheduler.step()
            self._optimizer.update_rho_t()
        else:
            # SAM optimizer
            loss.backward()
            self._optimizer.first_step(zero_grad=True)

            # TODO: maybe remove flatten output
            out_batch = self.model(x_batch, False)
            if self.plot_attention:
                self._attention_patterns.append(self.model.attention_pattern)

            loss = self._compute_loss(out_batch, y_batch)

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
            self._optimizer.second_step(zero_grad=True)

    # ==================== Custom Hooks ====================

    def _on_epoch_start(self, epoch):
        """Reset attention patterns at epoch start."""
        self._attention_patterns = []

    def _on_epoch_end(self, epoch):
        """Plot attention patterns at epoch end."""
        if self.plot_attention and epoch % 1 == 0:
            self._plot_attention_patterns(epoch)

    def _plot_attention_patterns(self, epoch):
        """Plot attention patterns for this epoch."""
        try:
            from src.utils.model_utils import (
                plot_samformer_attention_mean,
                plot_samformer_attention_mean_stats,
                plot_samformer_attention_variance,
            )

            attention_mean_path = self._plot_path / "attention" / "mean"
            attention_var_path = self._plot_path / "attention" / "variance"
            attention_mean_path.mkdir(parents=True, exist_ok=True)
            attention_var_path.mkdir(parents=True, exist_ok=True)

            plot_samformer_attention_mean(
                self._attention_patterns, epoch + 1, attention_mean_path
            )
            plot_samformer_attention_mean_stats(
                self._attention_patterns, epoch + 1, attention_mean_path
            )
            plot_samformer_attention_variance(
                self._attention_patterns, epoch + 1, attention_var_path
            )

            # self._logger.info(f"Attention plots saved for epoch {epoch + 1}")
        except Exception as e:
            self._logger.warning(f"Failed to plot attention: {e}")

    # ==================== Custom LR Handling ====================

    def _get_current_lr(self):
        """Get current LR (handles GSAM special case)."""
        if self.gsam:
            return self._lr_scheduler._last_lr[0]
        return super()._get_current_lr()

    def _step_scheduler(self):
        """Step scheduler (GSAM handles this differently)."""
        if not self.gsam:
            super()._step_scheduler()

    # ==================== Custom Test Evaluation ====================

    def _compute_test_metrics(self, preds, labels) -> Dict[str, float]:
        """SAMFormer-specific test evaluation with per-horizon metrics."""
        # Reshape predictions

        preds = preds.reshape(preds.shape[0], self.num_channels, self.pred_len)
        labels = labels.reshape(labels.shape[0], self.num_channels, self.pred_len)
        preds = torch.permute(preds, (0, 2, 1)).contiguous()
        labels = torch.permute(labels, (0, 2, 1)).contiguous()

        # Overall metrics
        # TODO: check what this actually computes

        overall_metrics = super()._compute_test_metrics(preds, labels)

        # Per-horizon metrics
        horizon_metrics = {metric: [] for metric in self._metric_calculators}

        for i in range(self.model.horizon):
            horizon_preds = preds[:, i, :].contiguous()
            horizon_labels = labels[:, i, :].contiguous()

            # Compute metrics for this horizon
            horizon_result = self._compute_metrics(horizon_preds, horizon_labels)

            # Log horizon metrics
            metric_str = ", ".join(
                [f"{k.upper()}: {v:.4f}" for k, v in horizon_result.items()]
            )
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

        # Plotting
        if self._enable_plotting:
            self._plot_test_results(preds, labels, horizon_metrics)

        return overall_metrics

    # TODO: what do we actually need from here?
    # also make plots look a bit nicer
    def _plot_test_results(self, preds, labels, horizon_metrics):
        """Generate test result plots."""
        try:
            from src.utils.plotting import (
                plot_mean_per_day,
                mean_branch_plot,
                branch_plot,
            )

            # Use primary metric for plotting
            primary_metric_values = horizon_metrics.get(
                self._primary_metric, horizon_metrics.get(self._get_loss_name(), [])
            )

            # Mean predictions per horizon
            per_day_preds = [preds[:, i, :].mean() for i in range(self.model.horizon)]
            per_day_labels = [labels[:, i, :].mean() for i in range(self.model.horizon)]

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
