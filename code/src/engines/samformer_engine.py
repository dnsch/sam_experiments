import torch
import numpy as np
from typing import Dict, List
from src.base.torch_engine import TorchEngine


class SAMFormer_Engine(TorchEngine):
    """
    SAMFormer pytorch trainer with custom hooks to capture attention patterns.
    Rest is done in TorchEngine.
    """

    def __init__(
        self,
        plot_attention=False,
        **args,
    ):
        super().__init__(**args)
        self.plot_attention = plot_attention

        # Attention tracking
        self._attention_patterns = []

    # ==================== Custom Hooks ====================

    def _on_epoch_start(self, epoch):
        """Reset attention patterns at epoch start."""
        if self.plot_attention:
            self._attention_patterns = []

    def _on_forward_pass(self):
        if self.plot_attention:
            self._attention_patterns.append(self.model.attention_pattern)

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

            # TODO: check this function, might remove it
            plot_samformer_attention_mean_stats(
                self._attention_patterns, epoch + 1, attention_mean_path
            )
            plot_samformer_attention_variance(
                self._attention_patterns, epoch + 1, attention_var_path
            )

            # self._logger.info(f"Attention plots saved for epoch {epoch + 1}")
        except Exception as e:
            self._logger.warning(f"Failed to plot attention: {e}")
