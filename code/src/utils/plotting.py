from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure matplotlib for better-looking plots

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))

from src.utils.metrics import TrainingMetrics

# ============================================================

# Plots

# ============================================================


def plot_loss_metric(
    train_values: List[float],
    val_values: List[float],
    metric_name: str,
    epochs: int,
    plot_path: Union[str, Path] = ".",
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot training and validation curves for a single metric.

    Args:
        train_values: List of training metric values per epoch
        val_values: List of validation metric values per epoch
        metric_name: Name of the metric (e.g., 'mae', 'rmse', 'mape')
        epochs: Total number of epochs
        plot_path: Base directory for saving plots
        figsize: Figure size as (width, height)
        show_best: Whether to mark the best validation value on the plot
    """
    plot_dir = Path(plot_path) / "statistics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    epoch_range = np.arange(1, epochs + 1)

    # Plot training curve
    ax.plot(
        epoch_range[: len(train_values)],
        train_values,
        label="Training",
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=max(1, epochs // 20),
        alpha=0.9,
    )

    # Plot validation curve
    ax.plot(
        epoch_range[: len(val_values)],
        val_values,
        label="Validation",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=4,
        markevery=max(1, epochs // 20),
        alpha=0.9,
    )

    # Styling
    ax.set_xlabel("Epoch", fontweight="medium")
    ax.set_ylabel("Value", fontweight="medium")

    # Format title
    title = metric_name.replace("_", " ").upper()
    ax.set_title(title, fontweight="bold", pad=15)

    # Grid
    ax.grid(True, linestyle=":", alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="best", framealpha=0.95, edgecolor="gray")

    # Tight layout
    plt.tight_layout()

    # Save
    plot_filename = plot_dir / f"train_val_{metric_name}_{epochs}epochs.png"
    plt.savefig(plot_filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stats(
    metrics: TrainingMetrics,
    last_epoch: int,
    plot_path: Path,
):
    """
    Generic plotting function for training metrics.

    Args:
        metrics: TrainingMetrics object containing all tracked metrics
        loss_name: Name of the loss metric used for optimization
        last_epoch: Last epoch number for x-axis
        plot_path: Path to save plots
    """
    all_metric_names = sorted(metrics.get_all_metric_names())

    # Plot combined train/val curves for all metrics
    for metric_name in all_metric_names:
        train_values = metrics.get_train_metric(metric_name)
        val_values = metrics.get_val_metric(metric_name)

        if train_values and val_values:
            plot_loss_metric(
                train_values=train_values,
                val_values=val_values,
                metric_name=metric_name,
                epochs=last_epoch,
                plot_path=plot_path,
            )


def plot_all_metrics_combined(
    metrics: TrainingMetrics,
    last_epoch: int,
    plot_path: Path,
    figsize: Optional[Tuple[int, int]] = None,
):
    """
    Plot all metrics stacked vertically in a single column.

    Args:
        metrics: TrainingMetrics object containing all tracked metrics
        last_epoch: Last epoch number
        plot_path: Path to save the plot
        figsize: Optional figure size (width, height). If None, auto-calculated.
    """

    all_metrics = sorted(metrics.get_all_metric_names())
    n_metrics = len(all_metrics)

    if n_metrics == 0:
        return

    # Vertical stacking: 1 column, n rows
    n_cols = 1
    n_rows = n_metrics

    # Auto-calculate figure size if not provided
    # Single column, so narrow width; height scales with number of metrics
    if figsize is None:
        figsize = (10, 4 * n_metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single metric case
    axes = [axes] if n_metrics == 1 else axes.flatten()

    epochs = list(range(1, last_epoch + 1))

    for idx, metric_name in enumerate(all_metrics):
        ax = axes[idx]

        train_vals = metrics.get_train_metric(metric_name)
        val_vals = metrics.get_val_metric(metric_name)

        if train_vals:
            ax.plot(
                epochs[: len(train_vals)],
                train_vals,
                label="Train",
                color="#2E86AB",
                linewidth=2,
                marker="o",
                markersize=4,
                markevery=max(1, last_epoch // 20),
                alpha=0.9,
            )
        if val_vals:
            ax.plot(
                epochs[: len(val_vals)],
                val_vals,
                label="Validation",
                color="#F77F00",
                linewidth=2,
                linestyle="--",
                marker="s",
                markersize=4,
                markevery=max(1, last_epoch // 20),
                alpha=0.9,
            )

        ax.set_title(f"{metric_name.upper()}", fontweight="bold", pad=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.upper())
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, linestyle=":", alpha=0.6, linewidth=0.8)
        ax.set_axisbelow(True)

    # Add overall title
    fig.suptitle(
        "Training Metrics Overview",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(
        plot_path / f"all_metrics_combined_{last_epoch}ep.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def branch_plot(preds, labels, var_index, plot_path, title):
    # Select the specified var_index
    preds_sensor = preds[:, :, var_index]
    labels_sensor = labels[:, :, var_index]
    # Define a set of distinguishable colors
    colors = ["orange", "green", "red", "purple", "brown", "pink"]

    # Create a plot
    plt.figure(figsize=(15, 7), dpi=500)

    for timepoint_idx in range(labels_sensor.shape[0]):
        labels_entry = labels_sensor[timepoint_idx, :]
        preds_entry = preds_sensor[timepoint_idx, :]

        plt.plot(
            range(timepoint_idx, len(labels_entry) + timepoint_idx),
            labels_entry,
            color="blue",
            linewidth=0.5,
            label="Ground Truth" if timepoint_idx == 0 else "",
        )
        plt.plot(
            range(timepoint_idx, len(preds_entry) + timepoint_idx),
            preds_entry,
            color=colors[timepoint_idx % len(colors)],
            linewidth=0.5,
            label=f"Prediction {timepoint_idx + 1}" if timepoint_idx == 0 else "",
        )

    # Adding labels, title, and legend
    plt.title(f"Ground Truth and Predictions for Sensor {var_index}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid()
    # Create custom legend entries
    legend_elements = [
        Line2D([0], [0], color="blue", lw=1, label="Ground Truth"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1,
            label=f"Predictions starting from first {preds.shape[0]} timesteps",
            linestyle="-",
            marker=None,
        ),
    ]

    plt.legend(handles=legend_elements, loc="upper right")

    # Save the plot
    plot_dir = Path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_dir / "statistics" / title
    plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)


def mean_branch_plot(preds, labels, plot_path, title):
    # Select the specified sensor
    preds_sensor = preds.mean(dim=2)
    labels_sensor = labels.mean(dim=2)
    # Define a set of distinguishable colors
    colors = ["orange", "green", "red", "purple", "brown", "pink"]

    # Create a plot
    plt.figure(figsize=(15, 7), dpi=500)

    # Plot all sequences for the specified sensor

    for timepoint_idx in range(labels_sensor.shape[0]):
        labels_entry = labels_sensor[timepoint_idx, :]
        preds_entry = preds_sensor[timepoint_idx, :]

        plt.plot(
            range(timepoint_idx, len(labels_entry) + timepoint_idx),
            labels_entry,
            color="blue",
            linewidth=0.5,
            label="Ground Truth" if timepoint_idx == 0 else "",
        )
        plt.plot(
            range(timepoint_idx, len(preds_entry) + timepoint_idx),
            preds_entry,
            color=colors[timepoint_idx % len(colors)],
            linewidth=0.5,
            label=f"Prediction {timepoint_idx + 1}" if timepoint_idx == 0 else "",
        )

    # Adding labels, title, and legend
    plt.title(f"Mean Ground Truth and Predictions for all Sensors")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid()
    # Create custom legend entries
    legend_elements = [
        Line2D([0], [0], color="blue", lw=1, label="Ground Truth"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1,
            label=f"Predictions starting from first {preds.shape[0]} timesteps",
            linestyle="-",
            marker=None,
        ),
    ]

    plt.legend(handles=legend_elements, loc="upper right")

    # Save the plot
    plot_dir = Path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_dir / "statistics" / title
    plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)


def plot_mean_per_day(mean_per_day_preds, mean_per_day_labels, plot_path, title):
    # Convert tensors to lists
    preds = [p.item() for p in mean_per_day_preds]
    labels = [l.item() for l in mean_per_day_labels]

    # Create a plot
    plt.figure(figsize=(15, 7), dpi=500)

    # Plot predictions and labels
    plt.plot(
        range(len(labels)), labels, color="blue", linewidth=0.5, label="Ground Truth"
    )
    plt.plot(
        range(len(preds)), preds, color="orange", linewidth=0.5, label="Predictions"
    )

    # Adding labels, title, and legend
    plt.title("Timestep Mean Labels and Predictions for all Sensors")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid()
    plt.legend(loc="upper right")
    # Save the plot
    plot_dir = Path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_dir / "statistics" / title
    plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)
