import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ============================================================

# Plots

# ============================================================


# Plotting functions
def plot_train_val_loss(total_train_loss, loss, loss_string, epochs, plot_path):
    plot_dir = Path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, epochs + 1),
        total_train_loss,
        label=f"Train {loss_string}",
        color="tab:blue",
    )
    plt.plot(
        range(1, epochs + 1),
        loss,
        label=f"Validation {loss_string}",
        linestyle="--",
        color="tab:orange",
    )

    # Adding title and labels
    plt.title("train_val_loss")
    plt.xlabel("Epochs")
    plt.ylabel(f"{loss_string}")

    # Show legend
    plt.legend()

    # Save the plot with the name including the number of epochs
    plot_filename = plot_dir / "statistics" / f"train_val_loss_{epochs}.png"
    plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)

    # Close the plot to free memory
    plt.close()


def plot_loss_metric(input_array, epochs, variable_name, color, plot_path):
    epoch_range = np.arange(1, epochs + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(epoch_range, input_array, label=variable_name, color=color)

    plt.title(f"{variable_name}", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss metric", fontsize=12)

    plt.grid(True)

    plt.legend()
    plt.tight_layout()

    plot_dir = Path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_filename = plot_dir / "statistics" / f"{variable_name}.png"
    plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)
    plt.close()


def plot_stats(
    total_train_loss,
    total_train_mape,
    total_train_rmse,
    total_valid_loss,
    total_valid_mape,
    total_valid_rmse,
    last_epoch,
    timeout,
    plot_path,
):
    if timeout:
        # plotting:
        plot_train_val_loss(
            total_train_loss,
            total_valid_loss,
            "mae",
            last_epoch,
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_train_loss,
            last_epoch,
            "train_mae",
            color="tab:blue",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_train_mape,
            last_epoch,
            "train_mape",
            color="tab:orange",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_train_rmse,
            last_epoch,
            "train_rmse",
            color="tab:green",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_valid_loss,
            last_epoch,
            "validation_mae",
            color="tab:blue",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_valid_mape,
            last_epoch,
            "validation_mape",
            color="tab:orange",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_valid_rmse,
            last_epoch,
            "validation_rmse",
            color="tab:green",
            plot_path=plot_path,
        )

    else:
        # plotting:
        plot_train_val_loss(
            total_train_loss,
            total_valid_loss,
            "masked_mae",
            last_epoch,
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_train_loss,
            last_epoch,
            "train_mae",
            color="tab:blue",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_train_mape,
            last_epoch,
            "train_mape",
            color="tab:orange",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_train_rmse,
            last_epoch,
            "train_rmse",
            color="tab:green",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_valid_loss,
            last_epoch,
            "validation_mae",
            color="tab:blue",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_valid_mape,
            last_epoch,
            "validation_mape",
            color="tab:orange",
            plot_path=plot_path,
        )
        plot_loss_metric(
            total_valid_rmse,
            last_epoch,
            "validation_rmse",
            color="tab:green",
            plot_path=plot_path,
        )


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
