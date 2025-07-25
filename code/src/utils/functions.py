import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Compute Hessian Eigenvalues and vectors
from scipy.sparse.linalg import LinearOperator, eigsh
import h5py
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[2]))
from extra.loss-landscape.net_plotter import name_direction_file


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


# TODO: Experimental vibe coded calculation of hessian directions
# Check for correctness and whether it's not enough to just calculate
# min/max hessian Eigenvalues or some other metric (trace?) to measure
# steepness of loss surface minimum
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)  # Added DataLoader, TensorDataset
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np
from typing import Tuple, List, Callable, Any, Dict, Optional


# Helper functions for flattening and unflattening parameters (unchanged)
def _flatten_params(params_iterable: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Flattens a tuple of tensors into a single 1D tensor."""
    return torch.cat([p.reshape(-1) for p in params_iterable])


def _unflatten_params(
    flat_params_tensor: torch.Tensor,
    param_infos: List[Tuple[torch.Size, torch.dtype, torch.device]],
) -> Tuple[torch.Tensor, ...]:
    """Unflattens a 1D tensor back into a tuple of tensors based on param_infos."""
    unflattened_list = []
    current_idx = 0
    for shape, dtype, device in param_infos:
        num_elements = torch.Size(shape).numel()
        param_slice = flat_params_tensor[current_idx : current_idx + num_elements]
        unflattened_list.append(param_slice.reshape(shape).to(dtype).to(device))
        current_idx += num_elements
    return tuple(unflattened_list)


def compute_dominant_hessian_directions(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_dataloader: DataLoader,  # Changed from data_x, data_y
    tol: float = 1e-5,
    maxiter: Optional[int] = None,
    ncv: Optional[int] = None,
) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Computes the dominant Hessian directions for a given model and loss function,
    using a DataLoader for the dataset.

    This function implements Algorithm 1 from the paper "Visualizing high-dimensional
    loss landscapes with Hessian directions" (arXiv:2208.13219v2). It finds the
    eigenvectors corresponding to the largest-magnitude positive eigenvalue (maxeigval, maxeigvec)
    and the largest-magnitude negative eigenvalue (mineigval, mineigvec) of the Hessian
    of the average loss over the provided train_dataloader.

    Args:
        model: The PyTorch model (nn.Module).
        criterion: The loss function (e.g., nn.CrossEntropyLoss()).
        train_dataloader: PyTorch DataLoader providing (X, Y) batches for the entire
                          dataset over which the Hessian of the average loss is computed.
        tol: Tolerance for eigsh solver.
        maxiter: Maximum number of iterations for eigsh solver.
        ncv: The number of Lanczos vectors generated by eigsh.
             Default is min(N, 20) for k=1 if not specified.

    Returns:
        A tuple (max_eigval, max_eigvec, min_eigval, min_eigvec):
        - max_eigval (float): The largest positive eigenvalue.
        - max_eigvec (torch.Tensor): The corresponding eigenvector (1D, flat).
        - min_eigval (float): The most negative eigenvalue.
        - min_eigvec (torch.Tensor): The corresponding eigenvector (1D, flat).
        Eigenvectors are on the same device as the model parameters.
    """
    model.eval()

    model_params_for_grad = tuple(p for p in model.parameters() if p.requires_grad)
    if not model_params_for_grad:
        raise ValueError("No parameters with requires_grad=True found in the model.")

    param_infos = [(p.shape, p.dtype, p.device) for p in model_params_for_grad]
    param_device = param_infos[0][2]
    param_dtype = param_infos[0][1]

    N = sum(p.numel() for p in model_params_for_grad)

    # Memoize the average loss and its first gradients to avoid recomputing for each HVP call
    # if the model parameters (point of evaluation for Hessian) are fixed.
    # For HVP, the point theta^* is fixed.

    # Pre-calculate average loss and its gradients once.
    model.zero_grad(set_to_none=True)
    accumulated_loss_static = torch.tensor(0.0, device=param_device, dtype=param_dtype)
    num_batches_static = 0
    for batch_x_static, batch_y_static in train_dataloader:
        batch_x_static = batch_x_static.to(param_device)
        batch_y_static = batch_y_static.to(param_device)
        outputs_static = model(batch_x_static)
        loss_on_batch_static = criterion(outputs_static, batch_y_static)
        accumulated_loss_static += loss_on_batch_static
        num_batches_static += 1

    if num_batches_static == 0:
        raise ValueError("train_dataloader is empty.")

    final_loss_for_hvp_static = accumulated_loss_static / num_batches_static

    # Calculate first-order gradients (dL/dtheta) for the average loss
    # create_graph=True is crucial for allowing higher-order derivatives for HVP
    grads_static = torch.autograd.grad(
        final_loss_for_hvp_static, model_params_for_grad, create_graph=True
    )

    def _hvp_matvec_fn(v_np: np.ndarray) -> np.ndarray:
        """Computes Hessian-vector product H @ v using pre-computed static grads."""
        # model.zero_grad(set_to_none=True) # Not needed here as grads_static are from a fixed point

        v_torch_flat = torch.from_numpy(v_np).to(dtype=param_dtype, device=param_device)
        v_tuple = _unflatten_params(v_torch_flat, param_infos)

        # Dot product of static gradients and vector v: (dL/dtheta)^T @ v
        grad_v_dot_product = sum(
            torch.sum(g_static * vt_e) for g_static, vt_e in zip(grads_static, v_tuple)
        )

        # Second-order gradients (gradient of the dot product w.r.t. parameters) -> H @ v
        # create_graph=False as this is the final gradient for HVP
        hvp_tuple = torch.autograd.grad(
            grad_v_dot_product,
            model_params_for_grad,
            retain_graph=True,  # <<< THIS IS THE KEY CHANGE
            create_graph=False,
        )

        hvp_flat = _flatten_params(hvp_tuple)
        return hvp_flat.cpu().numpy()

    L1 = LinearOperator(shape=(N, N), matvec=_hvp_matvec_fn, dtype=np.float32)

    try:
        eigval1_mag_np, eigvec1_np = eigsh(
            L1, k=1, which="LM", tol=tol, maxiter=maxiter, ncv=ncv
        )
    except Exception as e:
        raise RuntimeError(
            f"eigsh for L1 failed: {e}. Consider adjusting tol, maxiter, or ncv, or check HVP computation."
        )

    eigval1 = float(eigval1_mag_np[0])
    eigvec1_flat = torch.from_numpy(eigvec1_np[:, 0]).to(
        dtype=param_dtype, device=param_device
    )

    def _shifted_hvp_matvec_fn(v_np: np.ndarray) -> np.ndarray:
        hvp_val_np = _hvp_matvec_fn(v_np)
        shifted_hvp_val_np = hvp_val_np - eigval1 * v_np
        return shifted_hvp_val_np

    L2 = LinearOperator(shape=(N, N), matvec=_shifted_hvp_matvec_fn, dtype=np.float32)

    try:
        eigval2_shifted_mag_np, eigvec2_np = eigsh(
            L2, k=1, which="LM", tol=tol, maxiter=maxiter, ncv=ncv
        )
    except Exception as e:
        raise RuntimeError(
            f"eigsh for L2 failed: {e}. Consider adjusting tol, maxiter, or ncv, or check HVP computation."
        )

    eigval2_shifted = float(eigval2_shifted_mag_np[0])
    eigvec2_flat = torch.from_numpy(eigvec2_np[:, 0]).to(
        dtype=param_dtype, device=param_device
    )

    eigval2 = eigval2_shifted + eigval1

    if eigval1 >= 0:
        max_eigval = eigval1
        max_eigvec = eigvec1_flat
        min_eigval = eigval2
        min_eigvec = eigvec2_flat
    else:
        min_eigval = eigval1
        min_eigvec = eigvec1_flat
        max_eigval = eigval2
        max_eigvec = eigvec2_flat

    if min_eigval > max_eigval:
        max_eigval, min_eigval = min_eigval, max_eigval
        max_eigvec, min_eigvec = min_eigvec, max_eigvec

    return max_eigval, max_eigvec, min_eigval, min_eigvec


def unflatten_to_weights(flat_params, model):
    """
    Converts a flat 1D tensor of parameters back into a list of tensors
    with the same shapes as the model's parameters.

    Args:
        flat_params (torch.Tensor): A 1D tensor containing all model parameters.
        model (torch.nn.Module): The model with the target parameter shapes.

    Returns:
        list[torch.Tensor]: A list of tensors with shapes matching the model's parameters.
    """
    unflattened = []
    current_idx = 0
    for param in model.parameters():
        num_elements = param.numel()
        # Slice the flat tensor to get the elements for the current parameter
        param_slice = flat_params[current_idx : current_idx + num_elements]
        # Reshape the slice to the correct dimensions and add to the list
        unflattened.append(param_slice.view(param.size()))
        current_idx += num_elements
    return unflattened


# --- Main function to save eigenvectors ---
def save_eigenvectors_to_hdf5(args, net, max_evec, min_evec, output_dir=""):
    """
    Converts two 1D PyTorch eigenvector tensors into a loss-landscape-compatible
    HDF5 file by reshaping them to match the model's parameter structure.

    Args:
        args (object): argparse.Namespace with arguments for name_direction_file.
        net (torch.nn.Module): The model to which the directions apply.
        max_evec (torch.Tensor): The first direction vector (1D).
        min_evec (torch.Tensor): The second direction vector (1D).
        output_dir (str, optional): Directory to save the file in.
    """
    # Basic validation
    if not isinstance(max_evec, torch.Tensor) or not isinstance(min_evec, torch.Tensor):
        raise TypeError("max_evec and min_evec must be PyTorch tensors.")
    if max_evec.ndim != 1 or min_evec.ndim != 1:
        raise ValueError("Eigenvectors must be 1-dimensional tensors.")

    # --- CORE LOGIC CHANGE ---
    # Un-flatten the 1D eigenvectors into a list of tensors matching the model's parameter shapes.
    print("Reshaping flat eigenvectors to match model parameter shapes...")
    xdirection = unflatten_to_weights(max_evec, net)
    ydirection = unflatten_to_weights(min_evec, net)
    # --- END OF CHANGE ---

    # Generate the filename using the same utility as the plotting script
    h5_filename = name_direction_file(args)

    if output_dir:
        output_path = SCRIPT_DIR / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        final_filepath = output_path / Path(h5_filename).name
    else:
        h5_path = SCRIPT_DIR / h5_filename
        leaf_dir = h5_path.parent
        if leaf_dir:
            leaf_dir.mkdir(parents=True, exist_ok=True)
        final_filepath = h5_path
        print(f"Saving compatible HDF5 file to: {final_filepath}")

    # --- SAVING LOGIC CHANGE ---
    # Create and write to HDF5 file using the compatible utility and correct names
    with h5py.File(final_filepath, "w") as f:
        h5_util.write_list(f, "xdirection", xdirection)
        h5_util.write_list(f, "ydirection", ydirection)
    # --- END OF CHANGE ---

    return final_filepath
