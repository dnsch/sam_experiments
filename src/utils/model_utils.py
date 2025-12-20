import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
# ============================================================

# Models

# ============================================================


# Samformer Functions
# TODO: this should not be reserved to SAMFormer
# put it somewhere s.t. it becomes clear that it can be used with other
# architectures aswell
def load_optimizer(model, args, logger):
    """
    Loads the optimizer based on the choice provided in args.
    """
    try:
        optimizer_class = getattr(torch.optim, args.optimizer)

        if args.sam:
            logger.info(f"Optimizer class: {optimizer_class}")
            return optimizer_class
        elif args.gsam:
            logger.info(f"Optimizer class: {optimizer_class}")
            optimizer = optimizer_class(
                model.parameters(), lr=args.lrate, weight_decay=args.wdecay
            )
            logger.info(optimizer)
            return optimizer
        else:
            # no Sharpness Aware Minimization
            optimizer = optimizer_class(
                model.parameters(), lr=args.lrate, weight_decay=args.wdecay
            )
            logger.info(optimizer)
            return optimizer
    except AttributeError:
        raise ValueError(f"Optimizer '{args.optimizer}' not found in torch.optim.")


# Plot attention_matrix


class AttentionExtractor:
    """Extract and store attention weights during forward pass"""

    def __init__(self):
        self.keys = None
        self.queries = None
        self.attention_weights = None

    def extract_attention(self, model, x):
        """
        Extract attention weights from SAMFormer model

        Args:
            model: SAMFormer model
            x: Input tensor

        Returns:
            attention_weights: Tensor of shape (batch, seq_len, seq_len)
        """
        with torch.no_grad():
            # Get keys and queries
            # Assuming x shape is (batch, seq_len, features)
            keys = model.compute_keys(x)  # (batch, seq_len, 16)
            queries = model.compute_queries(x)  # (batch, seq_len, 16)

            # Compute attention: Q * K^T / sqrt(d_k)
            d_k = keys.size(-1)
            attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (
                d_k**0.5
            )

            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_scores, dim=-1)

        return attention_weights


# TODO: change name, would make sense to extract attention patterns for all
# transformer architectures
def plot_samformer_attention_mean(
    attention_patterns_per_epoch,
    epoch,
    save_path,
    max_display_size=100,
):
    """
    Plot attention matrix for SAMFormer model

    Args:
        model: SAMFormer model
        x_batch: Input batch tensor
        epoch: Current epoch number
        save_path: Path to save the plot
        sample_idx: Which sample from batch to visualize
        max_display_size: Maximum sequence length to display
    """
    save_path.mkdir(parents=True, exist_ok=True)

    attention_patterns_mean = (
        attention_patterns_per_epoch[-2].mean(dim=0).detach().cpu().numpy()
    )

    # Limit display size for readability
    if attention_patterns_mean.shape[0] > max_display_size:
        attention_patterns_mean = attention_patterns_mean[
            :max_display_size, :max_display_size
        ]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(attention_patterns_mean, vmin=0, vmax=1, cmap="Reds", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", rotation=270, labelpad=20)

    # Set labels
    ax.set_title(f"SAMFormer Attention Matrix - Epoch {epoch}", fontsize=14, pad=20)
    ax.set_xlabel("Key Position (Time Step)", fontsize=12)
    ax.set_ylabel("Query Position (Time Step)", fontsize=12)

    # Add grid for better readability
    ax.grid(False)

    # Save figure
    filename = f"samformer_attention_epoch_{epoch:03d}.png"
    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=150, bbox_inches="tight")
    plt.close()

    return attention_patterns_mean


def plot_samformer_attention_mean_stats(attention_patterns_per_epoch, epoch, save_path):
    """
    Plot attention statistics (mean attention per position, entropy, etc.)

    Args:
        model: SAMFormer model
        x_batch: Input batch tensor
        epoch: Current epoch number
        save_path: Path to save the plot
        sample_idx: Which sample from batch to visualize
    """
    save_path.mkdir(parents=True, exist_ok=True)

    attention_patterns_mean = (
        torch.cat(attention_patterns_per_epoch, dim=0)
        .mean(dim=0)
        .detach()
        .cpu()
        .numpy()
    )

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Attention heatmap
    ax = axes[0, 0]
    im = ax.imshow(attention_patterns_mean, cmap="viridis", aspect="auto")
    ax.set_title("Attention Matrix")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax)

    # 2. Mean attention received (column-wise mean)
    ax = axes[0, 1]
    mean_attention_received = attention_patterns_mean.mean(axis=0)
    ax.plot(mean_attention_received, linewidth=2)
    ax.set_title("Average Attention Received per Position")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean Attention Weight")
    ax.grid(True, alpha=0.3)

    # 3. Mean attention given (row-wise mean)
    ax = axes[1, 0]
    mean_attention_given = attention_patterns_mean.mean(axis=1)
    ax.plot(mean_attention_given, linewidth=2, color="orange")
    ax.set_title("Average Attention Given per Position")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean Attention Weight")
    ax.grid(True, alpha=0.3)

    # 4. Attention entropy (measure of focus)
    ax = axes[1, 1]
    # Compute entropy for each query position
    epsilon = 1e-10
    entropy = -np.sum(
        attention_patterns_mean * np.log(attention_patterns_mean + epsilon), axis=1
    )
    ax.plot(entropy, linewidth=2, color="green")
    ax.set_title("Attention Entropy per Query Position")
    ax.set_xlabel("Query Position")
    ax.set_ylabel("Entropy (higher = more distributed)")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"SAMFormer Attention Analysis - Epoch {epoch}", fontsize=16, y=1.00)
    plt.tight_layout()

    filename = f"samformer_attention_stats_epoch_{epoch:03d}.png"
    plt.savefig(save_path / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_samformer_attention_variance(
    attention_patterns_per_epoch,
    epoch,
    save_path,
    max_display_size=100,
):
    """
    Plot attention matrix for SAMFormer model

    Args:
        model: SAMFormer model
        x_batch: Input batch tensor
        epoch: Current epoch number
        save_path: Path to save the plot
        sample_idx: Which sample from batch to visualize
        max_display_size: Maximum sequence length to display
    """
    save_path.mkdir(parents=True, exist_ok=True)

    attention_patterns_var = (
        torch.cat(attention_patterns_per_epoch, dim=0)
        .var(dim=0)  # unbiased=True by default
        .detach()
        .cpu()
        .numpy()
    )

    # Limit display size for readability
    if attention_patterns_var.shape[0] > max_display_size:
        attention_patterns_var = attention_patterns_var[
            :max_display_size, :max_display_size
        ]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(attention_patterns_var, cmap="Reds", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", rotation=270, labelpad=20)

    # Set labels
    ax.set_title(f"SAMFormer Attention Matrix - Epoch {epoch}", fontsize=14, pad=20)
    ax.set_xlabel("Key Position (Time Step)", fontsize=12)
    ax.set_ylabel("Query Position (Time Step)", fontsize=12)

    # Add grid for better readability
    ax.grid(False)

    # Save figure
    filename = f"samformer_attention_epoch_{epoch:03d}.png"
    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=150, bbox_inches="tight")
    plt.close()

    return attention_patterns_var


# StatsForecast functions
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.models import AutoMFLES
import warnings


def _get_param_with_warning(args, param_name, default_value, arg_name=None):
    """
    Extract parameter from args or use default with warning.

    Args:
        args: Argument object containing model_name and parameters
        param_name: Parameter name for the model
        default_value: Default value if parameter not in args
        arg_name: Attribute name in args if different from param_name

    Returns:
        Parameter value from args or default

    Warns:
        UserWarning: If parameter not found in args, indicating which
            default value is being used
    """
    arg_name = arg_name or param_name

    if hasattr(args, arg_name):
        return getattr(args, arg_name)
    else:
        warnings.warn(
            f"[{args.model_name.upper()}] Parameter '{arg_name}' not found in args. "
            f"Using default value: {default_value}",
            UserWarning,
        )
        return default_value


def get_nixtla_model(args):
    """
    Create Nixtla StatsForecast model from arguments.

    Parameters not found in args will use model defaults, raising a warning
    that specifies which defaults were applied.

    Args:
        args: Argument object with 'model_name' and model-specific parameters

    Returns:
        Initialized Nixtla model instance

    Raises:
        ValueError: If args.model_name is not supported
    """
    model_name_lower = args.model_name.lower()

    if model_name_lower == "autoarima":
        # Extract parameters with defaults and warnings
        seasonal_periods = _get_param_with_warning(
            args, "season_length", 24, "seasonal_periods"
        )
        max_p = _get_param_with_warning(args, "max_p", 3)
        max_q = _get_param_with_warning(args, "max_q", 3)
        max_P = _get_param_with_warning(args, "max_P", 2)
        max_Q = _get_param_with_warning(args, "max_Q", 2)
        max_d = _get_param_with_warning(args, "max_d", 2)
        max_D = _get_param_with_warning(args, "max_D", 1)
        seasonal = _get_param_with_warning(args, "seasonal", True)
        auto_arima = _get_param_with_warning(args, "auto_arima", True)

        # Handle d and D based on auto_arima
        if auto_arima:
            d_val = None
            D_val = None
        else:
            d_val = _get_param_with_warning(args, "d", 1)
            D_val = _get_param_with_warning(args, "D", 1)

        model = AutoARIMA(
            season_length=seasonal_periods,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            max_d=max_d,
            max_D=max_D,
            d=d_val,
            D=D_val,
            stepwise=True,
            approximation=True,
            seasonal=seasonal,
            ic="aic",
        )

    elif model_name_lower == "automfles":
        # Extract parameters with defaults and warnings
        season_length = _get_param_with_warning(args, "season_length", 24)

        # Handle season_length as list
        if isinstance(season_length, list):
            if len(season_length) == 0:
                season_length = None
            elif len(season_length) == 1:
                season_length = season_length[0]

        # Use horizon as test_size if available, otherwise default
        if hasattr(args, "horizon"):
            test_size = args.horizon
        else:
            test_size = _get_param_with_warning(args, "test_size", 96)

        n_windows = _get_param_with_warning(args, "n_windows", 2)
        metric = _get_param_with_warning(args, "metric", "smape")
        verbose = _get_param_with_warning(args, "verbose", False)
        prediction_intervals = _get_param_with_warning(
            args, "prediction_intervals", None
        )

        model = AutoMFLES(
            test_size=test_size,
            season_length=season_length,
            n_windows=n_windows,
            metric=metric,
            verbose=verbose,
            prediction_intervals=prediction_intervals,
        )

    else:
        raise ValueError(
            f"Unknown model: '{args.model_name}'. Supported models: 'arima', 'mfles'"
        )

    return model


def get_statsforecast_model(args, freq="H"):
    """
    Create StatsForecast instance with configured model.

    Args:
        args: Argument object with model configuration
        freq: Time series frequency (default: 'H' for hourly)

    Returns:
        StatsForecast instance with the configured model
    """

    model = get_nixtla_model(args)
    n_cores = _get_param_with_warning(args, "n_cores", -1)

    sf = StatsForecast(
        models=[model],
        freq=freq,
        n_jobs=n_cores,
    )

    return sf


def statsforecast_to_tensor(df, variable_name, flatten=False):
    """
    Convert statsforecast format DataFrame to PyTorch tensor.

    Args:
        df: StatsForecast DataFrame with columns ['unique_id', 'ds', variable_name]
        flatten: bool, if True returns shape [1, total_length],
                 if False returns shape [n_series, n_timesteps_per_series]

    Returns:
        torch.Tensor: Tensor in requested format
    """
    # Sort by unique_id and ds to ensure proper ordering
    df_sorted = df.sort_values(["unique_id", "ds"])

    # Convert values to tensor
    values = df_sorted[variable_name].values
    tensor_data = torch.tensor(values, dtype=torch.float32)

    if flatten:
        # Return flattened tensor with shape [1, total_length]
        return tensor_data.unsqueeze(0)
    else:
        # Calculate dimensions for reshaping
        unique_ids = df_sorted["unique_id"].unique()
        n_series = len(unique_ids)
        n_timesteps = len(df_sorted) // n_series

        # Reshape to [n_series, n_timesteps_per_series]
        return tensor_data.reshape(n_series, n_timesteps)


# TODO: delete?
def tensor_to_sliding_windows(tensor, seq_len, pred_len=0, time_increment=1):
    """
    Convert flattened tensor to sliding window format.

    Args:
        tensor: torch.Tensor of shape [1, total_length]
        seq_len: int, length of each sequence window
        pred_len: int, length of prediction window (default 0 if not used)
        time_increment: int, step size between windows (default 1)

    Returns:
        torch.Tensor: Sliding windows with shape [n_windows, seq_len]
    """

    # Squeeze to get 1D tensor
    data = tensor.squeeze(0)  # Shape: [20160]

    # Calculate number of possible windows
    n_samples = data.shape[0] - (seq_len - 1) - pred_len

    if n_samples <= 0:
        raise ValueError(
            f"Not enough data points. Need at least {seq_len + pred_len}, got {data.shape[0]}"
        )

    # Create sliding windows
    windows = []
    for i in range(0, n_samples, time_increment):
        # window = data[i : i + seq_len]
        # window = data[(i + seq_len) : (i + seq_len + pred_len)].T)
        window = data[(i + seq_len) : (i + seq_len + pred_len)]
        windows.append(window)

    # Stack into tensor
    return torch.stack(windows)
