import argparse

"""
File to build model and configuration specific arguments
"""


# Helper functions
# Add argument to an existing group helper
def _get_argument_group(parser, title):
    """Retrieve an existing argument group by title."""
    for group in parser._action_groups:
        if group.title == title:
            return group
    return None


# Base configuration (used in all experiments)
def get_base_config():
    """Base configuration parser with core training arguments."""
    parser = argparse.ArgumentParser(
        description="Configurations for Model Training and Evaluation:"
    )

    # Model
    model_group = parser.add_argument_group("Model", "Model settings")

    model_group.add_argument(
        "--mode",
        type=str,
        default="train",
        metavar="MODE",
        help="operation mode: train | test | eval",
    )
    model_group.add_argument(
        "--seed",
        type=int,
        default=2023,
        metavar="N",
        help="random seed for reproducibility",
    )
    # TODO: add additional loss functions
    model_group.add_argument(
        "--loss_name",
        "-l",
        default="mse",
        metavar="LOSS",
        help="loss functions: crossentropy | mse",
    )

    # Dataset
    dataset_group = parser.add_argument_group("Dataset", "Dataset settings")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="",
        required=True,
        metavar="NAME",
        help="name of the dataset to use",
    )
    dataset_group.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        metavar="RATIO",
        help="ratio of data to use for training (0.0-1.0)",
    )
    dataset_group.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        metavar="RATIO",
        help="ratio of data to use for validation (0.0-1.0)",
    )
    dataset_group.add_argument(
        "--raw_data", action="store_true", default=False, help="do not normalize data"
    )
    dataset_group.add_argument(
        "--noaug", default=False, action="store_true", help="no data augmentation"
    )
    # TODO: what exactly is this doing again?
    dataset_group.add_argument(
        "--label_corrupt_prob",
        type=float,
        default=0.0,
        metavar="PROB",
        help="probability of corrupting labels for robustness testing (0.0-1.0)",
    )

    # parser.add_argument(
    #     "--trainloader", default="", help="path to the dataloader with random labels"
    # )
    # parser.add_argument(
    #     "--testloader", default="", help="path to the testloader with random labels"
    # )

    # Experiment
    # seq_len denotes input history length, horizon denotes output future length
    exp_group = parser.add_argument_group("Experiment", "Experiment settings")
    # TODO: is this really needed?
    exp_group.add_argument(
        "--input_dim",
        type=int,
        default=3,
        metavar="N",
        help="number of input features/dimensions",
    )
    exp_group.add_argument(
        "--output_dim",
        type=int,
        default=1,
        metavar="N",
        help="number of output features/dimensions",
    )

    return parser


# Additional arguments


def _add_time_series_forecast_args(parser):
    """Add time series forecasting-specific arguments to Experiment group."""
    # Add to existing Experiment group
    exp_group = _get_argument_group(parser, "Experiment")
    if exp_group:
        # seq_len denotes input history length, horizon denotes output future length
        exp_group.add_argument(
            "--seq_len",
            type=int,
            default=512,
            metavar="N",
            help="input sequence length (history length)",
        )
        exp_group.add_argument(
            "--horizon",
            type=int,
            default=96,
            metavar="N",
            help="output prediction horizon (future length)",
        )

    return parser


def _add_deep_learning_args(parser):
    """Extended configuration parser including deep learning model arguments."""

    # Hardware
    hw_group = parser.add_argument_group("Hardware", "Hardware settings")
    hw_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        metavar="DEV",
        help="device to use for training (cpu or cuda:0, cuda:1, etc.)",
    )

    deep_learning_group = parser.add_argument_group(
        "Deep Learning", "Deep Learning settings"
    )

    # Optimizer
    deep_learning_group.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        metavar="OPT",
        help="Optimizer to use (e.g., Adam, SGD, Adagrad). Use same case as class names in torch.optim",
    )

    # Hyperparameters
    deep_learning_group.add_argument(
        "--batch_size",
        type=int,
        default=256,
        metavar="N",
        help="batch size for training",
    )
    deep_learning_group.add_argument(
        "--lrate",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate",
    )
    # TODO: check help here again
    deep_learning_group.add_argument(
        "--wdecay",
        type=float,
        default=1e-5,
        metavar="WD",
        help="weight decay (L2 regularization)",
    )
    deep_learning_group.add_argument(
        "--clip_grad_value",
        type=float,
        default=0,
        metavar="VAL",
        help="gradient clipping value (0 = no clipping)",
    )

    # Add to existing Experiment group
    exp_group = _get_argument_group(parser, "Experiment")
    if exp_group:
        exp_group.add_argument(
            "--max_epochs",
            type=int,
            default=100,
            metavar="N",
            help="maximum number of training epochs",
        )
        exp_group.add_argument(
            "--patience",
            type=int,
            default=30,
            metavar="N",
            help="patience for early stopping (number of epochs without improvement)",
        )

    return parser


def _add_loss_landscape_args(parser):
    """Extended configuration parser including loss landscape and plotting arguments."""

    # Loss landscape group with description
    ll_group = parser.add_argument_group(
        "Loss Landscape Visualization",
        "Loss Landscape Visualization settings,\n For more details, visit: https://github.com/tomgoldstein/loss-landscape",
    )
    ll_group.add_argument(
        "--plot_surface_mpi",
        "-m",
        action="store_true",
        help="use mpi for loss landscape computation",
    )
    ll_group.add_argument(
        "--plot_surface_cuda",
        "-c",
        action="store_true",
        help="use cuda for loss landscape computation",
    )
    ll_group.add_argument(
        "--threads",
        default=2,
        type=int,
        metavar="N",
        help="number of threads",
    )
    ll_group.add_argument(
        "--ngpu",
        type=int,
        default=1,
        metavar="N",
        help="number of GPUs to use for each rank, useful for data parallel evaluation",
    )

    # Model parameters group
    model_params_group = parser.add_argument_group(
        "Loss Landscape Visualization", "Model Parameters"
    )
    model_params_group.add_argument(
        "--model",
        default="samformer",
        metavar="NAME",
        help="model name",
    )
    model_params_group.add_argument(
        "--model_file",
        default="experiments/samformer/ETTh1/final_model_s2024.pt",
        metavar="PATH",
        help="path to the trained model file",
    )
    model_params_group.add_argument(
        "--model_file2",
        default="",
        metavar="PATH",
        help="use (model_file2 - model_file) as the xdirection",
    )
    model_params_group.add_argument(
        "--model_file3",
        default="",
        metavar="PATH",
        help="use (model_file3 - model_file) as the ydirection",
    )

    # Direction parameters group
    dir_group = parser.add_argument_group(
        "Loss Landscape Visualization",
        "Direction Parameters",
    )
    dir_group.add_argument(
        "--dir_file",
        default="",
        metavar="PATH",
        help="specify the name of direction file, or the path to an existing direction file",
    )
    dir_group.add_argument(
        "--dir_type",
        default="weights",
        metavar="TYPE",
        help="direction type: weights | states (including BN's running_mean/var)",
    )
    dir_group.add_argument(
        "--x",
        default="-1:1:51",
        metavar="MIN:MAX:NUM",
        help="A string with format xmin:x_max:xnum",
    )
    dir_group.add_argument(
        "--y",
        default=None,
        metavar="MIN:MAX:NUM",
        help="A string with format ymin:ymax:ynum",
    )
    dir_group.add_argument(
        "--xnorm",
        default="",
        metavar="TYPE",
        help="direction normalization: filter | layer | weight",
    )
    dir_group.add_argument(
        "--ynorm",
        default="",
        metavar="TYPE",
        help="direction normalization: filter | layer | weight",
    )
    dir_group.add_argument(
        "--xignore",
        default="",
        metavar="TYPE",
        help="ignore bias and BN parameters: biasbn",
    )
    dir_group.add_argument(
        "--yignore",
        default="",
        metavar="TYPE",
        help="ignore bias and BN parameters: biasbn",
    )
    dir_group.add_argument(
        "--same_dir",
        action="store_true",
        default=False,
        help="use the same random direction for both x-axis and y-axis",
    )
    dir_group.add_argument(
        "--idx",
        default=0,
        type=int,
        metavar="N",
        help="the index for the repeatness experiment",
    )
    dir_group.add_argument(
        "--surf_file",
        default="",
        metavar="PATH",
        help="customize the name of surface file, could be an existing file.",
    )
    dir_group.add_argument(
        "--hessian_directions",
        action="store_true",
        default=False,
        help="create hessian eigenvectors directions h5 file",
    )

    # Plot parameters group
    plot_group = parser.add_argument_group(
        "Loss Landscape Visualization", "Plot Parameters"
    )
    plot_group.add_argument(
        "--proj_file",
        default="",
        metavar="PATH",
        help="the .h5 file contains projected optimization trajectory.",
    )
    plot_group.add_argument(
        "--loss_max",
        default=5,
        type=float,
        metavar="VAL",
        help="Maximum value to show in 1D plot",
    )
    plot_group.add_argument(
        "--vmax",
        default=10,
        type=float,
        metavar="VAL",
        help="Maximum value to map",
    )
    plot_group.add_argument(
        "--vmin",
        default=0.1,
        type=float,
        metavar="VAL",
        help="Minimum value to map",
    )
    plot_group.add_argument(
        "--vlevel",
        default=0.5,
        type=float,
        metavar="VAL",
        help="plot contours every vlevel",
    )
    plot_group.add_argument(
        "--show", action="store_true", default=False, help="show plotted figures"
    )
    plot_group.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="use log scale for loss values",
    )
    plot_group.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="plot figures after computation",
    )

    return parser


def _add_sam_args(parser):
    """Add SAM (Sharpness Aware Minimization) arguments."""
    sam_group = parser.add_argument_group(
        "SAM Optimization", "Sharpness Aware Minimization settings"
    )

    sam_group.add_argument(
        "--no_sam",
        action="store_true",
        default=False,
        help="don't use Sharpness Aware Minimization",
    )
    sam_group.add_argument(
        "--rho",
        type=float,
        default=0.5,
        metavar="RHO",
        help="neighborhood size for SAM",
    )
    return parser


def _add_gsam_args(parser):
    """Add GSAM (Gradient-based SAM) arguments."""
    gsam_group = parser.add_argument_group(
        "GSAM Optimization", "Gradient-based Sharpness Aware Minimization settings"
    )

    gsam_group.add_argument(
        "--gsam",
        action="store_true",
        default=False,
        help="use GSAM (Gradient-based SAM) instead of standard SAM",
    )
    gsam_group.add_argument(
        "--gsam_alpha",
        type=float,
        default=0.5,
        metavar="ALPHA",
        help="alpha parameter for GSAM",
    )
    gsam_group.add_argument(
        "--gsam_rho_max",
        type=float,
        default=0.5,
        metavar="RHO_MAX",
        help="maximum rho value for GSAM",
    )
    gsam_group.add_argument(
        "--gsam_rho_min",
        type=float,
        default=0.1,
        metavar="RHO_MIN",
        help="minimum rho value for GSAM",
    )
    gsam_group.add_argument(
        "--gsam_adaptive",
        action="store_true",
        default=False,
        help="use adaptive GSAM with dynamic rho scheduling",
    )
    return parser


# Model specific arguments
def _add_samformer_args(parser):
    """Add SAMFormer model architecture arguments."""

    # SAMFormer specific arguments
    samformer_group = parser.add_argument_group(
        "SAMFormer Model", "SAMFormer-specific model architecture hyperparameters"
    )

    samformer_group.add_argument(
        "--use_revin",
        action="store_true",
        default=True,
        help="use reversible instance normalization",
    )
    samformer_group.add_argument(
        "--num_channels",
        type=int,
        default=7,
        metavar="N",
        help="number of input channels",
    )
    samformer_group.add_argument(
        "--hid_dim", type=int, default=16, metavar="N", help="hidden dimension size"
    )
    return parser


def _add_statsforecast_args(parser):
    """Add StatsForecast arguments."""
    statsforecast_group = parser.add_argument_group(
        "StatsForecast", "StatsForecast-specific settings"
    )

    statsforecast_group.add_argument(
        "--freq",
        type=str,
        default="h",
        metavar="FREQ",
        help="frequency of the data (see pandas available frequencies, e.g., 'h' for hourly, 'D' for daily)",
    )
    statsforecast_group.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        metavar="N",
        help="number of jobs used in parallel processing (-1 for all cores)",
    )

    return parser


def _add_auto_arima_args(parser):
    """Add Auto ARIMA arguments."""
    auto_arima_group = parser.add_argument_group(
        "Auto ARIMA", "Auto ARIMA model-specific settings"
    )

    # StatsForecast specific parameters
    # TODO: change this some auto value as in the code down below
    auto_arima_group.add_argument(
        "--n_cores",
        type=int,
        default=8,
        metavar="N",
        help="number of cores for parallel processing",
    )

    # ARIMA specific parameters
    auto_arima_group.add_argument(
        "--seasonal",
        action="store_true",
        default=True,
        help="use seasonal ARIMA",
    )
    auto_arima_group.add_argument(
        "--seasonal_periods",
        type=int,
        default=12,
        metavar="N",
        help="seasonal periods",
    )
    auto_arima_group.add_argument(
        "--auto_arima",
        type=bool,
        default=True,
        metavar="BOOL",
        help="use auto ARIMA parameter selection",
    )
    auto_arima_group.add_argument(
        "--p",
        type=int,
        default=1,
        metavar="N",
        help="autoregressive order",
    )
    auto_arima_group.add_argument(
        "--d",
        type=int,
        default=1,
        metavar="N",
        help="differencing order",
    )
    auto_arima_group.add_argument(
        "--q",
        type=int,
        default=1,
        metavar="N",
        help="moving average order",
    )
    auto_arima_group.add_argument(
        "--P",
        type=int,
        default=1,
        metavar="N",
        help="seasonal AR order",
    )
    auto_arima_group.add_argument(
        "--D",
        type=int,
        default=1,
        metavar="N",
        help="seasonal differencing order",
    )
    auto_arima_group.add_argument(
        "--Q",
        type=int,
        default=1,
        metavar="N",
        help="seasonal MA order",
    )
    auto_arima_group.add_argument(
        "--max_p",
        type=int,
        default=3,
        metavar="N",
        help="maximum AR order",
    )
    auto_arima_group.add_argument(
        "--max_q",
        type=int,
        default=3,
        metavar="N",
        help="maximum MA order",
    )
    auto_arima_group.add_argument(
        "--max_d",
        type=int,
        default=2,
        metavar="N",
        help="maximum differencing order",
    )
    auto_arima_group.add_argument(
        "--max_P",
        type=int,
        default=2,
        metavar="N",
        help="maximum seasonal AR order",
    )
    auto_arima_group.add_argument(
        "--max_Q",
        type=int,
        default=2,
        metavar="N",
        help="maximum seasonal MA order",
    )
    auto_arima_group.add_argument(
        "--max_D",
        type=int,
        default=1,
        metavar="N",
        help="maximum seasonal differencing order",
    )

    return parser


# Model configurations
def get_samformer_config():
    """
    Complete SAMFormer configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Deep learning config (optimizer, hyperparameters)

    - Loss landscape config (visualization and plotting)
    - SAMFormer model architecture

    - SAM optimization
    - GSAM optimization

    """
    # Start with loss landscape config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)

    # Add SAMFormer-specific configurations
    parser = _add_samformer_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    # Add loss landscape configuration
    parser = _add_loss_landscape_args(parser)

    return parser


def get_autoarima_config():
    """
    Complete Auto ARIMA configuration parser.

    Includes:


    - Base config (hardware, model, dataset, experiment)
    - StatsForecast config (frequency, parallel processing)

    - Auto ARIMA model-specific config (seasonal parameters, ARIMA orders)

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add StatsForecast configuration
    parser = _add_statsforecast_args(parser)

    # Add Auto ARIMA-specific configurations
    parser = _add_auto_arima_args(parser)

    return parser
