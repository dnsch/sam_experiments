import argparse

"""
File to build model and configuration specific arguments
"""

# TODO: add vgg arg group
# then add this to the _add_vgg_args
#
#
# Override after all groups are added
# model_group = _get_argument_group(parser, "Model")
# if model_group:
#     # Find and modify the existing argument
#     for action in model_group._group_actions:
#         if action.dest == "loss_name":
#             action.default = "cross-entropy"
#             action.help = "loss functions: cross-entropy"
#             break


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
        description="Configurations for Model Training and Evaluation:",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=1,
        metavar="N",
        help="random seed for reproducibility",
    )
    # TODO: add additional loss functions
    model_group.add_argument(
        "--loss_name",
        "-l",
        default="mse",
        metavar="LOSS",
        help="loss functions: mse | mae | rmse | mape",
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

    deep_learning_group.add_argument(
        "--use_revin",
        action="store_true",
        default=False,
        help="use reversible instance normalization",
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
            default=None,
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


def _add_tsmixer_args(parser):
    """Add TSMixer model architecture arguments."""

    # TSMixer specific arguments
    tsmixer_group = parser.add_argument_group(
        "TSMixer Model", "TSMixer-specific model architecture hyperparameters"
    )

    tsmixer_group.add_argument(
        "--num_channels",
        type=int,
        default=7,
        metavar="N",
        help="number of input channels",
    )
    tsmixer_group.add_argument(
        "--activation_fn",
        type=str,
        default="relu",
        metavar="ACTIVATION",
        help="activation function for TSMixer",
    )
    tsmixer_group.add_argument(
        "--num_blocks",
        type=int,
        default=2,
        metavar="N",
        help="number of mixer blocks",
    )
    tsmixer_group.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        metavar="RATE",
        help="dropout rate for TSMixer",
    )
    tsmixer_group.add_argument(
        "--ff_dim",
        type=int,
        default=64,
        metavar="N",
        help="feedforward dimension in mixer layers",
    )
    tsmixer_group.add_argument(
        "--normalize_before",
        type=bool,
        default=True,
        metavar="BOOL",
        help="whether to normalize before mixer layers",
    )
    tsmixer_group.add_argument(
        "--norm_type",
        type=str,
        default="batch",
        choices=["batch", "layer"],
        metavar="TYPE",
        help="type of normalization",
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
        default="H",
        metavar="FREQ",
        help="frequency of the data (see pandas available frequencies, e.g., 'H' for hourly, 'D' for daily)",
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
        default=24,
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


def _add_auto_mfles_args(parser):
    """Add Auto MFLES arguments."""
    auto_mfles_group = parser.add_argument_group(
        "Auto MFLES", "Auto MFLES model-specific settings"
    )

    # MFLES specific parameters
    auto_mfles_group.add_argument(
        "--season_length",
        type=int,
        nargs="*",
        default=24,
        metavar="N",
        help="Seasonal period(s). For hourly data: 24=daily, 168=weekly. If not specified, automatically determined",
    )
    auto_mfles_group.add_argument(
        "--n_windows",
        type=int,
        default=2,
        metavar="N",
        help="Number of windows for cross-validation",
    )
    auto_mfles_group.add_argument(
        "--step_size",
        type=int,
        default=None,
        metavar="N",
        help="Step size for rolling window validation. If None, equals test_size",
    )
    auto_mfles_group.add_argument(
        "--metric",
        type=str,
        default="smape",
        choices=["smape", "mase", "rmse", "mae", "mape"],
        metavar="METRIC",
        help="Metric for model selection",
    )
    auto_mfles_group.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed information during fitting",
    )
    auto_mfles_group.add_argument(
        "--alias",
        type=str,
        default="AutoMFLES",
        metavar="ALIAS",
        help="Model alias name",
    )
    auto_mfles_group.add_argument(
        "--prediction_intervals",
        type=str,
        default=None,
        metavar="INTERVALS",
        help="Prediction intervals configuration",
    )

    return parser


def _add_auto_tbats_args(parser):
    """Add Auto TBATS arguments."""
    auto_tbats_group = parser.add_argument_group(
        "Auto TBATS", "Auto TBATS model-specific settings"
    )

    # TBATS specific parameters
    auto_tbats_group.add_argument(
        "--season_length",
        type=int,
        nargs="+",
        default=[24],
        metavar="N",
        help="Seasonal period(s). For hourly data: 24=daily, 168=weekly. Can specify multiple (e.g., 24 168 for both daily and weekly patterns)",
    )
    auto_tbats_group.add_argument(
        "--use_boxcox",
        type=bool,
        default=False,
        metavar="BOOL",
        help="Use Box-Cox transformation. If None, automatically determined",
    )
    auto_tbats_group.add_argument(
        "--bc_lower_bound",
        type=float,
        default=0.0,
        metavar="BOUND",
        help="Lower bound for Box-Cox parameter",
    )
    auto_tbats_group.add_argument(
        "--bc_upper_bound",
        type=float,
        default=1.0,
        metavar="BOUND",
        help="Upper bound for Box-Cox parameter",
    )
    auto_tbats_group.add_argument(
        "--use_trend",
        type=bool,
        default=None,
        metavar="BOOL",
        help="Use trend component. If None, automatically determined",
    )
    auto_tbats_group.add_argument(
        "--use_damped_trend",
        type=bool,
        default=None,
        metavar="BOOL",
        help="Use damped trend. If None, automatically determined",
    )
    auto_tbats_group.add_argument(
        "--use_arma_errors",
        type=bool,
        default=True,
        metavar="BOOL",
        help="Use ARMA errors",
    )
    auto_tbats_group.add_argument(
        "--alias",
        type=str,
        default="AutoTBATS",
        metavar="ALIAS",
        help="Model alias name",
    )

    return parser


def _add_historic_average_args(parser):
    """Add Historic Average arguments."""
    historic_average_group = parser.add_argument_group(
        "Historic Average", "Historic Average model-specific settings"
    )

    historic_average_group.add_argument(
        "--alias",
        type=str,
        default="HistoricAverage",
        metavar="ALIAS",
        help="Model alias name",
    )

    return parser


def _add_naive_args(parser):
    """Add Naive arguments."""
    naive_group = parser.add_argument_group("Naive", "Naive model-specific settings")

    naive_group.add_argument(
        "--alias", type=str, default="Naive", metavar="ALIAS", help="Model alias name"
    )

    return parser


def _add_seasonal_naive_args(parser):
    """Add Seasonal Naive arguments."""
    seasonal_naive_group = parser.add_argument_group(
        "Seasonal Naive", "Seasonal Naive model-specific settings"
    )

    seasonal_naive_group.add_argument(
        "--season_length",
        type=int,
        default=24,
        metavar="N",
        help="Seasonal period length (e.g., 24 for daily seasonality in hourly data)",
    )
    seasonal_naive_group.add_argument(
        "--alias",
        type=str,
        default="SeasonalNaive",
        metavar="ALIAS",
        help="Model alias name",
    )

    return parser


def _add_seasonal_exponential_smoothing_args(parser):
    """Add Seasonal Exponential Smoothing arguments."""
    seasonal_exponential_smoothing_group = parser.add_argument_group(
        "Seasonal Exponential Smoothing",
        "Seasonal Exponential Smoothing model-specific settings",
    )

    seasonal_exponential_smoothing_group.add_argument(
        "--season_length",
        type=int,
        default=24,
        metavar="N",
        help="Seasonal period length (e.g., 24 for daily seasonality in hourly data)",
    )
    seasonal_exponential_smoothing_group.add_argument(
        "--prediction_intervals",
        type=str,
        default=None,
        metavar="INTERVALS",
        help="Prediction intervals configuration",
    )
    seasonal_exponential_smoothing_group.add_argument(
        "--alias",
        type=str,
        default="SeasonalExponentialSmoothingOptimized",
        metavar="ALIAS",
        help="Model alias name",
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


def get_tsmixer_config():
    """
    Complete TSMixer configuration parser.

    Includes:


    - Base config (hardware, model, dataset, experiment)
    - Deep learning config (optimizer, hyperparameters)


    - Loss landscape config (visualization and plotting)
    - TSMixer model architecture


    - SAM optimization
    - GSAM optimization

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)

    # Add TSMixer-specific configurations
    parser = _add_tsmixer_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    # Add loss landscape configuration
    parser = _add_loss_landscape_args(parser)

    return parser


def get_auto_arima_config():
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


def get_automfles_config():
    """
    Complete Auto MFLES configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, horizon)

    - StatsForecast config (frequency, parallel processing)
    - Auto MFLES model-specific config (seasonal parameters, validation settings)

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add StatsForecast configuration
    parser = _add_statsforecast_args(parser)

    # Add Auto MFLES-specific configurations
    parser = _add_auto_mfles_args(parser)

    return parser


def get_auto_tbats_config():
    """
    Complete Auto TBATS configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, horizon)

    - StatsForecast config (frequency, parallel processing)
    - Auto TBATS model-specific config (seasonal parameters, Box-Cox, trend, ARMA)

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add StatsForecast configuration
    parser = _add_statsforecast_args(parser)

    # Add Auto TBATS-specific configurations
    parser = _add_auto_tbats_args(parser)

    return parser


def get_historic_average_config():
    """
    Complete Historic Average configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, horizon)

    - StatsForecast config (frequency, parallel processing)
    - Historic Average model-specific config

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_historic_average_args(parser)
    return parser


def get_naive_config():
    """
    Complete Naive configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, horizon)

    - StatsForecast config (frequency, parallel processing)
    - Naive model-specific config

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_naive_args(parser)
    return parser


def get_seasonal_naive_config():
    """
    Complete Seasonal Naive configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, horizon)

    - StatsForecast config (frequency, parallel processing)
    - Seasonal Naive model-specific config

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_seasonal_naive_args(parser)
    return parser


def get_seasonal_exponential_smoothing_config():
    """
    Complete Seasonal Exponential Smoothing configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, horizon)

    - StatsForecast config (frequency, parallel processing)
    - Seasonal Exponential Smoothing model-specific config

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_seasonal_exponential_smoothing_args(parser)
    return parser
