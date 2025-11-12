import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# TODO: change paths here, don't need all of em
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))

from src.utils.args import get_public_config

from src.base.nixtla_engine import NixtlaEngine
from src.utils.dataloader import StatsforecastDataloader
from src.utils.logging import get_logger

import numpy as np
import torch
import pandas as pd
import time

from statsforecast import StatsForecast
from statsforecast.models import AutoMFLES

import pdb
import multiprocessing

torch.set_num_threads(3)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    # StatsForecast specific parameters
    parser.add_argument(
        "--n_cores", type=int, default=8, help="Number of cores for parallel processing"
    )

    # MFLES specific parameters

    parser.add_argument(
        "--season_length",
        type=int,
        nargs="*",  # Use '*' to allow empty list (for None behavior)
        default=24,
        help="Seasonal period(s). For hourly data: 24=daily, 168=weekly. If not specified, automatically determined",
    )
    parser.add_argument(
        "--n_windows",
        type=int,
        default=2,
        help="Number of windows for cross-validation",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=None,
        help="Step size for rolling window validation. If None, equals test_size",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="smape",
        choices=["smape", "mase", "rmse", "mae", "mape"],
        help="Metric for model selection",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Print detailed information during fitting",
    )
    parser.add_argument(
        "--alias", type=str, default="AutoMFLES", help="Model alias name"
    )
    args = parser.parse_args()

    if args.model_name == "":
        args.model_name = "mfles"
    if args.dataset == "":
        args.dataset = "ETTh1"

    base_dir = SCRIPT_DIR.parents[2] / "results"

    log_dir = "{}/{}/{}/seq_len_{}_pred_len_{}/".format(
        base_dir,
        args.model_name,
        args.dataset,
        args.seq_len,
        args.horizon,
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def run_experiments_on_data_list(
    data_list, scaler_list=None, args=None, logger=None, log_dir=None
):
    """
    Execute training for each data entry in data_list using StatsForecast/AutoARIMA.

    Args:
        data_list: List of data splits (each containing train/val/test dataframes)
        args: Arguments object containing hyperparameters and configuration
        logger: Logger object
        log_dir: Directory for logging

    Returns:
        results: List of results from each experiment
    """
    # Set CPU cores for parallel processing
    n_cores = min(8, multiprocessing.cpu_count())

    results = []

    # Iterate through each data entry
    for idx, data in enumerate(data_list):
        print(f"\n{'=' * 60}")
        print(f"Processing Experiment {idx + 1}/{len(data_list)}")
        print(f"{'=' * 60}\n")

        # Create the AutoTBATS model

        # TODO: change parameter values to ones from args
        sf_mfles_model = AutoMFLES(
            test_size=96,  # Required: validation set size
            season_length=24,
            n_windows=2,  # Number of cross-validation windows
            metric="smape",  # Model selection metric
            verbose=False,
            prediction_intervals=None,  # Can be configured separately if needed
        )
        # Create StatsForecast instance with the model
        sf = StatsForecast(
            models=[sf_mfles_model],
            freq="H",
            n_jobs=-1,
        )
        from src.utils.functions import get_statsforecast_model

        model = get_statsforecast_model(args)
        pdb.set_trace()

        # Define loss function
        loss_fn = torch.nn.MSELoss()

        # Create experiment-specific log directory
        # experiment_log_dir = f"{log_dir}/experiment_{idx}"
        experiment_log_dir = log_dir / f"experiment_{idx}"

        # Create the engine
        # TODO: scaler None here, that means we can't scale the data back
        # but as we compare it to other scaled data and preds anyway, maybe
        # this option is not needed
        engine = NixtlaEngine(
            model=model,
            dataloader=data,
            scaler=None,
            pred_len=args.horizon,
            loss_fn=loss_fn,
            backend="statsforecast",
            num_channels=data[0]["unique_id"].nunique(),
            logger=logger,
            log_dir=experiment_log_dir,
            seed=args.seed,
        )

        # Train the model
        result = engine.train()

        results.append(result)

        print(f"\nCompleted Experiment {idx + 1}/{len(data_list)}\n")

    return results


def main():
    args, log_dir, logger = get_config()

    set_seed(args.seed)

    dataset_name = args.dataset
    time_increment = 1

    # The statsforecast models work a bit different from the others,
    # first, we load a dedicated dataloader instance
    dataloader_instance = StatsforecastDataloader(
        dataset=dataset_name,
        args=args,
        logger=logger,
        merge_train_val=True,  # Merge train and val
    )
    data = dataloader_instance.get_dataloader()

    # Execute all experiments

    results = run_experiments_on_data_list(
        data_list=data, args=args, logger=logger, log_dir=log_dir
    )

    # Analyze results
    for idx, result in enumerate(results):
        print(f"Experiment {idx}: {result}")


if __name__ == "__main__":
    main()
