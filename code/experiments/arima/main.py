import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# TODO: change paths here, don't need all of em
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "pyhessian"))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "loss_landscape"))

from src.utils.args import get_public_config
from src.engines.arima_engine import ARIMAEngine
from src.utils.dataloader import DartsDataloader
from src.utils.logging import get_logger

import numpy as np
import torch
import pandas as pd
import time

from darts import TimeSeries
from darts.models import ARIMA, AutoARIMA
from darts.metrics import mse, rmse, mae, mape
from darts.utils.utils import ModelMode


import pdb

torch.set_num_threads(3)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    # ARIMA specific parameters
    parser.add_argument("--seasonal", type=bool, default=True, help="Use seasonal ARIMA")
    parser.add_argument("--seasonal_periods", type=int, default=12, help="Seasonal periods")
    parser.add_argument("--auto_arima", type=bool, default=True, help="Use auto ARIMA parameter selection")
    parser.add_argument("--p", type=int, default=1, help="AR order (if not auto)")
    parser.add_argument("--d", type=int, default=1, help="Differencing order (if not auto)")
    parser.add_argument("--q", type=int, default=1, help="MA order (if not auto)")
    parser.add_argument("--P", type=int, default=1, help="Seasonal AR order (if not auto)")
    parser.add_argument("--D", type=int, default=1, help="Seasonal differencing order (if not auto)")
    parser.add_argument("--Q", type=int, default=1, help="Seasonal MA order (if not auto)")
    parser.add_argument("--max_p", type=int, default=3, help="Maximum AR order (auto ARIMA)")
    parser.add_argument("--max_q", type=int, default=3, help="Maximum MA order (auto ARIMA)")
    parser.add_argument("--max_d", type=int, default=2, help="Maximum differencing order (auto ARIMA)")
    parser.add_argument("--max_P", type=int, default=2, help="Maximum seasonal AR order (auto ARIMA)")
    parser.add_argument("--max_Q", type=int, default=2, help="Maximum seasonal MA order (auto ARIMA)")
    parser.add_argument("--max_D", type=int, default=1, help="Maximum seasonal differencing order (auto ARIMA)")
    
    args = parser.parse_args()

    if args.model_name == "":
        args.model_name = "arima_darts"
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

def main():
    args, log_dir, logger = get_config()
    
    set_seed(args.seed)
    
    dataloader = DartsDataloader(
        dataset=args.dataset,
        args=args,
        logger=logger
    )
    
    train_series, val_series, test_series = dataloader.get_timeseries()
    dataloader_sliding_window = dataloader.get_sliding_window_dataloader()
    pdb.set_trace()
    
    engine = ARIMAEngine(
        train_series=train_series,
        val_series=val_series,
        test_series=test_series,
        args=args,
        logger=logger,
        log_dir=log_dir
    )

    # Read predictions from CSV file
    predictions_path = log_dir / "val_predictions.csv"
    predictions_df = pd.read_csv(predictions_path)

    engine.plot_results(predictions_df)
    
    if args.mode == "train":
        engine.train()
    elif args.mode == "test":
        engine.evaluate("test")
    else:
        engine.train()
        engine.evaluate("test")

if __name__ == "__main__":
    main()
