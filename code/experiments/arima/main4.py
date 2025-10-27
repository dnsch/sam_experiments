import sys
from pathlib import Path

from torch.utils import data

SCRIPT_DIR = Path(__file__).resolve().parent
# TODO: change paths here, don't need all of em
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "pyhessian"))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "loss_landscape"))

from src.utils.args import get_public_config

# from src.engines.arima_engine import ARIMA_Engine
from src.base.nixtla_engine import NixtlaEngine
from src.utils.dataloader import StatsforecastDataloader
from src.utils.logging import get_logger

import numpy as np
import torch
import pandas as pd
import time

# from darts.models import AutoARIMA
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# from darts import TimeSeries
# from darts.models import ARIMA, AutoARIMA
# from darts.metrics import mse, rmse, mae, mape
# from darts.utils.utils import ModelMode

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

    # StatsForecast specific parameters
    # TODO: change this some auto value as in the code down below
    parser.add_argument("--n_cores", type=int, default=8, help="Autoregressive order")
    # ARIMA specific parameters
    parser.add_argument(
        "--seasonal", type=bool, default=True, help="Use seasonal ARIMA"
    )
    parser.add_argument(
        "--seasonal_periods", type=int, default=12, help="Seasonal periods"
    )
    parser.add_argument(
        "--auto_arima",
        type=bool,
        default=True,
        help="Use auto ARIMA parameter selection",
    )
    parser.add_argument("--p", type=int, default=1, help="Autoregressive order")
    parser.add_argument("--d", type=int, default=1, help="Differencing order")
    parser.add_argument("--q", type=int, default=1, help="Moving average order")
    parser.add_argument("--P", type=int, default=1, help="Seasonal AR order")
    parser.add_argument("--D", type=int, default=1, help="Seasonal differencing order")
    parser.add_argument("--Q", type=int, default=1, help="Seasonal MA order")
    parser.add_argument("--max_p", type=int, default=3, help="Maximum AR order")
    parser.add_argument("--max_q", type=int, default=3, help="Maximum MA order")
    parser.add_argument(
        "--max_d", type=int, default=2, help="Maximum differencing order"
    )
    parser.add_argument(
        "--max_P", type=int, default=2, help="Maximum seasonal AR order"
    )
    parser.add_argument(
        "--max_Q", type=int, default=2, help="Maximum seasonal MA order"
    )
    parser.add_argument(
        "--max_D",
        type=int,
        default=1,
        help="Maximum seasonal differencing order",
    )

    args = parser.parse_args()

    if args.model_name == "":
        args.model_name = "arima"
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

    dataset_name = args.dataset
    time_increment = 1
    # The statsforecast models work a bit different from the others,
    # first, we load a dedicated dataloader instance
    dataloader_instance = StatsforecastDataloader(
        dataset="ETTh1",
        args=args,
        logger=logger,
        merge_train_val=True,  # Merge train and val
    )
    data = dataloader_instance.get_dataloader()
    import multiprocessing

    # Since we won't utilize the GPU, we set the available cpu core count
    n_cores = min(8, multiprocessing.cpu_count())  # avoid oversubscription

    # We specify the model, in our case, an auto arima with a season lenght of
    # 24 (hours)
    # sf_arima_model = AutoARIMA(
    #     season_length=24,  # daily seasonality for hourly data
    #     max_p=3,
    #     max_q=3,  # smaller AR/MA search
    #     max_P=1,
    #     max_Q=1,  # smaller seasonal AR/MA search
    #     d=1,
    #     D=1,  # fix differencing if reasonable for ETTh1
    #     stepwise=True,  # ensure stepwise search
    # )
    sf_arima_model = AutoARIMA(
        season_length=24,
        max_p=3,  # reduce from 3
        max_q=3,  # reduce from 3
        max_P=1,
        max_Q=1,
        d=1,
        D=1,
        stepwise=True,
        approximation=True,  # use approximation for speed
        seasonal=True,
        ic="aic",  # specify IC upfront
    )

    # StatsForecast allows for testing multiple models in one go,
    # for that, they need to be specified in the models array.
    # For now, we only use the autoarima model

    sf = StatsForecast(
        models=[sf_arima_model],
        freq="H",
        # n_jobs=n_cores,
        n_jobs=-1,
    )

    # Use all available CPU cores
    # n_cores = multiprocessing.cpu_count()
    # print(f"Using {n_cores} CPU cores")
    #
    # # sf = StatsForecast(
    # #     models=[AutoARIMA(season_length=24)],
    # #     freq="H",
    # #     n_jobs=n_cores,  # or n_jobs=-1 for all cores
    # # )
    #
    # # model = ARIMA(p=1, d=1, q=1, seasonal_order=(1, 1, 1, 12))
    # sf_arima_model = AutoARIMA(season_length=24)
    # sf = StatsForecast(
    #     models=[sf_arima_model],
    #     freq="H",
    #     n_jobs=-1,  # or n_jobs=-1 for all cores
    # )
    loss_fn = torch.nn.MSELoss()

    # TODO: change loss_fn to args
    engine = NixtlaEngine(
        model=sf,
        dataloader=data,
        pred_len=args.horizon,
        loss_fn=loss_fn,
        backend="statsforecast",
        logger=logger,
        log_dir=log_dir,
        seed=args.seed,
    )

    engine.train()
    pdb.set_trace()
    # ARIMA per component
    arima_overall, arima_windows, arima_details = engine.evaluate_arima_multivariate(
        darts_dl=darts_dl,
        step=args.seq_len,
        metric="MAE",
        merge_val_into_train=True,
        expand_train=True,
        seasonal=True,  # set according to your data
        m=24,  # e.g., hourly data with daily seasonality
        use_cached_orders=False,
    )
    pdb.set_trace()
    # dataloader_sliding_window = dataloader.get_sliding_window_dataloader()
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    endog_data = pd.concat([data["train_loader"], data["val_loader"]])

    mod = sm.tsa.SARIMAX(
        endog_data.values[:, 0],
        exog=endog_data.values[:, 1:],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
    )

    print("Samformer avg MAE:", samformer_overall)
    print("ARIMA avg MAE:", arima_overall)

    # Step 2: Fit the model with concatenated data
    res = mod.fit()
    # mod = sm.tsa.SARIMAX(order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    #
    # res = mod.fit(
    #     concatenated_data.values[:, 0],
    # )
    # mod = sm.tsa.SARIMAX(
    #     data["train_loader"].values[:, 0], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)
    # )
    #
    #

    # Python
    # Samformer
    sam_dl = SamformerDataloader(dataset_name, args, logger, time_increment=1)
    samformer_overall, samformer_windows = evaluate_samformer_on_windows(
        model=samformer_model,
        sam_dl=sam_dl,
        step=args.seq_len,
        metric="MAE",
        device="cuda",
    )

    # Forecast the next steps
    forecast_steps = 96  # Forecast for the next year
    forecast = res.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(
        start=pd.to_datetime(data["test_loader"].index[0]),
        periods=forecast_steps,
        freq="h",
    )
    original_index = pd.date_range(
        start=pd.to_datetime(concatenated_data.index[0]),
        periods=concatenated_data.values.shape[0] + 1,
        freq="h",
    )

    first_96_timesteps = data["test_loader"].values[:96, 0]  # First 96 values
    first_96_index = forecast_index

    plt.figure(figsize=(12, 6))

    plt.plot(
        first_96_index,
        first_96_timesteps,
        label="First 96 Timesteps",
        color="red",
        linewidth=1.5,
    )
    plt.plot(
        original_index[-50:],  # last 50 timestamps
        concatenated_data.values[-50:, 0],  # last 50 values
        label="Last 50 Points",
        color="blue",
    )

    plt.plot(
        forecast_index,
        forecast.predicted_mean,
        label="Forecast",
        color="orange",
        linestyle="--",
    )

    conf_int = forecast.conf_int()
    plt.fill_between(
        forecast_index,
        conf_int[:, 0],
        conf_int[:, 1],
        color="orange",
        alpha=0.3,
    )

    # Add labels and legend
    plt.title("SARIMA Model Forecast for ETTh1 Data")
    plt.xlabel("Time")
    plt.ylabel("Electricity Transformer Temperature")
    plt.legend()
    plt.show()

    pdb.set_trace()

    mod = sm.tsa.SARIMAX(
        concatenated_data.values[:, 0], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)
    )
    res = mod.fit()

    loss_fn = torch.nn.MSELoss()

    engine = SAMFormer_Engine(
        model=model,
        dataloader=dataloader,
        # scaler=scaler,
        scaler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        clip_grad_value=0,
        # clip_grad_value=4,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        batch_size=args.batch_size,
        num_channels=dataloader["train_loader"].dataset[0][0].shape[0],
        pred_len=args.horizon,
        no_sam=args.no_sam,
        use_revin=args.use_revin,
        gsam=args.gsam,
    )

    pdb.set_trace()

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
