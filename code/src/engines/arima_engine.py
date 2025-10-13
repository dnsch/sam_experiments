from bdb import set_trace
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils.functions import (
    plot_stats,
    branch_plot,
    plot_mean_per_day,
    mean_branch_plot,
)

from src.base.torch_engine import BaseEngine
from darts.models import ARIMA, AutoARIMA, VARIMA
from darts.utils.timeseries_generation import holidays_timeseries
from darts.metrics import mse, rmse, mae, mape
from darts import TimeSeries
import pdb

import matplotlib.pyplot as plt
import numpy as np
import os


# (As the ARIMA model is not implemented in pytorch, it doesn't make sense that
# it inherits from the BaseEngine)
# let's try to implement this in the same fashion as the other enignes,
# and scrap the parts that aren't needed


class ARIMA_Engine(BaseEngine):
    """
    ARIMA forecasting engine using Darts library with multivariate support
    """

    # TODO: check what to do with the **args here, are they really needed? Do
    # we want kwargs instead?
    def __init__(
        self,
        # batch_size=1,
        # num_channels=1,
        pred_len=1,
        # no_sam=True,
        # use_revin=True,
        # gsam=False,
        **args,
    ):
        super(ARIMA_Engine, self).__init__(**args)
        # TODO: check if all class attributes should be named with a leading
        # underscore
        self.pred_len = pred_len
        self.random_state = self._seed
        # self._timeout = True
        self._plot_path = self._save_path / "plots"
        # Initialize metrics
        self._mape = self._to_device(MeanAbsolutePercentageError())
        self._rmse = self._to_device(MeanSquaredError(squared=False))

    def train(self):
        """Handles the model fitting based on user args"""
        self.logger.info("Start training!")
        # TODO: turn this N here into a parameter, experiment with different
        # values to get a good trade-off between speed and performance
        N = 200  # tune based on accuracy/speed trade-off
        train_df = self.dataloader["train_loader"]
        train_df_window = (
            train_df.sort_values(["unique_id", "ds"])
            .groupby("unique_id", as_index=False)
            .tail(N)
        )
        # train_data = train_df_window
        train_data = train_df

        # Fit the model
        t1 = time.time()
        self.model.fit(train_data)
        t2 = time.time()
        pdb.set_trace()
        # TODO: come up with better filename_suffix
        filename_suffix = str(self.pred_len) + str(self.random_state)
        self.model.save_statsforecast_model(
            save_path=self._save_path, filename_suffix=filename_suffix
        )

        # Get train and test data
        test_data = self.dataloader["test_loader"]

        # Generate predictions for 96 steps ahead
        h = 96  # Forecast 96 steps into the future
        predictions = self.model.predict(h=h, level=[95])

        # Get all unique IDs
        unique_ids = train_data["unique_id"].unique()

        # Create output directory
        output_dir = "arima_plots_96_steps"
        os.makedirs(output_dir, exist_ok=True)

        # Create plots for each unique_id
        for uid in tqdm(unique_ids, desc="Creating focused plots"):
            # Filter data for current unique_id
            train_df_full = train_data[train_data["unique_id"] == uid].copy()
            test_df = test_data[test_data["unique_id"] == uid].copy()
            pred = predictions[predictions["unique_id"] == uid].copy()

            # Get only the last 48 training steps
            train_df = train_df_full.tail(48).copy()

            # Limit test data to 96 steps (in case there's more)
            test_df = test_df.head(96).copy()

            plt.figure(figsize=(14, 6))

            # Plot last 48 training steps
            plt.plot(
                train_df["ds"],
                train_df["y"],
                label="Training Data (Last 48 steps)",
                color="blue",
                linewidth=2,
            )

            # Plot actual test values (up to 96 steps)
            if not test_df.empty:
                plt.plot(
                    test_df["ds"],
                    test_df["y"],
                    label=f"Actual Test Values ({len(test_df)} steps)",
                    color="green",
                    linewidth=2,
                )

            # Plot 96-step predictions
            plt.plot(
                pred["ds"],
                pred["AutoARIMA"],
                label="96-Step Forecast",
                color="red",
                linestyle="--",
                linewidth=2,
            )

            # Plot confidence intervals
            if "AutoARIMA-lo-95" in pred.columns and "AutoARIMA-hi-95" in pred.columns:
                plt.fill_between(
                    pred["ds"],
                    pred["AutoARIMA-lo-95"],
                    pred["AutoARIMA-hi-95"],
                    alpha=0.3,
                    color="red",
                    label="95% Confidence Interval",
                )

            # Add vertical line to separate training and forecast periods
            if not train_df.empty:
                last_train_date = train_df["ds"].iloc[-1]
                plt.axvline(
                    x=last_train_date,
                    color="black",
                    linestyle=":",
                    alpha=0.7,
                    label="Train/Test Split",
                )

            plt.title(f"ARIMA Focused Forecast View - {uid}")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            # Save the plot
            filename = os.path.join(output_dir, f"arima_focused_96_forecast_{uid}.png")
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"✅ Created {len(unique_ids)} focused forecast plots!")
        print(f"📁 Plots saved in '{output_dir}' folder")

    # if self.args.arima_type == "VARIMA":
    #     model = VARIMA()
    #     model_string = "VARIMA"
    # elif self.args.arima_type == "AutoARIMA":
    #     model = AutoARIMA(
    #         start_p=0,
    #         start_q=0,
    #         max_p=self.args.max_p,
    #         max_q=self.args.max_q,
    #         max_d=self.args.max_d,
    #         seasonal=self.args.seasonal,
    #         start_P=0,
    #         start_Q=0,
    #         max_P=self.args.max_P,
    #         max_Q=self.args.max_Q,
    #         max_D=self.args.max_D,
    #         eason_length=self.args.seasonal_periods if self.args.seasonal else 1,
    #         random_state=self.args.seed,
    #     )
    #     model_string = "AutoARIMA"
    # else:
    #     if self.args.seasonal:
    #         seasonal_order = (
    #             self.args.P,
    #             self.args.D,
    #             self.args.Q,
    #             self.args.seasonal_periods,
    #         )
    #     else:
    #         seasonal_order = None
    #
    #     model = ARIMA(
    #         p=self.args.p,
    #         d=self.args.d,
    #         q=self.args.q,
    #         seasonal_order=seasonal_order,
    #         random_state=self.args.seed,
    #     )
    #     model_string = "ARIMA"

    # if self.args.arima_type == "VARIMA":
    #     if self.args.arima_holidays:
    #         # Use future covariates
    #         # from https://unit8co.github.io/darts/generated_api/darts.models.forecasting.varima.html
    #         # "e.g. encode each timestep whether it is on a holiday"
    #         future_cov = holidays_timeseries(series.time_index, "CN", add_length=6)
    #         total_start_time = time.time()
    #         model.fit(series, future_covariates=future_cov)
    #     else:
    #         total_start_time = time.time()
    #         model.fit(series)
    #
    # else:
    # Fit separate model for each feature with progress bar
    # pbar = tqdm(
    #     enumerate(self.feature_names),
    #     total=self.n_features,
    #     desc=f"Training {model_string} models",
    #     unit="feature",
    # )
    #
    # total_start_time = time.time()
    # for i, feature_name in pbar:
    #     # Update progress bar description
    #     pbar.set_description(f"Training {model_type} for {feature_name}")
    #
    #     feature_start_time = time.time()
    #
    #     # Extract univariate series for this feature
    #     univariate_series = self.train_series.univariate_component(feature_name)
    #
    #     # Add a progress bar for model fitting
    #     fitting_pbar = tqdm(total=1, desc=f"Fitting {feature_name}", leave=False)
    #     model.fit(univariate_series)
    #     fitting_pbar.update(1)
    #     fitting_pbar.close()
    #
    #     # Store the model
    #     self.models[feature_name] = model
    #
    #     feature_end_time = time.time()
    #     feature_duration = feature_end_time - feature_start_time
    #
    #     # Update progress bar with timing info
    #     pbar.set_postfix(
    #         {
    #             "time": f"{feature_duration:.1f}s",
    #             "avg_time": f"{(time.time() - total_start_time) / (i + 1):.1f}s",
    #         }
    #     )
    #
    #     self.logger.info(
    #         f"Feature {feature_name} training completed in {feature_duration:.2f} seconds"
    #     )
    #
    # pbar.close()
    #
    # total_end_time = time.time()
    # self.logger.info(
    #     f"All ARIMA models training completed in {total_end_time - total_start_time:.2f} seconds"
    # )
    #
    # # Evaluate on validation set
    # self.evaluate("val")
    #

    # def _predict_multivariate(self, n_periods, historical_series):
    #     """Generate predictions for all features"""
    #     predictions_dict = {}
    #
    #     # Add progress bar for prediction phase
    #     pbar = tqdm(self.feature_names, desc="Generating predictions", unit="feature")
    #
    #     for feature_name in pbar:
    #         pbar.set_description(f"Predicting {feature_name}")
    #
    #         # Get univariate series for this feature
    #         univariate_series = historical_series.univariate_component(feature_name)
    #
    #         # Generate prediction for this feature
    #         feature_prediction = self.models[feature_name].predict(n=n_periods)
    #         predictions_dict[feature_name] = feature_prediction.values()
    #
    #     pbar.close()
    #
    #     # Combine predictions into multivariate TimeSeries
    #     combined_values = np.column_stack(
    #         [predictions_dict[name] for name in self.feature_names]
    #     )
    #     combined_predictions = TimeSeries.from_values(
    #         combined_values, columns=self.feature_names
    #     )
    #
    #     return combined_predictions

    import numpy as np
    from typing import Tuple, List, Dict
    from darts.models import AutoARIMA, ARIMA

    def evaluate_arima_multivariate(
        self,
        darts_dl,  # instance of DartsDataloader (with rolling_eval_iterator)
        step: int,  # e.g., args.seq_len for non-overlapping windows
        metric: str = "MAE",  # "MAE", "RMSE", "MAPE"
        merge_val_into_train: bool = True,  # train on train+val initially
        expand_train: bool = True,  # expanding training as the window slides
        seasonal: bool = False,  # set True if your data is seasonal
        m: int = 1,  # seasonal period (e.g., 24 for hourly daily)
        use_cached_orders: bool = True,  # cache orders from initial fit to speed up subsequent fits
    ) -> Tuple[float, List[float], List[Dict]]:
        """
        Returns:
        overall_avg_loss: average across all components and all windows
        window_losses: per-window average loss across components
        details: per-window dicts with component-wise losses (optional for debugging)
        """

        # Choose metric function
        def _loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
            if metric.upper() == "MAE":
                return float(np.mean(np.abs(y_pred - y_true)))
            elif metric.upper() == "RMSE":
                return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            elif metric.upper() == "MAPE":
                denom = np.clip(np.abs(y_true), 1e-8, None)
                return float(np.mean(np.abs((y_pred - y_true) / denom)))
            else:
                raise ValueError(f"Unknown metric '{metric}'")

        # Cache orders on the initial training cut if requested
        orders_cache = None
        if use_cached_orders:
            pdb.set_trace()
            init_train_end = (
                darts_dl.val_end if merge_val_into_train else darts_dl.train_end
            )
            init_train_series = (
                darts_dl.train_series.append(darts_dl.val_series)
                if merge_val_into_train
                else darts_dl.train_series
            )
            orders_cache = {}
            for name in darts_dl.feature_names:
                ts_comp = init_train_series.univariate_component(name)
                auto = AutoARIMA(seasonal=seasonal, m=m)
                auto.fit(ts_comp)
                pm = auto.model  # underlying pmdarima model
                orders_cache[name] = {
                    "order": pm.order,  # (p,d,q)
                    "seasonal_order": pm.seasonal_order,  # (P,D,Q,m)
                }

        window_losses: List[float] = []
        details: List[Dict] = []

        for fold in darts_dl.rolling_eval_iterator(
            step=step,
            merge_val_into_train=merge_val_into_train,
            expand_train=expand_train,
        ):
            ts_train = fold["train_series"]
            y_true = fold["y"]  # shape (D, pred_len)
            D = y_true.shape[0]

            comp_losses = {}
            comp_preds = []

            for i, name in enumerate(darts_dl.feature_names):
                ts_comp = ts_train.univariate_component(name)
                pdb.set_trace()

                if use_cached_orders:
                    ords = orders_cache[name]
                    model = ARIMA(
                        order=ords["order"], seasonal_order=ords["seasonal_order"]
                    )
                else:
                    model = ARIMA(p=1, d=1, q=1, seasonal_order=(1, 1, 1, 12))
                    # model = AutoARIMA(seasonal=seasonal, season_length=m)

                model.fit(ts_comp)
                y_pred_ts = model.predict(n=darts_dl.pred_len)  # univariate TimeSeries
                y_pred = y_pred_ts.values().ravel()  # (pred_len,)
                pdb.set_trace()
                plt.figure(figsize=(12, 6))

                # Plot actual values
                plt.plot(y_true[i], label=f"Actual {name}", color="blue", linewidth=2)

                # Plot predictions
                plt.plot(
                    y_pred,
                    label=f"Predicted {name}",
                    color="red",
                    linestyle="--",
                    linewidth=2,
                )

                plt.title(f"ARIMA Predictions vs Actual - {name} (Window {fold['k']})")
                plt.xlabel("Time Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save the plot
                plt.savefig(
                    f"arima_prediction_{name}_window_{fold['k']}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.show()  # Remove this if you don't want to display interactively

                comp_preds.append(y_pred)
                pdb.set_trace()

                comp_losses[name] = _loss(y_pred, y_true[i])
                print(i)

            comp_preds = np.stack(comp_preds, axis=0)  # (D, pred_len)
            avg_loss = float(np.mean(list(comp_losses.values())))
            window_losses.append(avg_loss)
            details.append(
                {
                    "k": fold["k"],
                    "in_start": fold["in_start"],
                    "in_end": fold["in_end"],
                    "out_end": fold["out_end"],
                    "component_losses": comp_losses,
                    "avg_loss": avg_loss,
                }
            )

        overall_avg_loss = float(np.mean(window_losses)) if window_losses else np.nan
        return overall_avg_loss, window_losses, details

    def evaluate(self, test_set="test"):
        """Evaluate ARIMA model"""

        self.logger.info(f"Evaluating on {mode} set...")

        start_time = time.time()

        if mode == "val":
            input_series = self.train_series
            label_series = self.val_series
        else:  # test
            # Use train + val for test prediction
            input_series = self.train_series.append(self.val_series)
            label_series = self.test_series

            for feature_name in pbar:
                pbar.set_description(f"Refitting {feature_name}")
                univariate_series = historical_series.univariate_component(feature_name)
                self.models[feature_name].fit(univariate_series)

            pbar.close()

        # Generate predictions
        n_periods = len(true_series)
        predictions = self._predict_multivariate(n_periods, historical_series)

        end_time = time.time()

        # Compute metrics
        mse_score = mse(true_series, predictions)
        rmse_score = rmse(true_series, predictions)
        mae_score = mae(true_series, predictions)

        try:
            mape_score = mape(true_series, predictions)
            if np.isinf(mape_score) or np.isnan(mape_score):
                mape_score = float("inf")
        except:
            mape_score = float("inf")

        self.logger.info(f"{mode.upper()} Results:")
        self.logger.info(
            f"MSE: {mse_score:.6f}, RMSE: {rmse_score:.6f}, MAE: {mae_score:.6f}, MAPE: {mape_score:.4f}%"
        )
        self.logger.info(f"Evaluation time: {end_time - start_time:.4f}s")

        # Compute per-feature metrics with progress bar
        self.logger.info("Computing per-feature metrics...")
        pbar = tqdm(self.feature_names, desc="Computing metrics", unit="feature")

        feature_results = []
        for feature_name in pbar:
            pbar.set_description(f"Metrics for {feature_name}")

            true_feature = true_series.univariate_component(feature_name)
            pred_feature = predictions.univariate_component(feature_name)

            feature_mse = mse(true_feature, pred_feature)
            feature_rmse = rmse(true_feature, pred_feature)
            feature_mae = mae(true_feature, pred_feature)

            try:
                feature_mape = mape(true_feature, pred_feature)
                if np.isinf(feature_mape) or np.isnan(feature_mape):
                    feature_mape = float("inf")
            except:
                feature_mape = float("inf")

            feature_results.append(
                {
                    "feature": feature_name,
                    "mse": feature_mse,
                    "rmse": feature_rmse,
                    "mae": feature_mae,
                    "mape": feature_mape,
                }
            )

        pbar.close()

        # Log per-feature results
        self.logger.info("Per-feature results:")
        for result in feature_results:
            self.logger.info(
                f"  {result['feature']}: MSE={result['mse']:.6f}, "
                f"RMSE={result['rmse']:.6f}, MAE={result['mae']:.6f}, "
                f"MAPE={result['mape']:.4f}%"
            )

        # Save predictions
        predictions_path = self.log_dir / f"{mode}_predictions.csv"
        predictions.to_dataframe().to_csv(predictions_path)
        self.logger.info(f"Predictions saved to {predictions_path}")
        return mse_score, rmse_score, mae_score, mape_score

    def plot_results(self, predictions_df):
        """Plot results using predictions DataFrame and true test values."""
        # Extract predictions from the DataFrame
        preds = predictions_df.drop(
            columns=["Unnamed: 0"]
        ).values  # Drop index column and convert to numpy array

        # Extract true test values from the test_series
        true_labels = (
            self.test_series.values()
        )  # Assuming test_series is a pandas DataFrame

        # Convert to appropriate formats if necessary (e.g., tensors or numpy arrays)
        preds = np.array(preds)
        labels = np.array(true_labels)

        # Ensure preds and labels are in the correct shape for plotting
        # For example, if you want to plot mean across sensors
        # mean_preds = preds.mean(axis=0, keepdims=True)  # Mean predictions across rows
        # mean_labels = labels.mean(axis=0, keepdims=True)  # Mean labels across rows
        #
        # # Plot mean per day
        # plot_mean_per_day(
        #     mean_preds.flatten(),  # Flatten to 1D array for plotting
        #     mean_labels.flatten(),  # Flatten to 1D array for plotting
        #     self._plot_path,
        #     "mean_per_day_performance_plot.png"
        # )

        # Plot mean branch for first 5 and first 100 predictions
        pdb.set_trace()
        # mean_branch_plot(
        #     preds[:5, :],  # Adding a new axis for compatibility
        #     labels[:5, :],  # Adding a new axis for compatibility
        #     self._plot_path,
        #     "mean_performance_plot_first_5.png"
        # )
        # mean_branch_plot(
        #     preds[:100, :],  # Adding a new axis for compatibility
        #     labels[:100, :],  # Adding a new axis for compatibility
        #     self._plot_path,
        #     "mean_performance_plot_first_100.png"
        # )

        # Assuming var_index is defined or passed as an argument
        var_index = 0  # Adjust this to the appropriate variable index
        branch_plot(
            preds[:5, :, np.newaxis],
            labels[:5, :, np.newaxis],
            var_index,
            self._plot_path,
            f"sensor_{var_index}_branch_plot_first_5.png",
        )
        branch_plot(
            preds[:100, :, np.newaxis],
            labels[:100, :, np.newaxis],
            var_index,
            self._plot_path,
            f"sensor_{var_index}_branch_plot_first_100.png",
        )
