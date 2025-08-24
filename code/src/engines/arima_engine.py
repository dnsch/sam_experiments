import torch
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

from darts.models import ARIMA, AutoARIMA, VARIMA
from darts.utils.timeseries_generation import holidays_timeseries
from darts.metrics import mse, rmse, mae, mape
from darts import TimeSeries
import pdb


# As the ARIMA model is not implemented in pytorch, it doesn't make sense that
# it inherits from the BaseEngine
class ARIMAEngine:
    """
    ARIMA forecasting engine using Darts library with multivariate support
    """

    def __init__(
        self, train_series, val_series, test_series, args, logger, log_dir, **kwargs
    ):
        self.train_series = train_series
        self.val_series = val_series
        self.test_series = test_series
        self.args = args
        self.logger = logger
        self.log_dir = log_dir
        self._plot_path = self.log_dir / "plots"
        self.models = {}  # Dictionary to store models for each feature
        self.pred_len = args.horizon
        self.feature_names = train_series.columns
        self.n_features = len(self.feature_names)

        self.logger.info(f"Detected {self.n_features} features: {self.feature_names}")

    def train(self):
        """Handles the model fitting based on user args"""
        if self.args.arima_type == "VARIMA":
            model = VARIMA()
            model_string = "VARIMA"
        elif self.args.arima_type == "AutoARIMA":
            model = AutoARIMA(
                start_p=0,
                start_q=0,
                max_p=self.args.max_p,
                max_q=self.args.max_q,
                max_d=self.args.max_d,
                seasonal=self.args.seasonal,
                start_P=0,
                start_Q=0,
                max_P=self.args.max_P,
                max_Q=self.args.max_Q,
                max_D=self.args.max_D,
                season_length=self.args.seasonal_periods if self.args.seasonal else 1,
                random_state=self.args.seed,
            )
            model_string = "AutoARIMA"
        else:
            if self.args.seasonal:
                seasonal_order = (
                    self.args.P,
                    self.args.D,
                    self.args.Q,
                    self.args.seasonal_periods,
                )
            else:
                seasonal_order = None

            model = ARIMA(
                p=self.args.p,
                d=self.args.d,
                q=self.args.q,
                seasonal_order=seasonal_order,
                random_state=self.args.seed,
            )
            model_string = "ARIMA"

        self.logger.info(f"Start {model_string} training...")

        if self.args.arima_type == "VARIMA":
            if self.args.arima_holidays:
                # Use future covariates
                # from https://unit8co.github.io/darts/generated_api/darts.models.forecasting.varima.html
                # "e.g. encode each timestep whether it is on a holiday"
                future_cov = holidays_timeseries(series.time_index, "CN", add_length=6)
                total_start_time = time.time()
                model.fit(series, future_covariates=future_cov)
            else:
                total_start_time = time.time()
                model.fit(series)

        else:
            # Fit separate model for each feature with progress bar
            pbar = tqdm(
                enumerate(self.feature_names),
                total=self.n_features,
                desc=f"Training {model_string} models",
                unit="feature",
            )

            total_start_time = time.time()
            for i, feature_name in pbar:
                # Update progress bar description
                pbar.set_description(f"Training {model_type} for {feature_name}")

                feature_start_time = time.time()

                # Extract univariate series for this feature
                univariate_series = self.train_series.univariate_component(feature_name)

                # Create and fit model
                model = self._create_arima_model()

                # Add a progress bar for model fitting
                fitting_pbar = tqdm(
                    total=1, desc=f"Fitting {feature_name}", leave=False
                )
                model.fit(univariate_series)
                fitting_pbar.update(1)
                fitting_pbar.close()

                # Store the model
                self.models[feature_name] = model

                feature_end_time = time.time()
                feature_duration = feature_end_time - feature_start_time

                # Update progress bar with timing info
                pbar.set_postfix(
                    {
                        "time": f"{feature_duration:.1f}s",
                        "avg_time": f"{(time.time() - total_start_time) / (i + 1):.1f}s",
                    }
                )

                self.logger.info(
                    f"Feature {feature_name} training completed in {feature_duration:.2f} seconds"
                )

            pbar.close()

        total_end_time = time.time()
        self.logger.info(
            f"All ARIMA models training completed in {total_end_time - total_start_time:.2f} seconds"
        )

        # Evaluate on validation set
        self.evaluate("val")

    def _predict_multivariate(self, n_periods, historical_series):
        """Generate predictions for all features"""
        predictions_dict = {}

        # Add progress bar for prediction phase
        pbar = tqdm(self.feature_names, desc="Generating predictions", unit="feature")

        for feature_name in pbar:
            pbar.set_description(f"Predicting {feature_name}")

            # Get univariate series for this feature
            univariate_series = historical_series.univariate_component(feature_name)

            # Generate prediction for this feature
            feature_prediction = self.models[feature_name].predict(n=n_periods)
            predictions_dict[feature_name] = feature_prediction.values()

        pbar.close()

        # Combine predictions into multivariate TimeSeries
        combined_values = np.column_stack(
            [predictions_dict[name] for name in self.feature_names]
        )
        combined_predictions = TimeSeries.from_values(
            combined_values, columns=self.feature_names
        )

        return combined_predictions

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
