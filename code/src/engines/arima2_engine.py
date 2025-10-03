import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.utils.functions import (
    plot_stats,
    branch_plot,
    plot_mean_per_day,
    mean_branch_plot,
)

# from darts.metrics import mse, rmse, mae, mape
import pdb


class ARIMAEngine:
    """
    ARIMA forecasting engine using statsmodels
    """

    def __init__(self, dataloader, args, logger, log_dir, **kwargs):
        self.dataloader = dataloader
        self.logger = logger
        self.log_dir = log_dir
        self._plot_path = self.log_dir / "plots"
        # self.models = {}  # Dictionary to store models for each feature
        self.pred_len = args.horizon
        self.feature_names = train_series.columns
        self.n_features = len(self.feature_names)

        self.logger.info(f"Detected {self.n_features} features: {self.feature_names}")

    def train(self):
        """Handles the model fitting based on user args"""
        self.logger.info(f"Start training ARIMA models...")

        # Fit separate model for each feature with progress bar
        pbar = tqdm(
            enumerate(self.feature_names),
            total=self.n_features,
            desc="Training ARIMA models",
            unit="feature",
        )

        total_start_time = time.time()
        for i, feature_name in pbar:
            # Update progress bar description
            pbar.set_description(f"Training ARIMA for {feature_name}")

            feature_start_time = time.time()

            # Extract univariate series for this feature
            univariate_series = self.train_series[feature_name]

            # Create and fit model
            if self.args.seasonal:
                seasonal_order = (
                    self.args.P,
                    self.args.D,
                    self.args.Q,
                    self.args.seasonal_periods,
                )
                model = SARIMAX(
                    univariate_series,
                    order=(self.args.p, self.args.d, self.args.q),
                    seasonal_order=seasonal_order,
                )
            else:
                model = sm.tsa.ARIMA(
                    univariate_series, order=(self.args.p, self.args.d, self.args.q)
                )

            # Fit the model
            model_fit = model.fit()

            # Store the model
            self.models[feature_name] = model_fit

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
            univariate_series = historical_series[feature_name]

            # Generate prediction for this feature
            feature_prediction = self.models[feature_name].get_forecast(steps=n_periods)
            predictions_dict[feature_name] = feature_prediction.predicted_mean.values

        pbar.close()

        # Combine predictions into DataFrame
        combined_predictions = pd.DataFrame(predictions_dict)

        return combined_predictions

    def evaluate(self, test_set="test"):
        """Evaluate ARIMA model"""

        self.logger.info(f"Evaluating on {test_set} set...")

        start_time = time.time()

        if test_set == "val":
            input_series = self.train_series
            label_series = self.val_series
        else:  # test
            # Use train + val for test prediction
            input_series = self.train_series.append(self.val_series)
            label_series = self.test_series

        # Generate predictions
        n_periods = len(label_series)
        predictions = self._predict_multivariate(n_periods, input_series)

        end_time = time.time()

        # Compute metrics
        mse_score = mse(label_series, predictions)
        rmse_score = rmse(label_series, predictions)
        mae_score = mae(label_series, predictions)

        try:
            mape_score = mape(label_series, predictions)
            if np.isinf(mape_score) or np.isnan(mape_score):
                mape_score = float("inf")
        except:
            mape_score = float("inf")

        self.logger.info(f"{test_set.upper()} Results:")
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

            true_feature = label_series[feature_name]
            pred_feature = predictions[feature_name]

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
        predictions_path = self.log_dir / f"{test_set}_predictions.csv"
        predictions.to_csv(predictions_path)
        self.logger.info(f"Predictions saved to {predictions_path}")
        return mse_score, rmse_score, mae_score, mape_score

    def plot_results(self, predictions_df):
        """Plot results using predictions DataFrame and true test values."""
        # Extract predictions from the DataFrame
        preds = predictions_df.values  # Convert to numpy array
        true_labels = (
            self.test_series.values
        )  # Assuming test_series is a pandas DataFrame

        # Plot mean branch for first 5 and first 100 predictions
        var_index = 0  # Adjust this to the appropriate variable index
        branch_plot(
            preds[:5, np.newaxis],
            true_labels[:5, np.newaxis],
            var_index,
            self._plot_path,
            f"sensor_{var_index}_branch_plot_first_5.png",
        )
        branch_plot(
            preds[:100, np.newaxis],
            true_labels[:100, np.newaxis],
            var_index,
            self._plot_path,
            f"sensor_{var_index}_branch_plot_first_100.png",
        )
