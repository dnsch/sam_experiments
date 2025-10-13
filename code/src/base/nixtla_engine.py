from pathlib import Path
import time
import torch
import numpy as np

from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics
import pdb


class NixtlaEngine:
    def __init__(
        self,
        model,
        dataloader,
        pred_len,
        loss_fn,
        backend,
        log_dir,
        logger,
        seed,
        **kwargs,
    ):
        # super().__init__()
        self.model = model
        self._dataloader = dataloader
        self._loss_fn = loss_fn
        self.backend = backend
        self.pred_len = pred_len
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed

        if backend == "statsforecast":
            from statsforecast import StatsForecast

            # models = model if isinstance(model, (list, tuple)) else [model]
            # self._obj = StatsForecast(models=list(models), freq=freq, n_jobs=n_jobs)
            # self._fit_fn = lambda df, X: self._obj.fit(df=df, X_df=X)
            # self._forecast_fn = lambda h, X, level=None: self._obj.forecast(
            #     h=h, X_df=X, level=level
            # )

        elif backend == "neuralforecast":
            from neuralforecast import NeuralForecast

            models = model if isinstance(model, (list, tuple)) else [model]
            self._obj = NeuralForecast(models=list(models), freq=freq)
            # Some NF versions accept futr_df in fit; if not, drop it here.
            self._fit_fn = lambda df, X: self._obj.fit(df=df, val_size=0)
            self._forecast_fn = lambda h, X, level=None: self._obj.predict(
                h=h, futr_df=X
            )

        elif backend == "mlforecast":
            from mlforecast import MLForecast

            # model here is typically a regressor spec; MLForecast builds features internally.
            self._obj = (
                model
                if isinstance(model, MLForecast)
                else MLForecast(models=model, freq=freq)
            )
            # MLForecast uses different signatures:
            self._fit_fn = lambda df, X: self._obj.fit(
                df=df, id_col="unique_id", time_col="ds", target_col="y"
            )
            # Many workflows need a future_df with known exogenous/calendar for h steps
            self._forecast_fn = lambda h, X, level=None: self._obj.predict(
                h=h, new_df=X
            )

        else:
            raise ValueError(f"Unknown backend: {backend}")

    # utils
    def save_statsforecast_model(self, save_path, filename_suffix):
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f"model_{filename_suffix}.pkl"
        output_file = save_path / filename
        self.model.save(path=output_file, max_size=None, trim=None)

    def load_statsforecast_model(self, model_dir, filename_suffix):
        filename = model_dir / f"model_{filename_suffix}.pkl"
        loaded_model = self.model.load(filename)
        return loaded_model

    # Training
    def train(self):
        self._logger.info("Start training!")
        # TODO: turn this N here into a parameter, experiment with different
        # values to get a good trade-off between speed and performance
        N = 200  # tune based on accuracy/speed trade-off
        train_df = self._dataloader["train_loader"]
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
        filename_suffix = str(self.pred_len) + str(self._seed)
        self.save_statsforecast_model(
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

        wait, min_loss = 0, np.inf

        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate("val")
            v2 = time.time()

            cur_lr = (
                self._lrate
                if self._lr_scheduler is None
                else self._lr_scheduler.get_last_lr()[0]
            )
            if self._lr_scheduler:
                self._lr_scheduler.step()

            self._logger.info(
                f"Epoch: {epoch + 1:03d}, Train Loss: {mtrain_loss:.4f}, Train RMSE: {mtrain_rmse:.4f}, Train MAPE: {mtrain_mape:.4f}, "
                f"Valid Loss: {mvalid_loss:.4f}, Valid RMSE: {mvalid_rmse:.4f}, Valid MAPE: {mvalid_mape:.4f}, "
                f"Train Time: {(t2 - t1):.4f}s/epoch, Valid Time: {(v2 - v1):.4f}s, LR: {cur_lr:.4e}"
            )

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info(
                    f"Val loss decrease from {min_loss:.4f} to {mvalid_loss:.4f}"
                )
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        f"Early stop at epoch {epoch + 1}, loss = {min_loss:.6f}"
                    )
                    break

        self.evaluate("test")

    def fit(self, train_df, X_df=None):
        self._fit_fn(train_df, X_df)
        return self

    def forecast(self, h, X_df=None, level=None):
        return self._forecast_fn(h, X_df, level)
