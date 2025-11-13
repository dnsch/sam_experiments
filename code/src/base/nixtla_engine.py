from pathlib import Path
import time
import torch
import numpy as np

from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics
from src.utils.functions import statsforecast_to_tensor
from tqdm import tqdm
from src.utils.functions import (
    branch_plot,
    plot_mean_per_day,
    mean_branch_plot,
)


class NixtlaEngine:
    def __init__(
        self,
        model,
        dataloader,
        scaler=None,
        pred_len=1,
        loss_fn=None,
        backend=None,
        num_channels=1,
        log_dir=None,
        logger=None,
        seed=1,
        args=None,
        **kwargs,
    ):
        # super().__init__()
        self.model = model
        self._dataloader = dataloader
        self._scaler = scaler
        self._loss_fn = loss_fn
        self.backend = backend
        self.num_channels = num_channels
        self.pred_len = pred_len
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self._plot_path = self._save_path / "plots"
        self._plot_path.mkdir(parents=True, exist_ok=True)
        self.alias = args.alias

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
    def save_statsforecast_model(self):
        saved_models_path = self._save_path / "saved_models"
        saved_models_path.mkdir(parents=True, exist_ok=True)
        model_filename = f"model_seed_{self._seed}.pkl"
        output_file = saved_models_path / model_filename
        self.model.save(path=output_file, max_size=None, trim=None)

    def load_statsforecast_model(self):
        saved_models_path = self._save_path / "saved_models"
        filename = saved_models_path / f"model_seed_{self._seed}.pkl"
        self.model.load(filename)

    # Training
    def train(self):
        self._logger.info("Start fitting the model!")
        # TODO: turn this N here into a parameter, experiment with different
        # values to get a good trade-off between speed and performance
        N = 20  # tune based on accuracy/speed trade-off
        train_data = self._dataloader[0]

        # Fit the model
        t1 = time.time()
        self.model.fit(train_data)
        t2 = time.time()
        # TODO: check if arguments should be allowed for this function
        # maybe allow for different naming etc.
        self.save_statsforecast_model()
        # TODO: save that somewhere else and return it via a getter method
        result = self.evaluate("test")
        return result

    def evaluate(self, mode):
        if mode == "test":
            self.load_statsforecast_model()
            # self.load_model(self._save_path)

        preds = []
        labels = []

        if mode == "test":
            preds = []
            labels = []
            test_data = self._dataloader[1]
            predictions = self.model.predict(h=self.pred_len)
            # We only take the values and not the confidence intervals

            out_batch = statsforecast_to_tensor(predictions, self.alias, True)

            label = statsforecast_to_tensor(test_data, "y", True)

            preds.append(out_batch.cpu())
            labels.append(label.cpu())

            # Get the preds and labels in the correct format s.t. we can
            # compare them with the deep learning architecture forecasts
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            preds, labels = [
                preds.reshape(preds.shape[0], self.num_channels, self.pred_len),
                labels.reshape(labels.shape[0], self.num_channels, self.pred_len),
            ]
            preds, labels = [
                torch.permute(preds, (0, 2, 1)),
                torch.permute(labels, (0, 2, 1)),
            ]

            # TODO: change this based on the loss we chose in the args.
            mse = self._loss_fn(preds, labels).item()
            # TODO: add these metrics
            mape = 0.0
            rmse = 0.0

            log = "Test MSE: {:.4f},"
            self._logger.info(log.format(mse))

            mean_per_day_mse = []
            per_day_preds = []
            per_day_labels = []

            for i in range(self.pred_len):
                mse = self._loss_fn(preds[:, i, :], labels[:, i, :])
                log = "Horizon {:d}, Test MSE: {:.4f}"
                self._logger.info(log.format(i + 1, mse))
                mean_per_day_mse.append(mse)

                per_day_preds.append(preds[:, i, :].mean())
                per_day_labels.append(labels[:, i, :].mean())

            log = "Average per day Test MSE: {:.4f},"
            self._logger.info(log.format(np.mean(mean_per_day_mse)))
            var_index = 0
            plot_mean_per_day(
                per_day_preds,
                per_day_labels,
                self._plot_path,
                f"mean_per_day_performance_plot.png",
            )
            branch_plot(
                preds[:5, :, :],
                labels[:5, :, :],
                var_index,
                self._plot_path,
                f"sensor_{var_index}_branch_plot_first_5.png",
            )
            return np.mean(mean_per_day_mse)

    def fit(self, train_df, X_df=None):
        self._fit_fn(train_df, X_df)
        return self

    def forecast(self, h, X_df=None, level=None):
        return self._forecast_fn(h, X_df, level)
