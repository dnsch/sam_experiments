import torch
import time
import numpy as np
from src.base.engine import BaseEngine
from src.utils.functions import (
    plot_stats,
    branch_plot,
    plot_mean_per_day,
    mean_branch_plot,
)

import pdb
from torchmetrics.regression import MeanAbsolutePercentageError, MeanSquaredError


class SAMFormer_Engine(BaseEngine):
    """
    SAMFormer pytorch trainer implemented in the sklearn fashion
    """

    def __init__(
        self,
        batch_size=1,
        num_channels=1,
        pred_len=1,
        no_sam=True,
        use_revin=True,
        gsam=False,
        **args,
    ):
        super(SAMFormer_Engine, self).__init__(**args)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.no_sam = no_sam
        self.use_revin = use_revin
        self.gsam = gsam
        self.random_state = self._seed
        self._timeout = True
        self._epochs = 0
        self._plot_path = self._save_path / "plots"
        # Initialize metrics
        self._mape = self._to_device(MeanAbsolutePercentageError())
        self._rmse = self._to_device(MeanSquaredError(squared=False))

    import numpy as np
    import torch

    def evaluate_samformer_on_windows(
        model,  # your trained Samformer model (in eval mode)
        sam_dl,  # instance of SamformerDataloader
        step: int,  # match ARIMA step (e.g., args.seq_len)
        metric: str = "MAE",
        device: str = "cuda",  # or "cpu"
    ) -> Tuple[float, List[float]]:
        """
        Assumes model returns predictions shaped like (batch, D*pred_len) or (batch, D, pred_len).
        Converts predictions back to raw units using sam_dl.scaler, then averages across components.
        """

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

        # Build non-overlapping sliding windows over scaled test segment
        test_loader = sam_dl.get_test_sliding_loader(
            stride=step, batch_size=1, flatten_y=False
        )

        model.eval()
        window_losses = []

        with torch.no_grad():
            for xb, yb_scaled in test_loader:
                # xb: (1, D, seq_len), yb_scaled: (1, D, pred_len) in scaled units
                xb = xb.to(device)
                # Forward pass
                yhat_scaled = model(
                    xb
                )  # ensure shape matches (1, D*pred_len) or (1, D, pred_len)

                # Reshape to (D, pred_len) in scaled units
                if yhat_scaled.dim() == 2:  # (1, D*pred_len)
                    D = yb_scaled.shape[1]
                    pred_len = yb_scaled.shape[2]
                    yhat_scaled = yhat_scaled.view(1, D, pred_len)
                yhat_scaled = yhat_scaled.squeeze(0).cpu().numpy()  # (D, pred_len)
                yb_scaled = yb_scaled.squeeze(0).cpu().numpy()  # (D, pred_len)

                # Inverse-transform both predictions and targets to raw units
                # StandardScaler expects shape (n_samples, D); we apply per time step
                D, pred_len = yhat_scaled.shape
                yhat_raw = np.zeros_like(yhat_scaled)
                y_true_raw = np.zeros_like(yb_scaled)
                for t in range(pred_len):
                    yhat_raw[:, t] = sam_dl.scaler.inverse_transform(
                        yhat_scaled[:, t].reshape(1, -1)
                    ).ravel()
                    y_true_raw[:, t] = sam_dl.scaler.inverse_transform(
                        yb_scaled[:, t].reshape(1, -1)
                    ).ravel()

                # Average loss across components
                loss = _loss(yhat_raw, y_true_raw)
                window_losses.append(loss)

        overall_avg_loss = float(np.mean(window_losses)) if window_losses else np.nan
        return overall_avg_loss, window_losses

    def train_batch(self):
        self.model.train()
        train_loss = []
        train_mape = []
        train_rmse = []

        for batch_idx, data in enumerate(self._dataloader["train_loader"]):
            x_batch, y_batch = data

            x_batch = self._to_device(x_batch)
            y_batch = self._to_device(y_batch)
            out_batch = self.model(x_batch, True)

            loss = self._loss_fn(out_batch, y_batch)
            mape = self._mape(out_batch, y_batch).item()
            rmse = self._rmse(out_batch, y_batch).item()
            cur_lr = 0
            if not self.no_sam:
                loss.backward()
                if self.gsam:
                    self._optimizer.set_closure(self._loss_fn, x_batch, y_batch)
                    out_batch, loss = self._optimizer.step()
                    cur_lr = self._lr_scheduler._last_lr[0]
                    self._lr_scheduler.step()
                    self._optimizer.update_rho_t()
                else:
                    self._optimizer.first_step(zero_grad=True)

                    out_batch = self.model(x_batch, True)
                    loss = self._loss_fn(out_batch, y_batch)

                    loss.backward()
                    if self._clip_grad_value != 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self._clip_grad_value
                        )
                    self._optimizer.second_step(zero_grad=True)
            else:
                self._optimizer.zero_grad()
                loss.backward()
                if self._clip_grad_value != 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self._clip_grad_value
                    )
                self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

        if self.gsam:
            return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), cur_lr
        else:
            return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    def train(self):
        self._logger.info("Start training!")

        wait = 0
        min_loss = np.inf
        total_train_loss = []
        total_train_mape = []
        total_train_rmse = []
        total_valid_loss = []
        total_valid_mape = []
        total_valid_rmse = []
        b1 = time.time()
        for epoch in range(self._max_epochs):
            t1 = time.time()
            if self.gsam:
                mtrain_loss, mtrain_mape, mtrain_rmse, cur_lr = self.train_batch()
            else:
                mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate("val")
            v2 = time.time()

            if self.gsam:
                cur_lr = cur_lr
            else:
                if self._lr_scheduler is None:
                    cur_lr = self._lrate
                else:
                    cur_lr = self._lr_scheduler.get_last_lr()[0]
                    self._lr_scheduler.step()

            message = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}"
            self._logger.info(
                message.format(
                    epoch + 1,
                    mtrain_loss,
                    mtrain_rmse,
                    mtrain_mape,
                    mvalid_loss,
                    mvalid_rmse,
                    mvalid_mape,
                    (t2 - t1),
                    (v2 - v1),
                    cur_lr,
                )
            )

            total_valid_loss.append(mvalid_loss)

            # added:
            total_train_loss.append(mtrain_loss)
            total_train_mape.append(mtrain_mape)
            total_train_rmse.append(mtrain_rmse)
            total_valid_mape.append(mvalid_mape)
            total_valid_rmse.append(mvalid_rmse)

            model_list_save_path = self._save_path / "saved_models/"
            self.save_current_model(model_list_save_path, epoch)
            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info(
                    "Val loss decrease from {:.4f} to {:.4f}".format(
                        min_loss, mvalid_loss
                    )
                )
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        "Early stop at epoch {}, loss = {:.6f}".format(
                            epoch + 1, min_loss
                        )
                    )
                    self._epochs = epoch + 1
                    break
            b2 = time.time()
            if self._timeout:
                if (b2 - b1) > (6 * 60 * 60):
                    print("Timeout reached, training stopped.")
                    print(f"{epoch + 1}")
                    self._epochs = epoch + 1
                    break

            self._epochs = epoch + 1
        try:
            plot_stats(
                total_train_loss,
                total_train_mape,
                total_train_rmse,
                total_valid_loss,
                total_valid_mape,
                total_valid_rmse,
                self._epochs,
                self._timeout,
                self._plot_path,
            )
        except:
            print("plotting not successful")

        self.evaluate("test")

    def evaluate(self, mode):
        if mode == "test":
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self._dataloader["val_loader"]):
                # X (b, t, n, f), label (b, t, n, 1)
                x_batch, y_batch = data
                x_batch, y_batch = self._to_device(self._to_tensor([x_batch, y_batch]))
                out_batch = self.model(x_batch, True)
                preds.append(out_batch.squeeze(-1).cpu())
                labels.append(y_batch.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        if mode == "val":
            mse = self._loss_fn(preds, labels).item()
            mape = self._mape(preds, labels).item()
            rmse = self._rmse(preds, labels).item()
            return mse, mape, rmse

        elif mode == "test":
            preds = []
            labels = []
            with torch.no_grad():
                for batch_idx, data in enumerate(self._dataloader["test_loader"]):
                    # X (b, t, n, f), label (b, t, n, 1)
                    X, label = data
                    X, label = self._to_device(self._to_tensor([X, label]))
                    out_batch = self.model(X, True)
                    preds.append(out_batch.cpu())
                    labels.append(label.cpu())

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            mse = self._loss_fn(preds, labels).item()
            mape = 0.0
            rmse = 0.0

            log = "Test MSE: {:.4f},"
            self._logger.info(log.format(mse))

            mean_per_day_mse = []
            per_day_preds = []
            per_day_labels = []
            preds, labels = [
                preds.reshape(preds.shape[0], self.num_channels, self.pred_len),
                labels.reshape(labels.shape[0], self.num_channels, self.pred_len),
            ]
            preds, labels = [
                torch.permute(preds, (0, 2, 1)),
                torch.permute(labels, (0, 2, 1)),
            ]
            for i in range(self.model.horizon):
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
            mean_branch_plot(
                preds[:5, :, :],
                labels[:5, :, :],
                self._plot_path,
                f"mean_performance_plot_first_5",
            )
            mean_branch_plot(
                preds[:100, :, :],
                labels[:100, :, :],
                self._plot_path,
                f"mean_performance_plot_first_100",
            )
            branch_plot(
                preds[:5, :, :],
                labels[:5, :, :],
                var_index,
                self._plot_path,
                f"sensor_{var_index}_branch_plot_first_5.png",
            )
            branch_plot(
                preds[:100, :, :],
                labels[:100, :, :],
                var_index,
                self._plot_path,
                f"sensor_{var_index}_branch_plot_first_100.png",
            )
