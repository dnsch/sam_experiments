import torch
import time
import numpy as np
from src.base.torch_engine import TorchEngine
from src.utils.functions import (
    plot_stats,
    branch_plot,
    plot_mean_per_day,
    mean_branch_plot,
    plot_samformer_attention_mean,
    plot_samformer_attention_mean_stats,
    plot_samformer_attention_variance,
)
from torchmetrics.regression import MeanAbsolutePercentageError, MeanSquaredError

from typing import Tuple, List, Dict


class SAMFormer_Engine(TorchEngine):
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
        plot_attention=False,
        **args,
    ):
        super(SAMFormer_Engine, self).__init__(**args)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.no_sam = no_sam
        self.use_revin = use_revin
        self.gsam = gsam
        self.plot_attention = plot_attention
        self.random_state = self._seed
        self._timeout = True
        self._epochs = 0
        self._plot_path = self._save_path / "plots"
        # Initialize metrics
        self._mape = self._to_device(MeanAbsolutePercentageError())
        self._rmse = self._to_device(MeanSquaredError(squared=False))

    def train_batch(self):
        self.model.train()
        train_loss = []
        train_mape = []
        train_rmse = []
        if self.plot_attention:
            attention_patterns_per_batch = []

        for batch_idx, data in enumerate(self._dataloader["train_loader"]):
            x_batch, y_batch = data

            x_batch = self._to_device(x_batch)
            y_batch = self._to_device(y_batch)

            out_batch = self.model(x_batch, True)

            attention_patterns_per_batch.append(self.model.attention_pattern)
            # attention_patterns_per_epoch.append(tensor.detach().cpu().numpy())

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
            return (
                np.mean(train_loss),
                np.mean(train_mape),
                np.mean(train_rmse),
                cur_lr,
                attention_patterns_per_batch,
            )
        else:
            return (
                np.mean(train_loss),
                np.mean(train_mape),
                np.mean(train_rmse),
                attention_patterns_per_batch,
            )

        # if self.gsam:
        #     return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), cur_lr
        # else:
        #     return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

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

        # Create attention plots directory
        attention_mean_plot_path = self._plot_path / "attention" / "mean"
        attention_mean_plot_path.mkdir(parents=True, exist_ok=True)
        attention_var_plot_path = self._plot_path / "attention" / "variance"
        attention_var_plot_path.mkdir(parents=True, exist_ok=True)

        # Frequency of attention plotting
        attention_plot_freq = 1

        attention_patterns_per_epoch_list = []

        b1 = time.time()
        for epoch in range(self._max_epochs):
            t1 = time.time()
            if self.gsam:
                (
                    mtrain_loss,
                    mtrain_mape,
                    mtrain_rmse,
                    cur_lr,
                    attention_patterns_per_epoch,
                ) = self.train_batch()
                self.model.attention_pattern.mean(axis=0).detach().cpu().numpy()
            else:
                mtrain_loss, mtrain_mape, mtrain_rmse, attention_patterns_per_epoch = (
                    self.train_batch()
                )

            attention_patterns_per_epoch_list.append(attention_patterns_per_epoch)
            # if self.gsam:
            #     mtrain_loss, mtrain_mape, mtrain_rmse, cur_lr = self.train_batch()
            # else:
            #     mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate("val")
            v2 = time.time()

            # Plot attention weights
            if epoch % attention_plot_freq == 0 or epoch == self._max_epochs - 1:
                try:
                    self.model.eval()

                    # Simple attention heatmap
                    plot_samformer_attention_mean(
                        attention_patterns_per_epoch,
                        epoch=epoch + 1,
                        save_path=attention_mean_plot_path,
                    )

                    # Detailed attention statistics
                    plot_samformer_attention_mean_stats(
                        attention_patterns_per_epoch,
                        epoch=epoch + 1,
                        save_path=attention_mean_plot_path,
                    )

                    # Simple attention heatmap
                    plot_samformer_attention_variance(
                        attention_patterns_per_epoch,
                        epoch=epoch + 1,
                        save_path=attention_var_plot_path,
                    )

                    self._logger.info(f"Attention plots saved for epoch {epoch + 1}")
                except Exception as e:
                    self._logger.warning(f"Failed to plot attention: {e}")

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

        result = self.evaluate("test")
        return result

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

            # # If we want to scale the values back
            # # TODO:not sure if we maybe should just also scale arima values to
            # # have a easier comparison
            # preds, labels = [
            #     self._scaler.inverse_transform(preds.squeeze()),
            #     self._scaler.inverse_transform(labels.squeeze()),
            # ]
            # preds, labels = [
            #     self._to_tensor(preds).unsqueeze(0),
            #     self._to_tensor(labels).unsqueeze(0),
            # ]

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
            # TODO: add possibility for other losses
            return np.mean(mean_per_day_mse)
