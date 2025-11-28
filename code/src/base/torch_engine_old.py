from pathlib import Path
import time
import torch
import numpy as np

from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics


class TorchEngine:
    def __init__(
        self,
        device,
        model,
        dataloader,
        scaler,
        loss_fn,
        lrate,
        optimizer,
        scheduler,
        clip_grad_value,
        max_epochs,
        patience,
        log_dir,
        logger,
        seed,
    ):
        # super().__init__()
        self._device = device
        self.model = model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        # self._save_path = (
        #     Path(log_dir) if Path(log_dir).is_absolute() else SCRIPT_DIR / log_dir
        # )
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed

        self._logger.info(f"The number of parameters: {self.model.param_num()}")

    def _to_device(self, tensors):
        return (
            [tensor.to(self._device) for tensor in tensors]
            if isinstance(tensors, list)
            else tensors.to(self._device)
        )

    def _to_numpy(self, tensors):
        return (
            [tensor.detach().cpu().numpy() for tensor in tensors]
            if isinstance(tensors, list)
            else tensors.detach().cpu().numpy()
        )

    def _to_tensor(self, nparray):
        return (
            [torch.tensor(array, dtype=torch.float32) for array in nparray]
            if isinstance(nparray, list)
            else torch.tensor(nparray, dtype=torch.float32)
        )

    def _inverse_transform(self, tensors):
        inv = lambda tensor: self._scaler.inverse_transform(tensor)
        return (
            [inv(tensor) for tensor in tensors]
            if isinstance(tensors, list)
            else inv(tensors)
        )

    def save_current_model(self, save_path, epoch):
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f"model_{epoch}.pt"
        torch.save(self.model.state_dict(), filename)

    def save_model(self, save_path):
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f"final_model_s{self._seed}.pt"
        torch.save(self.model.state_dict(), filename)

    def load_model(self, save_path):
        filename = save_path / f"final_model_s{self._seed}.pt"
        self.model.load_state_dict(torch.load(filename))

    def return_save_model(self, save_path):
        filename = save_path / f"final_model_s{self._seed}.pt"
        self.model.load_state_dict(torch.load(filename))
        return self.model

    def return_specific_model(self, save_path):
        self.model.load_state_dict(torch.load(save_path))
        return self.model

    def train_batch(self):
        self.model.train()
        train_loss, train_mape, train_rmse = [], [], []
        self._dataloader["train_loader"].shuffle()

        for X, label in self._dataloader["train_loader"].get_iterator():
            self._optimizer.zero_grad()
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print("Check mask value", mask_value)

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            self._iter_cnt += 1

        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    def train(self):
        self._logger.info("Start training!")
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

    def evaluate(self, mode):
        if mode == "test":
            self.load_model(self._save_path)
        self.model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for X, label in self._dataloader[f"{mode}_loader"].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == "val":
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == "test":
            test_mae, test_mape, test_rmse = [], [], []
            print("Check mask value", mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                self._logger.info(
                    f"Horizon {i + 1}, Test MAE: {res[0]:.4f}, Test RMSE: {res[2]:.4f}, Test MAPE: {res[1]:.4f}"
                )
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            self._logger.info(
                f"Average Test MAE: {np.mean(test_mae):.4f}, Test RMSE: {np.mean(test_rmse):.4f}, Test MAPE: {np.mean(test_mape):.4f}"
            )
