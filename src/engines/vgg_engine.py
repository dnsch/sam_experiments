import torch
import time
import numpy as np
from src.base.torch_engine import TorchEngine
from src.utils.functions import (
    plot_train_val_loss,
    plot_loss_metric,
)


class VGG_Engine(TorchEngine):
    """
    VGG pytorch trainer based off:
    https://github.com/tomgoldstein/loss-landscape/blob/master/cifar10/main.py
    """

    def __init__(
        self,
        batch_size=1,
        no_sam=True,
        **args,
    ):
        super(VGG_Engine, self).__init__(**args)
        self.batch_size = batch_size
        self.no_sam = no_sam
        self.random_state = self._seed
        self._timeout = True
        self._epochs = 0
        self._plot_path = self._save_path / "plots"

    def train_batch(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        if isinstance(self._loss_fn, torch.nn.CrossEntropyLoss):
            for batch_idx, data in enumerate(self._dataloader["train_loader"]):
                x_batch, y_batch = data

                x_batch = self._to_device(x_batch)
                y_batch = self._to_device(y_batch)
                out_batch = self.model(x_batch)

                total += self.batch_size

                loss = self._loss_fn(out_batch, y_batch)
                if not self.no_sam:
                    loss.backward()
                    self._optimizer.first_step(zero_grad=True)

                    out_batch = self.model(x_batch)
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
                train_loss += loss.item() * self.batch_size
                _, predicted = torch.max(out_batch.data, 1)
                correct += predicted.eq(y_batch.data).cpu().sum().item()

        return train_loss / total, 100 - 100.0 * correct / total

    def train(self):
        self._logger.info("Start training!")

        wait = 0
        min_loss = np.inf
        total_train_loss = []
        total_valid_loss = []
        b1 = time.time()
        for epoch in range(self._max_epochs):
            t1 = time.time()
            loss, train_err = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            test_loss, test_err = self.evaluate("val")
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = "Epoch: {:03d}, Train Loss: {:.4f}, Train Err: {:.4f}, Test Loss: {:.4f}, Test Err: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}"
            self._logger.info(
                message.format(
                    epoch + 1,
                    loss,
                    train_err,
                    test_loss,
                    test_err,
                    (t2 - t1),
                    (v2 - v1),
                    cur_lr,
                )
            )

            total_train_loss.append(loss)
            total_valid_loss.append(test_loss)
            model_list_save_path = self._save_path / "saved_models/"
            self.save_current_model(model_list_save_path, epoch)
            if test_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info(
                    "Val loss decrease from {:.4f} to {:.4f}".format(min_loss, test_loss)
                )
                min_loss = test_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        "Early stop at epoch {}, loss = {:.6f}".format(epoch + 1, min_loss)
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
            plot_train_val_loss(
                total_train_loss,
                total_valid_loss,
                "CrossEntropyLoss",
                self._epochs,
                plot_path=self._plot_path,
            )
            plot_loss_metric(
                total_train_loss,
                self._epochs,
                "train_CrossEntropyLoss",
                color="tab:blue",
                plot_path=self._plot_path,
            )
            plot_loss_metric(
                total_valid_loss,
                self._epochs,
                "validation_CrossEntropyLoss",
                color="tab:blue",
                plot_path=self._plot_path,
            )
        except:
            print("plotting not successful")

        self.evaluate("test")

    def evaluate(self, mode):
        if mode == "test":
            self.load_model(self._save_path)
        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            if isinstance(self._loss_fn, torch.nn.CrossEntropyLoss):
                for batch_idx, data in enumerate(self._dataloader["test_loader"]):
                    x_batch, y_batch = data
                    x_batch = self._to_device(x_batch)
                    y_batch = self._to_device(y_batch)
                    out_batch = self.model(x_batch)
                    total += self.batch_size
                    loss = self._loss_fn(out_batch, y_batch)
                    test_loss += loss.item() * self.batch_size
                    _, predicted = torch.max(out_batch.data, 1)
                    correct += predicted.eq(y_batch.data).cpu().sum().item()

        if mode == "val":
            return test_loss / total, 100 - 100.0 * correct / total
        elif mode == "test":
            return test_loss / total, 100 - 100.0 * correct / total
