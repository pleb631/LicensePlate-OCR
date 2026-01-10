import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np


from lprec import models
from .datasets import LicensePlateUtils

__all__ = ["PLModel"]


def WarmupLR(optimizer, T_max, warmup_epoch, lr, LRf):
    # lambda0 = lambda cur_iter: \
    #     (0.1*ETA_MIN_LR+(lr-0.1*ETA_MIN_LR)*(cur_iter/warmup_epoch))/lr \
    #         if cur_iter<warmup_epoch else \
    #     (ETA_MIN_LR+0.5*(lr-ETA_MIN_LR)*(1.0+math.cos((cur_iter-warmup_epoch)/(T_max-warmup_epoch)*math.pi)))/lr
    # return optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda0)
    ETA_MIN_LR = lr * (LRf ** len(T_max))
    print(ETA_MIN_LR)
    lambda0 = (
        lambda cur_iter: (ETA_MIN_LR + (lr - ETA_MIN_LR) * (cur_iter / warmup_epoch))
        / lr
        if cur_iter < warmup_epoch
        else LRf ** sum(1 for i in T_max if cur_iter > i)
    )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)


class Optimizer(object):
    def __init__(self, class_path, args=[], kwargs={}):
        self.class_path = class_path
        self.args = args
        self.kwargs = kwargs

    def __call__(self, parameters=None, optimizer=None):
        if parameters:
            return eval(self.class_path)(parameters, *self.args, **self.kwargs)
        return eval(self.class_path)(optimizer, *self.args, **self.kwargs)


class Loss(nn.Module):
    def __init__(self, blank,th=0.1,top_k_ratio=0.3):
        super().__init__()
        self.loss = nn.CTCLoss(blank, zero_infinity=True, reduction="none")
        self.top_k_ratio = top_k_ratio
        self.th = th

    def forward(self, x, batch, input_lengths, use_ohem=False):
        loss = self.loss(
            x,
            batch["label_tensor"],
            input_lengths=input_lengths,
            target_lengths=batch["label_length"],
        )
        loss = loss / batch["label_length"].float()
        if not use_ohem:
            return loss.mean()
        
        sorted_loss, _ = torch.sort(loss, descending=True)
        num_above = (sorted_loss >= self.th).sum().item()

        k = max(num_above, int(self.top_k_ratio * loss.size(0)))
        k = max(1, k)

        selected_loss = sorted_loss[:k]

        return selected_loss.mean()


class PLModel(pl.LightningModule):
    def __init__(
        self,
        model_type="LPRNet",
        optimizer: Optimizer = None,
        lr_scheduler=None,
        checkpoint=None,
        blank=0,
        num_classes=68,
        aux_loss=False,
        dropout=0.0,
        se=False,
        th=0.1,
        use_ohem=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        if checkpoint is not None:
            checkpoint = self.load_from_checkpoint(checkpoint)
            self.model = checkpoint.model
            self.criterion = checkpoint.criterion
        else:
            self.model = getattr(models, model_type)(
                num_classes=num_classes, dropout=dropout, se=se, **kwargs
            )
            print(self.model)
            self.criterion = Loss(blank,th=th)
            self.ce_loss = None
            if aux_loss:
                self.ce_loss = nn.CrossEntropyLoss()
        self.val_outputs = []
        self.use_ohem=use_ohem

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        batch_size = len(batch["label"])
        y_hat = self.model(batch["plate_image_tensor"]).squeeze()
        y_hat = torch.log_softmax(y_hat, dim=1)
        input_lengths = torch.full(
            size=(y_hat.size(0),),
            fill_value=y_hat.size(2),
            dtype=torch.long,
            device=y_hat.device,
        )
        loss = self.criterion(
            torch.permute(y_hat, (2, 0, 1)), batch, input_lengths, self.use_ohem
        )
        self.log_dict(
            {"train/loss": loss}, batch_size=batch_size, prog_bar=True, sync_dist=True
        )
        self.log_dict(
            {"train/lr": float(self.optimizer.param_groups[0]["lr"])},
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        correct, total = self._accuracy(y_hat, batch["label"])
        self.log_dict(
            {"train/acc": correct / total},
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["plate_image_tensor"]).squeeze()
        y_hat = torch.log_softmax(y_hat, dim=1)
        input_lengths = torch.full(
            size=(y_hat.size(0),),
            fill_value=y_hat.size(2),
            dtype=torch.long,
            device=y_hat.device,
        )
        loss = self.criterion(
            torch.permute(y_hat, (2, 0, 1)), batch, input_lengths, False
        )
        correct, total = self._accuracy(y_hat, batch["label"])
        self.val_outputs.append(
            {"correct": correct, "total": total, "val/loss": loss.item()}
        )

    def on_validation_epoch_end(self) -> None:
        correct = np.sum([x["correct"] for x in self.val_outputs])
        total = np.sum([x["total"] for x in self.val_outputs])
        loss = np.mean([x["val/loss"] for x in self.val_outputs])
        self.log_dict({"val/loss": loss, "val/acc": correct / total}, sync_dist=True)
        self.val_outputs.clear()

    def configure_optimizers(self):
        if self.hparams.optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = self.hparams.optimizer(self.model.parameters())
        if self.hparams.lr_scheduler is None:
            return self.optimizer
        else:
            self.lr_scheduler = self.hparams.lr_scheduler(self.optimizer)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "monitor": "val/loss",
        }

    def _accuracy(self, y_hat, y):
        tp = 0
        for output, label in zip(y_hat.detach().cpu(), y):
            predict = LicensePlateUtils.decode(output)
            if predict == label:
                tp += 1
        return tp, len(y)
