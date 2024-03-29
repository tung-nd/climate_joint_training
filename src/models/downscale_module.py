from audioop import bias
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import mse, mse_val, rmse, pearson, mean_bias


def interpolate_input(x: torch.Tensor, y: torch.Tensor):
    # interpolate input to match output size
    n, t, _, _, _ = x.shape
    x = x.flatten(0, 1)
    out_h, out_w = y.shape[-2], y.shape[-1]
    x = torch.nn.functional.interpolate(x, (out_h, out_w), mode="bilinear")
    x = x.unflatten(0, sizes=(n, t))
    return x


class DownscaleLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: str = 'adam',
        lr: float = 0.001,
        weight_decay: float = 0.005,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if optimizer == 'adam':
            self.optim_cls = torch.optim.Adam
        elif optimizer == 'adamw':
            self.optim_cls = torch.optim.AdamW
        else:
            raise NotImplementedError('Only support Adam and AdamW')

    def forward(self, x):
        return self.net.predict(x)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        x = interpolate_input(x, y)
        
        loss_dict, _ = self.net.forward(x, y, variables, out_variables, [mse], lat=None)
        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        x = interpolate_input(x, y)

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[mse_val, rmse, pearson, mean_bias],
            lat=None,
            clim=None,
            log_postfix=None,
        )
        
        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size = len(x)
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        x = interpolate_input(x, y)
        
        all_loss_dicts = self.net.evaluate(
            x,
            y,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[mse_val, rmse, pearson, mean_bias],
            lat=None,
            clim=None,
            log_postfix=None,
        )
        
        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size = len(x)
            )
        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = self.optim_cls(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {"params": no_decay, "lr": self.hparams.lr, "weight_decay": 0},
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
