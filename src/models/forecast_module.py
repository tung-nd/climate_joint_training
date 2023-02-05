from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
)


class ForecastLitModule(LightningModule):
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

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_climatology(self, clim):
        self.val_clim = clim

    def set_test_climatology(self, clim):
        self.test_clim = clim

    def training_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        loss_dict, _ = self.net.forward(x, y, variables, out_variables, [lat_weighted_mse], lat=self.lat)
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
        
        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix=log_postfix,
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
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        
        pred_steps = 1
        log_steps = [1]
        log_days = [int(self.pred_range / 24)]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            self.test_clim,
            variables,
            out_variables,
            pred_steps,
            [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            self.denormalization,
            lat=self.lat,
            log_steps=log_steps,
            log_days=log_days,
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
