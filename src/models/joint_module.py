from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from src.models.components.resnet import ResNet
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import lat_weighted_acc, lat_weighted_mse, lat_weighted_rmse, mse, rmse, pearson, mean_bias


class JointLitModule(LightningModule):
    def __init__(
        self,
        forecast_net: ResNet,
        downscale_net: ResNet,
        optimize_forecast: bool = False, # additional loss on forecast
        optimizer: str = 'adam',
        lr: float = 0.001,
        weight_decay: float = 0.005,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["forecast_net", "downscale_net"])
        self.forecast_net = forecast_net
        self.downscale_net = downscale_net
        if optimizer == 'adam':
            self.optim_cls = torch.optim.Adam
        elif optimizer == 'adamw':
            self.optim_cls = torch.optim.AdamW
        else:
            raise NotImplementedError('Only support Adam and AdamW')

    def forward(self, x):
        forecasted = self.forecast_net.predict(x)
        return self.downscale_net.predict(forecasted)

    def set_denormalization_forecast(self, mean, std):
        self.forecast_denormalization = transforms.Normalize(mean, std)
        
    def set_denormalization_downscale(self, mean, std):
        self.downscale_denormalization = transforms.Normalize(mean, std)
        
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
        inp, out_forecast, out_downscale, _, out_forecast_vars, out_downscale_vars = batch
        
        # forecast 
        loss_forecast_dict, _, forecast_pred = self.forecast_net.forward(inp, out_forecast, out_forecast_vars, [lat_weighted_mse], lat=self.lat, return_pred=True)
        loss_forecast_dict = loss_forecast_dict[0]
        for var in loss_forecast_dict.keys():
            self.log(
                "train_joint/" + "forecast_" + var,
                loss_forecast_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            
        # downscale
        loss_downscale_dict, _ = self.downscale_net.forward(forecast_pred, out_downscale, out_downscale_vars, [mse], lat=None, return_pred=False)
        loss_downscale_dict = loss_downscale_dict[0]
        for var in loss_downscale_dict.keys():
            self.log(
                "train_joint/" + "downscale_" + var,
                loss_downscale_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        
        # final loss
        if self.hparams.optimize_forecast:
            loss = loss_forecast_dict["loss"] + loss_downscale_dict["loss"]
        else:
            loss = loss_downscale_dict["loss"]
            
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        inp, out_forecast, out_downscale, inp_vars, out_forecast_vars, out_downscale_vars = batch
        
        # forecast
        pred_steps = 1
        log_steps = [1]
        log_days = [int(self.pred_range / 24)]

        all_forecast_loss_dicts, forecast_pred = self.forecast_net.rollout(
            inp,
            out_forecast,
            self.val_clim,
            inp_vars,
            out_forecast_vars,
            pred_steps,
            [lat_weighted_rmse, lat_weighted_acc],
            self.forecast_denormalization,
            lat=self.lat,
            log_steps=log_steps,
            log_days=log_days,
        )
        loss_forecast_dict = {}
        for d in all_forecast_loss_dicts:
            for k in d.keys():
                loss_forecast_dict[k] = d[k]

        for var in loss_forecast_dict.keys():
            self.log(
                "val_joint/" + "forecast_" + var,
                loss_forecast_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            
        # downscale
        all_downscale_loss_dicts, _ = self.downscale_net.upsample(
            forecast_pred.squeeze(), out_downscale, out_downscale_vars, self.downscale_denormalization, [rmse, pearson, mean_bias]
        )
        loss_downscale_dict = {}
        for d in all_downscale_loss_dicts:
            for k in d.keys():
                loss_downscale_dict[k] = d[k]

        for var in loss_downscale_dict.keys():
            self.log(
                "val_joint/" + "downscale_" + var,
                loss_downscale_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def test_step(self, batch: Any, batch_idx: int):
        inp, out_forecast, out_downscale, inp_vars, out_forecast_vars, out_downscale_vars = batch
        
        # forecast
        pred_steps = 1
        log_steps = [1]
        log_days = [int(self.pred_range / 24)]

        all_forecast_loss_dicts, forecast_pred = self.forecast_net.rollout(
            inp,
            out_forecast,
            self.val_clim,
            inp_vars,
            out_forecast_vars,
            pred_steps,
            [lat_weighted_rmse, lat_weighted_acc],
            self.forecast_denormalization,
            lat=self.lat,
            log_steps=log_steps,
            log_days=log_days,
        )
        loss_forecast_dict = {}
        for d in all_forecast_loss_dicts:
            for k in d.keys():
                loss_forecast_dict[k] = d[k]

        for var in loss_forecast_dict.keys():
            self.log(
                "test_joint/" + "forecast_" + var,
                loss_forecast_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            
        # downscale
        all_downscale_loss_dicts, _ = self.downscale_net.upsample(
            forecast_pred.squeeze(), out_downscale, out_downscale_vars, self.downscale_denormalization, [rmse, pearson, mean_bias]
        )
        loss_downscale_dict = {}
        for d in all_downscale_loss_dicts:
            for k in d.keys():
                loss_downscale_dict[k] = d[k]

        for var in loss_downscale_dict.keys():
            self.log(
                "test_joint/" + "downscale_" + var,
                loss_downscale_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

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

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
