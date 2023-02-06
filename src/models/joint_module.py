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
    mse, mse_val, rmse,
    pearson, mean_bias
)
from .downscale_module import interpolate_input


class JointLitModule(LightningModule):
    def __init__(
        self,
        forecast_net: torch.nn.Module,
        downscale_net: torch.nn.Module,
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
        
    def set_highres_lat_lon(self, lat, lon):
        self.lat_highres = lat
        self.lon_highres = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_climatology(self, clim):
        self.val_clim = clim

    def set_test_climatology(self, clim):
        self.test_clim = clim
        
    def set_val_climatology_highres(self, clim):
        self.val_clim_highres = clim

    def set_test_climatology_highres(self, clim):
        self.test_clim_highres = clim

    def training_step(self, batch: Any, batch_idx: int):
        inp, out_forecast, out_downscale, in_vars, out_downscale_vars = batch
        out_forecast_vars = in_vars
        
        # forecast 
        loss_forecast_dict, _, forecast_pred = self.forecast_net.forward(
            inp,
            out_forecast,
            in_vars,
            out_forecast_vars,
            [lat_weighted_mse],
            lat=self.lat,
            return_pred=True)
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
        forecast_pred = forecast_pred.unsqueeze(1)
        forecast_pred = interpolate_input(forecast_pred, out_downscale)
        loss_downscale_dict, _ = self.downscale_net.forward(
            forecast_pred,
            out_downscale,
            out_forecast_vars,
            out_downscale_vars,
            [lat_weighted_mse],
            lat=self.lat_highres,
            return_pred=False
        )
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
        inp, out_forecast, out_downscale, in_vars, out_downscale_vars = batch
        out_forecast_vars = in_vars
        
        # forecast
        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"
            
        all_forecast_loss_dicts, forecast_pred = self.forecast_net.evaluate(
            inp,
            out_forecast,
            in_vars,
            out_forecast_vars,
            transform=self.forecast_denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix=log_postfix,
            return_pred=True
        )
        loss_forecast_dict = {}
        for d in all_forecast_loss_dicts:
            for k in d.keys():
                loss_forecast_dict[k] = d[k]

        for var in loss_forecast_dict.keys():
            self.log(
                "val_joint/" + "lowres_" + var,
                loss_forecast_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            
        # downscale
        forecast_pred = forecast_pred.unsqueeze(1)
        forecast_pred = interpolate_input(forecast_pred, out_downscale)
        all_downscale_loss_dicts = self.downscale_net.evaluate(
            forecast_pred,
            out_downscale,
            out_forecast_vars,
            out_downscale_vars,
            transform=self.downscale_denormalization,
            metrics=[pearson, mean_bias, lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat_highres,
            clim=self.val_clim_highres,
            log_postfix=log_postfix,
        )
        loss_downscale_dict = {}
        for d in all_downscale_loss_dicts:
            for k in d.keys():
                loss_downscale_dict[k] = d[k]

        for var in loss_downscale_dict.keys():
            self.log(
                "val_joint/" + "highres_" + var,
                loss_downscale_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        

    def test_step(self, batch: Any, batch_idx: int):
        inp, out_forecast, out_downscale, in_vars, out_downscale_vars = batch
        out_forecast_vars = in_vars
        
        # forecast
        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"
            
        all_forecast_loss_dicts, forecast_pred = self.forecast_net.evaluate(
            inp,
            out_forecast,
            in_vars,
            out_forecast_vars,
            transform=self.forecast_denormalization,
            metrics=[pearson, mean_bias, lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
            return_pred=True
        )
        loss_forecast_dict = {}
        for d in all_forecast_loss_dicts:
            for k in d.keys():
                loss_forecast_dict[k] = d[k]

        for var in loss_forecast_dict.keys():
            self.log(
                "test_joint/" + "lowres_" + var,
                loss_forecast_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            
        # downscale
        forecast_pred = forecast_pred.unsqueeze(1)
        forecast_pred = interpolate_input(forecast_pred, out_downscale)
        all_downscale_loss_dicts = self.downscale_net.evaluate(
            forecast_pred,
            out_downscale,
            out_forecast_vars,
            out_downscale_vars,
            transform=self.downscale_denormalization,
            metrics=[mse_val, rmse, pearson, mean_bias, lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat_highres,
            clim=self.test_clim_highres,
            log_postfix=log_postfix,
        )
        loss_downscale_dict = {}
        for d in all_downscale_loss_dicts:
            for k in d.keys():
                loss_downscale_dict[k] = d[k]

        for var in loss_downscale_dict.keys():
            self.log(
                "test_joint/" + "highres_" + var,
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
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
