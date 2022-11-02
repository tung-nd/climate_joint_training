import os

from pytorch_lightning.utilities.cli import LightningCLI

from src.models.joint_module import JointLitModule
from src.datamodules.era5_joint_datamodule import ERA5JointDataModule


def main():
    cli = LightningCLI(
        model_class=JointLitModule,
        datamodule_class=ERA5JointDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    # set denormalization for forecast
    forecast_normalization = cli.datamodule.get_out_forecast_transforms()
    forecast_mean_norm, forecast_std_norm = forecast_normalization.mean, forecast_normalization.std
    forecast_mean_denorm, forecast_std_denorm = -forecast_mean_norm / forecast_std_norm, 1 / forecast_std_norm
    cli.model.set_denormalization_forecast(forecast_mean_denorm, forecast_std_denorm)
    
    # set denormalization for downscale
    downscale_normalization = cli.datamodule.get_out_downscale_transforms()
    downscale_mean_norm, downscale_std_norm = downscale_normalization.mean, downscale_normalization.std
    downscale_mean_denorm, downscale_std_denorm = -downscale_mean_norm / downscale_std_norm, 1 / downscale_std_norm
    cli.model.set_denormalization_downscale(downscale_mean_denorm, downscale_std_denorm)
    
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.pred_range)
    cli.model.set_val_climatology(cli.datamodule.get_climatology(split='val'))
    cli.model.set_test_climatology(cli.datamodule.get_climatology(split='test'))

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
