import os

from pytorch_lightning.cli import LightningCLI

from src.models.downscale_module import DownscaleLitModule
from src.datamodules.era5_single_task_datamodule import ERA5SingleTaskDataModule

os.environ["NCCL_P2P_DISABLE"] = "1"


def main():
    cli = LightningCLI(
        model_class=DownscaleLitModule,
        datamodule_class=ERA5SingleTaskDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        # auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.get_out_transforms()
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path='best')


if __name__ == "__main__":
    main()
