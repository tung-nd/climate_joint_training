import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .era5_dataset import ERA5Joint

def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out_forecast = torch.stack([batch[i][1] for i in range(len(batch))])
    out_downscale = torch.stack([batch[i][2] for i in range(len(batch))])
    vars = batch[0][3]
    out_forecast_vars = batch[0][4]
    out_downscale_vars = batch[0][5]
    return inp, out_forecast, out_downscale, vars, out_forecast_vars, out_downscale_vars

class ERA5JointDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        highres_root_dir,
        inp_vars,
        inp_pressure_levels,
        out_forecast_vars,
        out_forecast_pressure_levels,
        out_downscale_vars,
        out_downscale_pressure_levels,
        pred_range,
        train_start_year,
        val_start_year,
        test_start_year,
        end_year: int = 2018,
        subsample: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        assert end_year >= test_start_year and test_start_year > val_start_year and val_start_year > train_start_year

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        train_years = range(train_start_year, val_start_year)
        self.train_dataset = ERA5Joint(
            root_dir, highres_root_dir, inp_vars, inp_pressure_levels,
            out_forecast_vars, out_forecast_pressure_levels,
            out_downscale_vars, out_downscale_pressure_levels,
            pred_range, train_years, subsample, 'train'
        )

        val_years = range(val_start_year, test_start_year)
        self.val_dataset = ERA5Joint(
            root_dir, highres_root_dir, inp_vars, inp_pressure_levels,
            out_forecast_vars, out_forecast_pressure_levels,
            out_downscale_vars, out_downscale_pressure_levels,
            pred_range, val_years, subsample, 'val'
        )
        self.val_dataset.set_normalize(self.train_dataset.inp_transform, self.train_dataset.out_forecast_transform, self.train_dataset.out_downscale_transform)

        test_years = range(test_start_year, end_year + 1)
        self.test_dataset = ERA5Joint(
            root_dir, highres_root_dir, inp_vars, inp_pressure_levels,
            out_forecast_vars, out_forecast_pressure_levels,
            out_downscale_vars, out_downscale_pressure_levels,
            pred_range, test_years, subsample, 'test'
        )
        self.test_dataset.set_normalize(self.train_dataset.inp_transform, self.train_dataset.out_forecast_transform, self.train_dataset.out_downscale_transform)

    def get_lat_lon(self):
        return self.train_dataset.lat, self.train_dataset.lon

    def get_out_forecast_transforms(self):
        return self.train_dataset.out_forecast_transform
    
    def get_out_downscale_transforms(self):
        return self.train_dataset.out_downscale_transform

    def get_climatology(self, split='val'):
        if split == 'val':
            return self.val_dataset.get_climatology()
        elif split == 'test':
            return self.test_dataset.get_climatology()
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

# era5_module = ERA5DataModule(
#     root_dir='/datadrive/datasets/5.625deg',
#     inp_vars=['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind'],
#     out_vars=['2m_temperature'],
#     train_start_year=1979,
#     val_start_year=2015,
#     test_start_year=2017,
#     end_year=2018,
#     pred_range=6,
#     batch_size=64,
#     num_workers=1,
#     pin_memory=False
# )
# train_loader = era5_module.train_dataloader()
# for x, y in train_loader:
#     print (x.shape)
#     print (y.shape)
#     break