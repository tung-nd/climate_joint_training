import os
from typing import Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from .era5_iterdataset import NpyReader, Forecast, Downscale, IndividualDataIter, ShuffleIterableDataset


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    variables = batch[0][2]
    out_variables = batch[0][3]
    return inp, out, variables, out_variables


class ERA5SingleTaskDataModule(LightningDataModule):
    def __init__(
        self,
        task,
        inp_root_dir,
        out_root_dir,
        in_vars,
        out_vars,
        history: int = 1,
        window: int = 6,
        pred_range=6,
        subsample=1,
        buffer_size=10000,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        out_vars = out_vars if out_vars is not None else in_vars
        self.hparams.out_vars = out_vars

        if task == "forecasting":
            assert inp_root_dir == out_root_dir
            self.dataset_caller = Forecast
            self.dataset_arg = {
                "pred_range": pred_range,
                "history": history,
                "window": window,
            }
        else:  # downscaling
            self.dataset_caller = Downscale
            self.dataset_arg = {}

        self.inp_lister_train = list(
            dp.iter.FileLister(os.path.join(inp_root_dir, "train"))
        )
        self.out_lister_train = list(
            dp.iter.FileLister(os.path.join(out_root_dir, "train"))
        )

        self.inp_lister_val = list(
            dp.iter.FileLister(os.path.join(inp_root_dir, "val"))
        )
        self.out_lister_val = list(
            dp.iter.FileLister(os.path.join(out_root_dir, "val"))
        )

        self.inp_lister_test = list(
            dp.iter.FileLister(os.path.join(inp_root_dir, "test"))
        )
        self.out_lister_test = list(
            dp.iter.FileLister(os.path.join(out_root_dir, "test"))
        )

        self.transforms = self.get_normalize(inp_root_dir, in_vars)
        self.output_transforms = self.get_normalize(out_root_dir, out_vars)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.out_root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.out_root_dir, "lon.npy"))
        return lat, lon

    def get_normalize(self, root_dir, variables):
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        mean = []
        for var in variables:
            if var != "total_precipitation":
                mean.append(normalize_mean[var])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in variables])
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_out_transforms(self):
        return self.output_transforms

    def get_climatology(self, split="val"):
        path = os.path.join(self.hparams.out_root_dir, split, "climatology.npz")
        clim_dict = np.load(path)
        clim = np.concatenate([clim_dict[var] for var in self.hparams.out_vars])
        clim = torch.from_numpy(clim)
        return clim

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                IndividualDataIter(
                    self.dataset_caller(
                        NpyReader(
                            inp_file_list=self.inp_lister_train,
                            out_file_list=self.out_lister_train,
                            variables=self.hparams.in_vars,
                            out_variables=self.hparams.out_vars,
                            shuffle=True,
                        ),
                        **self.dataset_arg,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                ),
                buffer_size=self.hparams.buffer_size,
            )

            self.data_val = IndividualDataIter(
                self.dataset_caller(
                    NpyReader(
                        inp_file_list=self.inp_lister_val,
                        out_file_list=self.out_lister_val,
                        variables=self.hparams.in_vars,
                        out_variables=self.hparams.out_vars,
                        shuffle=False,
                    ),
                    **self.dataset_arg,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
            )

            self.data_test = IndividualDataIter(
                self.dataset_caller(
                    NpyReader(
                        inp_file_list=self.inp_lister_test,
                        out_file_list=self.out_lister_test,
                        variables=self.hparams.in_vars,
                        out_variables=self.hparams.out_vars,
                        shuffle=False,
                    ),
                    **self.dataset_arg,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )