import os
import xarray as xr
import glob
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.datamodules import NAME_TO_VAR, DEFAULT_PRESSURE_LEVELS, CONSTANTS, SINGLE_LEVEL_VARS, PRESSURE_LEVEL_VARS

def load_from_nc(data_dir, variables, pressure_levels, years):
    """
    data_dir: the data root directory
    variables: list of variables names
    pressure_levels: the pressure levels used for each variables
    """
    constant_names = [name for name in variables if NAME_TO_VAR[name] in CONSTANTS]
    ps = glob.glob(os.path.join(data_dir, 'constants', '*.nc'))
    if len(ps) == 0:
        constants = None
    else:
        all_constants = xr.open_mfdataset(ps, combine='by_coords')
        constants = {name: all_constants[NAME_TO_VAR[name]] for name in constant_names}

    non_const_names = [name for name in variables if name not in constant_names]
    data_dict = {}
    for name in non_const_names:
        if name in SINGLE_LEVEL_VARS:
            data_dict[name] = []
        elif name in PRESSURE_LEVEL_VARS:
            for level in pressure_levels[name]:
                data_dict[f'{name}_{level}'] = []
        else:
            raise NotImplementedError(f'{name} is not either in single-level or pressure-level dict')

    for year in tqdm(years):
        for name in non_const_names:
            dir_var = os.path.join(data_dir, name)
            ps = glob.glob(os.path.join(dir_var, f'*{year}*.nc'))
            xr_data = xr.open_mfdataset(ps, combine='by_coords')
            xr_data = xr_data[NAME_TO_VAR[name]]

            if len(xr_data.shape) == 3: # single level, 8760, 32, 64
                xr_data = xr_data.expand_dims(dim={'level': np.array([0])}, axis=1)
                data_dict[name].append(xr_data)
            else: # pressure level
                for level in pressure_levels[name]:
                    xr_data_level = xr_data.sel(level=[level])
                    data_dict[f'{name}_{level}'].append(xr_data_level)
    
    data_dict = {k: xr.concat(data_dict[k], dim='time') for k in data_dict.keys()}
    # precipitation and solar radiation miss a few data points in the beginning
    len_min = min([data_dict[k].shape[0] for k in data_dict.keys()])
    data_dict = {k: data_dict[k][-len_min:] for k in data_dict.keys()}
    
    return data_dict, constants

def get_lat_lon(data_dir, variables, years):
    # lat lon is stored in each of the nc files, just need to load one and extract
    dir_var = os.path.join(data_dir, variables[0])
    year = years[0]
    ps = glob.glob(os.path.join(dir_var, f'*{year}*.nc'))
    xr_data = xr.open_mfdataset(ps, combine='by_coords')
    lat = xr_data['lat'].to_numpy()
    lon = xr_data['lon'].to_numpy()
    return lat, lon

class ERA5Forecast(Dataset):
    def __init__(self, root_dir, in_vars, in_pressure_levels, out_vars, out_pressure_levels, pred_range, years, subsample=1, partition='train'):
        print (f'Creating {partition} dataset from netCDF files')
        super().__init__()
        
        self.root_dir = root_dir
        self.pred_range = pred_range
        self.years = years
        self.subsample = subsample
        
        self.data_dict, _ = load_from_nc(root_dir, in_vars, in_pressure_levels, years)
        
        in_vars_pressure = []
        for var in in_vars:
            if var in in_pressure_levels:
                for p in in_pressure_levels[var]:
                    in_vars_pressure.append(var + '_' + str(p))
            else:
                in_vars_pressure.append(var)
        self.in_vars = in_vars_pressure
        inp_data = xr.concat([self.data_dict[k] for k in in_vars_pressure], dim='level')
                
        out_vars_pressure = []
        for var in out_vars:
            if var in out_pressure_levels:
                for p in out_pressure_levels[var]:
                    out_vars_pressure.append(var + '_' + str(p))
            else:
                out_vars_pressure.append(var)
        self.out_vars = out_vars_pressure
        out_data = xr.concat([self.data_dict[k] for k in out_vars_pressure], dim='level')

        self.inp_data = inp_data[0 : -pred_range : subsample].to_numpy().astype(np.float32)
        self.out_data = out_data[pred_range::subsample].to_numpy().astype(np.float32)

        assert len(self.inp_data) == len(self.out_data)
        
        self.lat, self.lon = get_lat_lon(root_dir, in_vars, years)

        if partition == 'train':
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
        else:
            self.inp_transform = None
            self.out_transform = None

        del self.data_dict

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def get_climatology(self):
        return torch.from_numpy(self.out_data.mean(axis=0))

    def __getitem__(self, index):
        inp = torch.from_numpy(self.inp_data[index])
        out = torch.from_numpy(self.out_data[index])
        return self.inp_transform(inp), self.out_transform(out), self.in_vars, self.out_vars

    def __len__(self):
        return len(self.inp_data)
    

class ERA5Downscaling(Dataset):
    def __init__(self, root_dir, highres_root_dir, in_vars, in_pressure_levels, out_vars, out_pressure_levels, years, subsample=1, partition='train'):
        print (f'Creating {partition} dataset')
        super().__init__()
        
        self.root_dir = root_dir
        self.highres_root_dir = highres_root_dir
        self.in_vars = in_vars
        self.in_pressure_levels = in_pressure_levels
        self.out_vars = out_vars
        self.out_pressure_levels = out_pressure_levels
        
        self.data_dict, _ = load_from_nc(root_dir, in_vars, in_pressure_levels, years)
        self.highres_data_dict, _ = load_from_nc(highres_root_dir, out_vars, out_pressure_levels, years)

        in_vars_pressure = []
        for var in in_vars:
            if var in in_pressure_levels:
                for p in in_pressure_levels[var]:
                    in_vars_pressure.append(var + '_' + str(p))
            else:
                in_vars_pressure.append(var)
        self.in_vars = in_vars_pressure
        inp_data = xr.concat([self.data_dict[k] for k in in_vars_pressure], dim='level')
                
        out_vars_pressure = []
        for var in out_vars:
            if var in out_pressure_levels:
                for p in out_pressure_levels[var]:
                    out_vars_pressure.append(var + '_' + str(p))
            else:
                out_vars_pressure.append(var)
        self.out_vars = out_vars_pressure
        out_data = xr.concat([self.highres_data_dict[k] for k in out_vars_pressure], dim='level')

        self.inp_data = inp_data[::subsample].to_numpy().astype(np.float32)
        self.out_data = out_data[::subsample].to_numpy().astype(np.float32)

        assert len(self.inp_data) == len(self.out_data)
        
        self.lat, self.lon = get_lat_lon(root_dir, in_vars, years)

        self.downscale_ratio = self.out_data.shape[-1] // self.inp_data.shape[-1]

        if partition == 'train':
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
        else:
            self.inp_transform = None
            self.out_transform = None

        del self.data_dict
        del self.highres_data_dict

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def get_climatology(self):
        return torch.from_numpy(self.out_data.mean(axis=0))

    def __getitem__(self, index):
        inp = torch.from_numpy(self.inp_data[index])
        out = torch.from_numpy(self.out_data[index])
        return self.inp_transform(inp), self.out_transform(out), self.in_vars, self.out_vars

    def __len__(self):
        return len(self.inp_data)


# dataset = ERA5('/datadrive/datasets/5.625deg', ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential'], [1979, 1980])
# for k in dataset.data_dict.keys():
#     print (k)
#     print (dataset.data_dict[k].shape)
# x = dataset[0]
# print (x.shape)
# print (len(dataset))
# print (dataset.normalize_mean)
# print (dataset.normalize_std)

# dataset = ERA5Forecast('/datadrive/datasets/5.625deg', ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential'], ['2m_temperature'], 6, [1979, 1980], 'train')
# print (len(dataset))
# x, y = dataset[0]
# print (x.shape)
# print (y.shape)