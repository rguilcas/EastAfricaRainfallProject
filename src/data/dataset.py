"""
File documentation:
This file defines a custom PyTorch Dataset called AtmosphereToRainfallDataset , which is used for loading and preprocessing the dataset.
The dataset loads atmospheric predictor variables from either a single NetCDF file or a directory of per-variable NetCDF files, and rainfall target variables from NetCDF files using xarray.
It also applies any specified transformations to the input features and target variable.
"""

import glob
import os
from torch.utils.data import Dataset
import torch
import xarray as xr
import numpy as np
import pandas as pd

class AtmosphereToRainfallDataset(Dataset):
    """
    A custom PyTorch Dataset for loading atmospheric predictor variables and rainfall target variables from NetCDF files.
    Predictor inputs can come from a single file or from a directory where each file stores one variable.
    The dataset loads the data using xarray, applies any specified transformations, and returns samples in a format suitable for training a regression model.
    """
    def __init__(self, 
                 data_in_name='lowres_era5_features.nc', 
                 data_target_name='EastAfrica_mean_CHIRPS.nc', 
                 transform_X=None, transform_y=None,
                 data_path = '/Data/gfi/users/rogui7909/data/AfricaPrecip',
                 time_slice = (None,None),
                 forecast_horizon_days=0):
        """
        Initializes the dataset by loading the input and target data from NetCDF files, applying any specified transformations, and preparing the data for indexing.
        The dataset also aligns inputs and target data and determines their common time steps. The data are filtered accordingly.
        """
        ds_in = xr.open_dataarray(os.path.join(data_path, data_in_name))
        ds_target = xr.open_dataarray(os.path.join(data_path, data_target_name))
        ds_target = ds_target.assign_coords(time=pd.to_datetime(ds_target.time.dt.date))

        # Some target files store values as [time, dayofyear].
        # Align each time step with its own day-of-year so targets become 1D over time.
        if "dayofyear" in ds_target.dims:
            time_doy = xr.DataArray(
                pd.to_datetime(ds_target.time.values).dayofyear,
                coords={"time": ds_target.time.values},
                dims=("time",),
            )
            ds_target = ds_target.sel(dayofyear=time_doy)
            ds_target = ds_target.squeeze()

        # computes common times between predictors and targets and filter the data accordingly.
        common_times = np.intersect1d(ds_in.time.values, ds_target.time.values)
        ds_total = xr.Dataset({'predictors': ds_in, 'targets': ds_target}).sel(time=common_times)
        # store xarray dataset and transformations as attributes.
        self.ds = ds_total.sel(time=common_times).sel(time=slice(time_slice[0], time_slice[1]))
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.forecast_horizon_days = int(forecast_horizon_days)
        self.times = self.ds.time.values

    def close(self):
        self.ds.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        """
        Determines the number of samples in the dataset, which is the length of the time dimension after filtering for common times and splits.
        """
        return max(0, self.times.size - self.forecast_horizon_days)

    def __getitem__(self, idx):
        """
        Determines how one sample is retrieved. Here we select the appropriate time step from the dataset and apply any transformations if specified.
        """
        # Support convenient slicing/index lists for notebook inspection, e.g. dataset[:10].
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self[i] for i in idx]
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        # Convert xarray object to tensors.
        x = torch.as_tensor(self.ds.predictors.isel(time=idx).values, dtype=torch.float32)
        if self.forecast_horizon_days > 0:
            y = torch.as_tensor(
                self.ds.targets.isel(time=slice(idx, idx + self.forecast_horizon_days + 1)).values,
                dtype=torch.float32,
            )
        else:
            y = torch.as_tensor(self.ds.targets.isel(time=idx).values, dtype=torch.float32)
        # Keep target as a 1D tensor of length 1 so batched targets are [B, 1].
        if y.ndim == 0:
            y = y.unsqueeze(0)
        # Apply transformations if specified.
        if self.transform_X:
            x = self.transform_X(x)
        if self.transform_y:
            y = self.transform_y(y)
        # return a dictionary with keys 'x' and 'y' containing the input features and target variable, respectively, as well as the index for potential debugging or analysis purposes.
        return {"x": x, "y": y, "idx": idx}
    
    def sel(self, time='2020-01-01'):
        """
        Determines how one sample is retrieved. Here we select the appropriate time step from the dataset and apply any transformations if specified.
        """
        # Convert xarray object to tensors.
        x = torch.as_tensor(self.ds.predictors.sel(time=time).values, dtype=torch.float32)
        if self.forecast_horizon_days > 0:
            idx = int(np.where(self.times == np.datetime64(time))[0][0])
            y = torch.as_tensor(
                self.ds.targets.isel(time=slice(idx, idx + self.forecast_horizon_days + 1)).values,
                dtype=torch.float32,
            )
        else:
            y = torch.as_tensor(self.ds.targets.sel(time=time).values, dtype=torch.float32)
        # Keep target as a 1D tensor of length 1 so batched targets are [B, 1].
        if y.ndim == 0:
            y = y.unsqueeze(0)
        # Apply transformations if specified.
        if self.transform_X:
            x = self.transform_X(x)
        if self.transform_y:
            y = self.transform_y(y)
        # return a dictionary with keys 'x' and 'y' containing the input features and target variable, respectively, as well as the index for potential debugging or analysis purposes.
        return {"x": x, "y": y, "time": time}