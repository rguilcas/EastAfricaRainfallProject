"""
File documentation:
This file defines the MyDataModule class, which is a PyTorch Lightning DataModule for loading and preprocessing the dataset. 
It handles the data loading for training, validation, testing, and prediction phases. 
The DataModule also applies the specified transformations to the input features and target variable, fitting the transformations on the training data.
"""

import os
import torch

import lightning as L
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import AtmosphereToRainfallDataset, load_predictors

random_seed = 42

class MyDataModule(L.LightningDataModule):
    def __init__(self, data_in_name, data_target_name, 
                 data_path='./data',
                 batch_size=32,
                 years_split = [('1979','2008'), ('2009','2014'), ('2015','2024')],
                 transform_X=None, transform_y=None):
        """
        Initializes the MyDataModule with the specified parameters. 
        It takes the names of the input and target data files, the path to the data directory, batch size, years for train/validation/test splits, and any transformations to apply to the input features and target variable.
        """
        super().__init__()
        self.data_in_name = data_in_name
        self.data_target_name = data_target_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_years, self.val_years, self.test_years = years_split
        self.transform_X = transform_X
        self.transform_y = transform_y
        self._transforms_fitted = False
        predictor_data = load_predictors(data_path, data_in_name)
        self.image_shape = tuple(predictor_data.isel(time=0).shape)
        self.image_size = int(np.prod(self.image_shape[-2:]))

    def _ensure_transforms_fitted(self):
        """
        Helper function that ensures that transforms are fitted on the training split.
        This is needed for test/predict-only workflows that skip trainer.fit().
        """
        if self._transforms_fitted:
            return

        x_fitted = self.transform_X is None or getattr(self.transform_X, "fitted", False)
        y_fitted = self.transform_y is None or getattr(self.transform_y, "fitted", False)
        if x_fitted and y_fitted:
            self._transforms_fitted = True
            return

        print("Fitting transforms on the training data...")
        dataset_train_for_fit = AtmosphereToRainfallDataset(
            self.data_in_name,
            self.data_target_name,
            data_path=self.data_path,
            time_slice=self.train_years,
        )
        if self.transform_X is not None and not getattr(self.transform_X, "fitted", False):
            self.transform_X.fit(dataset_train_for_fit)
        if self.transform_y is not None and not getattr(self.transform_y, "fitted", False):
            self.transform_y.fit(dataset_train_for_fit)
        self._transforms_fitted = True

    def setup(self, stage='fit'):
        """
        This method is called by PyTorch Lightning to set up the datasets for training, validation, testing, and prediction.
        It creates the appropriate datasets for each phase, applies the specified transformations, and fits the transformations on the training data if they haven't been fitted yet.
        """
        # Training phase
        if stage == 'fit':
            # Data splitting, we keep a sequential split now for train/validation/test, but we can choose to shuffle them here.
            self.dataset_train = AtmosphereToRainfallDataset(
                self.data_in_name, 
                self.data_target_name, 
                data_path=self.data_path,
                time_slice=self.train_years)
            self.dataset_val = AtmosphereToRainfallDataset(
                self.data_in_name, 
                self.data_target_name, 
                data_path=self.data_path,
                time_slice=self.val_years)
            # Data preprocessing: fit on train split once and reuse for all stages.
            self._ensure_transforms_fitted()
            self.dataset_train.transform_X = self.transform_X
            self.dataset_val.transform_X = self.transform_X
            self.dataset_train.transform_y = self.transform_y
            self.dataset_val.transform_y = self.transform_y
            
        # Testing phase
        if stage == 'test':
            self._ensure_transforms_fitted()
            self.dataset_test = AtmosphereToRainfallDataset(self.data_in_name, self.data_target_name, 
                                                            data_path=self.data_path, time_slice=self.test_years,
                                                            transform_X = self.transform_X, transform_y = self.transform_y)
        
        # Prediction phase
        if stage == 'predict':
            self._ensure_transforms_fitted()
            self.dataset_predict = AtmosphereToRainfallDataset(self.data_in_name, self.data_target_name, 
                                                            data_path=self.data_path, time_slice = (None, None),
                                                            transform_X = self.transform_X, transform_y = self.transform_y)
        

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset. The training data is shuffled to ensure that the model does not learn any spurious patterns from the order of the data.
        This is used when running trainer.fit().
        """
        # We still shuffle the training data.
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(random_seed))
    
    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset. The validation data is not shuffled.
        This is used when running trainer.fit().
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns the DataLoader for the testing dataset. The testing data is not shuffled.
        This is used when running trainer.test().
        """
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        """
        Returns the DataLoader for the full dataset used for prediction.  
        This is used when running trainer.predict().
        The data is not shuffled to maintain the original order for analysis.
        We could choose to have other datasets for prediction, for example, we could have a dataset for future years that we want to predict on, but here we just use the full dataset.
        """
        return DataLoader(self.dataset_predict, batch_size=self.batch_size)
    