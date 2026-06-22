"""
File documentation:
This file defines custom data transformations for preprocessing the dataset.
"""

import torch

class StandardScalerX:
    """
    Standardizes the input data by removing the mean and scaling to unit variance.
    The mean and standard deviation are computed from the training dataset and stored as attributes.
    """
    def __init__(self, eps=1e-6):
        self.meanX = None
        self.stdX = None
        self.fitted = False
        self.eps = eps

    def fit(self, dataset):
        X = torch.stack([dataset[i]['x'] for i in range(len(dataset))])
        self.meanX = X.mean(dim=0)
        self.stdX = X.std(dim=0, unbiased=False)
        self.stdX = torch.clamp(self.stdX, min=self.eps)

        self.fitted = True

    def __call__(self, x):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted")
        mean_x = self.meanX.to(device=x.device, dtype=x.dtype)
        std_x = self.stdX.to(device=x.device, dtype=x.dtype)
        return (x - mean_x) / std_x



class MinMaxScalerY:
    """
    Scales the target data to a given range (default is [0, 1]) by subtracting the minimum and dividing by the range (max - min).
    The minimum and maximum values are computed from the training dataset and stored as attributes.
    """
    def __init__(self, eps=1e-6):
        self.maxY = None
        self.minY = None
        self.fitted = False
        self.eps = eps

    def fit(self, dataset):
        Y = torch.stack([dataset[i]['y'] for i in range(len(dataset))])
        self.maxY = Y.max(dim=0).values
        self.minY = Y.min(dim=0).values
        self.fitted = True

    def inverse_transform(self, y):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted")
        max_y = self.maxY.to(device=y.device, dtype=y.dtype)
        min_y = self.minY.to(device=y.device, dtype=y.dtype)
        return y * (max_y - min_y + self.eps) + min_y

    def __call__(self, y):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted")
        max_y = self.maxY.to(device=y.device, dtype=y.dtype)
        min_y = self.minY.to(device=y.device, dtype=y.dtype)
        return (y - min_y) / (max_y - min_y + self.eps)
    

class Log1pY:
    """
    Scales the target data by applying the log1p transformation, which is defined as log(1 + y).
    This transformation is useful for target variables that are strictly positive and have a skewed distribution.
    """
    def __init__(self, eps=1e-6):
        self.fitted = True
    def inverse_transform(self, y):
        return torch.expm1(y)
    def __call__(self, y):
        return torch.log1p(y)