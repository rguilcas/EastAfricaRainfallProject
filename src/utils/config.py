"""
File documentation:
This file defines helper functions related to configuration management and data processing.
"""
import os
from src.data import transforms
from pathlib import Path
import pandas as pd
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def _make_transform(name):
	"""
	Helper function to create a transform instance from its name. If the name is None or "None", returns None.
	"""
	if name in (None, "None"):
		return None
	return getattr(transforms, name)()

def _build_data_slice(times, years_split):
	"""
	Helper function to build a data slice array that indicates whether each time point belongs to the training, validation, or test split based on the provided years_split.
	"""
	ts = pd.to_datetime(times)
	split = pd.Series("train", index=ts)
	_, val_years, test_years = years_split
	split[val_years[0]:val_years[1]] = "val"
	split[test_years[0]:test_years[1]] = "test"
	return split.to_numpy()