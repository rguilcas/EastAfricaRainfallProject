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


def resolve_task_type(config):
	"""
	Returns normalized task type from config and validates supported values.
	"""
	task_type = config.get("trainer", {}).get("task_type", "regression").lower()
	if task_type not in {"regression", "classification"}:
		raise ValueError(
			f"Unsupported trainer.task_type '{task_type}'. "
			"Expected one of: regression, classification."
		)
	return task_type


def get_classification_positive_weight(config):
	"""
	Returns the positive-class weight used by CrossEntropyLoss for binary classification.

	For BinaryScalerY with quantile q, this uses:
	positive_weight = 1 / (1 - q)
	"""
	transform_y = config.get("data", {}).get("transform_y")
	quantile = 0.5

	if isinstance(transform_y, dict) and transform_y.get("name") == "BinaryScalerY":
		quantile = float(transform_y.get("quantile", 0.5))
	elif isinstance(transform_y, str) and transform_y == "BinaryScalerY":
		quantile = 0.5

	if quantile >= 1.0:
		raise ValueError("BinaryScalerY quantile must be < 1.0 to compute class weight.")
	if quantile < 0.0:
		raise ValueError("BinaryScalerY quantile must be >= 0.0.")

	return 1.0 / (1.0 - quantile)
    

def _make_transform(name):
	"""
	Helper function to create a transform instance.

	Supported formats:
	- None or "None"
	- "TransformClassName"
	- {"name": "TransformClassName", ...kwargs}
	"""
	if name in (None, "None"):
		return None
	if isinstance(name, str):
		return getattr(transforms, name)()
	if isinstance(name, dict):
		transform_name = name.get("name")
		if transform_name in (None, "None"):
			return None
		kwargs = {k: v for k, v in name.items() if k != "name"}
		return getattr(transforms, transform_name)(**kwargs)
	raise TypeError(
		"data.transform_X / data.transform_y must be None, a transform name string, "
		"or a dict like {'name': 'BinaryScalerY', ...}."
	)

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