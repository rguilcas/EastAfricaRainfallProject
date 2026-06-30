"""
File documentation:
This file contains utility functions for loading trained models from checkpoints.
The main function, get_trained_model, loads a trained model from a specified checkpoint path and returns the model instance along with the PyTorch Lightning Trainer and the DataModule used for predictions.
"""

import inspect

import lightning as L

from src.data.datamodule import MyDataModule
from src.models.lightning_module import BinaryClassificationModel, RegressionModel
import src.models.models as models
from src.utils.config import _make_transform, get_classification_positive_weight, resolve_task_type


def get_trained_model(config, checkpoint_path):
    """
    Loads a trained model from a checkpoint path and returns the model instance along
    with the PyTorch Lightning Trainer and the DataModule used for predictions.
    """
    task_type = resolve_task_type(config)
    forecast_horizon_days = int(
        config["data"].get("forecast_horizon_days", config["data"].get("forecast_horizon_weeks", 0))
    )
    horizon_steps = forecast_horizon_days + 1

    # Gets the transformations from config. If a name is None or "None", no transform is applied.
    transform_X = _make_transform(config["data"].get("transform_X"))
    transform_y = _make_transform(config["data"].get("transform_y"))

    datamodule = MyDataModule(
        data_in_name=config["data"]["data_in_name"],
        data_target_name=config["data"]["data_target_name"],
        data_path=config["data"]["data_path"],
        batch_size=config["data"]["batch_size"],
        transform_X=transform_X,
        transform_y=transform_y,
        forecast_horizon_days=forecast_horizon_days,
    )

    model_class = getattr(models, config["model"]["model_name"])
    model_config = {
        **config["model"],
        "image_size": datamodule.image_size,
        "n_channels_input_cnn": config["model"]["n_channels_input_cnn"],
        "input_size": datamodule.image_size * config["model"]["n_channels_input_cnn"],
        "target_size": (2 * horizon_steps) if task_type == "classification" else horizon_steps,
    }

    constructor_params = inspect.signature(model_class.__init__).parameters
    allowed_kwargs = {
        name: value
        for name, value in model_config.items()
        if name in constructor_params and name != "self"
    }
    neural_network = model_class(**allowed_kwargs)

    if task_type == "classification":
        positive_class_weight = get_classification_positive_weight(config)
        lightning_model = BinaryClassificationModel.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            model=neural_network,
            learning_rate=config["trainer"]["learning_rate"],
            positive_class_weight=positive_class_weight,
            horizon_steps=horizon_steps,
        )
    else:
        target_inverse_transform = None
        if transform_y is not None and hasattr(transform_y, "inverse_transform"):
            target_inverse_transform = transform_y.inverse_transform

        lightning_model = RegressionModel.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            model=neural_network,
            learning_rate=config["trainer"]["learning_rate"],
            target_inverse_transform=target_inverse_transform,
        )

    trainer = L.Trainer(
        logger=False,
        accelerator=config["trainer"].get("accelerator", "auto"),
        devices=config["trainer"].get("devices", "auto"),
    )
    return lightning_model, trainer, datamodule
