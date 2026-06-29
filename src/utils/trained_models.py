"""
File documentation:
This file contains utility functions for loading trained models from checkpoints.
The main function, get_trained_model, loads a trained model from a specified checkpoint path and returns the model instance along with the PyTorch Lightning Trainer and the DataModule used for predictions.
"""

import lightning as L
import inspect 
from src.data.datamodule import MyDataModule
from src.models.lightning_module import RegressionModel
import src.models.models as models
from src.utils.config import _make_transform 


def get_trained_model(config, checkpoint_path):
    """
    Loads a trained model from a checkpoint path and returns the model instance along with the PyTorch Lightning Trainer and the DataModule used for predictions.
    """
    # Gets the transformation from the src.data.transforms module based on the name specified in the config. If the name is None or "None", no transformation is applied.
    transform_X = _make_transform(config["data"].get("transform_X"))
    transform_y = _make_transform(config["data"].get("transform_y"))

	# Loads the datamodel using MyDataModule class, which handles loading the dataset and applying the specified transformations.
    datamodule = MyDataModule(
		data_in_name=config["data"]["data_in_name"],
		data_target_name=config["data"]["data_target_name"],
		data_path=config["data"]["data_path"],
		batch_size=config["data"]["batch_size"],
		transform_X=transform_X,
		transform_y=transform_y,
	)

	# Builds the model architecture based on the model name specified in the config. The model is initialized with the appropriate parameters, including the image size and number of input channels for the CNN.
    model_class = getattr(models, config["model"]["model_name"])
    model_config = {
		**config["model"],
		"image_size": datamodule.image_size,
		"n_channels_input_cnn": config["model"]["n_channels_input_cnn"],
		"input_size": datamodule.image_size * config["model"]["n_channels_input_cnn"],
		"target_size": 1,
	}
	# Filters the model_config to only include parameters that are accepted by the model constructor, and then initializes the model.
    constructor_params = inspect.signature(model_class.__init__).parameters
    allowed_kwargs = {
		name: value
		for name, value in model_config.items()
		if name in constructor_params and name != "self"
	}
    neural_network = model_class(**allowed_kwargs)

	# If the target transformation has an inverse_transform method, it is passed to the Lightning model to be applied to the predictions before they are returned.
    target_inverse_transform = None
    if transform_y is not None and hasattr(transform_y, "inverse_transform"):
        target_inverse_transform = transform_y.inverse_transform

	# Loads the trained model checkpoint using the RegressionModel Lightning module, which wraps the neural network and handles the prediction logic.
    lightning_model = RegressionModel.load_from_checkpoint(
		checkpoint_path,
		map_location="cpu",
		model=neural_network,
		learning_rate=config["trainer"]["learning_rate"],
		target_inverse_transform=target_inverse_transform,
	)

	# Defines a PyTorch Lightning Trainer and uses it to run the prediction on the full dataset using the datamodule. 
    trainer = L.Trainer(
		logger=False,
		accelerator=config["trainer"].get("accelerator", "auto"),
		devices=config["trainer"].get("devices", "auto"),
	)
    return lightning_model, trainer, datamodule