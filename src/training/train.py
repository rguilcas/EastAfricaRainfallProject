"""
File documentation:
This file is the main training script that trains a model following the given configuration file. 
It loads the configuration from a YAML file, initializes the model and data loaders, and starts the training process.
"""

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import src.models.models as models
from src.data.datamodule import MyDataModule
from src.models.lightning_module import RegressionModel
from src.utils.config import load_config, _make_transform

import argparse
import os
import uuid
import yaml
import inspect
import numpy as np

def main(config):
    """
    Main function to run the training.
    It sets up the W&B logger, initializes the data module and model based on the configuration, and starts the training process using PyTorch Lightning's Trainer.
    """

    # determines project dir and creates an individual run directory for this training run. 
    project_dir = os.path.join(config['logging']['base_dir'], config['logging']['project'])
    run_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join(project_dir, run_id)
    # Check that the run_id does not already exist to avoid overwriting previous runs. If it does, generate a new one.
    while os.path.exists(out_dir):
        run_id = uuid.uuid4().hex[:8]
        out_dir = os.path.join(project_dir, run_id)
    os.makedirs(out_dir)

    # Sets up the W&B logger with the specified project name, entity, run name, and configuration. The logs will be saved in the output directory for this run.
    wandb_logger = WandbLogger(
        project=config['logging']['project'],
        entity=config['logging'].get('entity'),
        name=config['logging'].get('run_name'),
        id=run_id,
        version=run_id,
        config=config,
        save_dir=out_dir
    )
    print(f"W&B run id: {run_id}")
    print(f"W&B run url: {wandb_logger.experiment.url}")

    # Save config file for reproducibility
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Gets the transformation from the src.data.transforms module based on the name specified in the config. If the name is None or "None", no transformation is applied.
    transform_X = _make_transform(config["data"].get("transform_X"))
    transform_y = _make_transform(config["data"].get("transform_y"))

    # Loads the datamodule using MyDataModule class, which handles loading the dataset and applying the specified transformations.
    datamodule = MyDataModule(
        data_in_name=config['data']['data_in_name'],
        data_target_name=config['data']['data_target_name'],
        data_path=config['data']['data_path'],
        batch_size=config['data']['batch_size'],
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
    constructor_params = inspect.signature(model_class.__init__).parameters
    allowed_kwargs = {
		name: value
		for name, value in model_config.items()
		if name in constructor_params and name != "self"
	}
    neural_network = model_class(**allowed_kwargs)

    # If the target transformation has an inverse_transform method, it is passed to the RegressionModel to be applied to the predictions and targets during training and validation, allowing them to be evaluated on the original scale of the target variable.
    target_inverse_transform = None
    if transform_y is not None and hasattr(transform_y, 'inverse_transform'):
        target_inverse_transform = transform_y.inverse_transform
    
    # Initializes the RegressionModel Lightning module, which wraps the neural network and handles the training logic, including applying the target inverse transformation if specified.
    lightning_model = RegressionModel(
        neural_network,
        learning_rate=config['trainer']['learning_rate'],
        target_inverse_transform=target_inverse_transform,
    )

    # Sets up the PyTorch Lightning Trainer with the specified maximum epochs, accelerator, devices, logger, and callbacks for model checkpointing and early stopping based on validation loss.
    trainer = L.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        logger=wandb_logger,
        default_root_dir=out_dir,
        # We save checkpoints in the output directory for this run, and monitor the validation loss to save the best model and stop training if it does not improve for a certain number of epochs.
        callbacks=[
            ModelCheckpoint(dirpath=os.path.join(out_dir, "checkpoints"), 
                            filename="best",
                            save_top_k=1, 
                            monitor="val_loss",
                            mode="min"),
            EarlyStopping(monitor="val_loss", patience=5, mode='min'),
        ],
    )
    
    # Main training loop: fits the model using the datamodule, which handles loading the data and applying the specified transformations.
    trainer.fit(lightning_model, datamodule=datamodule)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a rainfall regression model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)



