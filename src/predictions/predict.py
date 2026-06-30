"""
File documentation:
This file is the main prediction script that generates predictions from a trained model checkpoint.
It loads the model checkpoint, applies the necessary data transformations, and generates predictions on the full dataset
The predictions are then saved to a CSV file for further analysis.
"""


import argparse
from pathlib import Path

import pandas as pd
import torch

from src.utils.config import load_config, _build_data_slice, resolve_task_type
from src.utils.trained_models import get_trained_model


def main(config, checkpoint_path, output_path):
	"""
	Main function to run the prediction. 
	It loads the model checkpoint, applies the necessary data transformations, and generates predictions on the full dataset. 
	The predictions are then saved to a CSV file for further analysis.
	"""
	lightning_model, trainer, datamodule = get_trained_model(config, checkpoint_path)
	task_type = resolve_task_type(config)
	forecast_horizon_days = int(
		config["data"].get("forecast_horizon_days", config["data"].get("forecast_horizon_weeks", 0))
	)
	horizon_steps = forecast_horizon_days + 1
	
	# Make all predictions and concatenate the results into tensors for predictions, targets, and time indices.
	prediction_batches = trainer.predict(lightning_model, datamodule=datamodule)
	if task_type == "classification":
		pred_classes = torch.cat([batch[0].detach().cpu() for batch in prediction_batches], dim=0)
		pred_pos_prob = torch.cat([batch[1].detach().cpu() for batch in prediction_batches], dim=0)
		targets = torch.cat([batch[2].detach().cpu() for batch in prediction_batches], dim=0)
		if pred_classes.ndim == 1:
			pred_classes = pred_classes.unsqueeze(-1)
		if pred_pos_prob.ndim == 1:
			pred_pos_prob = pred_pos_prob.unsqueeze(-1)
		if targets.ndim == 1:
			targets = targets.unsqueeze(-1)
		time_indices = torch.cat([batch[3].detach().cpu() for batch in prediction_batches], dim=0).numpy()
	else:
		preds = torch.cat([batch[0].detach().cpu() for batch in prediction_batches], dim=0)
		targets = torch.cat([batch[1].detach().cpu() for batch in prediction_batches], dim=0)
		if preds.ndim == 1:
			preds = preds.unsqueeze(-1)
		if targets.ndim == 1:
			targets = targets.unsqueeze(-1)
		time_indices = torch.cat([batch[2].detach().cpu() for batch in prediction_batches], dim=0).numpy()
		# If the prediction is negative, set it to 0, since rainfall cannot be negative.
		preds = torch.clamp(preds, min=0.0)
	
	# Get the actual times of the predictions if data have been shuffled.
	times = datamodule.dataset_predict.times[time_indices]

	if task_type == "regression":
		# If the target transformation has an inverse_transform method, it is applied to both the predictions and targets to bring them back to the original scale.
		target_inverse_transform = lightning_model.target_inverse_transform
		if target_inverse_transform is not None:
			preds = target_inverse_transform(preds)
			targets = target_inverse_transform(targets)

	years_split = (datamodule.train_years, datamodule.val_years, datamodule.test_years)
	# Creates a DataFrame with the predictions, targets, and data slice (train/val/test) for each time point.
	if task_type == "classification":
		data = {"data_slice": _build_data_slice(times, years_split)}
		for h in range(horizon_steps):
			data[f"prediction_t_plus_{h}"] = pred_classes[:, h].numpy()
			data[f"prediction_positive_prob_t_plus_{h}"] = pred_pos_prob[:, h].numpy()
			data[f"target_t_plus_{h}"] = targets[:, h].numpy()
		df = pd.DataFrame(data, index=pd.to_datetime(times))
	else:
		data = {"data_slice": _build_data_slice(times, years_split)}
		for h in range(horizon_steps):
			data[f"prediction_t_plus_{h}"] = preds[:, h].numpy()
			data[f"target_t_plus_{h}"] = targets[:, h].numpy()
		df = pd.DataFrame(data, index=pd.to_datetime(times))
	df.index.name = "time"
	df = df.sort_index()
	# Saves the DataFrame to a CSV file at the specified output path. The output directory is created if it does not exist.
	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_file)
	print(f"Saved predictions to: {output_file}")


if __name__ == "__main__":
	# Parses command-line arguments for the run ID, project root, checkpoint name, and output path for the predictions.
	parser = argparse.ArgumentParser(description="Run inference from a trained checkpoint.")
	parser.add_argument(
		"--run-id",
		type=str,
		required=True,
		help="Name of the run to load a checkpoint from. Should correspond to a W&B run id.",
	)
	parser.add_argument(
		"--project-root",
		type=str,
		default="bccr-ml-course",
		help="Name of the output project folder where the run is stored.",
	)
	parser.add_argument(
		"--checkpoint-name",
		type=str,
		default="best.ckpt",
		help="Checkpoint filename under outputs/<project>/<run-id>/checkpoints/.",
	)
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=None,
		help="Deprecated alias of --checkpoint-name.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Path for the predictions to be saved.",    
	)
	args = parser.parse_args()
	if args.checkpoint:
		args.checkpoint_name = args.checkpoint

	config_path = f"outputs/{args.project_root}/{args.run_id}/config.yaml"
	checkpoint_path = f"outputs/{args.project_root}/{args.run_id}/checkpoints/{args.checkpoint_name}"
	run_output_dir = Path(f"outputs/{args.project_root}/{args.run_id}")
	checkpoint_file = Path(checkpoint_path)
	checkpoint_path = str(checkpoint_file)
	if args.output is None:
		args.output = str(run_output_dir / "predictions.csv")
	# Load configuration file
	config = load_config(config_path)
	# Run the main prediction function with the loaded configuration, checkpoint path, and output path for the predictions.
	main(config=config, checkpoint_path=checkpoint_path, output_path=args.output)