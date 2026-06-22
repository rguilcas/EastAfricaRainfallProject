#!/usr/bin/env bash

# --- File Documentation ---
# This file is the main training script that trains a model following the given configuration file. 
# It loads the configuration from a YAML file, initializes the model and data loaders, and starts the training process.
# --------------------------

# Exit on error
set -e

# Go to project root (important!)
cd ../


# Default config file name. You can specify a different config file using the --config argument when running the script.
CONFIG_FILE="default_config.yml"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ ! -f "configs/$CONFIG_FILE" ]; then
    echo "Error: Config file not found: configs/$CONFIG_FILE"
    exit 1
fi

# Run python training script
python -m src.training.train \
    --config "configs/$CONFIG_FILE"