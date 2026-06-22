#!/usr/bin/env bash

# --- File Documentation ---
# This file makes predictions with a trained model. 
# It loads the model checkpoint from a specified training run, applies the necessary data transformations, and generates predictions on the full dataset. 
# The predictions are then saved to a CSV file for further analysis.
# --------------------------


# Exit on error
set -e

# Go to project root
cd ../

# Run ID is added as a command line argument. You can find the run ID in the Weights & Biases dashboard. It is a unique identifier for each training run, and you can use it to load the model checkpoint and logs for that run.


# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run-id) RUN_ID="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$RUN_ID" ]; then
    echo "Error: Please provide the run-id you want to predict on. using the --run-id argument."
    exit 1
fi


python -m src.predictions.predict \
    --run-id "$RUN_ID" \
    --project-root "bccr-ml-course" \
    --checkpoint "best.ckpt" \
