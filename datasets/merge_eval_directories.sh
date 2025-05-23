#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Requirements : VoxCeleb1 and voxconverse_test_SV dataset"

# Define paths
# These paths are relative to the project root where this script is expected to be run.
VOX1_EVALS_DIR="vox1-evals"
VOXCELEB1_SRC_PATH="./voxceleb1"             # Default directory path for Voxceleb1 wav files
VOXCONVERSE_SRC_PATH="./voxconverse_test_SV" # Default directory path for VoxConverse test SV wav files

echo "Creating $VOX1_EVALS_DIR directory if it does not exist..."
mkdir -p "$VOX1_EVALS_DIR"

echo "Creating symbolic links for VoxCeleb1 contents in $VOX1_EVALS_DIR..."
if [ -d "$VOXCELEB1_SRC_PATH" ]; then
    for item in "$VOXCELEB1_SRC_PATH"/*; do
        if [ -e "$item" ]; then
            ln -sfn "$(realpath "$item")" "$VOX1_EVALS_DIR/$(basename "$item")"
        fi
    done
    echo "Symbolic links for VoxCeleb1 created."
else
    echo "Warning: Source directory $VOXCELEB1_SRC_PATH does not exist. Skipping VoxCeleb1 symlinks."
    echo "Please ensure VoxCeleb1 dataset is at $VOXCELEB1_SRC_PATH."
fi

echo "Creating symbolic links for VoxConverse test set contents in $VOX1_EVALS_DIR..."
if [ -d "$VOXCONVERSE_SRC_PATH" ]; then
    for item in "$VOXCONVERSE_SRC_PATH"/*; do
        if [ -e "$item" ]; then 
            ln -sfn "$(realpath "$item")" "$VOX1_EVALS_DIR/$(basename "$item")"
        fi
    done
    echo "Symbolic links for VoxConverse created."
else
    echo "Warning: Source directory $VOXCONVERSE_SRC_PATH does not exist. Skipping VoxConverse symlinks."
    echo "This might be because it was not downloaded correctly by make_vcmix_testset.py, or the path is incorrect."
fi

echo "Setup for vox1-evals finished. The directory $VOX1_EVALS_DIR should now contain symbolic links to the datasets."
echo "Please verify the contents of $VOX1_EVALS_DIR."
