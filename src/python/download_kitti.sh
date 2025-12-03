#!/bin/bash

# Determine the script's absolute directory path first
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Navigate up two levels (from src/python) to reach the project root
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

# --- Configuration ---
DATA_DIR="$PROJECT_ROOT/data"
URLS_FILE="$DATA_DIR/urls.txt"

echo "Starting FULL KITTI Odometry Dataset download..."
echo "Project Root: $PROJECT_ROOT"
echo "Target Data Directory: $DATA_DIR"
echo "--------------------------------------------------------"
echo "WARNING: This will download approximately 80GB of data."
echo "Zips will be deleted immediately after extraction to save space."
echo "--------------------------------------------------------"

# 1. Clean up previous specific sequence folders (optional, to keep it clean)
# We preserve kitti_00 if you want to keep the demo data, otherwise you can delete it manually.
# rm -rf "$DATA_DIR/kitti_00" 

# 2. Check if the URLs file exists
if [ ! -f "$URLS_FILE" ]; then
    echo "Error: URL file not found at $URLS_FILE."
    echo "Please create data/urls.txt with the correct KITTI zip links."
    exit 1
fi

# 3. Download the main archives
# The -c flag allows resuming if the 80GB download gets interrupted
echo "Downloading files listed in $URLS_FILE..."
wget -c -i "$URLS_FILE" -P "$DATA_DIR"

# 4. Unzip Archives and Clean Up Immediately
# KITTI zips extract into a 'dataset' folder by default.
# We use 'unzip -n' which skips files if they already exist.

# --- Calibration ---
CALIB_ZIP="$DATA_DIR/data_odometry_calib.zip"
echo "Unzipping Calibration data..."
if [ -f "$CALIB_ZIP" ]; then
    unzip -n -q "$CALIB_ZIP" -d "$DATA_DIR"
    echo "Deleting $CALIB_ZIP..."
    rm "$CALIB_ZIP" # Cleanup step
else
    echo "Warning: data_odometry_calib.zip not found."
fi

# --- Poses ---
POSES_ZIP="$DATA_DIR/data_odometry_poses.zip"
echo "Unzipping Ground Truth Poses..."
if [ -f "$POSES_ZIP" ]; then
    unzip -n -q "$POSES_ZIP" -d "$DATA_DIR"
    echo "Deleting $POSES_ZIP..."
    rm "$POSES_ZIP" # Cleanup step
else
    echo "Warning: data_odometry_poses.zip not found."
fi

# --- Velodyne Point Clouds ---
VELODYNE_ZIP="$DATA_DIR/data_odometry_velodyne.zip"
echo "Unzipping Velodyne Point Clouds (This will take a while)..."
if [ -f "$VELODYNE_ZIP" ]; then
    unzip -n -q "$VELODYNE_ZIP" -d "$DATA_DIR"
    echo "Deleting $VELODYNE_ZIP..."
    rm "$VELODYNE_ZIP" # Cleanup step
else
    echo "Warning: data_odometry_velodyne.zip not found."
fi

echo "--- Download, Extraction, and Immediate Cleanup Complete! ---"
echo "Your data should be organized under $DATA_DIR/dataset."