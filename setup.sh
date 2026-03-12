#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Verifying Kaggle CLI..."
kaggle --version

echo "Creating dataset folders..."
mkdir -p traffic_ai_dataset/raw_downloads
mkdir -p traffic_ai_dataset/images
mkdir -p traffic_ai_dataset/videos
mkdir -p traffic_ai_dataset/processed

echo "Starting dataset pipeline..."
python traffic_dataset_pipeline.py
