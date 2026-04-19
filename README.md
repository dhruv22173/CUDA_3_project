# GPU Batch Image Processing

## Description
This project demonstrates GPU-based batch image processing using PyTorch. A large number of images are processed in parallel using a Sobel edge detection filter.

## GPU Usage
CUDA-enabled GPU is used to accelerate convolution operations.

## Dataset
100+ images placed in the input folder.

## Setup
pip install -r requirements.txt

## Run
python main.py

## Output
Processed images are stored in the output folder.

## Results
Execution time comparison between CPU and GPU is stored in results/execution_log.txt.