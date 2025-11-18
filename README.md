# Bottle Cap Detection MLOps

## Introduction

An end-to-end MLOps project for detecting bottle caps using YOLO models, integrated with Weights & Biases (W&B) for experiment tracking. This repository includes scripts for training, inference, and testing of the model. The goal is to detect and classify bottle caps into three categories (light_blue, dark_blue, other).

## Features

- Model: YOLOv11-Nano, selected after experiments against v6, v8, and v10.
- CLI: A complete Python CLI program called bsort for all project interactions.
- CI/CD Pipeline: A GitHub Action that automatically runs:
  - Code Linting (Pylint)
  - Code Formatting (Black, isort)
  - Unit Testing (Pytest with mocks)
  - Docker Image Building
- Containerization: A Dockerfile for a fully reproducible, deployable environment.
- Experiment Tracking: All training runs are logged and versioned using W&B (wandb.ai).
- Configuration: All parameters are managed in a settings.yaml file.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/bottle_caps_detection.git
   cd bottle_caps_detection
   ```

2. Create a conda environment and install dependencies:

   ```bash
   conda create -n bsort_env python=3.10
   conda activate bsort_env
   ```

3. Install the project in editable mode following the pyproject.toml:

   ```bash
   pip install -e .
   ```

## Usage (CLI)

The `bsort` CLI program provides the following commands:

1. Train the model:

    This command trains the YOLO model using the specified configuration file and automatically export the trained model to FP32 ONNX format.

    Note : To log training runs to W&B, ensure you have a W&B account and are logged in via either with setting the WANDB_API_KEY environment variable or running `wandb login` in your terminal. also run `yolo settings wandb=True` to enable W&B logging in YOLO.

   ```bash
   bsort train --config settings.yaml
   ```

2. Quantize the model:

    This command quantizes the FP32 ONNX model to INT8 format using calibration data.

   ```bash
   bsort quantize --config settings.yaml
   ```

3. Inference on an image:

    This command runs inference on a single image using the specified ONNX model.

   ```bash
   bsort infer --image path/to/image.jpg --config settings.yaml
   ```

## Benchmark Results

The final model selected was YOLOv11-Nano. The model is then exported to FP32 ONNX format and benchmarked on local CPU.

| Model          | Precision | Recall | mAP@0.5 | mAP@0.5:0.95  |
|----------------|-----------|--------|---------|---------------|
| YOLOv11-Nano   | 0.86      | 1.00   | 0.99    | 0.88          |

| CPU Model          | Model                    |Average Inference Time (ms) | FPS   |
|--------------------|--------------------------|----------------------------|-------|
| AMD Ryzen 9 5900HS | YOLOv11-Nano(FP32 ONNX)  | 90.57                      | 11.04 |
| AMD Ryzen 9 5900HS | YOLOv11-Nano(INT8 ONNX)  | 174.88                     | 5.72  |

On a local AMD Ryzen 9 5900HS CPU, the YOLOv11-Nano model in FP32 ONNX format achieved an average inference time of 90.57 ms, corresponding to approximately 11.04 FPS. It also reported an average inference time of 174.88 ms (5.72 FPS) on INT8 ONNX format, possibly due to the overhead of dequantization during inference.

In order to achieve the constraint of 5-10ms on Raspberry Pi, further optimizations such as quantization to INT8 and model pruning and accelerator hardware (e.g., Coral TPU) are recommended sincethese accelerators are have specialized hardware for INT8 operations which will eliminate the dequantization overhead seen on CPU.

## CI/CD

The project includes a GitHub Actions workflow that automatically runs on every push and pull request to the main branch. The workflow performs the following steps:

1.Lint & Test:

- Installs all project dependencies.
- Checks for code style errors with pylint.
- Checks for formatting errors with black --check.
- Checks for import sorting errors with isort --check.
- Runs the full pytest unit test suite.

2.Build Docker Image:

- Only runs if Lint & Test passes and the push is to the main branch.
- Builds the Dockerfile to ensure the application is container-ready.
