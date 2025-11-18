"""
Module to handle training of YOLO models with W&B integration.
"""

import os

from ultralytics import YOLO
import wandb


def run_training(
    model_name: str,
    dataset_path: str,
    project_name: str,
    epochs: int,
    batch_size: int,
    onnx_model_path: str,
    seed: int,
):
    """
    Logs into W&B and runs the YOLO training process.
    Args:
        model_name (str): Name or path of the YOLO model to train.
        dataset_path (str): Path to the dataset YAML file.
        project_name (str): W&B project name for logging.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        onnx_model_path (str): Path to save the exported ONNX model.
        seed (int): Random seed for reproducibility.
    Returns:
        None
    """
    try:
        wandb.login()
        use_wandb = True
    except Exception as e:
        print(f"Could not log in to W&B. Skipping. Error: {e}")
        use_wandb = False

    # Load the model
    model = YOLO(model_name)

    results = model.train(
        data=dataset_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        project=project_name if use_wandb else None,
        name=f"{model_name}-local",
        exist_ok=True,
        seed=seed,
    )

    best_model_path = results.save_dir / "weights" / "best.pt"
    print(f"Best model saved to: {best_model_path}")

    try:
        best_model = YOLO(best_model_path)

        print("Exporting best model to ONNX...")

        exported_onnx_source_path = best_model.export(
            format="onnx", imgsz=640, dynamic=True, opset=12, simplify=True
        )

        print(
            f"Moving exported model from {exported_onnx_source_path} to {onnx_model_path}"
        )

        os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

        os.replace(exported_onnx_source_path, onnx_model_path)

        print(f"Successfully exported and moved ONNX model to '{onnx_model_path}'")
    except Exception as e:
        print(f"Failed to export model to ONNX format. Error: {e}")
