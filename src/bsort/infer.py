"""
Module for running inference using a YOLO model.
"""

import os

from ultralytics import YOLO


def run_inference(model_path: str, image_path: str):
    """
    Runs ONNX inference on a single image and saves the result to the outputs folder.
    Args:
        model_path (str): Path to the ONNX model file.
        image_path (str): Path to the input image.
    Returns:
        None
    """

    model = YOLO(model_path)

    results = model(image_path)

    os.makedirs("outputs", exist_ok=True)

    output_path = (
        f"outputs/{os.path.basename(image_path).replace('.jpg', '')}-prediction.jpg"
    )
    results[0].save(filename=output_path)
