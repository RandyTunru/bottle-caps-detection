"""
Module to handle ONNX INT8 quantization using Post-Training Quantization (PTQ).
"""

import os
import glob
import cv2
import numpy as np
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat


class BottleCapDataReader(CalibrationDataReader):
    """
    Data reader for feeding calibration images to the ONNX quantizer.
    It reads images from a folder and pre-processes them.
    """

    def __init__(self, image_folder: str, input_shape: tuple = (1, 3, 640, 640)):
        """
        Args:
            image_folder (str): Path to the folder containing calibration images.
            input_shape (tuple): The model's input shape (B, C, H, W).
        """
        self.image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
        self.input_shape = input_shape
        self.input_name = "images" 
        self.iterator = iter(self.image_files)
        print(f"Found {len(self.image_files)} calibration images.")

    def get_next(self):
        try:
            image_path = next(self.iterator)
            
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            resized = cv2.resize(image_rgb, (self.input_shape[3], self.input_shape[2]))
            
            preprocessed_image = (resized / 255.0).transpose(2, 0, 1).astype(np.float32)
            
            input_tensor = np.expand_dims(preprocessed_image, axis=0)
            
            return {self.input_name: input_tensor}

        except StopIteration:
            return None  

def run_quantization(
    fp32_model_path: str,
    int8_model_path: str,
    calibration_path: str
):
    """
    Performs INT8 Post-Training Quantization.
    Args:
        fp32_model_path (str): Path to the FP32 ONNX model.
        int8_model_path (str): Path to save the INT8 quantized ONNX model.
        calibration_path (str): Path to the folder with calibration images.
    """
    if not os.path.exists(fp32_model_path):
        print(f"Error: FP32 model not found at {fp32_model_path}")
        print("Please run 'bsort train' first to generate it.")
        return

    print("--- Starting INT8 Quantization (this may take a few minutes) ---")

    calibrator = BottleCapDataReader(calibration_path)
    
    quantize_static(
        model_input=fp32_model_path,
        model_output=int8_model_path,
        calibration_data_reader=calibrator,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        op_types_to_quantize=['Conv', 'MatMul'] # Key operators for YOLO
    )
    
    print("--- Quantization Complete ---")
    print(f"INT8 model saved to: {int8_model_path}")