import os
import pytest
import torch
import torch.nn as nn

@pytest.fixture(scope="session")
def dummy_onnx_model(tmp_path_factory):
    """
    Creates a minimal ONNX model file that can be loaded
    by onnxruntime for testing.
    """
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "dummy_model.onnx"

    dummy_input = torch.randn(1, 10)
    model = DummyModel()
    
    torch.onnx.export(
        model,
        dummy_input,
        str(model_path),  
        input_names=["input"],
        output_names=["output"],
    )

    return str(model_path)