"""
Tests for the CLI commands in bsort
Examples:
    pytest test/test_cli.py
"""

from unittest.mock import MagicMock, call

import yaml
from typer.testing import CliRunner

from bsort.main import app

runner = CliRunner()


def test_cli_train_command(mocker, tmp_path):
    """
    Tests the 'bsort train' command.
    Args:
        mocker: Pytest mocker fixture.
        tmp_path: Pytest temporary path fixture.
    Returns:
        None
    """

    mocker.patch("wandb.login")

    mock_yolo_class = mocker.patch("bsort.train.YOLO")

    mock_model_instance = MagicMock()

    mock_yolo_class.return_value = mock_model_instance

    mock_results = MagicMock()
    mock_results.save_dir = tmp_path
    mock_model_instance.train.return_value = mock_results

    expected_best_path = tmp_path / "weights" / "best.pt"

    mock_model_instance.export.return_value = "fake/path/best.onnx"

    mocker.patch("os.replace")
    mocker.patch("os.makedirs")

    config = {
        "model_name": "yolo11n.pt",
        "dataset_yaml": "fake/data.yaml",
        "project_name": "Test-Project",
        "epochs": 1,
        "batch_size": 1,
        "seed": 42,
        "onnx_model_path": "models/fake.onnx",
    }
    config_path = tmp_path / "test_settings.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = runner.invoke(app, ["train", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Starting Model Training" in result.stdout
    assert "Training Complete" in result.stdout

    assert mock_yolo_class.call_count == 2
    constructor_calls = mock_yolo_class.call_args_list
    assert constructor_calls[0] == call("yolo11n.pt")
    assert constructor_calls[1] == call(expected_best_path)

    mock_model_instance.train.assert_called_once_with(
        data="fake/data.yaml",
        epochs=1,
        batch=1,
        imgsz=640,
        project="Test-Project",
        name="yolo11n.pt-local",
        exist_ok=True,
        seed=42,
    )

    mock_model_instance.export.assert_called_once_with(
        format="onnx", imgsz=640, dynamic=True, opset=12, half=True, simplify=True
    )

    mock_model_instance.export.assert_called()


def test_cli_infer_command(mocker, dummy_onnx_model, tmp_path):
    """
    Tests the 'bsort infer' command.
    Args:
        mocker: Pytest mocker fixture.
        dummy_onnx_model: Path to a dummy ONNX model fixture.
        tmp_path: Pytest temporary path fixture.
    Returns:
        None
    """

    mock_yolo_class = mocker.patch("bsort.infer.YOLO")
    mock_model_instance = MagicMock()
    mock_yolo_class.return_value = mock_model_instance

    mock_results = [MagicMock()]
    mock_model_instance.return_value = mock_results

    config = {"onnx_model_path": dummy_onnx_model}
    config_path = tmp_path / "test_settings.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    fake_image_path = tmp_path / "test.jpg"
    fake_image_path.touch()

    result = runner.invoke(
        app, ["infer", "--image", str(fake_image_path), "--config", str(config_path)]
    )

    assert result.exit_code == 0
    assert "Running Inference" in result.stdout
    assert "Inference Complete" in result.stdout

    mock_yolo_class.assert_called_once_with(dummy_onnx_model)

    mock_model_instance.assert_called_once_with(str(fake_image_path))

    mock_results[0].save.assert_called_once_with(
        filename=f"outputs/{fake_image_path.stem}-prediction.jpg"
    )


def test_cli_infer_no_image_fail():
    """
    Tests the 'bsort infer' command with a non-existent image path.
    Args:
        None
    Returns:
        None
    """

    result = runner.invoke(app, ["infer", "--image", "nonexistent.jpg"])

    assert result.exit_code != 0
    assert "Error: Image file not found" in result.stdout
