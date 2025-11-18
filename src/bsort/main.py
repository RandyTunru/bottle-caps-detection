import os
from typing import Annotated

import typer
import yaml

from . import infer as infer_model
from . import train as train_model

app = typer.Typer(help="Bottle Cap Detection CLI.")


def load_config(config_path: str) -> dict:
    """Loads the YAML config file."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        raise typer.Exit(code=1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.command()
def train(
    config: Annotated[
        str,
        typer.Option(
            "--config",
            "-c",
            help="Path to the settings.yaml file.",
        ),
    ],
):
    """
    Train a new model using the specified config file.
    """
    print(f"Loading config from {config}")
    params = load_config(config)

    print("Starting Model Training")
    train_model.run_training(
        model_name=params["model_name"],
        dataset_path=params["dataset_yaml"],
        project_name=params["project_name"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        onnx_model_path=params["onnx_model_path"],
        seed=params["seed"],
    )
    print("Training Complete")


@app.command()
def infer(
    image: Annotated[
        str,
        typer.Option(
            "--image",
            "-i",
            help="Path to the input image.",
        ),
    ],
    config: Annotated[
        str,
        typer.Option(
            "--config",
            "-c",
            help="Path to the settings.yaml file.",
        ),
    ] = "settings.yaml",
):
    """
    Run inference on a single image using the production ONNX model.
    """
    if not os.path.exists(image):
        print(f"Error: Image file not found at {image}")
        raise typer.Exit(code=1)

    params = load_config(config)
    model_path = params["onnx_model_path"]

    if not os.path.exists(model_path):
        print(f"Error: ONNX model not found at {model_path}")
        print("Please run training and export first.")
        raise typer.Exit(code=1)

    print("Running Inference")
    print(f"Model: {model_path}")
    print(f"Image: {image}")

    infer_model.run_inference(
        model_path=model_path,
        image_path=image,
    )
    print("Inference Complete")
    print("Prediction saved to 'outputs' folder.")


if __name__ == "__main__":
    app()
