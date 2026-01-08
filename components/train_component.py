from kfp import dsl
from kfp.dsl import Input, Output, Model, Metrics


@dsl.component(
    base_image="your-docker-registry/kubeflow-fashion-mnist:latest"
)
def train_model(
    learning_rate: float,
    epochs: int,
    hidden_units: int,
    model: Output[Model],
    metrics: Output[Metrics],
):
    """
    Trains a Fashion-MNIST model and outputs a SavedModel + metrics
    """

    import subprocess
    import json
    import os

    model_dir = model.path

    cmd = [
        "python",
        "/app/train.py",
        f"--learning-rate={learning_rate}",
        f"--epochs={epochs}",
        f"--hidden-units={hidden_units}",
        f"--model-dir={model_dir}",
    ]

    subprocess.run(cmd, check=True)

    # Read metrics written by train.py
    metrics_file = os.path.join(model_dir, "metrics.txt")

    accuracy = None
    loss = None

    with open(metrics_file) as f:
        for line in f:
            key, value = line.strip().split("=")
            if key == "accuracy":
                accuracy = float(value)
            elif key == "loss":
                loss = float(value)

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("loss", loss)