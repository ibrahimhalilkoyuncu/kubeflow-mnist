from kfp import dsl
from kfp.dsl import Output, Model, Artifact, Metrics


@dsl.component(
    base_image="kubeflow-fashion-mnist:1.0"
)
def train_model(
    learning_rate: float,
    epochs: int,
    hidden_units: int,
    model: Output[Model],
    metrics: Output[Metrics],
):
    import subprocess
    import os
    import json

    model_dir = model.path
    os.makedirs(model_dir, exist_ok=True)

    subprocess.run([
        "python",
        "/app/train.py",
        f"--learning-rate={learning_rate}",
        f"--epochs={epochs}",
        f"--hidden-units={hidden_units}",
        f"--model-dir={model_dir}",
        f"--metrics-dir={metrics.path}",
    ], check=True)
    
    # Read metrics and log them
    metrics_file = os.path.join(metrics.path, "metrics.json")
    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)
    
    metrics.log_metric("accuracy", metrics_data["accuracy"])
    metrics.log_metric("loss", metrics_data["loss"])
    metrics.log_metric("learning_rate", learning_rate)