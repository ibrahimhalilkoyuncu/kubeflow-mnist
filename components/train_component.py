from kfp import dsl
from kfp.dsl import Output, Model, Metrics


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

    cmd = [
        "python",
        "/app/train.py",
        f"--learning-rate={learning_rate}",
        f"--epochs={epochs}",
        f"--hidden-units={hidden_units}",
        f"--model-dir={model.path}",
        f"--metrics-dir={metrics.path}",
    ]

    subprocess.run(cmd, check=True)