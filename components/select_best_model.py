from kfp.dsl import component, Input, Output, Model, Metrics
from typing import List


@component
def select_best_model(
    models: List[Input[Model]],
    metrics: List[Input[Metrics]],
    best_model: Output[Model],
):
    import json
    import shutil
    import os

    best_accuracy = -1.0
    best_model_path = None

    for model_artifact, metrics_artifact in zip(models, metrics):
        metrics_file = os.path.join(metrics_artifact.path, "metrics.json")

        with open(metrics_file) as f:
            data = json.load(f)

        accuracy = data["accuracy"]

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = model_artifact.path

    # Promote best model
    shutil.copytree(best_model_path, best_model.path)

    print(f"✅ Best model selected with accuracy={best_accuracy}")
