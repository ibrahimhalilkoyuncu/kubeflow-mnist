from kfp import dsl
from kfp.dsl import Input, Output, Model, Metrics


@dsl.component(
    base_image="python:3.11"
)
def select_best_model_manual(
    model_1: Input[Model],
    metrics_1: Input[Metrics],
    model_2: Input[Model],
    metrics_2: Input[Metrics],
    model_3: Input[Model],
    metrics_3: Input[Metrics],
    best_model: Output[Model],
):
    """
    Select the best model based on accuracy metric from 3 training runs
    
    Args:
        model_1, model_2, model_3: Model artifacts from training runs
        metrics_1, metrics_2, metrics_3: Metrics artifacts from training runs
        best_model: Output artifact for the best selected model
    """
    import os
    import json
    import shutil

    models = [model_1, model_2, model_3]
    metrics = [metrics_1, metrics_2, metrics_3]
    
    best_idx = -1
    best_acc = -1.0

    # Iterate through metrics to find best accuracy
    for i, metric_artifact in enumerate(metrics):
        # Read metrics from metadata
        if hasattr(metric_artifact, 'metadata') and metric_artifact.metadata:
            acc = metric_artifact.metadata.get("accuracy", 0.0)
            lr = metric_artifact.metadata.get("learning_rate", 0.0)
        else:
            # Fallback: read from metrics.json file
            metric_path = os.path.join(metric_artifact.path, "metrics.json")
            if os.path.exists(metric_path):
                with open(metric_path) as f:
                    data = json.load(f)
                    acc = data.get("accuracy", 0.0)
                    lr = data.get("learning_rate", 0.0)
            else:
                acc = 0.0
                lr = 0.0
        
        print(f"Model {i+1} (lr={lr}): accuracy={acc}")

        if acc > best_acc:
            best_acc = acc
            best_idx = i

    print(f"\nBest model: Model {best_idx+1} with accuracy={best_acc}")

    # Copy best model to output
    # TensorFlow Serving requires models to be in a versioned directory (e.g., /1/)
    best_model_artifact = models[best_idx]
    
    if os.path.exists(best_model.path):
        shutil.rmtree(best_model.path)
    
    # Create version directory for TensorFlow Serving
    version_dir = os.path.join(best_model.path, "1")
    os.makedirs(version_dir, exist_ok=True)
    
    # Copy model files to versioned directory
    for item in os.listdir(best_model_artifact.path):
        src = os.path.join(best_model_artifact.path, item)
        dst = os.path.join(version_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    print(f"Model saved to versioned directory: {version_dir}")
