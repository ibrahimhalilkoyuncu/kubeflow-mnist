from kfp import dsl
from kfp import compiler

from components.train_component import train_model


@dsl.pipeline(
    name="fashion-mnist-hyperparameter-pipeline",
    description="Train Fashion-MNIST model with parallel hyperparameter runs"
)
def fashion_mnist_pipeline(
    epochs: int = 5,
    hidden_units: int = 128,
):
    """
    Kubeflow pipeline with parallel hyperparameter training
    """

    # Hyperparameters to try
    learning_rates = [0.001, 0.0005, 0.0001]

    # Fan-out: parallel training runs
    with dsl.ParallelFor(learning_rates) as lr:
        task = train_model(
            learning_rate=lr,
            epochs=epochs,
            hidden_units=hidden_units,
        )