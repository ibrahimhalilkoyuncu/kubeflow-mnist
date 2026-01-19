from kfp import dsl
from components.train_component import train_model
from components.select_best_model import select_best_model_manual


@dsl.pipeline(
    name="fashion-mnist-hyperparameter-pipeline",
    description="Train Fashion-MNIST with parallel hyperparameter runs and select best model",
)
def fashion_mnist_pipeline(
    epochs: int = 5,
    hidden_units: int = 128,
):
    # Train with different learning rates
    train_task_1 = train_model(
        learning_rate=0.001,
        epochs=epochs,
        hidden_units=hidden_units,
    )
    
    train_task_2 = train_model(
        learning_rate=0.0005,
        epochs=epochs,
        hidden_units=hidden_units,
    )
    
    train_task_3 = train_model(
        learning_rate=0.0001,
        epochs=epochs,
        hidden_units=hidden_units,
    )

    # Select best model
    select_best_model_manual(
        model_1=train_task_1.outputs["model"],
        metrics_1=train_task_1.outputs["metrics"],
        model_2=train_task_2.outputs["model"],
        metrics_2=train_task_2.outputs["metrics"],
        model_3=train_task_3.outputs["model"],
        metrics_3=train_task_3.outputs["metrics"],
    )
