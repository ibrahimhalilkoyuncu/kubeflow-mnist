import argparse
import os
import tensorflow as tf
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--model-dir", type=str, default="/mnt/model")
    parser.add_argument("--metrics-dir", type=str, required=True)
    return parser.parse_args()


def load_data():
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


def build_model(hidden_units, learning_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    args = parse_args()

    print("Training parameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Hidden units: {args.hidden_units}")
    print(f"  Model dir: {args.model_dir}")

    x_train, y_train, x_test, y_test = load_data()

    model = build_model(args.hidden_units, args.learning_rate)

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        validation_data=(x_test, y_test)
    )

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Final evaluation - loss: {loss}, accuracy: {accuracy}")

    os.makedirs(args.model_dir, exist_ok=True)
    model.save(args.model_dir)

    os.makedirs(args.metrics_dir, exist_ok=True)

    metrics = {
        "accuracy": accuracy,
        "loss": loss
    }

    # Write metrics for downstream Kubeflow steps
    with open(os.path.join(args.metrics_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
