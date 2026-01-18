from kfp import compiler
from kfp.compiler.compiler_utils import KubernetesManifestOptions

from pipeline.pipeline import fashion_mnist_pipeline


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=fashion_mnist_pipeline,
        package_path="fashion-mnist-pipeline-k8s.yaml",

        kubernetes_manifest_format=True,
        kubernetes_manifest_options=KubernetesManifestOptions(
            pipeline_name="fashion-mnist-hyperparameter-pipeline",
            pipeline_display_name="Fashion MNIST Hyperparameter Pipeline",
            pipeline_version_name="v1",
            pipeline_version_display_name="Fashion MNIST HP v1",
            namespace="kubeflow",
            include_pipeline_manifest=True,
        ),
    )