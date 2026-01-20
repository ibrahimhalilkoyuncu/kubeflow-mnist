from kfp import dsl
from kfp.dsl import Input, Model


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["kubernetes==28.1.0"]
)
def deploy_to_kserve(
    model: Input[Model],
    model_name: str = "fashion-mnist-model",
    namespace: str = "kubeflow",
    min_replicas: int = 1,
    max_replicas: int = 3,
):
    """
    Deploy the best model to KServe InferenceService
    
    Args:
        model: The best model artifact from the pipeline
        model_name: Name for the InferenceService
        namespace: Kubernetes namespace to deploy to
        min_replicas: Minimum number of replicas
        max_replicas: Maximum number of replicas
    """
    import os
    from kubernetes import client, config
    
    # Load in-cluster config
    config.load_incluster_config()
    
    # Get the model URI from the artifact
    # The model.uri will be something like: minio://mlpipeline/v2/artifacts/<pipeline-run-id>/<task-id>/best_model
    model_uri = model.uri
    
    # Convert minio:// to s3:// for KServe
    # KServe uses S3 protocol to access MinIO
    if model_uri.startswith("minio://"):
        # Extract the path after minio://
        path = model_uri.replace("minio://", "")
        # Construct S3 URI pointing to MinIO service
        storage_uri = f"s3://{path}"
    else:
        storage_uri = model_uri
    
    print(f"Model URI: {model_uri}")
    print(f"Storage URI for KServe: {storage_uri}")
    
    # Define the InferenceService manifest
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "annotations": {
                "sidecar.istio.io/inject": "true"
            }
        },
        "spec": {
            "predictor": {
                "serviceAccountName": "kserve-sa",
                "tensorflow": {
                    "storageUri": storage_uri,
                    "runtimeVersion": "2.14.0",
                    "resources": {
                        "requests": {
                            "cpu": "500m",
                            "memory": "1Gi"
                        },
                        "limits": {
                            "cpu": "1",
                            "memory": "2Gi"
                        }
                    },
                    "protocolVersion": "v1"
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "scaleTarget": 1,
                "scaleMetric": "concurrency"
            }
        }
    }
    
    # Create custom object API client
    api = client.CustomObjectsApi()
    
    try:
        # Try to get existing InferenceService
        existing = api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name
        )
        
        # Update existing InferenceService
        print(f"Updating existing InferenceService: {model_name}")
        api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
            body=inference_service
        )
        print(f"InferenceService {model_name} updated successfully")
        
    except client.exceptions.ApiException as e:
        if e.status == 404:
            # Create new InferenceService
            print(f"Creating new InferenceService: {model_name}")
            api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=inference_service
            )
            print(f"InferenceService {model_name} created successfully")
        else:
            raise
    
    print(f"\nDeployment complete!")
    print(f"Model: {model_name}")
    print(f"Namespace: {namespace}")
    print(f"Storage URI: {storage_uri}")
