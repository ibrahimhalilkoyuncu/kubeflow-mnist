#!/usr/bin/env python3
"""
Test script for Fashion MNIST inference service
This script sends test images to the KServe inference endpoint and displays predictions with images
"""

import json
import requests
import numpy as np
from tensorflow import keras
import subprocess
import matplotlib.pyplot as plt

# Fashion MNIST class names
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def get_inference_url():
    """Get the inference service URL - using localhost with port-forward"""
    # Always use localhost since we're using port-forward
    url = "http://localhost:8080"
    print("Using port-forward URL. Make sure port-forward is running:")
    print("kubectl port-forward -n kubeflow <pod-name> 8080:8080")
    return url

def load_test_data(num_samples=5):
    """Load test images from Fashion MNIST dataset"""
    print("Loading Fashion MNIST test data...")
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # Normalize the images
    x_test = x_test.astype('float32') / 255.0
    
    # Select random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    test_images = x_test[indices]
    test_labels = y_test[indices]
    
    return test_images, test_labels

def prepare_request_data(images):
    """Prepare images for TensorFlow Serving request"""
    # TensorFlow Serving expects input in the format:
    # {"instances": [[image_data]]}
    instances = images.reshape(images.shape[0], -1).tolist()
    return {"instances": instances}

def send_inference_request(url, data):
    """Send inference request to the model"""
    # TensorFlow Serving REST API endpoint
    endpoint = f"{url}/v1/models/fashion-mnist-model:predict"
    
    headers = {
        "Content-Type": "application/json",
        "Host": "fashion-mnist-model-predictor.kubeflow.example.com"  # Required for Istio routing
    }
    
    try:
        print(f"\nSending request to: {endpoint}")
        response = requests.post(endpoint, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def display_predictions_with_images(images, true_labels, predictions):
    """Display prediction results with images"""
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    
    if predictions is None or 'predictions' not in predictions:
        print("No predictions received!")
        return
    
    pred_probs = np.array(predictions['predictions'])
    pred_classes = np.argmax(pred_probs, axis=1)
    
    # Create figure with subplots
    num_samples = len(images)
    fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        true_class = CLASS_NAMES[true_labels[i]]
        pred_class = CLASS_NAMES[pred_classes[i]]
        confidence = pred_probs[i][pred_classes[i]] * 100
        
        # Display image
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')
        
        # Set title with prediction
        is_correct = true_class == pred_class
        color = 'green' if is_correct else 'red'
        title = f"True: {true_class}\nPred: {pred_class}\n{confidence:.1f}%"
        axes[i].set_title(title, color=color, fontsize=10, weight='bold')
        
        # Print detailed results
        print(f"\nSample {i+1}:")
        print(f"  True Label:      {true_class}")
        print(f"  Predicted Label: {pred_class}")
        print(f"  Confidence:      {confidence:.2f}%")
        print(f"  Status:          {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        
        # Show top 3 predictions
        top3_indices = np.argsort(pred_probs[i])[-3:][::-1]
        print(f"  Top 3 predictions:")
        for idx in top3_indices:
            print(f"    - {CLASS_NAMES[idx]}: {pred_probs[i][idx]*100:.2f}%")
    
    plt.tight_layout()
    plt.savefig('inference_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: inference_results.png")
    plt.show()
    
    # Calculate accuracy
    accuracy = np.mean(pred_classes == true_labels) * 100
    print("\n" + "="*80)
    print(f"Overall Accuracy: {accuracy:.2f}% ({np.sum(pred_classes == true_labels)}/{len(true_labels)} correct)")
    print("="*80)

def main():
    """Main function to test the inference service"""
    print("="*80)
    print("Fashion MNIST Inference Test")
    print("="*80)
    
    # Get inference URL
    url = get_inference_url()
    print(f"\nInference Service URL: {url}")
    
    # Load test data
    num_samples = 5
    test_images, test_labels = load_test_data(num_samples)
    print(f"Loaded {num_samples} test samples")
    
    # Prepare request
    request_data = prepare_request_data(test_images)
    print(f"Request data shape: {len(request_data['instances'])} instances, each with {len(request_data['instances'][0])} features")
    
    # Send inference request
    predictions = send_inference_request(url, request_data)
    
    # Display results with images
    display_predictions_with_images(test_images, test_labels, predictions)
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)

if __name__ == "__main__":
    main()
