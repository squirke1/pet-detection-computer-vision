"""
Example Python client for Pet Detection API

Demonstrates how to interact with the API programmatically.
"""

import requests
from pathlib import Path
from typing import Dict, List, Optional


class PetDetectionClient:
    """Client for Pet Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> Dict:
        """
        Check API health status.
        
        Returns:
            Health check response
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def detect(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None
    ) -> Dict:
        """
        Run pet detection on an image.
        
        Args:
            image_path: Path to image file
            conf_threshold: Optional confidence threshold override
            
        Returns:
            Detection results
        """
        # Prepare file
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        files = {
            'file': (image_path.name, open(image_path, 'rb'), 'image/jpeg')
        }
        
        # Prepare parameters
        params = {}
        if conf_threshold is not None:
            params['conf'] = conf_threshold
        
        # Make request
        response = requests.post(
            f"{self.base_url}/detect",
            files=files,
            params=params
        )
        response.raise_for_status()
        
        return response.json()
    
    def list_models(self) -> Dict:
        """
        List available models.
        
        Returns:
            Models information
        """
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def print_detections(self, result: Dict) -> None:
        """
        Pretty print detection results.
        
        Args:
            result: Detection result from API
        """
        if not result.get('success'):
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return
        
        num_detections = result['num_detections']
        inference_time = result['inference_time_ms']
        model = result['model_name']
        
        print(f"\n{'='*60}")
        print(f"DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"Inference Time: {inference_time:.2f} ms")
        print(f"Detections: {num_detections}")
        print(f"{'='*60}\n")
        
        if num_detections > 0:
            for i, det in enumerate(result['detections'], 1):
                print(f"{i}. {det['class_name'].upper()}")
                print(f"   Confidence: {det['confidence']:.2%}")
                print(f"   BBox: {det['bbox']}\n")
        else:
            print("No pets detected.\n")


def example_basic_detection():
    """Example: Basic detection on a single image."""
    print("\n=== Example 1: Basic Detection ===\n")
    
    # Create client
    client = PetDetectionClient()
    
    # Check health
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Model Loaded: {health['model_loaded']}\n")
    
    # Run detection
    result = client.detect("data/raw/test_image.jpg")
    client.print_detections(result)


def example_custom_threshold():
    """Example: Detection with custom confidence threshold."""
    print("\n=== Example 2: Custom Threshold ===\n")
    
    client = PetDetectionClient()
    
    # Higher threshold = fewer but more confident detections
    result = client.detect(
        "data/raw/test_image.jpg",
        conf_threshold=0.5
    )
    
    client.print_detections(result)


def example_batch_processing():
    """Example: Process multiple images."""
    print("\n=== Example 3: Batch Processing ===\n")
    
    client = PetDetectionClient()
    
    image_paths = [
        "data/raw/image1.jpg",
        "data/raw/image2.jpg",
        "data/raw/image3.jpg"
    ]
    
    results = []
    for img_path in image_paths:
        try:
            result = client.detect(img_path)
            results.append({
                'image': img_path,
                'detections': result['num_detections'],
                'time_ms': result['inference_time_ms']
            })
        except Exception as e:
            print(f"❌ Failed processing {img_path}: {e}")
    
    # Summary
    total_detections = sum(r['detections'] for r in results)
    avg_time = sum(r['time_ms'] for r in results) / len(results)
    
    print(f"\nProcessed {len(results)} images")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {avg_time:.2f} ms")


def example_error_handling():
    """Example: Proper error handling."""
    print("\n=== Example 4: Error Handling ===\n")
    
    client = PetDetectionClient()
    
    try:
        result = client.detect("nonexistent.jpg")
        client.print_detections(result)
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    # Run examples
    try:
        example_basic_detection()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_custom_threshold()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_error_handling()
    except Exception as e:
        print(f"Example 4 failed: {e}")
