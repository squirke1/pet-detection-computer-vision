"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_image():
    """Create a simple test image (100x100 RGB)."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a white rectangle to have some features
    cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
    return image


@pytest.fixture
def sample_image_with_edges():
    """Create a test image with clear edges."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create a pattern with edges
    cv2.rectangle(image, (10, 10), (50, 50), (255, 0, 0), 2)
    cv2.circle(image, (75, 75), 20, (0, 255, 0), 2)
    return image


@pytest.fixture
def sample_gray_image():
    """Create a simple grayscale test image."""
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), 255, -1)
    return image


@pytest.fixture
def sample_detections():
    """Sample detection data for testing."""
    return {
        'boxes': [[10, 10, 50, 50], [60, 60, 90, 90]],
        'class_names': ['dog', 'cat'],
        'confidences': [0.95, 0.87],
        'class_ids': [0, 1]
    }


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temporary directory with test images."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    # Create a few test images
    for i in range(3):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = image_dir / f"test_image_{i}.jpg"
        cv2.imwrite(str(img_path), img)
    
    return image_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for output files."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
