"""
Integration tests for inference scripts.

These tests verify that the complete inference workflows work correctly.
Note: These tests will be skipped if the model file is not available.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys
import subprocess

# Add src directory to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Check if model exists
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "yolov8n-pets.pt"
MODEL_EXISTS = MODEL_PATH.exists()

# Skip all tests in this module if model doesn't exist
pytestmark = pytest.mark.skipif(
    not MODEL_EXISTS,
    reason="YOLOv8 model not available for integration testing"
)


@pytest.fixture
def test_image_with_pet(tmp_path):
    """
    Create a synthetic test image.
    Note: This won't actually contain a pet, but will test the inference pipeline.
    """
    image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    # Add some structure
    cv2.rectangle(image, (100, 100), (300, 300), (255, 128, 0), -1)
    
    image_path = tmp_path / "test_pet.jpg"
    cv2.imwrite(str(image_path), image)
    return image_path


class TestInferOnImageScript:
    """Integration tests for infer_on_image.py script."""
    
    def test_script_runs_successfully(self, test_image_with_pet, tmp_path):
        """Test that the inference script runs without errors."""
        output_path = tmp_path / "output.jpg"
        
        cmd = [
            sys.executable,
            str(src_path / "infer_on_image.py"),
            "--image", str(test_image_with_pet),
            "--model", str(MODEL_PATH),
            "--output", str(output_path),
            "--conf", "0.25"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check that script completed successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        assert output_path.exists(), "Output image was not created"
    
    def test_script_with_high_confidence(self, test_image_with_pet, tmp_path):
        """Test inference with high confidence threshold."""
        output_path = tmp_path / "output_high_conf.jpg"
        
        cmd = [
            sys.executable,
            str(src_path / "infer_on_image.py"),
            "--image", str(test_image_with_pet),
            "--model", str(MODEL_PATH),
            "--output", str(output_path),
            "--conf", "0.9"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert output_path.exists()
    
    def test_script_with_missing_image(self, tmp_path):
        """Test that script handles missing image gracefully."""
        cmd = [
            sys.executable,
            str(src_path / "infer_on_image.py"),
            "--image", "/nonexistent/image.jpg",
            "--model", str(MODEL_PATH)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should exit with error code
        assert result.returncode != 0


class TestInferOnFolderScript:
    """Integration tests for infer_on_folder.py script."""
    
    def test_folder_inference(self, temp_image_dir, tmp_path):
        """Test batch inference on a folder of images."""
        output_dir = tmp_path / "outputs"
        
        cmd = [
            sys.executable,
            str(src_path / "infer_on_folder.py"),
            "--input", str(temp_image_dir),
            "--model", str(MODEL_PATH),
            "--output", str(output_dir),
            "--conf", "0.25"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        assert output_dir.exists()
        
        # Check that output images were created
        output_images = list(output_dir.glob("*.jpg"))
        assert len(output_images) > 0, "No output images were created"
    
    def test_folder_with_no_images(self, tmp_path):
        """Test inference on empty folder."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "outputs"
        
        cmd = [
            sys.executable,
            str(src_path / "infer_on_folder.py"),
            "--input", str(empty_dir),
            "--model", str(MODEL_PATH),
            "--output", str(output_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should complete but with message about no images
        assert result.returncode == 0


class TestEnhancedInferenceScript:
    """Integration tests for infer_enhanced.py script."""
    
    def test_enhanced_inference_basic(self, test_image_with_pet, tmp_path):
        """Test enhanced inference without extra features."""
        output_path = tmp_path / "enhanced_output.jpg"
        
        cmd = [
            sys.executable,
            str(src_path / "infer_enhanced.py"),
            "--image", str(test_image_with_pet),
            "--model", str(MODEL_PATH),
            "--output", str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        assert output_path.exists()
    
    def test_enhanced_inference_with_edges(self, test_image_with_pet, tmp_path):
        """Test enhanced inference with edge detection."""
        output_path = tmp_path / "enhanced_edges.jpg"
        
        cmd = [
            sys.executable,
            str(src_path / "infer_enhanced.py"),
            "--image", str(test_image_with_pet),
            "--model", str(MODEL_PATH),
            "--output", str(output_path),
            "--show-edges"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert output_path.exists()
    
    def test_enhanced_inference_with_keypoints(self, test_image_with_pet, tmp_path):
        """Test enhanced inference with keypoint detection."""
        output_path = tmp_path / "enhanced_keypoints.jpg"
        
        cmd = [
            sys.executable,
            str(src_path / "infer_enhanced.py"),
            "--image", str(test_image_with_pet),
            "--model", str(MODEL_PATH),
            "--output", str(output_path),
            "--show-keypoints"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert output_path.exists()
    
    def test_enhanced_inference_all_features(self, test_image_with_pet, tmp_path):
        """Test enhanced inference with all features enabled."""
        output_path = tmp_path / "enhanced_all.jpg"
        
        cmd = [
            sys.executable,
            str(src_path / "infer_enhanced.py"),
            "--image", str(test_image_with_pet),
            "--model", str(MODEL_PATH),
            "--output", str(output_path),
            "--show-edges",
            "--show-keypoints"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert output_path.exists()


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_complete_detection_workflow(self, test_image_with_pet, tmp_path):
        """Test complete workflow: load -> detect -> save."""
        from utils import load_image, save_image
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(str(MODEL_PATH))
        
        # Load image
        image = load_image(str(test_image_with_pet))
        assert image is not None
        
        # Run inference
        results = model(image, conf=0.25, verbose=False)
        assert len(results) > 0
        
        # Save result
        output_path = tmp_path / "workflow_output.jpg"
        result_img = results[0].plot()
        success = save_image(result_img, str(output_path))
        
        assert success
        assert output_path.exists()
