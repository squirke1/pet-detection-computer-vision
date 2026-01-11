"""
Unit tests for utility functions in src/utils.py
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src directory to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils import (
    load_image,
    save_image,
    draw_detections,
    get_image_files,
    extract_edge_features,
    detect_keypoints_sift,
    detect_keypoints_orb,
    detect_contours,
    draw_enhanced_detections
)


class TestLoadImage:
    """Tests for load_image function."""
    
    def test_load_existing_image(self, tmp_path, sample_image):
        """Test loading an existing image file."""
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), sample_image)
        
        loaded = load_image(str(image_path))
        assert loaded is not None
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == sample_image.shape
    
    def test_load_nonexistent_image(self):
        """Test loading a non-existent image returns None."""
        result = load_image("/nonexistent/path/image.jpg")
        assert result is None
    
    def test_load_invalid_image(self, tmp_path):
        """Test loading an invalid image file returns None."""
        invalid_path = tmp_path / "invalid.jpg"
        invalid_path.write_text("not an image")
        
        result = load_image(str(invalid_path))
        assert result is None


class TestSaveImage:
    """Tests for save_image function."""
    
    def test_save_image_success(self, tmp_path, sample_image):
        """Test successfully saving an image."""
        output_path = tmp_path / "output" / "test.jpg"
        
        result = save_image(sample_image, str(output_path))
        assert result is True
        assert output_path.exists()
    
    def test_save_image_creates_directory(self, tmp_path, sample_image):
        """Test that save_image creates parent directories."""
        output_path = tmp_path / "deep" / "nested" / "path" / "test.jpg"
        
        result = save_image(sample_image, str(output_path))
        assert result is True
        assert output_path.exists()
        assert output_path.parent.exists()
    
    def test_save_invalid_image(self, tmp_path):
        """Test saving invalid data returns False."""
        output_path = tmp_path / "test.jpg"
        invalid_data = "not an image"
        
        # This should raise an exception, not return False
        # cv2.imwrite doesn't handle invalid data gracefully
        with pytest.raises((cv2.error, TypeError)):
            save_image(invalid_data, str(output_path))


class TestDrawDetections:
    """Tests for draw_detections function."""
    
    def test_draw_single_detection(self, sample_image):
        """Test drawing a single detection box."""
        boxes = [[10, 10, 50, 50]]
        class_names = ['dog']
        confidences = [0.95]
        class_ids = [0]
        
        result = draw_detections(sample_image, boxes, class_names, confidences, class_ids)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
        # Image should be modified (different from original)
        assert not np.array_equal(result, sample_image)
    
    def test_draw_multiple_detections(self, sample_image, sample_detections):
        """Test drawing multiple detection boxes."""
        result = draw_detections(
            sample_image,
            sample_detections['boxes'],
            sample_detections['class_names'],
            sample_detections['confidences'],
            sample_detections['class_ids']
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
    
    def test_draw_with_different_classes(self, sample_image):
        """Test drawing boxes for different classes (dog and cat)."""
        boxes = [[10, 10, 40, 40], [50, 50, 80, 80]]
        class_names = ['dog', 'cat']
        confidences = [0.9, 0.85]
        class_ids = [0, 1]
        
        result = draw_detections(sample_image, boxes, class_names, confidences, class_ids)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape


class TestGetImageFiles:
    """Tests for get_image_files function."""
    
    def test_get_image_files_from_directory(self, temp_image_dir):
        """Test getting image files from a directory."""
        files = get_image_files(str(temp_image_dir))
        
        assert len(files) == 3
        assert all(f.suffix == '.jpg' for f in files)
        assert all(f.exists() for f in files)
    
    def test_get_image_files_with_different_extensions(self, tmp_path):
        """Test getting files with different image extensions."""
        # Create images with different extensions
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            img_path = tmp_path / f"image{ext}"
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
        
        files = get_image_files(str(tmp_path))
        assert len(files) == 4
    
    def test_get_image_files_empty_directory(self, tmp_path):
        """Test getting files from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        files = get_image_files(str(empty_dir))
        assert files == []
    
    def test_get_image_files_nonexistent_directory(self):
        """Test getting files from non-existent directory."""
        files = get_image_files("/nonexistent/directory")
        assert files == []


class TestExtractEdgeFeatures:
    """Tests for extract_edge_features function."""
    
    def test_extract_edge_features_returns_dict(self, sample_image_with_edges):
        """Test that extract_edge_features returns a dictionary."""
        result = extract_edge_features(sample_image_with_edges)
        
        assert isinstance(result, dict)
        assert 'canny' in result
        assert 'sobel' in result
        assert 'laplacian' in result
        assert 'gray' in result
    
    def test_edge_features_correct_types(self, sample_image_with_edges):
        """Test that edge features have correct types."""
        result = extract_edge_features(sample_image_with_edges)
        
        for key, value in result.items():
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.uint8
    
    def test_edge_features_correct_shapes(self, sample_image_with_edges):
        """Test that edge features have correct shapes."""
        result = extract_edge_features(sample_image_with_edges)
        h, w = sample_image_with_edges.shape[:2]
        
        assert result['canny'].shape == (h, w)
        assert result['sobel'].shape == (h, w)
        assert result['laplacian'].shape == (h, w)
        assert result['gray'].shape == (h, w)
    
    def test_edge_detection_finds_edges(self, sample_image_with_edges):
        """Test that edge detection actually finds edges."""
        result = extract_edge_features(sample_image_with_edges)
        
        # Should have non-zero pixels where edges exist
        assert np.any(result['canny'] > 0)
        assert np.any(result['sobel'] > 0)
        assert np.any(result['laplacian'] > 0)


class TestDetectKeypointsSIFT:
    """Tests for detect_keypoints_sift function."""
    
    def test_detect_sift_keypoints(self, sample_image_with_edges):
        """Test SIFT keypoint detection."""
        keypoints, descriptors = detect_keypoints_sift(sample_image_with_edges)
        
        assert isinstance(keypoints, (list, tuple))
        # May or may not find keypoints depending on image
        if len(keypoints) > 0:
            assert descriptors is not None
            assert isinstance(descriptors, np.ndarray)
    
    def test_sift_max_keypoints_limit(self, sample_image_with_edges):
        """Test that max_keypoints parameter limits keypoint count."""
        keypoints_small, _ = detect_keypoints_sift(sample_image_with_edges, max_keypoints=10)
        keypoints_large, _ = detect_keypoints_sift(sample_image_with_edges, max_keypoints=500)
        
        assert isinstance(keypoints_small, (list, tuple))
        assert isinstance(keypoints_large, (list, tuple))
        # Should find more keypoints with higher limit
        assert len(keypoints_small) <= len(keypoints_large)


class TestDetectKeypointsORB:
    """Tests for detect_keypoints_orb function."""
    
    def test_detect_orb_keypoints(self, sample_image_with_edges):
        """Test ORB keypoint detection."""
        keypoints, descriptors = detect_keypoints_orb(sample_image_with_edges)
        
        assert isinstance(keypoints, (list, tuple))
        # May or may not find keypoints depending on image
        if len(keypoints) > 0:
            assert descriptors is not None
            assert isinstance(descriptors, np.ndarray)
    
    def test_orb_max_keypoints_limit(self, sample_image_with_edges):
        """Test that max_keypoints parameter is respected."""
        keypoints, _ = detect_keypoints_orb(sample_image_with_edges, max_keypoints=10)
        
        assert isinstance(keypoints, (list, tuple))
        assert len(keypoints) <= 10


class TestDetectContours:
    """Tests for detect_contours function."""
    
    def test_detect_contours(self, sample_image_with_edges):
        """Test contour detection."""
        contours = detect_contours(sample_image_with_edges)
        
        assert isinstance(contours, list)
        # Should find at least some contours in the test image
        assert len(contours) > 0
    
    def test_contours_min_area_filter(self, sample_image_with_edges):
        """Test that min_area parameter filters small contours."""
        contours_small = detect_contours(sample_image_with_edges, min_area=10)
        contours_large = detect_contours(sample_image_with_edges, min_area=1000)
        
        # Should filter more contours with larger min_area
        assert len(contours_large) <= len(contours_small)
    
    def test_contours_are_numpy_arrays(self, sample_image_with_edges):
        """Test that contours are numpy arrays."""
        contours = detect_contours(sample_image_with_edges)
        
        for contour in contours:
            assert isinstance(contour, np.ndarray)


class TestDrawEnhancedDetections:
    """Tests for draw_enhanced_detections function."""
    
    def test_draw_enhanced_basic(self, sample_image, sample_detections):
        """Test basic enhanced detection drawing."""
        result = draw_enhanced_detections(
            sample_image,
            sample_detections['boxes'],
            sample_detections['class_names'],
            sample_detections['confidences'],
            sample_detections['class_ids']
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
    
    def test_draw_enhanced_with_edges(self, sample_image, sample_detections):
        """Test enhanced drawing with edge overlay."""
        result = draw_enhanced_detections(
            sample_image,
            sample_detections['boxes'],
            sample_detections['class_names'],
            sample_detections['confidences'],
            sample_detections['class_ids'],
            show_edges=True
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
    
    def test_draw_enhanced_with_keypoints(self, sample_image_with_edges, sample_detections):
        """Test enhanced drawing with keypoints."""
        result = draw_enhanced_detections(
            sample_image_with_edges,
            sample_detections['boxes'],
            sample_detections['class_names'],
            sample_detections['confidences'],
            sample_detections['class_ids'],
            show_keypoints=True
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_with_edges.shape
    
    def test_draw_enhanced_with_all_features(self, sample_image_with_edges, sample_detections):
        """Test enhanced drawing with all features enabled."""
        result = draw_enhanced_detections(
            sample_image_with_edges,
            sample_detections['boxes'],
            sample_detections['class_names'],
            sample_detections['confidences'],
            sample_detections['class_ids'],
            show_edges=True,
            show_keypoints=True
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_with_edges.shape
