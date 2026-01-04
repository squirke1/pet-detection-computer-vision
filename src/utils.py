"""
Utility functions for pet detection inference.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array, or None if loading fails
    """
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    return image


def draw_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    class_names: List[str],
    confidences: List[float],
    class_ids: List[int]
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Input image
        boxes: List of bounding boxes [x1, y1, x2, y2]
        class_names: List of class names
        confidences: List of confidence scores
        class_ids: List of class IDs
        
    Returns:
        Image with drawn detections
    """
    output_image = image.copy()
    
    # Define colors for different classes
    colors = {
        'dog': (255, 0, 0),    # Blue
        'cat': (0, 255, 0),    # Green
    }
    
    for box, class_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[class_id]
        color = colors.get(class_name.lower(), (0, 255, 255))  # Default yellow
        
        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {conf:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            output_image,
            (x1, y1 - label_height - baseline - 10),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            output_image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return output_image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to disk.
    
    Args:
        image: Image to save
        output_path: Path where to save the image
        
    Returns:
        True if successful, False otherwise
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"Saved result to: {output_path}")
    else:
        print(f"Error: Could not save image to {output_path}")
    
    return success


def get_image_files(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[Path]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory to search
        extensions: Tuple of valid image extensions
        
    Returns:
        List of image file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}")
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def extract_edge_features(image: np.ndarray) -> dict:
    """
    Extract edge detection features from an image.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Dictionary containing different edge detection results
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    canny = cv2.Canny(gray, 50, 150)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
    
    # Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    return {
        'canny': canny,
        'sobel': sobel_combined,
        'laplacian': laplacian,
        'gray': gray
    }


def detect_keypoints_sift(image: np.ndarray, max_keypoints: int = 500) -> Tuple[List, Optional[np.ndarray]]:
    """
    Detect SIFT keypoints and descriptors.
    
    Args:
        image: Input image (BGR format)
        max_keypoints: Maximum number of keypoints to detect
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def detect_keypoints_orb(image: np.ndarray, max_keypoints: int = 500) -> Tuple[List, Optional[np.ndarray]]:
    """
    Detect ORB keypoints and descriptors.
    
    Args:
        image: Input image (BGR format)
        max_keypoints: Maximum number of keypoints to detect
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def detect_contours(image: np.ndarray, min_area: int = 500) -> List:
    """
    Detect contours in an image.
    
    Args:
        image: Input image (BGR format)
        min_area: Minimum contour area to filter small contours
        
    Returns:
        List of contours (each contour is a numpy array of points)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return large_contours


def draw_enhanced_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    class_names: List[str],
    confidences: List[float],
    class_ids: List[int],
    show_edges: bool = True,
    show_keypoints: bool = False
) -> np.ndarray:
    """
    Draw enhanced detections with optional edge detection and keypoints.
    
    Args:
        image: Input image
        boxes: List of bounding boxes [x1, y1, x2, y2]
        class_names: List of class names
        confidences: List of confidence scores
        class_ids: List of class IDs
        show_edges: Whether to overlay edge detection
        show_keypoints: Whether to show keypoints
        
    Returns:
        Image with enhanced visualizations
    """
    output_image = image.copy()
    
    # Add edge overlay if requested
    if show_edges:
        edges = extract_edge_features(image)
        # Create colored edge overlay (cyan edges)
        edge_overlay = np.zeros_like(image)
        edge_overlay[edges['canny'] > 0] = [255, 255, 0]  # Cyan edges
        output_image = cv2.addWeighted(output_image, 0.85, edge_overlay, 0.15, 0)
    
    # Add keypoints if requested
    if show_keypoints:
        keypoints, _ = detect_keypoints_orb(image, max_keypoints=200)
        output_image = cv2.drawKeypoints(
            output_image, keypoints, None,
            color=(0, 255, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    
    # Draw standard YOLO detections on top
    colors = {
        'dog': (255, 0, 0),    # Blue
        'cat': (0, 255, 0),    # Green
    }
    
    for box, class_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[class_id]
        color = colors.get(class_name.lower(), (0, 255, 255))
        
        # Draw bounding box with thicker line
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{class_name}: {conf:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        cv2.rectangle(
            output_image,
            (x1, y1 - label_height - baseline - 10),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            output_image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    
    return output_image
