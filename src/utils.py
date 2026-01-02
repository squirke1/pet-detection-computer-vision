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
