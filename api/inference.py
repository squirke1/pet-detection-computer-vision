"""
Inference logic wrapper for API.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics.models.yolo import YOLO

from api.models import Detection


class InferenceEngine:
    """Handles model loading and inference."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to YOLOv8 model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.model: Optional[YOLO] = None
        self.class_names = {0: 'dog', 1: 'cat'}  # Default COCO pet classes
        
    def load_model(self) -> None:
        """Load the YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        
        # Update class names from model if available
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def predict(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> Tuple[List[Detection], float]:
        """
        Run inference on an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Optional confidence threshold override
            
        Returns:
            Tuple of (detections list, inference_time_ms)
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        assert self.model is not None, "Model must be loaded"
        
        # Use instance threshold if not overridden
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        # Run inference
        start_time = time.perf_counter()
        results = self.model(image, conf=conf, verbose=False)[0]
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse results
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for bbox, conf, class_id in zip(boxes, confidences, class_ids):
                detection = Detection(
                    class_id=int(class_id),
                    class_name=self.class_names.get(int(class_id), f"class_{class_id}"),
                    confidence=float(conf),
                    bbox=bbox.tolist()
                )
                detections.append(detection)
        
        return detections, inference_time_ms
    
    def predict_from_bytes(
        self,
        image_bytes: bytes,
        conf_threshold: Optional[float] = None
    ) -> Tuple[List[Detection], float, Tuple[int, int, int]]:
        """
        Run inference on image bytes.
        
        Args:
            image_bytes: Raw image bytes
            conf_threshold: Optional confidence threshold override
            
        Returns:
            Tuple of (detections, inference_time_ms, image_shape)
            
        Raises:
            ValueError: If image cannot be decoded
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Run inference
        detections, inference_time = self.predict(image, conf_threshold)
        
        return detections, inference_time, image.shape
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {
                "loaded": False,
                "path": str(self.model_path),
                "size_mb": 0.0
            }
        
        size_mb = self.model_path.stat().st_size / (1024 * 1024)
        
        return {
            "loaded": True,
            "path": str(self.model_path),
            "size_mb": round(size_mb, 2),
            "class_names": self.class_names,
            "conf_threshold": self.conf_threshold
        }
