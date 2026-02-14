#!/usr/bin/env python3
"""Generate example detection outputs for blog post."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ultralytics import YOLO  # type: ignore
import cv2
import numpy as np
from utils import draw_detections, extract_edge_features, detect_keypoints_orb

# Setup paths
MODEL_PATH = Path('models/yolov8n-pets.pt')
INPUT_DIR = Path('data/raw')
RESULTS_DIR = Path('examples/results')

# Load model
print("Loading model...")
model = YOLO(str(MODEL_PATH))
print("✓ Model loaded\n")

# Get all images
images = sorted(list(INPUT_DIR.glob('*.jpg')))[:5]  # Take first 5

print(f"Processing {len(images)} images...\n")

for img_path in images:
    print(f"Processing: {img_path.name}")
    
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ✗ Failed to read image")
        continue
    
    # Run detection
    results = model(img, conf=0.3, verbose=False)[0]
    
    # Create detection visualization
    det_img = img.copy()
    
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        class_names = {15: 'cat', 16: 'dog'}
        
        for bbox, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Determine class name
            if class_id in class_names:
                label = class_names[class_id]
            else:
                label = model.names[class_id]
            
            # Color based on class
            color = (255, 100, 50) if label == 'dog' else (50, 200, 100)
            
            # Draw box
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label_text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(det_img, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            cv2.putText(det_img, label_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        num_detections = len(boxes)
    else:
        num_detections = 0
    
    # Save detection result
    det_output = RESULTS_DIR / 'detections' / img_path.name
    cv2.imwrite(str(det_output), det_img)
    print(f"  ✓ Detection saved ({num_detections} pets found)")
    
    # Create enhanced version with CV features
    enhanced_img = det_img.copy()
    
    # Extract edges
    edge_features = extract_edge_features(img)
    edges = edge_features['canny']
    
    # Overlay edges in cyan
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[:, :, 1] = 255  # Cyan = BGR (255, 255, 0)
    edges_colored[:, :, 2] = 255
    mask = edges > 0
    enhanced_img[mask] = cv2.addWeighted(enhanced_img[mask], 0.7, edges_colored[mask], 0.3, 0)
    
    # Detect and draw keypoints
    keypoints, descriptors = detect_keypoints_orb(img, max_keypoints=100)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(enhanced_img, (x, y), 3, (0, 255, 255), -1)
        cv2.circle(enhanced_img, (x, y), 6, (0, 255, 255), 1)
    
    # Save enhanced result
    enh_output = RESULTS_DIR / 'enhanced' / img_path.name
    cv2.imwrite(str(enh_output), enhanced_img)
    print(f"  ✓ Enhanced saved ({len(keypoints)} keypoints)")
    
    print()

print("✅ All examples generated!")
print(f"\nResults saved to: {RESULTS_DIR}/")
