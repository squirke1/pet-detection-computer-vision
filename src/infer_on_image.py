#!/usr/bin/env python3
"""
Inference script for detecting pets in a single image using YOLOv8.

Usage:
    python infer_on_image.py --image path/to/image.jpg [--model path/to/model.pt] [--output path/to/output.jpg]
"""

import argparse
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

from utils import load_image, draw_detections, save_image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect pets in a single image')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='../models/yolov8n-pets.pt',
        help='Path to YOLOv8 model (default: ../models/yolov8n-pets.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output image (default: ../outputs/detections/output_<original_name>)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display result in a window'
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {args.model}")
        print("Please download a YOLOv8 model or train your own.")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load image
    print(f"Loading image from {args.image}...")
    image = load_image(args.image)
    if image is None:
        sys.exit(1)
    
    # Run inference
    print(f"Running inference (confidence threshold: {args.conf})...")
    results = model(image, conf=args.conf, verbose=False)
    
    # Process results
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = result.names
    
    # Print detections
    print(f"\nDetected {len(boxes)} object(s):")
    for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
        class_name = class_names[class_id]
        print(f"  {i+1}. {class_name} (confidence: {conf:.2f})")
    
    # Draw detections on image
    output_image = draw_detections(
        image,
        boxes.tolist(),
        class_names,
        confidences.tolist(),
        class_ids.tolist()
    )
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.image)
        output_filename = f"output_{input_path.name}"
        output_path = f"../outputs/detections/{output_filename}"
    else:
        output_path = args.output
    
    # Save result
    save_image(output_image, output_path)
    
    # Show result if requested
    if args.show:
        import cv2
        cv2.imshow('Pet Detection', output_image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n✅ Inference complete!")


if __name__ == '__main__':
    main()
