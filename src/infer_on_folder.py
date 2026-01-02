#!/usr/bin/env python3
"""
Batch inference script for detecting pets in multiple images from a folder.

Usage:
    python infer_on_folder.py --input path/to/folder [--model path/to/model.pt] [--output path/to/output/folder]
"""

import argparse
from pathlib import Path
import sys
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

from utils import load_image, draw_detections, save_image, get_image_files


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect pets in all images in a folder')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input folder containing images'
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
        default='../outputs/detections',
        help='Path to output folder (default: ../outputs/detections)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--skip-no-detections',
        action='store_true',
        help='Skip saving images with no detections'
    )
    
    return parser.parse_args()


def process_image(model, image_path, output_dir, conf_threshold, skip_no_detections):
    """
    Process a single image.
    
    Args:
        model: YOLO model
        image_path: Path to input image
        output_dir: Output directory path
        conf_threshold: Confidence threshold
        skip_no_detections: Whether to skip images with no detections
        
    Returns:
        Number of detections found
    """
    # Load image
    image = load_image(str(image_path))
    if image is None:
        return 0
    
    # Run inference
    results = model(image, conf=conf_threshold, verbose=False)
    
    # Process results
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = result.names
    
    num_detections = len(boxes)
    
    # Skip if no detections and flag is set
    if skip_no_detections and num_detections == 0:
        return 0
    
    # Draw detections
    output_image = draw_detections(
        image,
        boxes.tolist(),
        class_names,
        confidences.tolist(),
        class_ids.tolist()
    )
    
    # Save result
    output_path = output_dir / f"output_{image_path.name}"
    save_image(output_image, str(output_path))
    
    return num_detections


def main():
    """Main batch inference function."""
    args = parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {args.model}")
        print("Please download a YOLOv8 model or train your own.")
        sys.exit(1)
    
    # Get image files
    print(f"Scanning for images in {args.input}...")
    image_files = get_image_files(args.input)
    
    if not image_files:
        print(f"No images found in {args.input}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image(s)")
    
    # Load model
    print(f"Loading model from {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all images
    print(f"\nProcessing images (confidence threshold: {args.conf})...")
    total_detections = 0
    images_with_detections = 0
    
    for image_path in tqdm(image_files, desc="Processing"):
        num_detections = process_image(
            model,
            image_path,
            output_dir,
            args.conf,
            args.skip_no_detections
        )
        
        total_detections += num_detections
        if num_detections > 0:
            images_with_detections += 1
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total images processed: {len(image_files)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(image_files):.2f}")
    print(f"Results saved to: {output_dir}")
    print("="*50)
    print("\n✅ Batch inference complete!")


if __name__ == '__main__':
    main()
