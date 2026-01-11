#!/usr/bin/env python3
"""
Enhanced inference script for detecting pets with edge detection features.

This script combines YOLOv8 object detection with computer vision features:
- Edge detection (Canny, Sobel, Laplacian)
- Keypoint detection (SIFT, ORB)
- Contour detection

Usage:
    python infer_enhanced.py --image path/to/image.jpg [options]
"""

import argparse
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

from utils import (
    load_image, 
    save_image, 
    extract_edge_features,
    detect_keypoints_sift,
    detect_keypoints_orb,
    detect_contours,
    draw_enhanced_detections
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced pet detection with edge detection features'
    )
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
        help='Path to save output image (default: ../outputs/detections/enhanced_<original_name>)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--show-edges',
        action='store_true',
        help='Overlay edge detection on output'
    )
    parser.add_argument(
        '--show-keypoints',
        action='store_true',
        help='Show keypoint detection on output'
    )
    parser.add_argument(
        '--analyze-features',
        action='store_true',
        help='Print detailed feature analysis'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display result in a window'
    )
    
    return parser.parse_args()


def analyze_features(image, verbose=True):
    """
    Analyze and extract features from the image.
    
    Args:
        image: Input image
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with feature analysis results
    """
    # Edge detection
    edges = extract_edge_features(image)
    
    # Keypoint detection
    keypoints_sift, descriptors_sift = detect_keypoints_sift(image)
    keypoints_orb, descriptors_orb = detect_keypoints_orb(image)
    
    # Contour detection
    contours = detect_contours(image, min_area=500)
    
    results = {
        'edges': edges,
        'sift_keypoints': len(keypoints_sift),
        'orb_keypoints': len(keypoints_orb),
        'contours': len(contours)
    }
    
    if verbose:
        print("\n" + "="*60)
        print("FEATURE ANALYSIS")
        print("="*60)
        print(f"\n🔍 Edge Detection:")
        print(f"  • Canny edges detected: {edges['canny'].sum() // 255} pixels")
        print(f"  • Sobel edges detected: {edges['sobel'].sum() // 255} pixels")
        print(f"  • Laplacian edges detected: {edges['laplacian'].sum() // 255} pixels")
        
        print(f"\n📍 Keypoint Detection:")
        print(f"  • SIFT keypoints: {len(keypoints_sift)}")
        print(f"  • ORB keypoints: {len(keypoints_orb)}")
        
        print(f"\n🔲 Contour Detection:")
        print(f"  • Large contours found: {len(contours)} (area > 500px²)")
        
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            print(f"  • Largest contour area: {max(areas):.0f}px²")
            print(f"  • Average contour area: {sum(areas)/len(areas):.0f}px²")
        print("="*60 + "\n")
    
    return results


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
    
    # Analyze features if requested
    if args.analyze_features:
        import cv2
        analyze_features(image, verbose=True)
    
    # Run YOLO inference
    print(f"\nRunning YOLO inference (confidence threshold: {args.conf})...")
    results = model(image, conf=args.conf, verbose=False)
    
    # Process results
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = result.names
    
    # Print detections
    print(f"\n🐾 Detected {len(boxes)} pet(s):")
    for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
        class_name = class_names[class_id]
        x1, y1, x2, y2 = box
        print(f"  {i+1}. {class_name.upper()} (confidence: {conf:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    # Draw enhanced detections
    output_image = draw_enhanced_detections(
        image,
        boxes.tolist(),
        class_names,
        confidences.tolist(),
        class_ids.tolist(),
        show_edges=args.show_edges,
        show_keypoints=args.show_keypoints
    )
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.image)
        output_filename = f"enhanced_{input_path.name}"
        output_path = f"../outputs/detections/{output_filename}"
    else:
        output_path = args.output
    
    # Save result
    save_image(output_image, output_path)
    
    # Show result if requested
    if args.show:
        import cv2
        cv2.imshow('Enhanced Pet Detection', output_image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n✅ Enhanced inference complete!")
    
    # Print feature info if enabled
    if args.show_edges or args.show_keypoints:
        print("\n💡 Visualization features enabled:")
        if args.show_edges:
            print("  • Edge detection overlay (cyan)")
        if args.show_keypoints:
            print("  • ORB keypoints (yellow circles)")


if __name__ == '__main__':
    main()
