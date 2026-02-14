#!/usr/bin/env python3
"""Create side-by-side comparison images."""

import cv2
import numpy as np
from pathlib import Path

# Setup paths
ORIG_DIR = Path('examples/results/originals')
DET_DIR = Path('examples/results/detections')
ENH_DIR = Path('examples/results/enhanced')
COMP_DIR = Path('examples/results/comparisons')

images = sorted(list(ORIG_DIR.glob('*.jpg')))[:5]

print(f"Creating {len(images)} comparison images...\n")

for img_path in images:
    print(f"Processing: {img_path.name}")
    
    # Load all three versions
    orig = cv2.imread(str(ORIG_DIR / img_path.name))
    det = cv2.imread(str(DET_DIR / img_path.name))
    enh = cv2.imread(str(ENH_DIR / img_path.name))
    
    if orig is None or det is None or enh is None:
        print(f"  ✗ Failed to load images")
        continue
    
    # Resize all to same height
    target_height = 400
    h, w = orig.shape[:2]
    target_width = int(w * target_height / h)
    
    orig_resized = cv2.resize(orig, (target_width, target_height))
    det_resized = cv2.resize(det, (target_width, target_height))
    enh_resized = cv2.resize(enh, (target_width, target_height))
    
    # Create labels
    label_height = 40
    label_width = target_width
    
    def create_label(text, color=(100, 100, 255)):
        label = np.ones((label_height, label_width, 3), dtype=np.uint8) * 255
        cv2.putText(label, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2, cv2.LINE_AA)
        return label
    
    label_orig = create_label("Original")
    label_det = create_label("Detection", (255, 100, 50))
    label_enh = create_label("Enhanced + CV Features", (50, 200, 100))
    
    # Stack images vertically with labels
    orig_with_label = np.vstack([label_orig, orig_resized])
    det_with_label = np.vstack([label_det, det_resized])
    enh_with_label = np.vstack([label_enh, enh_resized])
    
    # Add spacing between images
    spacing = np.ones((target_height + label_height, 20, 3), dtype=np.uint8) * 255
    
    # Combine horizontally
    comparison = np.hstack([
        orig_with_label,
        spacing,
        det_with_label,
        spacing,
        enh_with_label
    ])
    
    # Add title bar
    title_height = 60
    title_bar = np.ones((title_height, comparison.shape[1], 3), dtype=np.uint8) * 240
    title_text = f"Pet Detection Pipeline: {img_path.stem.replace('_', ' ').title()}"
    cv2.putText(title_bar, title_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (50, 50, 50), 2, cv2.LINE_AA)
    
    final = np.vstack([title_bar, comparison])
    
    # Save
    output_path = COMP_DIR / img_path.name
    cv2.imwrite(str(output_path), final)
    print(f"  ✓ Saved: {output_path.name}")

print(f"\n✅ All comparisons created in: {COMP_DIR}/")
