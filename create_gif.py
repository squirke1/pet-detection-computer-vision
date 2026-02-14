#!/usr/bin/env python3
"""Create an animated GIF showing detection in action."""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Setup paths
DET_DIR = Path('examples/results/detections')
OUTPUT_DIR = Path('examples/results/videos')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(list(DET_DIR.glob('*.jpg')))[:5]

print(f"Creating animated GIF from {len(images)} images...")

# Load and resize all images
frames = []
target_size = (640, 480)

for img_path in images:
    print(f"  Loading: {img_path.name}")
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    # Resize
    img_resized = cv2.resize(img, target_size)
    
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Add frame number overlay
    frame_num = len(frames) + 1
    cv2.putText(img_rgb, f"Frame {frame_num}/{len(images)}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img_rgb, f"Frame {frame_num}/{len(images)}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_rgb)
    frames.append(pil_img)

# Create GIF
output_path = OUTPUT_DIR / 'detection_demo.gif'
print(f"\nCreating GIF: {output_path}")

# Save as animated GIF
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=1000,  # 1 second per frame
    loop=0  # Loop forever
)

print(f"✓ GIF created: {output_path}")
print(f"  Frames: {len(frames)}")
print(f"  Duration: {len(frames)} seconds")
print(f"  Size: {target_size[0]}x{target_size[1]}")

# Also create a faster version
output_fast = OUTPUT_DIR / 'detection_demo_fast.gif'
frames[0].save(
    output_fast,
    save_all=True,
    append_images=frames[1:],
    duration=500,  # 0.5 seconds per frame
    loop=0
)
print(f"✓ Fast GIF created: {output_fast}")

print(f"\n✅ Animated GIFs ready for blog post!")
