#!/usr/bin/env python3
"""Create heatmap visualization showing pet activity zones."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ultralytics import YOLO

# Setup
MODEL_PATH = Path('models/yolov8n-pets.pt')
INPUT_DIR = Path('examples/results/originals')
OUTPUT_DIR = Path('examples/results/heatmaps')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading model...")
model = YOLO(str(MODEL_PATH))
print("✓ Model loaded\n")

# Get all images
images = sorted(list(INPUT_DIR.glob('*.jpg')))

# Create a base image size for heatmap
heatmap_size = (720, 1280, 3)  # HxW format
heatmap_density = np.zeros(heatmap_size[:2], dtype=np.float32)

print(f"Processing {len(images)} images for heatmap...\n")

# Collect detection centers
detections_list = []

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    # Run detection
    results = model(img, conf=0.3, verbose=False)[0]
    
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            # Calculate center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Normalize to heatmap size
            h, w = img.shape[:2]
            norm_x = int(cx * heatmap_size[1] / w)
            norm_y = int(cy * heatmap_size[0] / h)
            
            detections_list.append((norm_x, norm_y))
            print(f"  {img_path.name}: Detection at ({norm_x}, {norm_y})")

print(f"\n Total detections: {len(detections_list)}")

# Create heatmap with Gaussian blur
for x, y in detections_list:
    # Add Gaussian blob
    y1, y2 = max(0, y-60), min(heatmap_size[0], y+60)
    x1, x2 = max(0, x-60), min(heatmap_size[1], x+60)
    
    Y, X = np.ogrid[y1:y2, x1:x2]
    gaussian = np.exp(-((X-x)**2 + (Y-y)**2) / (2 * 40**2))
    heatmap_density[y1:y2, x1:x2] += gaussian * 100

# Apply additional blur
heatmap_density = cv2.GaussianBlur(heatmap_density, (51, 51), 0)

# Normalize
if heatmap_density.max() > 0:
    heatmap_density = heatmap_density / heatmap_density.max()

# Create visualization
print("\nCreating heatmap visualizations...")

# 1. Pure heatmap
plt.figure(figsize=(16, 9))
plt.imshow(heatmap_density, cmap='jet', interpolation='bilinear')
plt.colorbar(label='Activity Density')
plt.title('Pet Activity Heatmap\n(Showing where pets were detected across multiple images)', 
         fontsize=16, pad=20)
plt.xlabel('Horizontal Position (pixels)')
plt.ylabel('Vertical Position (pixels)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_pure.png', dpi=150, bbox_inches='tight')
print("  ✓ Pure heatmap saved")
plt.close()

# 2. Heatmap with zones overlay
plt.figure(figsize=(16, 9))
plt.imshow(heatmap_density, cmap='hot', interpolation='bilinear', alpha=0.8)
plt.colorbar(label='Activity Level')
plt.title('Pet Activity Zones Analysis', fontsize=16, pad=20)

# Add zone annotations
zones = [
    (0.25, 0.25, 'Zone A\nMedium Activity'),
    (0.75, 0.25, 'Zone B\nLow Activity'),
    (0.5, 0.75, 'Zone C\nHigh Activity'),
]

for zx, zy, label in zones:
    px = int(zx * heatmap_size[1])
    py = int(zy * heatmap_size[0])
    plt.plot(px, py, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=2)
    plt.text(px, py - 50, label, color='white', fontsize=10, 
            ha='center', va='top', weight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.xlabel('Horizontal Position (pixels)')
plt.ylabel('Vertical Position (pixels)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_zones.png', dpi=150, bbox_inches='tight')
print("  ✓ Zones heatmap saved")
plt.close()

# 3. Heatmap with statistics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# Main heatmap
im = ax1.imshow(heatmap_density, cmap='viridis', interpolation='bilinear')
ax1.set_title('Pet Activity Heatmap', fontsize=14, pad=15)
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
plt.colorbar(im, ax=ax1, label='Density')

# Statistics
stats_text = f"""
🐾 Pet Detection Statistics

Total Detections: {len(detections_list)}
Images Processed: {len(images)}
Average Detections/Image: {len(detections_list)/len(images):.1f}

Activity Zones:
• High Activity: {np.sum(heatmap_density > 0.7)/(heatmap_size[0]*heatmap_size[1])*100:.1f}%
• Medium Activity: {np.sum((heatmap_density > 0.3) & (heatmap_density <= 0.7))/(heatmap_size[0]*heatmap_size[1])*100:.1f}%
• Low Activity: {np.sum(heatmap_density <= 0.3)/(heatmap_size[0]*heatmap_size[1])*100:.1f}%

Peak Activity Location:
X: {np.unravel_index(heatmap_density.argmax(), heatmap_density.shape)[1]} px
Y: {np.unravel_index(heatmap_density.argmax(), heatmap_density.shape)[0]} px
"""

ax2.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.axis('off')
ax2.set_title('Analysis', fontsize=14, pad=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_with_stats.png', dpi=150, bbox_inches='tight')
print("  ✓ Stats heatmap saved")
plt.close()

print(f"\n✅ All heatmaps created in: {OUTPUT_DIR}/")
