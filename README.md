# pet-detection-computer-vision
machine vision experiment that detects pets in photos using YOLOv8 and OpenCV. The model identifies dogs and cats in real-world images and draws bounding boxes around them, showing how object detection can bring AI “eyes” to everyday scenes.
## Features
- 🐕 **YOLOv8 Pet Detection**: Fast and accurate detection of dogs and cats
- 🔍 **Edge Detection**: Canny, Sobel, and Laplacian edge detection
- 📍 **Keypoint Detection**: SIFT and ORB feature extraction
- 🔲 **Contour Analysis**: Boundary detection and shape analysis
- 🎨 **Enhanced Visualization**: Overlay edge detection and keypoints on YOLO results
## Project structure
```
pet-detection-yolov8/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ annotations/
├─ models/
│  └─ yolov8n-pets.pt
├─ notebooks/
│  ├─ 01_exploration.ipynb
│  ├─ 02_edge_detection_features.ipynb
│  └─ 03_yolo_edge_integration.ipynb
├─ src/
│  ├─ infer_on_image.py
│  ├─ infer_on_folder.py
│  ├─ infer_enhanced.py          # NEW: Enhanced inference with CV features
│  └─ utils.py
├─ outputs/
│  ├─ detections/
│  └─ logs/
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Usage

### Basic Detection
```bash
python src/infer_on_image.py --image data/raw/my_dogs.jpg
```

### Enhanced Detection with Edge Features
```bash
# With edge detection overlay
python src/infer_enhanced.py --image data/raw/my_dogs.jpg --show-edges

# With keypoint visualization
python src/infer_enhanced.py --image data/raw/my_dogs.jpg --show-keypoints

# With complete feature analysis
python src/infer_enhanced.py --image data/raw/my_dogs.jpg --show-edges --show-keypoints --analyze-features
```

## Gitflow CI/CD
- Branch model: develop is the integration branch, main contains production-ready code, and feature/*, release/*, hotfix/* branches inherit from the appropriate base following the Gitflow convention.
- `.github/workflows/gitflow-ci.yml` runs on pushes to all Gitflow branches and on pull requests targeting main or develop.
- The pipeline checks out the repo, sets up Python 3.11, installs any dependencies listed in `requirements.txt`, compiles sources for early syntax validation, and runs pytest if a `tests/` directory exists.
- Successful pushes to main, release/*, or hotfix/* additionally package the repository into `dist/pet-detection-yolov8.tar.gz` and upload it as a GitHub Actions artifact for distribution or deployment.
