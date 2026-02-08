# Pet Detection Computer Vision

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine vision project that combines **YOLOv8** object detection with **OpenCV** computer vision techniques for enhanced pet detection in images. The system identifies dogs and cats with bounding boxes while providing additional visual analysis through edge detection, keypoint extraction, and contour analysis.

> 🎯 **Purpose**: Demonstrate the integration of modern deep learning (YOLO) with traditional computer vision methods for robust object detection and analysis.

---computer-vision/
├── data/                          # Data directory
│   ├── raw/                       # Raw input images
│   ├── processed/                 # Preprocessed data
│   └── annotations/               # Annotation files
│
├── models/                        # Model files
│   └── yolov8n-pets.pt           # YOLOv8 trained model
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploration.ipynb      # YOLO exploration
│   ├── 02_edge_detection_features.ipynb  # CV features
│   └── 03_yolo_edge_integration.ipynb    # Integration demo
│
├── src/                           # Source code
│   ├── infer_on_image.py         # Single image inference
│   ├── infer_on_folder.py        # Batch processing
│   ├── infer_on_video.py         # Video & webcam processing
│   ├── infer_enhanced.py         # Enhanced inference with CV features
│   └── utils.py                  # Utility functions
│
├── tests/                         # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── conftest.py               # Pytest configuration
│
├── outputs/                       # Output directory
│   ├── detections/               # Detection results
│   └── logs/                     # Execution logs
│
├── .github/                       # GitHub Actions
│   └── workflows/
│       └── gitflow-ci.yml        # CI/CD pipeline
│
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
├── README.md                     # This file
├── LICENSE                       # MIT License
└── DIAGRAMS.md                   # Architecture diagrams
```

---

## 🖼️ Examples

### Detection Output

**Input Image** → **Standard Detection** → **Enhanced Detection**

The enhanced detection mode overlays:
- **Cyan edges**: Canny edge detection highlighting object boundaries
- **Yellow circles**: ORB keypoints showing distinctive features
- **Blue/Green boxes**: Pet detections (dogs=blue, cats=green) with confidence scores

### Sample Results

```
🐾 Detected 2 pet(s):
  1. DOG (confidence: 0.92) at [145, 234, 456, 678]
  2. CAT (confidence: 0.87) at [512, 123, 789, 456]
```

---

## 🛠️ Development

### Setting Up Development Environment

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install black flake8 mypy  # Code quality tools
   ```

2. Run tests:
   ```bash
   pytest tests/ -v --cov=src
   ```

3. Format code:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

### Project Workflow (Gitflow)

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/\***: Feature development branches
- **release/\***: Release preparation branches
- **hotfix/\***: Emergency fixes for production

**Creating a feature:**
```bash
git checkout develop
git checkout -b feature/my-new-feature
# Make changes, commit
git checkout develop
git merge --no-ff feature/my-new-feature
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suite
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v
```

### Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Test Requirements

- Model file (`models/yolov8n-pets.pt`) is optional for most tests
- Integration tests will be skipped if model is not available
- Test images are generated synthetically when needed

---

## 🔄allation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/squirke1/pet-detection-computer-vision.git
   cd pet-detection-computer-vision
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 model**
   ```bash
   # The model will be automatically downloaded on first run
   # Or manually place your trained model in models/yolov8n-pets.pt
   ```

### Requirements

- `opencv-python>=4.8.0` - Computer vision operations
- `numpy>=1.24.0` - Numerical computing
- `ultralytics>=8.0.0` - YOLOv8 framework
- `matplotlib>=3.7.0` - Visualization
- `pillow>=10.0.0` - Image processing

See [requirements.txt](requirements.txt) for complete list.

---

## 🎯 Quick Start

Detect pets in an image with one command:

```bash
python src/infer_on_image.py --image data/raw/my_dogs.jpg
```

Output will be saved to `outputs/detections/output_my_dogs.jpg`

For enhanced detection with edge features:

```bash
python src/infer_enhanced.py --image data/raw/my_dogs.jpg --show-edges --show-keypoints
```

---

## 📖 Usage

### Basic Detection

Detect pets in a single image:

```bash
python src/infer_on_image.py --image path/to/image.jpg
```

**Options:**
- `--model MODEL_PATH` - Path to YOLOv8 model (default: `models/yolov8n-pets.pt`)
- `--output OUTPUT_PATH` - Output image path
- `--conf THRESHOLD` - Confidence threshold (default: 0.25)
- `--show` - Display result in a window

**Example:**
```bash
python src/infer_on_image.py \
    --image data/raw/my_dogs.jpg \
    --conf 0.5 \
    --output results/my_detection.jpg \
    --show
```

### Enhanced Detection

Enhanced inference with computer vision features:

```bash
python src/infer_enhanced.py --image path/to/image.jpg [OPTIONS]
```

**Options:**
- `--show-edges` - Overlay edge detection (cyan overlay)
- `--show-keypoints` - Display ORB keypoints (yellow circles)
- `--analyze-features` - Print detailed feature analysis

**Examples:**

```bash
# Edge detection overlay
python src/infer_enhanced.py --image data/raw/my_dogs.jpg --show-edges

# Full feature visualization
python src/infer_enhanced.py \
    --image data/raw/my_dogs.jpg \
    --show-edges \
    --show-keypoints \
    --analyze-features

# Custom output path
python src/infer_enhanced.py \
    --image data/raw/my_dogs.jpg \
    --show-edges \
    --output custom_output.jpg
```

**Feature Analysis Output Example:**
```
============================================================
FEATURE ANALYSIS
============================================================

🔍 Edge Detection:
  • Canny edges detected: 12543 pixels
  • Sobel edges detected: 15287 pixels
  • Laplacian edges detected: 11893 pixels

📍 Keypoint Detection:
  • SIFT keypoints: 487
  • ORB keypoints: 500

🔲 Contour Detection:
  • Large contours found: 23 (area > 500px²)
  • Largest contour area: 45672px²
  • Average contour area: 3421px²
============================================================
```

### Batch Processing

Process all images in a folder:

```bash
python src/infer_on_folder.py --folder data/raw/ --output outputs/batch/
```

### Video Processing

Process video files or webcam streams:

```bash
# Process video file
python src/infer_on_video.py --video input.mp4 --output result.mp4

# Real-time webcam
python src/infer_on_video.py --webcam --display

# Skip frames for better performance
python src/infer_on_video.py --video input.mp4 --skip-frames 2
```

### REST API & Web Interface

Run the detection API server with web UI:

```bash
# Start API server
python -m uvicorn api.main:app --reload

# Or with Docker
docker-compose up

# Access web UI
open http://localhost:8000
```

**API Endpoints:**
- `POST /detect` - Upload image for pet detection
- `GET /health` - Health check and model status
- `GET /models` - List available models
- `GET /docs` - Interactive API documentation

See [API.md](API.md) for complete API documentation and examples.

### Model Training

Train or fine-tune a YOLOv8 model on your custom dataset:

```bash
# Prepare dataset (split train/val/test)
python src/prepare_data.py split --input data/raw --output data/processed

# Validate dataset
python src/prepare_data.py validate --data config/data.yaml

# Train model
python src/train.py --data config/data.yaml --epochs 100 --batch 16

# Resume training
python src/train.py --data config/data.yaml --resume runs/train/exp/weights/last.pt
```

See [TRAINING.md](TRAINING.md) for comprehensive training guide.

### Model Evaluation

Evaluate model performance with comprehensive metrics:

```bash
# Basic evaluation
python src/evaluate.py --model models/yolov8n-pets.pt --data config/data.yaml

# With visualizations and reports
python src/evaluate.py --model models/yolov8n-pets.pt --data config/data.yaml \
    --save-plots --save-txt --save-json

# Compare multiple models
python src/evaluate.py --compare models/model1.pt models/model2.pt models/model3.pt \
    --data config/data.yaml --save-plots

# Speed benchmark
python src/evaluate.py --model models/yolov8n-pets.pt --benchmark \
    --benchmark-samples 200
```

**Evaluation Metrics:**
- mAP@0.5 and mAP@0.5:0.95
- Precision, Recall, F1-Score
- Per-class performance
- Confusion matrix
- Inference speed (FPS)

### Visualization Tools

Visualize training progress and results:

```bash
# Plot training history
python src/visualize_metrics.py training --results runs/train/exp/results.csv \
    --output outputs/training_history.png

# Plot per-class performance
python src/visualize_metrics.py class-performance --metrics outputs/evaluation/metrics.json \
    --output outputs/class_performance.png
```

---

## 📁 Project Structure
```
pet-detection-computer-vision/
├─ config/
│  └─ data.yaml                  # Dataset configuration
├─ data/
│  ├─ raw/                       # Raw images
│  ├─ processed/                 # Processed datasets
│  └─ annotations/               # Annotation files
├─ models/
│  └─ yolov8n-pets.pt           # Trained model weights
├─ notebooks/
│  ├─ 01_exploration.ipynb
│  ├─ 02_edge_detection_features.ipynb
│  └─ 03_yolo_edge_integration.ipynb
├─ src/
│  ├─ infer_on_image.py          # Single image inference
│  ├─ infer_on_folder.py         # Batch processing (multiprocessing)
│  ├─ infer_on_video.py          # Video & webcam processing
│  ├─ infer_enhanced.py          # Enhanced inference with CV features
│  ├─ train.py                   # Model training pipeline
│  ├─ prepare_data.py            # Dataset preparation utilities
│  ├─ evaluate.py                # Model evaluation & metrics
│  ├─ visualize_metrics.py       # Visualization tools
│  └─ utils.py                   # Utility functions
├─ api/
│  ├─ main.py                    # FastAPI application
│  ├─ models.py                  # Pydantic request/response models
│  └─ inference.py               # Inference engine wrapper
├─ examples/
│  └─ api_client.py              # Python client examples
├─ outputs/
│  ├─ detections/                # Detection results
│  ├─ evaluation/                # Evaluation reports & plots
│  └─ logs/                      # Execution logs
├─ documentation/
│  └─ project-roadmap.md         # Development roadmap
├─ Dockerfile                    # Docker image definition
├─ docker-compose.yml            # Docker Compose configuration
├─ .dockerignore                 # Docker build exclusions
├─ requirements.txt              # Python dependencies
├─ api-requirements.txt          # API-specific dependencies
├─ README.md
├─ API.md                        # API documentation
├─ TRAINING.md                   # Training documentation
└─ LICENSE
```

---

## 🔄 CI/CD

### Automated Pipeline

The project uses **GitHub Actions** with a Gitflow-based CI/CD pipeline:

**Pipeline Stages:**

1. **Code Checkout** - Fetch repository code
2. **Environment Setup** - Python 3.11 installation
3. **Dependency Installation** - Install requirements
4. **Linting** - Code style validation
5. **Testing** - Run pytest suite
6. **Coverage** - Generate test coverage reports
7. **Artifact Build** - Package application (main branch only)
8. **Deployment** - Upload artifacts for distribution

**Trigger Conditions:**
- **feature/\*** - Linting + Tests
- **develop** - Full test suite + Coverage
- **main** - Full pipeline + Artifact creation + Release
- **Pull Requests** - All tests must pass

**Configuration:** See [.github/workflows/gitflow-ci.yml](.github/workflows/gitflow-ci.yml)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request to `develop` branch

### Contribution Guidelines

- Follow [PEP 8](https://pep8.org/) style guide
- Write tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR
- Use descriptive commit messages

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License
Copyright (c) 2025 Stephen Quirke
```

---

## 🙏 Acknowledgments

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** - YOLOv8 implementation
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **COCO Dataset** - Pre-trained model foundation
- **Open Source Community** - Various tools and libraries

---

## 📞 Contact & Support

- **Author**: Stephen Quirke
- **Repository**: [github.com/squirke1/pet-detection-computer-vision](https://github.com/squirke1/pet-detection-computer-vision)
- **Issues**: [Report a bug](https://github.com/squirke1/pet-detection-computer-vision/issues)

For questions or discussions, please open an issue on GitHub.

---

## 🗺️ Roadmap

See [documentation/project-roadmap.md](documentation/project-roadmap.md) for detailed development plans.

**Completed Features (v2.1):**
- ✅ Video processing and real-time detection
- ✅ Model training and fine-tuning utilities
- ✅ Model evaluation with comprehensive metrics
- ✅ Batch processing optimization (multiprocessing)
- ✅ Enhanced CV features integration
- ✅ REST API with FastAPI
- ✅ Docker containerization
- ✅ Web UI for image upload

**Upcoming Features:**
- 🔍 Multi-object tracking across frames (DeepSORT/ByteTrack)
- 📈 Advanced analytics and heatmaps
- 🔐 API authentication and rate limiting
- ☁️ Cloud deployment guides (AWS, GCP, Azure)

---

<p align="center">Made with ❤️ for the computer vision community</p>
