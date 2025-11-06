# pet-detection-computer-vision
machine vision experiment that detects pets in photos using YOLOv8 and OpenCV. The model identifies dogs and cats in real-world images and draws bounding boxes around them, showing how object detection can bring AI “eyes” to everyday scenes.

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
│  └─ 01_exploration.ipynb
├─ src/
│  ├─ infer_on_image.py
│  ├─ infer_on_folder.py
│  └─ utils.py
├─ outputs/
│  ├─ detections/
│  └─ logs/
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Gitflow CI/CD
- Branch model: develop is the integration branch, main contains production-ready code, and feature/*, release/*, hotfix/* branches inherit from the appropriate base following the Gitflow convention.
- `.github/workflows/gitflow-ci.yml` runs on pushes to all Gitflow branches and on pull requests targeting main or develop.
- The pipeline checks out the repo, sets up Python 3.11, installs any dependencies listed in `requirements.txt`, compiles sources for early syntax validation, and runs pytest if a `tests/` directory exists.
- Successful pushes to main, release/*, or hotfix/* additionally package the repository into `dist/pet-detection-yolov8.tar.gz` and upload it as a GitHub Actions artifact for distribution or deployment.
