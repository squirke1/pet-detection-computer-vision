# Training Pipeline - Quick Start Guide

## 📋 Prerequisites

- Labeled dataset in YOLO format
- GPU with CUDA support (recommended)
- Python 3.11+ with dependencies installed

## 🗂️ Dataset Structure

Your dataset should follow this structure:

```
data/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/ (optional)
    ├── images/
    └── labels/
```

### Label Format (YOLO)

Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
```

All values are normalized (0-1):
- `class_id`: Integer (0 for dog, 1 for cat)
- `x_center, y_center`: Center of bounding box
- `width, height`: Box dimensions

Example `img001.txt`:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

## 🛠️ Data Preparation

### 1. Split Existing Dataset

If you have all images in one folder:

```bash
python src/prepare_data.py split \
    --input path/to/all_images \
    --output data/raw \
    --train 0.8 \
    --val 0.1 \
    --test 0.1
```

### 2. Validate Dataset

Check your dataset for issues:

```bash
python src/prepare_data.py validate \
    --data-dir data/raw \
    --class-names dog cat
```

### 3. Visualize Samples

Preview annotations:

```bash
python src/prepare_data.py visualize \
    --image data/raw/train/images/sample.jpg \
    --label data/raw/train/labels/sample.txt \
    --class-names dog cat \
    --output visualization.jpg
```

## 🏋️ Training

### Quick Start Training

```bash
python src/train.py \
    --data-yaml config/data.yaml \
    --epochs 100 \
    --batch-size 16
```

### Training with Custom Model

```bash
# Small model (fast, good for testing)
python src/train.py --model yolov8n.pt --data-yaml config/data.yaml --epochs 100

# Medium model (balanced)
python src/train.py --model yolov8m.pt --data-yaml config/data.yaml --epochs 100

# Large model (best accuracy)
python src/train.py --model yolov8l.pt --data-yaml config/data.yaml --epochs 100
```

### Advanced Training Options

```bash
python src/train.py \
    --data-yaml config/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --device cuda \
    --workers 8 \
    --lr0 0.01 \
    --optimizer Adam \
    --patience 50 \
    --cache  # Cache images in RAM for faster training
```

### Resume Training

If training was interrupted:

```bash
python src/train.py --resume models/training/train_20260205_120000/weights/last.pt
```

## 📊 Monitoring Training

Training progress is automatically logged to:
```
models/training/train_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Latest checkpoint
├── results.png          # Training metrics plot
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png         # F1 curve
├── P_curve.png          # Precision curve
├── R_curve.png          # Recall curve
└── args.yaml            # Training configuration
```

### View Training Metrics

YOLOv8 automatically creates visualization plots in the training directory.

## 🎯 Hyperparameter Tuning

### Learning Rate

```bash
# Lower learning rate for fine-tuning
python src/train.py --data-yaml config/data.yaml --lr0 0.001

# Higher learning rate for faster convergence
python src/train.py --data-yaml config/data.yaml --lr0 0.01
```

### Data Augmentation

```bash
python src/train.py \
    --data-yaml config/data.yaml \
    --hsv-h 0.02 \      # Color jitter
    --hsv-s 0.7 \
    --hsv-v 0.4 \
    --degrees 10.0 \    # Rotation
    --translate 0.2 \   # Translation
    --scale 0.5 \       # Scale
    --fliplr 0.5 \      # Horizontal flip
    --mosaic 1.0 \      # Mosaic augmentation
    --mixup 0.1         # Mixup augmentation
```

### Image Size

```bash
# Faster training, lower accuracy
python src/train.py --data-yaml config/data.yaml --imgsz 320 --batch-size 32

# Balanced (default)
python src/train.py --data-yaml config/data.yaml --imgsz 640 --batch-size 16

# Better accuracy, slower training
python src/train.py --data-yaml config/data.yaml --imgsz 1024 --batch-size 8
```

## 🚀 Using Trained Model

After training, use your model for inference:

```bash
# Single image
python src/infer_on_image.py \
    --image test.jpg \
    --model models/training/train_20260205_120000/weights/best.pt

# Batch processing
python src/infer_on_folder.py \
    --input data/test/images \
    --output outputs/predictions \
    --model models/training/train_20260205_120000/weights/best.pt

# Video processing
python src/infer_on_video.py \
    video.mp4 \
    --model models/training/train_20260205_120000/weights/best.pt \
    --output-video output.mp4
```

## 💡 Tips

### For Small Datasets (<1000 images)

- Use pretrained model and fine-tune
- Increase augmentation
- Use smaller model (yolov8n)
- Lower batch size
- More epochs (200+)

```bash
python src/train.py \
    --model yolov8n.pt \
    --data-yaml config/data.yaml \
    --epochs 200 \
    --batch-size 8 \
    --mosaic 1.0 \
    --mixup 0.15
```

### For Large Datasets (>10000 images)

- Can train from scratch
- Use larger model (yolov8m or yolov8l)
- Higher batch size
- Less augmentation

```bash
python src/train.py \
    --model yolov8m.pt \
    --data-yaml config/data.yaml \
    --epochs 100 \
    --batch-size 32 \
    --cache
```

### For Limited GPU Memory

- Reduce batch size
- Reduce image size
- Use smaller model
- Disable caching

```bash
python src/train.py \
    --model yolov8n.pt \
    --data-yaml config/data.yaml \
    --batch-size 8 \
    --imgsz 416
```

## 📈 Expected Results

Typical training performance on pet detection:

| Model | Size | Params | mAP50 | Speed (ms) |
|-------|------|--------|-------|------------|
| YOLOv8n | 6.2MB | 3.2M | 88-92% | 2-3 |
| YOLOv8s | 22MB | 11.2M | 91-94% | 3-4 |
| YOLOv8m | 52MB | 25.9M | 93-96% | 5-7 |
| YOLOv8l | 87MB | 43.7M | 94-97% | 8-10 |

*mAP50 = Mean Average Precision at IoU=0.5*

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 4

# Reduce image size
--imgsz 320

# Use smaller model
--model yolov8n.pt
```

### Training Not Converging

- Check dataset labels are correct
- Reduce learning rate: `--lr0 0.001`
- Increase warmup: `--warmup-epochs 5.0`
- Adjust augmentation

### Overfitting

- Increase augmentation
- Add dropout: `--dropout 0.1`
- Reduce model complexity
- Get more training data

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Data Augmentation Guide](https://docs.ultralytics.com/modes/train/#augmentation-settings)
- [Hyperparameter Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
