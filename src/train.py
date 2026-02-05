#!/usr/bin/env python3
"""
Training script for YOLOv8 pet detection model.

Features:
- Custom dataset training
- Data augmentation
- Training monitoring with callbacks
- Checkpoint management
- Validation metrics
- Resume training support
- Hyperparameter tuning

Usage:
    # Basic training
    python train.py --data-yaml config/data.yaml --epochs 100
    
    # Resume from checkpoint
    python train.py --resume runs/train/exp/weights/last.pt
    
    # Fine-tune pretrained model
    python train.py --model yolov8n.pt --data-yaml config/data.yaml --epochs 50
"""

import argparse
from pathlib import Path
import sys
import yaml
from datetime import datetime
from typing import Optional, Dict

from ultralytics.models.yolo import YOLO


def validate_data_yaml(data_yaml_path: Path) -> Dict:
    """
    Validate data.yaml configuration file.
    
    Args:
        data_yaml_path: Path to data.yaml file
        
    Returns:
        Dictionary with validated configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not data_yaml_path.exists():
        raise ValueError(f"Data configuration not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in data.yaml: {field}")
    
    # Validate paths exist
    for path_key in ['train', 'val']:
        if path_key in config:
            path = Path(config[path_key])
            if not path.is_absolute():
                # Make path relative to data.yaml location
                path = data_yaml_path.parent / path
            if not path.exists():
                raise ValueError(f"{path_key} path does not exist: {path}")
    
    # Validate class count
    if len(config['names']) != config['nc']:
        raise ValueError(f"Number of classes ({config['nc']}) doesn't match names list length ({len(config['names'])})")
    
    print(f"✅ Data configuration validated:")
    print(f"   Classes: {config['nc']} ({', '.join(config['names'])})")
    print(f"   Train: {config['train']}")
    print(f"   Val: {config['val']}")
    
    return config


def print_training_header(config: 'TrainingConfig', save_dir: Path):
    """
    Print training configuration header.
    
    Args:
        config: Training configuration
        save_dir: Directory to save outputs
    """
    print("\n" + "=" * 60)
    print("🚀 Training Configuration")
    print("=" * 60)
    print(f"Dataset: {config.data_yaml}")
    print(f"Model: {config.model}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Image size: {config.imgsz}")
    print(f"Device: {config.device or 'auto'}")
    print(f"Workers: {config.workers}")
    print(f"Output: {save_dir}")
    print("=" * 60 + "\n")


class TrainingConfig:
    """Training configuration and hyperparameters."""
    
    def __init__(
        self,
        data_yaml: str,
        model: str = 'yolov8n.pt',
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        device: Optional[str] = None,
        workers: int = 8,
        project: str = '../models/training',
        name: Optional[str] = None,
        exist_ok: bool = False,
        pretrained: bool = True,
        optimizer: str = 'auto',
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: float = 3.0,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        flipud: float = 0.0,
        fliplr: float = 0.5,
        mosaic: float = 1.0,
        mixup: float = 0.0,
        copy_paste: float = 0.0,
        patience: int = 50,
        save: bool = True,
        save_period: int = -1,
        cache: bool = False,
        verbose: bool = True,
        seed: int = 0,
        deterministic: bool = True,
        single_cls: bool = False,
        resume: bool = False,
        amp: bool = True,
        fraction: float = 1.0,
        profile: bool = False,
        overlap_mask: bool = True,
        mask_ratio: int = 4,
        dropout: float = 0.0,
        val: bool = True
    ):
        """
        Initialize training configuration.
        
        Args:
            data_yaml: Path to data.yaml configuration
            model: Model to train (yolov8n.pt, yolov8s.pt, etc.)
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Input image size
            device: Device to train on (cuda, cpu, or None for auto)
            workers: Number of dataloader workers
            project: Project directory for training runs
            name: Run name (default: auto-generated)
            exist_ok: Whether to overwrite existing project/name
            pretrained: Use pretrained weights
            optimizer: Optimizer (auto, SGD, Adam, AdamW)
            lr0: Initial learning rate
            lrf: Final learning rate (lr0 * lrf)
            momentum: SGD momentum
            weight_decay: Weight decay
            warmup_epochs: Warmup epochs
            warmup_momentum: Warmup momentum
            warmup_bias_lr: Warmup bias learning rate
            box: Box loss gain
            cls: Class loss gain
            dfl: DFL loss gain
            hsv_h: HSV hue augmentation
            hsv_s: HSV saturation augmentation
            hsv_v: HSV value augmentation
            degrees: Rotation augmentation degrees
            translate: Translation augmentation
            scale: Scale augmentation
            shear: Shear augmentation
            perspective: Perspective augmentation
            flipud: Vertical flip probability
            fliplr: Horizontal flip probability
            mosaic: Mosaic augmentation probability
            mixup: Mixup augmentation probability
            copy_paste: Copy-paste augmentation probability
            patience: Early stopping patience
            save: Save checkpoints
            save_period: Save checkpoint every x epochs
            cache: Cache images (ram or disk)
            verbose: Verbose logging
            seed: Random seed
            deterministic: Deterministic training
            single_cls: Train as single-class dataset
            resume: Resume training from checkpoint
            amp: Automatic mixed precision
            fraction: Dataset fraction to use
            profile: Profile training
            overlap_mask: Overlap mask for segmentation
            mask_ratio: Mask downsample ratio
            dropout: Dropout rate
            val: Validate during training
        """
        self.data_yaml = data_yaml
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.device = device
        self.workers = workers
        self.project = project
        self.name = name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exist_ok = exist_ok
        self.pretrained = pretrained
        self.optimizer = optimizer
        self.lr0 = lr0
        self.lrf = lrf
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.box = box
        self.cls = cls
        self.dfl = dfl
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.fliplr = fliplr
        self.mosaic = mosaic
        self.mixup = mixup
        self.copy_paste = copy_paste
        self.patience = patience
        self.save = save
        self.save_period = save_period
        self.cache = cache
        self.verbose = verbose
        self.seed = seed
        self.deterministic = deterministic
        self.single_cls = single_cls
        self.resume = resume
        self.amp = amp
        self.fraction = fraction
        self.profile = profile
        self.overlap_mask = overlap_mask
        self.mask_ratio = mask_ratio
        self.dropout = dropout
        self.val = val
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for YOLO training."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def train_model(config: TrainingConfig) -> Path:
    """
    Train YOLOv8 model with given configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Path to best model weights
    """
    # Validate data configuration
    data_config = validate_data_yaml(Path(config.data_yaml))
    
    # Setup output directory
    save_dir = Path(config.project) / config.name
    
    # Load model
    print(f"\n📦 Loading model: {config.model}")
    model = YOLO(config.model)
    
    # Print model info
    print(f"   Model type: {model.model_name if hasattr(model, 'model_name') else 'YOLOv8'}")
    print(f"   Pretrained: {config.pretrained}")
    
    # Print training header
    print_training_header(config, save_dir)
    
    # Train
    results = model.train(**config.to_dict())
    
    # Get best model path
    best_model_path = save_dir / 'weights' / 'best.pt'
    
    if best_model_path.exists():
        print(f"\n🎯 Best model saved: {best_model_path}")
        
        # Validate best model
        print(f"\n📊 Validating best model...")
        metrics = model.val()
        
        print(f"\n✅ Validation Metrics:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
    
    return best_model_path


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='🐕 Pet Detection Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch with custom dataset
  python train.py --data-yaml config/pets.yaml --epochs 100
  
  # Fine-tune pretrained model
  python train.py --model yolov8n.pt --data-yaml config/pets.yaml --epochs 50
  
  # Train with larger model and higher resolution
  python train.py --model yolov8m.pt --data-yaml config/pets.yaml --imgsz 1024 --batch-size 8
  
  # Resume training from checkpoint
  python train.py --resume runs/train/exp/weights/last.pt
  
  # Fast training for testing
  python train.py --data-yaml config/pets.yaml --epochs 10 --imgsz 320 --batch-size 32
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data-yaml',
        type=str,
        help='Path to data.yaml configuration file'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Model to train (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint path'
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda, cpu, or None for auto)')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    
    # Output configuration
    parser.add_argument('--project', type=str, default='../models/training', help='Project directory')
    parser.add_argument('--name', type=str, default=None, help='Run name')
    parser.add_argument('--exist-ok', action='store_true', help='Overwrite existing project/name')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='auto', help='Optimizer (auto, SGD, Adam, AdamW)')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    
    # Augmentation parameters
    parser.add_argument('--hsv-h', type=float, default=0.015, help='HSV hue augmentation')
    parser.add_argument('--hsv-s', type=float, default=0.7, help='HSV saturation augmentation')
    parser.add_argument('--hsv-v', type=float, default=0.4, help='HSV value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotation degrees')
    parser.add_argument('--translate', type=float, default=0.1, help='Translation fraction')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale fraction')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Horizontal flip probability')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup augmentation probability')
    
    # Advanced options
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')
    parser.add_argument('--fraction', type=float, default=1.0, help='Dataset fraction to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    
    # Handle resume training
    if args.resume:
        print(f"🔄 Resuming training from: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
        print("\n✅ Training resumed and completed!")
        return
    
    # Validate required arguments
    if not args.data_yaml:
        parser.error("--data-yaml is required for new training")
    
    # Create training configuration
    config = TrainingConfig(
        data_yaml=args.data_yaml,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        patience=args.patience,
        cache=args.cache,
        fraction=args.fraction,
        seed=args.seed
    )
    
    try:
        # Train model
        best_model = train_model(config)
        
        print(f"\n{'=' * 60}")
        print("🎉 Training Complete!")
        print(f"{'=' * 60}")
        print(f"Best model: {best_model}")
        print(f"Training logs: {config.project}/{config.name}")
        print(f"{'=' * 60}\n")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
