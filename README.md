# Retina CycleGAN: Fundus ↔ OCT Translation

CycleGAN implementation for translating between fundus and OCT retinal images in diabetic retinopathy cases.

## Quick Start

### Installation
```bash
pip install torch torchvision pandas pillow
mkdir -p data/fundus/trainA data/oct/trainB outputs
```

### Data Preparation
```bash
# Filter DR-positive images from dataset
python filter_oct_and_fundus.py

# Or prepare from zip archives
python prepare_fundus_data.py --raw_dir /path/to/zips --train_dir data/fundus/trainA --val_dir data/fundus/val
```

### Training
```bash
# Basic training
python train.py --dataA data/fundus/trainA --dataB data/oct/trainB --epochs 30

# Enhanced training (with perceptual loss)
python "import os, pandas as pd.py" --dataA data/fundus/trainA --dataB data/oct/trainB --epochs 30
```

## Architecture

- **Generators**: ResNet-based encoder-decoder with 6 residual blocks
- **Discriminators**: PatchGAN for texture realism
- **Losses**: Adversarial + Cycle Consistency + Identity + Perceptual (VGG)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 8 | Training batch size |
| `--img_size` | 128 | Image resolution |
| `--epochs` | 30 | Training epochs |
| `--lambda_cycle` | 10.0 | Cycle loss weight |

## Files

- `filter_oct_and_fundus.py` - Filter DR-positive images from dataset
- `prepare_fundus_data.py` - Extract archives and split data
- `train.py` - Standard CycleGAN training
- `import os, pandas as pd.py` - Enhanced training with perceptual loss

## Output

Training generates sample images (`epoch{N}_sample.jpg`) and checkpoints (`checkpoint_{N}.pth`) in the `outputs/` directory.

## Resume Training
```bash
python train.py --resume outputs/checkpoint_15.pth
```


*Currently working to Imporve it*

## Clinical Use

Trained specifically on diabetic retinopathy (NPDR/PDR) cases for cross-modal image synthesis and research. Not for clinical diagnosis.
