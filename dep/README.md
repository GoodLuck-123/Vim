# Vision Mamba - Dense Depth Estimation (dep)

## Overview

This directory contains an adaptation of Vision Mamba for **monocular dense depth estimation** on NYU Depth v2 dataset. It's modified from the semantic segmentation (`seg/`) implementation to predict pixel-wise continuous depth maps instead of semantic classes.

## Key Changes from `seg/` to `dep/`

| Component | seg/ (Segmentation) | dep/ (Depth Estimation) |
|-----------|-------------------|------------------------|
| **Output** | Multi-class logits (150 classes for ADE20K) | Single-channel depth (1 channel) |
| **Loss** | CrossEntropyLoss | SILogLoss (scale-invariant log loss) |
| **Metrics** | mIoU, mAcc, mDice | AbsRel, RMSE, δ<1.25, δ<1.25² |
| **Head** | UPerHead | DepthHead (modified from UPerHead) |
| **Dataset** | ADE20K (semantic labels) | NYU Depth v2 (depth maps) |
| **Depth Range** | N/A | [0.001, 10.0] meters |
| **Backbone** | VisionMambaSeg | VisionMambaSeg (unchanged) |

## Installation

```bash
# Activate environment
source /home/dji/miniforge3/bin/activate vim

# Install core dependencies
pip install -r dep-requirements.txt

# (Optional) Fix yapf compatibility for config dumping
pip install --upgrade yapf==0.32.0
```

## Dataset Setup

Prepare NYU Depth v2 in this structure:
```
/path/to/nyu_depth_v2/
├── images/
│   ├── training/      (795 RGB images, 480×640)
│   └── validation/    (654 RGB images, 480×640)
└── depth/
    ├── training/      (16-bit PNG depth maps, mm scale)
    └── validation/    (16-bit PNG depth maps, mm scale)
```

Update `configs/_base_/datasets/nyu_depth_v2.py`:
```python
data_root = '/path/to/nyu_depth_v2'  # Set your actual path
```

## Quick Start

### Training (4 GPUs)
```bash
bash scripts/ft_vim_tiny_dep.sh
```

### Single GPU Training
```bash
cd /home/dji/projects/Vim/dep
python train.py configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  --work-dir work_dirs/vimdep-t \
  --gpus 1
```

### Evaluation
```bash
bash scripts/eval_vim_tiny_dep.sh work_dirs/vimdep-t/latest.pth
```

## Architecture

**Model Pipeline:**
```
RGB Input (B, 3, 512, 512)
    ↓
VisionMambaSeg Backbone
    ├─ Layer 5: (B, 192, 32, 32)
    ├─ Layer 11: (B, 192, 32, 32)
    ├─ Layer 17: (B, 192, 32, 32)
    └─ Layer 23: (B, 192, 32, 32)
    ↓ [FPN Upsampling to 4 scales]
    ↓
DepthHead (Neck + Decoder)
    ├─ PPM pooling (1, 2, 3, 6)
    ├─ Feature fusion
    └─ 1×1 Conv → (B, 1, 512, 512)
    ↓
Sigmoid + Range Scaling
    ↓
Depth Map (B, 1, 512, 512) ∈ [0.001, 10.0]
```

## Custom Modules

### 1. DepthHead (`decode_heads/depth_head.py`)
- Single-channel decoder based on UPerHead
- Outputs raw logits → sigmoid activation → range mapping
- Compatible with mmseg's EncoderDecoder model

### 2. SILogLoss & BerHuLoss (`losses/silog_loss.py`)
- **SILogLoss**: Standard depth loss, scale-invariant in log space
- **BerHuLoss**: Reverse Huber loss, robust for outliers

### 3. NYUDepthV2Dataset (`datasets/nyu_depth_v2.py`)
- Dataset class with standard depth metrics
- AbsRel, RMSE, log-RMSE, δ threshold evaluation

### 4. LoadDepthAnnotation (`datasets/pipelines/depth_loading.py`)
- Pipeline transform for loading 16-bit PNG depth (mm → m)
- Stores in `gt_semantic_seg` key for compatibility

## Configs

**Model Config**: `configs/_base_/models/upernet_vim.py`
- Backbone: VisionMambaSeg (embed_dim=384, depth=12)
- Head: DepthHead (min_depth=1e-3, max_depth=10.0)
- Loss: SILogLoss (variance_focus=0.85)

**Experiment Config**: `configs/vim/depth/depth_vim_tiny_24_512_60k.py`
- Optimizer: AdamW (lr=1e-4, layer decay=0.92)
- Scheduler: Poly with linear warmup (1500 iters)
- Iterations: 60k
- Batch: 8 per GPU
- Input: 512×512

## Evaluation Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **AbsRel** | Absolute relative error: mean(\|pred-gt\|/gt) | ↓ |
| **SqRel** | Squared relative error | ↓ |
| **RMSE** | Root mean squared error | ↓ |
| **RMSElog** | RMSE in log space | ↓ |
| **δ<1.25** | % predictions with max(pred/gt, gt/pred)<1.25 | ↑ |
| **δ<1.25²** | % predictions with max(pred/gt, gt/pred)<1.5625 | ↑ |
| **δ<1.25³** | % predictions with max(pred/gt, gt/pred)<1.953 | ↑ |

## Notes

- **Framework**: Fully compatible with mmsegmentation v0.29.1
- **Data Keys**: Depth stored in `gt_semantic_seg` for EncoderDecoder compatibility
- **GPU**: Distributed training ready (DDP), supports FP16
- **Batch Size**: 8 images/GPU (60k iterations ≈ 7500 epochs on NYU train set)

## Files Modified/Created

**New Files (8):**
- `decode_heads/depth_head.py`, `decode_heads/__init__.py`
- `losses/silog_loss.py`, `losses/__init__.py`
- `datasets/nyu_depth_v2.py`, `datasets/__init__.py`
- `datasets/pipelines/depth_loading.py`, `datasets/pipelines/__init__.py`

**Modified Files (9):**
- `train.py`, `test.py` - Added custom module registration
- `configs/_base_/models/upernet_vim.py` - DepthHead + SILogLoss
- `configs/_base_/datasets/nyu_depth_v2.py` - NYU Depth v2 config
- `configs/_base_/schedules/schedule_60k.py` - Depth metrics
- `configs/vim/depth/*.py` - Experiment configs
- `scripts/*_dep.sh` - Training/eval scripts

## References

- NYU Depth v2 Dataset: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- SILog Loss: Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network" (NIPS 2014)
- Vision Mamba: https://github.com/hustvl/Vim
- mmsegmentation: https://github.com/open-mmlab/mmsegmentation

## Related Directories

- `../seg/` - Original semantic segmentation implementation
- `../det/` - Object detection implementation
- `../backbone/` - Vision Mamba backbone (shared)
- `../mmcv_custom/` - Custom mmseg utilities (shared)
