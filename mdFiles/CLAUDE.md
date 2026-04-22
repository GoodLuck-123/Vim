# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Vision Mamba (Vim)** is a multi-task computer vision framework using bidirectional state space models (SSMs) as an alternative to Vision Transformers. The codebase is organized by vision task:

- **`vim/`**: ImageNet classification (pretraining / finetuning)
- **`det/`**: Object detection (Detectron2-based)
- **`seg/`**: Semantic segmentation (MMSegmentation-based)
- **`dep/`**: Dense depth estimation (NYU Depth v2)
- **`mamba-1p1p1/`**: Local Mamba SSM implementation
- **`causal-conv1d/`**: CUDA kernels for causal 1D convolution

## Environment Setup

Each task directory has independent Python version and dependency requirements:

```bash
# vim/ (ImageNet pretraining)
conda create -n vim python=3.10.13
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r vim/vim_requirements.txt
pip install -e causal-conv1d>=1.1.0
pip install -e mamba-1p1p1

# det/ (detection)
conda create -n vim_det python=3.9.19
cd det && pip install -e .
# Also install pycocotools, Shapely, and link COCO datasets

# seg/ / dep/ (segmentation / depth)
conda create -n vim_seg python=3.9.19
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116
pip install -U openmim && mim install mmcv-full==1.7.1
pip install mmsegmentation==0.29.1
```

For AMD GPUs: Use `rocm/pytorch` Docker image instead.

## Training & Evaluation

### ImageNet Classification (`vim/`)

**Pretraining:**
```bash
cd vim
bash scripts/pt-vim-t.sh  # Pretrain Vim-tiny
bash scripts/pt-vim-s.sh  # Pretrain Vim-small
```

**Finetuning:**
```bash
bash scripts/ft-vim-t.sh  # Finetune Vim-tiny
bash scripts/ft-vim-s.sh  # Finetune Vim-small
```

**Evaluation:**
```bash
python main.py --eval --resume /path/to/ckpt \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --data-path /path/to/imagenet
```

Key flags: `--model`, `--batch-size`, `--epochs`, `--lr`, `--drop-path`, `--data-path`, `--eval`

### Object Detection (`det/`)

```bash
cd det
bash scripts/ft_vim_tiny_vimdet.sh    # Train
bash scripts/eval_vim_tiny_vimdet.sh  # Evaluate
```

### Semantic Segmentation (`seg/`)

```bash
cd seg
bash scripts/ft_vim_tiny_upernet.sh   # Train
bash scripts/eval_vim_tiny_upernet.sh # Evaluate
```

Update `seg/configs/_base_/datasets/ade20k.py` with dataset path before training.

### Dense Depth Estimation (`dep/`)

```bash
cd dep
# Configure data_root in configs/_base_/datasets/nyu_depth_v2.py first
bash scripts/ft_vim_tiny_depth.sh     # Train
bash scripts/eval_vim_tiny_depth.sh   # Evaluate
```

Depth dataset: 16-bit PNG maps, normalized to [0.001, 10.0] meters.

## Code Architecture

### Model Definition

**vim/models_mamba.py**: Core architecture
- `PatchEmbed`: 2D image to patch embedding
- `Block`: Residual block with Mamba mixer and LayerNorm/RMSNorm
- Model classes: `VisionMamba`, `vim_tiny_patch16_224`, etc.
- Uses `timm` (pytorch-image-models) registry for model registration

**Key design**: Bidirectional Mamba blocks replace self-attention; position embeddings preserve spatial structure.

### Training Pipeline

**vim/main.py**: Training entry point (based on DeiT training script)
- Argument parsing via `get_args_parser()`
- Uses `timm` components: schedulers, optimizers, data augmentation, model EMA
- Distributed training support via `torch.distributed`
- MLflow integration for experiment logging

**vim/engine.py**: Training/evaluation loops
- `train_one_epoch()`: Standard supervised training
- `evaluate()`: Validation on test set

**vim/datasets.py**: Data loading
- ImageNet / custom dataset loaders

**vim/augment.py**: Data augmentation (RandAugment, Mixup, CutMix)

**vim/losses.py**: Loss functions
- `DistillationLoss`: Knowledge distillation support
- Cross-entropy with label smoothing

### Downstream Task Implementations

- **det/**: Detectron2 backbone integration
- **seg/**: MMSegmentation UPerHead with Vim backbone
- **dep/**: Custom depth head replacing classification head; uses SILog loss

## Common Development Tasks

### Adding a New Model Variant

1. Define architecture in `vim/models_mamba.py` (extend `VisionMamba` class)
2. Add model factory function with `@register_model` decorator
3. Update `__all__` export list
4. Create/update training script in `vim/scripts/`

### Modifying Training Loop

- Edit `vim/main.py` for high-level args / setup
- Edit `vim/engine.py` for epoch/batch logic
- Augmentation: modify `vim/augment.py`
- Losses: extend `vim/losses.py`

### Adapting to New Downstream Task

1. Copy relevant task dir (`seg/` → `custom_task/`)
2. Replace backbone: import Vim from `vim/models_mamba.py`
3. Define task-specific head (detection / segmentation / regression)
4. Update loss function and metrics
5. Configure dataset loading and augmentation

## Key Dependencies

- **torch 2.1.1** (1.12.1 for seg/dep): GPU tensor ops
- **timm 0.4.12**: Vision backbone registry, schedulers, augmentations
- **mamba-ssm** (local): State space model blocks
- **causal-conv1d** (local CUDA): Efficient 1D convolution
- **MMSegmentation / Detectron2**: Task-specific frameworks
- **einops**: Tensor rearrangement
- **transformers**: Tokenizers, config loading

## GPU Training Notes

- NVIDIA GPUs: torch cu118
- AMD GPUs: Use ROCm Docker (`rocm/pytorch:rocm6.2_...`)
- Batch size depends on model size; reduce if OOM
- Mixed precision training (fp16/bf16) supported via AMP
- Distributed training: Use `torch.distributed.launch` or `submitit`

## File Sizes & Checkpoints

Pretrained model weights available on HuggingFace:
- Vim-tiny: 7M params (~26MB checkpoint)
- Vim-small: 26M params (~98MB checkpoint)
- Vim-base: 98M params (~375MB checkpoint)

Downloaded weights cached in `~/.cache/huggingface/` by default.
