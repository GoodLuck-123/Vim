#!/bin/bash
# 5080 Quick Setup Script - 可复制粘贴执行
# Usage: bash SETUP_5080_QUICK.sh

set -e

echo "=== Vision Mamba Depth - 5080 Quick Setup ==="

# Step 1: Create environment
echo "Step 1: Creating conda environment..."
conda create -n vim_5080 python=3.9.19 -y
eval "$(conda shell.bash hook)"
conda activate vim_5080

# Step 2: Install PyTorch (cu118 for RTX 5080)
echo "Step 2: Installing PyTorch 1.12.1 cu118..."
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu118

# Step 3: Install dependencies
echo "Step 3: Installing base dependencies..."
pip install numpy==1.23.4 imageio scipy Pillow opencv-python

# Step 4: Install MMSegmentation stack
echo "Step 4: Installing MMSegmentation 0.29.1..."
pip install mmcv-full==1.7.1 mmsegmentation==0.29.1

# Step 5: Install Vision Mamba dependencies
echo "Step 5: Installing Vision Mamba dependencies..."
pip install timm einops

# Step 6: Setup LD_LIBRARY_PATH
echo "Step 6: Configuring LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

# Step 7: Compile causal-conv1d (with Ada architecture)
echo "Step 7: Compiling causal-conv1d (may take 5-10 minutes)..."
export CUDA_ARCH="sm_89"
export MAX_JOBS=2
cd causal-conv1d
python setup.py build_ext --inplace
cd ..
python -c "import causal_conv1d_cuda; print('✓ causal-conv1d compiled successfully')"

# Step 8: Compile mamba-1p1p1
echo "Step 8: Compiling mamba-1p1p1 (may take 5-10 minutes)..."
cd mamba-1p1p1
python setup.py build_ext --inplace
cd ..
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('✓ mamba-1p1p1 compiled successfully')"

# Step 9: Verify full stack
echo "Step 9: Verifying full stack..."
python -c "
from mamba_ssm.modules.mamba_simple import Mamba
from causal_conv1d import causal_conv1d_fn
import torch

x = torch.randn(2, 64, 100, device='cuda')
model = Mamba(d_model=64).cuda()
y = model(x)
print(f'✓ All backends loaded and tested: {y.shape}')
"

# Step 10: Verify dep module
echo "Step 10: Verifying depth module..."
cd dep
python -c "
from backbone import VisionMambaSeg
from decode_heads import DepthHead
print('✓ DepthHead and VisionMambaSeg imported')
"
cd ..

echo ""
echo "=== ✓ Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Update data_root in dep/configs/_base_/datasets/nyu_depth_v2.py"
echo "2. Test CNN baseline: cd dep && python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py --launcher none --gpus 1"
echo "3. Train Vim: python train.py configs/vim/depth/depth_vim_tiny_24_512_60k.py --launcher none --gpus 1"
echo ""
echo "For detailed guide, see: SETUP_5080.md"
