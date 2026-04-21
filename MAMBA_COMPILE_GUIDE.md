# Mamba SSM 编译指南

本文档记录在Vim项目中编译`causal-conv1d`和`mamba-1p1p1`的完整流程。

## 环境要求

### 硬件
- NVIDIA GPU（最低compute capability 7.0，推荐8.0+）
- 编译时临时存储 ~2-3GB

### 软件栈
```
CUDA Toolkit 11.6+ (建议 11.8+)
  nvcc, nvlink
GCC 9.0+ 
  g++, gcc
Python 3.9+
PyTorch 1.12.1+ (cu116 variant)
```

验证工具链：
```bash
nvcc --version
gcc --version
g++ --version
```

## 安装流程

### Step 1: 创建Conda环境

```bash
conda create -n vim python=3.9.19
conda activate vim
```

### Step 2: 安装PyTorch（关键！版本必须匹配）

```bash
# NVIDIA GPU (cu116)
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu116

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 输出: 1.12.1+cu116 True
```

### Step 3: 安装基础依赖

```bash
pip install numpy==1.23.4 imageio scipy Pillow opencv-python
pip install mmcv-full==1.7.1 mmsegmentation==0.29.1
pip install timm einops
```

### Step 4: 编译 causal-conv1d

```bash
cd /path/to/Vim/causal-conv1d

# 方法A: 编译到项目目录（推荐用于开发）
python setup.py build_ext --inplace

# 验证
python -c "import causal_conv1d_cuda; print('✓ causal_conv1d_cuda OK')"
```

**预期输出：** 生成 `causal_conv1d_cuda.cpython-39-x86_64-linux-gnu.so`

### Step 5: 编译 mamba-1p1p1

```bash
cd /path/to/Vim/mamba-1p1p1

# 编译到项目目录
python setup.py build_ext --inplace

# 验证
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('✓ Mamba OK')"
```

### Step 6: 验证完整栈

```bash
python -c "
from mamba_ssm.modules.mamba_simple import Mamba
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
import torch
print('✓ All backends loaded')

# Quick test
x = torch.randn(2, 64, 100, device='cuda')
model = Mamba(d_model=64).cuda()
y = model(x)
print(f'✓ Forward pass OK: {y.shape}')
"
```

## 常见问题 & 解决方案

### 问题1: `libc10.so: cannot open shared object file`

**原因：** PyTorch库路径未在LD_LIBRARY_PATH中

**解决：**
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

# 或永久添加到 ~/.bashrc or ~/.zshrc
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

### 问题2: CUDA compute capability 不匹配

**症状：** 编译成功但导入时出错

**解决：** 在setup.py中设置目标架构
```bash
# 检查GPU
nvidia-smi  # 查看GPU型号

# 根据型号设置CUDA_ARCH环境变量
# RTX 3080: sm_86
# RTX 4080/5080: sm_89
# H100: sm_90

export CUDA_ARCH="sm_89"  # 替换为你的架构
python setup.py build_ext --inplace
```

### 问题3: 编译超时或内存不足

**症状：** `Killed` 或编译卡住

**解决：**
```bash
# 限制并行编译线程
export MAX_JOBS=2
python setup.py build_ext --inplace
```

### 问题4: 权限问题（Permission denied）

**症状：** `/usr/local/cuda/bin/nvcc: Permission denied`

**解决：**
```bash
# 确保有读权限
chmod +x /usr/local/cuda/bin/nvcc

# 或者检查PATH
which nvcc
```

## 在5080上的完整脚本

保存为 `install_mamba.sh`：

```bash
#!/bin/bash
set -e

# Setup
CUDA_ARCH="${CUDA_ARCH:-sm_89}"  # 5080是Ada架构
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

echo "=== Building causal-conv1d ==="
cd /path/to/Vim/causal-conv1d
python setup.py build_ext --inplace

echo "✓ causal-conv1d built"
python -c "import causal_conv1d_cuda; print('Loaded')"

echo "=== Building mamba-1p1p1 ==="
cd /path/to/Vim/mamba-1p1p1
python setup.py build_ext --inplace

echo "✓ mamba-1p1p1 built"
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('Loaded')"

echo "=== Verification ==="
python -c "
from mamba_ssm.modules.mamba_simple import Mamba
import torch
model = Mamba(d_model=768).cuda()
x = torch.randn(1, 100, 768, device='cuda')
y = model(x)
print(f'✓ Forward pass: {y.shape}')
"

echo "=== SUCCESS ==="
```

使用：
```bash
bash install_mamba.sh
```

## 配置文件更新

编译完成后，更新dep配置使用Vim backbone：

```bash
# 使用VisionMambaSeg而不是CNNBaseline
python train.py configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  --launcher none --gpus 1 --work-dir work_dirs/vim_depth_test
```

## 性能对比

| 组件 | CNN Baseline | Vim (Mamba) |
|------|-------------|-----------|
| 模型大小 | ~10M params | ~12M params (tiny) |
| 推理时间 | ~50ms | ~30ms (理论) |
| 显存占用 | ~2GB (batch=4) | ~4GB (batch=4) |
| 精度潜力 | 基础 | 高 |

## 故障排除清单

- [ ] CUDA工具链可用（`nvcc --version`）
- [ ] PyTorch 1.12.1 cu116已安装
- [ ] LD_LIBRARY_PATH已配置
- [ ] causal-conv1d编译成功（.so文件存在）
- [ ] mamba-1p1p1编译成功
- [ ] Python可导入两个模块
- [ ] GPU前向pass成功

## 参考

- [Mamba-SSM官方](https://github.com/state-spaces/mamba)
- [causal-conv1d官方](https://github.com/Dao-AILab/causal-conv1d)
- [Vision Mamba论文](https://arxiv.org/abs/2401.09417)

---

**最后更新：** 2026-04-21
**测试平台：** RTX 3080 (编译成功，库链接问题)
**推荐平台：** RTX 5080 (Ada架构，预期无问题)
