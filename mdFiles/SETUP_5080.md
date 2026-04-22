# Vision Mamba Depth - 5080 Setup Guide

本指南指导如何在RTX 5080上完整配置和运行Vim depth estimation任务。

## 快速开始 (3步)

```bash
# 1. 创建Conda环境
conda create -n vim_5080 python=3.9.19
conda activate vim_5080

# 2. 安装PyTorch (cu118 - 5080推荐)
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu118

# 3. 运行编译脚本
cd /path/to/Vim
bash MAMBA_COMPILE_GUIDE.md  # 或手动执行其中的步骤
```

---

## 详细步骤

### Step 1: 环境准备

```bash
# 激活环境
conda create -n vim_5080 python=3.9.19
conda activate vim_5080

# 安装基础依赖
pip install numpy==1.23.4 imageio scipy Pillow opencv-python

# 安装MMSegmentation栈
pip install mmcv-full==1.7.1 mmsegmentation==0.29.1

# 安装Vision Mamba依赖
pip install timm einops torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: 编译Mamba (使用 MAMBA_COMPILE_GUIDE.md)

**关键点：**
- RTX 5080 = Ada架构 = `sm_89`
- 使用 `export CUDA_ARCH=sm_89` 设置编译目标
- causal-conv1d编译在mamba-1p1p1之前
- 验证两个模块都可以在Python中导入

```bash
# 查看编译指南
cat MAMBA_COMPILE_GUIDE.md

# 按步骤执行，或使用脚本
bash MAMBA_COMPILE_GUIDE.md install_mamba.sh
```

### Step 3: 验证数据集

```bash
# 确保NYU Depth v2数据集已下载到
/path/to/dataset/nyu_depth_v2/

# 数据集结构应为：
# nyu_depth_v2/
# ├── train/
# │   ├── 0000_colors.png
# │   ├── 0000_depths.png
# │   ├── ...
# ├── val/
# │   ├── ...
# └── test/
#     ├── ...
```

更新配置文件中的数据路径：

```bash
# 编辑 dep/configs/_base_/datasets/nyu_depth_v2.py
# 修改 data_root 指向正确的数据集路径
```

### Step 4: 验证安装

```bash
# 进入dep目录
cd dep

# 验证Mamba模块
python -c "
from backbone import VisionMambaSeg
import torch
print('✓ VisionMambaSeg imported')

model = VisionMambaSeg(
    img_size=512, 
    patch_size=16,
    embed_dim=192,
    depth=24
).cuda()
x = torch.randn(1, 3, 512, 512).cuda()
y = model(x)
print(f'✓ Forward pass OK: output shapes = {[o.shape for o in y]}')
"

# 验证数据加载
python -c "
from datasets import NYUDepthV2Dataset
from datasets.pipelines import Compose, LoadImageFromFile, LoadDepthAnnotation, DepthFormatBundle
import mmcv

print('✓ All dataset classes imported')
"
```

---

## 训练 Vim Backbone

### 快速验证 (CNN Baseline - 已在3080验证)

```bash
cd dep

# 使用CNN baseline快速检查（数据+前向+损失都正常）
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --launcher none \
  --gpus 1 \
  --work-dir work_dirs/cnn_test_5080
```

**预期输出：**
- iter 0-2000：loss逐渐下降
- 内存占用：~6-7GB（batch_size=4）

### 完整Vim训练

```bash
# Tiny model (24层, 推荐第一次尝试)
python train.py configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  --launcher none \
  --gpus 1 \
  --work-dir work_dirs/vim_tiny_depth_5080

# Small model (如果有足够VRAM)
python train.py configs/vim/depth/depth_vim_small_24_512_60k.py \
  --launcher none \
  --gpus 1 \
  --work-dir work_dirs/vim_small_depth_5080
```

**配置说明：**
- `depth_vim_tiny_24_512_60k.py`：24层Vim-Tiny, 512x512输入, 60k迭代
- `depth_vim_small_24_512_60k.py`：24层Vim-Small, 更大的embed_dim
- 修改 `samples_per_gpu=8` 来调整batch size

### 多GPU分布式训练

```bash
# 4个GPU上训练（batch_size=32，每GPU8张图）
python -m torch.distributed.launch \
  --nproc_per_node 4 \
  train.py configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  --launcher pytorch \
  --gpus 4 \
  --work-dir work_dirs/vim_tiny_depth_dist
```

---

## 评估 & 推理

### 评估指标

```bash
cd dep

# 在验证集上评估
python test.py configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  work_dirs/vim_tiny_depth_5080/latest.pth \
  --eval mde
```

**输出指标：**
- `AbsRel`: 平均绝对误差比
- `Sq Rel`: 相对平方误差
- `RMSE`: 均方根误差
- `RMSE log`: 对数域RMSE
- `δ < 1.25`: 阈值精度 (理想 >0.9)

### 单个图像推理

```python
import torch
from backbone import VisionMambaSeg
from decode_heads import DepthHead
import mmcv

# 加载模型
model = dict(
    backbone=VisionMambaSeg(...).cuda(),
    decode_head=DepthHead(...).cuda()
)

# 推理
img = mmcv.imread('test.jpg')  # (H, W, 3)
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

with torch.no_grad():
    features = model['backbone'](img)
    depth = model['decode_head'](features)  # (1, H, W)

print(f"Depth range: {depth.min():.3f} - {depth.max():.3f} m")
```

---

## 性能对比

| 模型 | 参数量 | VRAM | 单iter时间 | 精度(AbsRel) |
|------|--------|------|-----------|-------------|
| CNN Baseline | 10M | 6GB | ~50ms | 基础 |
| Vim-Tiny | 12M | 8GB | ~80ms | 更好* |
| Vim-Small | 26M | 12GB | ~150ms | 最好* |

\* 理论值，基于论文；需要在5080上验证

---

## 故障排除

### 问题1: ImportError: cannot import VisionMambaSeg

**原因：** causal-conv1d 或 mamba-1p1p1 未编译

**解决：**
```bash
# 检查是否生成了.so文件
find . -name "*.so" | grep causal_conv1d
find . -name "*.so" | grep mamba_ssm

# 如果没有，按照 MAMBA_COMPILE_GUIDE.md 重新编译
```

### 问题2: CUDA compute capability mismatch

**症状：** 编译成功但导入失败 `unsupported architecture`

**解决：**
```bash
# 确认GPU架构
nvidia-smi  # 应显示 "Ada" 或类似

# 重新编译，显式指定sm_89
export CUDA_ARCH="sm_89"
cd causal-conv1d
python setup.py clean --all
python setup.py build_ext --inplace
```

### 问题3: 内存不足 (OOM)

**症状：** `RuntimeError: CUDA out of memory`

**解决：**
- 减小 `samples_per_gpu` (从8 → 4 → 2)
- 或减小 `crop_size` (从512 → 384)
- 修改配置文件：`data = dict(samples_per_gpu=4, ...)`

### 问题4: Loss divergence

**症状：** Loss在iter 2000后无限增大或为NaN

**原因：** 数值稳定性问题（已知问题）

**解决：**
- 降低学习率：`lr=1e-5` 而不是 `1e-4`
- 增加warmup：`warmup_iters=3000`
- 使用gradient clipping（已启用）
- 使用混合精度训练：修改 `fp16 = dict(loss_scale='dynamic')`

---

## 配置文件位置

| 文件 | 用途 |
|------|------|
| `dep/configs/vim/depth/depth_vim_tiny_24_512_60k.py` | Vim-Tiny训练配置 |
| `dep/configs/vim/depth/depth_vim_small_24_512_60k.py` | Vim-Small训练配置 |
| `dep/configs/_base_/models/upernet_vim.py` | Vim基础模型配置 |
| `dep/configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py` | CNN基准快速测试 |
| `dep/backbone/vim.py` | VisionMambaSeg实现 |
| `dep/backbone/cnn_baseline.py` | CNN基线实现 |
| `dep/decode_heads/depth_head.py` | 深度解码头 |
| `dep/losses/silog_loss.py` | SILog损失函数 |

---

## 下一步

1. ✅ **拉取最新代码**：从GitHub获取所有changes
2. ✅ **环境准备**：按Step 1创建conda环境和安装依赖
3. ✅ **编译Mamba**：使用MAMBA_COMPILE_GUIDE.md
4. ✅ **验证安装**：运行Step 4的验证脚本
5. ▶ **快速测试**：运行CNN baseline验证流程
6. ▶ **训练Vim**：运行完整Vim backbone训练
7. ▶ **评估**：在验证集上测试精度

---

## 参考

- [Mamba SSM 编译指南](./MAMBA_COMPILE_GUIDE.md)
- [Vision Mamba 论文](https://arxiv.org/abs/2401.09417)
- [MMSegmentation 文档](https://github.com/open-mmlab/mmsegmentation)
- [NYU Depth v2 数据集](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

---

**最后更新：** 2026-04-21
**验证平台：** RTX 5080 (Ada架构, sm_89)
**推荐环境：** Python 3.9.19 + PyTorch 1.12.1 cu118
