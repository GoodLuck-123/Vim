# Vision Mamba Depth Estimation - GPU Training Guide (容器部署版)

## 环境配置

### 容器环境已就绪

容器已按官方环境配置完整，无需额外安装：

```bash
# 容器预配置信息
- Python: 3.10.13
- PyTorch: 2.1.1 + cu118
- causal_conv1d: >= 1.1.0
- mamba: 1p1p1

# 进入容器后直接验证
python --version                           # 3.10.13
python -c "import torch; print(torch.__version__)"  # 2.1.1+cu118
python -c "import torch; print(torch.cuda.is_available())"  # True
```

### 快速验证容器内CUDA环境

```bash
# 检查GPU识别
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"  # RTX 5080

# 检查Mamba CUDA ops
python -c "from causal_conv1d import causal_conv1d_fn; print('✓ causal_conv1d ok')"
python -c "from mamba_ssm import Mamba; print('✓ mamba_ssm ok')"
```

## 快速启动（5分钟内开始训练）

### 步骤1：进入容器

```bash
# 假设你已经启动了容器
docker run --gpus all -it -v /home/dji/projects/Vim:/workspace your_image_name bash

# 进入代码目录
cd /workspace/dep
```

### 步骤2：运行训练（单GPU）

```bash
python train.py \
    configs/vim/depth/depth_vim_tiny_24_512_60k.py \
    --work-dir work_dirs/vimdep-tiny \
    --launcher none \
    --gpus 1
```

### 步骤3：监控训练

在另一个终端：
```bash
# 查看日志
tail -f work_dirs/vimdep-tiny/20260413_171711.log

# 或启动TensorBoard（容器内）
tensorboard --logdir work_dirs/vimdep-tiny --port 6006

# 本地浏览：http://localhost:6006
```

## 配置文件优化

### 验证GPU优化配置已启用

`configs/vim/depth/depth_vim_tiny_24_512_60k.py` 中应包含：

```python
model = dict(
    backbone=dict(
        rms_norm=True,           # ✅ GPU优化：RMSNorm加速
        fused_add_norm=True,     # ✅ GPU优化：融合LayerNorm+残差
        # ... 其他参数
    ),
)

fp16 = None  # 可选：启用 FP16 混合精度训练
```

### 数据集配置

确保 `configs/_base_/datasets/nyu_depth_v2.py` 中的路径正确：

```python
data_root = '/home/dji/projects/Vim/data'  # 容器内数据路径

data = dict(
    samples_per_gpu=8,      # RTX 5080: 推荐 8-12
    workers_per_gpu=4,      # 数据加载线程
    train=dict(...),
)
```

## 运行脚本

### 单GPU训练（快速启动）

```bash
cd /home/dji/projects/Vim/dep

python train.py \
    configs/vim/depth/depth_vim_tiny_24_512_60k.py \
    --work-dir work_dirs/vimdep-tiny \
    --launcher none \
    --gpus 1
```

### 使用提供的脚本

编辑 `scripts/ft_vim_tiny_single_gpu.sh`：

```bash
#!/bin/bash
set -x

cd /home/dji/projects/Vim/dep

python train.py \
  configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  --work-dir work_dirs/vimdep-tiny \
  --launcher none \
  --gpus 1 \
  --options data.samples_per_gpu=8
```

运行：
```bash
bash scripts/ft_vim_tiny_single_gpu.sh
```

### 多GPU分布式训练

创建 `scripts/ft_vim_tiny_multi_gpu.sh`：

```bash
#!/bin/bash
set -x

cd /home/dji/projects/Vim/dep

# 4 GPU 分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    configs/vim/depth/depth_vim_tiny_24_512_60k.py \
    --work-dir work_dirs/vimdep-tiny-multi \
    --launcher pytorch \
    --options data.samples_per_gpu=4
```

运行：
```bash
bash scripts/ft_vim_tiny_multi_gpu.sh
```

### 恢复训练（断点续训）

```bash
python train.py \
    configs/vim/depth/depth_vim_tiny_24_512_60k.py \
    --work-dir work_dirs/vimdep-tiny \
    --launcher none \
    --resume-from work_dirs/vimdep-tiny/latest.pth \
    --gpus 1
```

### 评估训练完的模型

```bash
python test.py \
    configs/vim/depth/depth_vim_tiny_24_512_60k.py \
    work_dirs/vimdep-tiny/best.pth \
    --work-dir work_dirs/vimdep-tiny-eval \
    --eval AbsRel RMSE delta_1 \
    --launcher none \
    --gpu-id 0
```

## 性能优化

运行:
```bash
bash scripts/ft_vim_tiny_multi_gpu.sh
```

### 恢复训练（断点续训）

```bash
python train.py \
    configs/vim/depth/depth_vim_tiny_24_512_60k.py \
    --work-dir work_dirs/vimdep-tiny \
    --launcher none \
    --resume-from work_dirs/vimdep-tiny/latest.pth \
    --gpus 1
```

### 评估训练完的模型

```bash
python test.py \
    configs/vim/depth/depth_vim_tiny_24_512_60k.py \
    work_dirs/vimdep-tiny/best.pth \
    --work-dir work_dirs/vimdep-tiny-eval \
    --eval AbsRel RMSE delta_1 \
    --launcher none \
    --gpu-id 0
```

## 性能优化

### 1. 混合精度训练 (FP16)

在config中启用:

```python
fp16 = dict(loss_scale=512.)

optimizer_config = dict(
    type='DistOptimizerHook',
    use_fp16=True,  # ✅ 启用FP16
)
```

**优势**: 
- 显存占用减少 ~50%
- 训练速度快 15-20%
- 精度基本无损

### 2. 梯度累积

用于处理大batch size限制:

```python
# 在config中设置
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=2,  # 每2步更新一次权重
)
```

这等效于 `batch_size * 2` 但显存占用不增加。

### 3. 调整batch size

对于RTX 5080 (16GB显存):

```python
# 推荐配置
data = dict(
    samples_per_gpu=8,      # per GPU
    workers_per_gpu=4,
)
# 总batch size = 8 * 4 GPUs = 32
```

如果OOM，降低到:
```python
samples_per_gpu=4,  # 使用梯度累积补偿
```

### 4. 学习率调整

更新learning rate scheduler:

```python
# Linear warmup + Poly decay
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# AdamW优化器
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.92)
)
```

## 常见问题解决

### Q1: CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案（RTX 5080 16GB）**:
```bash
# 方案A：降低batch size
python train.py ... --options data.samples_per_gpu=4

# 方案B：启用梯度累积
# 在config中设置 update_interval=2

# 方案C：启用FP16混合精度
# 在config中设置 fp16=dict(loss_scale=512.)
```

**推荐顺序**: 先试 samples_per_gpu=4 → 再启用FP16 → 最后梯度累积

### Q2: 训练速度慢

**症状**: GPU利用率低 (<50%)

**容器内检查**:
```bash
# 看GPU显存/占用率是否持续满载
nvidia-smi
# 或实时监控
watch -n 1 nvidia-smi

# 增加数据加载线程
python train.py ... --options data.workers_per_gpu=8

# 确保GPU优化已启用
# 检查 rms_norm=True, fused_add_norm=True 在config中
```

### Q3: 模型不收敛

**症状**: Loss不下降或剧烈震荡

**检查项**:
1. **学习率过大**: 试试降低到 `5e-5` 或 `1e-5`
2. **数据问题**: 检查深度值是否在 `[1e-3, 10.0]` 米范围内
3. **预热不足**: 增加 `warmup_iters` 从 1500 到 2000-3000
4. **Loss计算**: 查看日志是否有NaN或Inf

### Q4: NaN Loss

**症状**: Loss变成 NaN 或 Inf

**原因**: 
- 深度值有0或负数
- log运算中的数值不稳定
- 无效像素未正确处理

**容器内检查**:
```bash
# 检查数据加载
python -c "
from datasets import NYUDepthV2Dataset
ds = NYUDepthV2Dataset(data_root='/home/dji/projects/Vim/data', img_dir='nyu2_train')
sample = ds[0]
print(f'Depth range: {sample[\"gt_semantic_seg\"].min():.4f} - {sample[\"gt_semantic_seg\"].max():.4f}')
"
```

### Q5: 容器内找不到数据

**症状**: `FileNotFoundError: [Errno 2] No such file or directory`

**容器内检查**:
```bash
# 确认数据路径
ls /home/dji/projects/Vim/data/nyu2_train | head -5
ls /home/dji/projects/Vim/data/nyu2_train/bedroom_0020_out/ | head -5

# 确认配置中的数据根目录
grep "data_root" configs/_base_/datasets/nyu_depth_v2.py
```

如果路径不同，更新 `configs/_base_/datasets/nyu_depth_v2.py`:
```python
data_root = '/your/actual/data/path'
```

## 容器内监测训练

### 实时监控GPU

```bash
# 容器内实时显卡利用率
nvidia-smi

# 或持续监控
watch -n 1 nvidia-smi
```

### 查看训练日志

```bash
# 容器内查看日志
tail -f work_dirs/vimdep-tiny/20260413_171711.log

# 或查看最近100行
tail -100 work_dirs/vimdep-tiny/*.log
```

### 启动TensorBoard（可选）

```bash
# 容器内启动
tensorboard --logdir work_dirs/vimdep-tiny --port 6006

# 本地浏览：http://localhost:6006
# （需要容器有--port 6006:6006映射）
```

### 关键监控指标

训练中应注意：
- **Training Loss**: SILogLoss应逐步下降（初值通常 2-5）
- **GPU Memory**: 应接近满载 (~14-15GB for RTX 5080)
- **GPU Utilization**: 应 >90%
- **Data Loading**: 应 <5% 时间开销

## 数据准备

### NYU Depth v2 数据集

**当前路径**: `/home/dji/projects/Vim/data`

**结构**:
```
data/
├── nyu2_train/          # 284个场景
│   ├── bedroom_0001_out/
│   │   ├── 1.jpg       # RGB图像 (640x480)
│   │   ├── 1.png       # 深度图 (16-bit, 单位: mm)
│   │   ├── 2.jpg
│   │   ├── 2.png
│   │   └── ...
│   └── ...
└── nyu2_test/           # 1308个场景
    └── ...
```

**容器内验证数据**:
```bash
# 检查训练集
ls /home/dji/projects/Vim/data/nyu2_train | wc -l      # 应该是 284

# 检查样本
ls /home/dji/projects/Vim/data/nyu2_train/bedroom_0020_out/

# 检查图像和深度对
file /home/dji/projects/Vim/data/nyu2_train/bedroom_0020_out/1.jpg
file /home/dji/projects/Vim/data/nyu2_train/bedroom_0020_out/1.png
```

**数据格式转换（如需要）**:
```python
# 如果数据格式不同，使用以下转换
import imageio
import numpy as np

# Mat格式 → PNG
depth_mat = scipy.io.loadmat('depth.mat')['depths']  # (480, 640)
depth_mm = (depth_mat * 1000).astype(np.uint16)      # 转为mm
imageio.imwrite('depth.png', depth_mm)
```

## 容器内快速检查清单

首次在RTX 5080容器中训练前：

- [ ] 容器内环境验证：`python -c "import torch; print(torch.cuda.is_available())"`  ✅ True
- [ ] Mamba CUDA ops可用：`python -c "from causal_conv1d import causal_conv1d_fn; print('OK')"`
- [ ] 数据集路径存在：`ls /home/dji/projects/Vim/data/nyu2_train`
- [ ] 配置参数检查：`rms_norm=True` 和 `fused_add_norm=True` 已启用
- [ ] Batch size设置：`samples_per_gpu=8` (或根据显存调整)
- [ ] 工作目录可写：`mkdir -p work_dirs/vimdep-tiny`

## 训练时间估计

基于RTX 5080 (16GB显存)：

| 模型 | 配置 | 显存 | 60k iters时间 |
|------|------|------|-------------|
| Tiny | 192-dim, 24-layer | ~14GB | ~4-6小时 |
| Small | 384-dim, 24-layer | OOM | 需降batch |

**推荐**: 从 **Tiny** 开始验证训练流程

## 模型保存位置

训练完后的检查点在：

```bash
work_dirs/vimdep-tiny/
├── latest.pth              # 最新检查点 (自动覆盖)
├── best.pth                # 最佳检查点 (最小AbsRel)
├── {timestamp}.log         # 文本日志
├── {timestamp}.log.json    # JSON格式日志
└── {iter_num}.pth          # 定期保存的检查点
```

**导出最佳模型**:
```bash
# 容器内
cp work_dirs/vimdep-tiny/best.pth best_model.pth

# 复制到本地（如需要）
# 在容器外：docker cp container_id:/path/to/best_model.pth ./
```

## 超参数微调建议

如果训练效果不理想，按顺序尝试：

### 1. Loss不下降

```bash
# 降低学习率
python train.py ... --options optimizer.lr=5e-5

# 增加预热步数
python train.py ... --options lr_config.warmup_iters=3000
```

### 2. 显存不足

```bash
# 降低batch size
python train.py ... --options data.samples_per_gpu=4

# 启用FP16（在config中）
fp16 = dict(loss_scale=512.)
```

### 3. 训练速度慢

```bash
# 增加数据加载线程
python train.py ... --options data.workers_per_gpu=8

# 确保GPU优化启用
# 检查 rms_norm=True, fused_add_norm=True
```

## 保存和部署

### 在容器内评估训练完的模型

```bash
python test.py \
    configs/vim/depth/depth_vim_tiny_24_512_60k.py \
    work_dirs/vimdep-tiny/best.pth \
    --work-dir work_dirs/vimdep-tiny-eval \
    --eval AbsRel RMSE delta_1 \
    --launcher none \
    --gpu-id 0
```

### 转换为推理格式（可选）

```python
import torch
from mmseg.models import build_segmentor
from mmcv import Config

cfg = Config.fromfile('configs/vim/depth/depth_vim_tiny_24_512_60k.py')
model = build_segmentor(cfg.model)
checkpoint = torch.load('work_dirs/vimdep-tiny/best.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 保存为TorchScript
scripted = torch.jit.script(model)
scripted.save('model.pt')
```

## 参考资源

- [MMSegmentation Docs](https://mmsegmentation.readthedocs.io/)
- [Vision Mamba Paper](https://arxiv.org/abs/2401.09417)
- [NYU Depth v2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
