# CNN Backbone - 对比实验计划

**Date:** 2026-04-21  
**Location:** RTX 5080  
**Goal:** 识别并修复loss divergence问题

---

## 📝 实验设计

### 3个对比实验

#### Experiment 1: Original Config (Baseline)
```bash
Config: depth_cnn_tiny_512_60k_full.py
名称: CNN-Original
特点: 完全保持原始配置
参数:
  - lr=1e-4, warmup=1500, warmup_ratio=1e-6
  - loss scale=10.0
  - grad_clip=None

运行:
cd dep
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --launcher none --gpus 1 \
  --work-dir work_dirs/exp1_original \
  --seed 42

预期结果: iter 2000后loss发散 (与3080一致)
```

#### Experiment 2: Loss Stable Only
```bash
Config: depth_cnn_tiny_512_60k_full.py (+ 修改的loss)
名称: CNN-StableLoss
特点: 仅改进loss函数的数值稳定性

改进:
  - loss scale: 10.0 → 1.0
  - 添加epsilon项防止sqrt(negative)
  
其他保持不变:
  - lr=1e-4 (原始)
  - warmup=1500 (原始)
  - grad_clip=None (原始)

运行:
# 先确保loss已修改
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --launcher none --gpus 1 \
  --work-dir work_dirs/exp2_stable_loss \
  --seed 42

预期结果: Loss应该更稳定，但可能在iter 3000+发散
```

#### Experiment 3: Full Improvement Stack
```bash
Config: depth_cnn_tiny_512_60k_stable.py (新配置)
名称: CNN-FullStable
特点: 组合所有改进

改进:
  - loss scale: 10.0 → 1.0 ✓
  - lr: 1e-4 → 5e-5 ✓
  - warmup: 1500 → 5000 ✓
  - warmup_ratio: 1e-6 → 0.1 ✓
  - grad_clip: None → max_norm=1.0 ✓
  - total_iters: 1000 → 60000 ✓

运行:
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py \
  --launcher none --gpus 1 \
  --work-dir work_dirs/exp3_full_stable \
  --seed 42

预期结果: Loss应该平滑下降到60k iterations而不发散
```

---

## 🎯 执行流程 (4小时)

### Step 1: 检查环境 (5 min)
```bash
# 确保在正确的环境
conda activate vim_5080
cd /home/dji/projects/Vim

# 验证imports
python -c "
from dep.losses import SILogLoss
from dep.backbone import CNNBaseline
print('✓ Imports OK')
"
```

### Step 2: 准备数据 (1 min)
```bash
# 检查NYU数据集位置
ls dep/configs/_base_/datasets/nyu_depth_v2.py

# 确保数据路径正确（如果需要更新）
# vim dep/configs/_base_/datasets/nyu_depth_v2.py
```

### Step 3: 运行对比实验 (并行)

**Terminal 1 - Exp1 (Original)**
```bash
cd /home/dji/projects/Vim/dep
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --launcher none --gpus 1 \
  --work-dir work_dirs/exp1_original \
  --seed 42 2>&1 | tee exp1.log
```

等待它跑到iter 2000左右...

**在另一个Terminal监控：**
```bash
# 实时查看loss
tail -f work_dirs/exp1_original/run.log | grep "loss_depth"

# 或用TensorBoard
tensorboard --logdir work_dirs/exp1_original/
```

**预期日志输出：**
```
iter 100: loss_depth 2.354
iter 500: loss_depth 1.234
iter 1000: loss_depth 0.832
iter 1500: loss_depth 0.756
iter 2000: loss_depth nan  ← HERE: Loss diverges
```

### Step 4: 运行Exp2和Exp3

**一旦Exp1失败或达到iter 2000，启动Exp2：**
```bash
cd /home/dji/projects/Vim/dep
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --launcher none --gpus 1 \
  --work-dir work_dirs/exp2_stable_loss \
  --seed 42 2>&1 | tee exp2.log
```

**同时启动Exp3：**
```bash
cd /home/dji/projects/Vim/dep
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py \
  --launcher none --gpus 1 \
  --work-dir work_dirs/exp3_full_stable \
  --seed 42 2>&1 | tee exp3.log
```

---

## 📊 监控与分析

### Real-time Monitoring

**监控Loss曲线：**
```bash
# 使用TensorBoard (推荐)
tensorboard --logdir work_dirs/ --port 6006

# 或手动查看logs
for dir in work_dirs/exp*/; do
  echo "=== $(basename $dir) ==="
  tail -20 $dir/run.log | grep "loss_depth"
done
```

### Checkpoints

三个实验都会自动保存checkpoints：
```bash
work_dirs/
├── exp1_original/
│   ├── run.log
│   ├── latest.pth
│   └── iter_*.pth
├── exp2_stable_loss/
│   ├── run.log
│   ├── latest.pth
│   └── iter_*.pth
└── exp3_full_stable/
    ├── run.log
    ├── latest.pth
    └── iter_*.pth
```

---

## 📈 结果分析框架

### 对比指标

创建分析脚本 `analyze_experiments.py`：

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

def extract_losses(log_file):
    """Extract loss values from log file"""
    losses = []
    iterations = []
    
    with open(log_file) as f:
        for line in f:
            if 'loss_depth' in line:
                # Parse line like: "iter 100: loss_depth 2.354"
                parts = line.split()
                try:
                    iter_num = int(parts[1].rstrip(':'))
                    loss_val = float(parts[-1])
                    iterations.append(iter_num)
                    losses.append(loss_val)
                except:
                    pass
    
    return iterations, losses

# Compare experiments
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (exp_dir, title) in enumerate([
    ('exp1_original', 'Original Config'),
    ('exp2_stable_loss', 'Loss Stable Only'),
    ('exp3_full_stable', 'Full Stable Stack'),
]):
    log_file = f'work_dirs/{exp_dir}/run.log'
    
    if Path(log_file).exists():
        iters, losses = extract_losses(log_file)
        axes[idx].plot(iters, losses, 'b-', linewidth=2)
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Loss')
        axes[idx].grid(True, alpha=0.3)
        
        # Mark where loss diverges
        for i, loss in enumerate(losses):
            if loss > 100 or loss != loss:  # NaN check
                axes[idx].axvline(iters[i], color='r', linestyle='--', label='Divergence')
                break

plt.tight_layout()
plt.savefig('experiment_comparison.png')
print("Saved: experiment_comparison.png")
```

**运行分析：**
```bash
python analyze_experiments.py
```

### 关键观察点

| 指标 | Exp1-Original | Exp2-Loss | Exp3-Full |
|------|---|---|---|
| Loss @ iter 100 | ~2.3 | ~2.3 | ~2.3 |
| Loss @ iter 1000 | ~0.8 | ~0.9 | ~1.0 |
| Loss @ iter 2000 | NaN/Inf | ? | ? |
| Loss @ iter 5000 | - | ? | ? |
| Divergence @ iter | ~2000 | ? | 不发散? |

---

## ✅ 成功标志

### 如果改进有效

```
✅ Exp1: Loss @ iter 2000 = NaN (确认问题存在)
✅ Exp2: Loss @ iter 3000+ = 比Exp1稳定
✅ Exp3: Loss @ iter 60000 = 持续下降，无NaN
```

### 如果改进无效

```
❌ Exp2 & Exp3 的loss仍然在iter 2000-3000发散
   → 问题可能是：
      1. 数据异常（某些batch特别难）
      2. 模型架构问题
      3. 需要更激进的改进
```

---

## 🔧 如果仍然失败，尝试这些

### Option A: 更激进的学习率降低
```python
# 改为5e-6而不是5e-5
optimizer = dict(lr=5e-6, ...)
```

### Option B: 替换Loss函数
```python
# 使用BerHuLoss代替SILogLoss
loss_decode=dict(type='BerHuLoss', threshold=0.2)
```

### Option C: 检查数据异常
```bash
# 可视化一些样本
python -c "
from dep.datasets import NYUDepthV2Dataset
dataset = NYUDepthV2Dataset(...)
for i in range(10):
    sample = dataset[i]
    depth = sample['gt_semantic_seg']
    print(f'Sample {i}: min={depth.min()}, max={depth.max()}, mean={depth.mean()}')
    # 检查是否有异常值
"
```

### Option D: 改进模型架构
```python
# 增加特征维度
embed_dim=80  # 从64→80
depths=(3, 4, 6, 3)  # 增加blocks
```

---

## 📝 记录模板

创建文件 `EXPERIMENT_RESULTS.md` 来记录结果：

```markdown
# Experiment Results - 2026-04-21

## Exp1: Original Config
- Start time: HH:MM
- End time / Failure: HH:MM (iter XXXX)
- Loss @ iter 100: X.XXX
- Loss @ iter 1000: X.XXX
- Loss @ iter 2000: X.XXX (or NaN)
- Observation: ...

## Exp2: Stable Loss Only
- Start time: HH:MM
- Status: Running / Complete / Failed
- Key observation: ...

## Exp3: Full Stable Stack
- Start time: HH:MM
- Status: Running / Complete / Failed
- Final loss @ iter 60k: X.XXX
- Key observation: ...

## Conclusion
Which experiment succeeded? Which improvements were most effective?
```

---

## 🚀 预计时间表

| Phase | Duration | Tasks |
|-------|----------|-------|
| Setup | 10 min | 验证环境、准备数据 |
| Exp1 | 30-60 min | 原始配置（预期iter 2000失败） |
| Exp2+3 | 2-4 hour | 改进配置（并行运行）|
| 分析 | 15 min | 生成图表、总结结论 |
| **总计** | **3-5 hours** | **一个实验周期** |

---

## ✨ 最终目标

完成这个实验后，你将得到：
1. ✅ 明确的loss divergence根本原因
2. ✅ 有效的改进方案（优先级排序）
3. ✅ 稳定的CNN训练pipeline
4. ✅ 可应用于Vim backbone的改进

然后可以信心满满地向Vim backbone应用相同改进！

---

**Ready?** 开始Exp1吧！🚀

