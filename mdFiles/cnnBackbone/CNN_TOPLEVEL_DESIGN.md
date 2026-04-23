# CNN Backbone - 顶层设计与改进方案

**Date:** 2026-04-21  
**Status:** 在5080上重新诊断 & 优化设计  
**Target:** 消除loss divergence，建立稳定训练pipeline

---

## 📊 问题诊断

### 当前状态 (3080验证)
```
✓ 数据加载：正常
✓ 前向传播：正常 (45ms/iter)
✓ Loss计算：正常
✓ Backward：正常
❌ 训练稳定性：iter 2000后loss发散
```

### Loss Divergence 的根本原因

从3080的实验来看，**不是代码bug**，而是**训练动力学问题**：

```
Stage 1 (iter 0-2000): Loss = 2.5 → 0.8 ✓ 正常下降
Stage 2 (iter 2000+): Loss = 0.8 → NaN ❌ 发散

可能的原因：
1. 学习率过高 (1e-4可能在后期太激进)
2. Warmup不足 (1500 iter可能太短)
3. 数据分布问题 (某些batch特别难)
4. 累积数值误差 (SILog Loss中log操作)
```

---

## 🎯 改进方案 (3个优先级)

### 🔴 Priority 1: 数值稳定性 (最可能有效)

**问题：** SILogLoss中的数值溢出

```python
# 当前实现 (潜在问题)
log_diff = torch.log(pred) - torch.log(target)
loss = sqrt(mean(d²) - λ*mean(d)²) * 10.0  # 乘以10可能放大误差
```

**改进方案：**

```python
# 方案A: 归一化loss
loss = sqrt(...) * 1.0  # 改为 * 1.0 而不是 * 10.0
# 原因：* 10可能导致梯度过大，尤其在iter后期

# 方案B: 添加数值稳定项
eps = 1e-8
loss = sqrt(
    (log_diff ** 2).mean() - λ * (log_diff.mean() ** 2) + eps
)
# 防止sqrt内部为负或极小

# 方案C: 用更稳定的BerHuLoss替代
# BerHu是分段线性，更数值稳定
```

**实施步骤：**
1. 创建 `silog_loss_stable.py` (改进版)
2. 在config中切换到新loss
3. 对比训练曲线

---

### 🟡 Priority 2: 学习率调度优化

**问题：** 固定lr或线性衰减可能不适合深度任务

```python
# 当前配置
optimizer = dict(lr=1e-4, ...)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,  # ← 只有1500 iter warmup
    warmup_ratio=1e-6,
    power=1.0,
)
```

**改进方案：**

```python
# 方案A: 增加warmup，降低峰值学习率
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=5000,        # ← 从1500→5000 (iter总数的8%)
    warmup_ratio=0.1,         # ← 从1e-6→0.1 (循序渐进)
    power=1.0,
)
optimizer = dict(lr=5e-5, ...)  # ← 从1e-4→5e-5 (降50%)

# 方案B: 添加学习率上界
optimizer_config = dict(
    type='DistOptimizerHook',
    grad_clip=dict(max_norm=1.0),  # ← 添加梯度裁剪
)

# 方案C: 使用Cosine衰减（更平滑）
lr_config = dict(
    policy='cosineanneal',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.1,
)
```

---

### 🟢 Priority 3: 数据和模型细调

**问题A：** 深度数据范围是否正确标准化

```python
# 检查项：
# 1. NYU数据集深度范围应该是 [0.001m, 10.0m]
# 2. 是否有异常值（太大或太小）
# 3. 是否有NaN或Inf

# 改进：添加数据验证
@PIPELINES.register_module()
class DepthValueClipping:
    """Clip depth to valid range [min_depth, max_depth]"""
    def __init__(self, min_depth=0.001, max_depth=10.0):
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def __call__(self, results):
        results['gt_semantic_seg'] = torch.clamp(
            results['gt_semantic_seg'],
            min=self.min_depth,
            max=self.max_depth
        )
        return results
```

**问题B：** CNN架构可能欠拟合

```python
# 当前：4-stage encoder [64, 128, 256, 384]
# 改进方向：

# 方案A: 增加特征维度
embed_dim=80  # 从64→80
depths=(3, 4, 6, 3)  # 增加blocks

# 方案B: 添加Skip connections
# 当前可能没有充分的信息流

# 方案C: 使用更强的norm (GroupNorm vs BatchNorm)
# BatchNorm对小batch size敏感
```

---

## 🔬 实施计划

### Phase 1: 稳定性验证 (Day 1)

**Step 1.1: 创建改进的Loss版本**
```bash
# 编辑 dep/losses/silog_loss.py
# 修改：去掉 * 10.0，改为 * 1.0
# 添加：epsilon项防止sqrt(负数)
```

**Step 1.2: 创建改进的Config**
```bash
# 创建 dep/configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py
# 改动：
# - lr: 1e-4 → 5e-5
# - warmup_iters: 1500 → 5000
# - warmup_ratio: 1e-6 → 0.1
# - 添加 grad_clip=1.0
```

**Step 1.3: 对比实验**
```bash
# 实验A: 原始配置 (baseline)
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --work-dir work_dirs/cnn_baseline_original

# 实验B: 稳定化Loss
# (修改loss文件后)
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --work-dir work_dirs/cnn_baseline_stable_loss

# 实验C: 改进学习率调度
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py \
  --work-dir work_dirs/cnn_baseline_stable_lr

# 实验D: 组合改进
# (同时应用C和Loss改进)
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py \
  --work-dir work_dirs/cnn_baseline_stable_all
```

**监控指标：**
```bash
# 每个实验都关注：
# 1. iter 100: loss 应该 ~2.0
# 2. iter 1000: loss 应该 ~0.8
# 3. iter 2000: loss 应该 继续下降 (不发散)
# 4. iter 5000: loss 是否稳定 (没有NaN)
```

### Phase 2: 长期训练验证 (Day 2-3)

一旦Phase 1找到稳定配置，进行完整60k iteration训练

```bash
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py \
  --launcher none --gpus 1 \
  --work-dir work_dirs/cnn_60k_final \
  --seed 42
```

**监控：**
- TensorBoard: `tensorboard --logdir work_dirs/`
- 每1000 iter保存checkpoint
- Loss曲线应该单调非增（允许小波动）

### Phase 3: 转移到Vim (Day 4+)

一旦CNN稳定，应用相同改进到Vim配置

```python
# dep/configs/vim/depth/depth_vim_tiny_24_512_60k.py
optimizer = dict(
    lr=5e-5,  # 应用相同学习率策略
    ...
)
lr_config = dict(
    warmup_iters=5000,
    warmup_ratio=0.1,
)
```

---

## 📋 改进检查清单

### Loss稳定性改进
- [ ] 移除SILogLoss中的`* 10.0`缩放
- [ ] 添加epsilon项到sqrt内部
- [ ] 考虑备选方案：用BerHuLoss替代SILogLoss
- [ ] 验证log操作的数值范围 (pred和target都应该>0)

### 学习率优化
- [ ] 创建新config: `depth_cnn_tiny_512_60k_stable.py`
- [ ] 设置 `lr=5e-5` (从1e-4→5e-5)
- [ ] 设置 `warmup_iters=5000` (从1500→5000)
- [ ] 设置 `warmup_ratio=0.1` (从1e-6→0.1)
- [ ] 添加 `grad_clip=dict(max_norm=1.0)`

### 数据验证
- [ ] 检查深度值范围 [0.001, 10.0]
- [ ] 添加DepthValueClipping pipeline
- [ ] 验证没有NaN/Inf值

### 实验跟踪
- [ ] 对比实验A (原始baseline)
- [ ] 对比实验B (改进loss)
- [ ] 对比实验C (改进lr)
- [ ] 对比实验D (组合改进)
- [ ] 记录所有Loss曲线

---

## 🎓 理论基础

### 为什么SILog会发散？

```
SILog = sqrt(E[d²] - λ * E[d]²) * 10

当 E[d²] < λ * E[d]² 时 → sqrt(负数) → NaN
这可能在iter后期发生，因为：
1. λ=0.85很高，容易导致内部为负
2. * 10放大梯度，在后期training激活的变化可能导致数值不稳定
```

### 为什么减少学习率？

```
Depth estimation中，早期学习率1e-4可能太激进：
- 深度是连续值，不是分类问题
- Loss landscape可能有sharp minima
- 降到5e-5能让优化器更稳定探索
```

### 为什么增加warmup？

```
NYU Depth v2有50K样本，60k iterations意味着batch接近1个epoch/iter
- 更长的warmup能让模型gradually adapt
- 1500→5000更符合标准做法 (8-10% of total iters)
```

---

## 🚀 快速开始 (立即执行)

### 方案A: 最小改动 (10分钟)
```python
# 1. 编辑 dep/losses/silog_loss.py，第64行：
# 改为：) * 1.0  # 从 * 10.0

# 2. 运行实验
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --work-dir work_dirs/test_minimal_fix --seed 42
```

### 方案B: 完整改进 (30分钟)
```bash
# 1. 创建改进的config
cp dep/configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
   dep/configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py

# 2. 编辑新config (改lr, warmup, grad_clip)

# 3. 同时修改loss (去掉*10)

# 4. 运行对比实验
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py \
  --work-dir work_dirs/test_stable_all --seed 42
```

---

## 📊 预期改进

| 指标 | 原始 | 改进后 |
|------|------|--------|
| iter 2000 Loss | 发散 | 继续下降 |
| 训练稳定性 | 不稳定 | 稳定 |
| 最终AbsRel | - | 更好 |
| 代码改动 | - | 最小 |

---

## ✅ 成功标志

训练到iter 5000时：
- ✅ Loss没有NaN
- ✅ Loss没有无限增长
- ✅ Loss平滑单调递减
- ✅ 没有CUDA错误

---

**下一步：** 准备好后，开始Phase 1实验，我会一起跟进分析结果。

