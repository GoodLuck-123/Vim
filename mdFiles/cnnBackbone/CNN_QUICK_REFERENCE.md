# CNN Backbone - Quick Reference Card

## 🎯 核心问题
```
Loss diverges at iter ~2000
Likely cause: Numerical instability in loss function + aggressive learning rate
```

## 🔧 3个改进已应用

### 1️⃣ Loss Stability (已修改)
```diff
- loss = sqrt(...) * 10.0
+ loss = sqrt(...) * 1.0
+ variance_term = clamp(variance_term, min=1e-8)
```
**文件:** `dep/losses/silog_loss.py`

### 2️⃣ Learning Rate Schedule (新Config)
```diff
- lr: 1e-4 → 5e-5
- warmup: 1500 → 5000
- warmup_ratio: 1e-6 → 0.1
+ grad_clip: max_norm=1.0
```
**文件:** `dep/configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py`

### 3️⃣ Training Configuration
```diff
- total_iters: 1000 → 60000
+ checkpoint_interval: 5000
+ evaluation_interval: 10000
```

---

## 🚀 3个对比实验

### Exp1: Original (Baseline)
```bash
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --work-dir work_dirs/exp1_original --seed 42
# 预期: Loss @ iter ~2000 = NaN
```

### Exp2: Loss Stable Only  
```bash
# 确保 dep/losses/silog_loss.py 已修改
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --work-dir work_dirs/exp2_stable_loss --seed 42
# 预期: Loss更稳定，但可能还会发散
```

### Exp3: Full Improvements
```bash
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py \
  --work-dir work_dirs/exp3_full_stable --seed 42
# 预期: Loss smooth descent to iter 60k
```

---

## 📊 监控Loss

```bash
# Terminal 1: 运行训练
cd dep
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py \
  --launcher none --gpus 1 --work-dir work_dirs/test --seed 42

# Terminal 2: 实时查看Loss
tail -f work_dirs/test/run.log | grep "loss_depth"

# Terminal 3: TensorBoard
tensorboard --logdir work_dirs/test/
```

---

## ✅ 成功检查清单

### Before Experiment
- [ ] 确认在5080上
- [ ] 激活vim_5080环境
- [ ] 检查GPU: `nvidia-smi`
- [ ] 检查数据集路径
- [ ] 验证loss已修改: `grep "* 1.0" dep/losses/silog_loss.py`

### During Experiment
- [ ] Exp1: 观察iter 2000是否loss diverges
- [ ] Exp2: 检查loss是否比Exp1更稳定
- [ ] Exp3: 监控loss是否平滑下降

### After Experiment
- [ ] [ ] 生成loss对比图表
- [ ] [ ] 选择最稳定的配置
- [ ] [ ] 应用到Vim backbone
- [ ] [ ] 记录结论到EXPERIMENT_RESULTS.md

---

## 🎓 理论总结

```
为什么loss会发散?

SILog Loss 公式:
  loss = sqrt(E[log(pred/target)²] - λ * E[log(pred/target)]²) * 10

问题:
  1. 当内部为负时 → sqrt(negative) = NaN
  2. * 10放大梯度 → 后期training可能爆炸

解决方案:
  1. * 1.0代替* 10.0 → 降低梯度尺度
  2. clamp内部 → 防止sqrt(negative)
  3. 降低学习率 → 更稳定的优化
  4. 增加warmup → 更平滑的学习

预期效果:
  ✓ Loss继续下降而不发散
  ✓ 稳定收敛到全部60k iterations
  ✓ 更低的最终AbsRel误差
```

---

## 📋 关键文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `dep/losses/silog_loss.py` | ✏️ 已修改 | Loss scale: 10→1, 加epsilon |
| `dep/configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py` | ✨ 新建 | 改进的配置 |
| `CNN_TOPLEVEL_DESIGN.md` | 📖 参考 | 完整设计文档 |
| `CNN_EXPERIMENT_PLAN.md` | 📖 参考 | 详细实验计划 |

---

## 💡 如果仍然失败

按优先级尝试:

1. **更激进的lr**
   ```python
   # 改为 1e-5 或 5e-6
   optimizer = dict(lr=1e-5, ...)
   ```

2. **切换Loss函数**
   ```python
   loss_decode=dict(type='BerHuLoss', threshold=0.2)
   ```

3. **检查数据异常**
   ```bash
   python -c "
   from dep.datasets import NYUDepthV2Dataset
   d = NYUDepthV2Dataset(...)
   s = d[0]
   print(s['gt_semantic_seg'].min(), s['gt_semantic_seg'].max())
   "
   ```

4. **改进模型**
   ```python
   embed_dim=80  # 增大特征维度
   depths=(3, 4, 6, 3)  # 增加blocks
   ```

---

## 🎉 完成后

```bash
# 一旦CNN稳定，应用相同策略到Vim:
# 1. 更新 dep/configs/vim/depth/depth_vim_tiny_24_512_60k.py
# 2. 应用相同的:
#    - lr = 5e-5
#    - warmup_iters = 5000
#    - warmup_ratio = 0.1
#    - grad_clip = dict(max_norm=1.0)
# 3. 训练Vim backbone
```

---

**Status:** Ready to Execute 🚀

