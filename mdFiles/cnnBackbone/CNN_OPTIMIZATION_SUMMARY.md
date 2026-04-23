# CNN Backbone优化 - 完整总结

**Date:** 2026-04-21  
**Status:** 顶层设计完成，改进方案已实施  
**下一步:** 在5080上执行对比实验

---

## 📊 问题分析

### 现状
- ✅ CNN backbone在3080验证成功（数据→模型→loss→梯度都正常）
- ❌ 训练不稳定：loss在iter ~2000后发散（NaN/Inf）
- 🔬 根本原因：数值稳定性问题 + 学习率过激进

### 关键洞察
```
Loss Divergence 不是代码bug，而是训练动力学问题：
- SILog Loss * 10.0 放大梯度 → 后期training不稳定
- 学习率1e-4对深度估计可能太高 → 产生sharp minima
- Warmup只有1500 iter → 模型没有充分适应
```

---

## 🔧 实施改进（已完成）

### 改进1: Loss数值稳定性
**文件修改:** `dep/losses/silog_loss.py`
```python
# 前
loss = sqrt(...) * 10.0

# 后
variance_term = clamp(variance_term, min=1e-8)  # 防止sqrt(negative)
loss = sqrt(variance_term) * 1.0  # 降低梯度尺度
```
**预期效果:** Loss不再NaN，更稳定

### 改进2: 学习率调度优化
**文件创建:** `dep/configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py`
```python
optimizer = dict(lr=5e-5, ...)  # 1e-4 → 5e-5
lr_config = dict(
    warmup_iters=5000,  # 1500 → 5000
    warmup_ratio=0.1,   # 1e-6 → 0.1
)
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0),  # 新增
)
```
**预期效果:** 更平滑的学习，避免参数更新过度

### 改进3: 训练配置完善
**扩展配置:**
```python
total_iters = 60000  # 完整训练，不只是快速测试
checkpoint_interval = 5000  # 频繁保存checkpoint
evaluation_interval = 10000  # 定期评估
```

---

## 🧪 对比实验设计

### 3个受控实验

**Exp1: Original Baseline**
```bash
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --work-dir work_dirs/exp1_original
```
预期: iter 2000后loss=NaN (复现问题)

**Exp2: Loss Stable Only**
```bash
# 应用改进1，保持其他不变
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --work-dir work_dirs/exp2_stable_loss
```
预期: Loss比Exp1稳定，但可能还会发散

**Exp3: Full Improvements**
```bash
# 应用改进1+2+3
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py \
  --work-dir work_dirs/exp3_full_stable
```
预期: Loss平滑下降到iter 60k，无NaN

### 对比指标

```
                 Exp1-Original  Exp2-Loss  Exp3-Full
iter 100          ~2.3          ~2.3        ~2.3
iter 1000         ~0.8          ~0.9        ~1.0
iter 2000         NaN           ?            ?
iter 5000         -             ?            ?
Status            Failed ✗      Better?      Stable? ✓
```

---

## 📈 预期改进效果

### 对Loss曲线
```
原始:   ╱╲╱╲╱╲      (iter 2000后发散)
改进:   ╱╲╲╲╲╲╲     (持续下降)
```

### 对训练时间
- Exp1: 0.5h (失败于iter 2000)
- Exp2: 0.5-1h (可能部分改善)
- Exp3: 2-3h (完整60k iterations)

### 对最终性能
```
CNN-Original: 无法完成完整训练
CNN-Stable:   能够完整训练 → 得到最终AbsRel误差
```

---

## ✅ 实验执行清单

### 准备阶段
- [ ] 确认在5080
- [ ] 激活vim_5080环境
- [ ] 验证修改: `grep "* 1.0" dep/losses/silog_loss.py`
- [ ] 检查新config: `ls dep/configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py`
- [ ] 检查数据集

### 实验阶段
- [ ] Exp1: 运行原始配置 → 确认loss diverges
- [ ] Exp2: 运行改进loss → 观察是否改善
- [ ] Exp3: 运行完整改进 → 验证完整60k训练
- [ ] 监控loss曲线 (TensorBoard)
- [ ] 保存所有logs

### 分析阶段
- [ ] 生成Loss对比图表
- [ ] 分析各实验的优缺点
- [ ] 确定最优配置
- [ ] 记录结论

### 后续阶段
- [ ] 应用到Vim backbone配置
- [ ] 更新DEPTH_MODULE_STATUS.md
- [ ] 准备Vim完整训练

---

## 📚 参考文档

已创建的文档：
1. **CNN_TOPLEVEL_DESIGN.md** - 完整设计与分析 (15页)
2. **CNN_EXPERIMENT_PLAN.md** - 详细实验计划 (12页)
3. **CNN_QUICK_REFERENCE.md** - 快速参考卡 (1页)
4. **CNN_OPTIMIZATION_SUMMARY.md** - 本文档

文件位置：
```
/home/dji/projects/Vim/
├── CNN_TOPLEVEL_DESIGN.md       ← 设计原理
├── CNN_EXPERIMENT_PLAN.md       ← 实验过程
├── CNN_QUICK_REFERENCE.md       ← 快速查询
├── CNN_OPTIMIZATION_SUMMARY.md  ← 总结 (本文件)
├── dep/
│   ├── losses/silog_loss.py           ✏️ 已修改
│   ├── configs/cnn_baseline/
│   │   ├── depth_cnn_tiny_512_60k_full.py      (原始)
│   │   └── depth_cnn_tiny_512_60k_stable.py    ✨ 新建
│   ├── train.py
│   └── test.py
```

---

## 🎯 关键决策点

### 1. 为什么改loss * 10.0 → * 1.0?
- 原因：* 10放大梯度，可能导致后期数值不稳定
- 验证：Exp2会测试仅改这个的效果
- 风险：较低（可还原）

### 2. 为什么降学习率 1e-4 → 5e-5?
- 原因：深度估计中1e-4可能太激进，5e-5更稳定
- 验证：Exp3会测试完整效果
- 基准：基于标准最佳实践

### 3. 为什么增加warmup 1500 → 5000?
- 原因：更长warmup让模型gradual adapt，避免急剧变化
- 标准：通常应为总iterations的8-10%
- 5000/60000 = 8.3% ✓

### 4. 为什么添加grad_clip?
- 原因：梯度爆炸是常见原因，clip可防止发散
- 值：max_norm=1.0是标准设置
- 影响：轻微（通常只clip异常梯度）

---

## 🚀 成功标志

### 最小成功
```
✓ Exp1: 复现loss divergence @ iter ~2000
✓ Exp2: Loss比Exp1更稳定（至少延后发散）
✓ Exp3: Loss能否完整训练到iter 60k不发散?
```

### 完全成功
```
✓ Exp3: Loss平滑单调递减
✓ Loss曲线没有NaN/Inf
✓ 训练完整60k iterations
✓ 得到最终AbsRel < 0.35
✓ 可以作为Vim baseline
```

---

## ❌ 失败时的备选方案

### 如果Exp3仍然发散
按优先级尝试：
1. 更激进的lr: 5e-6 或 1e-6
2. 切换loss: BerHuLoss代替SILogLoss
3. 检查数据: 验证没有极端异常值
4. 改进模型: 增加hidden_dim或depth

---

## 💼 项目进度

```
前期 (3080验证):
  ✓ CNN pipeline实现
  ✓ 数据加载验证
  ✓ 模型架构验证
  ✓ 识别问题：loss divergence

当前 (5080准备):
  ✓ 问题诊断完成
  ✓ 改进方案设计
  ✓ 实验计划制定
  ▶ 开始对比实验 ← YOU ARE HERE

后续 (5080训练):
  ⏳ 验证改进有效
  ⏳ 完整CNN训练
  ⏳ 应用到Vim
  ⏳ Vim完整训练
```

---

## 📝 记录模板

创建 `EXPERIMENT_RESULTS.md` 记录：

```markdown
# CNN Optimization Experiment Results

## Environment
- GPU: RTX 5080
- Date: 2026-04-21
- Python/PyTorch: [version]

## Exp1: Original Baseline
- Start: [time]
- Status: [Running/Complete/Failed]
- Loss @ iter 100: X.XXX
- Loss @ iter 1000: X.XXX
- Loss @ iter 2000: [NaN / X.XXX]
- Observation: ...

## Exp2: Loss Stable
- Start: [time]
- Status: ...
- Loss @ iter 5000: [?]
- Observation: ...

## Exp3: Full Stable
- Start: [time]
- Status: ...
- Loss @ iter 60000: [?]
- Observation: ...

## Conclusion
Which improvements were most effective?
Recommendation for Vim backbone?
```

---

## ✨ 总结

你现在拥有：
1. ✅ **完整的问题诊断** - 知道发生了什么和为什么
2. ✅ **基于理论的改进方案** - 3个互补的改进
3. ✅ **受控的实验设计** - 能隔离每个改进的效果
4. ✅ **详细的执行指南** - CNN_EXPERIMENT_PLAN.md
5. ✅ **快速参考资料** - CNN_QUICK_REFERENCE.md

**下一步：** 在5080上执行3个对比实验，验证哪些改进有效，然后应用到Vim backbone！

---

**Expected Timeline:**
- 实验执行：3-5小时
- 分析结果：15分钟
- 应用到Vim：15分钟

**Total:** 一个下午 → 得到稳定的CNN + 改进方案应用到Vim

🚀 Ready to optimize!

