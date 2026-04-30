# Vision Mamba Depth Estimation — 项目状态与改进计划

**更新日期:** 2026-04-30
**状态:** 训练优化完成 — 增强/EdgeLoss/bf16/SM_120/batch_size 全部实施，d1 ~88.4% plateau（受限于 uint8 数据 + Vim-Tiny 容量）

**更新日期:** 2026-05-01
**状态:** torch.compile 导致 GroupNorm 全部冻结 + 深度图网纹 → 确认禁用 compile（详见 三、8号问题）

---

## 一、版本演进与结果

| 版本 | Git | 主要改动 | AbsRel | d1 | 备注 |
|------|-----|----------|--------|-----|------|
| v1.0.0 | `3d30bfd` | Stable baseline, SILog, 无增强 | 0.266 | 60.0% | uint8数据正确加载 |
| v1.0.1 | `af55bcb` | 空间增强 (scale/crop/flip/color) | — | +0.4pp | 微弱提升 |
| v1.0.1.1 | `551228a` | +RandomDepthScale/GaussianBlur/Noise | — | ~88.4% | 增强收益有限 |
| v1.0.1.2 | `bb42390` | +EdgeLoss (Sobel gradient L1) | — | ~88.4% | 无显著提升 |
| v1.0.2 | `0e05368` | bf16 + torch.compile | — | ~88.4% | 1.4×加速, 精度无损 |
| v1.0.3 | `a331c52` | SM_120 native compile + bs/lr联动 | — | 训练中 | 0.31s/iter @ bs=32 |
| v1.0.4 | `—` | **禁用 torch.compile** (GN freeze bug), 保留 bf16+SM_120+增强 | — | 0.89 @ iter_42000 | compile 使所有 GN 层 γ=1.0/β=0.0 冻结, 深度图现 16px 网纹; 禁用后恢复正常 |

---

## 二、数据集问题 — 根因分析

### 训练数据格式

```
文件: /home/eric/infra/Vim/data/nyu2_train/ (50688 对)
深度格式: uint8 PNG (8-bit 灰度)
编码: 像素值直接代表厘米 (cm), 值域 [0, 255]
实际范围: [0.2m, 2.55m] ← 硬上限
测试格式: uint16 PNG (16-bit), 毫米编码, 值域 [0.7m, 10m]
```

### 影响

- uint8只能表示0-255 → /100后 = 0-2.55m
- 所有 >2.55m 的像素被裁剪为255 (2.55m)
- 训练中位深度 0.78m vs 测试 2.10m — 分布严重不匹配
- **模型从未见过2.55m以上的深度 → 无法学会预测大尺度场景**

### 解决方案

获取NYUv2原始uint16训练数据，或对当前数据做尺度增强(depth scaling augmentation)。

---

## 三、已解决问题汇总

### 环境问题

| # | 问题 | 根因 | 修复 | 文件 |
|---|------|------|------|------|
| 1 | RTX 5080 sm_120 CUDA扩展编译失败 | Blackwell架构不在mamba/causal-conv1d setup.py arch列表 | setup.py加`arch=compute_120,code=sm_120`, 重编译含native sm_120 SASS | causal-conv1d/setup.py, mamba-1p1p1/setup.py |
| 2 | `torch.load` 报 UnpicklingError | PyTorch 2.6+ `weights_only` 默认改为 True | 所有 `torch.load()` 加 `weights_only=False` | backbone/vim.py, visualize_depth.py |
| 3 | MMCV._ext 导入失败 | mmcv-full 缺少 _ext 模块 (无CUDA编译) | 创建 stub 模块 (`mmcv/_ext.py`) 模拟缺失接口 | site-packages/mmcv/_ext.py |
| 4 | Python版本不兼容 | vim需要3.10, dep需要3.9 | conda env用Python 3.9.19 | conda环境 |
| 5 | PyTorch版本升级 | 5080需CUDA 12.4+, 旧torch 1.12不支持 | torch 2.8.0+cu128 | 环境 |
| 6 | LD_LIBRARY_PATH 缺少torch lib | mamba .so链接不到torch | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH` | ~/.zshrc |

### 架构问题

| # | 问题 | 根因 | 修复 | 文件 |
|---|------|------|------|------|
| 1 | 双向Mamba forward特征收集bug | `forward_features`中out_indices对应的特征未正确提取 | 修复特征收集循环逻辑 | backbone/vim.py (之前修) |
| 2 | BN train/eval不一致 | Vim backbone特征极不稳定(running_var达500K+), BN在eval模式用错误统计量 | 改用GroupNorm (无running stats, train/eval一致) | configs/.../depth_vim_tiny_24_512_60k_single.py |
| 3 | decode_head forward_test有sigmoid | depth regression不应有sigmoid (原用于segmentation) | 移除sigmoid, 用clamp替代 | decode_heads/depth_head.py (之前修) |
| 4 | BerHuLoss variance_focus参数泄露 | 父类config的`variance_focus`传给了BerHuLoss(不需要此参数) | `_delete_=True` 清掉旧参数 | configs/.../depth_vim_tiny_24_512_60k_single.py |
| 5 | DepthEncoderDecoder未注册 | 自定义segmentor需显式注册 | `@SEGMENTORS.register_module()` | segmentors/depth_encoder_decoder.py |
| 6 | Vim pretrained权重加载 | pretrained有cls_token, depth模型没有 → pos_embed reshape崩溃 | init_weights: 移除cls_token+pos_embed第一个位置, 然后interpolate | backbone/vim.py |
| 7 | pos_embed尺寸不匹配 | pretrained 14×14 (224²), depth 32×32 (512²) | interpolate_pos_embed bicubic插值 | backbone/vim.py |
| 8 | RMSNorm→LayerNorm bias缺失 | pretrained用RMSNorm(无bias), depth用LayerNorm(有bias) | strict=False, bias随机初始化 | — |

### 训练问题

| # | 问题 | 根因 | 修复 | 文件 |
|---|------|------|------|------|
| 1 | 训练GT深度10x scale error | 训练PNG是uint8(cm→/100), 但误用/1000(mm) | dtype检测: uint8÷100, uint16÷1000 | datasets/pipelines/depth_loading.py, datasets/nyu_depth_v2.py |
| 2 | 从零训练不收敛 | 24层Mamba + 仅50K图片, 无预训练权重 | 加载Vim官方ImageNet预训练权重 | config中 backbone.pretrained='...' |
| 3 | BerHuLoss学训练分布均值 | BerHu是scale-dependent loss, 模型学预测0.8m (训练中位) | 换SILogLoss (scale-invariant in log space) | configs/.../single.py |
| 4 | Eval指标无意义 | 测试分布与训练分布不同, 无scale对齐 | Per-image LS log-space alignment before metrics | datasets/nyu_depth_v2.py |
| 5 | **torch.compile 导致 GroupNorm 全部冻结 + 深度图网纹** | compile 将 GN backward 的空间归约 (Σ over H×W) 融合进 CUDA kernel, bf16 下 16384 次累加导致小梯度 underflow → γ/β 梯度归约结果趋近 0。AdamW exp_avg 始终 ≈ 0, 权重永远停在 γ=1.0, β=0.0。同时 ReLU mask 可能错误传播到 GN 梯度路径。效果: 3×3 conv 补偿 GN 缺失 → 高噪 UPerHead 输出 → bilinear 上采样产生 16px 周期网纹。GPU kernel 内静默失败, 精度照涨 (LS alignment 兜底), 无 NaN 无报错 | **禁用 compile** (`compile = 'default'` 注释掉), GN 恢复正常训练, 网纹消失 | configs/.../single_aug.py, decode_heads/depth_head.py |

### 脚本/工具问题

| # | 问题 | 修复 | 文件 |
|---|------|------|------|
| 1 | visualize_depth.py导入链断裂 | 显式import所有注册模块 (backbone/head/loss/segmentor/dataset) | visualize_depth.py |
| 2 | Vim输入尺寸不匹配 | Vim需512×512, 测试图是480×640 → F.interpolate | visualize_depth.py |
| 3 | 批推理对比 (GT vs Pred) | 新建debug推理脚本 | debug_inference.py |

---

## 四、当前训练配置

```python
# depth_vim_tiny_24_512_60k_single_aug.py
backbone: VisionMambaSeg (Vim-Tiny, 24层, pretrained=ImageNet)
decode_head: DepthHead (GroupNorm, SILogLoss only, EdgeLoss 已删除)
optimizer: AdamW lr=2e-4, weight_decay=0.02, grad_clip=max_norm=5.0
schedule: poly, warmup=500, max_iters=60000
batch: 32, workers: 16, input: 512×512
fp16: bf16 autocast, compile: 已禁用 (torch.compile 导致 GN 冻结+网纹, 见三/训练问题#5)
augmentation: RandomDepthScale + GaussianBlur + GaussianNoise + PhotoMetricDistortion
eval: per-image LS log-space alignment (AbsRel, RMSE, δ1/2/3) ⚠️ 早期会严重虚高, Spearman 才是真实质量
```

---

## 五、改进计划

### Phase 1: 数据质量 ✅ (已完成)

1. **获取uint16训练数据** ⬜ (未做)
2. **深度数据增强** ✅
   - `RandomDepthScale` ×[0.8, 1.2] (v1.0.1.1)
   - Random horizontal flip (已有)
   - `RandomGaussianBlur` + `RandomGaussianNoise` (v1.0.1.1)
   - `PhotoMetricDistortion` (已有)

### Phase 2: 训练策略 ✅ (核心项已完成)

3. **分层学习率** ⬜ (未做)
4. **SSI loss** ⬜ (未做, SILog足够)
5. **混合精度训练** ✅
   - bf16 via `torch.amp.autocast` (v1.0.2)
   - 1.4×加速, VRAM -31%, 精度无损
6. **EdgeLoss** ✅ (v1.0.1.2)
   - Sobel gradient L1, loss_weight=0.1
   - 无显著提升, 保留为辅助loss
7. **SM_120 native compile** ✅ (v1.0.3)
   - mamba-1p1p1 + causal-conv1d 重编译含 sm_120 SASS
   - 消除 Blackwell PTX JIT 开销
8. **torch.compile** ✅ (v1.0.2)
   - decode_head only (backbone Mamba ops 不兼容)

### Phase 3: 模型增强

9. **多尺度测试** ⬜
10. **Vim-Small升级** ⬜

### Phase 4: 评估与对比

11. **标准benchmark对比** ⬜
12. **跨数据集泛化** ⬜

---

## 六、关键文件索引

```
dep/
├── backbone/vim.py                  # VisionMambaSeg — pretrained加载, 特征提取
├── decode_heads/depth_head.py       # UPerHead→单通道depth, clamp [1e-3, 10]
├── losses/silog_loss.py             # SILogLoss + BerHuLoss
├── datasets/nyu_depth_v2.py         # 数据加载 + eval (LS alignment)
├── datasets/pipelines/depth_loading.py  # uint8/uint16自适应加载
├── segmentors/depth_encoder_decoder.py  # EncoderDecoder for depth
├── configs/vim/depth/
│   ├── depth_vim_tiny_24_512_60k.py        # 基础训练config
│   └── depth_vim_tiny_24_512_60k_single.py # 单GPU config (当前使用)
├── train.py                         # 训练入口
├── test.py                          # 评估入口
├── visualize_depth.py               # 可视化: GT vs Pred / Original vs Depth side-by-side视频
│                                     #   视频模式: 2面板(原始|深度)，FPS=原视频，无多视图分割
│                                     #   参数: --config --checkpoint --mode --input --output
└── debug_inference.py               # 快速推理debug脚本

data/
├── nyu2_train/                      # 50,688对 uint8 (0-2.55m)
└── nyu2_test/                       # 654对 uint16 (0-10m)

work_dirs/vim_t_midclstok_76p1acc.pth  # Vim-Tiny ImageNet预训练权重 (76.1% top-1)
```

---

## 七、常用命令

```bash
# 训练 (单GPU)
python train.py configs/vim/depth/depth_vim_tiny_24_512_60k_single.py \
  --launcher none --gpus 1 --work-dir work_dirs/depth_vim_tiny_24_512_60k

# 评估
python test.py configs/vim/depth/depth_vim_tiny_24_512_60k_single.py \
  checkpoint.pth

# 可视化 — pairs模式 (GT vs Predicted)
python visualize_depth.py --mode pairs \
  --config configs/vim/depth/depth_vim_tiny_24_512_60k_single.py \
  --checkpoint checkpoint.pth \
  --input ../data/nyu2_test \
  --output depth_comparison.mp4

# 可视化 — video模式 (Original | Depth)
python visualize_depth.py --mode video \
  --input ../data/test01.mp4 \
  --output test01_depth_pred.mp4

# GPU监控
nvidia-smi -l 1
```
