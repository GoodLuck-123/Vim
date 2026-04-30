# 模型结构与数据管道归档

**更新日期:** 2026-04-30

本文档记录每次稳定训练结果的完整模型结构与数据管道，用于对比实验与结果复现。

---

## 一、当前最佳结果: Vim-Tiny + DepthHead (SILogLoss)

**工作目录:** `work_dirs/depth_vim_tiny_24_512_60k_uint8`
**配置文件:** `configs/vim/depth/depth_vim_tiny_24_512_60k_single.py`

### 结果

| 指标 | 数值 |
|------|------|
| AbsRel | 0.266 |
| SqRel | 0.173 |
| RMSE | 0.825 |
| RMSElog | 0.307 |
| delta_1 (δ<1.25) | 60.0% |
| delta_2 (δ<1.25²) | 84.5% |
| delta_3 (δ<1.25³) | 94.0% |

### 1. Backbone: VisionMambaSeg

```
总参数量: 7.56M (backbone only)
完整模型: 11.81M (含decode_head)

基础模型: VisionMamba (Vim-Tiny, ImageNet预训练)
预训练权重: vim_t_midclstok_76p1acc.pth (ImageNet top-1: 76.1%)
```

**结构参数:**

| 参数 | 值 | 说明 |
|------|-----|------|
| `img_size` | 512 | 输入尺寸, 224→512 re-interpolate pos_embed |
| `patch_size` | 16 | patch embedding stride, 512/16=32 tokens per side |
| `embed_dim` | 192 | 特征维度 |
| `depth` | 24 | Mamba block 层数 |
| `in_chans` | 3 | RGB 输入 |
| `out_indices` | [5, 11, 17, 23] | 4个stage输出层索引 (1/4尺度特征) |
| `bimamba_type` | `v2` | 双向Mamba v2实现 |
| `if_cls_token` | False | 无分类token (depth不需要) |
| `if_abs_pos_embed` | True | 绝对位置编码 |
| `if_rope` | False | 不使用RoPE |
| `if_rope_residual` | False | — |
| `rms_norm` | False | 使用LayerNorm (非RMSNorm) |
| `residual_in_fp32` | False | — |
| `fused_add_norm` | False | — |
| `final_pool_type` | `all` | 取所有token的特征 |
| `if_divide_out` | True | — |
| `if_bidirectional` | True | 双向Mamba: 偶数层forward, 奇数层backward flip, 求和合并 |

**FPN neck (built-in):**

| 操作 | 说明 |
|------|------|
| fpn1 → out[0] | ConvTranspose2d ×2: embed_dim→embed_dim, scale ×4 |
| fpn2 → out[1] | ConvTranspose2d ×1: embed_dim→embed_dim, scale ×2 |
| fpn3 → out[2] | Identity (1/4尺度) |
| fpn4 → out[3] | MaxPool2d: kernel=2, stride=2 (1/8尺度) |

输出: 4个特征图 `[(B,192,H/4,W/4), (B,192,H/8,W/8), (B,192,H/16,W/16), (B,192,H/32,W/32)]`

**Pretrained权重加载流程:**
1. `torch.load(weights_only=False)` 加载ImageNet checkpoint
2. 移除 `head.weight`, `head.bias`
3. 移除 `rope.freqs_cos`, `rope.freqs_sin` (如果存在)
4. 移除 `cls_token` (pretrained有, depth model没有)
5. `pos_embed[:, 1:, :]` 去掉cls_token位置, bicubic interpolate 14×14→32×32
6. `strict=False` 加载 (RMSNorm→LayerNorm bias随机初始化)

### 2. Decode Head: DepthHead

```
参数量: 4.24M
继承: UPerHead (mmseg) → 单通道depth输出替代多分类logits
```

**结构参数:**

| 参数 | 值 | 说明 |
|------|-----|------|
| `type` | `DepthHead` | 自定义, 继承UPerHead |
| `in_channels` | [192, 192, 192, 192] | 4个FPN输入通道 |
| `in_index` | [0, 1, 2, 3] | 对应backbone的4个输出 |
| `channels` | 192 | head内部通道数 |
| `pool_scales` | (1, 2, 3, 6) | PSP金字塔池化尺度 |
| `dropout_ratio` | 0.1 | FPN+PSP融合后的dropout |
| `num_classes` | 1 | 单通道深度值 |
| `norm_cfg` | GroupNorm(num_groups=32) | 无running stats, train/eval一致 |
| `align_corners` | False | 上采样对齐 |
| `min_depth` | 0.001 | 输出下界 (m) |
| `max_depth` | 10.0 | 输出上界 (m) |

**Forward流程:**
```
backbone features (4 scales)
  → UPerHead FPN + PSP multi-scale fusion
  → (B, 1, H/4, W/4) 特征图
  → F.interpolate(scale_factor=4) → (B, 1, H, W)
  → F.softplus() + min_depth  ← 确保正值, 适配log空间SILogLoss
  → squeeze(1) → (B, H, W)
```

**测试时:**
```
forward → torch.clamp(output, min_depth, max_depth) → (B, H, W)
```

### 3. Loss: SILogLoss

```
类型: Scale-Invariant Log Loss (Eigen et al.)
文件: losses/silog_loss.py
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `variance_focus` λ | 0.85 | 方差项权重 |
| `loss_weight` | 1.0 | loss全局权重 |
| `scale_factor` | 1.0 | 稳定版本(原paper用10.0, 数值不稳定) |

**公式:**
```
L = α · sqrt(mean(d²) - λ · mean(d)²)
d = log(pred) - log(gt)
α = 1.0

其中: pred ≥ 1e-6 (clamp), gt ≥ 1e-6 (clamp)
仅计算有效像素: gt > 0
```

**为什么不用BerHu:**
- BerHu是scale-dependent loss, 模型倾向学训练分布均值 (0.78m)
- SILog在log空间计算, 对scale不敏感, 学到更好的相对深度结构
- BerHu实验: AbsRel=0.70, d1=0.6% (完全崩)

### 4. 数据管道

**训练数据:** NYU Depth v2 — 50,688对 (uint8 PNG)
**测试数据:** NYU Depth v2 — 654对 (uint16 PNG)

| 配置 | 值 |
|------|-----|
| `data_root` | `../data` |
| `img_dir` (train) | `nyu2_train` |
| `ann_dir` (train) | `nyu2_train` |
| `img_dir` (test) | `nyu2_test` |
| `ann_dir` (test) | `nyu2_test` |

**训练pipeline:**

```
1. LoadImageFromFile              — 加载 RGB (jpg)
2. LoadDepthAnnotation            — 加载 depth (png), dtype自适应转meters:
                                    uint8 → float32 / 100.0  (cm→m, 范围0-2.55m)
                                    uint16 → float32 / 1000.0 (mm→m, 范围0-10m)
3. Resize(img_scale=512, ratio_range=(0.5, 2.0))  — 随机缩放
4. RandomCrop(512×512, cat_max_ratio=0.75)        — 随机裁剪
5. RandomFlip(prob=0.5)           — 随机水平翻转
6. PhotoMetricDistortion          — 亮度/对比度/饱和度扰动
7. Normalize                      — ImageNet: mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]
8. Pad(512×512, pad_val=0)        — 补齐
9. ImageToTensor                  — HWC→CHW tensor
10. Collect                       — 收集 img + gt_semantic_seg
```

**测试pipeline:**

```
1. LoadImageFromFile
2. MultiScaleFlipAug(img_scale=512, flip=False)
   └── Resize(keep_ratio=False), Normalize, ImageToTensor, Collect
```

**深度加载关键fix:**
- uint8 (训练): 值域[0,255] → /100 = [0, 2.55m]
- uint16 (测试): 值域[0,65535] → /1000 = [0, 65m], 实际[0.7m, 10m]
- uint8硬上限2.55m → 超过此深度的像素被裁剪 → 模型未见2.55m+深度
- **影响:** 测试集2.55m+区域预测偏小, 但相对深度关系正确

### 5. 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `optimizer` | AdamW | — |
| `lr` | 2e-4 | 基础学习率 |
| `betas` | (0.9, 0.999) | Adam动量 |
| `weight_decay` | 0.02 | 权重衰减 |
| `grad_clip` | max_norm=5.0 | 梯度裁剪 |
| `lr schedule` | poly, power=1.0 | 多项式衰减 |
| `warmup` | linear, 500 iters | warmup_ratio=1e-6 |
| `total_iters` | 60,000 | 约1 epoch (50688/8=6336 iters per epoch) |
| `batch_size` | 8 per GPU | 单GPU训练 |
| `input_size` | 512×512 | 固定尺寸 |
| `fp16` | None | 未启用混合精度 |

### 6. 评估协议

| 参数 | 值 |
|------|-----|
| `eval_interval` | 1000 iters |
| `save_best` | AbsRel (越小越好) |
| `checkpoint_interval` | 5000 iters |
| `max_keep_ckpts` | 3 |

**Per-image LS Log-space Alignment:**
```
对每张预测图单独求解最优scale/shift:
  min_{s,t} ||s·log(pred) + t - log(gt)||²
  pred_aligned = exp(s·log(pred) + t)
  clamp to [1e-3, 10.0]
  然后计算 AbsRel, SqRel, RMSE, RMSElog, δ1/2/3
```

---

## 二、CNN Baseline (对照实验)

**配置文件:** `configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py`

**用途:** 数据管道调试, 无需Mamba CUDA编译

### Backbone: CNNBaseline

```
参数量: 0.91M (backbone), 8.80M (完整模型)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `type` | `CNNBaseline` | 简单4-stage CNN |
| `in_chans` | 3 | RGB输入 |
| `embed_dim` | 64 | 基础通道数 |
| `depths` | (2, 2, 6, 2) | 4个stage的层数 |

**4-stage结构:**
```
Stage 0: Conv2d(3→64) ×2, stride=2 → (B,64,H/2,W/2)
Stage 1: Conv2d(64→128) ×2, stride=2 → (B,128,H/4,W/4)
Stage 2: Conv2d(128→256) ×6, stride=2 → (B,256,H/8,W/8)
Stage 3: Conv2d(256→384) ×2, stride=2 → (B,384,H/16,W/16)
```

### Decode Head: DepthHead (同Vim配置)

| 参数 | 值 |
|------|-----|
| `in_channels` | [64, 128, 256, 384] |
| `channels` | 256 |
| `norm_cfg` | BatchNorm |
| `loss_decode` | SILogLoss |

### 训练差异

| 参数 | CNN | Vim-Tiny |
|------|-----|----------|
| lr | 5e-5 | 2e-4 |
| warmup_iters | 5000 | 500 |
| warmup_ratio | 0.1 | 1e-6 |
| weight_decay | 0.05 | 0.02 |
| grad_clip | max_norm=1.0 | max_norm=5.0 |

CNN从头训练, 无预训练权重, 需要更保守的lr和更长warmup。

---

## 三、Eval指标对照 (CNN vs Vim-Tiny)

| 指标 | CNN Baseline (from scratch) | Vim-Tiny (pretrained) |
|------|----------------------------|----------------------|
| AbsRel | ~0.32 | 0.266 |
| delta_1 | ~46% | 60.0% |
| delta_2 | ~77% | 84.5% |
| delta_3 | ~92% | 94.0% |

Vim-Tiny预训练权重从ImageNet迁移视觉特征, 相对CNN有显著提升。

---

## 四、关键自定义模块文件

```
dep/
├── backbone/vim.py                  # VisionMambaSeg: 注册为@BACKBONES
├── backbone/cnn_baseline.py         # CNNBaseline: 注册为@BACKBONES
├── decode_heads/depth_head.py       # DepthHead: 注册为@HEADS
├── losses/silog_loss.py             # SILogLoss, BerHuLoss: 注册为@LOSSES
├── datasets/nyu_depth_v2.py         # NYUDepthV2Dataset: 注册为@DATASETS
├── datasets/pipelines/depth_loading.py  # LoadDepthAnnotation: 注册为@PIPELINES
├── segmentors/depth_encoder_decoder.py  # DepthEncoderDecoder: 注册为@SEGMENTORS
├── configs/vim/depth/
│   └── depth_vim_tiny_24_512_60k_single.py  # 当前使用config
└── configs/cnn_baseline/
    └── depth_cnn_tiny_512_60k_stable.py     # CNN baseline config
```

---

## 五、版本演进

### v1.0.0 — Stable Baseline (3d30bfd)

**配置:** `depth_vim_tiny_24_512_60k_single.py`
**训练:** AdamW lr=2e-4, bs=8, 60K iters, 无混合精度, 无增强

| AbsRel | RMSE | d1 | d2 | d3 |
|--------|------|----|----|-----|
| 0.266 | 0.825 | 60.0% | 84.5% | 94.0% |

基础pipeline: Resize→RandomCrop→RandomFlip→PhotoMetricDistortion→Normalize→Pad

---

### v1.0.1 — Spatial Augmentation (af55bcb)

**新增:** depth值空间缩放 + 空间裁剪 + 空间翻转 + RGB颜色增强
**结果:** d1 提升约 0.4 个百分点
**结论:** 空间增强有微弱正向作用，但 uint8 数据上限仍是瓶颈

---

### v1.0.1.1 — RGB Augmentation (551228a)

**新增 pipeline transforms (`depth_loading.py`):**

| Transform | 参数 | 作用 |
|-----------|------|------|
| `RandomDepthScale` | ×[0.8, 1.2], prob=0.5 | 模拟不同距离场景 |
| `RandomGaussianBlur` | kernel∈[3,5,7,9], σ∈[0.1,2.0], prob=0.5 | 降低纹理依赖 |
| `RandomGaussianNoise` | σ∈[3,15], prob=0.5 | 模拟传感器噪声 |

**结果:** d1 ~88.2-88.4%，与v1.0.1接近，增强带来的收益有限

---

### v1.0.1.2 — EdgeLoss (bb42390)

**新增 `losses/silog_loss.py` → EdgeLoss:**
- 3×3 Sobel 梯度提取 pred/GT 边缘
- L1 loss on gradient magnitude difference
- loss_weight=0.1 (辅助loss)

**结果:** 与v1.0.1.1基本一致，边缘loss无显著提升
**结论:** SILogLoss 本身对边缘已有较好约束，EdgeLoss 与主 loss 目标部分重叠

---

### v1.0.2 — BF16 + torch.compile (0e05368)

**新增 `mmcv_custom/train_api.py`:**

| 优化 | 实现 | 效果 |
|------|------|------|
| bf16 mixed precision | `torch.amp.autocast('cuda', dtype=torch.bfloat16)` wrap model.forward | **1.4× 加速**, VRAM -31%, 精度无损 |
| torch.compile | `torch.compile(decode_head, mode='default')` | decode_head仅占小部分计算, 无明显加速 |
| torch.compile (backbone) | 尝试失败 | Mamba CUDA ops (selective_scan, causal_conv1d) 造成 Dynamo graph break, 反复重编译, loss spike (0.12→0.51) |

**bf16 不适用之处:** 无。bf16 与 fp32 指数位相同，不需要 loss scaling。

**结果:** 训练速度从 ~0.5s/iter 降到 ~0.36s/iter (bs=8, workers=8)

---

### v1.0.3 — SM_120 Native Compile (a331c52)

**问题:** RTX 5080 (Blackwell, compute capability 12.0) 运行 Mamba CUDA kernel 时，nvcc 编译的 .so 最高只到 sm_90 (Hopper)，driver 在首次 launch 时做 PTX→SASS JIT 编译，每次 kernel 变体有额外开销。

**修改:**
- `mamba-1p1p1/setup.py` — 添加 `arch=compute_120,code=sm_120` (CUDA ≥ 12.0)
- `causal-conv1d/setup.py` — 同上
- 重编译后 `selective_scan_cuda.so` 和 `causal_conv1d_cuda.so` 包含 native sm_120 SASS

**验证:** `cuobjdump --list-text` 确认 sm_120 已编译进 .so

**速度影响:** 消除 PTX JIT 开销，per-iteration 约 5-15% 提升（与 worker 提升叠加后从 ~0.36s → ~0.31s/iter）

**Batch size 联动调整 (`depth_vim_tiny_24_512_60k_single_aug.py`):**

| 参数 | v1.0.2 | v1.0.3 | 说明 |
|------|--------|--------|------|
| `samples_per_gpu` | 8 | 32 | 4× (48 OOM) |
| `workers_per_gpu` | 8 | 16 | 9800X3D 16线程 |
| `lr` | 2e-4 | 2e-4 | AdamW保守, 不严格线性缩放 |
| `max_iters` | 30000 | 10000 | 总样本 320k (原 240k) |

warmup_iters / weight_decay / grad_clip 保持不变 — GroupNorm不受batch size影响。

---

## 六、已知限制

1. **uint8训练数据上限:** 训练GT硬上限2.55m (255cm/100), 模型无法学>2.55m的绝对深度
2. **训练/测试分布不匹配:** train中位0.78m vs test中位2.10m, 虽然LS alignment在eval时校正scale, 但限制了预测的绝对精度
3. **单尺度推理:** 仅512×512固定尺寸, 无multi-scale/flip测试增强
4. **无分层学习率:** 未启用LayerDecayOptimizerConstructor, backbone和head使用相同lr
5. **无混合精度:** fp16/bf16未启用, 训练速度可优化
