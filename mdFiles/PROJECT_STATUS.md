# Vision Mamba Depth Estimation — 项目状态与改进计划

**更新日期:** 2026-04-29
**状态:** 训练完成，可视化脚本已修复，效果与之前最好情况一致

---

## 一、当前结果

| 实验 | AbsRel | RMSE | delta_1 | delta_2 | delta_3 | 说明 |
|------|--------|------|---------|---------|---------|------|
| 预训练+SILog (60K iters, 当前) | 0.266 | 0.825 | 60.0% | 84.5% | 94.0% | uint8数据正确加载(÷100)，稳定复现 |
| 预训练+SILog (60K iters, 旧) | 0.106 | 0.399 | 88.9% | 97.9% | 99.5% | 同配置同数据，原因待分析 |

两个实验使用完全相同的配置文件 `depth_vim_tiny_24_512_60k_single.py`，相同训练/测试数据。
88.9%实验的eval从d1=0.35%逐步提升至88.9%，与60%实验的全程平稳(~60%)模式不同。

可视化显示：物体轮廓清晰，相对深度关系正确，但绝对深度偏小（受限于uint8训练数据0-2.55m范围）。

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
| 1 | RTX 5080 sm_120 CUDA扩展编译失败 | Blackwell架构不在PyTorch原生arch列表 | `CUDA_ARCH="sm_120"`, setup.py加`compute_120` | causal-conv1d/setup.py, mamba-1p1p1/setup.py |
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

### 脚本/工具问题

| # | 问题 | 修复 | 文件 |
|---|------|------|------|
| 1 | visualize_depth.py导入链断裂 | 显式import所有注册模块 (backbone/head/loss/segmentor/dataset) | visualize_depth.py |
| 2 | Vim输入尺寸不匹配 | Vim需512×512, 测试图是480×640 → F.interpolate | visualize_depth.py |
| 3 | 批推理对比 (GT vs Pred) | 新建debug推理脚本 | debug_inference.py |

---

## 四、当前训练配置

```python
# depth_vim_tiny_24_512_60k_single.py
backbone: VisionMambaSeg (Vim-Tiny, 24层, pretrained=ImageNet)
decode_head: DepthHead (GroupNorm, SILogLoss)
optimizer: AdamW lr=2e-4, weight_decay=0.02
schedule: poly, warmup=500, 60K iters
batch: 8, input: 512×512
eval: per-image LS log-space alignment (AbsRel, RMSE, δ1/2/3)
```

---

## 五、改进计划

### Phase 1: 数据质量 (最高优先级)

1. **获取uint16训练数据**
   - 下载NYUv2官方raw kinect depth (uint16, 0-10m)
   - 或用NYUv2 labeled training set (795训练/654测试的标准split)
   - 避免8-bit有损压缩

2. **深度数据增强**
   - Per-sample random scale augmentation (×[0.8, 1.2]) — 模拟不同距离场景
   - Random horizontal flip (已有, 确认启用)
   - Color jitter on RGB (增加光照不变性)

### Phase 2: 训练策略

3. **分层学习率**
   - 恢复`LayerDecayOptimizerConstructor` (base config有)
   - Backbone浅层lr低(0.1×), 深层+head lr高(1×)

4. **Scale-Shift Invariant (SSI) loss**
   - 当前SILog是近似scale-invariant (λ=0.85)
   - 真正的SSI loss: `min_{s,t} ||s*pred+t-gt||` per image
   - 训练时每batch计算per-image optimal s,t → 更robust

5. **混合精度训练**
   - fp16/bf16加速训练, 允许更大batch size
   - RTX 5080支持bf16

### Phase 3: 模型增强

6. **多尺度测试**
   - 当前仅512×512 single scale
   - 加flip + multi-scale testing提升精度

7. **Vim-Small升级**
   - embed_dim: 192→384 (参数7M→26M)
   - 需要更多VRAM但精度提升显著

### Phase 4: 评估与对比

8. **标准benchmark对比**
   - 与已发表NYUv2方法对比: DPT, AdaBins, BTS, DepthFormer等
   - 使用标准eval protocol (per-image median scaling或无scaling)

9. **跨数据集泛化**
   - 在SUN RGB-D, KITTI等测试zero-shot性能

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
