# Depth Estimation Troubleshooting Guide

本文档汇总了Vim depth depth estimation的常见问题和解决方案。

## 数据加载问题

### 问题：FileNotFoundError: depth/data_root not found

**症状：**
```
FileNotFoundError: /path/to/nyu_depth_v2 is not found
```

**原因：** 数据路径配置错误

**解决：**
```bash
# 1. 检查数据集位置
ls /your/dataset/nyu_depth_v2/train/
# 应该看到: 0000_colors.png, 0000_depths.png, ...

# 2. 更新配置文件
# 编辑 dep/configs/_base_/datasets/nyu_depth_v2.py
data_root = '/your/actual/path/to/nyu_depth_v2'

# 3. 验证配置
python -c "
from mmcv import Config
cfg = Config.fromfile('configs/_base_/datasets/nyu_depth_v2.py')
print('data_root:', cfg.data_root)
"
```

### 问题：RuntimeError: LoadDepthAnnotation not registered

**症状：**
```
RuntimeError: LoadDepthAnnotation is not registered in PIPELINES
```

**原因：** 自定义pipeline未正确导入

**解决：**
```python
# 确保在train.py/test.py中导入了custom modules
from datasets.pipelines import depth_loading  # 这会触发@PIPELINES.register_module()

# 验证
from mmseg.datasets import PIPELINES
assert 'LoadDepthAnnotation' in PIPELINES.module_dict
```

---

## 模型加载问题

### 问题：ImportError: cannot import VisionMambaSeg

**症状：**
```
ImportError: cannot import name 'VisionMambaSeg' from backbone
```

**原因：** 
1. Mamba编译失败
2. LD_LIBRARY_PATH未配置

**解决：**

```bash
# 方法1：检查.so文件
find . -name "causal_conv1d_cuda*.so" -o -name "mamba*.so"

# 如果为空，说明编译失败，重新编译
cd causal-conv1d
python setup.py clean --all
python setup.py build_ext --inplace

# 方法2：设置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

# 方法3：测试导入
python -c "import causal_conv1d_cuda; print('✓')"
python -c "from mamba_ssm import mamba_simple; print('✓')"
```

### 问题：RuntimeError: CUDA compute capability mismatch

**症状：**
```
RuntimeError: The CUDA compute capability of your GPU does not match the PTX code
Supported: sm_87, but got sm_90
```

**原因：** 编译时的CUDA架构与运行GPU不匹配

**解决：**

```bash
# 1. 查看GPU型号和架构
nvidia-smi

# 查看支持的架构映射
# RTX 3080: sm_86
# RTX 4080: sm_89
# RTX 5080: sm_89 (Ada)
# H100: sm_90

# 2. 重新编译
export CUDA_ARCH="sm_89"  # 根据GPU修改
cd causal-conv1d
python setup.py clean --all
python setup.py build_ext --inplace
cd ../mamba-1p1p1
python setup.py clean --all
python setup.py build_ext --inplace
cd ..

# 3. 验证
python -c "
import causal_conv1d_cuda
import torch
print('CUDA Capability:', torch.cuda.get_device_capability())
"
```

---

## 训练问题

### 问题：TypeError: forward() got unexpected keyword argument

**症状：**
```
TypeError: forward() got an unexpected keyword argument 'return_loss'
```

**原因：** mmseg版本与深度模块版本不兼容

**解决：**
```bash
# 确保使用正确版本
pip install mmsegmentation==0.29.1 --force-reinstall

# 验证版本
python -c "import mmseg; print(mmseg.__version__)"
```

### 问题：Loss is NaN or Inf

**症状：**
```
iter 100: loss_depth nan
```

**原因：** 
1. 数值溢出（深度值过大或过小）
2. 学习率过高
3. 数据预处理不正确

**解决：**

```python
# 方法1：检查数据范围
# 编辑 dep/configs/vim/depth/depth_vim_tiny_24_512_60k.py
# 深度值应在 [0.001, 10.0] 范围内

# 方法2：降低学习率
optimizer = dict(
    type='AdamW',
    lr=1e-5,  # 从1e-4降低到1e-5
    ...
)

# 方法3：增加Warmup
lr_config = dict(
    ...
    warmup_iters=3000,  # 增加warmup
)

# 方法4：启用混合精度 (可选)
fp16 = dict(loss_scale='dynamic')

# 方法5：检查数据标准化
# 在 dep/datasets/nyu_depth_v2.py 中验证深度值范围
```

### 问题：CUDA Out of Memory (OOM)

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
```

**原因：** Batch size过大或模型过大

**解决：**

```python
# 方法1：减小batch size
# 编辑 dep/configs/vim/depth/depth_vim_tiny_24_512_60k.py
data = dict(
    samples_per_gpu=4,  # 从8改为4
    workers_per_gpu=16,
)

# 方法2：减小输入尺寸
crop_size = (384, 384)  # 从512改为384

# 方法3：减小模型大小
# 使用CNN baseline进行快速测试：
# python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py

# 方法4：启用梯度累积
# 修改 optimizer_config
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=2,  # 累积2个batch再更新
)
```

---

## 评估问题

### 问题：KeyError when evaluating

**症状：**
```
KeyError: 'depth' not in evaluation outputs
```

**原因：** 评估指标配置不正确

**解决：**
```bash
# 在test.py中明确指定评估指标
python test.py config.py checkpoint.pth --eval mde

# 或在配置文件中添加
evaluation = dict(
    interval=1000,
    metric='mde',  # Mean Depth Error
)
```

### 问题：AbsRel/RMSE指标异常高

**症状：**
```
AbsRel: 0.85 (期望 < 0.3)
RMSE: 5.0  (期望 < 1.0)
```

**原因：** 
1. 模型未正确收敛
2. 数据预处理/后处理错误
3. 评估逻辑有bug

**解决：**
```python
# 1. 检查验证损失是否下降
# 查看 work_dirs/*/tf_logs/ 中的TensorBoard日志

# 2. 手动检查单个预测
import torch
from mmcv import Config, load

cfg = Config.fromfile('config.py')
img = cv2.imread('test.jpg')
# 推理并可视化结果

# 3. 检查深度值范围
print('Min depth:', depth.min())
print('Max depth:', depth.max())
print('Mean depth:', depth.mean())
# 应该在 [0.001, 10.0] 范围内
```

---

## 性能问题

### 问题：训练速度过慢

**症状：**
```
iter 100 (100/60000): loss_depth 2.345 (100 ms/iter)
```

**原因：** 
1. 数据加载瓶颈
2. 编译配置不优化

**解决：**

```python
# 方法1：增加数据加载线程
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,  # 增加worker数量
)

# 方法2：启用pin_memory
cfg.data.train.pipeline[0] = dict(
    type='LoadImageFromFile',
    to_float32=True,
    pin_memory=True,  # 锁定内存
)

# 方法3：使用混合精度
fp16 = dict(loss_scale='dynamic')

# 方法4：检查是否使用了GPU
nvidia-smi  # 应该看到高利用率
```

---

## 文件和路径问题

### 检查清单

```bash
# 1. 验证项目结构
ls -la dep/
# 应该有: backbone/, datasets/, decode_heads/, losses/, configs/, train.py, test.py

# 2. 验证数据集
ls -la /your/nyu_depth_v2/train/ | head -10
# 应该看到: 0000_colors.png, 0000_depths.png, ...

# 3. 验证配置文件
ls -la dep/configs/vim/depth/
# 应该有: depth_vim_tiny_24_512_60k.py, depth_vim_small_24_512_60k.py

# 4. 验证编译产物
find . -name "*.so" | grep -E "(causal_conv1d|mamba)"
# 应该找到至少2个.so文件

# 5. 验证backbone文件
ls -la dep/backbone/
# 应该有: __init__.py, cnn_baseline.py, vim.py
```

---

## 调试技巧

### 启用详细日志

```bash
# 设置日志级别
export PYTHONUNBUFFERED=1

# 运行训练并保存日志
python train.py config.py \
  --launcher none \
  --gpus 1 \
  --work-dir work_dirs/debug 2>&1 | tee train.log

# 查看日志
tail -100 train.log
```

### 逐步验证

```bash
# 1. 验证数据加载
python -c "
from datasets import NYUDepthV2Dataset
from mmcv import Config

cfg = Config.fromfile('configs/_base_/datasets/nyu_depth_v2.py')
dataset = NYUDepthV2Dataset(cfg.data.train)
sample = dataset[0]
print('Sample keys:', sample.keys())
print('Image shape:', sample['img'].shape)
print('Depth shape:', sample['gt_semantic_seg'].shape)
"

# 2. 验证模型前向
python -c "
from mmseg.models import build_segmentor
from mmcv import Config

cfg = Config.fromfile('configs/vim/depth/depth_vim_tiny_24_512_60k.py')
model = build_segmentor(cfg.model).cuda()
x = torch.randn(1, 3, 512, 512).cuda()
y = model(x)
print('Output shape:', y.shape)
"

# 3. 验证损失
python -c "
from losses import SILogLoss
import torch

loss_fn = SILogLoss()
pred = torch.randn(2, 512, 512, requires_grad=True).cuda()
target = torch.randn(2, 512, 512).cuda() * 5 + 1  # 在[0,10]范围

loss = loss_fn(pred, target)
print('Loss:', loss)
loss.backward()
print('✓ Backward pass OK')
"
```

---

## 快速问题排查

| 症状 | 最可能原因 | 第一步检查 |
|------|----------|----------|
| ImportError: VisionMambaSeg | Mamba未编译 | `find . -name "*.so"` |
| CUDA OOM | Batch size过大 | 修改 `samples_per_gpu` |
| Loss is NaN | 数据范围错误 | 检查深度值是否在[0.001,10.0] |
| 训练超慢 | I/O瓶颈 | 增加 `workers_per_gpu` |
| 文件找不到 | 路径配置错误 | 检查 `data_root` 设置 |
| 类型错误 | 版本不兼容 | 验证mmseg==0.29.1 |
| 精度差 | 未收敛 | 检查是否需要pre-trained weights |

---

## 获得帮助

1. **检查日志**：`tail -50 work_dirs/*/run.log`
2. **查看配置**：`cat dep/configs/vim/depth/depth_vim_tiny_24_512_60k.py`
3. **验证环境**：`python -c "import torch, mmseg, timm; print('OK')"`
4. **测试单个组件**：运行上面的"逐步验证"脚本

---

**最后更新：** 2026-04-21
