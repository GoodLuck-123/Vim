"""Quick debug script to check model predictions."""
import torch
import numpy as np
import imageio
import os.path as osp
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmseg.models import build_segmentor
from mmseg.datasets.pipelines import Compose
from mmseg.datasets import build_dataset

import backbone.vim
import decode_heads.depth_head
import losses.silog_loss
import segmentors
import datasets.nyu_depth_v2  # register NYUDepthV2Dataset
import datasets.pipelines.depth_loading  # register LoadDepthAnnotation

checkpoint = 'work_dirs/depth_vim_tiny_24_512_60k/iter_5000.pth'
config_path = 'configs/vim/depth/depth_vim_tiny_24_512_60k_single.py'

cfg = Config.fromfile(config_path)
print("Building model...")
model = build_segmentor(cfg.model)
ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()
model.cuda()

# Load test dataset for GT access
cfg.data.val.test_mode = True
dataset = build_dataset(cfg.data.val)

# Build test pipeline
test_pipeline = Compose(cfg.data.val.pipeline)

print(f"Dataset size: {len(dataset)}")

all_preds = []
all_gts = []
total_pixels = 0

for idx in range(min(10, len(dataset))):
    # Load one sample using the dataset's raw info
    img_path = osp.join(dataset.data_root, dataset.img_dir, dataset.img_infos[idx]['filename'])

    # Apply test pipeline
    data = dict(img=img_path, img_prefix=None, seg_prefix=None, img_info=dict(filename=img_path))
    data = test_pipeline(data)

    # MultiScaleFlipAug: img is list of tensors, img_metas is list
    img = data['img'] if isinstance(data['img'], list) else [data['img']]
    img_tensor = img[0]  # take first scale
    img_meta = data['img_metas']
    if hasattr(img_meta, 'data'):
        img_meta = img_meta.data
    if isinstance(img_meta, dict):
        img_meta = [img_meta]

    with torch.no_grad():
        seg_logit = model.simple_test(img_tensor.unsqueeze(0).cuda(), img_meta, rescale=True)
        if isinstance(seg_logit, list):
            pred_np = seg_logit[0]
        else:
            pred_np = seg_logit

    # Load GT
    ann_dir = osp.join(dataset.data_root, dataset.ann_dir)
    gt_file = osp.join(ann_dir, dataset.img_infos[idx]['ann']['seg_map'])
    gt = imageio.imread(gt_file).astype(np.float32) / 1000.0

    print(f"\n--- Sample {idx} ---")
    print(f"  Pred: shape={pred_np.shape}, mean={pred_np.mean():.4f}, std={pred_np.std():.4f}, "
          f"min={pred_np.min():.4f}, max={pred_np.max():.4f}")
    print(f"  Pred percentiles: p1={np.percentile(pred_np, 1):.4f}, p5={np.percentile(pred_np, 5):.4f}, "
          f"p50={np.percentile(pred_np, 50):.4f}, p95={np.percentile(pred_np, 95):.4f}, p99={np.percentile(pred_np, 99):.4f}")
    print(f"  GT:   shape={gt.shape}, mean={gt.mean():.4f}, std={gt.std():.4f}, "
          f"min={gt.min():.4f}, max={gt.max():.4f}")

    # Per-sample metrics
    pred_valid = np.clip(pred_np.reshape(gt.shape), 1e-3, 10.0)
    gt_valid = np.clip(gt, 1e-3, 10.0)
    valid_mask = gt_valid > 0
    if valid_mask.any():
        pred_v = pred_valid[valid_mask]
        gt_v = gt_valid[valid_mask]
        abs_rel = np.mean(np.abs(pred_v - gt_v) / gt_v)
        rmse = np.sqrt(np.mean((pred_v - gt_v) ** 2))
        ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
        a1 = np.mean(ratio < 1.25)
        a2 = np.mean(ratio < 1.25 ** 2)
        a3 = np.mean(ratio < 1.25 ** 3)
        print(f"  AbsRel={abs_rel:.4f}, RMSE={rmse:.4f}, d1={a1:.4f}, d2={a2:.4f}, d3={a3:.4f}")
        total_pixels += pred_v.size
        all_preds.append(pred_v)
        all_gts.append(gt_v)

if all_preds:
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    print(f"\n=== GLOBAL ({total_pixels} pixels) ===")
    print(f"Pred: mean={all_preds.mean():.4f}, std={all_preds.std():.4f}, min={all_preds.min():.4f}, max={all_preds.max():.4f}")
    print(f"GT:   mean={all_gts.mean():.4f}, std={all_gts.std():.4f}, min={all_gts.min():.4f}, max={all_gts.max():.4f}")
    abs_rel = np.mean(np.abs(all_preds - all_gts) / all_gts)
    rmse = np.sqrt(np.mean((all_preds - all_gts) ** 2))
    ratio = np.maximum(all_preds / all_gts, all_gts / all_preds)
    a1 = np.mean(ratio < 1.25)
    a2 = np.mean(ratio < 1.25 ** 2)
    a3 = np.mean(ratio < 1.25 ** 3)
    print(f"AbsRel={abs_rel:.4f}, RMSE={rmse:.4f}, d1={a1:.4f}, d2={a2:.4f}, d3={a3:.4f}")
