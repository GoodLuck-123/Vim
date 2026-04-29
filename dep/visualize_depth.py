"""Depth visualization: image-pair comparison or single-video inference."""
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path

# Register custom modules BEFORE mmseg model builder
import segmentors
from decode_heads import depth_head
from losses import silog_loss
from datasets import nyu_depth_v2
from datasets.pipelines import depth_loading
try:
    from backbone import vim
except Exception:
    pass
from backbone import cnn_baseline

from mmcv import Config
from mmseg.models import build_segmentor

# ImageNet normalization (matches training config)
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def load_model(config_path, checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.cuda()
    model.eval()
    return model, cfg


def preprocess_frame(bgr, size=(512, 512)):
    """Preprocess BGR frame for model: resize, RGB, normalize, NCHW tensor."""
    img = cv2.resize(bgr, size)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - IMG_MEAN) / IMG_STD
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()
    return tensor


def depth_to_colormap(depth):
    """Convert depth map to colored image (turbo colormap with dynamic range)."""
    depth = np.squeeze(depth).astype(np.float32)
    valid = (depth > 0) & (depth < 100)
    if valid.sum() > 0:
        vmin = np.percentile(depth[valid], 2)
        vmax = np.percentile(depth[valid], 98)
    else:
        vmin, vmax = 0.001, 10.0
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def infer(model, img_tensor):
    """Run inference, return depth map (H, W) in meters."""
    with torch.no_grad():
        x = model.backbone(img_tensor)
        pred = model.decode_head.forward(x)  # (B, H, W) or (B, 1, H, W)
    pred = pred.detach().cpu().numpy().squeeze()
    return pred


# ---- Mode: image pairs (GT vs Predicted side-by-side) ----

def create_comparison_video(config_path, checkpoint_path, data_dir, output_video):
    model, _ = load_model(config_path, checkpoint_path)
    data_path = Path(data_dir)

    jpg_map = {f.stem: f for f in data_path.glob('*.jpg')}
    png_map = {f.stem: f for f in data_path.glob('*.png')}
    common = sorted(jpg_map.keys() & png_map.keys(),
                    key=lambda x: int(x) if x.isdigit() else x)

    if not common:
        print(f"No matched jpg/png pairs found in {data_dir}")
        return

    print(f"Found {len(common)} matched pairs")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 5, (1024, 512))

    for idx, stem in enumerate(common):
        try:
            # Load GT (with uint8/uint16 detection)
            gt_raw = cv2.imread(str(png_map[stem]), cv2.IMREAD_ANYDEPTH)
            if gt_raw is None:
                continue
            if gt_raw.dtype == np.uint8:
                gt_depth = gt_raw.astype(np.float32) / 100.0
            else:
                gt_depth = gt_raw.astype(np.float32) / 1000.0

            # Load and preprocess color image
            bgr = cv2.imread(str(jpg_map[stem]))
            if bgr is None:
                continue
            tensor = preprocess_frame(bgr)

            # Predict
            pred = infer(model, tensor)

            # Resize GT to match prediction
            gt_depth = cv2.resize(gt_depth, (512, 512))

            gt_color = depth_to_colormap(gt_depth)
            pred_color = depth_to_colormap(pred)

            frame = np.hstack([gt_color, pred_color])
            cv2.putText(frame, 'GT Depth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, 'Predicted', (522, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(common)}")
        except Exception as e:
            print(f"  Error on {stem}: {e}")
            import traceback; traceback.print_exc()
            continue

    out.release()
    print(f"Video saved to {output_video}")


# ---- Mode: video inference (RGB | Depth side-by-side) ----

def create_video_depth(config_path, checkpoint_path, input_video, output_video):
    model, _ = load_model(config_path, checkpoint_path)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Cannot open video: {input_video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input: {orig_w}x{orig_h}, {fps:.1f} fps, {total_frames} frames")

    # Output: original | depth side-by-side, 512px height
    panel_h = 512
    panel_w = int(orig_w * panel_h / orig_h)
    out_w = panel_w * 2
    out_h = panel_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))
    print(f"Output: {out_w}x{out_h} @ {fps:.1f}fps, writing to {output_video}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_resized = cv2.resize(frame, (panel_w, out_h))
        tensor = preprocess_frame(frame)
        pred = infer(model, tensor)
        pred_color = depth_to_colormap(pred)
        pred_color = cv2.resize(pred_color, (panel_w, out_h))

        frame_out = np.hstack([orig_resized, pred_color])
        cv2.putText(frame_out, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_out, 'Depth', (panel_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame_out)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    print(f"Video saved to {output_video} ({frame_idx} frames)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth visualization')
    parser.add_argument('--config', default='configs/vim/depth/depth_vim_tiny_24_512_60k_single.py',
                        help='Model config path')
    parser.add_argument('--checkpoint', default='work_dirs/depth_vim_tiny_24_512_60k/best_AbsRel_iter_59000.pth',
                        help='Checkpoint path')
    parser.add_argument('--mode', choices=['pairs', 'video'], default='video',
                        help='pairs: GT-vs-Pred from jpg/png dir; video: infer from video file')
    parser.add_argument('--input', default='../data/test01.mp4',
                        help='Input: data directory (pairs mode) or video file (video mode)')
    parser.add_argument('--output', default='depth_output.mp4',
                        help='Output video path')

    args = parser.parse_args()

    if args.mode == 'pairs':
        create_comparison_video(args.config, args.checkpoint, args.input, args.output)
    else:
        create_video_depth(args.config, args.checkpoint, args.input, args.output)
