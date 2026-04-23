import cv2
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, '/home/dji/projects/Vim/dep')

# Register custom modules FIRST
from decode_heads import depth_head
from losses import silog_loss
from backbone import cnn_baseline

from mmcv import Config
from mmseg.models import build_segmentor
import mmcv

def depth_to_colormap(depth):
    """Convert depth map to colored image (turbo colormap with dynamic range)"""
    depth = np.squeeze(depth)

    # Dynamic range calculation - exclude outliers
    valid_mask = (depth > 0) & (depth < 100)
    if valid_mask.sum() > 0:
        min_d = np.percentile(depth[valid_mask], 2)
        max_d = np.percentile(depth[valid_mask], 98)
    else:
        min_d = 0.001
        max_d = 10.0

    # Normalize to [0, 1]
    depth_norm = np.clip((depth - min_d) / (max_d - min_d + 1e-6), 0, 1)
    # Apply TURBO colormap for better contrast
    depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return depth_color

def create_depth_video(config_path, checkpoint_path, data_dir, output_video_path, num_samples=100):
    """Create side-by-side video of GT depth vs predicted depth"""

    print("Loading model...")
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.cuda()
    model.eval()

    print(f"Loading dataset from {data_dir}...")
    data_path = Path(data_dir)

    # Match jpg/png by filename stem to avoid mispairing
    jpg_map = {f.stem: f for f in data_path.glob('*.jpg')}
    png_map = {f.stem: f for f in data_path.glob('*.png')}
    common = sorted(jpg_map.keys() & png_map.keys(), key=lambda x: int(x) if x.isdigit() else x)
    common = common[:num_samples]

    if not common:
        print(f"No matched jpg/png pairs found in {data_dir}")
        return

    print(f"Found {len(common)} matched pairs")

    # Video settings
    frame_width = 1024
    frame_height = 512
    fps = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Creating video: {output_video_path}")

    for idx, stem in enumerate(common):
        color_file = jpg_map[stem]
        depth_file = png_map[stem]
        try:
            # Load GT depth
            gt_depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH).astype(np.float32)
            if gt_depth_raw is None:
                print(f"  Skip {idx}: can't load depth")
                continue
            gt_depth = gt_depth_raw / 1000.0  # mm to m

            # Load color image
            color_img = cv2.imread(str(color_file))
            if color_img is None:
                print(f"  Skip {idx}: can't load color")
                continue

            H, W = color_img.shape[:2]
            color_img = cv2.resize(color_img, (512, 512))

            # Prepare input
            color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            color_tensor = torch.from_numpy(color_rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0

            # Predict - just forward pass
            with torch.no_grad():
                backbone_out = model.backbone(color_tensor)
                pred_depth = model.decode_head.forward(backbone_out)

            # Detach and move to CPU, ensure it's a copy
            pred_depth = pred_depth.detach().cpu().numpy().copy()
            if len(pred_depth.shape) == 3:
                pred_depth = pred_depth[0].copy()

            gt_depth = cv2.resize(gt_depth, (512, 512)).copy()

            # Convert to color - ensure output is copied
            gt_color = depth_to_colormap(gt_depth).copy()
            pred_color = depth_to_colormap(pred_depth).copy()

            # Stack side by side
            frame = np.hstack([gt_color, pred_color]).copy()

            # Add labels
            cv2.putText(frame, 'GT Depth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, 'Predicted', (512+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write frame
            out.write(frame)

            # Explicitly clean up GPU memory
            del color_tensor, backbone_out, pred_depth, gt_color, pred_color, frame
            torch.cuda.empty_cache()

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(common)}")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    out.release()
    print(f"✓ Video saved to {output_video_path}")

if __name__ == '__main__':
    config = 'configs/cnn_baseline/depth_cnn_tiny_512_60k_stable.py'
    checkpoint = 'work_dirs/cnn_stable/latest.pth'
    data_dir = '/home/dji/projects/Vim/data/nyu2_train/bedroom_0010_out'
    output = 'depth_comparison.mp4'

    create_depth_video(config, checkpoint, data_dir, output, num_samples=100)
