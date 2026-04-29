import os
import os.path as osp
import numpy as np
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class NYUDepthV2Dataset(CustomDataset):
    """NYU Depth v2 dataset for monocular depth estimation.

    Dataset paper: https://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf

    Data structure:
    - nyu2_train/
      - scene_0001/
        - 1.jpg (RGB image)
        - 1.png (depth annotation in mm, 16-bit)
        - ...
    - nyu2_test/
      - ...

    Args:
        pipeline (list): Processing pipeline
        data_root (str): Data root directory containing nyu2_train/ and nyu2_test/
        img_dir (str): Image subdirectory name (e.g., 'nyu2_train')
        ann_dir (str): Not used for NYUv2, annotations are in same directory as images
        **kwargs: Other arguments passed to CustomDataset
    """

    CLASSES = ('depth',)
    PALETTE = None

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None):
        """Load annotations. Supports scene-subdir layout and flat layout."""
        img_infos = []
        img_dir_path = osp.join(self.data_root, img_dir)

        if not osp.exists(img_dir_path):
            raise FileNotFoundError(f'Directory not found: {img_dir_path}')

        entries = os.listdir(img_dir_path)
        has_subdirs = any(osp.isdir(osp.join(img_dir_path, e)) for e in entries)

        if has_subdirs:
            # Layout 1: scene subdirectories with .jpg/.png pairs
            for scene_name in sorted(os.listdir(img_dir_path)):
                scene_path = osp.join(img_dir_path, scene_name)
                if not osp.isdir(scene_path):
                    continue
                for img_file in sorted(os.listdir(scene_path)):
                    if not img_file.endswith('.jpg'):
                        continue
                    depth_file = img_file.replace('.jpg', '.png')
                    depth_path = osp.join(scene_path, depth_file)
                    if not osp.exists(depth_path):
                        continue
                    img_info = dict(
                        filename=osp.join(scene_name, img_file),
                        ann=dict(seg_map=osp.join(scene_name, depth_file))
                    )
                    img_infos.append(img_info)
        else:
            # Layout 2: flat files — match *_colors.png with *_depth.png
            for rgb_file in sorted(entries):
                if not rgb_file.endswith('_colors.png'):
                    continue
                depth_file = rgb_file.replace('_colors.png', '_depth.png')
                depth_path = osp.join(img_dir_path, depth_file)
                if not osp.exists(depth_path):
                    continue
                img_info = dict(
                    filename=rgb_file,
                    ann=dict(seg_map=depth_file)
                )
                img_infos.append(img_info)

        print(f'Loaded {len(img_infos)} samples from {img_dir}')
        return img_infos

    def evaluate(self, results, metric='depth', **eval_kwargs):
        """Evaluate depth predictions.

        Args:
            results (list): Prediction results, each item is a depth map
            metric (str): Metric name, default 'depth'
            **eval_kwargs: Other evaluation arguments

        Returns:
            dict: Evaluation metrics including AbsRel, RMSE, delta_1, etc.
        """
        eval_metrics = {}

        # Load ground truth depths
        gt_depths = []
        for idx in range(len(self)):
            gt_depth_file = osp.join(
                self.data_root, self.ann_dir, self.img_infos[idx]['ann']['seg_map']
            )
            gt_depth = self._load_depth(gt_depth_file)
            gt_depths.append(gt_depth)

        # Compute metrics for each prediction
        abs_rel_list = []
        sqr_rel_list = []
        rmse_list = []
        rmse_log_list = []
        a1_list = []  # delta < 1.25
        a2_list = []  # delta < 1.25^2
        a3_list = []  # delta < 1.25^3

        for i, pred_depth in enumerate(results):
            if isinstance(pred_depth, dict) and 'depth' in pred_depth:
                pred_depth = pred_depth['depth']

            if isinstance(pred_depth, np.ndarray):
                pred_depth = pred_depth.squeeze()
            else:
                pred_depth = pred_depth.squeeze().cpu().numpy()

            gt_depth = gt_depths[i].squeeze()

            # Valid mask (depth > 0)
            valid_mask = gt_depth > 0

            if not valid_mask.any():
                continue

            pred_valid = pred_depth[valid_mask]
            gt_valid = gt_depth[valid_mask]

            # Clip to valid range
            pred_valid = np.clip(pred_valid, 1e-3, 10.0)
            gt_valid = np.clip(gt_valid, 1e-3, 10.0)

            # Per-image LS alignment in log space: solve min_{s,t} ||s*log(pred)+t - log(gt)||
            log_pred = np.log(pred_valid)
            log_gt = np.log(gt_valid)
            A = np.stack([log_pred, np.ones_like(log_pred)], axis=1)  # (N, 2)
            st, _, _, _ = np.linalg.lstsq(A, log_gt, rcond=None)
            pred_aligned = np.exp(st[0] * log_pred + st[1])
            pred_aligned = np.clip(pred_aligned, 1e-3, 10.0)

            # Absolute Relative Error
            abs_rel = np.mean(np.abs(pred_aligned - gt_valid) / gt_valid)
            abs_rel_list.append(abs_rel)

            # Squared Relative Error
            sqr_rel = np.mean(((pred_aligned - gt_valid) ** 2) / (gt_valid ** 2))
            sqr_rel_list.append(sqr_rel)

            # RMSE
            rmse = np.sqrt(np.mean((pred_aligned - gt_valid) ** 2))
            rmse_list.append(rmse)

            # RMSE in log space
            rmse_log = np.sqrt(np.mean((np.log(pred_aligned) - np.log(gt_valid)) ** 2))
            rmse_log_list.append(rmse_log)

            # Delta thresholds
            ratio = np.maximum(pred_aligned / gt_valid, gt_valid / pred_aligned)
            a1 = np.mean(ratio < 1.25)
            a1_list.append(a1)

            a2 = np.mean(ratio < 1.25 ** 2)
            a2_list.append(a2)

            a3 = np.mean(ratio < 1.25 ** 3)
            a3_list.append(a3)

        # Average metrics
        eval_metrics['AbsRel'] = np.mean(abs_rel_list)
        eval_metrics['SqRel'] = np.mean(sqr_rel_list)
        eval_metrics['RMSE'] = np.mean(rmse_list)
        eval_metrics['RMSElog'] = np.mean(rmse_log_list)
        eval_metrics['delta_1'] = np.mean(a1_list)
        eval_metrics['delta_2'] = np.mean(a2_list)
        eval_metrics['delta_3'] = np.mean(a3_list)

        return eval_metrics

    def _load_depth(self, depth_file):
        """Load depth map from file.

        Args:
            depth_file (str): Path to depth map file

        Returns:
            np.ndarray: Depth map
        """
        import imageio

        if depth_file.endswith('.png'):
            depth_raw = imageio.imread(depth_file)
            if depth_raw.dtype == np.uint8:
                depth = depth_raw.astype(np.float32) / 100.0   # cm to m
            else:
                depth = depth_raw.astype(np.float32) / 1000.0  # mm to m
        elif depth_file.endswith('.npy'):
            depth = np.load(depth_file).astype(np.float32)
        else:
            raise ValueError(f'Unsupported depth file format: {depth_file}')

        return depth

