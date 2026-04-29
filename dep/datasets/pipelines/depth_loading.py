import os.path as osp
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class DepthFormatBundle:
    """Simplistic formatter for depth estimation. Converts tensors to torch.Tensor."""

    def __call__(self, results):
        """Format results for depth estimation."""
        import torch
        import numpy as np

        if 'img' in results:
            if isinstance(results['img'], np.ndarray):
                results['img'] = torch.from_numpy(results['img'].transpose(2, 0, 1).copy()).contiguous()

        if 'gt_semantic_seg' in results:
            if isinstance(results['gt_semantic_seg'], np.ndarray):
                results['gt_semantic_seg'] = torch.from_numpy(results['gt_semantic_seg'][np.newaxis, ...].copy()).contiguous()

        # Clean up intermediate keys for collate
        keys_to_keep = ['img', 'gt_semantic_seg']
        keys_to_remove = [k for k in results.keys() if k not in keys_to_keep]
        for k in keys_to_remove:
            del results[k]

        return results


@PIPELINES.register_module()
class LoadDepthAnnotation:
    """Load depth annotations from file.

    Args:
        reduce_zero_label (bool): Unused, for compatibility with seg pipeline
    """

    def __init__(self, reduce_zero_label=False):
        """Initialize LoadDepthAnnotation."""
        pass

    def __call__(self, results):
        """Load depth annotation.

        Args:
            results (dict): Data pipeline results

        Returns:
            dict: Updated results with gt_semantic_seg containing depth map
        """
        import imageio
        import numpy as np

        # Get the depth file path from the annotation info
        # The img_infos from NYUDepthV2Dataset contains the full relative path
        ann_info = results.get('ann_info', {})
        seg_map = ann_info.get('seg_map')

        if seg_map is None:
            raise ValueError(f'No seg_map in ann_info. Available keys: {list(ann_info.keys())}')

        # Construct full depth file path using seg_prefix (set by CustomDataset)
        seg_prefix = results.get('seg_prefix', '')
        if seg_prefix:
            depth_path = osp.join(seg_prefix, seg_map)
        else:
            depth_path = seg_map

        # Load depth map
        if depth_path.endswith('.png'):
            depth_raw = imageio.imread(depth_path)
            # Detect dtype: uint8 = cm (/100), uint16 = mm (/1000)
            if depth_raw.dtype == np.uint8:
                depth = depth_raw.astype('float32') / 100.0
            else:
                depth = depth_raw.astype('float32') / 1000.0
        elif depth_path.endswith('.npy'):
            depth = np.load(depth_path).astype('float32')
        else:
            raise ValueError(f'Unsupported depth format: {depth_path}')

        # Store in gt_semantic_seg for compatibility with EncoderDecoder
        results['gt_semantic_seg'] = depth

        # Add to seg_fields for augmentation compatibility
        if 'seg_fields' not in results:
            results['seg_fields'] = []
        if 'gt_semantic_seg' not in results['seg_fields']:
            results['seg_fields'].append('gt_semantic_seg')

        # Initialize meta fields that may be accessed by downstream transforms
        if 'img_meta' not in results:
            results['img_meta'] = {}
        if 'flip' not in results:
            results['flip'] = False

        return results


@PIPELINES.register_module()
class RandomGaussianBlur:
    """Apply Gaussian blur to RGB image, leaving depth unchanged.

    Forces the model to rely on geometric structure rather than fine texture,
    which is desirable for depth estimation.

    Args:
        kernel_sizes (list): Candidate kernel sizes (odd). Default: [3, 5, 7, 9]
        sigma_min (float): Min sigma. Default: 0.1
        sigma_max (float): Max sigma. Default: 2.0
        prob (float): Probability of applying. Default: 0.5
    """

    def __init__(self, kernel_sizes=None, sigma_min=0.1, sigma_max=2.0, prob=0.5):
        self.kernel_sizes = kernel_sizes or [3, 5, 7, 9]
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = prob

    def __call__(self, results):
        import numpy as np
        import cv2
        if np.random.random() < self.prob:
            k = int(np.random.choice(self.kernel_sizes))
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)
            results['img'] = cv2.GaussianBlur(results['img'], (k, k), sigma)
        return results


@PIPELINES.register_module()
class RandomGaussianNoise:
    """Add Gaussian noise to RGB image, leaving depth unchanged.

    Simulates sensor noise to improve robustness. Forces the model to not
    overfit to pixel-perfect RGB values.

    Args:
        std_min (float): Min noise std (in [0, 255] scale). Default: 3.0
        std_max (float): Max noise std. Default: 15.0
        prob (float): Probability of applying. Default: 0.5
    """

    def __init__(self, std_min=3.0, std_max=15.0, prob=0.5):
        self.std_min = std_min
        self.std_max = std_max
        self.prob = prob

    def __call__(self, results):
        import numpy as np
        if np.random.random() < self.prob:
            std = np.random.uniform(self.std_min, self.std_max)
            noise = np.random.randn(*results['img'].shape).astype(np.float32) * std
            img = results['img'].astype(np.float32) + noise
            results['img'] = np.clip(img, 0, 255).astype(np.uint8)
        return results


@PIPELINES.register_module()
class RandomDepthScale:
    """Randomly scale depth values to simulate different scene scales.

    Multiplies GT depth by a random factor in [scale_min, scale_max].
    This helps the model generalize beyond the limited depth range of uint8
    training data (0-2.55m) by virtually stretching/compressing scene depth.

    Args:
        scale_min (float): Minimum scale factor. Default: 0.8
        scale_max (float): Maximum scale factor. Default: 1.2
        prob (float): Probability of applying the transform. Default: 0.5
    """

    def __init__(self, scale_min=0.8, scale_max=1.2, prob=0.5):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.prob = prob

    def __call__(self, results):
        import numpy as np
        if np.random.random() < self.prob:
            scale = np.random.uniform(self.scale_min, self.scale_max)
            depth = results['gt_semantic_seg'].copy()
            depth = depth * scale
            results['gt_semantic_seg'] = depth
        return results

