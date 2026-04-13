import os.path as osp
from mmseg.datasets.builder import PIPELINES


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

        # Get the depth file path from the annotation info
        # The img_infos from NYUDepthV2Dataset contains the full relative path
        ann_info = results.get('ann_info', {})
        seg_map = ann_info.get('seg_map')

        if seg_map is None:
            raise ValueError(f'No seg_map in ann_info. Available keys: {list(ann_info.keys())}')

        # Construct full depth file path
        data_root = results.get('data_root', '')
        if data_root:
            depth_path = osp.join(data_root, seg_map)
        else:
            depth_path = seg_map

        # Load depth map
        if depth_path.endswith('.png'):
            # Load 16-bit PNG (mm to m)
            depth = imageio.imread(depth_path).astype('float32') / 1000.0
        elif depth_path.endswith('.npy'):
            import numpy as np
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

        return results

