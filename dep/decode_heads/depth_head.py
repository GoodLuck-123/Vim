import torch
import torch.nn as nn
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads import UPerHead


@HEADS.register_module()
class DepthHead(UPerHead):
    """Depth estimation head for Vision Mamba.

    Inherits from UPerHead but outputs single-channel depth prediction
    instead of multi-class segmentation logits.

    Args:
        min_depth (float): Minimum depth value in meters. Default: 1e-3
        max_depth (float): Maximum depth value in meters. Default: 10.0
    """

    def __init__(self, min_depth=1e-3, max_depth=10.0, **kwargs):
        """Initialize DepthHead.

        Forces num_classes=1 for single-channel output.
        """
        kwargs['num_classes'] = 1
        super().__init__(**kwargs)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, inputs):
        """Forward pass returning raw logits (B, 1, H, W)."""
        x = self._transform_inputs(inputs)
        out = self.forward_head(x)
        return out

    def forward_head(self, x):
        """Decode using PPM and convolution layers."""
        # Process multi-scale features through PPM (if used in parent)
        if hasattr(self, 'ppm'):
            ppm_out = self.ppm(x[0])
            x = [ppm_out] + x[1:]

        # Fuse features from different scales
        if hasattr(self, '_forward_ppm'):
            # For UPerHead compatibility
            output = self.decode_head(x)
        else:
            # Simple decode: use the first feature
            output = self.cls_seg(x[0])

        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward pass for training.

        Note: gt_semantic_seg parameter actually contains gt_depth (by design).
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward pass for testing.

        Returns depth values clipped to [min_depth, max_depth].
        """
        seg_logits = self.forward(inputs)
        # Apply sigmoid and scale to [min_depth, max_depth]
        depth = seg_logits.sigmoid() * (self.max_depth - self.min_depth) + self.min_depth
        return depth
