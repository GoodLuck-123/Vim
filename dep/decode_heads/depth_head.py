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

        # Add adapter to project input features to the expected channel size
        in_ch = kwargs.get('in_channels', [512])[0]  # First input channel
        self.adapter = nn.Sequential(
            nn.Conv2d(in_ch, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        """Forward pass returning raw depth (B, H, W)."""
        x = self._transform_inputs(inputs)

        # Project features to cls_seg's expected channel size
        feat = self.adapter(x[0])  # (B, channels, H/4, W/4)
        out = self.cls_seg(feat)   # (B, 1, H/4, W/4)

        # Upsample to original image size (factor of 4)
        from torch.nn import functional as F
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        # Squeeze to (B, H, W) for loss compatibility
        out = out.squeeze(1)

        return out

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward pass for training.

        Note: gt_semantic_seg parameter actually contains gt_depth (by design).
        """
        seg_logits = self.forward(inputs)  # (B, H, W)

        # Unsqueeze back to (B, 1, H, W) for loss function
        seg_logits = seg_logits.unsqueeze(1)

        # Squeeze depth map if needed: (B, 1, H, W) -> (B, H, W)
        if gt_semantic_seg.dim() == 4 and gt_semantic_seg.shape[1] == 1:
            gt_semantic_seg = gt_semantic_seg.squeeze(1)

        # Call loss directly (bypass parent's resize logic)
        loss_decode = self.loss_decode(seg_logits, gt_semantic_seg)
        losses = dict(loss_depth=loss_decode)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward pass for testing.

        Returns depth values clipped to [min_depth, max_depth].
        """
        seg_logits = self.forward(inputs)
        # Apply sigmoid and scale to [min_depth, max_depth]
        depth = seg_logits.sigmoid() * (self.max_depth - self.min_depth) + self.min_depth
        return depth
