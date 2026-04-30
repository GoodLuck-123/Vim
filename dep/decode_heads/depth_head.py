import torch
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads import UPerHead


@HEADS.register_module()
class DepthHead(UPerHead):
    """Depth estimation head for Vision Mamba.

    Inherits from UPerHead but outputs single-channel depth prediction
    instead of multi-class segmentation logits.
    """

    def __init__(self, min_depth=1e-3, max_depth=10.0, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(**kwargs)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, inputs):
        """Forward using full UPerHead multi-scale fusion, returns (B, H, W)."""
        from torch.nn import functional as F
        out = super().forward(inputs)  # (B, 1, H/4, W/4) - full FPN+PSP fusion
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        # Softplus ensures positive values for log-space SILogLoss
        out = F.softplus(out) + self.min_depth
        return out.squeeze(1)          # (B, H, W)

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

        loss_decode = self.loss_decode(seg_logits, gt_semantic_seg)
        return dict(loss_depth=loss_decode)

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward pass for testing.

        Returns depth values clamped to [min_depth, max_depth].
        No sigmoid — depth is raw regression, not classification.
        """
        import torch
        seg_logits = self.forward(inputs)
        depth = torch.clamp(seg_logits, self.min_depth, self.max_depth)
        return depth
