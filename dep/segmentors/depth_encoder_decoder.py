import torch
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors import EncoderDecoder


@SEGMENTORS.register_module()
class DepthEncoderDecoder(EncoderDecoder):
    """EncoderDecoder for dense depth regression.

    Overrides simple_test to skip segmentation-specific threshold/argmax
    and return raw depth values.
    """

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test: backbone + decode_head forward, bypassing encode_decode.

        We bypass mmseg's encode_decode because it calls resize() which expects
        4D tensor (B, C, H, W), but our depth head returns (B, H, W).
        """
        x = self.extract_feat(img)
        seg_logit = self.decode_head.forward(x)  # (B, H, W)
        seg_logit = torch.clamp(seg_logit, self.decode_head.min_depth,
                                self.decode_head.max_depth)
        if seg_logit.dim() == 2:
            seg_logit = seg_logit.unsqueeze(0)

        # Rescale to original image size if requested
        if rescale and img_meta is not None:
            from torch.nn import functional as F
            if isinstance(img_meta, list):
                img_meta = img_meta[0]
            ori_shape = img_meta.get('ori_shape', img_meta.get('img_shape'))
            if ori_shape is not None and seg_logit.shape[-2:] != tuple(ori_shape[:2]):
                # Ensure 4D: (N, C, H, W)
                while seg_logit.dim() < 4:
                    seg_logit = seg_logit.unsqueeze(0)
                seg_logit = F.interpolate(seg_logit, size=tuple(ori_shape[:2]),
                                          mode='bilinear', align_corners=False)
                seg_logit = seg_logit.squeeze(0).squeeze(0)  # back to (H, W)

        seg_logit = seg_logit.cpu().numpy()
        return [seg_logit]

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations."""
        from mmseg.ops import resize
        seg_logits = []
        for img, img_meta in zip(imgs, img_metas):
            logit = self.inference(img, img_meta, rescale)
            seg_logits.append(logit)
        seg_logit = sum(seg_logits) / len(seg_logits)
        seg_logit = seg_logit.squeeze(1).cpu().numpy()
        seg_logit = list(seg_logit)
        return seg_logit
