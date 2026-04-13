import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES


@LOSSES.register_module()
class SILogLoss(nn.Module):
    """Scale-Invariant Log Loss for depth estimation (Eigen et al.).

    This is a standard loss for monocular depth estimation that is
    scale-invariant in the log space.

    Args:
        variance_focus (float): Weight parameter λ for variance term.
            Default: 0.85
        loss_weight (float): Weight of this loss. Default: 1.0
    """

    def __init__(self, variance_focus=0.85, loss_weight=1.0):
        """Initialize SILogLoss."""
        super().__init__()
        self.variance_focus = variance_focus
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        """Compute SILog loss.

        Args:
            pred (Tensor): Predicted depth, shape (B, 1, H, W) or (B, H, W)
            target (Tensor): Ground truth depth, same shape as pred
            weight: Unused, for compatibility
            avg_factor: Unused, for compatibility

        Returns:
            Tensor: Scalar loss value
        """
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        # Valid mask: depth > 0
        valid_mask = target > 0

        if not valid_mask.any():
            # If no valid pixels, return zero loss
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Extract valid predictions and targets
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        # Ensure positive values for log
        pred_valid = torch.clamp(pred_valid, min=1e-6)
        target_valid = torch.clamp(target_valid, min=1e-6)

        # Compute log differences
        log_diff = torch.log(pred_valid) - torch.log(target_valid)

        # SILog loss: sqrt(mean(d^2) - λ * mean(d)^2) * 10
        loss = torch.sqrt(
            (log_diff ** 2).mean() - self.variance_focus * (log_diff.mean() ** 2)
        ) * 10.0

        return self.loss_weight * loss


@LOSSES.register_module()
class BerHuLoss(nn.Module):
    """BerHu (Reverse Huber) Loss for depth estimation.

    Often used for indoor depth datasets like NYU Depth v2.
    More robust to outliers than L1 loss.

    Args:
        threshold (float): Threshold c for Huber. Default: 0.2
        loss_weight (float): Weight of this loss. Default: 1.0
    """

    def __init__(self, threshold=0.2, loss_weight=1.0):
        """Initialize BerHuLoss."""
        super().__init__()
        self.threshold = threshold
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        """Compute BerHu loss.

        Args:
            pred (Tensor): Predicted depth, shape (B, 1, H, W) or (B, H, W)
            target (Tensor): Ground truth depth, same shape as pred
            weight: Unused, for compatibility
            avg_factor: Unused, for compatibility

        Returns:
            Tensor: Scalar loss value
        """
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        # Valid mask: depth > 0
        valid_mask = target > 0

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Extract valid predictions and targets
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        diff = torch.abs(pred_valid - target_valid)
        delta = self.threshold * torch.max(diff).item()

        # BerHu = L1 for |diff| <= delta, else (|diff|^2 + delta^2) / (2*delta)
        mask_l1 = diff <= delta
        mask_l2 = ~mask_l1

        loss = torch.sum(diff[mask_l1]) + torch.sum(
            (diff[mask_l2] ** 2 + delta ** 2) / (2 * delta)
        )
        loss = loss / len(diff)

        return self.loss_weight * loss
