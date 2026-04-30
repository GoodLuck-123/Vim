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
        self.loss_name = 'loss_depth'

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

        # SILog loss: sqrt(mean(d^2) - λ * mean(d)^2)
        # Note: Original paper multiplies by 10, but this can cause numerical instability
        # in deep networks. We use scale_factor=1.0 for numerical stability.
        mean_squared = (log_diff ** 2).mean()
        squared_mean = (log_diff.mean() ** 2)
        variance_term = mean_squared - self.variance_focus * squared_mean

        # Clamp to prevent sqrt of negative numbers due to floating point errors
        variance_term = torch.clamp(variance_term, min=1e-8)

        loss = torch.sqrt(variance_term)
        # Apply scale factor (original: 10.0, stable version: 1.0)
        loss = loss * 1.0

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
        self.loss_name = 'loss_depth'

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


@LOSSES.register_module()
class EdgeLoss(nn.Module):
    """Gradient-based edge loss for depth estimation.

    Penalizes differences in spatial gradients between predicted and GT depth.
    Encourages the model to predict sharp object boundaries where GT has them,
    and smooth regions where GT is flat.

    Uses 3x3 Sobel filters to compute x/y gradients, then L1 loss between
    predicted and GT gradient magnitudes.

    Args:
        loss_weight (float): Weight of this loss. Default: 1.0
        scale (int): Scale at which to compute gradients. Default: 1 (full res)
            Set >1 to downsample first (handles multi-scale edge alignment).
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name = 'loss_edge'
        # Sobel kernels
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                 dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                 dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        # Valid mask
        valid_mask = target > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Ensure 4D for conv2d: (B, 1, H, W)
        p = pred.unsqueeze(1)
        t = target.unsqueeze(1)

        # Pad for same-size output
        p = torch.nn.functional.pad(p, (1, 1, 1, 1), mode='replicate')
        t = torch.nn.functional.pad(t, (1, 1, 1, 1), mode='replicate')

        # Compute gradients using depthwise conv
        grad_px = torch.nn.functional.conv2d(p, self.kernel_x)
        grad_py = torch.nn.functional.conv2d(p, self.kernel_y)
        grad_tx = torch.nn.functional.conv2d(t, self.kernel_x)
        grad_ty = torch.nn.functional.conv2d(t, self.kernel_y)

        # Gradient magnitude difference
        mag_p = torch.sqrt(grad_px ** 2 + grad_py ** 2 + 1e-6)
        mag_t = torch.sqrt(grad_tx ** 2 + grad_ty ** 2 + 1e-6)

        # Only compute on valid pixels
        vm = valid_mask.unsqueeze(1).float()

        edge_loss = (torch.abs(mag_p - mag_t) * vm).sum() / (vm.sum() + 1e-6)

        return self.loss_weight * edge_loss
