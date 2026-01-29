"""Loss function implementations for imbalanced learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustmentLoss(nn.Module):
    """Logit Adjustment Loss for long-tailed recognition."""

    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        # Numerical stability
        cls_num_list = torch.tensor(cls_num_list).float().cuda()
        prior = cls_num_list / (cls_num_list.sum() + 1e-12)
        self.logit_adjust = tau * torch.log(prior + 1e-12)

    def forward(self, logits, target):
        # Ensure logit_adjust is on the same device as logits
        if self.logit_adjust.device != logits.device:
            self.logit_adjust = self.logit_adjust.to(logits.device)
        logits_adj = logits - self.logit_adjust
        return F.cross_entropy(logits_adj, target)


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax Loss for long-tailed recognition."""

    def __init__(self, cls_num_list):
        super().__init__()
        self.sample_per_cls = torch.tensor(cls_num_list).float().cuda()

    def forward(self, logits, target):
        if self.sample_per_cls.device != logits.device:
            self.sample_per_cls = self.sample_per_cls.to(logits.device)

        spc = self.sample_per_cls.unsqueeze(0).expand(logits.shape[0], -1)
        logits_adj = logits + torch.log(spc + 1e-12)
        return F.cross_entropy(logits_adj, target)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
