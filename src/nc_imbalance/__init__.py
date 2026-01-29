"""Neural Collapse in Imbalanced Learning.

This package provides tools and experiments for studying neural collapse
phenomena in class-imbalanced learning scenarios.
"""

__version__ = "1.0.0"

from .models.architectures import ModelFactory, get_feature
from .data.imbalanced_dataset import ImbalancedDatasetGenerator
from .analysis.nc_metrics import NCAnalyzer
from .training.losses import (
    LogitAdjustmentLoss,
    BalancedSoftmaxLoss,
    FocalLoss
)

__all__ = [
    "ModelFactory",
    "get_feature",
    "ImbalancedDatasetGenerator",
    "NCAnalyzer",
    "LogitAdjustmentLoss",
    "BalancedSoftmaxLoss",
    "FocalLoss",
]
