"""KKT-Guided SVM Optimizer Package"""

__version__ = "1.0.0"
__author__ = "Fraud Detection Research Team"

from .core.kkt_conditions import KKTConditions, KKTMonitor
from .core.optimizer import AcceleratedGradientOptimizer
from .models.svm_model import KKTGuidedSVM

__all__ = [
    'KKTConditions',
    'KKTMonitor',
    'AcceleratedGradientOptimizer',
    'KKTGuidedSVM'
]
