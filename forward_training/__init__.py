"""
Forward-only training methods

Provides 3 training methods that do not rely on backpropagation:
  - SPSA zeroth-order optimization
  - HiMAL layer-wise dual-network training
  - Digital twin surrogate gradient training
"""

from .spsa import SPSAOptimizer, spsa_train_layer
from .himal import HiMALTrainer
from .twin import DigitalTwinTrainer

__all__ = [
    "SPSAOptimizer", "spsa_train_layer",
    "HiMALTrainer",
    "DigitalTwinTrainer",
]
