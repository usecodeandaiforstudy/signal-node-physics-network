"""
V3 Forward-Only Training Methods

Adaptations of v1 training methods for D-dimensional particles:
  - ProtoGuidedTrainerND: Pure physical-domain prototype-guided training
  - HiMALND: HiMAL dual-network layer-wise training (D-dimensional)
"""

from .proto_nd import ProtoGuidedTrainerND
from .himal_nd import HiMALND

__all__ = ["ProtoGuidedTrainerND", "HiMALND"]
