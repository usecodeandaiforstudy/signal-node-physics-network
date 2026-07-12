"""
Physical component simulation module

Provides a physical simulator compatible with the rbf_network parameter structure,
replacing standard numerical operations with non-ideal analog circuit elements.
"""

from .components import (
    differential_pair,
    gilbert_square,
    bjt_gaussian_kernel,
    current_mode_normalize,
    memristor_mvm,
    kirchhoff_sum,
    NoiseConfig,
    V_T, V_SUPPLY, I_BIAS_MAX,
)
from .layer import PhysicalRBFLayer
from .network import PhysicalResidualRBFNetwork

__all__ = [
    "differential_pair", "gilbert_square", "bjt_gaussian_kernel",
    "current_mode_normalize", "memristor_mvm", "kirchhoff_sum",
    "NoiseConfig",
    "V_T", "V_SUPPLY", "I_BIAS_MAX",
    "PhysicalRBFLayer", "PhysicalResidualRBFNetwork",
]
