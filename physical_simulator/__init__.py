"""
Physical component simulation module

Provides a physical simulator compatible with the rbf_network parameter structure,
replacing standard numerical operations with non-ideal analog circuit elements.
"""

from .components import (
    differential_pair,
    gilbert_square,
    bjt_gaussian_kernel,
    bjt_log_amplifier,
    gilbert_divide,
    current_mode_normalize,
    memristor_mvm,
    kirchhoff_sum,
    # New: ADC and analog gradient components
    analog_adc,
    analog_subtractor,
    analog_squarer,
    analog_summer,
    analog_loss_readout,
    analog_gilbert_multiplier,
    analog_capacitor_integrator,
    analog_gradient_mac,
    analog_center_gradient,
    analog_width_gradient,
    analog_weight_gradient,
    NoiseConfig,
    V_T, V_SUPPLY, I_BIAS_MAX,
)
from .layer import PhysicalRBFLayer
from .network import PhysicalResidualRBFNetwork

__all__ = [
    # Existing components
    "differential_pair", "gilbert_square", "bjt_gaussian_kernel",
    "bjt_log_amplifier", "gilbert_divide",
    "current_mode_normalize", "memristor_mvm", "kirchhoff_sum",
    # New: ADC and analog loss/gradient
    "analog_adc",
    "analog_subtractor", "analog_squarer", "analog_summer",
    "analog_loss_readout",
    "analog_gilbert_multiplier", "analog_capacitor_integrator",
    "analog_gradient_mac",
    "analog_center_gradient", "analog_width_gradient", "analog_weight_gradient",
    "NoiseConfig",
    "V_T", "V_SUPPLY", "I_BIAS_MAX",
    "PhysicalRBFLayer", "PhysicalResidualRBFNetwork",
]
