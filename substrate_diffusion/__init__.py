"""
Diffusion-field substrate for the RBF particle network.

Second physical instantiation of the same five-primitive mapping
(movable source | distance metric | decaying kernel | superposable readout |
tunable strength).  See substrate_diffusion/components.py for the physics and
the honest list of physical constraints.

    from substrate_diffusion import DiffusionRBFLayerND, DiffusionSubstrate, PRESETS

The layer is interface-compatible with simulator.layer_nd.PhysicalRBFLayerND:
same parameter vector layout, same forward(X) -> (out_gauss, out_act) contract,
same from_digital() export, so trained parameters move between substrates
unchanged.
"""

from .components import (                       # noqa: F401
    DiffusionNoiseConfig,
    DiffusionSubstrate,
    PRESETS,
    analytic_gaussian_1d,
    green_prefactor,
    mirror_positions,
    pulse_time,
    quantize_position,
    readout,
    solve_heat_1d,
    source_response,
    unit_peak_strength,
)
from .layer import DiffusionRBFLayerND, KERNELS   # noqa: F401

__all__ = [
    "DiffusionRBFLayerND",
    "DiffusionSubstrate",
    "DiffusionNoiseConfig",
    "PRESETS",
    "KERNELS",
    "pulse_time",
    "green_prefactor",
    "source_response",
    "unit_peak_strength",
    "solve_heat_1d",
    "analytic_gaussian_1d",
    "quantize_position",
    "readout",
    "mirror_positions",
]
