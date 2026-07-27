"""
V3 Physical Simulator for D-Dimensional RBF Layers

Extends v1 physical_simulator to D dimensions:
  - D instances of differential_pair per particle (linear scaling)
  - D instances of gilbert_square per particle (linear scaling)
  - kirchhoff_sum for Σ dxᵢ² (single, unchanged)
  - bjt_gaussian_kernel, current_mode_normalize, memristor_mvm unchanged
  - Wire-direct inter-layer connections (zero devices)

components.py from v1 is reused as-is — component logic is per-channel,
D-dimensionality is handled by multiple instantiations in the layer.
"""

from .layer_nd import PhysicalRBFLayerND
from .network_nd import PhysicalResidualRBFNetworkND

__all__ = ["PhysicalRBFLayerND", "PhysicalResidualRBFNetworkND"]
