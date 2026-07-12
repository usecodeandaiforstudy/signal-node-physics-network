"""
Physical ResidualRBFNetwork — 3-layer residual RBF particle network physical simulation

Shares the same parameter structure as rbf_network.core.ResidualRBFNetwork,
but each layer is a PhysicalRBFLayer (physical component implementation).
"""

import numpy as np
from typing import Tuple, List

from .layer import PhysicalRBFLayer


class PhysicalResidualRBFNetwork:
    """
    Physical simulation residual RBF network.

    Architecture:
        Input (x, y)
          → Layer1: PhysicalRBFLayer → f1
          → Layer2: PhysicalRBFLayer → delta2 → f2 = f1 + delta2
          → Layer3: PhysicalRBFLayer → delta3 → f3 = f2 + delta3
          → output = f3

    Parameters
    ----------
    layers : list of PhysicalRBFLayer
        3 physical RBF layers (must sync parameters from digital model)
    """

    def __init__(self, layers: List[PhysicalRBFLayer]):
        assert len(layers) == 3, "ResidualRBFNetwork requires exactly 3 layers"
        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]

    @classmethod
    def from_digital(cls, digital_model) -> "PhysicalResidualRBFNetwork":
        """Create physical simulator from PyTorch ResidualRBFNetwork"""
        layers = [
            PhysicalRBFLayer.from_digital(digital_model.layer1),
            PhysicalRBFLayer.from_digital(digital_model.layer2),
            PhysicalRBFLayer.from_digital(digital_model.layer3),
        ]
        return cls(layers)

    def sync_from_digital(self, digital_model):
        """Sync all parameters from PyTorch model"""
        self.layer1.sync_from_digital(digital_model.layer1)
        self.layer2.sync_from_digital(digital_model.layer2)
        self.layer3.sync_from_digital(digital_model.layer3)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Physical forward propagation (with all non-idealities).

        Parameters
        ----------
        x, y : (batch,)
            Input coordinates

        Returns
        -------
        output : (batch, output_dim)
            Network output
        """
        # Layer 1
        out_g1, out_a1 = self.layer1.forward(x, y)
        f1 = out_g1 + out_a1  # Dual-channel fusion: (batch, n_vars1)

        # Prepare input for Layer 2: project to (batch,) spatial coordinates
        g1_sq = out_g1.mean(-1) if out_g1.ndim > 1 and out_g1.shape[-1] > 1 else out_g1.reshape(-1)
        a1_sq = out_a1.mean(-1) if out_a1.ndim > 1 and out_a1.shape[-1] > 1 else out_a1.reshape(-1)

        # Layer 2: residual correction
        out_g2, out_a2 = self.layer2.forward(g1_sq, a1_sq)
        delta2 = out_g2 + out_a2  # (batch, n_vars2)

        # f2 = f1 + delta2 — shape alignment
        if f1.shape[1] > delta2.shape[1]:
            delta2_padded = np.zeros_like(f1)
            delta2_padded[:, :delta2.shape[1]] = delta2
            f2 = f1 + delta2_padded
        elif delta2.shape[1] > f1.shape[1]:
            f1_padded = np.zeros_like(delta2)
            f1_padded[:, :f1.shape[1]] = f1
            f2 = f1_padded + delta2
        else:
            f2 = f1 + delta2

        # Prepare input for Layer 3: project to (batch,) spatial coordinates
        g2_sq = out_g2.mean(-1) if out_g2.ndim > 1 and out_g2.shape[-1] > 1 else out_g2.reshape(-1)
        a2_sq = out_a2.mean(-1) if out_a2.ndim > 1 and out_a2.shape[-1] > 1 else out_a2.reshape(-1)

        # Layer 3: residual correction
        out_g3, out_a3 = self.layer3.forward(g2_sq, a2_sq)
        delta3 = out_g3 + out_a3  # (batch, n_vars3)

        # f3 = f2 + delta3 — shape alignment
        if f2.shape[1] > delta3.shape[1]:
            # f2 has more dimensions: retain f2 values for dimensions not covered by delta3
            delta3_padded = np.zeros_like(f2)
            delta3_padded[:, :delta3.shape[1]] = delta3
            f3 = f2 + delta3_padded
        elif delta3.shape[1] > f2.shape[1]:
            f2_padded = np.zeros_like(delta3)
            f2_padded[:, :f2.shape[1]] = f2
            f3 = f2_padded + delta3
        else:
            f3 = f2 + delta3

        # Output truncation: final output dimension = last layer's output dimension
        out_dim = delta3.shape[1]
        return f3[:, :out_dim]

    def forward_ideal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Ideal forward (disable all noise, for comparison/validation).
        """
        from .components import NoiseConfig
        original = NoiseConfig._current
        NoiseConfig.set_level("none")
        result = self.forward(x, y)
        NoiseConfig.set_level(original)
        return result

    def get_all_params_vector(self) -> np.ndarray:
        """Get flattened vector of all layer parameters (for SPSA)"""
        return np.concatenate([
            self.layer1.get_params_vector(),
            self.layer2.get_params_vector(),
            self.layer3.get_params_vector(),
        ])

    def set_all_params_from_vector(self, vec: np.ndarray):
        """Restore all layer parameters from flattened vector"""
        n1 = self.layer1.get_params_vector().size
        n2 = self.layer2.get_params_vector().size
        self.layer1.set_params_from_vector(vec[:n1])
        self.layer2.set_params_from_vector(vec[n1:n1 + n2])
        self.layer3.set_params_from_vector(vec[n1 + n2:])

    @property
    def n_total_params(self) -> int:
        return self.get_all_params_vector().size
