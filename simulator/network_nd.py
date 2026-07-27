"""
V3 Physical ResidualRBFNetworkND — 3-layer D-dimensional residual RBF network
with analog circuit simulation.

Key differences from v1 PhysicalResidualRBFNetwork:
  - Input: (batch, D) ndarray instead of separate (x, y) arrays
  - Inter-layer: wire-direct (no .mean(-1) projection needed)
    Each layer's output (batch, d_k) IS the next layer's input
  - Residual accumulation with zero-padding for dimension mismatch
  - Same 6-step circuit chain per layer (reusing v1 components.py)
"""

import numpy as np
from typing import List, Tuple, Optional

from physical_simulator.components import NoiseConfig


def _zero_pad_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two arrays with zero-padding for mismatched last dimensions."""
    da, db = a.shape[-1], b.shape[-1]
    if da > db:
        out = a.copy()
        out[:, :db] += b
        return out
    elif db > da:
        out = b.copy()
        out[:, :da] += a
        return out
    else:
        return a + b


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (digital-domain readout only)."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class PhysicalResidualRBFNetworkND:
    """
    Physical simulation of 3-layer D-dimensional residual RBF network.

    Architecture:
        Input X (batch, D)
          → Layer1: PhysicalRBFLayerND → f1 (batch, d1)
          → Layer2: PhysicalRBFLayerND → delta2 (batch, d2) → f2 = f1 + delta2
          → Layer3: PhysicalRBFLayerND → delta3 (batch, d3) → f3 = f2 + delta3
          → output = f3

    No inter-layer projection — wire-direct connections throughout.
    Residuals are zero-padded when dimensions don't match.

    Parameters
    ----------
    layers : list of PhysicalRBFLayerND
        3 physical RBF layers
    """

    def __init__(self, layers: List):
        assert len(layers) == 3, "ResidualRBFNetworkND requires exactly 3 layers"
        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]

    @classmethod
    def from_digital(cls, digital_model) -> "PhysicalResidualRBFNetworkND":
        """
        Create physical simulator from digital ResidualRBFNetworkND.

        Parameters
        ----------
        digital_model : ResidualRBFNetworkND
            PyTorch digital model

        Returns
        -------
        PhysicalResidualRBFNetworkND
        """
        from simulator.layer_nd import PhysicalRBFLayerND

        layers = [
            PhysicalRBFLayerND.from_digital(digital_model.layer1),
            PhysicalRBFLayerND.from_digital(digital_model.layer2),
            PhysicalRBFLayerND.from_digital(digital_model.layer3),
        ]
        return cls(layers)

    def sync_from_digital(self, digital_model):
        """Sync all parameters from digital PyTorch model to physical layers."""
        self.layer1.sync_from_digital(digital_model.layer1)
        self.layer2.sync_from_digital(digital_model.layer2)
        self.layer3.sync_from_digital(digital_model.layer3)

    def _zero_pad_add(self, base: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Add delta to base with zero-padding for dimension mismatch.

        Parameters
        ----------
        base : (batch, d_base)
        delta : (batch, d_delta)

        Returns
        -------
        result : (batch, max(d_base, d_delta))
        """
        b, db = base.shape
        _, dd = delta.shape

        if db > dd:
            padded = np.zeros_like(base)
            padded[:, :dd] = delta
            return base + padded
        elif dd > db:
            padded = np.zeros_like(delta)
            padded[:, :db] = base
            return padded + delta
        else:
            return base + delta

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Full forward propagation through the 3-layer physical network.

        Parameters
        ----------
        X : (batch, D) ndarray
            Input data in D-dimensional space

        Returns
        -------
        output : (batch, d3) ndarray
            Network output (last layer's output dimension)
        """
        # Layer 1
        out_g1, out_a1 = self.layer1.forward(X)
        f1 = out_g1 + out_a1  # (batch, d1) — fused dual-channel output

        # Layer 2: residual — input = Layer1 output directly (wire-direct)
        out_g2, out_a2 = self.layer2.forward(f1)
        delta2 = out_g2 + out_a2  # (batch, d2)
        f2 = self._zero_pad_add(f1, delta2)  # (batch, max(d1, d2))

        # Layer 3: residual — input = Layer2 output directly (wire-direct)
        out_g3, out_a3 = self.layer3.forward(f2)
        delta3 = out_g3 + out_a3  # (batch, d3)
        f3 = self._zero_pad_add(f2, delta3)

        # Output: last layer output dimension
        out_dim = delta3.shape[1]
        return f3[:, :out_dim]

    def forward_ideal(self, X: np.ndarray) -> np.ndarray:
        """
        Ideal forward (all noise disabled) for comparison/validation.
        """
        original = NoiseConfig._current
        NoiseConfig.set_level("none")
        result = self.forward(X)
        NoiseConfig.set_level(original)
        return result

    def get_all_params_vector(self) -> np.ndarray:
        """Flattened vector of all parameters (for SPSA or global FD)."""
        return np.concatenate([
            self.layer1.get_params_vector(),
            self.layer2.get_params_vector(),
            self.layer3.get_params_vector(),
        ])

    def set_all_params_from_vector(self, vec: np.ndarray):
        """Restore all layer parameters from flattened vector."""
        n1 = self.layer1.get_params_vector().size
        n2 = self.layer2.get_params_vector().size
        self.layer1.set_params_from_vector(vec[:n1])
        self.layer2.set_params_from_vector(vec[n1:n1 + n2])
        self.layer3.set_params_from_vector(vec[n1 + n2:])

    @property
    def n_total_params(self) -> int:
        return self.get_all_params_vector().size

    # ── Proto-weighted inference ──

    @staticmethod
    def _proto_predict(
        h: np.ndarray, proto_h: np.ndarray, proto_y: np.ndarray,
        temperature: float = 0.1,
    ) -> np.ndarray:
        """
        Predict y from h via prototype-weighted interpolation.
        This is a digital readout — not part of the physical forward path.

        Parameters
        ----------
        h : (N, h_dim)
        proto_h : (K, h_dim)
        proto_y : (K, d)
        temperature : float

        Returns
        -------
        pred : (N, d)
        """
        dist = np.linalg.norm(h[:, None, :] - proto_h[None, :, :], axis=2)
        weights = _softmax(-dist / temperature, axis=1)
        return weights @ proto_y

    def predict(
        self, X: np.ndarray,
        proto_h: List[np.ndarray], proto_y: List[np.ndarray],
        temperature: float = 0.1,
    ) -> np.ndarray:
        """
        Full physical forward + proto-weighted inference.

        1. Physical: X → L1 → L2 → L3 (pure analog)
        2. Per-layer proto_predict → accumulate residuals (digital readout)

        Parameters
        ----------
        X : (N, D)
        proto_h : list of 3 ndarrays (K, h_dim_i)
        proto_y : list of 3 ndarrays (K, d_y)
        temperature : float

        Returns
        -------
        pred : (N, d_y)
        """
        acc = None
        h = X.copy()

        layers = [self.layer1, self.layer2, self.layer3]
        for i, layer in enumerate(layers):
            out_g, out_a = layer.forward(h)
            h_out = out_g + out_a
            if h_out.shape[0] != h.shape[0]:
                h_out = h_out.T

            # Proto-weighted residual prediction (digital readout)
            if i < len(proto_h) and i < len(proto_y):
                pred_i = self._proto_predict(h_out, proto_h[i], proto_y[i], temperature)
            else:
                pred_i = h_out

            acc = pred_i.copy() if acc is None else _zero_pad_sum(acc, pred_i)
            h = h_out

        return acc

    # ── Constructors ──

    @classmethod
    def from_himal_trained(
        cls, layers_A: List, proto_h_A: List[np.ndarray] = None,
        proto_y_A: List[np.ndarray] = None,
    ) -> "PhysicalResidualRBFNetworkND":
        """
        Build physical network from HiMAL-trained A-network layers.

        Parameters
        ----------
        layers_A : list of PhysicalRBFLayerND
        proto_h_A, proto_y_A : optional prototype lists for inference

        Returns
        -------
        PhysicalResidualRBFNetworkND
        """
        net = cls([layers_A[0], layers_A[1], layers_A[2]])
        net._proto_h = proto_h_A or []
        net._proto_y = proto_y_A or []
        return net

    @classmethod
    def from_himal_dual(
        cls, layers_A: List, layers_B: List,
        proto_h_A: List[np.ndarray], proto_y_A: List[np.ndarray],
        proto_h_B: List[np.ndarray], proto_y_B: List[np.ndarray],
    ) -> Tuple["PhysicalResidualRBFNetworkND", "PhysicalResidualRBFNetworkND"]:
        """
        Build two physical networks from HiMAL dual-trained layers.

        Returns (net_A, net_B).
        """
        net_A = cls.from_himal_trained(layers_A, proto_h_A, proto_y_A)
        net_B = cls.from_himal_trained(layers_B, proto_h_B, proto_y_B)
        return net_A, net_B

    @staticmethod
    def predict_dual(
        net_A: "PhysicalResidualRBFNetworkND",
        net_B: "PhysicalResidualRBFNetworkND",
        X: np.ndarray, temperature: float = 0.1,
    ) -> np.ndarray:
        """Dual-network prediction: average of A and B proto-weighted outputs."""
        pred_A = net_A.predict(X, net_A._proto_h, net_A._proto_y, temperature)
        pred_B = net_B.predict(X, net_B._proto_h, net_B._proto_y, temperature)
        if pred_A.shape[-1] != pred_B.shape[-1]:
            pred_A = _zero_pad_sum(pred_A, np.zeros_like(pred_B))
            pred_B = _zero_pad_sum(np.zeros_like(pred_A), pred_B)
        return (pred_A + pred_B) / 2.0
