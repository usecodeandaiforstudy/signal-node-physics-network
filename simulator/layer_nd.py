"""
V3 Physical RBF Layer for D-Dimensional Particles

PhysicalRBFLayerND mirrors v1's PhysicalRBFLayer but handles D-dimensional
centers and input vectors.

Forward signal chain (6 components, same as v1):
  1. D x differential_pair for each dimension (dx_i = (x_i - xc_i) / h)
  2. D x gilbert_square for each dimension (dx_i^2)
  3. Single kirchhoff_sum across all D dimensions (d^2 = sum_i dx_i^2)
  4. bjt_gaussian_kernel x 2 channels (narrow k=0.5, wide k=0.3)
  5. current_mode_normalize (PU) x 2 channels
  6. memristor_mvm x 2 channels (W_gauss . z_gauss, W_act . z_act)

Reuses all components from physical_simulator.components without modification.
"""

import numpy as np
from typing import Tuple

from physical_simulator.components import (
    differential_pair, gilbert_square,
    bjt_gaussian_kernel, current_mode_normalize,
    memristor_mvm, kirchhoff_sum,
    NoiseConfig,
)


class PhysicalRBFLayerND:
    """
    Physical simulation RBF layer for D-dimensional particles.

    Differs from v1 PhysicalRBFLayer:
    - centers: (D, N) ndarray instead of separate (xc, yc) arrays
    - forward(X): X is (batch, D) instead of (x, y)
    - Distance: D pairs of differential_pair + gilbert_square, then kirchhoff_sum

    Parameters
    ----------
    centers : np.ndarray (D, N)
        Particle center coordinates (each column = one particle)
    h : np.ndarray (N,) or float
        Particle width parameter
    W_gauss : np.ndarray (n_vars, N)
        Gaussian channel linear weights
    W_act : np.ndarray (n_vars, N)
        Wide-kernel channel linear weights
    use_pu : bool
        Whether to use PU normalization
    """

    def __init__(
        self,
        centers: np.ndarray,
        h: np.ndarray,
        W_gauss: np.ndarray,
        W_act: np.ndarray,
        use_pu: bool = True,
    ):
        self.centers = np.asarray(centers, dtype=np.float32)
        if self.centers.ndim == 1:
            self.centers = self.centers.reshape(1, -1)
        self.h = np.atleast_1d(np.asarray(h, dtype=np.float32)).flatten()
        self.W_gauss = np.asarray(W_gauss, dtype=np.float32)
        self.W_act = np.asarray(W_act, dtype=np.float32)
        self.use_pu = use_pu

        self.dim = self.centers.shape[0]       # D
        self.n_particles = self.centers.shape[1]  # N
        self.n_vars = self.W_gauss.shape[0]       # d_out

    @classmethod
    def from_digital(cls, digital_layer) -> "PhysicalRBFLayerND":
        """
        Create physical layer from digital RBFLayerND.

        Parameters
        ----------
        digital_layer : RBFLayerND
            PyTorch digital layer

        Returns
        -------
        PhysicalRBFLayerND
        """
        import torch

        centers = digital_layer.centers()
        if isinstance(centers, torch.Tensor):
            centers = centers.detach().cpu().numpy()

        h = digital_layer.widths()
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().numpy()

        Wg = digital_layer.output_gauss.weight.detach().cpu().numpy()
        Wa = digital_layer.output_act.weight.detach().cpu().numpy()

        return cls(
            centers=centers, h=h,
            W_gauss=Wg, W_act=Wa,
            use_pu=digital_layer.use_pu,
        )

    def sync_from_digital(self, digital_layer):
        """Sync parameters from digital RBFLayerND (post-training update)."""
        import torch

        centers = digital_layer.centers()
        self.centers = centers.detach().cpu().numpy() if isinstance(centers, torch.Tensor) else np.asarray(centers)
        if self.centers.ndim == 1:
            self.centers = self.centers.reshape(1, -1)

        h = digital_layer.widths()
        self.h = h.detach().cpu().numpy().flatten() if isinstance(h, torch.Tensor) else np.atleast_1d(np.asarray(h)).flatten()

        self.W_gauss = digital_layer.output_gauss.weight.detach().cpu().numpy()
        self.W_act = digital_layer.output_act.weight.detach().cpu().numpy()

        self.dim = self.centers.shape[0]
        self.n_particles = self.centers.shape[1]
        self.n_vars = self.W_gauss.shape[0]

    def get_params_vector(self) -> np.ndarray:
        """Flatten all parameters into one vector (for SPSA / finite differences)."""
        return np.concatenate([
            self.centers.flatten(),         # D * N
            self.h.flatten(),               # N
            self.W_gauss.flatten(),         # n_vars * N
            self.W_act.flatten(),           # n_vars * N
        ])

    def set_params_from_vector(self, vec: np.ndarray):
        """Restore parameters from flattened vector."""
        D, N = self.dim, self.n_particles
        d_out = self.n_vars

        # centers: (D, N)
        n_c = D * N
        self.centers = vec[:n_c].reshape(D, N).copy()

        # h: (N,)
        self.h = vec[n_c:n_c + N].copy()

        # W_gauss: (n_vars, N)
        offset = n_c + N
        n_wg = d_out * N
        self.W_gauss = vec[offset:offset + n_wg].reshape(d_out, N).copy()

        # W_act: (n_vars, N)
        offset += n_wg
        self.W_act = vec[offset:offset + n_wg].reshape(d_out, N).copy()

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Physical forward propagation through D-dimensional analog circuit chain.

        Signal chain:
          D x differential_pair → D x gilbert_square → kirchhoff_sum(d²)
          → bjt_gaussian_kernel(x2) → current_mode_normalize(x2)
          → memristor_mvm(x2) → kirchhoff_sum(out_gauss, out_act)

        Parameters
        ----------
        X : (batch, D) ndarray
            Input vectors in D-dimensional space

        Returns
        -------
        out_gauss : (batch, n_vars) ndarray
        out_act   : (batch, n_vars) ndarray
        """
        noise_cfg = NoiseConfig.get_config()

        # Steps 1-2: For each dimension, differential_pair + gilbert_square
        # Accumulate d² = sum_i dx_i²
        d2 = None
        for i in range(self.dim):
            x_i = X[:, i].astype(np.float32).ravel()
            # centers[i, :] gives the i-th coord of all particles
            dx = differential_pair(
                x_i, self.centers[i, :], self.h,
                mismatch_std=noise_cfg["mismatch"],
                noise_std=noise_cfg["noise"],
            )
            # Single-dimension: gilbert_square(dx, zeros) = dx²
            dy_zero = np.zeros_like(dx)
            d2_i = gilbert_square(
                dx, dy_zero,
                mismatch_std=noise_cfg["mismatch"],
                noise_std=noise_cfg["noise"],
            )
            if d2 is None:
                d2 = d2_i
            else:
                # Step 3: kirchhoff_sum across dimensions
                d2 = kirchhoff_sum(d2, d2_i)

        if d2 is None:
            # Edge case: 0-dimensional input
            d2 = np.zeros((X.shape[0], self.n_particles), dtype=np.float32)

        # Step 4: BJT exponential — narrow + wide kernel
        v_t = noise_cfg.get("v_t", 1.0)
        z_gauss = bjt_gaussian_kernel(d2, k=0.5, temperature_voltage=v_t)
        z_act = bjt_gaussian_kernel(d2, k=0.3, temperature_voltage=v_t)

        # Step 5: Current mirror normalization (PU)
        if self.use_pu:
            z_gauss = current_mode_normalize(
                z_gauss, mirror_mismatch_std=noise_cfg["mismatch"] * 0.5,
            )
            z_act = current_mode_normalize(
                z_act, mirror_mismatch_std=noise_cfg["mismatch"] * 0.5,
            )

        # Step 6: Memristor MVM — (batch, N) → (batch, n_vars)
        out_gauss = memristor_mvm(
            z_gauss, self.W_gauss,
            conductance_drift_std=noise_cfg["drift"],
            read_noise_std=noise_cfg["noise"],
            stuck_fraction=noise_cfg["stuck"],
        )
        out_act = memristor_mvm(
            z_act, self.W_act,
            conductance_drift_std=noise_cfg["drift"],
            read_noise_std=noise_cfg["noise"],
            stuck_fraction=noise_cfg["stuck"],
        )

        return out_gauss, out_act
