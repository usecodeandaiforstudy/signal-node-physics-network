"""
Physical RBFLayer — single-layer RBF layer implemented with analog circuit elements

Shares the same parameter structure (xc, yc, h, W_gauss, W_act) as rbf_network.core.RBFLayer,
but forward propagation uses physical component models from physical_simulator.components.
"""

import numpy as np
from typing import Tuple

from .components import (
    differential_pair, gilbert_square,
    bjt_gaussian_kernel, current_mode_normalize,
    memristor_mvm, kirchhoff_sum,
    NoiseConfig,
)


class PhysicalRBFLayer:
    """
    Physical simulation RBF layer.

    Differs from the PyTorch RBFLayer:
    - Does not inherit from nn.Module
    - Parameters stored as numpy arrays
    - Forward propagation uses physical component models (with noise and mismatch)
    - Supports parameter sync from digital RBFLayer

    Parameters
    ----------
    xc : np.ndarray (n_particles,)
        Particle x coordinates
    yc : np.ndarray (n_particles,)
        Particle y coordinates
    h : np.ndarray (n_particles,) or float
        Particle width parameter
    W_gauss : np.ndarray (n_vars, n_particles)
        Gaussian channel linear weights
    W_act : np.ndarray (n_vars, n_particles)
        Wide-kernel channel linear weights
    use_pu : bool
        Whether to use PU normalization
    """
    def __init__(
        self,
        xc: np.ndarray,
        yc: np.ndarray,
        h: np.ndarray,
        W_gauss: np.ndarray,
        W_act: np.ndarray,
        use_pu: bool = True,
    ):
        self.xc = np.asarray(xc, dtype=np.float32).flatten()
        self.yc = np.asarray(yc, dtype=np.float32).flatten()
        self.h = np.asarray(h, dtype=np.float32).flatten()
        self.W_gauss = np.asarray(W_gauss, dtype=np.float32)
        self.W_act = np.asarray(W_act, dtype=np.float32)
        self.use_pu = use_pu
        self.n_particles = len(self.xc)

    @classmethod
    def from_digital(cls, digital_layer) -> "PhysicalRBFLayer":
        """Create physical layer by syncing parameters from PyTorch RBFLayer"""
        import torch
        xc, yc = digital_layer.centers()
        if isinstance(xc, torch.Tensor):
            xc = xc.detach().cpu().numpy()
            yc = yc.detach().cpu().numpy()
        h = digital_layer.widths()
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().numpy()
        Wg = digital_layer.output_gauss.weight.detach().cpu().numpy()
        Wa = digital_layer.output_act.weight.detach().cpu().numpy()

        return cls(
            xc=xc, yc=yc, h=h,
            W_gauss=Wg, W_act=Wa,
            use_pu=digital_layer.use_pu,
        )

    def sync_from_digital(self, digital_layer):
        """Sync parameters from PyTorch layer (post-training update)"""
        import torch
        xc, yc = digital_layer.centers()
        self.xc = xc.detach().cpu().numpy().flatten() if isinstance(xc, torch.Tensor) else np.asarray(xc).flatten()
        self.yc = yc.detach().cpu().numpy().flatten() if isinstance(yc, torch.Tensor) else np.asarray(yc).flatten()
        h = digital_layer.widths()
        self.h = h.detach().cpu().numpy().flatten() if isinstance(h, torch.Tensor) else np.asarray(h).flatten()
        self.W_gauss = digital_layer.output_gauss.weight.detach().cpu().numpy()
        self.W_act = digital_layer.output_act.weight.detach().cpu().numpy()

    def get_params_vector(self) -> np.ndarray:
        """Flatten all parameters into one vector (for SPSA/finite differences)"""
        return np.concatenate([
            self.xc.flatten(),
            self.yc.flatten(),
            self.h.flatten(),
            self.W_gauss.flatten(),
            self.W_act.flatten(),
        ])

    def set_params_from_vector(self, vec: np.ndarray):
        """Restore parameters from flattened vector"""
        n = self.n_particles
        n_wg = self.W_gauss.size
        n_wa = self.W_act.size

        self.xc = vec[0:n].copy()
        self.yc = vec[n:2*n].copy()
        self.h = vec[2*n:3*n].copy()
        offset = 3 * n
        self.W_gauss = vec[offset:offset + n_wg].reshape(self.W_gauss.shape).copy()
        offset += n_wg
        self.W_act = vec[offset:offset + n_wa].reshape(self.W_act.shape).copy()

    def forward(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Physical forward propagation.

        Signal chain:
          Differential amplifier → Gilbert square → BJT exponential kernel → Current mirror normalization → Memristor MVM → Kirchhoff summation

        Parameters
        ----------
        x, y : (batch,) or (batch, 1)
            Input coordinates

        Returns
        -------
        out_gauss, out_act : (batch, n_vars)
            Outputs of both channels
        """
        noise_cfg = NoiseConfig.get_config()

        # Step 1: Differential amplifier — (x-xc)/h and (y-yc)/h
        dx = differential_pair(
            x, self.xc, self.h,
            mismatch_std=noise_cfg["mismatch"],
            noise_std=noise_cfg["noise"],
        )
        dy = differential_pair(
            y, self.yc, self.h,
            mismatch_std=noise_cfg["mismatch"],
            noise_std=noise_cfg["noise"],
        )

        # Step 2: Gilbert squarer — d² = dx² + dy²
        d2 = gilbert_square(
            dx, dy,
            mismatch_std=noise_cfg["mismatch"],
            noise_std=noise_cfg["noise"],
        )

        # Step 3: BJT exponential — narrow kernel + wide kernel
        # V_T read from noise config: "none" mode = 1.0 (mathematical equivalent), otherwise = 0.026 (real circuit)
        z_gauss = bjt_gaussian_kernel(d2, k=0.5, temperature_voltage=noise_cfg.get("v_t", 1.0))
        z_act = bjt_gaussian_kernel(d2, k=0.3, temperature_voltage=noise_cfg.get("v_t", 1.0))

        # Step 4: Current mirror normalization (PU)
        if self.use_pu:
            z_gauss = current_mode_normalize(z_gauss, mirror_mismatch_std=noise_cfg["mismatch"] * 0.5)
            z_act = current_mode_normalize(z_act, mirror_mismatch_std=noise_cfg["mismatch"] * 0.5)

        # Step 5: Memristor MVM — (batch, n_particles) → (batch, n_vars)
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

        return out_gauss, out_act  # (batch, n_vars) each
