"""
V3 D-Dimensional RBF Particle Neural Network Core

Key differences from v1 (rbf_network/core.py):
  - Particles are D-dimensional vectors instead of 2D (xc, yc) pairs
  - Distance: d² = Σᵢ (xᵢ - xcᵢ)² / h²  (i = 1..D)
  - Inter-layer: wire-direct (no projection needed — output dim → next input dim)
  - No grid/boundary/fixed_points (simplified for D-dimensional domain)
  - Dual-channel (narrow 0.5 + wide 0.3 kernel) preserved
  - PU normalization preserved
  - Residual connections with zero-padding for dimension mismatch
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List


# ──────────────────────────────────────────────
#  RBF Activation Functions (identical to v1)
# ──────────────────────────────────────────────

def gaussian_kernel(d2: torch.Tensor) -> torch.Tensor:
    """Narrow Gaussian kernel: exp(-0.5 * d²), captures local details."""
    return torch.exp(-0.5 * d2)


def wide_kernel(d2: torch.Tensor) -> torch.Tensor:
    """Wide Gaussian kernel: exp(-0.3 * d²), captures global structure."""
    return torch.exp(-0.3 * d2)


# ──────────────────────────────────────────────
#  ShiftND — D-Dimensional Particle Position Manager
# ──────────────────────────────────────────────

class ShiftND(nn.Module):
    """
    Manages a set of N particles in D-dimensional space.

    Parameters
    ----------
    n_particles : int
        Number of RBF particles
    dim : int
        Dimensionality D of the input space
    init_range : tuple of float
        (min, max) range for random initialization of particle centers
    """

    def __init__(
        self,
        n_particles: int,
        dim: int,
        init_range: Tuple[float, float] = (-3.0, 3.0),
    ):
        super().__init__()
        self.n_particles = n_particles
        self.dim = dim

        lo, hi = init_range

        # centers: (D, N) — each column is one particle's D-dimensional coordinates
        centers = torch.empty(dim, n_particles, dtype=torch.float32)
        nn.init.uniform_(centers, lo, hi)
        self.centers = nn.Parameter(centers)

        # h: (N,) — scalar width per particle
        dsize = hi - lo
        h_init = (dsize / np.sqrt(max(n_particles, 1)))
        self.h = nn.Parameter(torch.full((n_particles,), h_init, dtype=torch.float32))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute D-dimensional normalized squared distances.

        Parameters
        ----------
        X : (batch, D) tensor
            Input vectors in D-dimensional space

        Returns
        -------
        d2 : (batch, N) tensor
            Squared normalized distances: d²_j = Σ_i ((X_i - centers_{i,j}) / h_j)²
        """
        # X: (B, D), centers: (D, N)
        # Reshape for broadcasting:
        #   X     → (B, D, 1)
        #   centers → (1, D, N)
        #   diff → (B, D, N)
        diff = X.unsqueeze(2) - self.centers.unsqueeze(0)  # (B, D, N)

        # h: (N,) → (1, 1, N)
        inv_h = 1.0 / self.h.view(1, 1, -1)
        scaled = diff * inv_h  # (B, D, N)

        # d² = Σ over D dimensions → (B, N)
        d2 = (scaled * scaled).sum(dim=1)
        return d2


# ──────────────────────────────────────────────
#  RBFLayerND — Single D-Dimensional RBF Layer
# ──────────────────────────────────────────────

class RBFLayerND(nn.Module):
    """
    Single D-dimensional RBF layer.

    Forward pass:
      1. Compute D-dimensional normalized distance to each particle
      2. Apply dual-channel Gaussian activation (narrow + wide)
      3. Optional PU normalization
      4. Linear projection to output dimension

    Parameters
    ----------
    dim : int
        Input dimensionality D
    n_particles : int
        Number of RBF particles
    n_vars : int
        Output dimension
    init_range : tuple of float
        Range for particle center initialization
    use_pu : bool
        Enable Partition-of-Unity normalization
    init_zero : bool
        If True, initialize linear weights to zero
    """

    def __init__(
        self,
        dim: int,
        n_particles: int,
        n_vars: int = 1,
        init_range: Tuple[float, float] = (-3.0, 3.0),
        use_pu: bool = True,
        init_zero: bool = True,
    ):
        super().__init__()
        self.shift = ShiftND(n_particles, dim, init_range)
        self.n_particles = n_particles
        self.dim = dim
        self.n_vars = n_vars
        self.use_pu = use_pu

        # Linear projections for dual channels
        self.output_gauss = nn.Linear(n_particles, n_vars, bias=not use_pu)
        self.output_act = nn.Linear(n_particles, n_vars, bias=not use_pu)

        if init_zero:
            nn.init.zeros_(self.output_gauss.weight)
            nn.init.zeros_(self.output_act.weight)
            if not use_pu:
                nn.init.zeros_(self.output_gauss.bias)
                nn.init.zeros_(self.output_act.bias)
        else:
            nn.init.xavier_uniform_(self.output_gauss.weight)
            nn.init.xavier_uniform_(self.output_act.weight)
            if not use_pu:
                nn.init.zeros_(self.output_gauss.bias)
                nn.init.zeros_(self.output_act.bias)

    def centers(self) -> torch.Tensor:
        """Return particle centers: (D, N)."""
        return self.shift.centers

    def widths(self) -> torch.Tensor:
        """Return particle widths: (N,)."""
        return self.shift.h

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        X : (batch, D) tensor
            Input in D-dimensional space

        Returns
        -------
        out_gauss : (batch, n_vars)
            Narrow Gaussian channel output
        out_act : (batch, n_vars)
            Wide Gaussian channel output
        """
        d2 = self.shift(X)  # (batch, N)

        # Narrow Gaussian channel
        z_gauss = gaussian_kernel(d2)  # (batch, N)
        if self.use_pu:
            z_gauss = z_gauss / (z_gauss.sum(dim=1, keepdim=True) + 1e-8)
        out_gauss = self.output_gauss(z_gauss)  # (batch, n_vars)

        # Wide Gaussian channel
        z_act = wide_kernel(d2)  # (batch, N)
        if self.use_pu:
            z_act = z_act / (z_act.sum(dim=1, keepdim=True) + 1e-8)
        out_act = self.output_act(z_act)  # (batch, n_vars)

        return out_gauss, out_act

    def sync_to_dict(self) -> dict:
        """Export layer parameters as a numpy dict (for physical simulator sync)."""
        return {
            "centers": self.centers().detach().cpu().numpy().copy(),
            "h": self.widths().detach().cpu().numpy().copy(),
            "W_gauss": self.output_gauss.weight.detach().cpu().numpy().copy(),
            "W_act": self.output_act.weight.detach().cpu().numpy().copy(),
            "use_pu": self.use_pu,
            "bias_gauss": self.output_gauss.bias.detach().cpu().numpy().copy()
            if self.output_gauss.bias is not None else None,
            "bias_act": self.output_act.bias.detach().cpu().numpy().copy()
            if self.output_act.bias is not None else None,
        }


# ──────────────────────────────────────────────
#  ResidualRBFNetworkND — 3-Layer Residual Network
# ──────────────────────────────────────────────

class ResidualRBFNetworkND(nn.Module):
    """
    Three-layer D-dimensional RBF particle neural network with residual connections.

    Architecture:
        Input X (batch, D)
          → Layer1: RBFLayerND → (g1, a1) → f1 = g1 + a1           (batch, d1)
          → Layer2: RBFLayerND → (g2, a2) → delta2 = g2 + a2        (batch, d2)
              f2 = pad(f1) + delta2
          → Layer3: RBFLayerND → (g3, a3) → delta3 = g3 + a3        (batch, d3)
              f3 = pad(f2) + delta3
          → output = f3

    Inter-layer: wire-direct — Layer k output IS Layer k+1 input.
    No mean(-1) projection needed because input dim naturally matches output dim
    of the previous layer.

    Residual zero-padding: when d_{k+1} > d_k or d_{k+1} < d_k, the smaller
    tensor is zero-padded before addition.

    Parameters
    ----------
    input_dim : int
        Dimensionality D of the input space
    hidden_dim : int
        Output dimension of Layer 1 (d1)
    output_dim : int
        Output dimension of Layer 3 (d3). Layer 2 uses output_dim as well.
    layer_particles : tuple of int
        Number of particles per layer [n1, n2, n3]
    init_range : tuple of float
        Range for particle center initialization
    task : str
        "regression" or "classification"
    num_classes : int
        Number of classes (for classification only)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 4,
        output_dim: int = 2,
        layer_particles: Tuple[int, int, int] = (20, 12, 8),
        init_range: Tuple[float, float] = (-3.0, 3.0),
        task: str = "regression",
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task = task

        n1, n2, n3 = layer_particles

        # Layer 1: D → d1
        self.layer1 = RBFLayerND(
            dim=input_dim, n_particles=n1, n_vars=hidden_dim,
            init_range=init_range, use_pu=True, init_zero=False,
        )
        # Layer 2: d1 → d2 (= output_dim)
        self.layer2 = RBFLayerND(
            dim=hidden_dim, n_particles=n2, n_vars=output_dim,
            init_range=init_range, use_pu=True, init_zero=True,
        )
        # Layer 3: d2 → d3 (= output_dim)
        self.layer3 = RBFLayerND(
            dim=output_dim, n_particles=n3, n_vars=output_dim,
            init_range=init_range, use_pu=True, init_zero=True,
        )

        # Classification head (optional)
        # NOTE: no Softmax — CrossEntropyLoss expects raw logits
        if task == "classification":
            self.head = nn.Linear(output_dim, num_classes)
        else:
            self.head = nn.Identity()

    def _zero_pad(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Zero-pad the smaller tensor to match the larger tensor's last dimension.

        Parameters
        ----------
        base : (B, db) tensor
        delta : (B, dd) tensor

        Returns
        -------
        result : (B, max(db, dd)) tensor
        """
        db, dd = base.shape[-1], delta.shape[-1]
        if db > dd:
            pad = torch.zeros(base.shape[0], db, device=base.device, dtype=base.dtype)
            pad[:, :dd] = delta
            return base + pad
        elif dd > db:
            pad = torch.zeros(delta.shape[0], dd, device=delta.device, dtype=delta.dtype)
            pad[:, :db] = base
            return pad + delta
        else:
            return base + delta

    def _prepare_input(self, signal: torch.Tensor, layer: RBFLayerND) -> torch.Tensor:
        """
        Truncate or zero-pad signal to match a layer's expected input dimension.

        Parameters
        ----------
        signal : (B, d_signal) tensor
            Accumulated residual signal from previous layers
        layer : RBFLayerND
            Target layer with known input dimension

        Returns
        -------
        input : (B, layer.dim) tensor
        """
        d_target = layer.dim
        d_current = signal.shape[-1]
        if d_current >= d_target:
            return signal[:, :d_target]
        else:
            pad = torch.zeros(signal.shape[0], d_target - d_current,
                              device=signal.device, dtype=signal.dtype)
            return torch.cat([signal, pad], dim=1)

    def forward(self, X: torch.Tensor, return_hidden: bool = False):
        """
        Forward pass through 3-layer residual network.

        Parameters
        ----------
        X : (batch, D) tensor
            Input in D-dimensional space
        return_hidden : bool
            If True, return intermediate outputs for diagnostics

        Returns
        -------
        output : (batch, output_dim) or (batch, num_classes)
            Final network output
        hidden : tuple (optional)
            (g1, a1), (g2, a2), (g3, a3)
        """
        # Layer 1
        out_g1, out_a1 = self.layer1(X)  # each (B, d1)
        f1 = out_g1 + out_a1              # (B, d1)

        # Layer 2: residual — input = Layer 1 output directly
        out_g2, out_a2 = self.layer2(f1)  # each (B, d2)
        delta2 = out_g2 + out_a2          # (B, d2)
        f2 = self._zero_pad(f1, delta2)   # (B, max(d1, d2))

        # Layer 3: residual — input = truncate/pad f2 to Layer 3 dim
        f2_input = self._prepare_input(f2, self.layer3)
        out_g3, out_a3 = self.layer3(f2_input)  # each (B, d3)
        delta3 = out_g3 + out_a3          # (B, d3)
        f3 = self._zero_pad(f2, delta3)

        # Truncate to output_dim
        output = f3[:, :self.output_dim]
        output = self.head(output)

        if return_hidden:
            return output, (out_g1, out_a1), (out_g2, out_a2), (out_g3, out_a3)
        return output

    def get_all_centers(self) -> List[np.ndarray]:
        """
        Get particle centers for all layers.

        Returns
        -------
        list of (D_i, N_i) ndarrays, one per layer
        """
        centers = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            c = layer.centers().detach().cpu().numpy()
            centers.append(c)
        return centers

    def get_all_params_vector(self) -> np.ndarray:
        """
        Flatten all trainable parameters into a single vector.

        Order: layer1.centers → layer1.h → layer1.W_gauss → layer1.W_act →
               layer2.centers → ... → layer3.W_act.

        Returns
        -------
        vec : 1D ndarray
        """
        parts = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            parts.append(layer.centers().detach().cpu().numpy().ravel())
            parts.append(layer.widths().detach().cpu().numpy().ravel())
            parts.append(layer.output_gauss.weight.detach().cpu().numpy().ravel())
            parts.append(layer.output_act.weight.detach().cpu().numpy().ravel())
        return np.concatenate(parts)

    def set_all_params_from_vector(self, vec: np.ndarray):
        """
        Restore all trainable parameters from a flattened vector.
        Inverse of get_all_params_vector.
        """
        device = next(self.parameters()).device
        offset = 0
        for layer in [self.layer1, self.layer2, self.layer3]:
            D, N = layer.dim, layer.n_particles
            d_out = layer.n_vars

            # centers: (D, N)
            n_centers = D * N
            c = torch.tensor(vec[offset:offset + n_centers], dtype=torch.float32,
                             device=device).reshape(D, N)
            layer.shift.centers.data.copy_(c)
            offset += n_centers

            # h: (N,)
            h = torch.tensor(vec[offset:offset + N], dtype=torch.float32, device=device)
            layer.shift.h.data.copy_(h)
            offset += N

            # W_gauss: (n_vars, N)
            n_wg = d_out * N
            wg = torch.tensor(vec[offset:offset + n_wg], dtype=torch.float32,
                              device=device).reshape(d_out, N)
            layer.output_gauss.weight.data.copy_(wg)
            offset += n_wg

            # W_act: (n_vars, N)
            wa = torch.tensor(vec[offset:offset + n_wg], dtype=torch.float32,
                              device=device).reshape(d_out, N)
            layer.output_act.weight.data.copy_(wa)
            offset += n_wg

    @property
    def n_total_params(self) -> int:
        return len(self.get_all_params_vector())
