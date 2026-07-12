"""
RBF Particle Neural Network Core Module

Core design principles:
- Each neuron is a movable "particle" in 2D space, with learnable coordinates (x, y) and width h (influence radius)
- Node coordinates and Linear weights are both trainable parameters (no additional neural network mapping per layer)
- Dual-channel parallel activation: Gaussian (narrow kernel, captures local) + Custom (wide kernel, captures global)
- Partition of Unity (PU) normalization ensures particle contributions sum to 1
- Supports residual connections (ResidualRBFNetwork), where each layer learns a correction to the previous layer's output
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List


# ──────────────────────────────────────────────
#  RBF Activation Functions
# ──────────────────────────────────────────────

def gaussian_kernel(dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Narrow Gaussian kernel: exp(-0.5 * d²), captures local details"""
    return torch.exp(-0.5 * (dx * dx + dy * dy))


def wide_kernel(dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Wide Gaussian kernel: exp(-0.3 * d²), captures global structure"""
    return torch.exp(-0.3 * (dx * dx + dy * dy))


# ──────────────────────────────────────────────
#  Shift2D — Particle Position Manager
# ──────────────────────────────────────────────

class Shift2D(nn.Module):
    """
    Manages a set of particles in 2D space.

    Particles are divided into two types:
    - free_points: trainable coordinates (move through space via gradient descent)
    - fixed_points: non-trainable anchor points (used for boundary constraints, etc.)

    Parameters
    ----------
    points : tuple of arrays
        Initial (x, y) coordinates of free particles
    fixed_points : tuple of arrays
        (x, y) coordinates of fixed particles
    fixed_h : bool
        True means all particles share a single width parameter
    grid_params : tuple, optional
        (xL, xR, T, nx, nt, endpoint) for generating interior grid points
    boundary_params : tuple, optional
        (xL, xR, T, nx, nt) for generating boundary points
    """
    def __init__(
        self,
        points: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([])),
        fixed_points: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([])),
        fixed_h: bool = False,
        grid_params: Optional[Tuple] = None,
        boundary_params: Optional[Tuple] = None,
    ):
        super().__init__()

        grid_points = self._make_grid(grid_params) if grid_params else (np.array([]), np.array([]))
        boundary_pts = self._make_boundary(boundary_params) if boundary_params else (np.array([]), np.array([]))

        points = (
            np.concatenate([points[0], grid_points[0]]),
            np.concatenate([points[1], grid_points[1]]),
        )
        fixed_points = (
            np.concatenate([fixed_points[0], boundary_pts[0]]),
            np.concatenate([fixed_points[1], boundary_pts[1]]),
        )

        n_free = len(points[0]) if points[0].size > 0 else 0
        n_fixed = len(fixed_points[0]) if fixed_points[0].size > 0 else 0
        self.n = n_free + n_fixed

        all_x = np.hstack([arr for arr in [points[0], fixed_points[0]] if arr.size > 0])
        dsize = np.ptp(all_x) if all_x.size > 0 else 1.0
        dx = dsize / np.sqrt(self.n) if self.n > 0 else 1.0

        self.x = nn.Parameter(torch.tensor(points[0], dtype=torch.float32) if n_free > 0 else torch.tensor([]))
        self.y = nn.Parameter(torch.tensor(points[1], dtype=torch.float32) if n_free > 0 else torch.tensor([]))

        self.register_buffer("xf", torch.tensor(fixed_points[0], dtype=torch.float32) if n_fixed > 0 else torch.tensor([]))
        self.register_buffer("yf", torch.tensor(fixed_points[1], dtype=torch.float32) if n_fixed > 0 else torch.tensor([]))

        self.fixed_h = fixed_h
        if fixed_h:
            self.h = nn.Parameter(torch.tensor(dx, dtype=torch.float32) if self.n > 0 else torch.tensor(1.0))
        else:
            self.h = nn.Parameter(dx * torch.ones(self.n, dtype=torch.float32) if self.n > 0 else torch.tensor([1.0]))

    def _make_grid(self, params: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        xL, xR, T, nx, nt, endpoint = params
        if endpoint:
            x, t = np.mgrid[slice(xL, xR, nx * 1j), slice(0.0, T, nt * 1j)]
            return (x.ravel(), t.ravel())
        dxb2 = 0.5 * (xR - xL) / (nx + 1)
        dt = T / (nt + 1)
        x, t = np.mgrid[slice(xL + dxb2, xR - dxb2, nx * 1j), slice(dt, T, nt * 1j)]
        return (x.ravel(), t.ravel())

    def _make_boundary(self, params: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        xL, xR, T, nx, nt = params
        x_bottom = np.linspace(xL, xR, nx)
        t_bottom = np.zeros(nx)
        x_left = np.full(nt, xL)
        t_left = np.linspace(T / nt, T, nt)
        x_right = np.full(nt, xR)
        t_right = np.linspace(T / nt, T, nt)
        xs = np.concatenate([x_bottom, x_left, x_right])
        ts = np.concatenate([t_bottom, t_left, t_right])
        return (xs, ts)

    def centers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.cat((self.x, self.xf)), torch.cat((self.y, self.yf))

    def widths(self) -> torch.Tensor:
        return self.h

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xc, yc = self.centers()
        if self.n == 0:
            return torch.zeros(1), torch.zeros(1)
        fac = 1.0 / self.widths()
        xh = (x.unsqueeze(1) - xc) * fac
        yh = (y.unsqueeze(1) - yc) * fac
        return xh, yh


# ──────────────────────────────────────────────
#  RBFLayer — Single RBF Layer
# ──────────────────────────────────────────────

class RBFLayer(nn.Module):
    """
    Single RBF layer.

    Design principle: trainable node coordinates + trainable linear weights.
    No additional neural network mapping is added inside each layer — each RBFLayer itself
    is a linear combiner whose expressiveness comes from the particle spatial distribution
    and the nonlinearity of the RBF kernel.

    Forward pass flow:
    1. Compute normalized distance from input (x, y) to each particle
    2. Compute two activation pathways in parallel: Gaussian + Custom
    3. Optional PU normalization
    4. Pass through separate Linear layers for output
    """
    def __init__(
        self,
        points: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([])),
        fixed_points: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([])),
        fixed_h: bool = False,
        n_vars: int = 1,
        use_pu: bool = True,
        grid_params: Optional[Tuple] = None,
        boundary_params: Optional[Tuple] = None,
        init_zero: bool = True,
    ):
        super().__init__()
        self.shift = Shift2D(points, fixed_points, fixed_h, grid_params, boundary_params)
        self.n_particles = self.shift.n
        self.use_pu = use_pu

        self.output_gauss = nn.Linear(self.n_particles, n_vars, bias=not use_pu)
        self.output_act = nn.Linear(self.n_particles, n_vars, bias=not use_pu)

        if init_zero:
            nn.init.zeros_(self.output_gauss.weight)
            nn.init.zeros_(self.output_act.weight)
            if not use_pu:
                nn.init.zeros_(self.output_gauss.bias)
                nn.init.zeros_(self.output_act.bias)
        else:
            nn.init.xavier_uniform_(self.output_gauss.weight)
            nn.init.xavier_uniform_(self.output_act.weight)

    def centers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.shift.centers()

    def widths(self) -> torch.Tensor:
        return self.shift.widths()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xc, yc = self.centers()
        if self.n_particles == 0:
            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            return (
                torch.zeros(x.shape[0], self.output_gauss.out_features, device=device),
                torch.zeros(x.shape[0], self.output_act.out_features, device=device),
            )
        fac = 1.0 / self.widths()
        xh = (x.unsqueeze(1) - xc) * fac
        yh = (y.unsqueeze(1) - yc) * fac

        z_gauss = gaussian_kernel(xh, yh)
        if self.use_pu:
            z_gauss = z_gauss / (z_gauss.sum(axis=1, keepdim=True) + 1e-8)
        out_gauss = self.output_gauss(z_gauss)

        z_act = wide_kernel(xh, yh)
        if self.use_pu:
            z_act = z_act / (z_act.sum(axis=1, keepdim=True) + 1e-8)
        out_act = self.output_act(z_act)

        return out_gauss, out_act


# ──────────────────────────────────────────────
#  RBFNetwork — Basic 3-Layer RBF Network (No Residual)
# ──────────────────────────────────────────────

class RBFNetwork(nn.Module):
    """
    Three-layer stacked RBF particle neural network (basic version, no residual connections).

    Architecture:
        Input (x, y)
          → Layer1: RBF → (out_g1, out_a1)
          → Layer2: RBF → (out_g2, out_a2)
          → Layer3: RBF → (out_g3, out_a3)
          → combined = out_g3 + out_a3
          → [optional] classifier/regressor head
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 1,
        output_dim: int = 1,
        n_particles: int = 20,
        x_range: Tuple[float, float] = (-10.0, 10.0),
        y_range: Tuple[float, float] = (-10.0, 10.0),
        task: str = "regression",
        num_classes: int = 2,
    ):
        super().__init__()
        self.xL, self.xR = x_range
        self.yL, self.yR = y_range
        self.task = task

        init_particles = lambda n: (
            np.random.uniform(self.xL, self.xR, n).astype(np.float32),
            np.random.uniform(self.yL, self.yR, n).astype(np.float32),
        )

        self.layer1 = RBFLayer(points=init_particles(n_particles), n_vars=hidden_dim, use_pu=True)
        self.layer2 = RBFLayer(points=init_particles(n_particles), n_vars=hidden_dim, use_pu=True)
        self.layer3 = RBFLayer(points=init_particles(n_particles), n_vars=output_dim, use_pu=True)

        if task == "classification":
            self.head = nn.Sequential(nn.Linear(output_dim, num_classes), nn.Softmax(dim=1))
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor, return_hidden: bool = False):
        out_g1, out_a1 = self.layer1(x, y)
        out_g2, out_a2 = self.layer2(out_g1.squeeze(-1), out_a1.squeeze(-1))
        out_g3, out_a3 = self.layer3(out_g2.squeeze(-1), out_a2.squeeze(-1))

        combined = out_g3 + out_a3
        output = self.head(combined)

        if return_hidden:
            return output, (out_g1, out_a1), (out_g2, out_a2), (out_g3, out_a3)
        return output

    def get_all_centers(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        centers = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            cx, cy = layer.centers()
            centers.append((cx.detach().numpy(), cy.detach().numpy()))
        return centers


# ──────────────────────────────────────────────
#  ResidualRBFNetwork — RBF Network with Residual Connections
# ──────────────────────────────────────────────

class ResidualRBFNetwork(nn.Module):
    """
    Three-layer RBF particle neural network with residual connections.

    Each layer learns a correction (delta) to the previous layer's output, rather than
    a completely new mapping. Similar to ResNet's skip connection, this helps gradient
    flow and deep training.

    Architecture:
        Input (x, y)
          → Layer1: RBF → (out_g1, out_a1) → f1 = out_g1 + out_a1
          → Layer2: RBF → (out_g2, out_a2) → delta2 = out_g2 + out_a2 → f2 = f1 + delta2
          → Layer3: RBF → (out_g3, out_a3) → delta3 = out_g3 + out_a3 → f3 = f2 + delta3
          → output = f3

    Supports independent particle counts per layer. For example, in electric field fitting:
    - first_layer_particles = number of charges (e.g., 3 point charges → 3 particles)
    - Uses covariance regularization to make first-layer particles converge to point charge positions

    Parameters
    ----------
    first_layer_particles : int
        Number of particles in the first layer (should equal the number of point charges in
        electric field fitting tasks)
    second_layer_particles : int
        Number of particles in the second layer
    third_layer_particles : int
        Number of particles in the third layer
    """
    def __init__(
        self,
        hidden_dim: int = 1,
        output_dim: int = 1,
        first_layer_particles: int = 30,
        second_layer_particles: int = 20,
        third_layer_particles: int = 20,
        x_range: Tuple[float, float] = (-10.0, 10.0),
        y_range: Tuple[float, float] = (-10.0, 10.0),
        init_zero: bool = False,
    ):
        super().__init__()
        self.xL, self.xR = x_range
        self.yL, self.yR = y_range

        init = lambda n: (
            np.random.uniform(self.xL, self.xR, n).astype(np.float32),
            np.random.uniform(self.yL, self.yR, n).astype(np.float32),
        )

        self.layer1 = RBFLayer(
            points=init(first_layer_particles), n_vars=hidden_dim, use_pu=True, init_zero=init_zero,
        )
        self.layer2 = RBFLayer(
            points=init(second_layer_particles), n_vars=output_dim, use_pu=True, init_zero=init_zero,
        )
        self.layer3 = RBFLayer(
            points=init(third_layer_particles), n_vars=output_dim, use_pu=True, init_zero=init_zero,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Layer 1
        out_g1, out_a1 = self.layer1(x, y)
        f1 = out_g1 + out_a1

        # Layer 2: residual — learn correction
        # Project to scalar: use mean as spatial coordinates for the next layer
        g1_coord = out_g1.mean(-1) if out_g1.ndim > 1 and out_g1.shape[-1] > 1 else out_g1.squeeze(-1)
        a1_coord = out_a1.mean(-1) if out_a1.ndim > 1 and out_a1.shape[-1] > 1 else out_a1.squeeze(-1)
        out_g2, out_a2 = self.layer2(g1_coord, a1_coord)
        delta2 = out_g2 + out_a2
        f2 = f1 + delta2

        # Layer 3: residual — learn correction
        g2_coord = out_g2.mean(-1) if out_g2.ndim > 1 and out_g2.shape[-1] > 1 else out_g2.squeeze(-1)
        a2_coord = out_a2.mean(-1) if out_a2.ndim > 1 and out_a2.shape[-1] > 1 else out_a2.squeeze(-1)
        out_g3, out_a3 = self.layer3(g2_coord, a2_coord)
        delta3 = out_g3 + out_a3
        f3 = f2 + delta3

        return f3
