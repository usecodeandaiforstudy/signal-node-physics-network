"""
Training utilities, visualization, evaluation functions, electric field data generation,
covariance regularization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple, Dict
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist

# Same correction as network/utils_nd.py: `font.sans-serif` rather than `font.family`,
# and families that exist on the platform rather than two Linux/macOS ones.  See the
# comment there for what the wrong form does to every figure.
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════
#  Training Functions
# ═══════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_classification: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        x_batch, y_batch, *targets = [b.to(device) for b in batch]
        optimizer.zero_grad()
        output = model(x_batch, y_batch)
        target = targets[0] if len(targets) == 1 else targets
        loss = criterion(output, target.long() if is_classification else target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    is_classification: bool = True,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    for batch in dataloader:
        x_batch, y_batch, *targets = [b.to(device) for b in batch]
        output = model(x_batch, y_batch)
        target = targets[0] if len(targets) == 1 else targets
        if is_classification:
            if criterion:
                total_loss += criterion(output, target.long()).item() * x_batch.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
        else:
            if criterion:
                total_loss += criterion(output, target.float()).item() * x_batch.size(0)
        total += len(target)
        all_preds.extend(preds.cpu().numpy() if is_classification else output.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    results = {"loss": total_loss / total if criterion else 0.0}
    if is_classification:
        results["accuracy"] = correct / total
    return results, all_preds, all_targets


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 200,
    is_classification: bool = True,
    scheduler: Optional = None,
    verbose_interval: int = 20,
) -> Dict[str, List[float]]:
    history = {"train_loss": [], "test_acc": [] if is_classification else []}
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, is_classification)
        history["train_loss"].append(train_loss)
        if scheduler:
            scheduler.step(train_loss)
        if (epoch + 1) % verbose_interval == 0:
            result, _, _ = evaluate(model, test_loader, criterion, device, is_classification)
            msg = f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}"
            if is_classification:
                history["test_acc"].append(result["accuracy"])
                msg += f", Test Acc: {result['accuracy']:.4f}"
            print(msg)
    return history


# ═══════════════════════════════════════════════════════════
#  Visualization Functions
# ═══════════════════════════════════════════════════════════

def plot_decision_boundary(
    model, X, y, device=torch.device("cpu"), resolution=0.02, margin=0.5, ax=None, title="Decision Boundary"
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    model.eval()
    with torch.no_grad():
        preds = model(
            torch.tensor(xx.ravel(), dtype=torch.float32).to(device),
            torch.tensor(yy.ravel(), dtype=torch.float32).to(device),
        ).argmax(dim=1).cpu().numpy().reshape(xx.shape)
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    ax.contourf(xx, yy, preds, cmap=cmap_light, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=30)
    ax.set_title(title)
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    return ax


def plot_particles(model, X=None, y=None, device=torch.device("cpu"), save_path=None):
    model.eval()
    if X is None:
        return
    x_t = torch.tensor(X[:, 0], dtype=torch.float32).to(device)
    y_t = torch.tensor(X[:, 1], dtype=torch.float32).to(device)
    with torch.no_grad():
        _, (out_g1, out_a1), (out_g2, out_a2), _ = model(x_t, y_t, return_hidden=True)
        out_g1, out_a1 = out_g1.cpu().squeeze().numpy(), out_a1.cpu().squeeze().numpy()
        out_g2, out_a2 = out_g2.cpu().squeeze().numpy(), out_a2.cpu().squeeze().numpy()
    centers = model.get_all_centers()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c=y if y is not None else "blue", cmap="coolwarm", alpha=0.6, s=30)
    axes[0].scatter(centers[0][0], centers[0][1], c="black", marker="X", s=100, label="Layer1 Particles")
    axes[0].set_title("Input Space & Layer1 Particle Centers"); axes[0].legend()
    axes[1].scatter(out_g1, out_a1, c=y if y is not None else "blue", cmap="coolwarm", alpha=0.6, s=30)
    axes[1].scatter(centers[1][0], centers[1][1], c="red", marker="X", s=100, label="Layer2 Particles")
    axes[1].set_title("Layer1 Output & Layer2 Particle Centers"); axes[1].legend()
    axes[2].scatter(out_g2, out_a2, c=y if y is not None else "blue", cmap="coolwarm", alpha=0.6, s=30)
    axes[2].scatter(centers[2][0], centers[2][1], c="green", marker="X", s=100, label="Layer3 Particles")
    axes[2].set_title("Layer2 Output & Layer3 Particle Centers"); axes[2].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"]); axes[0].set_title("Training Loss"); axes[0].grid(linestyle="--", alpha=0.6)
    if "test_acc" in history and history["test_acc"]:
        axes[1].plot(history["test_acc"]); axes[1].set_title("Test Accuracy"); axes[1].grid(linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.show()


def generate_rotation_data(n_samples=10000, x_range=(-10, 10), y_range=(-10, 10), angle=np.pi/6, noise=0.0):
    x = np.random.uniform(*x_range, n_samples).astype(np.float32)
    y = np.random.uniform(*y_range, n_samples).astype(np.float32)
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    if noise > 0:
        x_rot += np.random.normal(0, noise, n_samples)
        y_rot += np.random.normal(0, noise, n_samples)
    return x, y, x_rot.astype(np.float32), y_rot.astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  Electric Field Data Generation (Coulomb's Law)
# ═══════════════════════════════════════════════════════════

def generate_electric_field(
    charges: List[Tuple[float, float, float]],
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    resolution: float = 0.1,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate point-charge electric field distribution dataset using Coulomb's law.

    Parameters
    ----------
    charges : list of (q, x0, y0)
        List of point charges: (charge[C], x-coordinate[m], y-coordinate[m])
    x_range, y_range : tuple
        Spatial ranges
    resolution : float
        Grid resolution
    save_path : str, optional
        CSV save path

    Returns
    -------
    DataFrame with columns: x_position, y_position, electric_field_x, electric_field_y, electric_field_magnitude
    """
    k = 9e9  # Coulomb constant
    x = np.arange(x_range[0], x_range[1], resolution)
    y = np.arange(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    Ex = np.zeros_like(X, dtype=np.float64)
    Ey = np.zeros_like(Y, dtype=np.float64)

    for q, x0, y0 in charges:
        dx = X - x0
        dy = Y - y0
        r_sq = dx**2 + dy**2
        r = np.sqrt(r_sq)

        # Truncate near source points to avoid numerical overflow
        min_dist = np.sqrt(k * abs(q) / 1e12)
        mask = r < min_dist
        r[mask] = min_dist
        r_sq[mask] = min_dist**2

        E = k * q / r_sq
        Ex += E * dx / r
        Ey += E * dy / r

    E_mag = np.sqrt(Ex**2 + Ey**2)
    df = pd.DataFrame({
        "x_position": X.flatten(),
        "y_position": Y.flatten(),
        "electric_field_x": Ex.flatten(),
        "electric_field_y": Ey.flatten(),
        "electric_field_magnitude": E_mag.flatten(),
    })

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Electric field data saved: {save_path} ({len(df)} records)")

    return df


# ═══════════════════════════════════════════════════════════
#  Covariance Regularization — measures how well particle centers match data distribution
# ═══════════════════════════════════════════════════════════

def compute_cov_regularization(model, data: torch.Tensor) -> torch.Tensor:
    """
    Compute the Frobenius norm difference between the covariance matrix of node centers
    and the covariance matrix of the data.

    The goal of this loss: make the spatial distribution of particles in each layer
    consistent with the data distribution, preventing particles from becoming too
    clustered or too dispersed. In the electric field fitting task, this causes the
    first-layer particles to converge to the point-charge positions.

    Parameters
    ----------
    model : ResidualRBFNetwork
        Model to regularize
    data : (N, 2) tensor
        Input data tensor

    Returns
    -------
    loss : sum of Frobenius norms of covariance differences across all layers
    """
    cov_data = torch.tensor(np.cov(data.numpy(), rowvar=False), dtype=torch.float32)
    loss = 0.0
    for layer in [model.layer1, model.layer2, model.layer3]:
        cx, cy = layer.centers()
        if cx.numel() < 2:
            continue
        centers = torch.stack([cx, cy], dim=1)
        centered = centers - centers.mean(dim=0, keepdim=True)
        cov_centers = (centered.T @ centered) / (centers.shape[0] - 1)
        diff = cov_centers - cov_data
        loss += torch.norm(diff, p="fro") ** 2
    return loss


# ═══════════════════════════════════════════════════════════
#  Field Gradient Maximization Loss — drives Layer1 particles toward charge positions
# ═══════════════════════════════════════════════════════════

def compute_field_gradient_attractor(model, device="cpu") -> torch.Tensor:
    """
    Compute the gradient norm of the network output at the positions of Layer1 particles,
    and maximize this norm to attract the particles to the regions of strongest gradient
    in the electric field — i.e., near the point charges.

    Physical principle:
    - The electric field gradient is largest at point charges (essentially singularities)
    - Maximizing ||∇E|| at particle positions = particles are pulled toward charges

    Mathematics:
        L_attract = -Σ_i ||∇f₃(c_{1,i}, d_{1,i})||²

    where (c_{1,i}, d_{1,i}) are the coordinates of Layer1 particle i,
    and ∇f₃ is the gradient of the final network output w.r.t. the input coordinates.

    Warning: this loss produces gradients flowing through second-order derivatives,
    requiring retain_graph=True and create_graph=True, with high memory and
    computational overhead.

    Parameters
    ----------
    model : ResidualRBFNetwork
    device : str or torch.device

    Returns
    -------
    loss : negative sum of gradient norms (lower = larger gradient)
    """
    model.train()  # Must be in training mode to compute gradients

    xc, yc = model.layer1.centers()
    if xc.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Inputs that require gradients
    x_in = xc.to(device).detach().requires_grad_(True)
    y_in = yc.to(device).detach().requires_grad_(True)

    # Forward pass to get output
    output = model.layer1.forward(x_in, y_in)
    f1 = output[0] + output[1]

    out_g2, out_a2 = model.layer2(f1.squeeze(-1), f1.squeeze(-1))
    f2 = f1 + out_g2 + out_a2

    out_g3, out_a3 = model.layer3(out_g2.squeeze(-1), out_a2.squeeze(-1))
    f3 = f2 + out_g3 + out_a3

    # Compute the gradient of f3 w.r.t. (x, y) → ∇f₃
    grad_x = torch.autograd.grad(f3.sum(), x_in, create_graph=True, retain_graph=True)[0]
    grad_y = torch.autograd.grad(f3.sum(), y_in, create_graph=True, retain_graph=True)[0]

    # Gradient norm
    grad_norm_sq = grad_x**2 + grad_y**2

    # Negative sign: maximizing gradient norm → minimizing loss
    return -grad_norm_sq.sum()



def analyze_node_coverage(
    model, X: np.ndarray, layer_id: int = 1, width_multiplier: float = 3.0,
) -> Dict:
    """
    Analyze the coverage of the data space by the particles in a given layer.

    Uses PCA to compare the covariance eigen-directions of node centers and data,
    calculates coverage ratios, etc., to assess whether the number of nodes is
    reasonable. Output metrics include:
    - Covariance determinant ratio (nodes / data)
    - First principal component angle
    - Data point coverage rate (proportion falling within particle Gaussian radii)

    Parameters
    ----------
    model : ResidualRBFNetwork
    X : (N, 2) array
        Input data
    layer_id : int
        Layer to analyze (1/2/3)
    width_multiplier : float
        Width multiplier threshold (coverage radius = h * width_multiplier)
    """
    layers = [model.layer1, model.layer2, model.layer3]
    layer = layers[layer_id - 1]
    layer_names = {1: "Layer 1", 2: "Layer 2", 3: "Layer 3"}

    xc, yc = layer.centers()
    centers_np = torch.stack([xc, yc], dim=1).detach().numpy()
    N = centers_np.shape[0]
    if N < 2:
        print(f"Warning: {layer_names[layer_id]} has too few nodes, cannot analyze")
        return {}

    w = layer.widths().detach().numpy()
    if w.ndim == 0 or len(w) == 1:
        w = np.full(N, w.item() if w.ndim == 0 else w[0])

    # PCA
    cov_c = np.cov(centers_np, rowvar=False)
    cov_d = np.cov(X, rowvar=False)
    eig_c = np.linalg.eigh(cov_c)
    eig_d = np.linalg.eigh(cov_d)
    eigvals_c, eigvecs_c = eig_c[0][::-1], eig_c[1][:, ::-1]
    eigvals_d, eigvecs_d = eig_d[0][::-1], eig_d[1][:, ::-1]

    cos_angle = np.abs(np.dot(eigvecs_c[:, 0], eigvecs_d[:, 0]))
    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

    # Coverage
    dists = cdist(X, centers_np)
    radii = w * width_multiplier
    coverage = np.mean(np.any(dists <= radii, axis=1)) * 100

    det_ratio = np.linalg.det(cov_c) / np.linalg.det(cov_d)

    print(f"\n===== {layer_names[layer_id]} Node Analysis =====")
    print(f"Num nodes: {N}")
    print(f"Covariance determinant ratio (nodes/data): {det_ratio:.2f}")
    print(f"First principal component angle: {angle_deg:.1f}°")
    print(f"Data point coverage: {coverage:.1f}%")

    if angle_deg < 20:
        print("\u2713 Shape matches well")
    else:
        print("\u26a0 Shape match is poor")
    if coverage > 90:
        print("\u2713 Coverage is sufficient")
    elif coverage > 70:
        print("\u2192 Coverage is moderate")
    else:
        print("\u26a0 Insufficient coverage, consider increasing node count or width")

    return {"det_ratio": det_ratio, "angle_deg": angle_deg, "coverage": coverage}
