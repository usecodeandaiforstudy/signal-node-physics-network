"""
V3 Training utilities, visualization, data generation, and regularization
for D-dimensional RBF particle networks.

Adapted from v1 rbf_network/utils.py. Key differences:
  - Data generation functions support D-dimensional input
  - Visualization uses PCA to project D-dimensional states to 2D
  - Covariance regularization handles D-dimensional particle centers
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple, Dict
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
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
    """Standard single-epoch training loop for ResidualRBFNetworkND."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        X_batch, *targets = [b.to(device) for b in batch]
        optimizer.zero_grad()
        output = model(X_batch)
        target = targets[0] if len(targets) == 1 else targets
        loss = criterion(output, target.long() if is_classification else target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    is_classification: bool = True,
):
    """Standard evaluation for ResidualRBFNetworkND."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    for batch in dataloader:
        X_batch, *targets = [b.to(device) for b in batch]
        output = model(X_batch)
        target = targets[0] if len(targets) == 1 else targets
        if is_classification:
            if criterion:
                total_loss += criterion(output, target.long()).item() * X_batch.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
        else:
            if criterion:
                total_loss += criterion(output, target.float()).item() * X_batch.size(0)
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
    """Full training loop with evaluation."""
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
#  Data Generation
# ═══════════════════════════════════════════════════════════

def generate_rotation_data_nd(
    n_samples: int = 1000,
    input_dim: int = 4,
    noise: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate D-dimensional rotation regression data.

    Creates input vectors X uniformly distributed in [-1, 1]^D,
    then applies a random orthogonal rotation matrix to produce targets Y.
    The network must learn this rotation transformation.

    Parameters
    ----------
    n_samples : int
        Number of data points
    input_dim : int
        Dimensionality D
    noise : float
        Gaussian noise std added to targets
    seed : int
        Random seed for reproducibility

    Returns
    -------
    X : (n_samples, input_dim) ndarray
        Input data
    Y : (n_samples, input_dim) ndarray
        Rotated target (= X @ R^T)
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, (n_samples, input_dim)).astype(np.float32)

    # Generate random orthogonal rotation matrix via QR decomposition
    A = rng.randn(input_dim, input_dim)
    Q, R = np.linalg.qr(A)
    # Ensure det = +1 (proper rotation, no reflection)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    Y = X @ Q.T
    if noise > 0:
        Y += rng.normal(0, noise, Y.shape).astype(np.float32)

    return X, Y.astype(np.float32)


def generate_2d_data_for_nd(
    X_2d: np.ndarray,
) -> np.ndarray:
    """
    Wrap 2D classification data (e.g. from sklearn.make_moons) for ND network input.

    Parameters
    ----------
    X_2d : (N, 2) ndarray
        2D input data

    Returns
    -------
    X_nd : (N, 2) ndarray
        Same data, compatible with ND network's (batch, D) interface
    """
    return X_2d.astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  Visualization Functions
# ═══════════════════════════════════════════════════════════

def pca_project(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Project high-dimensional data to 2D via PCA for visualization.

    Parameters
    ----------
    X : (N, D) ndarray
        D-dimensional data
    n_components : int
        Target dimensionality (default 2)

    Returns
    -------
    X_2d : (N, n_components) ndarray
        PCA-projected data
    """
    if X.shape[1] <= n_components:
        return X[:, :n_components]
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def plot_decision_boundary(
    model, X, y, device=torch.device("cpu"),
    resolution=0.02, margin=0.5, ax=None, title="Decision Boundary",
):
    """
    Plot decision boundary for 2D input data.
    For D > 2, projects data to 2D via PCA first.

    Parameters
    ----------
    model : ResidualRBFNetworkND
    X : (N, D) ndarray
    y : (N,) ndarray (int)
    """
    if X.shape[1] > 2:
        X_2d = pca_project(X, 2)
        print(f"  (visualization: PCA-projected from {X.shape[1]}D to 2D)")
    else:
        X_2d = X

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = X_2d[:, 0].min() - margin, X_2d[:, 0].max() + margin
    y_min, y_max = X_2d[:, 1].min() - margin, X_2d[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )
    grid_2d = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    # To evaluate the model, we need D-dimensional input.
    # For D > 2: the grid in 2D PCA space does not have a unique inverse mapping.
    # We use the mean of the remaining PCA components as a constant background.
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        full_proj = pca.fit_transform(X)
        # Mean of the higher-order PCA components across all data
        remaining = X - pca.inverse_transform(full_proj)
        remaining_mean = remaining.mean(axis=0, keepdims=True)
        # For each grid point, reconstruct approximate D-dim input
        grid_full = pca.inverse_transform(grid_2d) + remaining_mean
    else:
        grid_full = grid_2d

    model.eval()
    with torch.no_grad():
        preds = model(
            torch.tensor(grid_full, dtype=torch.float32).to(device),
        ).argmax(dim=1).cpu().numpy().reshape(xx.shape)

    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    ax.contourf(xx, yy, preds, cmap=cmap_light, alpha=0.3)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=30)
    ax.set_title(title)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    return ax


def plot_particles(
    model, X=None, y=None, device=torch.device("cpu"), save_path=None,
):
    """
    Visualize particle centers across layers.
    For D > 2, uses PCA to project center positions to 2D.

    Parameters
    ----------
    model : ResidualRBFNetworkND
    X : (N, D) ndarray, optional
    y : (N,) ndarray, optional
    """
    model.eval()
    centers = model.get_all_centers()  # List[(D, N_layer_i) ndarray]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i in range(3):
        ax = axes[i]
        c_i = centers[i]  # (D, N_i) array of particle centers
        if c_i.shape[1] < 2:
            ax.text(0.5, 0.5, "Too few particles", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"Layer {i+1} Particle Centers")
            continue

        if c_i.shape[0] > 2:
            c_2d = pca_project(c_i.T, 2).T  # → (2, N_i)
        else:
            c_2d = c_i  # Already (2, N_i)

        # Plot input data if provided (2D projection)
        if X is not None and y is not None and i == 0:
            X_2d = pca_project(X, 2) if X.shape[1] > 2 else X
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", alpha=0.6, s=30)

        ax.scatter(c_2d[0], c_2d[1], c="black", marker="X", s=100,
                   label=f"L{i+1} Particles ({c_i.shape[1]})")
        ax.set_title(f"Layer {i+1} Particle Centers")
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history: dict):
    """Plot training loss and test accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("Training Loss")
    axes[0].grid(linestyle="--", alpha=0.6)
    if "test_acc" in history and history["test_acc"]:
        axes[1].plot(history["test_acc"])
        axes[1].set_title("Test Accuracy")
        axes[1].grid(linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
#  Covariance Regularization
# ═══════════════════════════════════════════════════════════

def compute_cov_regularization(model, data: torch.Tensor) -> torch.Tensor:
    """
    Frobenius norm difference between particle center covariance and data covariance.

    For D-dimensional networks: centers are (D, N) matrices, data is (batch, D).

    Parameters
    ----------
    model : ResidualRBFNetworkND
    data : (N, D) tensor

    Returns
    -------
    loss : sum of Frobenius norms of covariance differences across all layers
    """
    cov_data = torch.tensor(
        np.cov(data.numpy(), rowvar=False), dtype=torch.float32, device=data.device,
    )
    loss = 0.0
    for layer in [model.layer1, model.layer2, model.layer3]:
        centers = layer.centers()  # (D, N_particles)
        if centers.shape[1] < 2:
            continue
        # centers is (D, N); transpose to (N, D) for covariance
        c_t = centers.T
        centered = c_t - c_t.mean(dim=0, keepdim=True)
        cov_centers = (centered.T @ centered) / (c_t.shape[0] - 1)
        # Pad/truncate cov_centers to match cov_data shape
        d_data = cov_data.shape[0]
        d_cent = cov_centers.shape[0]
        if d_cent > d_data:
            cov_centers = cov_centers[:d_data, :d_data]
        elif d_cent < d_data:
            padded = torch.zeros(d_data, d_data, dtype=cov_centers.dtype, device=cov_centers.device)
            padded[:d_cent, :d_cent] = cov_centers
            cov_centers = padded
        diff = cov_centers - cov_data[:d_data, :d_data]
        loss += torch.norm(diff, p="fro") ** 2
    return loss


# ═══════════════════════════════════════════════════════════
#  Node Coverage Analysis
# ═══════════════════════════════════════════════════════════

def analyze_node_coverage(
    model, X: np.ndarray, layer_id: int = 1, width_multiplier: float = 3.0,
) -> dict:
    """
    Analyze coverage of D-dimensional data space by particles in a given layer.

    Uses PCA to compare covariance structure between particle centers and data.

    Parameters
    ----------
    model : ResidualRBFNetworkND
    X : (N, D) ndarray
    layer_id : int
        Layer index (1/2/3)
    width_multiplier : float
        Coverage radius = h * width_multiplier

    Returns
    -------
    Dict with coverage metrics
    """
    layers = [model.layer1, model.layer2, model.layer3]
    layer = layers[layer_id - 1]

    centers = layer.centers()  # (D, N)
    centers_np = centers.detach().numpy().T  # → (N, D)
    N = centers_np.shape[0]
    if N < 2:
        print(f"Warning: Layer {layer_id} has too few nodes, cannot analyze")
        return {}

    widths = layer.widths().detach().numpy()
    if widths.ndim == 0 or len(widths) == 1:
        w = np.full(N, widths.item() if widths.ndim == 0 else widths[0])
    else:
        w = widths

    # PCA comparison for top 2 components
    X_2d = pca_project(X, 2) if X.shape[1] > 2 else X
    centers_2d = pca_project(centers_np, 2) if centers_np.shape[1] > 2 else centers_np

    cov_c = np.cov(centers_2d, rowvar=False)
    cov_d = np.cov(X_2d, rowvar=False)
    eig_c = np.linalg.eigh(cov_c)
    eig_d = np.linalg.eigh(cov_d)
    eigvecs_c = eig_c[1][:, ::-1]
    eigvecs_d = eig_d[1][:, ::-1]

    cos_angle = np.abs(np.dot(eigvecs_c[:, 0], eigvecs_d[:, 0]))
    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

    # Coverage in D-dimensional space
    from scipy.spatial.distance import cdist
    dists = cdist(X, centers_np)
    radii = w * width_multiplier
    coverage = np.mean(np.any(dists <= radii, axis=1)) * 100

    det_ratio = np.linalg.det(cov_c) / (np.linalg.det(cov_d) + 1e-12)

    print(f"\n===== Layer {layer_id} Node Analysis =====")
    print(f"Num nodes: {N}, Input dim: {X.shape[1]}")
    print(f"Covariance determinant ratio (nodes/data): {det_ratio:.2f}")
    print(f"First principal component angle: {angle_deg:.1f}°")
    print(f"Data point coverage: {coverage:.1f}%")

    if angle_deg < 20:
        print("  Shape matches well")
    else:
        print("  Shape match is poor")
    if coverage > 90:
        print("  Coverage is sufficient")
    elif coverage > 70:
        print("  Coverage is moderate")
    else:
        print("  Insufficient coverage, consider increasing node count or width")

    return {"det_ratio": det_ratio, "angle_deg": angle_deg, "coverage": coverage}
