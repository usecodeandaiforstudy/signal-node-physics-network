"""
Experiment 4: SPSA Zero-Order Training vs Backpropagation

Verify: whether SPSA can train the network without any backpropagation.

Tests:
  4a. Rotation transformation regression: SPSA vs Autograd convergence comparison
  4b. Particle coordinate convergence analysis

Key metrics:
  - SPSA final R² vs Autograd final R²
  - Convergence speed (how many times more iterations SPSA needs)
  - Gradient estimation stability
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rbf_network import ResidualRBFNetwork
from rbf_network.utils import generate_rotation_data
from physical_simulator import PhysicalResidualRBFNetwork, NoiseConfig
from forward_training.spsa import SPSAOptimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def r2_score(true, pred):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def train_autograd(x_in, y_in, x_out, y_out, split, epochs=300):
    """Standard backpropagation training (baseline)"""
    train_ds = TensorDataset(
        torch.tensor(x_in[:split]), torch.tensor(y_in[:split]),
        torch.tensor(x_out[:split]).unsqueeze(1), torch.tensor(y_out[:split]).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model = ResidualRBFNetwork(
        hidden_dim=1, output_dim=2,
        first_layer_particles=20, second_layer_particles=15, third_layer_particles=15,
        x_range=(-10, 10), y_range=(-10, 10), init_zero=False,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by, tx, ty in train_loader:
            bx, by, tx, ty = bx.to(DEVICE), by.to(DEVICE), tx.to(DEVICE), ty.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx, by)
            loss = criterion(out[:, 0:1], tx) + criterion(out[:, 1:2], ty)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_loader))
        if (epoch + 1) % 50 == 0:
            print(f"  Autograd epoch {epoch+1:3d}/{epochs} | loss={losses[-1]:.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(x_in[split:]).to(DEVICE), torch.tensor(y_in[split:]).to(DEVICE))
        pred_x, pred_y = out[:, 0].cpu().numpy(), out[:, 1].cpu().numpy()
    r2 = (r2_score(x_out[split:], pred_x) + r2_score(y_out[split:], pred_y)) / 2

    return model, r2, losses


def train_spsa_physical(x_in, y_in, x_out, y_out, split, n_iter=500):
    """Train physical simulation layers with SPSA (layer-wise)"""
    x_test_x = x_in[split:].astype(np.float32)
    x_test_y = y_in[split:].astype(np.float32)

    # Initialize physical network (random)
    np.random.seed(42)
    from physical_simulator.components import NoiseConfig
    NoiseConfig.set_level("low")  # Low noise to simulate real environment

    # For simplicity, use physical single layer + SPSA
    # In practice SPSA trains the entire ResidualRBFNetwork
    from physical_simulator.layer import PhysicalRBFLayer

    # Create 3 layers
    layers = []
    prev_dim = None

    for i, (n_p, n_v) in enumerate([(20, 1), (15, 2), (15, 2)]):
        xc = np.random.uniform(-5, 5, n_p).astype(np.float32)
        yc = np.random.uniform(-5, 5, n_p).astype(np.float32)
        h = (10.0 / np.sqrt(n_p)) * np.ones(n_p, dtype=np.float32)
        W_gauss = np.random.uniform(-0.1, 0.1, (n_v, n_p)).astype(np.float32)
        W_act = np.random.uniform(-0.1, 0.1, (n_v, n_p)).astype(np.float32)
        layers.append(PhysicalRBFLayer(xc=xc, yc=yc, h=h, W_gauss=W_gauss, W_act=W_act, use_pu=True))

    physical = PhysicalResidualRBFNetwork(layers)

    # SPSA training (on the entire network)
    def loss_fn(params):
        physical.set_all_params_from_vector(params)
        out = physical.forward(x_in[:split].astype(np.float32),
                               y_in[:split].astype(np.float32))
        if out.ndim > 1 and out.shape[0] != split:
            out = out.T

        # MSE
        pred_x = out[:, 0]
        pred_y = out[:, 1]
        mse_x = np.mean((pred_x - x_out[:split]) ** 2)
        mse_y = np.mean((pred_y - y_out[:split]) ** 2)
        return float(mse_x + mse_y)

    init_params = physical.get_all_params_vector()
    opt = SPSAOptimizer(lr0=0.05, c0=0.02)
    params = opt.optimize(loss_fn, init_params, n_iter=n_iter, verbose=True, verbose_interval=50)

    physical.set_all_params_from_vector(params)
    NoiseConfig.set_level("none")
    out = physical.forward(x_test_x, x_test_y)
    if out.ndim > 1 and out.shape[0] != len(x_test_x):
        out = out.T
    pred_x, pred_y = out[:, 0], out[:, 1]
    r2 = (r2_score(x_out[split:], pred_x) + r2_score(y_out[split:], pred_y)) / 2

    history = opt.get_history()
    return r2, [h["loss"] for h in history]


def main():
    print("=" * 60)
    print("  Experiment 4: SPSA Zero-Order Training vs Backpropagation")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Data
    x_in, y_in, x_out, y_out = generate_rotation_data(n_samples=1000, angle=np.pi/6, noise=0.01)
    split = int(1000 * 0.8)

    # Autograd baseline
    print("\n>>> Autograd (Backpropagation) Training")
    model_bp, r2_bp, losses_bp = train_autograd(x_in, y_in, x_out, y_out, split, epochs=300)
    print(f"  Autograd R²: {r2_bp:.4f}")

    # SPSA training
    print("\n>>> SPSA (Zero-Order Optimization) Training")
    r2_spsa, losses_spsa = train_spsa_physical(x_in, y_in, x_out, y_out, split, n_iter=500)
    print(f"  SPSA R²:     {r2_spsa:.4f}")

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(losses_bp, alpha=0.7, label=f'Autograd (R²={r2_bp:.3f})')
    axes[0].plot(np.arange(len(losses_spsa)) * (len(losses_bp) / len(losses_spsa)),
                 losses_spsa, alpha=0.7, label=f'SPSA (R²={r2_spsa:.3f})')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Effective Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar(["Autograd (BP)", "SPSA (0-order)"], [r2_bp, r2_spsa],
                color=["blue", "orange"], alpha=0.7)
    axes[1].set_title("Final R² Comparison")
    axes[1].set_ylabel("R²"); axes[1].set_ylim(0, 1)
    for i, v in enumerate([r2_bp, r2_spsa]):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=12)

    fig.suptitle("Experiment 4: SPSA vs Backpropagation", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_spsa_vs_bp.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, '/')}")
    plt.close()

    # Ratio
    ratio = r2_spsa / r2_bp if r2_bp > 0 else 0
    print(f"\n  SPSA/Autograd R² ratio: {ratio:.2%}")

    return {"r2_bp": r2_bp, "r2_spsa": r2_spsa, "ratio": ratio}


if __name__ == "__main__":
    main()
