"""
Experiment 5: HiMAL Layer-Wise Training vs End-to-End Training

Verify: whether layer-wise independent training (no cross-layer backpropagation) can match end-to-end training accuracy.

Method:
  - HiMAL: Dual networks, layer-wise training, transition region alignment
  - Baseline: Standard autograd end-to-end training
  - Compare final accuracy

Key metrics:
  - HiMAL final accuracy vs end-to-end baseline
  - Intermediate representation quality after each layer's training
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
from forward_training.himal import HiMALTrainer
from physical_simulator.components import NoiseConfig
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def r2_score(true, pred):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def baseline_end_to_end(X, y, epochs=300):
    """End-to-end autograd training (baseline)"""
    # Convert to regression format
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    train_ds = TensorDataset(
        torch.tensor(X_tr[:, 0], dtype=torch.float32),
        torch.tensor(X_tr[:, 1], dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = ResidualRBFNetwork(
        hidden_dim=1, output_dim=1,
        first_layer_particles=20, second_layer_particles=12, third_layer_particles=12,
        x_range=(X[:, 0].min(), X[:, 0].max()),
        y_range=(X[:, 1].min(), X[:, 1].max()),
        init_zero=False,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        for bx, by, bt in train_loader:
            bx, by, bt = bx.to(DEVICE), by.to(DEVICE), bt.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx, by)
            loss = criterion(pred, bt)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te[:, 0]).to(DEVICE),
                     torch.tensor(X_te[:, 1]).to(DEVICE)).cpu().numpy()

    r2 = r2_score(y_te.flatten(), pred.flatten())
    print(f"  End-to-end R²: {r2:.4f}")
    return r2


def train_himal_regression(X, y):
    """HiMAL layer-wise training (fully forward, no backpropagation)"""
    print("\n--- HiMAL Layer-Wise Training (Regression) ---")

    # Normalize
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / (y_std + 1e-8)

    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_norm[:split], y_norm[split:]

    trainer = HiMALTrainer(
        layer_sizes=(20, 12, 12),
        output_dims=(1, 1, 1),
        x_range=(X[:, 0].min(), X[:, 0].max()),
        y_range=(X[:, 1].min(), X[:, 1].max()),
        alignment_weight=0.3,
    )

    layers_A, layers_B, data = trainer.train(
        X_train, y_train,
        split_method="median_x",
        epochs_per_layer=150,
        lr=0.01,
        verbose=True,
    )

    # Evaluation
    pred = trainer.predict(layers_A, layers_B, X_test)
    if pred.ndim > 1:
        pred = pred.reshape(-1)

    r2 = r2_score(y_test.flatten() if y_test.ndim > 1 else y_test, pred.flatten())

    # Denormalized evaluation
    y_test_orig = y_test * y_std + y_mean
    pred_orig = pred * y_std + y_mean
    mse_orig = np.mean((y_test_orig.flatten() - pred_orig.flatten()) ** 2)

    print(f"  HiMAL R² (normalized): {r2:.4f}")
    print(f"  HiMAL MSE (original):  {mse_orig:.4f}")

    return r2, layers_A, layers_B, trainer, X_train, y_train


def main():
    print("=" * 60)
    print("  Experiment 5: HiMAL Layer-Wise Training vs End-to-End Training")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Generate data
    x_in, y_in, x_out, y_out = generate_rotation_data(n_samples=800, angle=np.pi/6, noise=0.01)
    X = np.column_stack([x_in, y_in]).astype(np.float32)
    y = x_out.astype(np.float32).reshape(-1, 1)  # Only predict x_rotated

    # Baseline
    print("\n>>> Baseline: End-to-end (autograd)")
    r2_e2e = baseline_end_to_end(X, y, epochs=300)

    # HiMAL
    print("\n>>> HiMAL: Layer-wise (forward-only)")
    r2_himal, layers_A, layers_B, trainer, X_tr, y_tr = train_himal_regression(X, y)

    # Comparison plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["End-to-end (BP)", "HiMAL (layer-wise)"], [r2_e2e, r2_himal],
           color=["blue", "green"], alpha=0.7)
    ax.set_title("Experiment 5: HiMAL vs End-to-end")
    ax.set_ylabel("R²"); ax.set_ylim(0, 1)
    for i, v in enumerate([r2_e2e, r2_himal]):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_himal_vs_e2e.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, '/')}")
    plt.close()

    ratio = r2_himal / r2_e2e if r2_e2e > 0 else 0
    print(f"\n  HiMAL/E2E R² ratio: {ratio:.2%}")

    return {"r2_e2e": r2_e2e, "r2_himal": r2_himal, "ratio": ratio}


if __name__ == "__main__":
    main()
