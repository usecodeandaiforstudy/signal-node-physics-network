"""
Experiment 3: Noise Tolerance

Verify the impact of physical non-idealities (thermal noise, process mismatch, memristor drift) on inference accuracy.

Method:
  1. Train ResidualRBFNetwork with autograd (digital domain)
  2. Sync parameters to PhysicalResidualRBFNetwork
  3. Test physical inference accuracy at 4 noise levels
  4. Plot accuracy vs noise curve

Output: Noise tolerance curve demonstrating network robustness to non-idealities.
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
from rbf_network.utils import generate_rotation_data, generate_electric_field, compute_cov_regularization
from physical_simulator import PhysicalResidualRBFNetwork, NoiseConfig
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_digital_moons():
    """Train Moons classification model"""
    X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_tr[:, 0], dtype=torch.float32),
        torch.tensor(X_tr[:, 1], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = ResidualRBFNetwork(
        hidden_dim=1, output_dim=2,
        first_layer_particles=15, second_layer_particles=15, third_layer_particles=15,
        x_range=(X_tr[:, 0].min(), X_tr[:, 0].max()),
        y_range=(X_tr[:, 1].min(), X_tr[:, 1].max()),
        init_zero=False,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        for bx, by, bt in train_loader:
            bx, by, bt = bx.to(DEVICE), by.to(DEVICE), bt.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx, by)
            loss = criterion(pred, bt)
            loss.backward()
            optimizer.step()

    model.eval()
    return model, X_te, y_test


def test_noise_tolerance_classification(model, X_test, y_test):
    """Test classification accuracy under different noise levels"""
    physical = PhysicalResidualRBFNetwork.from_digital(model)
    noise_levels = ["none", "low", "medium", "high"]
    accuracies = []

    for level in noise_levels:
        NoiseConfig.set_level(level)

        preds = []
        x_np = np.asarray(X_test)
        for i in range(0, len(X_test), 100):
            batch_x = x_np[i:i+100, 0].astype(np.float32)
            batch_y = x_np[i:i+100, 1].astype(np.float32)

            # Multiple samples averaged (noise is random)
            batch_preds = []
            for _ in range(5):
                out = physical.forward(batch_x, batch_y)
                if out.ndim > 1 and out.shape[0] != len(batch_x):
                    out = out.T
                batch_preds.append(out)
            out = np.mean(batch_preds, axis=0)
            preds.append(out.argmax(axis=1))

        preds = np.concatenate(preds)
        acc = (preds == y_test).mean()
        accuracies.append(acc)
        print(f"  Noise '{level}': accuracy = {acc:.4f}")

    return noise_levels, accuracies


def test_noise_tolerance_regression():
    """Test noise tolerance for regression task"""
    print("\n--- Regression noise tolerance ---")

    x_in, y_in, x_out, y_out = generate_rotation_data(n_samples=1000, angle=np.pi/6, noise=0.01)
    split = int(1000 * 0.8)

    train_ds = TensorDataset(
        torch.tensor(x_in[:split]), torch.tensor(y_in[:split]),
        torch.tensor(x_out[:split]).unsqueeze(1), torch.tensor(y_out[:split]).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = ResidualRBFNetwork(
        hidden_dim=1, output_dim=2,
        first_layer_particles=20, second_layer_particles=15, third_layer_particles=15,
        x_range=(-10, 10), y_range=(-10, 10), init_zero=False,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        for bx, by, tx, ty in train_loader:
            bx, by, tx, ty = bx.to(DEVICE), by.to(DEVICE), tx.to(DEVICE), ty.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx, by)
            loss = criterion(out[:, 0:1], tx) + criterion(out[:, 1:2], ty)
            loss.backward()
            optimizer.step()

    model.eval()
    physical = PhysicalResidualRBFNetwork.from_digital(model)

    noise_levels = ["none", "low", "medium", "high"]
    r2_values = []

    x_test_x = x_in[split:].astype(np.float32)
    x_test_y = y_in[split:].astype(np.float32)
    y_test_x = x_out[split:]
    y_test_y = y_out[split:]

    # Digital baseline
    with torch.no_grad():
        out_d = model(torch.tensor(x_test_x), torch.tensor(x_test_y)).cpu().numpy()

    def r2(true, pred):
        ss_res = ((pred - true) ** 2).sum()
        ss_tot = ((true - true.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    r2_d = (r2(y_test_x, out_d[:, 0]) + r2(y_test_y, out_d[:, 1])) / 2
    print(f"  Digital baseline R²: {r2_d:.4f}")

    for level in noise_levels:
        NoiseConfig.set_level(level)
        preds = []
        for _ in range(5):
            out = physical.forward(x_test_x, x_test_y)
            if out.ndim > 1 and out.shape[0] != len(x_test_x):
                out = out.T
            preds.append(out)
        out_p = np.mean(preds, axis=0)
        r2_val = (r2(y_test_x, out_p[:, 0]) + r2(y_test_y, out_p[:, 1])) / 2
        r2_values.append(r2_val)
        print(f"  Noise '{level}': R² = {r2_val:.4f}")

    return noise_levels, r2_values


def main():
    print("=" * 60)
    print("  Experiment 3: Noise Tolerance")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Classification task
    print("\n>>> Classification task: Moons")
    model_cls, X_test_cls, y_test_cls = train_digital_moons()
    noise_levels, accuracies = test_noise_tolerance_classification(model_cls, X_test_cls, y_test_cls)

    # Regression task
    print("\n>>> Regression task: Rotation transformation")
    noise_levels, r2_values = test_noise_tolerance_regression()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(noise_levels, accuracies, 'o-', linewidth=2, markersize=8)
    axes[0].set_title("Classification Accuracy vs Noise")
    axes[0].set_xlabel("Noise Level"); axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1); axes[0].grid(alpha=0.3)

    axes[1].plot(noise_levels, r2_values, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_title("Regression R² vs Noise")
    axes[1].set_xlabel("Noise Level"); axes[1].set_ylabel("R²")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Experiment 3: Noise Tolerance", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_noise_tolerance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, '/')}")
    plt.close()

    return {"accuracies": accuracies, "r2_values": r2_values}


if __name__ == "__main__":
    main()
