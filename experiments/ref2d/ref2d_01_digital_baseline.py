"""
Experiment 1: Digital Baseline

Train RBFNetwork and ResidualRBFNetwork with standard autograd,
establishing digital-domain performance baselines (accuracy / R² / particle convergence quality).

Test tasks:
  1a. Moons binary classification (RBFNetwork)
  1b. Rotation transformation regression (RBFNetwork)
  1c. Electric field fitting (ResidualRBFNetwork)

Output: Baseline metrics serving as reference standards for subsequent physical experiments.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from rbf_network import RBFNetwork, ResidualRBFNetwork
from rbf_network.utils import generate_rotation_data, generate_electric_field, compute_cov_regularization
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_moons():
    """Moons binary classification baseline"""
    print("\n--- 1a: Moons Binary Classification (RBFNetwork) ---")

    X, y = make_moons(n_samples=800, noise=0.1, random_state=42)
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

    model = RBFNetwork(
        hidden_dim=1, output_dim=1, n_particles=20,
        x_range=(X_tr[:, 0].min(), X_tr[:, 0].max()),
        y_range=(X_tr[:, 1].min(), X_tr[:, 1].max()),
        task="classification", num_classes=2,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        for bx, by, bt in train_loader:
            bx, by, bt = bx.to(DEVICE), by.to(DEVICE), bt.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx, by), bt)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_te[:, 0], dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(X_te[:, 1], dtype=torch.float32).to(DEVICE)
        acc = (model(x_t, y_t).argmax(1).cpu() == torch.tensor(y_test)).float().mean().item()

    print(f"  Baseline accuracy: {acc:.4f}")
    return {"task": "moons", "metric": "accuracy", "value": acc}


def test_rotation():
    """Rotation transformation regression baseline"""
    print("\n--- 1b: Rotation Transformation Regression (RBFNetwork) ---")

    x_in, y_in, x_out, y_out = generate_rotation_data(n_samples=2000, angle=np.pi/6, noise=0.01)
    split = int(2000 * 0.8)

    train_ds = TensorDataset(
        torch.tensor(x_in[:split]), torch.tensor(y_in[:split]),
        torch.tensor(x_out[:split]).unsqueeze(1), torch.tensor(y_out[:split]).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model = RBFNetwork(
        hidden_dim=1, output_dim=2, n_particles=20,
        x_range=(-10, 10), y_range=(-10, 10), task="regression",
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        model.train()
        for bx, by, tx, ty in train_loader:
            bx, by, tx, ty = bx.to(DEVICE), by.to(DEVICE), tx.to(DEVICE), ty.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx, by)
            loss = criterion(out[:, 0:1], tx) + criterion(out[:, 1:2], ty)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_in[split:], dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_in[split:], dtype=torch.float32).to(DEVICE)
        out = model(x_t, y_t)
        pred_x, pred_y = out[:, 0].cpu().numpy(), out[:, 1].cpu().numpy()

    def r2(true, pred):
        ss_res = ((pred - true) ** 2).sum()
        ss_tot = ((true - true.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    r2x = r2(x_out[split:], pred_x)
    r2y = r2(y_out[split:], pred_y)
    avg_r2 = (r2x + r2y) / 2

    print(f"  Baseline R²_x: {r2x:.4f}, R²_y: {r2y:.4f}, Avg R²: {avg_r2:.4f}")
    return {"task": "rotation", "metric": "R²", "value": avg_r2}


def test_electric():
    """Electric field fitting baseline"""
    print("\n--- 1c: Electric Field Fitting (ResidualRBFNetwork) ---")

    charges = [(1e-9, 0, 2), (-1e-9, 0, -2), (0.5e-9, 3, 0)]
    df = generate_electric_field(charges, resolution=0.15)
    X_raw = df[["x_position", "y_position"]].values.astype(np.float32)
    y_raw = df["electric_field_magnitude"].values.astype(np.float32).reshape(-1, 1)

    X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
    y_mean, y_std = y_raw.mean(), y_raw.std()
    X = (X_raw - X_mean) / (X_std + 1e-8)
    y = (y_raw - y_mean) / (y_std + 1e-8)

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = ResidualRBFNetwork(
        hidden_dim=1, output_dim=1,
        first_layer_particles=3, second_layer_particles=6, third_layer_particles=9,
        x_range=(X[:, 0].min(), X[:, 0].max()),
        y_range=(X[:, 1].min(), X[:, 1].max()),
        init_zero=False,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        model.train()
        for bx, by in dataloader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx[:, 0], bx[:, 1])
            mse = criterion(pred, by)
            reg = compute_cov_regularization(model, bx.cpu())
            loss = mse + 0.01 * reg
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_all = model(torch.tensor(X[:, 0]).to(DEVICE), torch.tensor(X[:, 1]).to(DEVICE))
        mse = criterion(pred_all, torch.tensor(y).to(DEVICE)).item()

    # Particle-charge distance
    xc, yc = model.layer1.centers()
    x_real = xc.detach().numpy() * X_std[0] + X_mean[0]
    y_real = yc.detach().numpy() * X_std[1] + X_mean[1]
    charge_positions = np.array([(0, 2), (0, -2), (3, 0)])
    avg_dist = sum(
        min(np.sqrt((cx - qx)**2 + (cy - qy)**2) for qx, qy in charge_positions)
        for cx, cy in zip(x_real, y_real)
    ) / 3

    print(f"  Baseline MSE: {mse:.6f} (normalized)")
    print(f"  Avg particle-charge distance: {avg_dist:.3f}")
    return {"task": "electric", "metric": "MSE", "value": mse, "avg_dist": avg_dist}


def main():
    print("=" * 60)
    print("  Experiment 1: Digital Baseline")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    results = {
        "moons": test_moons(),
        "rotation": test_rotation(),
        "electric": test_electric(),
    }

    print("\n" + "=" * 60)
    print("  Baseline Results Summary:")
    for name, r in results.items():
        print(f"  {name}: {r['metric']} = {r['value']:.4f}")

    return results


if __name__ == "__main__":
    results = main()
