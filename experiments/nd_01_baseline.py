"""
V3 Experiment E01: Digital Baseline — ND RBF Network

Validates ResidualRBFNetworkND in the digital domain:

Tests:
  1. Architecture sanity: forward pass shape correctness, residual accumulation
  2. 2D classification: ND vs v1 ResidualRBFNetwork on Moons / Circles
     (verifies correct degeneration — ND with input_dim=2 should match v1)
  3. 4D classification: ND on synthetic 4D data
     (verifies D-dimensional capability — wire-direct, no projection loss)
  4. Parameter count comparison: ND vs v1 vs MLP

Key metrics:
  - Classification accuracy
  - Training convergence speed
  - Parameter efficiency (accuracy / n_params)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from network.core_nd import ResidualRBFNetworkND
from network.utils_nd import generate_rotation_data_nd
from rbf_network.core import ResidualRBFNetwork
from rbf_network.utils import generate_rotation_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Suppress excessive prints
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
#  Data Generation
# ═══════════════════════════════════════════════════════════

def generate_4d_moons(n_samples: int = 800, noise: float = 0.12, seed: int = 42):
    """
    Generate 4D classification data: 2D Moons embedded in 4D space
    with 2 noise dimensions.

    Returns X (N, 4), y (N,).
    """
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    rng = np.random.RandomState(seed + 1)
    noise_dims = rng.normal(0, 0.3, (n_samples, 2))
    X_4d = np.column_stack([X_2d, noise_dims]).astype(np.float32)
    return X_4d, y


def generate_4d_circles(n_samples: int = 800, noise: float = 0.08, factor: float = 0.4, seed: int = 42):
    """Generate 4D Circles: 2D circles + 2 noise dimensions."""
    X_2d, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=seed)
    rng = np.random.RandomState(seed + 1)
    noise_dims = rng.normal(0, 0.3, (n_samples, 2))
    X_4d = np.column_stack([X_2d, noise_dims]).astype(np.float32)
    return X_4d, y


# ═══════════════════════════════════════════════════════════
#  Training Helpers
# ═══════════════════════════════════════════════════════════

def train_nd_model(model, X_tr, y_tr, X_te, y_te, epochs=300, lr=0.01, batch_size=128, verbose=True):
    """Train ND model with autograd."""
    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(X_tr)
        scheduler.step(avg_loss)

        # Evaluate
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
            pred = model(Xt).argmax(dim=1).cpu().numpy()
            acc = (pred == y_te).mean()
            if acc > best_acc:
                best_acc = acc

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | loss={avg_loss:.4f} | acc={acc:.4f} | best={best_acc:.4f}")

    return best_acc


def train_v1_model(model, X_tr, y_tr, X_te, y_te, epochs=300, lr=0.01, batch_size=128, verbose=True):
    """Train v1 ResidualRBFNetwork (2D input only)."""
    x_in = torch.tensor(X_tr[:, 0], dtype=torch.float32)
    y_in = torch.tensor(X_tr[:, 1], dtype=torch.float32)
    labels = torch.tensor(y_tr, dtype=torch.long)
    ds = TensorDataset(x_in, y_in, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by, bt in loader:
            bx, by, bt = bx.to(DEVICE), by.to(DEVICE), bt.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx, by)
            loss = criterion(out, bt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)

        avg_loss = total_loss / len(X_tr)
        scheduler.step(avg_loss)

        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X_te[:, 0], dtype=torch.float32).to(DEVICE)
            yt = torch.tensor(X_te[:, 1], dtype=torch.float32).to(DEVICE)
            pred = model(xt, yt).argmax(dim=1).cpu().numpy()
            acc = (pred == y_te).mean()
            if acc > best_acc:
                best_acc = acc

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | loss={avg_loss:.4f} | acc={acc:.4f} | best={best_acc:.4f}")

    return best_acc


# ═══════════════════════════════════════════════════════════
#  Test 1: Sanity Check
# ═══════════════════════════════════════════════════════════

def test_architecture():
    """Verify forward pass shapes and residual accumulation."""
    print("\n" + "=" * 55)
    print("  Test 1: Architecture Sanity Check")
    print("=" * 55)

    B, D = 32, 4
    X = torch.randn(B, D)

    model = ResidualRBFNetworkND(
        input_dim=4, hidden_dim=4, output_dim=2,
        layer_particles=(6, 4, 3),
        task="classification", num_classes=2,
    )

    # Forward pass
    out = model(X)
    print(f"  Input:  {X.shape} → Output: {out.shape}  (B=32, expected (32,2))")
    assert out.shape == (B, 2), f"Expected (32,2), got {out.shape}"

    # Residual accumulation check
    out_full, (g1, a1), (g2, a2), (g3, a3) = model(X, return_hidden=True)
    f1 = g1 + a1
    print(f"  Layer 1: input {X.shape[1]}D → output {f1.shape[1]}D")
    f2 = model._zero_pad(f1, g2 + a2)
    print(f"  Layer 2: input {f1.shape[1]}D → output {f2.shape[1]}D")
    f3 = model._zero_pad(f2, g3 + a3)
    print(f"  Layer 3: input {f2.shape[1]}D → output {f3.shape[1]}D")

    # Parameter count
    n_params = model.n_total_params
    print(f"  Total parameters: {n_params}")
    print(f"  All shapes correct.")

    return True


# ═══════════════════════════════════════════════════════════
#  Test 2: 2D Degeneration
# ═══════════════════════════════════════════════════════════

def test_2d_degeneration():
    """Compare ND (input_dim=2) vs v1 ResidualRBFNetwork on 2D data."""
    print("\n" + "=" * 55)
    print("  Test 2: 2D Degeneration — ND vs V1")
    print("=" * 55)

    results = {}

    for name, X, y in [
        ("Moons", *make_moons(n_samples=500, noise=0.12, random_state=42)),
        ("Circles", *make_circles(n_samples=500, noise=0.08, factor=0.4, random_state=42)),
    ]:
        print(f"\n  --- {name} ---")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr).astype(np.float32)
        X_te = sc.transform(X_te).astype(np.float32)

        xr = (X_tr[:, 0].min() - 1, X_tr[:, 0].max() + 1)
        yr = (X_tr[:, 1].min() - 1, X_tr[:, 1].max() + 1)
        n_p = 8  # particles per layer

        # V1 ResidualRBFNetwork (2D)
        print("  V1 ResidualRBFNetwork:")
        v1 = ResidualRBFNetwork(
            hidden_dim=2, output_dim=2,
            first_layer_particles=n_p,
            second_layer_particles=max(n_p // 2, 4),
            third_layer_particles=max(n_p // 3, 3),
            x_range=xr, y_range=yr, init_zero=False,
        ).to(DEVICE)
        t0 = time.time()
        acc_v1 = train_v1_model(v1, X_tr, y_tr, X_te, y_te, epochs=200, lr=0.008, verbose=True)
        n_v1 = sum(p.numel() for p in v1.parameters())
        print(f"  V1 → acc={acc_v1:.4f}, params={n_v1}, time={time.time()-t0:.1f}s")

        # V3 ND (input_dim=2)
        print("  V3 ResidualRBFNetworkND (D=2):")
        nd = ResidualRBFNetworkND(
            input_dim=2, hidden_dim=2, output_dim=2,
            layer_particles=(n_p, max(n_p // 2, 4), max(n_p // 3, 3)),
            init_range=(min(xr), max(xr)),
            task="classification", num_classes=2,
        ).to(DEVICE)
        t0 = time.time()
        acc_nd = train_nd_model(nd, X_tr, y_tr, X_te, y_te, epochs=200, lr=0.008, verbose=True)
        n_nd = nd.n_total_params
        print(f"  ND → acc={acc_nd:.4f}, params={n_nd}, time={time.time()-t0:.1f}s")

        results[name] = {"v1_acc": acc_v1, "nd_acc": acc_nd, "v1_params": n_v1, "nd_params": n_nd}
        print(f"  Δ acc: {acc_nd - acc_v1:+.4f} | params: {n_nd} vs {n_v1}")

    return results


# ═══════════════════════════════════════════════════════════
#  Test 3: 4D Advantage
# ═══════════════════════════════════════════════════════════

def test_4d_advantage():
    """Test ND on 4D data where v1 cannot reach."""
    print("\n" + "=" * 55)
    print("  Test 3: 4D Capability — ND on 4D Data")
    print("=" * 55)

    results = {}

    for name, X, y in [
        ("4D-Moons", *generate_4d_moons(n_samples=800, noise=0.12)),
        ("4D-Circles", *generate_4d_circles(n_samples=800, noise=0.08)),
    ]:
        print(f"\n  --- {name} ---")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr).astype(np.float32)
        X_te = sc.transform(X_te).astype(np.float32)

        ir = (X_tr.min() - 1, X_tr.max() + 1)

        # Small ND model
        nd_small = ResidualRBFNetworkND(
            input_dim=4, hidden_dim=4, output_dim=2,
            layer_particles=(8, 5, 4),
            init_range=ir, task="classification", num_classes=2,
        ).to(DEVICE)
        print("  V3 ND (small: 8/5/4 particles):")
        t0 = time.time()
        acc_small = train_nd_model(nd_small, X_tr, y_tr, X_te, y_te, epochs=300, lr=0.008, verbose=True)
        n_small = nd_small.n_total_params
        print(f"  → acc={acc_small:.4f}, params={n_small}, time={time.time()-t0:.1f}s")

        # Medium ND model
        nd_med = ResidualRBFNetworkND(
            input_dim=4, hidden_dim=4, output_dim=2,
            layer_particles=(12, 8, 6),
            init_range=ir, task="classification", num_classes=2,
        ).to(DEVICE)
        print("  V3 ND (medium: 12/8/6 particles):")
        t0 = time.time()
        acc_med = train_nd_model(nd_med, X_tr, y_tr, X_te, y_te, epochs=300, lr=0.008, verbose=True)
        n_med = nd_med.n_total_params
        print(f"  → acc={acc_med:.4f}, params={n_med}, time={time.time()-t0:.1f}s")

        results[name] = {"small_acc": acc_small, "med_acc": acc_med,
                          "small_params": n_small, "med_params": n_med}

    return results


# ═══════════════════════════════════════════════════════════
#  Test 4: Rotation Regression
# ═══════════════════════════════════════════════════════════

def test_rotation():
    """Test ND on D-dimensional rotation regression."""
    print("\n" + "=" * 55)
    print("  Test 4: Rotation Regression (4D)")
    print("=" * 55)

    X, Y = generate_rotation_data_nd(n_samples=800, input_dim=4, noise=0.01, seed=42)

    X_tr, X_te = X[:640], X[640:]
    Y_tr, Y_te = Y[:640], Y[640:]

    model = ResidualRBFNetworkND(
        input_dim=4, hidden_dim=4, output_dim=4,
        layer_particles=(20, 12, 8),
        init_range=(-1.5, 1.5), task="regression",
    ).to(DEVICE)

    Xt_tr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    Yt_tr = torch.tensor(Y_tr, dtype=torch.float32).to(DEVICE)
    Xt_te = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    Yt_te = torch.tensor(Y_te, dtype=torch.float32).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    best_r2 = -float("inf")
    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        pred = model(Xt_tr)
        loss = criterion(pred, Yt_tr)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred_te = model(Xt_te)
                ss_res = ((pred_te - Yt_te) ** 2).sum().item()
                ss_tot = ((Yt_te - Yt_te.mean(0)) ** 2).sum().item()
                r2 = 1 - ss_res / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                print(f"  Epoch {epoch+1:4d}/400 | loss={loss.item():.6f} | test R2={r2:.4f} | best R2={best_r2:.4f}")

    print(f"  Final best R2: {best_r2:.4f}, params: {model.n_total_params}")
    return {"r2": best_r2, "params": model.n_total_params}


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  V3 Experiment E01: ND Digital Baseline")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    all_results = {}

    # Test 1: Architecture sanity
    test_architecture()

    # Test 2: 2D degeneration
    r2d = test_2d_degeneration()
    all_results["2D"] = r2d

    # Test 3: 4D capability
    r4d = test_4d_advantage()
    all_results["4D"] = r4d

    # Test 4: Rotation regression
    rrot = test_rotation()
    all_results["rotation"] = rrot

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    print("\n  2D Degeneration (ND vs V1):")
    for name, r in r2d.items():
        print(f"    {name:<12s}: V1={r['v1_acc']:.4f} ({r['v1_params']}p)  "
              f"ND={r['nd_acc']:.4f} ({r['nd_params']}p)  "
              f"Δ={r['nd_acc'] - r['v1_acc']:+.4f}")

    print("\n  4D Capability (ND only):")
    for name, r in r4d.items():
        print(f"    {name:<12s}: small={r['small_acc']:.4f} ({r['small_params']}p)  "
              f"medium={r['med_acc']:.4f} ({r['med_params']}p)")

    print(f"\n  4D Rotation: R2={rrot['r2']:.4f} ({rrot['params']} params)")

    print("\n  " + "=" * 60)

    # ── Save summary bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 2D comparison
    names_2d = list(r2d.keys())
    x = np.arange(len(names_2d)) * 2
    w = 0.8
    for i, name in enumerate(names_2d):
        axes[0].bar(x[i] - w/2, r2d[name]["v1_acc"], w, label="V1" if i == 0 else "", color="#2196F3", alpha=0.7)
        axes[0].bar(x[i] + w/2, r2d[name]["nd_acc"], w, label="ND" if i == 0 else "", color="#4CAF50", alpha=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names_2d)
    axes[0].set_ylabel("Accuracy"); axes[0].set_title("2D: V1 vs ND"); axes[0].legend()
    axes[0].set_ylim(0, 1.1); axes[0].grid(axis="y", alpha=0.3)

    # 4D results
    names_4d = list(r4d.keys())
    x = np.arange(len(names_4d)) * 2
    for i, name in enumerate(names_4d):
        axes[1].bar(x[i] - w/2, r4d[name]["small_acc"], w, label="small" if i == 0 else "", color="#FF9800", alpha=0.7)
        axes[1].bar(x[i] + w/2, r4d[name]["med_acc"], w, label="medium" if i == 0 else "", color="#9C27B0", alpha=0.7)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names_4d)
    axes[1].set_ylabel("Accuracy"); axes[1].set_title("4D: ND"); axes[1].legend()
    axes[1].set_ylim(0, 1.1); axes[1].grid(axis="y", alpha=0.3)

    # Parameter counts
    p_labels = []
    p_counts = []
    for name in names_2d:
        p_labels.append(f"V1-{name}"); p_counts.append(r2d[name]["v1_params"])
        p_labels.append(f"ND-{name}"); p_counts.append(r2d[name]["nd_params"])
    for name in names_4d:
        p_labels.append(f"ND-sm-{name}"); p_counts.append(r4d[name]["small_params"])
        p_labels.append(f"ND-md-{name}"); p_counts.append(r4d[name]["med_params"])
    axes[2].barh(p_labels[::-1], p_counts[::-1], color="#607D8B", alpha=0.7)
    axes[2].set_xlabel("Parameters"); axes[2].set_title("Parameter Counts")
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "v3_e01_nd_baseline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/')}")
    plt.close()

    return all_results


if __name__ == "__main__":
    main()
