"""
V3 Experiment E02: Physical Forward Verification — ND Network

Validates PhysicalRBFLayerND and PhysicalResidualRBFNetworkND:

Tests:
  1. Parameter sync: digital → physical, verify per-layer outputs match
     (noise off, all non-ideality sources disabled)
  2. Full network consistency: 3-layer residual chain matches digital
  3. Noise tolerance: evaluate at 4 noise levels (none / low / medium / high)
  4. Layer-wise analysis: per-layer digital-vs-physical correlation

Key metrics:
  - Physical vs digital output correlation (cosine similarity, R²)
  - Accuracy degradation across noise levels
  - Per-layer consistency
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from network.core_nd import ResidualRBFNetworkND
from simulator.layer_nd import PhysicalRBFLayerND
from simulator.network_nd import PhysicalResidualRBFNetworkND
from network.utils_nd import generate_rotation_data_nd
from physical_simulator import NoiseConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between flattened arrays."""
    a_f = a.ravel(); b_f = b.ravel()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-8))


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))


def train_digital_model(X_tr, y_tr, input_dim, particles, epochs=200):
    """Train a digital ND model for testing."""
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    n1, n2, n3 = particles
    model = ResidualRBFNetworkND(
        input_dim=input_dim, hidden_dim=4, output_dim=input_dim,
        layer_particles=(n1, n2, n3),
        init_range=(-2, 2), task="regression",
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════
#  Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  V3 Experiment E02: Physical Forward Verification")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Generate and train ──
    X, Y = generate_rotation_data_nd(n_samples=500, input_dim=4, noise=0.01, seed=42)
    X_tr, X_te = X[:400], X[400:]
    Y_tr, Y_te = Y[:400], Y[400:]

    print("\n  [1] Training digital model (D=4, 8/6/4 particles)...")
    t0 = time.time()
    digital = train_digital_model(X_tr, Y_tr, input_dim=4, particles=(8, 6, 4), epochs=200)
    print(f"      Done in {time.time()-t0:.1f}s")

    # Digital evaluation
    with torch.no_grad():
        Xt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        Yt = torch.tensor(Y_te, dtype=torch.float32).to(DEVICE)
        pred_digital = digital(Xt).cpu().numpy()
    r2_digital = r2_score(Y_te, pred_digital)
    print(f"      Digital R2: {r2_digital:.4f}")

    # ── Convert to physical ──
    print("\n  [2] Converting to physical layers...")
    physical = PhysicalResidualRBFNetworkND.from_digital(digital)
    print(f"      Total physical params: {physical.n_total_params}")

    # ── Per-layer consistency check ──
    print("\n  [3] Per-layer digital vs physical consistency (noise off):")
    NoiseConfig.set_level("none")

    pl_info = [
        ("Layer 1", physical.layer1, digital.layer1),
        ("Layer 2", physical.layer2, digital.layer2),
        ("Layer 3", physical.layer3, digital.layer3),
    ]

    # Build inputs for each layer
    cur_X = X_te.astype(np.float32)
    for name, pl, dl in pl_info:
        # Digital forward
        with torch.no_grad():
            Xt = torch.tensor(cur_X, dtype=torch.float32)
            dg, da = dl(Xt)
            d_out = (dg + da).cpu().numpy()

        # Physical forward
        pg, pa = pl.forward(cur_X)
        p_out = pg + pa

        cos = cosine_similarity(d_out, p_out)
        r2 = r2_score(d_out, p_out)
        max_err = np.abs(d_out - p_out).max()
        print(f"      {name}: cosine={cos:.6f}  R2={r2:.6f}  max_err={max_err:.2e}")

        # Next layer input = output
        cur_X = d_out

    # ── Full network consistency ──
    print("\n  [4] Full network digital vs physical:")
    pred_physical = physical.forward(X_te.astype(np.float32))
    cos_full = cosine_similarity(pred_digital, pred_physical)
    r2_full = r2_score(pred_digital, pred_physical)
    print(f"      cosine={cos_full:.6f}  R2={r2_full:.6f}")

    # ── Noise evaluation ──
    print("\n  [5] Physical noise evaluation:")
    noise_results = {}
    for level in ["none", "low", "medium", "high"]:
        NoiseConfig.set_level(level)
        preds = []
        for _ in range(3):
            p = physical.forward(X_te.astype(np.float32))
            preds.append(p)
        avg = np.mean(preds, axis=0)
        r2 = r2_score(Y_te, avg)
        noise_results[level] = r2
        print(f"      {level:<8s}: R2={r2:.6f}")

    NoiseConfig.set_level("none")

    # ── Summary plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter: digital vs physical
    axes[0].scatter(pred_digital.ravel(), pred_physical.ravel(), alpha=0.5, s=10)
    axes[0].plot([pred_digital.min(), pred_digital.max()],
                 [pred_digital.min(), pred_digital.max()], 'r--', lw=1)
    axes[0].set_xlabel("Digital prediction"); axes[0].set_ylabel("Physical prediction")
    axes[0].set_title(f"Digital vs Physical (cos={cos_full:.4f})")
    axes[0].grid(alpha=0.3)

    # Noise bar chart
    levels = list(noise_results.keys())
    values = list(noise_results.values())
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
    bars = axes[1].bar(levels, values, color=colors, alpha=0.7)
    for b, v in zip(bars, values):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                     f"{v:.3f}", ha="center", fontsize=11)
    axes[1].set_ylabel("R2"); axes[1].set_title("Noise Robustness")
    axes[1].set_ylim(0, 1.05); axes[1].grid(axis="y", alpha=0.3)

    # Per-layer cosine
    Xt = torch.tensor(X_te, dtype=torch.float32)
    l1_digital = digital.layer1(Xt)
    l1_digital_fused = (l1_digital[0] + l1_digital[1]).cpu().numpy()
    l1_phys = physical.layer1.forward(X_te.astype(np.float32))
    l1_phys_fused = l1_phys[0] + l1_phys[1]
    l1_cos = cosine_similarity(l1_digital_fused, l1_phys_fused)

    l2_digital = digital.layer2(torch.tensor(l1_digital_fused))
    l2_digital_fused = (l2_digital[0] + l2_digital[1]).cpu().numpy()
    l2_phys = physical.layer2.forward(l1_digital_fused)
    l2_phys_fused = l2_phys[0] + l2_phys[1]
    l2_cos = cosine_similarity(l2_digital_fused, l2_phys_fused)

    l3_digital = digital.layer3(torch.tensor(l2_digital_fused))
    l3_digital_fused = (l3_digital[0] + l3_digital[1]).cpu().numpy()
    l3_phys = physical.layer3.forward(l2_digital_fused)
    l3_phys_fused = l3_phys[0] + l3_phys[1]
    l3_cos = cosine_similarity(l3_digital_fused, l3_phys_fused)

    layer_names = ["L1", "L2", "L3"]
    axes[2].bar(layer_names, [l1_cos, l2_cos, l3_cos],
                color="#2196F3", alpha=0.7)
    axes[2].set_ylabel("Cosine Similarity"); axes[2].set_title("Per-Layer Consistency")
    axes[2].set_ylim(0.99, 1.001); axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "v3_e02_nd_physical.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/')}")
    plt.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Digital R2 (ground truth):     {r2_digital:.4f}")
    print(f"  Physical R2 (noise-free):      {r2_score(Y_te, pred_physical):.4f}")
    print(f"  Digital-vs-Physical cos:       {cos_full:.4f}")
    print(f"  Noise low → high R2 drop:      {noise_results['none']:.4f} → {noise_results['high']:.4f}")
    print("  " + "=" * 60)

    return {
        "digital_r2": r2_digital,
        "physical_r2_noise_free": r2_score(Y_te, pred_physical),
        "cos_similarity": cos_full,
        "noise_results": noise_results,
    }


if __name__ == "__main__":
    main()
