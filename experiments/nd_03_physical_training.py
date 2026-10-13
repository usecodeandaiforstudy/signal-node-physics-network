"""
V3 Physical Domain Training: Proto-Guided Pure Physical FD Training

Zero digital autograd — every gradient comes from finite differences
on the physical analog circuit simulator.

Compares:
  1. ProtoGuidedTrainerND (pure physical FD, from scratch)
  2. Digital baseline (autograd) for reference

Tests:
  - 2D Moons: verify pure physical training works from random init
  - Per-layer accuracy progression
  - Noise robustness at 4 levels
  - Decision boundary visualization
"""

import sys, os, time
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
from matplotlib.colors import ListedColormap

from network.core_nd import ResidualRBFNetworkND
from training.proto_nd import ProtoGuidedTrainerND
from physical_simulator import NoiseConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  Digital baseline (for reference)
# ═══════════════════════════════════════════════════════════

def train_digital(X_tr, y_tr, X_te, y_te, input_dim, particles, epochs=300):
    """Train digital ND model with autograd."""
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    n1, n2, n3 = particles
    model = ResidualRBFNetworkND(
        input_dim=input_dim, hidden_dim=4, output_dim=2,
        layer_particles=(n1, n2, n3),
        init_range=(-3, 3), task="classification", num_classes=2,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.008)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step(loss.item())

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_te).to(DEVICE)).argmax(1).cpu().numpy()
            acc = (pred == y_te).mean()
            if acc > best_acc:
                best_acc = acc

    return best_acc, model.n_total_params


# ═══════════════════════════════════════════════════════════
#  Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  V3 Physical Domain: Proto-Guided Pure FD Training")
    print("  ZERO digital autograd — all gradients from physical FD")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Data ──
    X, y = make_moons(n_samples=400, noise=0.12, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr_np = sc.fit_transform(X_tr).astype(np.float32)
    X_te_np = sc.transform(X_te).astype(np.float32)
    print(f"\n  Data: Moons, train={len(X_tr)}, test={len(X_te)}")
    print(f"  Input: 2D, classes=2")

    # ── Digital baseline ──
    particles = (3, 2, 2)
    print(f"\n  [1] Digital baseline (autograd, {particles} particles)...")
    t0 = time.time()
    acc_dig, n_dig = train_digital(X_tr_np, y_tr, X_te_np, y_te,
                                    input_dim=2, particles=particles, epochs=300)
    print(f"      Digital acc: {acc_dig:.4f}  params: {n_dig}  time: {time.time()-t0:.1f}s")

    # ── Proto-Guided Physical Training ──
    print(f"\n  [2] Proto-Guided Pure Physical FD Training ({particles} particles)")

    NoiseConfig.set_level("none")

    x_range = (X_tr_np[:, 0].min() - 1.0, X_tr_np[:, 0].max() + 1.0)
    y_range = (X_tr_np[:, 1].min() - 1.0, X_tr_np[:, 1].max() + 1.0)

    trainer = ProtoGuidedTrainerND(
        layer_sizes=particles,
        output_dims=(4, 3, 2),
        n_classes=2,
        input_dim=2,
        init_range=(min(x_range), max(x_range)),
        temperature=0.1,
        proto_reg_weight=0.1,
    )

    # Random init baseline
    input_dims = [2, 4, 3]  # D → d1 → d2
    rand_layers = [trainer._init_layer(s, id, d)
                   for s, id, d in zip(trainer.layer_sizes, input_dims, trainer.output_dims)]
    rand_protos = [trainer._init_prototypes(d) for d in trainer.output_dims]
    acc_rand = float(np.mean([
        (trainer.predict(rand_layers, rand_protos, X_te_np) == y_te).mean()
        for _ in range(3)
    ]))
    print(f"      Random init acc: {acc_rand:.4f} (expected ~{1/2:.2f})")

    # Train!
    t0 = time.time()
    layers, all_protos = trainer.train(
        X_tr_np, y_tr,
        epochs_per_layer=60, lr=0.015, eps=1e-3,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"      Training time: {train_time:.1f}s")

    # ── Evaluate ──
    NoiseConfig.set_level("none")
    acc_phys = (trainer.predict(layers, all_protos, X_te_np) == y_te).mean()
    delta = acc_phys - acc_rand
    print(f"\n  [3] Results:")
    print(f"      Digital (autograd):  {acc_dig:.4f} ({n_dig} params)")
    print(f"      Physical (pure FD):  {acc_phys:.4f} (delta={delta:+.4f})")

    # Per-layer accuracy
    print(f"\n  [4] Per-layer diagnostic:")
    cur_X = X_te_np.copy()
    layer_accs = []
    for i in range(3):
        acc_i = (trainer.predict_single(layers[i], all_protos[i], cur_X) == y_te).mean()
        layer_accs.append(acc_i)
        print(f"      Layer {i+1} only: {acc_i:.4f}")
        out_g, out_a = layers[i].forward(cur_X)
        cur_X = (out_g + out_a).astype(np.float32)

    # ── Noise evaluation ──
    print(f"\n  [5] Noise robustness:")
    noise_results = {}
    for level in ["none", "low", "medium", "high"]:
        NoiseConfig.set_level(level)
        preds = []
        for _ in range(3):
            preds.append(trainer.predict(layers, all_protos, X_te_np))
        stacked = np.stack(preds, axis=0)
        final = np.array([np.bincount(stacked[:, i]).argmax() for i in range(len(y_te))])
        a = (final == y_te).mean()
        noise_results[level] = a
        print(f"      {level:<8s}: {a:.4f}")
    NoiseConfig.set_level("none")

    # ── Decision boundary ──
    margin = 1.0
    xm, xM = X_te_np[:, 0].min() - margin, X_te_np[:, 0].max() + margin
    ym, yM = X_te_np[:, 1].min() - margin, X_te_np[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(xm, xM, 80), np.linspace(ym, yM, 80))
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    Z = trainer.predict(layers, all_protos, grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Decision boundary + particle positions
    ax = axes[0]
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]),
                alpha=0.35, levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_te_np[:, 0], X_te_np[:, 1], c=y_te,
               cmap=ListedColormap(["#CC0000", "#0000CC"]), s=10, alpha=0.6)
    colors = ["black", "orange", "green"]
    markers = ["X", "s", "D"]
    sizes = [160, 110, 80]
    for i, layer in enumerate(layers):
        c = layer.centers  # (D, N)
        ax.scatter(c[0], c[1], c=colors[i], marker=markers[i], s=sizes[i],
                   edgecolors="white", linewidths=0.8, label=f"L{i+1}", zorder=6)
    ax.set_xlim(xm, xM); ax.set_ylim(ym, yM)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Physical Proto-FD — Moons\n(acc={acc_phys:.3f}, digital={acc_dig:.3f})")
    ax.legend(fontsize=8)

    # Noise bar chart
    ax = axes[1]
    levels = list(noise_results.keys())
    vals = list(noise_results.values())
    clrs = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
    bars = ax.bar(levels, vals, color=clrs, alpha=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=11)
    ax.set_ylabel("Accuracy"); ax.set_title("Noise Robustness")
    ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.3)

    # Per-layer + Digital comparison
    ax = axes[2]
    labels = ["L1 only", "L2 only", "L3 only", "Ensemble", "Digital"]
    vals = layer_accs + [acc_phys, acc_dig]
    bar_colors = ["#2196F3"] * 3 + ["#4CAF50", "#FF9800"]
    bars = ax.bar(labels, vals, color=bar_colors, alpha=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Accuracy"); ax.set_title("Layer-wise Contribution")
    ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "v3_physical_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/')}")
    plt.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    print(f"  Digital (autograd):     {acc_dig:.4f}")
    print(f"  Physical (pure FD):     {acc_phys:.4f}")
    print(f"  Improvement over rand:  {delta:+.4f}")
    print(f"  Noise low → high:       {noise_results['low']:.4f} → {noise_results['high']:.4f}")
    print(f"  Training time:          {train_time:.1f}s")
    print("  " + "=" * 60)

    if delta > 0.1:
        print("\n  VERIFIED: Proto-guided FD enables pure physical training.")
    else:
        print("\n  NOTE: Limited improvement — may need more epochs or larger network.")

    return {
        "digital_acc": acc_dig, "physical_acc": acc_phys, "delta": delta,
        "noise": noise_results, "layer_accs": layer_accs, "time": train_time,
    }


if __name__ == "__main__":
    main()
