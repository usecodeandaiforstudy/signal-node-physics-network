"""
Experiment: Physical RBF with Finite-Difference Per-Layer Fine-Tuning

TRUE physical training experiment:
  - RBF layers → PhysicalRBFLayer (differential amplifier, Gilbert squarer,
    BJT exponential, current mirror, memristor MVM)
  - Fine-tuning: per-layer FINITE DIFFERENCES on physical components
    (zero autograd on the physical side — every gradient comes from
     numerical perturbation of the actual analog circuit chain)

Strategy:
  1. Digital pretrain (ResidualRBFNetwork, end-to-end autograd) → base weights
  2. Convert RBF params to PhysicalRBFLayer (one per digital layer)
  3. Physical per-layer FD fine-tuning (each layer tuned independently)
  4. Evaluate on physical simulator at 4 noise levels

The FD step is genuine physical training: each loss evaluation goes through
the 6-component analog circuit model, not PyTorch tensor math.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from rbf_network.core import ResidualRBFNetwork, RBFLayer
from physical_simulator.layer import PhysicalRBFLayer
from physical_simulator import NoiseConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
#  Digital Pretrain (ResidualRBFNetwork, autograd)
# ──────────────────────────────────────────────

def digital_pretrain(X_tr, y_tr, particles, n_epochs=200):
    """Train ResidualRBFNetwork end-to-end with autograd."""
    x_range = (X_tr[:, 0].min() - 0.5, X_tr[:, 0].max() + 0.5)
    y_range = (X_tr[:, 1].min() - 0.5, X_tr[:, 1].max() + 0.5)

    m = ResidualRBFNetwork(
        hidden_dim=2, output_dim=2,
        first_layer_particles=particles,
        second_layer_particles=max(particles // 2, 4),
        third_layer_particles=max(particles // 3, 3),
        x_range=x_range, y_range=y_range, init_zero=False,
    )
    opt = optim.Adam(m.parameters(), lr=0.005)
    xt = torch.tensor(X_tr[:, 0]); yt = torch.tensor(X_tr[:, 1])
    lb = torch.tensor(y_tr).long()
    for _ in range(n_epochs):
        m.train(); opt.zero_grad()
        nn.CrossEntropyLoss()(m(xt, yt), lb).backward(); opt.step()
    m.eval()
    return m


# ──────────────────────────────────────────────
#  Physical Finite-Difference Per-Layer Fine-Tuning
# ──────────────────────────────────────────────

def physical_fd_train_one_layer(phys_layers, layer_idx, X, y, n_epochs=10, lr=0.005, eps=1e-3):
    """Train ONE physical RBF layer with finite differences.

    Each parameter is perturbed ±eps, loss is evaluated via full residual
    forward (physical_forward_full). Only layer_idx's parameters are updated.
    """
    target = phys_layers[layer_idx]
    params = target.get_params_vector()
    n_params = len(params)
    y_lbl = y.astype(np.int64).ravel()

    def loss_fn(p):
        target.set_params_from_vector(p)
        logits = physical_forward_full(phys_layers, X)
        # logits is already (batch, 2)
        logits = logits - logits.max(1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return float(-np.log(probs[np.arange(len(y_lbl)), y_lbl] + 1e-8).mean())

    best_p, best_l = params.copy(), loss_fn(params)

    for ep in range(n_epochs):
        grad = np.zeros(n_params)
        for j in range(n_params):
            params[j] = params[j] + eps; lp = loss_fn(params)
            params[j] = params[j] - 2 * eps; lm = loss_fn(params)
            params[j] = params[j] + eps  # restore
            grad[j] = (lp - lm) / (2 * eps)

        params = params - lr * grad
        cl = loss_fn(params)
        if cl < best_l: best_l = cl; best_p = params.copy()

    target.set_params_from_vector(best_p)
    return best_l


def physical_forward_full(phys_layers, X):
    """Full physical forward through 3 stacked layers + residual."""
    x = X[:, 0].astype(np.float32).ravel()
    y = X[:, 1].astype(np.float32).ravel() if X.ndim > 1 and X.shape[1] >= 2 else x.copy()

    # Layer 1
    out_g1, out_a1 = phys_layers[0].forward(x, y)
    f1 = out_g1 + out_a1
    g1 = out_g1.mean(-1) if out_g1.ndim > 1 and out_g1.shape[-1] > 1 else out_g1.ravel()
    a1 = out_a1.mean(-1) if out_a1.ndim > 1 and out_a1.shape[-1] > 1 else out_a1.ravel()

    # Layer 2: residual
    out_g2, out_a2 = phys_layers[1].forward(g1, a1)
    delta2 = out_g2 + out_a2
    # Pad/truncate to match f1 shape
    if f1.shape[1] > delta2.shape[1]:
        d2 = np.zeros_like(f1); d2[:, :delta2.shape[1]] = delta2; delta2 = d2
    elif delta2.shape[1] > f1.shape[1]:
        f1p = np.zeros_like(delta2); f1p[:, :f1.shape[1]] = f1; f1 = f1p
    f2 = f1 + delta2
    g2 = out_g2.mean(-1) if out_g2.ndim > 1 and out_g2.shape[-1] > 1 else out_g2.ravel()
    a2 = out_a2.mean(-1) if out_a2.ndim > 1 and out_a2.shape[-1] > 1 else out_a2.ravel()

    # Layer 3: residual
    out_g3, out_a3 = phys_layers[2].forward(g2, a2)
    delta3 = out_g3 + out_a3
    if f2.shape[1] > delta3.shape[1]:
        d3 = np.zeros_like(f2); d3[:, :delta3.shape[1]] = delta3; delta3 = d3
    elif delta3.shape[1] > f2.shape[1]:
        f2p = np.zeros_like(delta3); f2p[:, :f2.shape[1]] = f2; f2 = f2p
    f3 = f2 + delta3

    # Output: first 2 dims = class logits
    out_dim = min(f3.shape[1], 2)
    return f3[:, :out_dim]


def eval_physical(phys_layers, X, y, n=3):
    preds = []
    for _ in range(n):
        o = physical_forward_full(phys_layers, X)
        preds.append(o)
    avg = np.mean(preds, axis=0)
    p = avg.argmax(axis=1) if avg.ndim > 1 else (avg > 0.5).astype(int)
    return (p == y).mean()


# ──────────────────────────────────────────────
#  Inter-layer input computation
# ──────────────────────────────────────────────

def compute_layer_inputs(phys_layers, X):
    """Pre-compute the input coordinates for each layer."""
    inputs = [X.copy()]
    cur_x = X[:, 0].astype(np.float32).ravel()
    cur_y = X[:, 1].astype(np.float32).ravel() if X.ndim > 1 and X.shape[1] >= 2 else cur_x.copy()

    for i in range(2):  # Only need inputs for layers 2 and 3
        out_g, out_a = phys_layers[i].forward(cur_x, cur_y)
        cur_x = out_g.mean(-1) if out_g.ndim > 1 and out_g.shape[-1] > 1 else out_g.ravel()
        cur_y = out_a.mean(-1) if out_a.ndim > 1 and out_a.shape[-1] > 1 else out_a.ravel()
        inputs.append(np.column_stack([cur_x, cur_y]).astype(np.float32))

    return inputs


# ──────────────────────────────────────────────
#  Experiment Runner
# ──────────────────────────────────────────────

def run_one(name, X, y, particles=8, pretrain_epochs=200, fd_epochs=10):
    print(f"\n{'=' * 55}")
    print(f"  Physical FD-RBF: {name} ({particles}p, {pretrain_epochs}ep pretrain, {fd_epochs}ep FD)")
    print(f"{'=' * 55}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr_np = sc.fit_transform(X_tr).astype(np.float32)
    X_te_np = sc.transform(X_te).astype(np.float32)
    print(f"  train={len(X_tr)} test={len(X_te)}")

    # 1. Digital pretrain (end-to-end ResidualRBFNetwork)
    t0 = time.time()
    digital = digital_pretrain(X_tr_np, y_tr, particles, pretrain_epochs)
    with torch.no_grad():
        acc_dig = (digital(torch.tensor(X_te_np[:, 0]), torch.tensor(X_te_np[:, 1])).argmax(1)
                   == torch.tensor(y_te).long()).float().mean().item()
    print(f"  [1] Digital pretrain acc: {acc_dig:.4f} (time={time.time()-t0:.1f}s)")

    # 2. Convert to physical layers
    phys_layers = [
        PhysicalRBFLayer.from_digital(digital.layer1),
        PhysicalRBFLayer.from_digital(digital.layer2),
        PhysicalRBFLayer.from_digital(digital.layer3),
    ]

    # Check physical accuracy before FD
    NoiseConfig.set_level("none")
    acc_phys_pre = eval_physical(phys_layers, X_te_np, y_te)
    print(f"      Physical pre-FD acc: {acc_phys_pre:.4f}")

    # 3. Physical FD fine-tuning
    t0 = time.time()
    print(f"  [2] Physical FD fine-tuning (per-layer, {fd_epochs}ep each):")
    for layer_idx in range(3):
        layer_inputs = compute_layer_inputs(phys_layers, X_tr_np)
        print(f"      Layer {layer_idx+1}...", end="", flush=True)
        physical_fd_train_one_layer(phys_layers, layer_idx,
                                    layer_inputs[layer_idx], y_tr,
                                    n_epochs=fd_epochs, lr=0.02, eps=2e-3)
        print(" done", flush=True)
    print(f"      FD train time: {time.time()-t0:.1f}s")

    acc_phys_post = eval_physical(phys_layers, X_te_np, y_te)
    delta = acc_phys_post - acc_phys_pre
    print(f"      Physical post-FD acc: {acc_phys_post:.4f} (delta={delta:+.4f})")

    # 4. Noise evaluation
    print(f"\n  [3] Physical noise evaluation:")
    for level in ["none", "low", "medium", "high"]:
        NoiseConfig.set_level(level)
        a = eval_physical(phys_layers, X_te_np, y_te)
        print(f"      {level:<10s} {a:.4f}")
    NoiseConfig.set_level("none")

    # 5. Decision boundary plot
    tag = name.lower().replace(" ", "_")
    margin = 1.0; xm, xM = X_te_np[:, 0].min() - margin, X_te_np[:, 0].max() + margin
    ym, yM = X_te_np[:, 1].min() - margin, X_te_np[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(xm, xM, 80), np.linspace(ym, yM, 80))
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    Z = physical_forward_full(phys_layers, grid).argmax(axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]), alpha=0.35, levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_te_np[:, 0], X_te_np[:, 1], c=y_te, cmap=ListedColormap(["#CC0000", "#0000CC"]), s=5, alpha=0.6)
    for i, layer in enumerate(phys_layers):
        ax.scatter(layer.xc, layer.yc, c=["black", "orange", "green"][i],
                   marker=["X", "s", "D"][i], s=[160, 110, 80][i],
                   edgecolors="white", linewidths=0.8, label=f"L{i+1}", zorder=6)
    ax.set_xlim(xm, xM); ax.set_ylim(ym, yM); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Physical FD-RBF — {name} (pre={acc_phys_pre:.3f} → post={acc_phys_post:.3f})")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"phys_fd_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"name": name, "pre": acc_phys_pre, "post": acc_phys_post, "delta": delta}


def main():
    print("Physical RBF with Finite-Difference Per-Layer Training")
    print("  FD gradient on physical simulator → no autograd on RBF layers\n")
    results = []

    X, y = make_moons(n_samples=1000, noise=0.12, random_state=42)
    results.append(run_one("Moons", X, y, particles=3, pretrain_epochs=25, fd_epochs=25))

    X, y = make_circles(n_samples=1000, noise=0.08, factor=0.4, random_state=42)
    results.append(run_one("Circles", X, y, particles=10, pretrain_epochs=80, fd_epochs=20))

    print("\n" + "=" * 55)
    print(f"  {'Dataset':<12s} {'Pre-FD':<10s} {'Post-FD':<10s} {'Delta':<10s}")
    print("  " + "-" * 38)
    for r in results:
        print(f"  {r['name']:<12s} {r['pre']:<10.4f} {r['post']:<10.4f} {r['delta']:+.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
