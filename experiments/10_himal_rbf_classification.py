"""
Experiment 10: HiMAL-Trained RBF on Physical Simulator

Follows the same HiMAL dual-network layer-wise training as experiment 09,
then syncs the trained model to PhysicalResidualRBFNetwork and evaluates
at 4 noise levels.

Key properties:
  - HiMAL training: dual networks (A/B), per-layer autograd, no cross-layer BP
  - Each layer has a Linear classifier head (n_vars -> n_classes)
  - After digital training: sync to physical simulator via from_digital()
  - Evaluate on physical components with noise injection
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rbf_network.core import RBFLayer
from physical_simulator.layer import PhysicalRBFLayer
from physical_simulator import NoiseConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
#  HiMAL-RBF Layer (same as experiment 09)
# ──────────────────────────────────────────────

class HiMALRBFLayer(nn.Module):
    """RBF layer with Linear classifier head for HiMAL layer-wise training."""

    def __init__(self, n_particles, n_vars, n_classes, x_range, y_range, init_zero=False):
        super().__init__()
        self.n_vars = n_vars
        init_pts = (
            np.random.uniform(x_range[0], x_range[1], n_particles).astype(np.float32),
            np.random.uniform(y_range[0], y_range[1], n_particles).astype(np.float32),
        )
        self.rbf = RBFLayer(points=init_pts, n_vars=n_vars, use_pu=True, init_zero=init_zero)
        self.head = nn.Linear(n_vars, n_classes)

    def forward(self, x, y):
        out_g, out_a = self.rbf(x, y)
        features = out_g + out_a
        return features, out_g, out_a

    def classify(self, features):
        return self.head(features)


# ──────────────────────────────────────────────
#  Data Partition
# ──────────────────────────────────────────────

def partition_data(X_tensor, y_tensor, method="median_x", transition_width=0.2):
    with torch.no_grad():
        if method == "radius":
            split_feat = np.sqrt(X_tensor[:, 0].numpy()**2 + X_tensor[:, 1].numpy()**2)
        elif method == "median_y":
            split_feat = X_tensor[:, 1].numpy()
        else:
            split_feat = X_tensor[:, 0].numpy()

        median = np.median(split_feat)
        rng = np.ptp(split_feat)
        width = transition_width * max(rng, 1e-6)

        mask_A = split_feat <= median
        mask_B = split_feat > median
        trans = (split_feat >= median - width) & (split_feat <= median + width)

    return {
        "XA": X_tensor[mask_A], "yA": y_tensor[mask_A],
        "XB": X_tensor[mask_B], "yB": y_tensor[mask_B],
        "XT": X_tensor[trans],
        "split_value": median,
    }


# ──────────────────────────────────────────────
#  HiMAL Training (same as experiment 09)
# ──────────────────────────────────────────────

def train_layer_pair(layer_A, layer_B, data, epochs=200, lr=0.005, gamma=0.3):
    opt = optim.Adam(list(layer_A.parameters()) + list(layer_B.parameters()), lr=lr)
    xA, yA_coord = data["XA"][:, 0], data["XA"][:, 1]
    xB, yB_coord = data["XB"][:, 0], data["XB"][:, 1]
    labels_A = data["yA"].long(); labels_B = data["yB"].long()
    has_transition = len(data["XT"]) > 0
    if has_transition: xT, yT_coord = data["XT"][:, 0], data["XT"][:, 1]

    for ep in range(epochs):
        opt.zero_grad()
        featA, _, _ = layer_A(xA, yA_coord); lossA = F.cross_entropy(layer_A.classify(featA), labels_A)
        featB, _, _ = layer_B(xB, yB_coord); lossB = F.cross_entropy(layer_B.classify(featB), labels_B)

        if has_transition:
            featTA, _, _ = layer_A(xT, yT_coord); featTB, _, _ = layer_B(xT, yT_coord)
            probTA = F.softmax(layer_A.classify(featTA), dim=1)
            probTB = F.softmax(layer_B.classify(featTB), dim=1)
            loss_align = 0.5 * (F.kl_div(probTA.log(), probTB, reduction='batchmean') +
                                F.kl_div(probTB.log(), probTA, reduction='batchmean'))
        else:
            loss_align = torch.tensor(0.0)

        loss = lossA + lossB + gamma * loss_align
        if torch.isnan(loss) or torch.isinf(loss): continue
        loss.backward(); opt.step()
    return layer_A, layer_B


def project_to_2d(out_g, out_a):
    if out_g.ndim > 1 and out_g.shape[-1] > 1:
        return out_g.mean(-1), out_a.mean(-1)
    return out_g.squeeze(-1), out_a.squeeze(-1)


def build_himal_dual_rbf(x_range, y_range, particles, n_classes=2, n_vars=6):
    p1, p2, p3 = particles, max(particles // 2, 4), max(particles // 3, 3)
    def make_one():
        return nn.ModuleList([
            HiMALRBFLayer(p1, n_vars, n_classes, x_range, y_range),
            HiMALRBFLayer(p2, n_vars, n_classes, x_range, y_range),
            HiMALRBFLayer(p3, n_vars, n_classes, x_range, y_range),
        ])
    return make_one(), make_one()


def himal_train(layers_A, layers_B, X_tr, y_tr, epochs_per_layer=200, lr=0.005, gamma=0.3, partition="median_x"):
    data = partition_data(X_tr, y_tr, method=partition)
    for i in range(len(layers_A)):
        xA_cur = data["XA"][:, 0]; xA_cur_y = data["XA"][:, 1] if data["XA"].ndim > 1 and data["XA"].shape[1] >= 2 else xA_cur.clone()
        xB_cur = data["XB"][:, 0]; xB_cur_y = data["XB"][:, 1] if data["XB"].ndim > 1 and data["XB"].shape[1] >= 2 else xB_cur.clone()
        xT_cur = torch.zeros(0); xT_cur_y = torch.zeros(0)
        if len(data["XT"]) > 0:
            xT_cur = data["XT"][:, 0]; xT_cur_y = data["XT"][:, 1] if data["XT"].shape[1] >= 2 else xT_cur.clone()

        layer_data = {
            "XA": torch.stack([xA_cur, xA_cur_y], dim=1), "yA": data["yA"],
            "XB": torch.stack([xB_cur, xB_cur_y], dim=1), "yB": data["yB"],
            "XT": torch.stack([xT_cur, xT_cur_y], dim=1) if len(data["XT"]) > 0 else torch.zeros(0, 2),
        }
        train_layer_pair(layers_A[i], layers_B[i], layer_data, epochs=epochs_per_layer, lr=lr, gamma=gamma)

        with torch.no_grad():
            _, out_gA, out_aA = layers_A[i](xA_cur, xA_cur_y)
            _, out_gB, out_aB = layers_B[i](xB_cur, xB_cur_y)
            gA, aA = project_to_2d(out_gA, out_aA); gB, aB = project_to_2d(out_gB, out_aB)
            if len(data["XT"]) > 0:
                _, out_gTA, out_aTA = layers_A[i](xT_cur, xT_cur_y)
                _, out_gTB, out_aTB = layers_B[i](xT_cur, xT_cur_y)
                gTA, aTA = project_to_2d(out_gTA, out_aTA); gTB, aTB = project_to_2d(out_gTB, out_aTB)
                gT = (gTA + gTB) / 2.0; aT = (aTA + aTB) / 2.0
            data["XA"] = torch.stack([gA, aA], dim=1); data["XB"] = torch.stack([gB, aB], dim=1)
            if len(data["XT"]) > 0: data["XT"] = torch.stack([gT, aT], dim=1)


def himal_predict(layers_A, layers_B, X):
    """Predict using dual networks with HiMAL architecture (digital)."""
    with torch.no_grad():
        cur_A, cur_B = X.clone(), X.clone()
        for i in range(len(layers_A)):
            xA = cur_A[:, 0]; yA = cur_A[:, 1] if cur_A.shape[1] >= 2 else xA
            xB = cur_B[:, 0]; yB = cur_B[:, 1] if cur_B.shape[1] >= 2 else xB
            featA, out_gA, out_aA = layers_A[i](xA, yA)
            featB, out_gB, out_aB = layers_B[i](xB, yB)
            if i < len(layers_A) - 1:
                gA, aA = project_to_2d(out_gA, out_aA); gB, aB = project_to_2d(out_gB, out_aB)
                cur_A = torch.stack([gA, aA], dim=1); cur_B = torch.stack([gB, aB], dim=1)
        logitsA = layers_A[-1].classify(featA); logitsB = layers_B[-1].classify(featB)
    return (logitsA + logitsB) / 2.0


# ──────────────────────────────────────────────
#  Convert HiMAL layers to ResidualRBFNetwork
# ──────────────────────────────────────────────

def himal_to_physical_layers(layers):
    """Convert HiMAL RBFLayers to PhysicalRBFLayers (3 layers per network)."""
    phys_layers = []
    for layer in layers:
        phys_layers.append(PhysicalRBFLayer.from_digital(layer.rbf))
    return phys_layers


def himal_physical_forward(phys_layers_A, phys_layers_B, X):
    """Forward through physical HiMAL layers (stacked, no residual)."""
    x = X[:, 0].astype(np.float32).ravel()
    y = X[:, 1].astype(np.float32).ravel() if X.ndim > 1 and X.shape[1] >= 2 else x.copy()

    cur_A_x, cur_A_y, cur_B_x, cur_B_y = x.copy(), y.copy(), x.copy(), y.copy()

    for i in range(len(phys_layers_A)):
        out_gA, out_aA = phys_layers_A[i].forward(cur_A_x, cur_A_y)
        out_gB, out_aB = phys_layers_B[i].forward(cur_B_x, cur_B_y)

        if i < len(phys_layers_A) - 1:
            cur_A_x = out_gA.mean(-1) if out_gA.ndim > 1 and out_gA.shape[-1] > 1 else out_gA.ravel()
            cur_A_y = out_aA.mean(-1) if out_aA.ndim > 1 and out_aA.shape[-1] > 1 else out_aA.ravel()
            cur_B_x = out_gB.mean(-1) if out_gB.ndim > 1 and out_gB.shape[-1] > 1 else out_gB.ravel()
            cur_B_y = out_aB.mean(-1) if out_aB.ndim > 1 and out_aB.shape[-1] > 1 else out_aB.ravel()

    # Fused output: (batch, n_vars) from each network
    fA = out_gA + out_aA
    fB = out_gB + out_aB
    return fA, fB


def eval_physical(phys_model, X, y, n=3):
    preds = []
    for _ in range(n):
        o = phys_model.forward(X[:, 0].astype(np.float32), X[:, 1].astype(np.float32))
        if o.ndim > 1 and o.shape[0] != len(y): o = o.T
        preds.append(o)
    avg = np.mean(preds, axis=0)
    p = avg.argmax(axis=1) if avg.ndim > 1 else (avg > 0.5).astype(int)
    return (p == y).mean()


# ──────────────────────────────────────────────
#  Experiment Runner
# ──────────────────────────────────────────────

def run_one(name, X, y, particles=8, epochs_per_layer=200, partition="median_x"):
    print(f"\n{'=' * 55}")
    print(f"  HiMAL->Physical: {name} ({particles}p, {epochs_per_layer}ep/layer, {partition})")
    print(f"{'=' * 55}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr_np = sc.fit_transform(X_tr).astype(np.float32); X_te_np = sc.transform(X_te).astype(np.float32)
    print(f"  train={len(X_tr)} test={len(X_te)}")

    X_tr_t = torch.tensor(X_tr_np); y_tr_t = torch.tensor(y_tr).long()
    X_te_t = torch.tensor(X_te_np); y_te_t = torch.tensor(y_te).long()

    x_range = (X_tr_np[:, 0].min() - 0.5, X_tr_np[:, 0].max() + 0.5)
    y_range = (X_tr_np[:, 1].min() - 0.5, X_tr_np[:, 1].max() + 0.5)

    # 1. HiMAL digital training
    t0 = time.time()
    layers_A, layers_B = build_himal_dual_rbf(x_range, y_range, particles)
    himal_train(layers_A, layers_B, X_tr_t, y_tr_t, epochs_per_layer=epochs_per_layer, partition=partition)

    # 2. Digital evaluation
    logits = himal_predict(layers_A, layers_B, X_te_t)
    acc_dig = (logits.argmax(1) == y_te_t).float().mean().item()
    print(f"  [1] HiMAL digital acc: {acc_dig:.4f} (time={time.time()-t0:.1f}s)")

    # 3. Convert to physical layers
    phys_A = himal_to_physical_layers(layers_A)
    phys_B = himal_to_physical_layers(layers_B)

    # Get classifier head weights (use last layer's head as numpy)
    head_A_W = layers_A[-1].head.weight.detach().cpu().numpy()
    head_A_b = layers_A[-1].head.bias.detach().cpu().numpy()
    head_B_W = layers_B[-1].head.weight.detach().cpu().numpy()
    head_B_b = layers_B[-1].head.bias.detach().cpu().numpy()

    # 4. Noise evaluation
    print(f"\n  [2] Physical noise evaluation:")
    for level in ["none", "low", "medium", "high"]:
        NoiseConfig.set_level(level)
        preds_A = []; preds_B = []
        for _ in range(3):
            fA, fB = himal_physical_forward(phys_A, phys_B, X_te_np)
            logits_A = fA @ head_A_W.T + head_A_b
            logits_B = fB @ head_B_W.T + head_B_b
            preds_A.append(logits_A); preds_B.append(logits_B)
        avg_A, avg_B = np.mean(preds_A, axis=0), np.mean(preds_B, axis=0)
        fused = (avg_A + avg_B) / 2.0
        p = fused.argmax(axis=1)
        acc = (p == y_te).mean()
        print(f"      {level:<10s} {acc:.4f}")
    NoiseConfig.set_level("none")

    # 5. Decision boundary (using physical model A + B averaged)
    tag = name.lower().replace(" ", "_")
    margin = 1.0; xm, xM = X_te_np[:, 0].min() - margin, X_te_np[:, 0].max() + margin
    ym, yM = X_te_np[:, 1].min() - margin, X_te_np[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(xm, xM, 100), np.linspace(ym, yM, 100))
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    fA_g, fB_g = himal_physical_forward(phys_A, phys_B, grid_pts)
    logits_A_g = fA_g @ head_A_W.T + head_A_b
    logits_B_g = fB_g @ head_B_W.T + head_B_b
    Z = ((logits_A_g + logits_B_g) / 2.0).argmax(axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]), alpha=0.35, levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_te_np[:, 0], X_te_np[:, 1], c=y_te, cmap=ListedColormap(["#CC0000", "#0000CC"]), s=5, alpha=0.6)
    for i, layer in enumerate([phys_A[0], phys_A[1], phys_A[2]]):
        ax.scatter(layer.xc, layer.yc, c=["black", "orange", "green"][i],
                   marker=["X", "s", "D"][i], s=[160, 110, 80][i],
                   edgecolors="white", linewidths=0.8, label=f"L{i+1}", zorder=6)
    ax.set_xlim(xm, xM); ax.set_ylim(ym, yM); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"HiMAL Physical RBF — {name} (digital acc={acc_dig:.3f})")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"10_himal_phys_{tag}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    return {"name": name, "acc_digital": acc_dig}


def main():
    print("Experiment 10: HiMAL-Trained RBF on Physical Simulator")
    print("  Training: HiMAL layer-wise (digital), Evaluation: physical at 4 noise levels\n")
    results = []

    X, y = make_moons(n_samples=1500, noise=0.12, random_state=42)
    results.append(run_one("Moons", X, y, particles=8, epochs_per_layer=200, partition="median_x"))

    X, y = make_circles(n_samples=1500, noise=0.08, factor=0.4, random_state=42)
    results.append(run_one("Circles", X, y, particles=20, epochs_per_layer=500, partition="radius"))

    print("\n" + "=" * 55)
    for r in results:
        print(f"  {r['name']:<12s} digital acc={r['acc_digital']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
