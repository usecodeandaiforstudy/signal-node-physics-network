"""
Experiment 9: HiMAL Dual-Network Layer-Wise RBF Classification (Digital)

Follows the HiMAL approach from reference: dual networks (A/B), prototype-based
layer-wise training, alignment loss on transition zone.

Key properties:
  - Dual ResidualRBFNetworks (A and B), each handles a different spatial region
  - Layer-wise training: each layer has learnable prototypes, trained independently
  - Each layer uses autograd within itself, NO cross-layer backpropagation
  - Alignment via KL divergence on transition zone samples
  - Final classifier head trained on combined features (frozen after layer training)

This is the true HiMAL approach — per-layer autograd, no SPSA, no finite differences.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
#  HiMAL-RBF Layer (with learnable prototypes)
# ──────────────────────────────────────────────

class HiMALRBFLayer(nn.Module):
    """RBF layer with per-class prototypes or Linear head for HiMAL layer-wise training.

    Forward returns:
      - features: fused dual-channel output (batch, n_vars)
      - out_g, out_a: raw gating/activation channels for inter-layer projection
    """

    def __init__(self, n_particles, n_vars, n_classes, x_range, y_range, init_zero=False, use_head=False):
        super().__init__()
        self.n_vars = n_vars
        self.use_head = use_head
        init_pts = (
            np.random.uniform(x_range[0], x_range[1], n_particles).astype(np.float32),
            np.random.uniform(y_range[0], y_range[1], n_particles).astype(np.float32),
        )
        self.rbf = RBFLayer(points=init_pts, n_vars=n_vars, use_pu=True, init_zero=init_zero)
        if use_head:
            self.head = nn.Linear(n_vars, n_classes)
        else:
            self.prototypes = nn.Parameter(torch.randn(n_classes, n_vars) * 0.5)

    def forward(self, x, y):
        out_g, out_a = self.rbf(x, y)
        features = out_g + out_a
        return features, out_g, out_a

    def classify(self, features, temp=0.1):
        """Convert features to class logits via prototypes or Linear head"""
        if self.use_head:
            return self.head(features)
        dist = torch.cdist(features, self.prototypes)
        return -dist / temp


# ──────────────────────────────────────────────
#  Data Partition
# ──────────────────────────────────────────────

def partition_data(X_tensor, y_tensor, method="median_x", transition_width=0.2):
    """Split data into A-zone, B-zone, and transition zone."""
    with torch.no_grad():
        if method == "median_x":
            split_feat = X_tensor[:, 0].numpy()
        elif method == "median_y":
            split_feat = X_tensor[:, 1].numpy()
        elif method == "radius":
            # Partition by distance from origin — useful for concentric patterns
            split_feat = np.sqrt(X_tensor[:, 0].numpy()**2 + X_tensor[:, 1].numpy()**2)
        else:
            split_feat = X_tensor[:, 0].numpy()

        median = np.median(split_feat)
        rng = np.ptp(split_feat)
        width = transition_width * max(rng, 1e-6)

        mask_A = split_feat <= median
        mask_B = split_feat > median
        trans = (split_feat >= median - width) & (split_feat <= median + width)

    XA, yA = X_tensor[mask_A], y_tensor[mask_A]
    XB, yB = X_tensor[mask_B], y_tensor[mask_B]
    XT = X_tensor[trans]
    return {
        "XA": XA, "yA": yA, "XB": XB, "yB": yB, "XT": XT,
        "split_value": median,
    }


# ──────────────────────────────────────────────
#  Layer-Pair Trainer
# ──────────────────────────────────────────────

def train_layer_pair(layer_A, layer_B, data, epochs=200, lr=0.005, temp=0.1, gamma=0.3):
    """Train one layer of the dual-network (A and B) simultaneously.

    Loss = L_fit_A + L_fit_B + gamma * L_align
      L_fit  : CE on prototype-distance logits (per network, on its own region)
      L_align: symmetric KL divergence of soft prototype assignments on transition zone
    """
    opt = optim.Adam(list(layer_A.parameters()) + list(layer_B.parameters()), lr=lr)

    xA, yA_coord = data["XA"][:, 0], data["XA"][:, 1]
    xB, yB_coord = data["XB"][:, 0], data["XB"][:, 1]
    labels_A = data["yA"].long()
    labels_B = data["yB"].long()

    has_transition = len(data["XT"]) > 0
    if has_transition:
        xT, yT_coord = data["XT"][:, 0], data["XT"][:, 1]

    for ep in range(epochs):
        opt.zero_grad()

        # ---- Network A: fit on region A ----
        featA, _, _ = layer_A(xA, yA_coord)
        logitsA = layer_A.classify(featA, temp)
        lossA = F.cross_entropy(logitsA, labels_A)

        # ---- Network B: fit on region B ----
        featB, _, _ = layer_B(xB, yB_coord)
        logitsB = layer_B.classify(featB, temp)
        lossB = F.cross_entropy(logitsB, labels_B)

        # ---- Alignment on transition zone ----
        if has_transition:
            featTA, _, _ = layer_A(xT, yT_coord)
            featTB, _, _ = layer_B(xT, yT_coord)
            logitsTA = layer_A.classify(featTA, temp)
            logitsTB = layer_B.classify(featTB, temp)
            probTA = F.softmax(logitsTA, dim=1)
            probTB = F.softmax(logitsTB, dim=1)
            loss_align = 0.5 * (F.kl_div(probTA.log(), probTB, reduction='batchmean') +
                                F.kl_div(probTB.log(), probTA, reduction='batchmean'))
        else:
            loss_align = torch.tensor(0.0)

        loss = lossA + lossB + gamma * loss_align

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                accA = (logitsA.argmax(1) == labels_A).float().mean().item()
                accB = (logitsB.argmax(1) == labels_B).float().mean().item()
            if ep == 0 or ep >= epochs - 20 or ep % 100 == 0:
                pass  # Silent training, print only final

    return layer_A, layer_B


# ──────────────────────────────────────────────
#  Full HiMAL Training Pipeline
# ──────────────────────────────────────────────

def build_himal_dual_rbf(x_range, y_range, particles, n_classes=2, n_vars=6, use_head=False):
    """Build a pair of 3-layer HiMAL-RBF networks (A and B)."""
    p1 = particles
    p2 = max(particles // 2, 4)
    p3 = max(particles // 3, 3)

    def make_one():
        return nn.ModuleList([
            HiMALRBFLayer(p1, n_vars, n_classes, x_range, y_range, init_zero=False, use_head=use_head),
            HiMALRBFLayer(p2, n_vars, n_classes, x_range, y_range, init_zero=False, use_head=use_head),
            HiMALRBFLayer(p3, n_vars, n_classes, x_range, y_range, init_zero=False, use_head=use_head),
        ])

    return make_one(), make_one()


def project_to_2d(out_g, out_a):
    """Project dual-channel output to (x_coord, y_coord) for next layer input."""
    if out_g.ndim > 1 and out_g.shape[-1] > 1:
        g = out_g.mean(-1)
        a = out_a.mean(-1)
    else:
        g = out_g.squeeze(-1)
        a = out_a.squeeze(-1)
    return g, a


def himal_train(layers_A, layers_B, X_tr, y_tr, epochs_per_layer=200, lr=0.005, gamma=0.3, partition="median_x"):
    """Full HiMAL training: layer-by-layer, no cross-layer gradients."""
    data = partition_data(X_tr, y_tr, method=partition)
    n_layers = len(layers_A)

    for i in range(n_layers):
        # Build current layer's training data (2D coordinates)
        xA_cur = data["XA"][:, 0] if data["XA"].ndim > 1 else data["XA"].squeeze(-1)
        xA_cur_y = data["XA"][:, 1] if data["XA"].ndim > 1 and data["XA"].shape[1] >= 2 else xA_cur.clone()
        xB_cur = data["XB"][:, 0] if data["XB"].ndim > 1 else data["XB"].squeeze(-1)
        xB_cur_y = data["XB"][:, 1] if data["XB"].ndim > 1 and data["XB"].shape[1] >= 2 else xB_cur.clone()

        if len(data["XT"]) > 0:
            xT_cur = data["XT"][:, 0] if data["XT"].ndim > 1 else data["XT"].squeeze(-1)
            xT_cur_y = data["XT"][:, 1] if data["XT"].ndim > 1 and data["XT"].shape[1] >= 2 else xT_cur.clone()
        else:
            xT_cur = xT_cur_y = torch.zeros(0)

        layer_data = {
            "XA": torch.stack([xA_cur, xA_cur_y], dim=1),
            "yA": data["yA"],
            "XB": torch.stack([xB_cur, xB_cur_y], dim=1),
            "yB": data["yB"],
            "XT": torch.stack([xT_cur, xT_cur_y], dim=1) if len(data["XT"]) > 0 else torch.zeros(0, 2),
        }

        train_layer_pair(layers_A[i], layers_B[i], layer_data, epochs=epochs_per_layer, lr=lr, gamma=gamma)

        # Compute next layer's input (freeze current layer)
        with torch.no_grad():
            features_A, out_gA, out_aA = layers_A[i](xA_cur, xA_cur_y)
            features_B, out_gB, out_aB = layers_B[i](xB_cur, xB_cur_y)
            gA, aA = project_to_2d(out_gA, out_aA)
            gB, aB = project_to_2d(out_gB, out_aB)

            if len(data["XT"]) > 0:
                features_TA, out_gTA, out_aTA = layers_A[i](xT_cur, xT_cur_y)
                features_TB, out_gTB, out_aTB = layers_B[i](xT_cur, xT_cur_y)
                gTA, aTA = project_to_2d(out_gTA, out_aTA)
                gTB, aTB = project_to_2d(out_gTB, out_aTB)
                gT = (gTA + gTB) / 2.0
                aT = (aTA + aTB) / 2.0

            data["XA"] = torch.stack([gA, aA], dim=1)
            data["XB"] = torch.stack([gB, aB], dim=1)
            if len(data["XT"]) > 0:
                data["XT"] = torch.stack([gT, aT], dim=1)

    return data


def himal_extract_features(layers_A, layers_B, X):
    """Extract final-layer prototype logits from both networks."""
    with torch.no_grad():
        cur_A = X.clone()
        cur_B = X.clone()
        for i in range(len(layers_A)):
            xA = cur_A[:, 0]
            yA = cur_A[:, 1] if cur_A.shape[1] >= 2 else xA
            xB = cur_B[:, 0]
            yB = cur_B[:, 1] if cur_B.shape[1] >= 2 else xB

            featA, out_gA, out_aA = layers_A[i](xA, yA)
            featB, out_gB, out_aB = layers_B[i](xB, yB)

            if i < len(layers_A) - 1:
                gA, aA = project_to_2d(out_gA, out_aA)
                gB, aB = project_to_2d(out_gB, out_aB)
                cur_A = torch.stack([gA, aA], dim=1)
                cur_B = torch.stack([gB, aB], dim=1)

        logitsA = layers_A[-1].classify(featA)
        logitsB = layers_B[-1].classify(featB)
    return logitsA, logitsB


# ──────────────────────────────────────────────
#  Experiment Runner
# ──────────────────────────────────────────────

def run_one(name, X, y, particles=8, epochs_per_layer=200, partition="median_x", use_head=False):
    print(f"\n{'=' * 55}")
    hdr = "HiMAL-RBF-head" if use_head else "HiMAL-RBF"
    print(f"  {hdr}: {name} ({particles}p, {epochs_per_layer}ep/layer, partition={partition})")
    print(f"{'=' * 55}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr_np = sc.fit_transform(X_tr).astype(np.float32)
    X_te_np = sc.transform(X_te).astype(np.float32)
    print(f"  train={len(X_tr)} test={len(X_te)}")

    X_tr_t = torch.tensor(X_tr_np)
    y_tr_t = torch.tensor(y_tr).long()
    X_te_t = torch.tensor(X_te_np)
    y_te_t = torch.tensor(y_te).long()

    x_range = (X_tr_np[:, 0].min() - 0.5, X_tr_np[:, 0].max() + 0.5)
    y_range = (X_tr_np[:, 1].min() - 0.5, X_tr_np[:, 1].max() + 0.5)

    t0 = time.time()
    layers_A, layers_B = build_himal_dual_rbf(x_range, y_range, particles, use_head=use_head)
    himal_train(layers_A, layers_B, X_tr_t, y_tr_t, epochs_per_layer=epochs_per_layer, partition=partition)
    train_time = time.time() - t0

    # Evaluate
    logitsA, logitsB = himal_extract_features(layers_A, layers_B, X_te_t)
    avg_logits = (logitsA + logitsB) / 2.0
    acc = (avg_logits.argmax(1) == y_te_t).float().mean().item()
    print(f"  HiMAL accuracy: {acc:.4f} (train time: {train_time:.1f}s)")

    # Decision boundary plot
    tag = name.lower().replace(" ", "_")
    margin = 1.0
    xm, xM = X_te_np[:, 0].min() - margin, X_te_np[:, 0].max() + margin
    ym, yM = X_te_np[:, 1].min() - margin, X_te_np[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(xm, xM, 100), np.linspace(ym, yM, 100))
    grid = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32))
    logitsA_g, logitsB_g = himal_extract_features(layers_A, layers_B, grid)
    Z = ((logitsA_g + logitsB_g) / 2.0).argmax(1).numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]), alpha=0.35, levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_te_np[:, 0], X_te_np[:, 1], c=y_te, cmap=ListedColormap(["#CC0000", "#0000CC"]), s=5, alpha=0.6)
    ax.set_xlim(xm, xM); ax.set_ylim(ym, yM)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"HiMAL Dual RBF — {name} (acc={acc:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"09_himal_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"name": name, "acc": acc}


def main():
    print("Experiment 9: HiMAL Dual-Network Layer-Wise RBF Classification")
    print("  Training: per-layer autograd, no cross-layer backpropagation\n")
    results = []

    X, y = make_moons(n_samples=1500, noise=0.12, random_state=42)
    results.append(run_one("Moons", X, y, particles=8, epochs_per_layer=200, partition="median_x", use_head=True))

    X, y = make_circles(n_samples=1500, noise=0.08, factor=0.4, random_state=42)
    results.append(run_one("Circles", X, y, particles=15, epochs_per_layer=400, partition="radius", use_head=True))

    print("\n" + "=" * 55)
    for r in results:
        print(f"  {r['name']:<12s} acc={r['acc']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
