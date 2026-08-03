"""
Experiment 8: Physical Network 2D Classification
Train digital model → sync to physical → evaluate + visualize at 4 noise levels
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from rbf_network import ResidualRBFNetwork
from physical_simulator import PhysicalResidualRBFNetwork, NoiseConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARTICLES = 6  # optimal from ablation study
EPOCHS = 100


def train_digital_baseline(X_tr, y_tr, particles, epochs):
    """Train digital ResidualRBFNetwork"""
    m = ResidualRBFNetwork(
        hidden_dim=1, output_dim=2,
        first_layer_particles=particles,
        second_layer_particles=max(particles // 2, 3),
        third_layer_particles=max(particles // 3, 2),
        x_range=(X_tr[:, 0].min() - 0.5, X_tr[:, 0].max() + 0.5),
        y_range=(X_tr[:, 1].min() - 0.5, X_tr[:, 1].max() + 0.5),
        init_zero=False,
    )
    opt = optim.Adam(m.parameters(), lr=0.005)
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(
            m(torch.tensor(X_tr[:, 0]), torch.tensor(X_tr[:, 1])),
            torch.tensor(y_tr).long(),
        )
        loss.backward()
        opt.step()
    return m


def eval_physical(phys_model, X, y, n_samples=5):
    """Evaluate physical model with multiple noise samples"""
    m = phys_model
    preds = []
    for _ in range(n_samples):
        out = m.forward(X[:, 0].astype(np.float32), X[:, 1].astype(np.float32))
        if out.ndim > 1 and out.shape[0] != len(X):
            out = out.T
        preds.append(out.argmax(axis=1))
    # majority vote
    preds = np.array(preds)
    from scipy import stats
    final = stats.mode(preds, axis=0, keepdims=False).mode
    return (final == y).mean()


def plot_decision(phys_model, X, y, level, acc, save_path):
    """Plot decision boundary with particle centers"""
    margin = 1.0
    xm, xM = X[:, 0].min() - margin, X[:, 0].max() + margin
    ym, yM = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(xm, xM, 120), np.linspace(ym, yM, 120))
    grid_x = xx.ravel().astype(np.float32)
    grid_y = yy.ravel().astype(np.float32)
    out = phys_model.forward(grid_x, grid_y)
    if out.ndim > 1 and out.shape[0] != len(grid_x):
        out = out.T
    Z = out.argmax(axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]), alpha=0.35, levels=[-0.5, 0.5, 1.5])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#CC0000", "#0000CC"]), s=5, alpha=0.6)

    colors = ["black", "orange", "green"]
    markers = ["X", "s", "D"]
    sizes = [160, 110, 80]
    for i in range(3):
        layer = [phys_model.layer1, phys_model.layer2, phys_model.layer3][i]
        ax.scatter(layer.xc, layer.yc, c=colors[i], marker=markers[i], s=sizes[i],
                   edgecolors="white", linewidths=0.8, label=f"Layer {i+1}", zorder=6)

    ax.set_xlim(xm, xM); ax.set_ylim(ym, yM)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Physical network — {level} noise (acc={acc:.3f}, {PARTICLES} particles)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_one(name, X, y, particles=8, epochs=150):
    print(f"\n{'=' * 50}")
    print(f"  {name} ({particles}p, {epochs} epochs)")
    print(f"{'=' * 50}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    print(f"  train={len(X_tr)} test={len(X_te)}")

    # Train digital
    t0 = time.time()
    digital = train_digital_baseline(X_tr, y_tr, particles, epochs)
    n_params = sum(p.numel() for p in digital.parameters())
    print(f"  Digital trained: {n_params} params, {time.time() - t0:.1f}s")

    # Digital accuracy
    digital.eval()
    with torch.no_grad():
        acc_d = (digital(torch.tensor(X_te[:, 0]), torch.tensor(X_te[:, 1])).argmax(1)
                 == torch.tensor(y_te)).float().mean().item()
    print(f"  Digital accuracy: {acc_d:.4f}")

    # Sync to physical
    physical = PhysicalResidualRBFNetwork.from_digital(digital)
    print(f"  Synced to physical simulator")

    # Evaluate at 4 noise levels
    print(f"\n  {'Noise':<8s} {'Phys Acc':<10s} {'vs Digital':<10s} {'Status'}")
    print(f"  {'-' * 8:<8s} {'-' * 10:<10s} {'-' * 10:<10s} {'-' * 10}")

    for level in ["none", "low", "medium", "high"]:
        NoiseConfig.set_level(level)
        acc_p = eval_physical(physical, X_te, y_te)
        delta = acc_p - acc_d
        status = "OK" if acc_p > 0.7 else "LOW"
        print(f"  {level:<8s} {acc_p:<10.4f} {delta:+.4f}       {status}")

        # visualize
        name_tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plot_decision(physical, X_te, y_te, level, acc_p,
                      os.path.join(OUT_DIR, f"08_{name_tag}_{level}.png"))

    NoiseConfig.set_level("none")
    return {"name": name, "digital_acc": acc_d, "params": n_params}


def main():
    print(f"Experiment 8: Physical Network Classification | Device: {DEVICE}")
    results = []

    # Circles — harder: needs more particles/epochs
    X, y = make_circles(n_samples=2000, noise=0.08, factor=0.4, random_state=42)
    results.append(run_one("Circles (factor=0.4)", X, y, particles=12, epochs=250))

    # Moons — simpler geometry
    X, y = make_moons(n_samples=2000, noise=0.12, random_state=42)
    results.append(run_one("Moons (noise=0.12)", X, y, particles=8, epochs=150))

    # Summary
    print("\n" + "=" * 55)
    for r in results:
        print(f"  {r['name']:<25s}  digital={r['digital_acc']:.3f}  params={r['params']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
