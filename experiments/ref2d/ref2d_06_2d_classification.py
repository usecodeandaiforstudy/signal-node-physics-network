"""
2D Classification Experiment: circles + moons (naturally 2D, no PCA needed)
RBF particle network vs MLP comparison + decision boundary visualization
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from rbf_network import ResidualRBFNetwork

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_decision(model, X, y, centers_list, title, path):
    """Plot decision boundary + particle centers"""
    margin = 1.0
    xm, xM = X[:,0].min()-margin, X[:,0].max()+margin
    ym, yM = X[:,1].min()-margin, X[:,1].max()+margin
    xx, yy = np.meshgrid(np.linspace(xm, xM, 150), np.linspace(ym, yM, 150))
    model.eval()
    with torch.no_grad():
        g = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        Z = model(g[:,0], g[:,1]).argmax(1).numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["#FFAAAA","#AAAAFF"]), alpha=0.35, levels=[-0.5,0.5,1.5])
    ax.scatter(X[:,0], X[:,1], c=y, cmap=ListedColormap(["#CC0000","#0000CC"]), s=5, alpha=0.6)

    colors=["black","orange","green"]; markers=["X","s","D"]; sizes=[140,90,65]
    for i,(cx,cy) in enumerate(centers_list):
        ax.scatter(cx, cy, c=colors[i], marker=markers[i], s=sizes[i],
                  edgecolors="white", linewidths=0.8, label=f"L{i+1}", zorder=6)
    ax.set_xlim(xm,xM); ax.set_ylim(ym,yM)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {os.path.basename(path)}")


def run_experiment(name, X, y, particles=25, epochs=200, lr=0.003):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr).astype(np.float32); X_te = sc.transform(X_te).astype(np.float32)
    print(f"  train: {len(X_tr)}, test: {len(X_te)}, 2 classes")

    # ---- RBF ----
    t0 = time.time()
    ds = TensorDataset(torch.tensor(X_tr[:,0]), torch.tensor(X_tr[:,1]), torch.tensor(y_tr).long())
    model = ResidualRBFNetwork(
        hidden_dim=1, output_dim=2,
        first_layer_particles=particles,
        second_layer_particles=max(particles//2,8),
        third_layer_particles=max(particles//3,6),
        x_range=(X_tr[:,0].min()-0.5, X_tr[:,0].max()+0.5),
        y_range=(X_tr[:,1].min()-0.5, X_tr[:,1].max()+0.5),
        init_zero=False,
    )
    dl = DataLoader(ds, 256, True)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for e in range(epochs):
        model.train()
        for bx, by, bc in dl:
            opt.zero_grad(); nn.CrossEntropyLoss()(model(bx, by), bc).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()

    model.eval()
    with torch.no_grad():
        acc_rbf = (model(torch.tensor(X_te[:,0]), torch.tensor(X_te[:,1])).argmax(1) == torch.tensor(y_te)).float().mean().item()
    n_rbf = sum(p.numel() for p in model.parameters()); t_rbf = time.time()-t0
    print(f"  RBF({particles}p): acc={acc_rbf:.4f}  params={n_rbf}  time={t_rbf:.1f}s")

    centers = []
    for layer in [model.layer1, model.layer2, model.layer3]:
        xc, yc = layer.centers()
        centers.append((xc.detach().numpy(), yc.detach().numpy()))

    # ---- MLP ----
    t0 = time.time()
    mlp = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,2))
    ds2 = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).long()); dl2 = DataLoader(ds2, 256, True)
    opt = optim.Adam(mlp.parameters(), lr=0.001)
    for e in range(epochs):
        for bx, bc in dl2:
            opt.zero_grad(); nn.CrossEntropyLoss()(mlp(bx), bc).backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        acc_mlp = (mlp(torch.tensor(X_te)).argmax(1) == torch.tensor(y_te)).float().mean().item()
    n_mlp = sum(p.numel() for p in mlp.parameters()); t_mlp = time.time()-t0
    print(f"  MLP:       acc={acc_mlp:.4f}  params={n_mlp}  time={t_mlp:.1f}s")

    # decision boundary
    plot_decision(model, X_te, y_te, centers,
                  f"{name} (RBF {particles}p, acc={acc_rbf:.3f}, {n_rbf} params)",
                  os.path.join(OUT_DIR, f"06_{name.replace(' ','_').lower()}.png"))

    return {"name": name, "acc_rbf": acc_rbf, "acc_mlp": acc_mlp, "params_rbf": n_rbf, "params_mlp": n_mlp}


def main():
    print("2D Classification: RBF Particle Network vs MLP")
    results = []

    # -- Circles: concentric rings, hard for linear methods --
    X, y = make_circles(n_samples=3000, noise=0.08, factor=0.4, random_state=42)
    results.append(run_experiment("Circles (factor=0.4)", X, y, particles=25, epochs=200))

    # -- Moons --
    X, y = make_moons(n_samples=3000, noise=0.12, random_state=42)
    results.append(run_experiment("Moons (noise=0.12)", X, y, particles=20, epochs=200))

    # -- Summary bar --
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(results))
    w = 0.35
    b1 = ax.bar(x_pos - w/2, [r["acc_rbf"] for r in results], w, label=f'RBF', color="#2196F3", alpha=0.85)
    b2 = ax.bar(x_pos + w/2, [r["acc_mlp"] for r in results], w, label='MLP (2-layer)', color="#FF9800", alpha=0.85)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_xticks(x_pos); ax.set_xticklabels([r["name"] for r in results])
    ax.set_title("RBF Particle Network vs MLP — 2D Classification")
    ax.legend()
    for bars, key in [(b1, "acc_rbf"), (b2, "acc_mlp")]:
        for bar, r in zip(bars, results):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                    f'{r[key]:.3f}', ha='center', fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "06_2d_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  saved: {os.path.basename(path)}")

    print("\n" + "=" * 50)
    for r in results:
        print(f"  {r['name']:<28s} RBF={r['acc_rbf']:.3f} ({r['params_rbf']}p)  MLP={r['acc_mlp']:.3f} ({r['params_mlp']}p)")
    print("=" * 50)

if __name__ == "__main__":
    main()
