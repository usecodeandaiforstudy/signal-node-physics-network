"""
4D Random Projection RBF Ensemble Network
D=4 → K random 4→2 projections → K 2D RBFs → average fusion
Comparison: RBF-2D (first two dims only) / MLP-4D (full dims)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rbf_network import ResidualRBFNetwork

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def train_model(model, opt, data_fn, X, epochs=100):
    for _ in range(epochs):
        model.train()
        for i in range(0, len(X), 256):
            opt.zero_grad()
            bx, by = data_fn(i, X)
            nn.CrossEntropyLoss()(model(*bx) if isinstance(bx, tuple) else model(bx), by).backward()
            opt.step()


def eval_acc(model, X_te, y_te, is_tuple=False):
    model.eval()
    with torch.no_grad():
        if is_tuple:
            pred = model(torch.tensor(X_te[:, 0]), torch.tensor(X_te[:, 1])).argmax(1)
        else:
            pred = model(torch.tensor(X_te)).argmax(1)
        return (pred == torch.tensor(y_te)).float().mean().item()


def main():
    log = open(os.path.join(OUT, "07_ensemble_4d.txt"), "w", buffering=1)
    def p(msg):
        print(msg); log.write(msg+"\n")

    D, N = 4, 3000
    X, y = make_classification(N, D, n_informative=4, n_redundant=0, n_clusters_per_class=1,
                               flip_y=0.05, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr).astype(np.float32); X_te = sc.transform(X_te).astype(np.float32)
    Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr).long()
    p(f"4D data: train={len(X_tr)} test={len(X_te)} classes=2")

    results = []

    # Baseline: RBF on first 2D only
    p("\n--- Baseline: RBF (2D only, first 2 features) ---")
    m1 = ResidualRBFNetwork(1, 2, 15, 8, 6,
        (X_tr[:,0].min()-0.5, X_tr[:,0].max()+0.5),
        (X_tr[:,1].min()-0.5, X_tr[:,1].max()+0.5), False)
    def d1(i, X):
        end = min(i+256, len(X))
        return (torch.tensor(X[i:end, 0]), torch.tensor(X[i:end, 1])), yt[i:end]
    t0 = time.time(); train_model(m1, optim.Adam(m1.parameters(), lr=0.003), d1, X_tr)
    acc = eval_acc(m1, X_te, y_te, is_tuple=True); np_ = sum(p.numel() for p in m1.parameters()); t = time.time()-t0
    p(f"  RBF-2D:  acc={acc:.4f}  params={np_}  time={t:.1f}s")
    results.append(("RBF (2D only)", acc, np_, t))

    # Projected Ensemble
    for K, parts in [(3, 8), (5, 8), (5, 12)]:
        p(f"\n--- Ensemble: K={K} proj, {parts} particles/subnet ---")
        class Ensemble(nn.Module):
            def __init__(self):
                super().__init__()
                P_list = []
                for _ in range(K):
                    Pk = torch.randn(D, 2) * 0.1
                    Pk = Pk / Pk.norm(dim=0, keepdim=True)
                    P_list.append(Pk)
                self.P = P_list
                self.nets = nn.ModuleList([ResidualRBFNetwork(1, 2, parts, max(parts//2,4), max(parts//3,3),
                    (-5,5),(-5,5), False) for _ in range(K)])
            def forward(self, x):
                outs = [self.nets[k](x @ self.P[k][:,0], x @ self.P[k][:,1]) for k in range(K)]
                return torch.stack(outs, 0).mean(0)

        ens = Ensemble()
        def d_ens(i, X):
            end = min(i+256, len(X))
            return (X[i:end],), yt[i:end]
        t0 = time.time(); train_model(ens, optim.Adam(ens.parameters(), lr=0.003), d_ens, Xt)
        acc = eval_acc(ens, X_te, y_te); np_ = sum(p.numel() for p in ens.parameters()); t = time.time()-t0
        p(f"  Ensemble(K={K},{parts}p): acc={acc:.4f}  params={np_}  time={t:.1f}s")
        results.append((f"ProjEnsemble (K={K}, {parts}p)", acc, np_, t))

    # MLP on full 4D
    p("\n--- MLP on full 4D ---")
    mlp = nn.Sequential(nn.Linear(D,128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
    def d_mlp(i, X):
        end = min(i+256, len(X))
        return (X[i:end],), yt[i:end]
    t0 = time.time(); train_model(mlp, optim.Adam(mlp.parameters(), lr=0.001), d_mlp, Xt, epochs=100)
    acc = eval_acc(mlp, X_te, y_te); np_ = sum(p.numel() for p in mlp.parameters()); t = time.time()-t0
    p(f"  MLP-4D: acc={acc:.4f}  params={np_}  time={t:.1f}s")
    results.append(("MLP (4D)", acc, np_, t))

    # Summary
    p("\n" + "=" * 60)
    p(f"  {'Model':<30s} {'Acc':<8s} {'Params':<8s} {'Time':<8s}")
    p("  " + "-" * 55)
    for name, acc, np_, t in results:
        p(f"  {name:<30s} {acc:<8.4f} {np_:<8d} {t:<8.1f}s")
    p("=" * 60)
    log.close()
    print(f"\nResults saved to: {os.path.join(OUT, '07_ensemble_4d.txt')}")
    return results


if __name__ == "__main__":
    main()
