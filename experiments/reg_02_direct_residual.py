"""
E05E-RBF-LAYERWISE: Pure Gaussian RBF + Layer-Wise Direct Residual
====================================================================
Tests whether layer-wise FD training with DIRECT residual targets
(unlike HiMAL's proto-weighted prediction) enables depth for RBF.

Key change vs original e05/e12 HiMAL:
  OLD:  residual = y - proto_predict(L1_output)  ← proto smoothing
  NEW:  residual = y - L1_output[:, 0]            ← exact per-point error

Architecture (3 layers, single network, no A/B split):
  L1: D→mid_dim, N1 particles  →  trained on full y
  L2: mid_dim→mid_dim, N2 p    →  trained on y - L1[:,0]
  L3: mid_dim→1, N3 p          →  trained on y - L1[:,0] - L2[:,0]

Inference: h1=L1(X); h2=L2(h1); h3=L3(h2); pred = h1[:,0]+h2[:,0]+h3[:,0]

Data: y = sin(3x1) + cos(2x2) + 0.5*x1*x2 + noise  (e05 function)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ═══════════════════════════════════════════════════════
#  Gaussian RBF Particle Layer
# ═══════════════════════════════════════════════════════

class RBFGaussianLayer:
    """Single-channel Gaussian RBF: d² → exp(-0.5·d²/h²) → MVM."""
    def __init__(self, centers, h, W):
        self.centers = np.asarray(centers, dtype=np.float32)
        if self.centers.ndim == 1: self.centers = self.centers.reshape(1, -1)
        self.h = np.asarray(h, dtype=np.float32)
        if self.h.ndim == 1: self.h = self.h.reshape(1, -1)
        self.W = np.asarray(W, dtype=np.float32)
        self.D = self.centers.shape[0]
        self.N = self.centers.shape[1]
        self.d_out = self.W.shape[0]

    def get_params_vector(self):
        return np.concatenate([
            self.centers.flatten(), self.h.flatten(), self.W.flatten()
        ]).astype(np.float32)

    def set_params_from_vector(self, vec):
        D, N, d_out = self.D, self.N, self.d_out
        p = 0; sz_c = D*N; sz_h = self.h.size
        self.centers = vec[p:p+sz_c].reshape(D, N).copy(); p += sz_c
        self.h = vec[p:p+sz_h].reshape(self.h.shape).copy(); p += sz_h
        self.W = vec[p:p+d_out*N].reshape(d_out, N).copy()

    def kernel_matrix(self, X):
        Xf = X.astype(np.float32)
        d2 = np.zeros((Xf.shape[0], self.N), dtype=np.float32)
        for i in range(self.D):
            dx = Xf[:, i:i+1] - self.centers[i:i+1, :]
            d2 += dx * dx
        h_sq = np.maximum(self.h.reshape(1, -1)**2, 1e-8)
        return np.exp(np.clip(-0.5 * d2 / h_sq, -50, 20))

    def forward(self, X):
        return self.kernel_matrix(X) @ self.W.T


def kmeans_init(X, n_particles, n_vars, random_state=42):
    X_np = np.asarray(X, dtype=np.float32)
    N_data, D = X_np.shape
    n_clusters = min(n_particles, N_data)
    if n_clusters >= 2:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        km.fit(X_np)
        centers = km.cluster_centers_.T.astype(np.float32)
        labels = km.labels_
        h = np.ones((1, n_clusters), dtype=np.float32)
        for k in range(n_clusters):
            mask = labels == k
            h[0, k] = max(np.linalg.norm(X_np[mask]-centers[:,k], axis=1).mean()*2.0, 0.02) if mask.sum()>1 else 0.1
        if n_particles > n_clusters:
            extra = n_particles - n_clusters
            idx = np.random.choice(n_clusters, extra, replace=True)
            c_extra = centers[:, idx] + np.random.normal(0, 0.05, (D, extra)).astype(np.float32)
            h_extra = h[:, idx] * 0.8
            centers = np.concatenate([centers, c_extra], axis=1)
            h = np.concatenate([h, h_extra], axis=1)
    else:
        lo, hi = float(X_np.min()), float(X_np.max())
        dsize = hi - lo if hi > lo else 2.0
        centers = np.random.uniform(lo-0.5, hi+0.5, (D, n_particles)).astype(np.float32)
        h = np.full((1, n_particles), dsize/max(np.sqrt(n_particles), 1.0), dtype=np.float32)
    h = np.maximum(h, 0.02)
    W = np.random.uniform(-0.3, 0.3, (n_vars, n_particles)).astype(np.float32)
    return RBFGaussianLayer(centers, h, W)


def fd_train_layer(layer, X, y, epochs, lr, eps, beta=0.85, verbose=True):
    vec = layer.get_params_vector().astype(np.float64)
    vel = np.zeros_like(vec)
    Xf, yf = X.astype(np.float32), y.astype(np.float32)
    if yf.ndim == 1: yf = yf.reshape(-1, 1)

    def loss(v):
        layer.set_params_from_vector(v.astype(np.float32))
        pred = layer.forward(Xf)
        m = min(pred.shape[1], yf.shape[1])
        return float(np.mean((pred[:, :m] - yf[:, :m])**2))

    for ep in range(epochs):
        grad = np.zeros_like(vec)
        for j in range(len(vec)):
            orig = vec[j]; vec[j] = orig + eps; Lp = loss(vec)
            vec[j] = orig - eps; Lm = loss(vec)
            vec[j] = orig; grad[j] = (Lp - Lm) / (2 * eps)
        vel[:] = beta * vel - lr * grad
        vec += vel
        if verbose and (ep+1) % 25 == 0:
            print(f"      ep {ep+1:3d}/{epochs} | loss={loss(vec):.6f}")

    layer.set_params_from_vector(vec.astype(np.float32))
    return layer


# ═══════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════

print("=" * 65)
print("  E05E-RBF: Gaussian RBF + Layer-Wise Direct Residual")
print("=" * 65)
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  y = sin(3x1) + cos(2x2) + 0.5*x1*x2 + noise")
print(f"  Method: K-Means init → FD layer-wise → direct residual targets")

np.random.seed(42)
N = 400
X = np.random.uniform(-2, 2, (N, 2)).astype(np.float32)
y = (np.sin(3*X[:,0]) + np.cos(2*X[:,1]) + 0.5*X[:,0]*X[:,1]
     + np.random.normal(0, 0.15, N)).astype(np.float32)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr).astype(np.float32)
X_te = sc.transform(X_te).astype(np.float32)
mean_baseline = float(np.mean((y_te - y_tr.mean())**2))
print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, Baseline MSE: {mean_baseline:.4f}")

def r2_score(pred, true):
    mse = float(np.mean((pred.ravel()-true.ravel())**2))
    return 1.0 - mse / max(mean_baseline, 1e-10)


# ═══════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════

CONFIGS = [
    # (N1, N2, N3, mid_dim, epochs, label)
    (20, 14, 10, 4, 60, "Small"),
    (30, 20, 14, 5, 80, "Medium"),
]

LR, EPS, BETA = 0.02, 0.01, 0.85
N_TRIALS = 3

print(f"\n  Configs: {[c[5] for c in CONFIGS]}")
print(f"  Trials: {N_TRIALS}, FD lr={LR}, eps={EPS}, beta={BETA}\n")

all_results = {}

for N1, N2, N3, mid_dim, EPOCHS, label in CONFIGS:
    print(f"{'─'*65}")
    print(f"  CONFIG: {label} — L1({mid_dim}D,{N1}p) → L2({mid_dim}D,{N2}p) → L3(1D,{N3}p)")
    print(f"  FD epochs: {EPOCHS}")
    print(f"{'─'*65}")

    r2_l1_list, r2_l2_list, r2_l3_list = [], [], []
    params_list = []

    for trial in range(N_TRIALS):
        seed = 42 + trial * 100
        np.random.seed(seed)
        t0 = time.time()

        # ── L1: D→mid_dim, N1 particles, target = y ──
        print(f"\n  Trial {trial+1} (seed={seed})")
        l1 = kmeans_init(X_tr, N1, mid_dim, random_state=seed)
        print(f"    L1: {l1.D}D→{mid_dim}D, {N1}p ({l1.get_params_vector().size} params)")
        fd_train_layer(l1, X_tr, y_tr.reshape(-1,1), EPOCHS, LR, EPS, BETA)
        h1_tr = l1.forward(X_tr); h1_te = l1.forward(X_te)
        pred1_tr = h1_tr[:, 0:1]; pred1_te = h1_te[:, 0:1]
        r2_l1 = r2_score(pred1_te, y_te)

        # ── L2: mid_dim→mid_dim, N2 particles, target = y - L1[:,0] ──
        residual1_tr = y_tr.reshape(-1,1) - pred1_tr
        l2 = kmeans_init(h1_tr, N2, mid_dim, random_state=seed+1)
        print(f"    L2: {mid_dim}D→{mid_dim}D, {N2}p ({l2.get_params_vector().size} params)")
        fd_train_layer(l2, h1_tr, residual1_tr, EPOCHS, LR, EPS, BETA)
        h2_tr = l2.forward(h1_tr); h2_te = l2.forward(h1_te)
        pred2_tr = h2_tr[:, 0:1]; pred2_te = h2_te[:, 0:1]
        r2_l2 = r2_score((pred1_te + pred2_te).ravel(), y_te)

        # ── L3: mid_dim→1, N3 particles, target = y - (L1+L2)[:,0] ──
        residual2_tr = y_tr.reshape(-1,1) - (pred1_tr + pred2_tr)
        l3 = kmeans_init(h2_tr, N3, 1, random_state=seed+2)
        print(f"    L3: {mid_dim}D→1D, {N3}p ({l3.get_params_vector().size} params)")
        fd_train_layer(l3, h2_tr, residual2_tr, EPOCHS, LR, EPS, BETA)
        h3_te = l3.forward(h2_te)
        pred3_te = h3_te
        r2_l3 = r2_score((pred1_te + pred2_te + pred3_te).ravel(), y_te)

        total_params = sum(l.get_params_vector().size for l in [l1, l2, l3])
        params_list.append(total_params)
        r2_l1_list.append(r2_l1); r2_l2_list.append(r2_l2); r2_l3_list.append(r2_l3)

        print(f"    Result: L1={r2_l1:.4f}  L1+L2={r2_l2:.4f}  L1+L2+L3={r2_l3:.4f}  "
              f"Δ(L2)={r2_l2-r2_l1:+.4f}  Δ(L3)={r2_l3-r2_l2:+.4f}  "
              f"({time.time()-t0:.0f}s, {total_params} params)")

    all_results[label] = {
        "L1": r2_l1_list, "L1+L2": r2_l2_list, "L1+L2+L3": r2_l3_list,
        "params": params_list,
    }


# ═══════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  SUMMARY")
print(f"{'='*65}")
print(f"  {'Config':<10s} {'L1 R²':>10s}  {'L1+L2 R²':>10s}  {'L1+L2+L3 R²':>10s}  "
      f"{'Δ(L1→L2)':>10s}  {'Δ(L2→L3)':>10s}  {'Total Δ':>10s}  {'Params':>8s}")
print(f"  {'─'*10} {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

for label in [c[5] for c in CONFIGS]:
    d = all_results[label]
    m1, m2, m3 = np.mean(d["L1"]), np.mean(d["L1+L2"]), np.mean(d["L1+L2+L3"])
    s1, s2, s3 = np.std(d["L1"]), np.std(d["L1+L2"]), np.std(d["L1+L2+L3"])
    d12, d23, dtot = m2-m1, m3-m2, m3-m1
    np_mean = int(np.mean(d["params"]))
    print(f"  {label:<10s} {m1:.4f}±{s1:.3f}  {m2:.4f}±{s2:.3f}  {m3:.4f}±{s3:.3f}  "
          f"{d12:+9.4f}  {d23:+9.4f}  {dtot:+9.4f}  {np_mean:>6d}")

# MLP baseline
print(f"\n  ── MLP baseline ──")
from sklearn.neural_network import MLPRegressor
for h in [(20,), (20,14), (20,14,10)]:
    mlp = MLPRegressor(hidden_layer_sizes=h, activation='relu',
                        max_iter=2000, random_state=42, early_stopping=False)
    mlp.fit(X_tr, y_tr.ravel())
    print(f"  MLP{tuple(h)}: R²={r2_score(mlp.predict(X_te), y_te):.4f}")

print(f"\n{'='*65}")
best_label = max([c[5] for c in CONFIGS],
                 key=lambda l: np.mean(all_results[l]["L1+L2+L3"]) - np.mean(all_results[l]["L1"]))
best_d = all_results[best_label]
best_delta = np.mean(best_d["L1+L2+L3"]) - np.mean(best_d["L1"])

if best_delta > 0.05:
    print(f"  ✓ Layer-wise RBF depth WORKS! Best Δ = {best_delta:+.4f} ({best_label})")
    print(f"  → Direct residual targets (no proto smoothing) is the key.")
    print(f"  → Each layer adds significant predictive power.")
elif best_delta > 0.02:
    print(f"  ~ Layer-wise RBF gives moderate depth gain: Δ = {best_delta:+.4f}")
else:
    print(f"  ✗ Depth gain still minimal: Δ = {best_delta:+.4f}")
