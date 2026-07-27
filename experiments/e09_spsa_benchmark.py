"""
SPSA vs Per-Parameter FD Speed & Accuracy Comparison

Tests:
  A. Same model config, two training modes
  B. Speed: forward passes per epoch counted
  C. Accuracy: loss curve comparison
  D. Scalability: how speed gap grows with network size
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from network.combined import CombinedHiMALRBF
from physical_simulator import NoiseConfig

NoiseConfig.set_level("none")


def mse(pred, y):
    return float(np.mean((pred.ravel() - y.ravel()) ** 2))


# ── Data ──
np.random.seed(42)
N = 200
X = np.random.uniform(-2, 2, (N, 2)).astype(np.float32)
y = (np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1])
     + 0.5 * X[:, 0] * X[:, 1]
     + np.random.normal(0, 0.15, N)).astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr).astype(np.float32)
X_te = sc.transform(X_te).astype(np.float32)
baseline = mse(np.full_like(y_te, y_tr.mean()), y_te)
n_params_per_net = 0  # computed below

print("=" * 60)
print("  SPSA vs Per-Parameter FD Comparison")
print("=" * 60)
print(f"  Baseline MSE: {baseline:.3f}")

# ── Test 1: Small network speed comparison ──
print("\n  [1] Small network (4,3,2), 50 epochs each...")

# FD mode
t0 = time.time()
m_fd = CombinedHiMALRBF(
    input_dim=2, layer_sizes=(4, 3, 2), output_dims=(3, 2, 1),
    n_protos=5, use_spsa=False,
)
m_fd.train(X_tr, y_tr, epochs_per_layer=30, lr=0.012, eps=1e-3, verbose=False)
t_fd = time.time() - t0
pred_fd = m_fd.predict(X_te)
mse_fd = mse(pred_fd, y_te)
n_params_per_net = m_fd.n_params_per_network

# SPSA mode
t0 = time.time()
m_spsa = CombinedHiMALRBF(
    input_dim=2, layer_sizes=(4, 3, 2), output_dims=(3, 2, 1),
    n_protos=5, use_spsa=True, spsa_avg=1,
)
m_spsa.train(X_tr, y_tr, epochs_per_layer=30, lr=0.012, eps=1e-3, verbose=False)
t_spsa = time.time() - t0
pred_spsa = m_spsa.predict(X_te)
mse_spsa = mse(pred_spsa, y_te)

print(f"  FD:    MSE={mse_fd:.4f}  time={t_fd:.1f}s  ({n_params_per_net} params/net)")
print(f"  SPSA:  MSE={mse_spsa:.4f}  time={t_spsa:.1f}s  (avg={m_spsa.spsa_avg})")
print(f"  Speedup: {t_fd/max(t_spsa,0.01):.1f}x")

# ── Test 2: SPSA with more epochs matches FD ──
print("\n  [2] SPSA more epochs vs FD fewer epochs (same wall time budget)...")
# Use FD time as budget, run SPSA with many more epochs
n_epochs_spsa = max(10, int(30 * t_fd / max(t_spsa, 0.01)))

t0 = time.time()
m_spsa2 = CombinedHiMALRBF(
    input_dim=2, layer_sizes=(4, 3, 2), output_dims=(3, 2, 1),
    n_protos=5, use_spsa=True, spsa_avg=1,
)
m_spsa2.train(X_tr, y_tr, epochs_per_layer=n_epochs_spsa, lr=0.012, eps=1e-3, verbose=False)
t_spsa2 = time.time() - t0
pred_spsa2 = m_spsa2.predict(X_te)
mse_spsa2 = mse(pred_spsa2, y_te)
print(f"  FD:    30 epochs, MSE={mse_fd:.4f}, time={t_fd:.1f}s")
print(f"  SPSA:  {n_epochs_spsa} epochs, MSE={mse_spsa2:.4f}, time={t_spsa2:.1f}s")
print(f"  SPSA has {n_epochs_spsa}x more iterations, MSE gap: {mse_spsa2-mse_fd:+.4f}")

# ── Test 3: Multi-avg SPSA ──
print("\n  [3] SPSA with multi-average (reduces gradient noise)...")
avg_results = []
for avg in [1, 3, 5]:
    t0 = time.time()
    m_a = CombinedHiMALRBF(
        input_dim=2, layer_sizes=(4, 3, 2), output_dims=(3, 2, 1),
        n_protos=5, use_spsa=True, spsa_avg=avg,
    )
    m_a.train(X_tr, y_tr, epochs_per_layer=30, lr=0.012, eps=1e-3, verbose=False)
    t_a = time.time() - t0
    p_a = m_a.predict(X_te)
    avg_results.append({
        "avg": avg, "mse": mse(p_a, y_te), "time": t_a,
    })
    print(f"  avg={avg}:  MSE={avg_results[-1]['mse']:.4f}  time={t_a:.1f}s")

# ── Test 4: Scalability ──
print("\n  [4] Speedup vs network size...")
sizes = [(3,2,2), (4,3,2), (6,4,3)]
for sz in sizes:
    n_p = 0
    # Count params without training
    tmp = CombinedHiMALRBF(input_dim=2, layer_sizes=sz, output_dims=(3,2,1), n_protos=4)
    tmp._trainer = type('T', (), {'use_spsa': False, 'spsa_avg': 1})()
    # Compute params
    from training.himal_nd import HiMALND
    hh = HiMALND(layer_sizes=sz, output_dims=(3,2,1), input_dim=2)
    A, B = hh._init_random_layer(sz[0], 2, 3)
    n_p = A.get_params_vector().size
    fd_cost_per_epoch = 2 * n_p * 2  # 2 networks, 2 fwd per param
    spsa_cost_per_epoch = 2 * 2  # 2 networks, 2 fwd per epoch
    speedup = fd_cost_per_epoch / spsa_cost_per_epoch
    print(f"  {sz}:  {n_p} params/net  →  SPSA is {speedup:.0f}x fewer fwd/epoch")

# ── Summary ──
print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
print(f"  Network:       (4,3,2) particles, {n_params_per_net} params/net")
print(f"  FD time:       {t_fd:.1f}s  (MSE={mse_fd:.4f})")
print(f"  SPSA time:     {t_spsa:.1f}s  (MSE={mse_spsa:.4f})")
print(f"  Raw speedup:   {t_fd/max(t_spsa,0.01):.1f}x")
print(f"  SPSA w/ {n_epochs_spsa}x epochs: MSE={mse_spsa2:.4f}")
print("=" * 60)
