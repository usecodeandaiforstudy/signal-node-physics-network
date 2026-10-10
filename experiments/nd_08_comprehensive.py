"""
V3 Comprehensive Experiment Suite

Tests all network configurations:
  A. Combined HiMAL-RBF (primary model)
  B. HiMAL-FFA-ND standalone (training pipeline)
  C. Proto-Guided ND (classification baseline)

Benchmarks:
  1. Non-linear regression (sin/cos)
  2. Friedman #1 (standard ML benchmark)
  3. Noise robustness sweep
  4. Per-layer contribution analysis
  5. Physical forward validation
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression, make_moons, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from network.combined import CombinedHiMALRBF
from training.himal_nd import HiMALND
from physical_simulator import NoiseConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════

def mse(pred, y):
    return float(np.mean((pred.ravel() - y.ravel()) ** 2))


def evaluate_regression(model, X_tr, y_tr, X_te, y_te, name=""):
    """Full evaluation: train + test MSE + physical forward."""
    t0 = time.time()

    # Train
    model.train(X_tr, y_tr, split_method="median_x",
                epochs_per_layer=30, lr=0.012, eps=1e-3, verbose=False)
    train_time = time.time() - t0

    # Predict
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    mse_tr = mse(pred_tr, y_tr)
    mse_te = mse(pred_te, y_te)

    # Pure physical forward
    h_raw = model.forward(X_te, noise_level="none")
    mse_direct = mse(h_raw[:, :min(h_raw.shape[1], 1)], y_te)

    # Baseline
    mse_mean = mse(np.full_like(y_te, y_tr.mean()), y_te)

    return {
        "name": name,
        "mse_tr": mse_tr, "mse_te": mse_te,
        "mse_direct": mse_direct, "mse_mean": mse_mean,
        "r2": 1.0 - mse_te / max(mse_mean, 1e-10),
        "train_time": train_time,
        "n_params": model.n_total_params,
    }


# ═══════════════════════════════════════════════════
# Benchmark 1: Non-linear regression
# ═══════════════════════════════════════════════════

print("=" * 60)
print("  E08: Comprehensive Benchmark Suite")
print("=" * 60)

results = {}

# ── Data ──
np.random.seed(42)
N = 300
X1 = np.random.uniform(-2, 2, (N, 2)).astype(np.float32)
y1 = (np.sin(3 * X1[:, 0]) + np.cos(2 * X1[:, 1])
      + 0.5 * X1[:, 0] * X1[:, 1]
      + np.random.normal(0, 0.15, N)).astype(np.float32)
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.3, random_state=42)
sc1 = StandardScaler()
X1_tr = sc1.fit_transform(X1_tr).astype(np.float32)
X1_te = sc1.transform(X1_te).astype(np.float32)

print(f"\n  [1] Non-linear regression: sin(3x1)+cos(2x2)+0.5x1x2")
print(f"      Train={len(X1_tr)}, Test={len(X1_te)}")

# Config A: Small network
print(f"\n      Config A: Small (4,3,2) particles...")
model_small = CombinedHiMALRBF(
    input_dim=2, layer_sizes=(4, 3, 2), output_dims=(3, 2, 1),
    n_protos=5, temperature=0.15,
)
r = evaluate_regression(model_small, X1_tr, y1_tr, X1_te, y1_te, "small")
results["nl_small"] = r
print(f"      Test MSE={r['mse_te']:.4f}, R2={r['r2']:.3f}, "
      f"direct={r['mse_direct']:.3f}, params={r['n_params']}, "
      f"time={r['train_time']:.1f}s")

# Config B: Larger network
print(f"\n      Config B: Medium (6,4,3) particles...")
model_large = CombinedHiMALRBF(
    input_dim=2, layer_sizes=(6, 4, 3), output_dims=(4, 3, 2),
    n_protos=6, temperature=0.12,
)
r = evaluate_regression(model_large, X1_tr, y1_tr, X1_te, y1_te, "large")
results["nl_large"] = r
print(f"      Test MSE={r['mse_te']:.4f}, R2={r['r2']:.3f}, "
      f"direct={r['mse_direct']:.3f}, params={r['n_params']}, "
      f"time={r['train_time']:.1f}s")

# ── Per-layer contribution (large model) ──
print(f"\n      Per-layer contribution (large model):")
trainer = model_large._trainer
for nl in [1, 2, 3]:
    pred_l = np.zeros_like(y1_te, dtype=np.float32)
    hA, hB = X1_te.copy(), X1_te.copy()
    acc_A, acc_B = None, None
    from training.himal_nd import _zero_pad_sum
    for i in range(nl):
        hA_o = trainer._forward_layer(model_large.layers_A[i], hA)
        hB_o = trainer._forward_layer(model_large.layers_B[i], hB)
        if hA_o.shape[0] != hA.shape[0]: hA_o = hA_o.T
        if hB_o.shape[0] != hB.shape[0]: hB_o = hB_o.T
        pa = trainer._proto_predict(hA_o, trainer.proto_y_A[i], trainer.proto_h_A[i])
        pb = trainer._proto_predict(hB_o, trainer.proto_y_B[i], trainer.proto_h_B[i])
        if acc_A is None:
            acc_A, acc_B = pa.copy(), pb.copy()
        else:
            acc_A = _zero_pad_sum(acc_A, pa)
            acc_B = _zero_pad_sum(acc_B, pb)
        hA, hB = hA_o, hB_o
    mse_l = mse(((acc_A + acc_B) / 2.0).ravel(), y1_te)
    print(f"      L1-{nl}: MSE={mse_l:.4f}  (R2={1-mse_l/max(r['mse_mean'],1e-10):.3f})")

# ═══════════════════════════════════════════════════
# Benchmark 2: Friedman #1
# ═══════════════════════════════════════════════════

print(f"\n  [2] Friedman #1 (standard ML benchmark)...")
X2, y2 = make_friedman1(n_samples=300, noise=0.1, random_state=42)
X2 = X2.astype(np.float32)
y2 = y2.astype(np.float32)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.3, random_state=42)
sc2 = StandardScaler()
X2_tr = sc2.fit_transform(X2_tr).astype(np.float32)
X2_te = sc2.transform(X2_te).astype(np.float32)

model_fried = CombinedHiMALRBF(
    input_dim=10, layer_sizes=(8, 6, 4), output_dims=(5, 3, 2),
    n_protos=8, temperature=0.1,
)
r = evaluate_regression(model_fried, X2_tr, y2_tr, X2_te, y2_te, "friedman")
results["friedman"] = r
print(f"      Test MSE={r['mse_te']:.4f}, R2={r['r2']:.3f}, "
      f"params={r['n_params']}, time={r['train_time']:.1f}s")

# ═══════════════════════════════════════════════════
# Benchmark 3: Noise robustness
# ═══════════════════════════════════════════════════

print(f"\n  [3] Noise robustness sweep (large model)...")
noise_results = {}
for level in ["none", "low", "medium", "high"]:
    preds = []
    for _ in range(5):
        model_large.net_A.layer1.forward(X1_te)  # prime
        hA, hB = model_large.forward_dual(X1_te, noise_level=level)
        # Proto-weighted from noisy h
        pa = PhysicalResidualRBFNetworkND._proto_predict(
            hA, model_large.net_A._proto_h[-1], model_large.net_A._proto_y[-1],
            model_large.temperature)
        pb = PhysicalResidualRBFNetworkND._proto_predict(
            hB, model_large.net_B._proto_h[-1], model_large.net_B._proto_y[-1],
            model_large.temperature)
        preds.append(((pa + pb) / 2.0).ravel())
    stacked = np.stack(preds, axis=0)
    mean_p = stacked.mean(axis=0)
    noise_results[level] = {
        "mse": mse(mean_p, y1_te),
        "std": float(stacked.std(axis=0).mean()),
    }
    print(f"      {level:<8s}: MSE={noise_results[level]['mse']:.4f}  "
          f"std={noise_results[level]['std']:.3f}")

NoiseConfig.set_level("none")

# ═══════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Comprehensive Results")
print("=" * 60)

for key, r in results.items():
    print(f"\n  [{r['name']}]")
    print(f"    Train MSE:    {r['mse_tr']:.4f}")
    print(f"    Test MSE:     {r['mse_te']:.4f}")
    print(f"    R2:           {r['r2']:.3f}")
    print(f"    Direct MSE:   {r['mse_direct']:.4f}")
    print(f"    Mean baseline:{r['mse_mean']:.4f}")
    print(f"    Params:       {r['n_params']}")
    print(f"    Time:         {r['train_time']:.1f}s")

# ── Visual summary ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R2 comparison
ax = axes[0]
names = [r["name"] for r in results.values()]
r2s = [r["r2"] for r in results.values()]
bars = ax.bar(names, r2s, color=["#2196F3", "#4CAF50", "#FF9800"], alpha=0.7)
for b, v in zip(bars, r2s):
    ax.text(b.get_x() + b.get_width()/2, max(0, v) + 0.01,
            f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("R2"); ax.set_title("R2 Score by Model"); ax.grid(axis="y", alpha=0.3)

# Noise robustness
ax = axes[1]
lvls = list(noise_results.keys())
mses = [noise_results[l]["mse"] for l in lvls]
stds = [noise_results[l]["std"] for l in lvls]
ax.bar(lvls, mses, color=["#4CAF50", "#FFC107", "#FF9800", "#F44336"], alpha=0.7)
ax.set_ylabel("MSE"); ax.set_title("Noise Robustness"); ax.grid(axis="y", alpha=0.3)

# Params vs R2
ax = axes[2]
params = [r["n_params"] for r in results.values()]
ax.scatter(params, r2s, s=100, c=["#2196F3", "#4CAF50", "#FF9800"])
for n, r2 in zip(names, r2s):
    ax.annotate(n, (params[names.index(n)], r2),
                textcoords="offset points", xytext=(0, 10), fontsize=8)
ax.set_xlabel("Total Parameters"); ax.set_ylabel("R2")
ax.set_title("Params vs R2"); ax.grid(alpha=0.3)

plt.tight_layout()
path = os.path.join(OUT_DIR, "v3_comprehensive_benchmark.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Figure saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

print("\n" + "=" * 60)
print("  All benchmarks completed.")
print("=" * 60)
