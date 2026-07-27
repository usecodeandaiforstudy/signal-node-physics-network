"""
V3 HiMAL-FFA-ND: Non-Linear Regression — Layer Contribution Test

Tests whether L2/L3 contribute on a genuinely non-linear task:
  y = sin(3*x1) + cos(2*x2) + 0.5*x1*x2 + noise

Checks:
  1. Per-layer contribution (L1 only / L1+L2 / full ensemble)
  2. Per-layer training loss progression
  3. Proto-weighted vs direct-h inference
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.himal_nd import HiMALND, _zero_pad_sum
from physical_simulator import NoiseConfig

NoiseConfig.set_level("none")

# ── Non-linear regression data ──
print("=" * 60)
print("  Non-Linear Regression: L2/L3 Contribution Test")
print("=" * 60)

np.random.seed(42)
N = 200
X = np.random.uniform(-2, 2, (N, 2)).astype(np.float32)
y = (np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 1]
     + np.random.normal(0, 0.15, N)).astype(np.float32)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr).astype(np.float32)
X_te = sc.transform(X_te).astype(np.float32)

mean_baseline = float(np.mean((y_te - y_tr.mean()) ** 2))
print(f"  y = sin(3x1) + cos(2x2) + 0.5x1x2 + noise")
print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, Baseline MSE: {mean_baseline:.2f}")

# ── Train ──
print("\n  Training HiMAL-FFA-ND (3, 2, 2) × 40 epochs...")
trainer = HiMALND(
    layer_sizes=(3, 2, 2),
    output_dims=(2, 2, 1),
    input_dim=2,
    init_range=(-3.0, 3.0),
    alignment_weight=0.3,
    n_protos=6,
    temperature=0.15,
)
layers_A, layers_B, _ = trainer.train(
    X_tr, y_tr, split_method="median_x",
    epochs_per_layer=40, lr=0.012, eps=1e-3,
    verbose=True,
)

# ── A: Per-layer contribution ──
print("\n  [A] Per-layer contribution (proto-weighted):")

def predict_layers(trainer, layers_A, layers_B, X, n_layers):
    """Proto-weighted inference using first n_layers only."""
    hA, hB = X.copy(), X.copy()
    acc_A, acc_B = None, None
    for i in range(n_layers):
        hA_out = trainer._forward_layer(layers_A[i], hA)
        hB_out = trainer._forward_layer(layers_B[i], hB)
        if hA_out.shape[0] != hA.shape[0]: hA_out = hA_out.T
        if hB_out.shape[0] != hB.shape[0]: hB_out = hB_out.T
        pa = trainer._proto_predict(hA_out, trainer.proto_y_A[i], trainer.proto_h_A[i])
        pb = trainer._proto_predict(hB_out, trainer.proto_y_B[i], trainer.proto_h_B[i])
        if acc_A is None:
            acc_A, acc_B = pa.copy(), pb.copy()
        else:
            acc_A = _zero_pad_sum(acc_A, pa)
            acc_B = _zero_pad_sum(acc_B, pb)
        hA, hB = hA_out, hB_out
    return ((acc_A + acc_B) / 2.0).ravel()

def mse(pred, y):
    return float(np.mean((pred - y) ** 2))

mse_l1 = mse(predict_layers(trainer, layers_A, layers_B, X_te, 1), y_te)
mse_l2 = mse(predict_layers(trainer, layers_A, layers_B, X_te, 2), y_te)
mse_l3 = mse(predict_layers(trainer, layers_A, layers_B, X_te, 3), y_te)

print(f"  L1 only:          MSE={mse_l1:.3f}  (R2={1-mse_l1/mean_baseline:.3f})")
print(f"  L1+L2:            MSE={mse_l2:.3f}  (Δ={mse_l1-mse_l2:+.3f}, L2 contrib={(mse_l1-mse_l2)/max(mean_baseline-mse_l1,1e-8)*100:.0f}%)")
print(f"  L1+L2+L3 (full):  MSE={mse_l3:.3f}  (Δ={mse_l2-mse_l3:+.3f}, L3 contrib={(mse_l2-mse_l3)/max(mean_baseline-mse_l1,1e-8)*100:.0f}%)")

# ── B: Old vs New predict ──
print("\n  [B] New (proto-weighted) vs Old (direct h):")
def old_predict(layers_A, layers_B, X):
    hA, hB = X.copy(), X.copy()
    for i in range(len(layers_A)):
        hA_out = trainer._forward_layer(layers_A[i], hA)
        hB_out = trainer._forward_layer(layers_B[i], hB)
        if hA_out.shape[0] != hA.shape[0]: hA_out = hA_out.T
        if hB_out.shape[0] != hB.shape[0]: hB_out = hB_out.T
        if i == 0:
            hA, hB = hA_out, hB_out
        else:
            hA = _zero_pad_sum(hA, hA_out)
            hB = _zero_pad_sum(hB, hB_out)
    return ((hA[:, 0] + hB[:, 0]) / 2.0)

mse_old = mse(old_predict(layers_A, layers_B, X_te), y_te)
print(f"  Old (direct h):   MSE={mse_old:.3f}  (R2={1-mse_old/mean_baseline:.3f})")
print(f"  New (proto-wtd):  MSE={mse_l3:.3f}  (R2={1-mse_l3/mean_baseline:.3f})")
print(f"  Improvement:      {(mse_old-mse_l3)/mse_old*100:.1f}%")

# ── C: Per-layer training loss ──
print("\n  [C] Per-layer direct-h fit (on A-zone):")
cur_X = X_tr
data_masks = trainer.split_data(X_tr, y_tr, method="median_x")["masks"]
for i in range(3):
    X_a = cur_X[data_masks["A"]]
    hA = trainer._forward_layer(layers_A[i], X_a)
    yA = y_tr[data_masks["A"]]
    if hA.shape[0] != len(yA): hA = hA.T
    mse_A = float(np.mean((hA[:, 0] - yA) ** 2))
    print(f"  L{i+1}: fit_A (direct) = {mse_A:.3f}  proto_y dim = {trainer.proto_y_A[i].shape}")
    # move to next layer input
    h_all = trainer._forward_layer(layers_A[i], cur_X)
    if h_all.shape[0] != len(cur_X): h_all = h_all.T
    cur_X = h_all

# ── Summary ──
print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
print(f"  Baseline (mean):   {mean_baseline:.3f}")
print(f"  L1 only:           {mse_l1:.3f}  (R2: {1-mse_l1/mean_baseline:.3f})")
print(f"  L1+L2:             {mse_l2:.3f}  (R2: {1-mse_l2/mean_baseline:.3f})")
print(f"  Full ensemble:     {mse_l3:.3f}  (R2: {1-mse_l3/mean_baseline:.3f})")
print(f"  Old direct h:      {mse_old:.3f}  (R2: {1-mse_old/mean_baseline:.3f})")
l2_gain = mse_l1 - mse_l2
l3_gain = mse_l2 - mse_l3
print(f"  L2 gain:           {l2_gain:+.3f} MSE")
print(f"  L3 gain:           {l3_gain:+.3f} MSE")
if l2_gain > 0.005:
    print(f"  [OK] L2 contributes positively on non-linear task")
else:
    print(f"  [--] L2 does not help on this task")
if l3_gain > 0.005:
    print(f"  [OK] L3 contributes positively on non-linear task")
else:
    print(f"  [--] L3 does not help on this task")
print("=" * 60)
