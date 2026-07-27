"""
V3 HiMAL-FFA-ND: Dual-Network Proto-Weighted Training (Debug Script)

Verifies:
  1. split_data — A/B/T partition
  2. layer init + forward — RBF physical forward
  3. train_one_layer — per-parameter FD with MSE loss
  4. train — full 3-layer pipeline, proto storage
  5. predict — proto-weighted inference
  6. loss decreases over training
  7. prediction better than random
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.himal_nd import HiMALND
from physical_simulator import NoiseConfig

# ── Setup ──
NoiseConfig.set_level("none")

# ── Test data: small 2D regression for speed ──
print("=" * 60)
print("  HiMAL-FFA-ND Debug Run")
print("=" * 60)

X, y = make_regression(n_samples=120, n_features=2, n_informative=2,
                        noise=0.15, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr).astype(np.float32)
X_te = sc.transform(X_te).astype(np.float32)
y_tr = y_tr.astype(np.float32)
y_te = y_te.astype(np.float32)

print(f"  Data: 120 samples, 2D input → scalar output")
print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

# ── Test 1: split_data ──
print("\n  [1] Testing split_data...")
trainer = HiMALND(
    layer_sizes=(3, 2, 2),
    output_dims=(3, 2, 1),
    input_dim=2,
    init_range=(-3.0, 3.0),
    alignment_weight=0.3,
    n_protos=4,
    temperature=0.1,
)
data = trainer.split_data(X_tr, y_tr, method="median_x")
print(f"      A: {len(data['XA'])}, B: {len(data['XB'])}, T: {len(data['XT'])}")
assert len(data['XA']) > 0, "A zone empty"
assert len(data['XB']) > 0, "B zone empty"
print("      PASS")

# ── Test 2: layer init + forward ──
print("\n  [2] Testing layer init + forward...")
layer_A, layer_B = trainer._init_random_layer(3, 2, 3)
hA = trainer._forward_layer(layer_A, data['XA'])
hB = trainer._forward_layer(layer_B, data['XB'])
print(f"      hA shape: {hA.shape}, hB shape: {hB.shape}")
assert hA.shape[0] == len(data['XA']), f"hA batch mismatch"
assert hA.shape[1] == 3, f"hA dim: expected 3, got {hA.shape[1]}"
print("      PASS")

# ── Test 3: compute_pair_loss ──
print("\n  [3] Testing _compute_pair_loss...")
loss = trainer._compute_pair_loss(layer_A, layer_B, data, trainer.alignment_weight)
print(f"      initial loss: {loss:.4f}")
assert np.isfinite(loss), "Loss NaN/inf"
print("      PASS")

# ── Test 4: train_one_layer ──
print("\n  [4] Testing train_one_layer (10 epochs)...")
layer_A, layer_B = trainer.train_one_layer(
    data, n_particles=3, input_dim=2, n_vars=3,
    epochs=10, lr=0.01, eps=1e-3, verbose=True,
)
loss_final = trainer._compute_pair_loss(layer_A, layer_B, data, trainer.alignment_weight)
print(f"      final loss: {loss_final:.4f}")
assert np.isfinite(loss_final), "Loss NaN/inf after training"
print("      PASS")

# ── Test 5: create_prototypes + proto_predict ──
print("\n  [5] Testing _create_prototypes + _proto_predict...")
y_2d = data['yA'].reshape(-1, 1) if data['yA'].ndim == 1 else data['yA']
y_proto, h_proto, assign = trainer._create_prototypes(y_2d, hA, n_protos=4)
pred_A = trainer._proto_predict(hA, y_proto, h_proto)
print(f"      y_protos: {y_proto.shape}, h_protos: {h_proto.shape}, pred: {pred_A.shape}")
assert pred_A.shape[0] == len(data['XA'])
assert pred_A.shape[1] == y_proto.shape[1]
print("      PASS")

# ── Test 6: full train pipeline ──
print("\n  [6] Testing full train pipeline...")
trainer2 = HiMALND(
    layer_sizes=(4, 3, 2),
    output_dims=(4, 3, 1),
    input_dim=2,
    init_range=(-3.0, 3.0),
    alignment_weight=0.3,
    n_protos=4,
    temperature=0.1,
)
layers_A, layers_B, data_split = trainer2.train(
    X_tr, y_tr, split_method="median_x",
    epochs_per_layer=40, lr=0.012, eps=1e-3,
    verbose=True,
)
assert len(trainer2.proto_h_A) == 3, f"Expected 3 proto layers, got {len(trainer2.proto_h_A)}"
print(f"      Stored prototypes: OK")
print("      PASS")

# ── Test 7: predict ──
print("\n  [7] Testing predict...")
pred_tr = trainer2.predict(layers_A, layers_B, X_tr)
pred_te = trainer2.predict(layers_A, layers_B, X_te)
print(f"      pred_tr: {pred_tr.shape}, pred_te: {pred_te.shape}")

mse_tr = float(np.mean((pred_tr.ravel() - y_tr) ** 2))
mse_te = float(np.mean((pred_te.ravel() - y_te) ** 2))
mean_pred = np.full_like(y_te, y_tr.mean())
mse_mean = float(np.mean((mean_pred - y_te) ** 2))

print(f"      Train MSE: {mse_tr:.4f}")
print(f"      Test MSE:  {mse_te:.4f}")
print(f"      Mean baseline: {mse_mean:.4f}")
assert mse_te < mse_mean * 1.2, f"Test MSE ({mse_te:.4f}) not better than mean ({mse_mean:.4f})"
print("      PASS")

# ── Test 8: loss decrease ──
print("\n  [8] Testing loss decreases...")
trainer3 = HiMALND(
    layer_sizes=(3, 2, 2), output_dims=(3, 2, 1),
    input_dim=2, init_range=(-3.0, 3.0), n_protos=4,
)
data3 = trainer3.split_data(X_tr, y_tr, method="median_x")
ld = {"XA": data3["XA"], "yA": data3["yA"],
      "XB": data3["XB"], "yB": data3["yB"],
      "XT": data3["XT"]}
lA, lB = trainer3._init_random_layer(3, 2, 3)
loss_init = trainer3._compute_pair_loss(lA, lB, ld, trainer3.alignment_weight)
lA2, lB2 = trainer3.train_one_layer(ld, 3, 2, 3, epochs=10, lr=0.01, eps=1e-3, verbose=False)
loss_end = trainer3._compute_pair_loss(lA2, lB2, ld, trainer3.alignment_weight)
print(f"      {loss_init:.4f} → {loss_end:.4f} (ratio={loss_end/loss_init:.3f})")
assert loss_end < loss_init, f"Loss did not decrease"
print("      PASS")

# ── Summary ──
print("\n" + "=" * 60)
print("  ALL TESTS PASSED")
print(f"  Test MSE: {mse_te:.4f} (baseline: {mse_mean:.4f})")
print("=" * 60)
