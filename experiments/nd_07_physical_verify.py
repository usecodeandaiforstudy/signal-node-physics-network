"""
V3 Physical RBF Network Verification

Tests:
  1. PhysicalResidualRBFNetworkND forward matches HiMALND forward
  2. Proto-weighted predict matches HiMALND.predict
  3. Pure physical forward (no softmax) works end-to-end
  4. Noise robustness of physical forward
  5. Digital RBF network vs Physical RBF network comparison
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.himal_nd import HiMALND, _zero_pad_sum
from simulator.network_nd import PhysicalResidualRBFNetworkND
from network.core_nd import ResidualRBFNetworkND
from physical_simulator import NoiseConfig

# ---------------------------------------------------------------------------
#  THIS SCRIPT'S RESULTS ARE NOW CITABLE.  It printed 22-36 lines and saved
#  NOTHING, so its numbers existed only in a terminal -- no artifact to compare
#  against, no way to notice a change.  This TEES stdout to a report instead of
#  touching the script's own code, so it saves exactly what it prints.
# ---------------------------------------------------------------------------
import atexit as _atexit, io as _io, os as _os, sys as _sys
_REPORT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                        "outputs", "nd_07_physical_verify_report.txt")
_os.makedirs(_os.path.dirname(_REPORT), exist_ok=True)

class _Tee(_io.TextIOBase):
    # I/O IS WRAPPED IN try/except ON PURPOSE, and this is not defensive padding:
    # at interpreter shutdown Python closes the real stdout and then calls flush()
    # here, so an unguarded flush raises ValueError DURING SHUTDOWN and Python turns
    # that into exit code 120.  The first version of this block did exactly that and
    # `nd_07` exited 120 instead of 0, which would break any assert-style check.
    def __init__(self, *streams):
        self._streams = streams
    def write(self, text):
        for s in self._streams:
            try:
                s.write(text)
            except ValueError:
                pass
        return len(text)
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except ValueError:
                pass

_fh = open(_REPORT, "w", encoding="utf-8")
_sys.stdout = _Tee(_sys.stdout, _fh)
_atexit.register(_fh.close)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ──
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

print("=" * 60)
print("  Physical RBF Network Verification")
print("=" * 60)
print(f"  Baseline MSE: {mean_baseline:.3f}")

# ── Step 1: Train HiMAL ──
print("\n  [1] Training HiMAL-FFA-ND (4,3,2)...")
NoiseConfig.set_level("none")
trainer = HiMALND(
    layer_sizes=(4, 3, 2), output_dims=(3, 2, 1),
    input_dim=2, init_range=(-3.0, 3.0),
    alignment_weight=0.3, n_protos=5, temperature=0.15,
)
layers_A, layers_B, _ = trainer.train(
    X_tr, y_tr, split_method="median_x",
    epochs_per_layer=15, lr=0.012, eps=1e-3, verbose=False,
)
pred_himal = trainer.predict(layers_A, layers_B, X_te).ravel()
mse_himal = float(np.mean((pred_himal - y_te) ** 2))
print(f"      HiMALND.predict MSE: {mse_himal:.3f}")

# ── Step 2: Build PhysicalRBFNetwork from trained layers ──
print("\n  [2] Building PhysicalResidualRBFNetworkND from HiMAL layers...")
net_A, net_B = PhysicalResidualRBFNetworkND.from_himal_dual(
    layers_A, layers_B,
    trainer.proto_h_A, trainer.proto_y_A,
    trainer.proto_h_B, trainer.proto_y_B,
)
print(f"      net_A params: {net_A.n_total_params}")
print(f"      net_B params: {net_B.n_total_params}")

# Verify layer output matches
hA0 = trainer._forward_layer(layers_A[0], X_te)
g1, a1 = net_A.layer1.forward(X_te)
h_pnet = g1 + a1
print(f"      L1 output match: max|hA - h_pnet| = {np.max(np.abs(hA0 - h_pnet)):.2e}")

# ── Step 3: Proto-weighted predict matches ──
print("\n  [3] Comparing PhysicalRBFNetwork.predict vs HiMALND.predict...")
pred_pnet = PhysicalResidualRBFNetworkND.predict_dual(net_A, net_B, X_te, temperature=0.15)
mse_pnet = float(np.mean((pred_pnet.ravel() - y_te) ** 2))
match_err = np.max(np.abs(pred_himal - pred_pnet.ravel()))
print(f"      HiMALND MSE:         {mse_himal:.4f}")
print(f"      PhysicalRBFNet MSE:  {mse_pnet:.4f}")
print(f"      Max difference:      {match_err:.2e}")
assert match_err < 1e-4, f"Predictions don't match: max diff = {match_err}"
print("      PASS: predictions match exactly")

# ── Step 4: Pure physical forward (no proto, no softmax) ──
print("\n  [4] Pure physical forward (analog-only, no softmax)...")
NoiseConfig.set_level("none")
f3 = net_A.forward(X_te)  # raw output, no prototype mapping
print(f"      Raw forward output shape: {f3.shape}")
print(f"      Raw output range: [{f3.min():.3f}, {f3.max():.3f}]")
# Direct h predict (first dim)
mse_direct = float(np.mean((f3[:, 0] - y_te) ** 2))
print(f"      Direct h[:,0] MSE: {mse_direct:.3f} (vs proto {mse_pnet:.3f})")
assert np.isfinite(f3).all(), "Forward produced NaN/inf"
print("      PASS: pure physical forward works")

# ── Step 5: Digital vs Physical comparison ──
print("\n  [5] Digital (PyTorch) vs Physical (numpy) comparison...")
from network.core_nd import RBFLayerND
from simulator.layer_nd import PhysicalRBFLayerND

# Build matching digital layer from physical parameters
layer_A0 = layers_A[0]
digital_layer = RBFLayerND(
    dim=layer_A0.dim,
    n_particles=layer_A0.n_particles,
    n_vars=layer_A0.n_vars,
    init_range=(-3, 3),
)
# Copy params: digital uses ShiftND internally
with torch.no_grad():
    digital_layer.shift.centers.copy_(torch.tensor(layer_A0.centers))
    digital_layer.shift.h.copy_(torch.tensor(layer_A0.h))
    digital_layer.output_gauss.weight.copy_(torch.tensor(layer_A0.W_gauss))
    digital_layer.output_act.weight.copy_(torch.tensor(layer_A0.W_act))

X_te_t = torch.tensor(X_te, dtype=torch.float32)
with torch.no_grad():
    out_g_d, out_a_d = digital_layer(X_te_t)
    out_digital = (out_g_d + out_a_d).numpy()

# Physical forward
out_g, out_a = layer_A0.forward(X_te)
out_physical = out_g + out_a

diff = np.max(np.abs(out_digital - out_physical))
print(f"      Max |digital - physical| = {diff:.2e}")
print(f"      Digital output range: [{out_digital.min():.3f}, {out_digital.max():.3f}]")
print(f"      Physical output range: [{out_physical.min():.3f}, {out_physical.max():.3f}]")
assert diff < 0.5, f"Digital-physical mismatch: max diff = {diff:.3f}"
print("      PASS: digital and physical forward produce similar output")

# ── Step 6: Noise sweep ──
print("\n  [6] Noise robustness of physical forward...")
for level in ["none", "low", "medium", "high"]:
    NoiseConfig.set_level(level)
    preds = []
    for _ in range(5):
        p = PhysicalResidualRBFNetworkND.predict_dual(net_A, net_B, X_te, temperature=0.15)
        preds.append(p.ravel())
    stacked = np.stack(preds, axis=0)
    mean_pred = stacked.mean(axis=0)
    std_pred = stacked.std(axis=0)
    mse_noise = float(np.mean((mean_pred - y_te) ** 2))
    print(f"      {level:<8s}: MSE={mse_noise:.4f}  mean_std={std_pred.mean():.3f}")

NoiseConfig.set_level("none")

# ── Summary ──
print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
print(f"  HiMALND.predict:          {mse_himal:.4f}")
print(f"  PhysicalRBFNet.predict:   {mse_pnet:.4f}  (match: {match_err:.1e})")
print(f"  Pure forward h[:,0]:      {mse_direct:.4f}")
print(f"  Digital-physical diff:    max {diff:.2e}")
print(f"  All tests: PASS")
print("=" * 60)
