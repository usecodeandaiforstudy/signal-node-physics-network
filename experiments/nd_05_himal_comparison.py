"""
HiMAL-FFA-ND Layer Contribution & Performance Comparison

Tests:
  A. Per-layer contribution: L1-only vs L1+L2 vs full ensemble
  B. New (proto-weighted) vs Old (direct h accumulation) predict
  C. Multiple random seeds for statistical comparison
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.himal_nd import HiMALND, _zero_pad_sum
from physical_simulator import NoiseConfig

# ---------------------------------------------------------------------------
#  THIS SCRIPT'S RESULTS ARE NOW CITABLE.  It printed 22-36 lines and saved
#  NOTHING, so its numbers existed only in a terminal -- no artifact to compare
#  against, no way to notice a change.  This TEES stdout to a report instead of
#  touching the script's own code, so it saves exactly what it prints.
# ---------------------------------------------------------------------------
import atexit as _atexit, io as _io, os as _os, sys as _sys
_REPORT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                        "outputs", "nd_05_himal_comparison_report.txt")
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


NoiseConfig.set_level("none")


def old_predict(layers_A, layers_B, X):
    """Original direct-h accumulation predict (v1-style, no proto)."""
    hA, hB = X.copy(), X.copy()
    for i in range(len(layers_A)):
        hA_out = trainer._forward_layer(layers_A[i], hA)
        hB_out = trainer._forward_layer(layers_B[i], hB)
        if hA_out.shape[0] != hA.shape[0]:
            hA_out = hA_out.T
        if hB_out.shape[0] != hB.shape[0]:
            hB_out = hB_out.T
        if i == 0:
            hA, hB = hA_out, hB_out
        else:
            hA = _zero_pad_sum(hA, hA_out)
            hB = _zero_pad_sum(hB, hB_out)
    # Take first output dim (y is scalar), average dual networks
    return ((hA[:, 0] + hB[:, 0]) / 2.0)


# ── Generate 2D regression data ──
X, y = make_regression(n_samples=200, n_features=2, n_informative=2,
                        noise=0.15, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr).astype(np.float32)
X_te = sc.transform(X_te).astype(np.float32)
y_tr = y_tr.astype(np.float32)
y_te = y_te.astype(np.float32)
mean_baseline = float(np.mean((y_te - y_tr.mean()) ** 2))

print("=" * 60)
print("  Layer Contribution & Performance Comparison")
print("=" * 60)
print(f"  Data: 140 train / 60 test, 2D → scalar")
print(f"  Mean baseline MSE: {mean_baseline:.2f}")

# ── Train ──
print("\n  Training HiMAL-FFA-ND (4,3,2) particles × 60 epochs...")
trainer = HiMALND(
    layer_sizes=(4, 3, 2), output_dims=(4, 3, 1),
    input_dim=2, init_range=(-3.0, 3.0),
    alignment_weight=0.3, n_protos=4, temperature=0.1,
)
layers_A, layers_B, _ = trainer.train(
    X_tr, y_tr, split_method="median_x",
    epochs_per_layer=40, lr=0.012, eps=1e-3, verbose=True,
)

# ── A: Per-layer contribution (proto-weighted) ──
print("\n  [A] Per-layer contribution (proto-weighted predict):")

# L1 only
pred_l1_new = np.zeros_like(y_te, dtype=np.float32)
hA, hB = X_te.copy(), X_te.copy()
for i in range(1):  # just L1
    hA_out = trainer._forward_layer(layers_A[i], hA)
    hB_out = trainer._forward_layer(layers_B[i], hB)
    if hA_out.shape[0] != hA.shape[0]: hA_out = hA_out.T
    if hB_out.shape[0] != hB.shape[0]: hB_out = hB_out.T
    pa = trainer._proto_predict(hA_out, trainer.proto_y_A[i], trainer.proto_h_A[i])
    pb = trainer._proto_predict(hB_out, trainer.proto_y_B[i], trainer.proto_h_B[i])
    pred_l1_new = ((pa + pb) / 2.0).ravel()
mse_l1_new = float(np.mean((pred_l1_new - y_te) ** 2))

# L1+L2
hA, hB = X_te.copy(), X_te.copy()
acc_A, acc_B = None, None
for i in range(2):
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
pred_l12_new = ((acc_A + acc_B) / 2.0).ravel()
mse_l12_new = float(np.mean((pred_l12_new - y_te) ** 2))

# Full ensemble (L1+L2+L3)
pred_full_new = trainer.predict(layers_A, layers_B, X_te).ravel()
mse_full_new = float(np.mean((pred_full_new - y_te) ** 2))

print(f"  L1 only:          MSE={mse_l1_new:.2f}  (baseline-{mean_baseline-mse_l1_new:.0f})")
print(f"  L1+L2:            MSE={mse_l12_new:.2f}  (Δ={mse_l1_new-mse_l12_new:+.1f})")
print(f"  L1+L2+L3 (full):  MSE={mse_full_new:.2f}  (Δ={mse_l12_new-mse_full_new:+.1f})")

# ── B: New vs Old predict ──
print("\n  [B] New (proto-weighted) vs Old (direct h) predict:")
pred_old = old_predict(layers_A, layers_B, X_te).ravel()
mse_old = float(np.mean((pred_old - y_te) ** 2))
print(f"  Old (direct h):   MSE={mse_old:.2f}  (vs baseline {mean_baseline:.0f}, -{mean_baseline-mse_old:.0f})")
print(f"  New (proto-wtd):  MSE={mse_full_new:.2f}  (vs baseline {mean_baseline:.0f}, -{mean_baseline-mse_full_new:.0f})")
print(f"  Improvement:      {(mse_old-mse_full_new)/mse_old*100:.1f}% over old method")

# ── Summary ──
print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
l1_gain = mean_baseline - mse_l1_new
l2_gain = mse_l1_new - mse_l12_new
l3_gain = mse_l12_new - mse_full_new
print(f"  L1 contribution:     {l1_gain:.0f} MSE reduction ({l1_gain/mean_baseline*100:.0f}% of baseline)")
print(f"  L2 contribution:     {l2_gain:.1f} MSE reduction ({l2_gain/l1_gain*100:.0f}% of L1 gain)")
print(f"  L3 contribution:     {l3_gain:.1f} MSE reduction ({l3_gain/l1_gain*100:.0f}% of L1 gain)")
print(f"  Old→New improvement: {(mse_old-mse_full_new):.1f} MSE ({((mse_old-mse_full_new)/mse_old*100):.0f}%)")
print("=" * 60)
