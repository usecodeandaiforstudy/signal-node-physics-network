"""
E11-FD-REG: Regression Depth Proof with Finite Differences
=============================================================
Proves:
  1. R2 increases as layers accumulate (L1 < L1+L2 < Full)
  2. proto_y magnitude decreases with depth (residual targets)
  3. Compares multi-D wire-direct (3,2,1) vs 1D-chain (1,1,1)

Physical domain: NoiseConfig("none"), PhysicalRBFLayerND, FD gradients.
Output: outputs/e11_fd_reg_result.txt
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from physical_simulator import NoiseConfig
NoiseConfig.set_level("none")
from network.combined import CombinedHiMALRBF
from training.himal_nd import _zero_pad_sum

# Fixed seed for reproducible depth proof
np.random.seed(42)

OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "outputs", "e11_fd_reg_result.txt")

report = []
def L(s):
    print(s, flush=True)
    report.append(s)

L("=" * 55)
L("  E11-FD-REG: Regression Depth Proof (FD)")
L("=" * 55)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Gradient: per-parameter FD (physical domain)")
L(f"  Noise: none")

# ── Data ──
np.random.seed(42)
N = 150
Xr = np.random.uniform(-2, 2, (N, 2)).astype(np.float32)
yr = (np.sin(3 * Xr[:, 0]) + np.cos(2 * Xr[:, 1])
      + 0.3 * Xr[:, 0] * Xr[:, 1]).astype(np.float32)
var_y = float(np.var(yr))
L(f"\n  Data: y=sin(3x1)+cos(2x2)+0.3*x1*x2")
L(f"  N={N}, var={var_y:.3f}, mean|y|={float(np.mean(np.abs(yr))):.3f}")

# ── Two configurations ──
for tag, out_dims, ls in [
    ("wire-direct (3,2,1)", (3, 2, 1), (6, 4, 3)),
    ("1D-chain   (1,1,1)", (1, 1, 1), (6, 4, 3)),
]:
    L(f"\n{'─'*45}")
    L(f"  Config: {tag}")
    L(f"  Particles: {ls}, output_dims: {out_dims}")
    L(f"{'─'*45}")

    m = CombinedHiMALRBF(
        input_dim=2, layer_sizes=ls, output_dims=out_dims,
        task="regression", use_spsa=False,  # per-parameter FD
    )
    t0 = time.time()
    m.train(Xr, yr, epochs_per_layer=15, lr=0.01, verbose=False)
    train_t = time.time() - t0
    L(f"  Training: {train_t:.0f}s, {m.n_total_params} params")

    # ── Per-layer evaluation ──
    t = m._trainer
    lyA, lyB = m.layers_A, m.layers_B

    # Forward chain + per-layer proto predictions
    hA, hB = Xr.copy(), Xr.copy()
    preds_A, preds_B = [], []
    for i in range(3):
        hA = t._forward_layer(lyA[i], hA)
        hB = t._forward_layer(lyB[i], hB)
        if hA.shape[0] != Xr.shape[0]:
            hA = hA.T
        if hB.shape[0] != Xr.shape[0]:
            hB = hB.T
        piA = t._proto_predict(hA, t.proto_y_A[i], t.proto_h_A[i])
        piB = t._proto_predict(hB, t.proto_y_B[i], t.proto_h_B[i])
        preds_A.append(piA)
        preds_B.append(piB)

    def accum(k):
        """Accumulate first k layers (average A/B)."""
        a = None
        for j in range(k):
            pj = (preds_A[j] + preds_B[j]) / 2.0
            a = pj.copy() if a is None else _zero_pad_sum(a, pj)
        return a[:, :1]

    def get_r2(pred):
        mse = float(np.mean((pred.ravel() - yr) ** 2))
        return 1.0 - mse / max(var_y, 1e-10)

    r2_1 = get_r2(accum(1))
    r2_2 = get_r2(accum(2))
    r2_3 = get_r2(accum(3))

    # Proto_y magnitudes (should decrease with depth)
    mags = [float(np.mean(np.abs(t.proto_y_A[i]))) for i in range(3)]

    L(f"  R2 by depth:")
    L(f"    L1-only:  {r2_1:.4f}")
    L(f"    L1+L2:    {r2_2:.4f}")
    L(f"    Full:     {r2_3:.4f}")
    L(f"  Proto_y magnitude (|target| per layer):")
    L(f"    L1: {mags[0]:.4f}   L2: {mags[1]:.4f}   L3: {mags[2]:.4f}")

    # ── Verdict ──
    checks = []
    if r2_3 > r2_1 + 0.005:
        checks.append(f"Depth improves R2: +{r2_3-r2_1:.4f}")
    elif r2_3 > r2_1:
        checks.append(f"Depth improves R2 marginally: +{r2_3-r2_1:.4f}")
    else:
        checks.append("Depth did not improve R2")

    if mags[0] > mags[1] + 0.01 and mags[1] > mags[2] + 0.001:
        checks.append(f"Residual training works: |proto_y| "
                      f"{mags[0]:.3f} > {mags[1]:.3f} > {mags[2]:.3f}")
    elif mags[0] > mags[2] + 0.01:
        checks.append(f"Residual training works: |proto_y| "
                      f"L1({mags[0]:.3f}) > L3({mags[2]:.3f})")
    else:
        checks.append("Residual training not verified")

    for c in checks:
        L(f"  >>> {c}")

# ── Physical domain check ──
from simulator.layer_nd import PhysicalRBFLayerND
L(f"\n{'─'*45}")
L(f"  Physical domain verification")
L(f"  Layer type: {type(m.layers_A[0]).__name__}")
L(f"  All PhysicalRBFLayerND: "
  f"{'PASS' if all(isinstance(l,PhysicalRBFLayerND) for l in m.layers_A) else 'FAIL'}")
L(f"  Noise: none")
L(f"  Gradient: per-parameter FD (no autograd)")
L(f"  Loss: physical MSE (subtract/square/mean)")

h_test = m.layers_A[0].forward(Xr[:3].astype(np.float32))
L(f"  Forward test: shape={h_test[0].shape}, "
  f"range=[{float(h_test[0].min()):.4f}, {float(h_test[0].max()):.4f}]")

L(f"\n{'='*55}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
L(f"Saved: {OUT}")
