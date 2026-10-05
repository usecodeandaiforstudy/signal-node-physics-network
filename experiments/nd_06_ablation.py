"""
V3 Ablation Study: HiMAL dual-network + RBF multi-layer effectiveness

Compares:
  A. Single vs Dual network: does A/B partition + alignment help?
  B. Single-layer vs Multi-layer: do L2/L3 contribute meaningfully?
  C. All trained with identical physical FD, same total particle budget
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.himal_nd import HiMALND, _zero_pad_sum
from simulator.layer_nd import PhysicalRBFLayerND
from physical_simulator import NoiseConfig

# ---------------------------------------------------------------------------
#  THIS SCRIPT'S RESULTS ARE NOW CITABLE.  It printed 22-36 lines and saved
#  NOTHING, so its numbers existed only in a terminal -- no artifact to compare
#  against, no way to notice a change.  This TEES stdout to a report instead of
#  touching the script's own code, so it saves exactly what it prints.
# ---------------------------------------------------------------------------
import atexit as _atexit, io as _io, os as _os, sys as _sys
_REPORT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                        "outputs", "nd_06_ablation_report.txt")
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

# ── Data: non-linear regression ──
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

def mse(pred, y):
    return float(np.mean((pred.ravel() - y) ** 2))

print("=" * 60)
print("  Ablation: Dual-Network + Multi-Layer Effectiveness")
print("=" * 60)
print(f"  y = sin(3x1)+cos(2x2)+0.5x1x2, baseline MSE={mean_baseline:.3f}")
print(f"  Train={len(X_tr)}, Test={len(X_te)}")

# ═══════════════════════════════════════════════
# Config: total particle budget = 8 per network
#   Dual+L1:   A=4, B=4  (layer 1 only)
#   Dual+L3:   A=(4,2,2), B=(4,2,2)  
#   Single+L3: 8 per layer, single network (no partition)
# ═══════════════════════════════════════════════

results = {}

# 1) Dual-network, single-layer (baseline for dual)
print("\n  [1] Dual-network, single-layer (4+4 particles)...")
t1 = HiMALND(layer_sizes=(4,), output_dims=(2,), input_dim=2, alignment_weight=0.3,
             n_protos=5, temperature=0.15)
t1._init_random_layer = lambda n, d_in, d_out: (
    t1._init_layer(n, d_in, d_out), t1._init_layer(n, d_in, d_out))

# Manually do single-layer dual training
d1 = t1.split_data(X_tr, y_tr, method="median_x")
ld1 = {"XA": d1["XA"], "yA": d1["yA"].reshape(-1,1),
       "XB": d1["XB"], "yB": d1["yB"].reshape(-1,1), "XT": d1["XT"]}
lA, lB = t1.train_one_layer(ld1, n_particles=4, input_dim=2, n_vars=2,
                             epochs=60, lr=0.012, eps=1e-3, verbose=False)

# Predict: just forward single layer, proto-weighted
t1.proto_h_A, t1.proto_y_A = [], []
t1.proto_h_B, t1.proto_y_B = [], []
y_t = d1["yA"].reshape(-1,1)
hA = t1._forward_layer(lA, d1["XA"])
if hA.shape[0] != len(d1["XA"]): hA = hA.T
yPA, hPA, _ = t1._create_prototypes(y_t, hA, 5)
t1.proto_y_A.append(yPA); t1.proto_h_A.append(hPA)
y_t2 = d1["yB"].reshape(-1,1)
hB = t1._forward_layer(lB, d1["XB"])
if hB.shape[0] != len(d1["XB"]): hB = hB.T
yPB, hPB, _ = t1._create_prototypes(y_t2, hB, 5)
t1.proto_y_B.append(yPB); t1.proto_h_B.append(hPB)

pred = t1.predict([lA], [lB], X_te)
results["dual_1layer"] = mse(pred, y_te)
print(f"      MSE={results['dual_1layer']:.3f}")

# 2) Dual-network, 3-layer
print("\n  [2] Dual-network, 3-layer (4,2,2 particles)...")
t2 = HiMALND(layer_sizes=(4, 2, 2), output_dims=(3, 2, 1), input_dim=2,
             alignment_weight=0.3, n_protos=5, temperature=0.15)
lA3, lB3, _ = t2.train(X_tr, y_tr, split_method="median_x",
                        epochs_per_layer=60, lr=0.012, eps=1e-3, verbose=False)
pred = t2.predict(lA3, lB3, X_te)
results["dual_3layer"] = mse(pred, y_te)
print(f"      MSE={results['dual_3layer']:.3f}")

# 3) Single-network, 3-layer (NO data partition, just one network)
print("\n  [3] Single-network, 3-layer (8,4,4) particles (no A/B split)...")

class SingleNetTrainer:
    """Single network, same RBF + FD training, no data partition."""
    def __init__(self, layer_sizes, output_dims, input_dim, init_range=(-3,3)):
        self.layer_sizes = layer_sizes
        self.output_dims = output_dims
        self.input_dim = input_dim
        self.init_range = init_range

    def _init(self, n, d_in, d_out):
        lo, hi = self.init_range
        xc = np.random.uniform(lo, hi, (d_in, n)).astype(np.float32)
        h = ((hi-lo)/np.sqrt(max(n,1))) * np.ones(n, dtype=np.float32)
        Wg = np.random.uniform(-0.1, 0.1, (d_out, n)).astype(np.float32)
        Wa = np.random.uniform(-0.1, 0.1, (d_out, n)).astype(np.float32)
        return PhysicalRBFLayerND(centers=xc, h=h, W_gauss=Wg, W_act=Wa, use_pu=True)

    def _forward(self, layer, X):
        out_g, out_a = layer.forward(X.astype(np.float32))
        return out_g + out_a

    def _loss(self, layer, X, y):
        h = self._forward(layer, X)
        if h.shape[0] != y.shape[0]: h = h.T
        d_out = min(h.shape[1], 1 if y.ndim==1 else y.shape[1])
        if y.ndim == 1: y2 = y.reshape(-1, 1)
        else: y2 = y
        return float(np.mean((h[:, :d_out] - y2[:, :d_out]) ** 2))

    def train_one(self, X, y, n, d_in, d_out, epochs=60, lr=0.012, eps=1e-3):
        layer = self._init(n, d_in, d_out)
        for epoch in range(epochs):
            vec = layer.get_params_vector()
            grad = np.zeros_like(vec)
            for j in range(len(vec)):
                vec[j] += eps
                layer.set_params_from_vector(vec)
                Lp = self._loss(layer, X, y)
                vec[j] -= 2*eps
                layer.set_params_from_vector(vec)
                Lm = self._loss(layer, X, y)
                vec[j] += eps
                grad[j] = (Lp - Lm) / (2*eps)
            vec = vec - lr * grad
            layer.set_params_from_vector(vec)
        return layer

    def train(self, X, y):
        layers = []
        cur_X = X.copy()
        cur_y = y.copy()
        cur_dim = self.input_dim
        acc = None
        for i in range(len(self.layer_sizes)):
            n, d_out = self.layer_sizes[i], self.output_dims[i]
            if i > 0 and acc is not None:
                cur_y = y.copy()
                if cur_y.ndim == 1: cur_y = cur_y.reshape(-1, 1)
                if acc.shape[1] > cur_y.shape[1]:
                    cur_y = cur_y - acc[:, :cur_y.shape[1]]
                else:
                    cur_y = cur_y - acc
            layer = self.train_one(cur_X, cur_y, n, cur_dim, d_out,
                                   epochs=60, lr=0.012, eps=1e-3)
            layers.append(layer)
            h = self._forward(layer, cur_X)
            if h.shape[0] != cur_X.shape[0]: h = h.T
            acc = h.copy() if acc is None else _zero_pad_sum(acc, h)
            cur_X = h.copy()
            cur_dim = d_out
        return layers

    def predict(self, layers, X):
        h = X.copy()
        acc = None
        for layer in layers:
            h_out = self._forward(layer, h)
            if h_out.shape[0] != h.shape[0]: h_out = h_out.T
            acc = h_out.copy() if acc is None else _zero_pad_sum(acc, h_out)
            h = h_out
        return acc[:, 0]  # first output dim

st = SingleNetTrainer(layer_sizes=(8, 4, 4), output_dims=(4, 3, 2), input_dim=2)
layers_single = st.train(X_tr, y_tr)
pred = st.predict(layers_single, X_te)
results["single_3layer"] = mse(pred, y_te)
print(f"      MSE={results['single_3layer']:.3f}")

# 4) Single-network, single-layer (same total budget: 8 particles, 1 layer)
print("\n  [4] Single-network, single-layer (8 particles)...")
st2 = SingleNetTrainer(layer_sizes=(8,), output_dims=(2,), input_dim=2)
l_s = st2.train_one(X_tr, y_tr, n=8, d_in=2, d_out=2, epochs=60, lr=0.012, eps=1e-3)
pred = st2.predict([l_s], X_te)
results["single_1layer"] = mse(pred, y_te)
print(f"      MSE={results['single_1layer']:.3f}")

# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Results: Dual-network vs Multi-layer Ablation")
print("=" * 60)
print(f"  Baseline (mean predict):  {mean_baseline:.3f}")
print()

# Multi-layer effect (fixed architecture)
single_1 = results["single_1layer"]
single_3 = results["single_3layer"]
dual_1 = results["dual_1layer"]
dual_3 = results["dual_3layer"]

print(f"  Single-net 1-layer:  {single_1:.3f}")
print(f"  Single-net 3-layer:  {single_3:.3f}  (multi-layer gain: {single_1-single_3:+.3f})")
print()
print(f"  Dual-net 1-layer:    {dual_1:.3f}")
print(f"  Dual-net 3-layer:    {dual_3:.3f}  (multi-layer gain: {dual_1-dual_3:+.3f})")
print()

# Dual-network effect (fixed depth)
print(f"  Single 1-layer:      {single_1:.3f}")
print(f"  Dual   1-layer:      {dual_1:.3f}    (dual-net gain: {single_1-dual_1:+.3f})")
print()
print(f"  Single 3-layer:      {single_3:.3f}")
print(f"  Dual   3-layer:      {dual_3:.3f}    (dual-net gain: {single_3-dual_3:+.3f})")
print()

# Overall best
best = min(results.values())
print(f"  Best overall: {best:.3f} ({[k for k,v in results.items() if v==best][0]})")
print(f"  Worst: {max(results.values()):.3f} (baseline: {mean_baseline:.3f})")
print("=" * 60)
