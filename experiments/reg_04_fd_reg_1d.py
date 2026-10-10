"""
E11-FD-REG-1D-Single: 1D sin(x) Regression — Single-Network Layer-Wise FD
===========================================================================
Uses a SINGLE network (no A/B split) with the same layer-wise residual
training logic as HiMAL but applied to all data.

Key improvements over v1:
  - Smart init: evenly-spaced particles + sharp kernels (h ~ 0.5-0.8)
  - ALL output dims learn target (wire-direct friendly)
  - Larger FD step (eps=0.01) for better gradient estimates

Purpose: Compare single-network vs HiMAL dual-network on smooth 1D regression.

Output: outputs/reg_04_fd_reg_1d.png + _report.txt
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physical_simulator import NoiseConfig
NoiseConfig.set_level("none")
from simulator.layer_nd import PhysicalRBFLayerND

# ── Config ──
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(OUT_DIR, "reg_04_fd_reg_1d_report.txt")
PLOT_FILE   = os.path.join(OUT_DIR, "reg_04_fd_reg_1d.png")

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

report = []
def L(s):
    print(s, flush=True)
    report.append(s)


# ═══════════════════════════════════════════════════════════════
#  Single-Network Layer-Wise FD Trainer
# ═══════════════════════════════════════════════════════════════

class SingleNetLayerwiseFD:
    """Single-network layer-wise FD trainer with smart initialization."""

    def __init__(
        self,
        input_dim: int = 1,
        layer_sizes: tuple = (25, 18, 12),
        output_dims: tuple = (4, 3, 2),
        init_range: tuple = (-7.0, 7.0),
        n_protos: int = 12,
        temperature: float = 0.1,
    ):
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dims = output_dims
        self.init_range = init_range
        self.n_protos = n_protos
        self.temperature = temperature
        self.layers: list = []
        self.proto_h: list = []
        self.proto_y: list = []

    @staticmethod
    def _softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @staticmethod
    def _create_prototypes(y, h, n_protos):
        y_flat = y.ravel() if y.ndim > 1 else y
        N = len(y_flat)
        n_protos = min(n_protos, N)
        idx = np.argsort(y_flat)
        bin_size = N // n_protos
        assign = np.zeros(N, dtype=np.int32)
        y_2d = y.reshape(N, -1) if y.ndim == 1 else y
        out_dim = y_2d.shape[1]
        y_protos = np.zeros((n_protos, out_dim), dtype=np.float32)
        h_protos = np.zeros((n_protos, h.shape[-1]), dtype=np.float32)
        for k in range(n_protos):
            start = k * bin_size
            end = (k + 1) * bin_size if k < n_protos - 1 else N
            bin_idx = idx[start:end]
            assign[bin_idx] = k
            y_protos[k] = y_2d[bin_idx].mean(axis=0)
            h_protos[k] = h[bin_idx].mean(axis=0)
        return y_protos, h_protos, assign

    def _proto_predict(self, h, y_protos, h_protos):
        K = len(y_protos)
        dist = np.linalg.norm(h[:, None, :] - h_protos[None, :, :], axis=2)
        weights = self._softmax(-dist / self.temperature, axis=1)
        return weights @ y_protos

    def _init_layer(self, n_particles, input_dim, n_vars):
        """Smart init: evenly-spaced centers with sharp kernels."""
        lo, hi = self.init_range
        dsize = hi - lo
        # Evenly spaced with small random jitter
        centers_base = np.linspace(lo, hi, n_particles).reshape(1, -1).astype(np.float32)
        jitter = np.random.uniform(-0.3 * dsize / n_particles, 0.3 * dsize / n_particles,
                                   (1, n_particles)).astype(np.float32)
        centers = centers_base + jitter
        # Sharp h: data_range / (4 * sqrt(n))
        h = (dsize / (4.0 * np.sqrt(max(n_particles, 1)))) * np.ones(n_particles, dtype=np.float32)
        W_gauss = np.random.uniform(-0.3, 0.3, (n_vars, n_particles)).astype(np.float32)
        W_act = np.zeros_like(W_gauss)
        return PhysicalRBFLayerND(centers=centers, h=h, W_gauss=W_gauss, W_act=W_act, use_pu=False)

    def _forward_layer(self, layer, X):
        out_g, out_a = layer.forward(X.astype(np.float32))
        return out_g + out_a

    def _mse_loss(self, layer, X, y_target):
        """ALL output dims learn the target (wire-direct friendly)."""
        h = self._forward_layer(layer, X)
        if h.shape[0] != y_target.shape[0]:
            h = h.T
        if y_target.shape[1] == 1 and h.shape[1] > 1:
            y_broad = np.tile(y_target, (1, h.shape[1]))
        else:
            y_broad = y_target[:, :min(h.shape[1], y_target.shape[1])]
        return float(np.mean((h[:, :y_broad.shape[1]] - y_broad) ** 2))

    def train_one_layer(self, X, y_target, n_particles, input_dim, n_vars,
                        epochs=200, lr=0.02, eps=0.01, verbose=True):
        layer = self._init_layer(n_particles, input_dim, n_vars)
        y_2d = y_target.reshape(-1, 1) if y_target.ndim == 1 else y_target

        for epoch in range(epochs):
            vec = layer.get_params_vector()
            grad = np.zeros_like(vec, dtype=np.float64)
            for j in range(len(vec)):
                orig = vec[j]
                vec[j] = orig + eps
                layer.set_params_from_vector(vec.astype(np.float32))
                Lp = self._mse_loss(layer, X, y_2d)
                vec[j] = orig - eps
                layer.set_params_from_vector(vec.astype(np.float32))
                Lm = self._mse_loss(layer, X, y_2d)
                vec[j] = orig
                layer.set_params_from_vector(vec.astype(np.float32))
                grad[j] = (Lp - Lm) / (2 * eps)
            vec = vec - lr * grad
            layer.set_params_from_vector(vec.astype(np.float32))

            if verbose and (epoch + 1) % 40 == 0:
                loss = self._mse_loss(layer, X, y_2d)
                print(f"  Layer epoch {epoch+1:3d}/{epochs} | loss={loss:.6f}")
        return layer

    def train(self, X, y, epochs_per_layer=100, lr=0.02, eps=0.01, verbose=True):
        N = len(X)
        y_orig = y.reshape(-1, 1) if y.ndim == 1 else y.copy()
        self.layers, self.proto_h, self.proto_y = [], [], []
        cur_X = X.copy()
        cur_dim = self.input_dim
        acc_pred = None

        for i in range(3):
            if verbose:
                print(f"\n===== Single-Net Layer {i+1} "
                      f"({self.layer_sizes[i]} particles, "
                      f"in={cur_dim}D, out={self.output_dims[i]}D) =====")
            n_p = self.layer_sizes[i]
            n_v = self.output_dims[i]
            if i == 0:
                cur_y = y_orig.copy()
            else:
                cur_y = y_orig - acc_pred[:, :y_orig.shape[1]]

            layer = self.train_one_layer(cur_X, cur_y, n_p, cur_dim, n_v,
                                         epochs=epochs_per_layer, lr=lr, eps=eps,
                                         verbose=verbose)
            self.layers.append(layer)
            h = self._forward_layer(layer, cur_X)
            if h.shape[0] != N:
                h = h.T
            y_2d = cur_y.reshape(-1, 1) if cur_y.ndim == 1 else cur_y
            y_proto, h_proto, _ = self._create_prototypes(y_2d, h, self.n_protos)
            self.proto_h.append(h_proto)
            self.proto_y.append(y_proto)

            pred_i = self._proto_predict(h, y_proto, h_proto)
            if acc_pred is None:
                acc_pred = pred_i.copy()
            else:
                da, db = acc_pred.shape[-1], pred_i.shape[-1]
                if da > db:
                    acc_pred[:, :db] += pred_i
                elif db > da:
                    tmp = np.zeros((N, db), dtype=np.float32)
                    tmp[:, :da] += acc_pred
                    tmp[:, :db] += pred_i
                    acc_pred = tmp
                else:
                    acc_pred += pred_i
            cur_X = h.copy()
            cur_dim = n_v
        return self

    def predict(self, X):
        h = X.copy()
        acc = None
        for i in range(len(self.layers)):
            h = self._forward_layer(self.layers[i], h)
            if h.shape[0] != X.shape[0]:
                h = h.T
            pi = self._proto_predict(h, self.proto_y[i], self.proto_h[i])
            if acc is None:
                acc = pi.copy()
            else:
                da, db = acc.shape[-1], pi.shape[-1]
                if da > db:
                    acc[:, :db] += pi
                elif db > da:
                    tmp = np.zeros((X.shape[0], db), dtype=np.float32)
                    tmp[:, :da] += acc
                    tmp[:, :db] += pi
                    acc = tmp
                else:
                    acc += pi
        return acc[:, :1]

    def predict_per_layer(self, X):
        results, h, acc = [], X.copy(), None
        for i in range(len(self.layers)):
            h = self._forward_layer(self.layers[i], h)
            if h.shape[0] != X.shape[0]:
                h = h.T
            pi = self._proto_predict(h, self.proto_y[i], self.proto_h[i])
            if acc is None:
                acc = pi.copy()
            else:
                da, db = acc.shape[-1], pi.shape[-1]
                if da > db:
                    acc[:, :db] += pi
                elif db > da:
                    tmp = np.zeros((X.shape[0], db), dtype=np.float32)
                    tmp[:, :da] += acc
                    tmp[:, :db] += pi
                    acc = tmp
                else:
                    acc += pi
            results.append(acc[:, :1].copy())
        return results


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

np.random.seed(42)

L("=" * 55)
L("  E11-FD-REG-1D-SINGLE: 1D sin(x) — Single Network Layer-Wise FD")
L("=" * 55)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Data: y = sin(x) + N(0, 0.05^2), x in [{-np.pi*2:.1f}, {2*np.pi:.1f}]")
L(f"  Method: Single network, all data → one network")
L("")

N = 200
x_range = (-2 * np.pi, 2 * np.pi)
X = np.random.uniform(*x_range, (N, 1)).astype(np.float32)
y_true = np.sin(X[:, 0]).astype(np.float32)
noise = np.random.normal(0, 0.05, N).astype(np.float32)
y = (y_true + noise).astype(np.float32)

var_y = float(np.var(y))
L(f"  N={N}, var(y)={var_y:.3f}")

layer_sizes = (25, 18, 12)
output_dims = (4, 3, 2)
init_range = (-7.0, 7.0)

trainer = SingleNetLayerwiseFD(
    input_dim=1, layer_sizes=layer_sizes, output_dims=output_dims,
    init_range=init_range, n_protos=12, temperature=0.1,
)

L(f"  Layer sizes: {layer_sizes}, output_dims: {output_dims}")
L(f"  init_range: {init_range}, h ~ {init_range[1]-init_range[0]:.0f}/(4*sqrt(N))")
L(f"  Per-parameter FD, all output dims learn target")
L("")

t0 = time.time()
trainer.train(X, y, epochs_per_layer=100, lr=0.02, eps=0.01, verbose=False)
train_t = time.time() - t0
n_params = sum(l.get_params_vector().size for l in trainer.layers)
L(f"  Training: {train_t:.0f}s, {n_params} params")

X_dense = np.linspace(x_range[0], x_range[1], 300).reshape(-1, 1).astype(np.float32)
y_dense_true = np.sin(X_dense[:, 0]).astype(np.float32)

def r2_score(pred, true):
    mse = float(np.mean((pred.ravel() - true.ravel()) ** 2))
    return 1.0 - mse / max(float(np.var(true)), 1e-10)

y_pred = trainer.predict(X_dense).ravel()
r2_dense = r2_score(y_pred, y_dense_true)

per_layer = trainer.predict_per_layer(X)
r2_1 = r2_score(per_layer[0], y)
r2_2 = r2_score(per_layer[1], y)
r2_3 = r2_score(per_layer[2], y)

mags = [float(np.mean(np.abs(trainer.proto_y[i]))) for i in range(3)]

L(f"\n  --- Results ---")
L(f"  R^2 by depth (train set):")
L(f"    L1-only:  {r2_1:.4f}")
L(f"    L1+L2:    {r2_2:.4f}")
L(f"    Full:     {r2_3:.4f}")
L(f"  R^2 on dense grid (vs clean sin): {r2_dense:.4f}")
L(f"  Proto_y magnitude:")
L(f"    L1: {mags[0]:.4f}   L2: {mags[1]:.4f}   L3: {mags[2]:.4f}")

if mags[0] > mags[1] + 0.01:
    L(f"  >>> Residual training OK")
else:
    L(f"  >>> Residual training NOT verified")

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(X[:, 0], y, c="lightgray", s=15, alpha=0.7, label="train data (noisy)")
ax.plot(X_dense[:, 0], y_dense_true, "k--", linewidth=1.5, label="y = sin(x)")
ax.plot(X_dense[:, 0], y_pred, "b-", linewidth=2, label=f"Single-Net predict (R^2={r2_dense:.4f})")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Single Network: sin(x) fitting")
ax.legend(fontsize=8); ax.set_xlim(x_range)

ax = axes[1]
ax.plot(X[:, 0], per_layer[0].ravel(), "o", markersize=3, alpha=0.6,
        label=f"L1 (R^2={r2_1:.3f})")
ax.plot(X[:, 0], (per_layer[1] - per_layer[0]).ravel(), "s", markersize=3, alpha=0.6,
        label="L2 residual")
ax.plot(X[:, 0], (per_layer[2] - per_layer[1]).ravel(), "^", markersize=3, alpha=0.6,
        label="L3 residual")
ax.set_xlabel("x"); ax.set_ylabel("contribution")
ax.set_title("Per-layer proto contributions")
ax.legend(fontsize=8); ax.set_xlim(x_range)

ax = axes[2]
colors = ["#2196F3", "#FF9800", "#4CAF50"]
for i, layer in enumerate(trainer.layers):
    cx = layer.centers[0, :]
    ax.plot(cx, np.full_like(cx, i), "o", color=colors[i], markersize=8, alpha=0.7,
            label=f"L{i+1} ({len(cx)}p)")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["L1", "L2", "L3"])
ax.set_xlabel("x")
ax.set_title("Particle center positions")
ax.legend(fontsize=8); ax.set_xlim(x_range)

fig.suptitle("E11-FD-REG-1D: Single-Network sin(x) Regression", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.close()
L(f"\n  Plot saved: {os.path.relpath(PLOT_FILE, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

L(f"\n{'='*55}")
if r2_dense > 0.8:
    L(f"  VERDICT: PASS (R^2={r2_dense:.4f} > 0.80)")
elif r2_dense > 0.5:
    L(f"  VERDICT: PARTIAL (R^2={r2_dense:.4f}, 0.5-0.8)")
else:
    L(f"  VERDICT: NEEDS WORK (R^2={r2_dense:.4f} < 0.50)")
L(f"{'='*55}")

L(f"\nSaved: {REPORT_FILE}")
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
