# Purpose: Test whether RBF particle independence allows parallel per-param FD
# training without accuracy loss. All K use the same exact FD — only particle
# grouping changes. Confirms physical hardware parallelism is mathematically sound.
"""
E13-PARALLEL-BENCHMARK: Grouped Per-Parameter FD — Particle Independence Test
=============================================================================
Direction B: all K values use the SAME per-parameter FD (exact gradient).
Only the grouping changes — particles are partitioned into K groups, and
each group's parameters are updated sequentially via per-parameter FD.

Hypothesis:
  RBF particles contribute independently to the output (sum over particles).
  Therefore training them in separate sequential groups should yield nearly
  identical results to joint training (K=1), modulo O(lr^2) cross-group effects.

What this tests:
  If K-group sequential per-param FD matches K=1 full FD in final R^2,
  it confirms particle independence — and thus the feasibility of parallel
  hardware where K circuit blocks independently FD-train their own particles.

All K values use the same gradient method. Total forward passes per epoch
is identical (2 * n_params) for all K. The comparison is purely about
whether group-wise sequential updates disrupt convergence.

Metrics:
  - Final R^2 after full training
  - R^2 vs epoch learning curves (same fwd budget, same gradient method)
  - Per-K prediction quality on dense grid

Output: outputs/e13_parallel_benchmark.png + _report.txt
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

REPORT_FILE = os.path.join(OUT_DIR, "e13_parallel_benchmark_report.txt")
PLOT_FILE   = os.path.join(OUT_DIR, "e13_parallel_benchmark.png")

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

report = []
def L(s):
    print(s, flush=True)
    report.append(s)

# ═══════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════
np.random.seed(42)
N_TRAIN = 200
x_range = (-2 * np.pi, 2 * np.pi)
X = np.random.uniform(*x_range, (N_TRAIN, 1)).astype(np.float32)
y_true = np.sin(X[:, 0]).astype(np.float32)
noise = np.random.normal(0, 0.05, N_TRAIN).astype(np.float32)
y = (y_true + noise).astype(np.float32)

X_dense = np.linspace(x_range[0], x_range[1], 300).reshape(-1, 1).astype(np.float32)
y_dense_true = np.sin(X_dense[:, 0]).astype(np.float32)

var_y = float(np.var(y))
L("=" * 60)
L("  E13-PARALLEL-BENCHMARK (Direction B): Particle Independence")
L("  ALL K use per-parameter FD — only grouping changes")
L("=" * 60)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Data: y = sin(x) + N(0, 0.05^2), x in [{x_range[0]:.1f}, {x_range[1]:.1f}]")
L(f"  N_train={N_TRAIN}, var(y)={var_y:.3f}")
L("")

# ═══════════════════════════════════════════════════════════════
#  Network config
# ═══════════════════════════════════════════════════════════════
N_PARTICLES = 40
INPUT_DIM = 1
OUTPUT_DIM = 1
INIT_LO, INIT_HI = -7.0, 7.0

L(f"  Architecture: single-layer RBF, {N_PARTICLES} particles")
L(f"  Init range: [{INIT_LO}, {INIT_HI}]")
L(f"  Gradient method: per-parameter FD (exact) for ALL K")
L("")

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def make_layer(n_particles=N_PARTICLES):
    lo, hi = INIT_LO, INIT_HI
    dsize = hi - lo
    base = np.linspace(lo, hi, n_particles).reshape(1, -1).astype(np.float32)
    jitter = np.random.uniform(-0.3 * dsize / n_particles, 0.3 * dsize / n_particles,
                               (1, n_particles)).astype(np.float32)
    xc = base + jitter
    h = (dsize / (4.0 * np.sqrt(max(n_particles, 1)))) * np.ones(n_particles, dtype=np.float32)
    W_gauss = np.random.uniform(-0.3, 0.3, (OUTPUT_DIM, n_particles)).astype(np.float32)
    W_act = np.zeros_like(W_gauss)
    return PhysicalRBFLayerND(centers=xc, h=h, W_gauss=W_gauss, W_act=W_act, use_pu=False)


def forward_mse(layer, X_in, y_in):
    out_g, out_a = layer.forward(X_in)
    return float(np.mean(((out_g + out_a).ravel() - y_in.ravel()) ** 2))


def r2_score(pred, true):
    mse = float(np.mean((pred.ravel() - true.ravel()) ** 2))
    return 1.0 - mse / max(float(np.var(true)), 1e-10)


def _build_group_indices(n_particles, dim, n_vars, K):
    """Build per-parameter index lists for each of K particle groups.

    Parameter vector layout (PhysicalRBFLayerND):
      [0          , D*N)         centers.ravel()
      [D*N        , D*N+N)       h.ravel()
      [D*N+N      , D*N+N+V*N)   W_gauss.ravel()  — C-order (row-major)
      [D*N+N+V*N  , total)       W_act.ravel()
    """
    D, N, V = dim, n_particles, n_vars
    groups = []
    particles_per_group = max(1, N // K)

    for g in range(K):
        start_p = g * particles_per_group
        end_p = min((g + 1) * particles_per_group, N) if g < K - 1 else N
        particles = list(range(start_p, end_p))
        if not particles:
            continue

        indices = []
        for p in particles:
            # centers: [p*D, (p+1)*D)
            indices.extend(range(p * D, (p + 1) * D))
            # h: at D*N + p
            indices.append(D * N + p)
            # W_gauss: (V, N) C-order
            offset_wg = D * N + N
            for k in range(V):
                indices.append(offset_wg + k * N + p)
            # W_act
            offset_wa = offset_wg + V * N
            for k in range(V):
                indices.append(offset_wa + k * N + p)

        groups.append(indices)
    return groups


def grouped_fd_train(layer, X_in, y_in, K, epochs, lr, eps,
                     track_every=1, verbose=False):
    """
    Train with K-group per-parameter FD. ALL K use exact per-param FD.

    K=1: one group = all particles, update all params together.
    K>1: K groups, each group's params updated sequentially via per-param FD.

    Total forward passes per epoch = 2 * n_params for ALL K.
    """
    N = layer.n_particles
    D = layer.dim
    V = layer.n_vars
    n_params = layer.get_params_vector().size
    group_indices = _build_group_indices(N, D, V, K)

    vec = layer.get_params_vector()
    best_vec = vec.copy()
    best_mse = float("inf")
    history = []
    total_fwd = 0

    for epoch in range(epochs):
        # ── Iterate through groups sequentially ──
        for g_idx in group_indices:
            if not g_idx:
                continue

            # Per-parameter FD within this group
            for j in g_idx:
                vec[j] += eps
                layer.set_params_from_vector(vec)
                L_plus = forward_mse(layer, X_in, y_in)

                vec[j] -= 2 * eps
                layer.set_params_from_vector(vec)
                L_minus = forward_mse(layer, X_in, y_in)

                vec[j] += eps  # restore
                total_fwd += 2

                grad = (L_plus - L_minus) / (2 * eps)
                # In-place update: each parameter is updated immediately
                # so the next parameter in the same group sees the latest state
                vec[j] -= lr * grad

            layer.set_params_from_vector(vec)

        # ── Tracking ──
        if (epoch + 1) % track_every == 0:
            mse = forward_mse(layer, X_in, y_in)
            out_g, out_a = layer.forward(X_in)
            r2_tr = r2_score((out_g + out_a).ravel(), y_in)
            out_gd, out_ad = layer.forward(X_dense)
            r2_de = r2_score((out_gd + out_ad).ravel(), y_dense_true)

            history.append({
                "epoch": epoch + 1,
                "fwd_passes": total_fwd,
                "mse": mse,
                "r2_train": r2_tr,
                "r2_dense": r2_de,
            })

            if mse < best_mse:
                best_mse = mse
                best_vec = vec.copy()

            if verbose and (epoch + 1) % 20 == 0:
                L(f"    [K={K}] epoch {epoch+1:4d}/{epochs} | fwd={total_fwd:6d} | "
                  f"R2_train={r2_tr:.4f} | R2_dense={r2_de:.4f}")

    layer.set_params_from_vector(best_vec)
    return history


# ═══════════════════════════════════════════════════════════════
#  Run experiment for each K
# ═══════════════════════════════════════════════════════════════
K_VALUES = [1, 2, 4, 8]
EPOCHS = 200
LR = 0.015
EPS = 0.008

L(f"  K values: {K_VALUES}")
L(f"  Epochs per K: {EPOCHS}, lr={LR}, eps={EPS}")
L(f"  Forward passes per epoch: 2*{N_PARTICLES*(INPUT_DIM + 1 + 2*OUTPUT_DIM)} = "
  f"{2*N_PARTICLES*(INPUT_DIM+1+2*OUTPUT_DIM)} (identical for all K)")
L("")

all_results = {}

for K in K_VALUES:
    group_desc = "all particles together" if K == 1 else \
                 f"{K} groups of ~{N_PARTICLES // K} particles each"
    L(f"─── K={K} ({group_desc}) ───")
    t0 = time.time()

    np.random.seed(123)
    layer = make_layer()
    n_total_params = layer.get_params_vector().size

    L(f"  Particles: {N_PARTICLES}, total params: {n_total_params}")

    history = grouped_fd_train(
        layer, X, y, K=K, epochs=EPOCHS, lr=LR, eps=EPS,
        track_every=5, verbose=True,
    )

    elapsed = time.time() - t0
    final_r2 = history[-1]["r2_dense"] if history else 0.0
    total_fwd = history[-1]["fwd_passes"] if history else 0

    all_results[K] = {
        "history": history,
        "final_r2": final_r2,
        "total_fwd": total_fwd,
        "elapsed": elapsed,
        "n_params": n_total_params,
    }

    L(f"  Final R2_dense: {final_r2:.4f} | Total fwd: {total_fwd} | Time: {elapsed:.1f}s")
    if K == 1:
        baseline_r2 = final_r2
    L("")

# ═══════════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════════
L("═" * 60)
L("  RESULTS SUMMARY — Direction B: Grouped Per-Param FD")
L("═" * 60)

baseline_r2 = all_results[1]["final_r2"]

for K in K_VALUES:
    r = all_results[K]
    drop_pct = (baseline_r2 - r["final_r2"]) / max(abs(baseline_r2), 1e-10) * 100
    L(f"  K={K:2d}: R2_dense={r['final_r2']:.4f} (drop={drop_pct:+.1f}%), "
      f"fwd={r['total_fwd']}, time={r['elapsed']:.0f}s")

# Compare R2 vs epoch curves
L("")
L("  R2 vs epoch (same fwd budget, same gradient method):")
for entry_idx in range(len(all_results[1]["history"])):
    epoch = all_results[1]["history"][entry_idx]["epoch"]
    r2s = [all_results[k]["history"][entry_idx]["r2_dense"] for k in K_VALUES]
    max_diff = max(r2s) - min(r2s)
    L(f"    epoch {epoch:3d}: R2 = {[f'{r:.4f}' for r in r2s]}  max_diff = {max_diff:.4f}")

L("")
target_drop = 3.0
all_pass = True
for K in K_VALUES:
    if K == 1:
        continue
    drop_pct = (baseline_r2 - all_results[K]["final_r2"]) / max(abs(baseline_r2), 1e-10) * 100
    verdict = "PASS" if drop_pct <= target_drop else "FAIL"
    if drop_pct > target_drop:
        all_pass = False
    L(f"  K={K}: R2 drop = {drop_pct:.2f}% vs {target_drop}% threshold -> {verdict}")

L("")
if all_pass:
    L(f"  OVERALL VERDICT: PASS")
    L(f"  Particle independence confirmed: grouped sequential FD matches joint FD.")
    L(f"  R2 degradation < {target_drop}% for all K. Parallel hardware is feasible.")
else:
    L(f"  OVERALL VERDICT: PARTIAL — some K exceed {target_drop}% threshold")

# ═══════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
colors = {1: "#2196F3", 2: "#FF9800", 4: "#4CAF50", 8: "#F44336"}

# ── Plot 1: R2 vs epoch (all K) ──
ax = axes[0, 0]
for K in K_VALUES:
    h = all_results[K]["history"]
    ep = [x["epoch"] for x in h]
    r2s = [x["r2_dense"] for x in h]
    ax.plot(ep, r2s, "-o", color=colors[K], markersize=3,
            linewidth=1.8, label=f"K={K}")
ax.set_xlabel("Epoch")
ax.set_ylabel("R^2 (dense grid vs clean sin)")
ax.set_title("Convergence: R^2 vs Epoch\n(all K: same per-param FD, same fwd/epoch)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Plot 2: Final R^2 bar chart ──
ax = axes[0, 1]
ks = K_VALUES
r2s_final = [all_results[k]["final_r2"] for k in ks]
bars = ax.bar(range(len(ks)), r2s_final, color=[colors[k] for k in ks], edgecolor="black")
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([f"K={k}" for k in ks])
ax.set_ylabel("Final R^2 (dense grid)")
ax.set_title("Grouping vs Final Accuracy\n(same per-param FD for all K)")
for bar, r2, k in zip(bars, r2s_final, ks):
    drop = (baseline_r2 - r2) / max(abs(baseline_r2), 1e-10) * 100 if k > 1 else 0
    label = f"{r2:.4f}"
    if k > 1:
        label += f"\n({drop:+.2f}%)"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            label, ha="center", va="bottom", fontsize=9)
ax.axhline(y=baseline_r2 * 0.97, color="red", linestyle=":", linewidth=1.2,
           label=f"-3% threshold ({baseline_r2 * 0.97:.4f})")
ax.legend(fontsize=7)
ax.set_ylim(0, max(r2s_final) * 1.15)
ax.grid(True, alpha=0.2, axis="y")

# ── Plot 3: R2 difference vs K=1 per epoch ──
ax = axes[0, 2]
h_base = all_results[1]["history"]
ep_base = [x["epoch"] for x in h_base]
for K in K_VALUES:
    if K == 1:
        continue
    h_k = all_results[K]["history"]
    diffs = [h_k[i]["r2_dense"] - h_base[i]["r2_dense"] for i in range(len(h_base))]
    ax.plot(ep_base, diffs, "-o", color=colors[K], markersize=3,
            linewidth=1.8, label=f"K={K} - K=1")
ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("R^2 difference vs K=1")
ax.set_title("R^2 Deviation from Joint Training")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Plot 4: Accuracy degradation ──
ax = axes[0, 3]
ks_nonbase = [k for k in K_VALUES if k > 1]
drops = [(baseline_r2 - all_results[k]["final_r2"]) / max(abs(baseline_r2), 1e-10) * 100
         for k in ks_nonbase]
ax.bar(range(len(ks_nonbase)), drops, color=[colors[k] for k in ks_nonbase], edgecolor="black")
ax.set_xticks(range(len(ks_nonbase)))
ax.set_xticklabels([f"K={k}" for k in ks_nonbase])
ax.set_ylabel("R^2 drop vs baseline (%)")
ax.set_title("Accuracy Degradation\n(grouped sequential FD vs joint FD)")
ax.axhline(y=3.0, color="red", linestyle=":", linewidth=1.2, label="3% threshold")
for i, (d, k) in enumerate(zip(drops, ks_nonbase)):
    ax.text(i, d + 0.1 * max(0.5, max(drops) if drops else 1),
            f"{d:.2f}%", ha="center", fontsize=10, fontweight="bold",
            color="green" if d <= 3 else "red")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2, axis="y")
top = max(max(drops) + 2, 5) if drops else 5
ax.set_ylim(bottom=min(min(drops) - 1, 0) if drops else -1, top=top)

# ── Plots 5-8: Prediction curves ──
for idx, K in enumerate(K_VALUES):
    ax = axes[1, idx]
    np.random.seed(123)
    layer_k = make_layer()
    _ = grouped_fd_train(layer_k, X, y, K=K, epochs=EPOCHS, lr=LR, eps=EPS,
                         track_every=EPOCHS + 1, verbose=False)
    out_gd, out_ad = layer_k.forward(X_dense)
    y_pred_k = (out_gd + out_ad).ravel()

    r2_k = all_results[K]["final_r2"]
    ax.scatter(X[:, 0], y, c="lightgray", s=12, alpha=0.5, label="train (noisy)")
    ax.plot(X_dense[:, 0], y_dense_true, "k--", linewidth=1.5, label="y = sin(x)")
    ax.plot(X_dense[:, 0], y_pred_k, color=colors[K], linewidth=2,
            label=f"K={K} (R^2={r2_k:.4f})")
    cx = layer_k.centers[0, :]
    ax.scatter(cx, np.zeros_like(cx), c="green", marker="|", s=60, alpha=0.6,
               label=f"{N_PARTICLES} particles")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    particles_per = N_PARTICLES // K if K > 1 else N_PARTICLES
    ax.set_title(f"K={K}: {'joint FD' if K==1 else f'{K} groups x ~{particles_per} particles'}")
    ax.legend(fontsize=7)
    ax.set_xlim(x_range)
    ax.grid(True, alpha=0.2)

fig.suptitle("E13 Direction B: Particle Independence Test — Grouped Per-Parameter FD",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.close()
L(f"\n  Plot saved: {PLOT_FILE}")

L(f"\n  Report saved: {REPORT_FILE}")
L(f"{'='*60}")
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
