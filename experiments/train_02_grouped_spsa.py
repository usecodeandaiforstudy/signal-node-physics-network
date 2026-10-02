# Purpose: Compare grouped SPSA (K groups, approximate) vs per-parameter FD
# (exact) for training RBF networks. Measures speed/accuracy tradeoff when
# replacing exact FD with block-wise SPSA approximation for hardware efficiency.
"""
E14-PARALLEL-SPSA: Grouped SPSA vs Per-Parameter FD — Speed/Accuracy Tradeoff
==============================================================================
Direction A: compare per-parameter FD (exact gradient) with grouped SPSA
(approximate gradient) at increasing parallelism levels.

Method:
  - K=1: per-parameter FD baseline (2*n_params fwd/epoch, exact gradient)
  - K>1: particles split into K groups; within each group, simultaneous
         SPSA perturbation with n_avg averages (2 fwd/group/avg).
         Total = 2*K*n_avg fwd/epoch.

Question:
  How much accuracy do we lose by replacing exact per-param FD with
  K-group SPSA approximation? Is the speedup worth the degradation?
  Target: < 3% R^2 drop for K <= 8.

Task: 1D sin(x) regression, single-layer RBF, 40 particles, 160 params.

Output: outputs/e14_parallel_spsa.png + _report.txt
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

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(OUT_DIR, "e14_parallel_spsa_report.txt")
PLOT_FILE   = os.path.join(OUT_DIR, "e14_parallel_spsa.png")

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
L("  E14-PARALLEL-SPSA: Grouped SPSA vs Per-Param FD")
L("=" * 60)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Data: y = sin(x) + N(0, 0.05^2), N_train={N_TRAIN}")
L("")

# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════
N_PARTICLES = 40
INPUT_DIM = 1
OUTPUT_DIM = 1
INIT_LO, INIT_HI = -7.0, 7.0
K_VALUES = [1, 2, 4, 8]
EPOCHS = 200
LR = 0.015
EPS = 0.008
N_AVG = 3       # SPSA averages per group
MAX_FWD = 64000 # cap total forward passes at K=1 budget

L(f"  Architecture: single-layer RBF, {N_PARTICLES} particles ({160} params)")
L(f"  K={K_VALUES}, epochs={EPOCHS}, lr={LR}, eps={EPS}, n_avg={N_AVG}")
L(f"  Max fwd budget: {MAX_FWD} (K=1 baseline)")
L("")

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def make_layer(n_particles=N_PARTICLES):
    lo, hi = INIT_LO, INIT_HI
    dsize = hi - lo
    base = np.linspace(lo, hi, n_particles).reshape(1, -1).astype(np.float32)
    jitter = np.random.uniform(-0.3*dsize/n_particles, 0.3*dsize/n_particles,
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


def _build_group_masks(n_particles, dim, n_vars, n_params, K):
    """Boolean masks for K particle groups in the flat parameter vector."""
    D, N, V = dim, n_particles, n_vars
    masks = []
    particles_per_group = max(1, N // K)

    for g in range(K):
        start_p = g * particles_per_group
        end_p = min((g + 1) * particles_per_group, N) if g < K - 1 else N
        particles = list(range(start_p, end_p))
        if not particles:
            continue

        mask = np.zeros(n_params, dtype=bool)
        for p in particles:
            # centers is (D, N), .ravel() is C-order, so particle p's centre
            # coordinates are at d*N + p -- NOT p*D + d.  The earlier version
            # used p*D and therefore masked the same dimension of different
            # particles; results previously reported from this script must be
            # regenerated.
            for d in range(D):
                mask[d * N + p] = True
            mask[D * N + p] = True
            offset_wg = D * N + N
            for k in range(V):
                mask[offset_wg + k * N + p] = True
            offset_wa = offset_wg + V * N
            for k in range(V):
                mask[offset_wa + k * N + p] = True
        masks.append(mask)
    return masks


def train_with_budget(layer, X_in, y_in, K, lr, eps, n_avg,
                      max_fwd, track_every=5, verbose=False):
    """
    Train until max_fwd forward passes budget is exhausted.

    K=1: per-parameter sequential FD (2*n_params fwd/epoch).
    K>1: K-group SPSA (2*K*n_avg fwd/epoch).
    """
    N = layer.n_particles
    D = layer.dim
    V = layer.n_vars
    n_params = layer.get_params_vector().size
    masks = _build_group_masks(N, D, V, n_params, K) if K > 1 else None

    vec = layer.get_params_vector()
    best_vec = vec.copy()
    best_mse = float("inf")
    history = []
    total_fwd = 0
    epoch = 0

    while total_fwd < max_fwd:
        epoch += 1

        # ── K=1: per-parameter sequential FD ──
        if K == 1:
            if total_fwd + 2 * n_params > max_fwd:
                break
            grad = np.zeros(n_params, dtype=np.float64)
            for j in range(n_params):
                vec[j] += eps
                layer.set_params_from_vector(vec)
                L_plus = forward_mse(layer, X_in, y_in)
                vec[j] -= 2 * eps
                layer.set_params_from_vector(vec)
                L_minus = forward_mse(layer, X_in, y_in)
                vec[j] += eps
                grad[j] = (L_plus - L_minus) / (2 * eps)
            total_fwd += 2 * n_params
            vec = vec - lr * grad.astype(vec.dtype)
            layer.set_params_from_vector(vec)

        # ── K>1: grouped SPSA ──
        else:
            for g in range(K):
                mask = masks[g]
                mask_f = mask.astype(np.float32)
                if not mask.any():
                    continue

                grad_g = np.zeros(n_params, dtype=np.float64)
                for _ in range(n_avg):
                    if total_fwd + 2 > max_fwd:
                        break
                    delta = np.random.choice([-1.0, 1.0], size=n_params).astype(np.float32)
                    delta *= mask_f
                    if not np.any(delta):
                        continue

                    vec_plus = vec + eps * delta
                    layer.set_params_from_vector(vec_plus)
                    L_plus = forward_mse(layer, X_in, y_in)

                    vec_minus = vec - eps * delta
                    layer.set_params_from_vector(vec_minus)
                    L_minus = forward_mse(layer, X_in, y_in)

                    layer.set_params_from_vector(vec)
                    total_fwd += 2

                    # g_j = (L+ - L-) / (2*eps) * delta_j  (since delta_j = +/-1)
                    grad_g += (L_plus - L_minus) / (2.0 * eps) * delta

                grad_g /= max(n_avg, 1)
                vec = vec - lr * grad_g.astype(vec.dtype)
                layer.set_params_from_vector(vec)

        # ── Tracking ──
        if epoch % track_every == 0 or total_fwd >= max_fwd:
            mse = forward_mse(layer, X_in, y_in)
            out_g, out_a = layer.forward(X_in)
            r2_tr = r2_score((out_g + out_a).ravel(), y_in)
            out_gd, out_ad = layer.forward(X_dense)
            r2_de = r2_score((out_gd + out_ad).ravel(), y_dense_true)

            history.append({
                "epoch": epoch,
                "fwd_passes": total_fwd,
                "mse": mse,
                "r2_train": r2_tr,
                "r2_dense": r2_de,
            })

            if mse < best_mse:
                best_mse = mse
                best_vec = vec.copy()

            if verbose:
                mode = "FD" if K == 1 else f"SPSA-{K}g"
                L(f"    [{mode}] epoch {epoch:4d} | fwd={total_fwd:6d}/{max_fwd} | "
                  f"R2_train={r2_tr:.4f} | R2_dense={r2_de:.4f}")

    layer.set_params_from_vector(best_vec)
    return history


# ═══════════════════════════════════════════════════════════════
#  Run experiment
# ═══════════════════════════════════════════════════════════════
all_results = {}

for K in K_VALUES:
    method = "per-param FD (exact)" if K == 1 else \
             f"{K}-group SPSA (n_avg={N_AVG}, {2*K*N_AVG} fwd/epoch)"
    group_desc = "joint" if K == 1 else \
                 f"{K} groups x ~{N_PARTICLES//K} particles"
    L(f"─── K={K}: {method} ({group_desc}) ───")
    t0 = time.time()

    np.random.seed(123)
    layer = make_layer()

    history = train_with_budget(
        layer, X, y, K=K, lr=LR, eps=EPS, n_avg=N_AVG,
        max_fwd=MAX_FWD, track_every=20 if K > 1 else 20,
        verbose=True,
    )

    elapsed = time.time() - t0
    final_r2 = history[-1]["r2_dense"]
    total_fwd = history[-1]["fwd_passes"]

    # Find convergence point (95% of final R2)
    target_r2 = final_r2 * 0.95
    conv_fwd = None
    for h in history:
        if h["r2_dense"] >= target_r2:
            conv_fwd = h["fwd_passes"]
            break

    all_results[K] = {
        "history": history,
        "final_r2": final_r2,
        "total_fwd": total_fwd,
        "conv_fwd": conv_fwd,
        "elapsed": elapsed,
    }

    fwd_per_epoch = 2 * layer.get_params_vector().size if K == 1 else 2 * K * N_AVG
    L(f"  Final R2_dense: {final_r2:.4f} | fwd: {total_fwd} | "
      f"conv at: {conv_fwd} fwd | fwd/epoch: {fwd_per_epoch} | time: {elapsed:.1f}s")
    if K == 1:
        baseline_r2 = final_r2
        baseline_conv = conv_fwd
    L("")

# ═══════════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════════
L("═" * 60)
L("  RESULTS SUMMARY — Grouped SPSA vs Per-Param FD")
L("═" * 60)

L(f"\n  {'K':<4} {'R2_dense':<12} {'drop':<10} {'conv_fwd':<12} {'epochs':<8} {'speedup':<10} {'time':<8}")
L(f"  {'-'*60}")
for K in K_VALUES:
    r = all_results[K]
    drop_pct = (baseline_r2 - r["final_r2"]) / max(abs(baseline_r2), 1e-10) * 100
    n_epochs = len(r["history"])
    speedup = baseline_conv / max(r["conv_fwd"], 1) if r["conv_fwd"] and baseline_conv else float("nan")
    L(f"  {K:<4} {r['final_r2']:<12.4f} {drop_pct:+.2f}%{'':>4} "
      f"{r['conv_fwd'] or 'N/A':<12} {n_epochs:<8} "
      f"{speedup:.1f}x{'':>5} {r['elapsed']:.0f}s")

# R2 vs forward passes comparison at key milestones
L("")
L("  R2 vs forward passes (same fwd budget):")
# Find common fwd milestones
milestones = [500, 1000, 2000, 5000, 10000, 20000, 40000, 64000]
for m in milestones:
    r2_at_m = {}
    for K in K_VALUES:
        for h in all_results[K]["history"]:
            if h["fwd_passes"] >= m:
                r2_at_m[K] = h["r2_dense"]
                break
        if K not in r2_at_m:
            r2_at_m[K] = all_results[K]["final_r2"]
    r2s = list(r2_at_m.values())
    max_diff = max(r2s) - min(r2s) if r2s else 0
    L(f"    fwd={m:6d}: K=1:{r2_at_m.get(1,0):.4f} K=2:{r2_at_m.get(2,0):.4f} "
      f"K=4:{r2_at_m.get(4,0):.4f} K=8:{r2_at_m.get(8,0):.4f}  max_diff={max_diff:.4f}")

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
    L(f"  OVERALL: PASS — SPSA approximation preserves accuracy within {target_drop}%.")
    L(f"  Grouped SPSA is a viable replacement for per-param FD in RBF networks.")
else:
    L(f"  OVERALL: PARTIAL — some K exceed {target_drop}% threshold.")

# ═══════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
colors = {1: "#2196F3", 2: "#FF9800", 4: "#4CAF50", 8: "#F44336"}

# ── Plot 1: R2 vs forward passes ──
ax = axes[0, 0]
for K in K_VALUES:
    h = all_results[K]["history"]
    fwds = [x["fwd_passes"] for x in h]
    r2s = [x["r2_dense"] for x in h]
    label = f"K={K} (FD)" if K == 1 else f"K={K} (SPSA)"
    ax.plot(fwds, r2s, "-o", color=colors[K], markersize=3,
            linewidth=1.8, label=label)
ax.axhline(y=baseline_r2 * 0.95, color="gray", linestyle="--", linewidth=1,
           label=f"95% baseline ({baseline_r2*0.95:.3f})")
ax.set_xlabel("Forward passes")
ax.set_ylabel("R^2 (dense grid)")
ax.set_title("Convergence: R^2 vs Forward Passes\n(equal fwd budget)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

# ── Plot 2: Final R^2 bar chart ──
ax = axes[0, 1]
ks = K_VALUES
r2s_final = [all_results[k]["final_r2"] for k in ks]
bars = ax.bar(range(len(ks)), r2s_final, color=[colors[k] for k in ks], edgecolor="black")
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([f"K={k}\n(FD)" if k == 1 else f"K={k}\n(SPSA)" for k in ks])
ax.set_ylabel("Final R^2 (dense grid)")
ax.set_title("Final Accuracy\n(FD baseline vs grouped SPSA)")
for bar, r2, k in zip(bars, r2s_final, ks):
    drop = (baseline_r2 - r2) / max(abs(baseline_r2), 1e-10) * 100 if k > 1 else 0
    label = f"{r2:.4f}"
    if k > 1:
        label += f"\n({drop:+.2f}%)"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            label, ha="center", va="bottom", fontsize=9)
ax.axhline(y=baseline_r2 * 0.97, color="red", linestyle=":", linewidth=1.2,
           label=f"-3% ({baseline_r2*0.97:.4f})")
ax.legend(fontsize=7)
ax.set_ylim(0, max(r2s_final) * 1.15)
ax.grid(True, alpha=0.2, axis="y")

# ── Plot 3: Convergence speed ──
ax = axes[0, 2]
conv_fwds = [all_results[k]["conv_fwd"] or 0 for k in ks]
colors_list = [colors[k] for k in ks]
bars = ax.bar(range(len(ks)), conv_fwds, color=colors_list, edgecolor="black")
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([f"K={k}\n(FD)" if k==1 else f"K={k}\n(SPSA)" for k in ks])
ax.set_ylabel("Forward passes to 95% R^2")
ax.set_title("Convergence Speed\n(lower = faster convergence)")
for bar, cf, k in zip(bars, conv_fwds, ks):
    if cf > 0 and baseline_conv:
        speedup = baseline_conv / max(cf, 1)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(conv_fwds)*0.03,
                f"{cf}\n({speedup:.1f}x)", ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.2, axis="y")

# ── Plot 4: Accuracy degradation ──
ax = axes[0, 3]
ks_nonbase = [k for k in K_VALUES if k > 1]
drops = [(baseline_r2 - all_results[k]["final_r2"]) / max(abs(baseline_r2), 1e-10) * 100
         for k in ks_nonbase]
ax.bar(range(len(ks_nonbase)), drops, color=[colors[k] for k in ks_nonbase], edgecolor="black")
ax.set_xticks(range(len(ks_nonbase)))
ax.set_xticklabels([f"K={k}\n({2*k*N_AVG} fwd/epoch)" for k in ks_nonbase])
ax.set_ylabel("R^2 drop vs FD baseline (%)")
ax.set_title("SPSA Accuracy Cost")
ax.axhline(y=3.0, color="red", linestyle=":", linewidth=1.2, label="3% threshold")
for i, (d, k) in enumerate(zip(drops, ks_nonbase)):
    ax.text(i, d + 0.15, f"{d:.2f}%", ha="center", fontsize=10, fontweight="bold",
            color="green" if d <= 3 else "red")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2, axis="y")
top = max(max(drops) + 2, 5)
ax.set_ylim(bottom=min(min(drops) - 1, 0), top=top)

# ── Plots 5-8: Prediction curves ──
for idx, K in enumerate(K_VALUES):
    ax = axes[1, idx]
    np.random.seed(123)
    layer_k = make_layer()
    _ = train_with_budget(layer_k, X, y, K=K, lr=LR, eps=EPS, n_avg=N_AVG,
                          max_fwd=MAX_FWD, track_every=MAX_FWD+1, verbose=False)
    out_gd, out_ad = layer_k.forward(X_dense)
    y_pred_k = (out_gd + out_ad).ravel()
    r2_k = all_results[K]["final_r2"]

    ax.scatter(X[:, 0], y, c="lightgray", s=12, alpha=0.5, label="train (noisy)")
    ax.plot(X_dense[:, 0], y_dense_true, "k--", linewidth=1.5, label="y = sin(x)")
    ax.plot(X_dense[:, 0], y_pred_k, color=colors[K], linewidth=2,
            label=f"K={K} (R^2={r2_k:.4f})")
    cx = layer_k.centers[0, :]
    ax.scatter(cx, np.zeros_like(cx), c="green", marker="|", s=60, alpha=0.6)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    method_str = "per-param FD" if K == 1 else f"{K}-group SPSA"
    ax.set_title(f"K={K}: {method_str}\nR^2={r2_k:.4f}")
    ax.legend(fontsize=7)
    ax.set_xlim(x_range)
    ax.grid(True, alpha=0.2)

fig.suptitle("E14: Grouped SPSA vs Per-Parameter FD — Speed/Accuracy Tradeoff",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.close()
L(f"\n  Plot saved: {PLOT_FILE}")

L(f"\n  Report saved: {REPORT_FILE}")
L(f"{'='*60}")
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
