"""
Ablation C -?Mismatch sensitivity: which non-ideality damages which parameters
============================================================================
Project: singal-node-physics-network

FALSIFIABLE PREDICTION
----------------------
The architecture stores two very different kinds of analog quantity:

    geometry : center coordinates + widths  -> a voltage / bias-current
               reference.  Real circuits damage these ADDITIVELY (gain and
               width mismatch, process spread).
    readout  : memristor conductances W     -> damaged MULTIPLICATIVELY
               (conductance drift, stuck devices, read noise).

Both effects already exist in the production simulator:
    differential_pair  : h_noisy = h * (1 + N(0, mismatch))      -> GEOMETRY
    gilbert_square     : d2     *= (1 + N(0, mismatch))          -> GEOMETRY
    current_mode_norm  : mirror copy-ratio error                  -> GEOMETRY
    memristor_mvm      : g_matrix = W * (1 + N(0, drift))         -> WEIGHTS

NoiseConfig only exposes four presets that move mismatch and drift TOGETHER
(components.py:490-495), so the two axes have never been separated.  This
script separates them and tests:

    H1: drift sensitivity increases with the fraction of learned capacity
        that sits in the readout matrix.
    H2: mismatch sensitivity increases with the fraction of learned capacity
        that sits in the geometry.

If H1/H2 hold, a real hardware design conclusion follows: the analog precision
budget should be spent on whichever quantity the layer actually learns with,
which is a decision the current codebase cannot inform.

PROTOCOL
--------
Models come from ablation A's cache (outputs/abl_cache) when present, otherwise
they are trained here with identical hyper-parameters.  Each model is evaluated
on a 5x5 grid of (mismatch, drift) with `--reps` noise realizations per cell.
No model selection is performed anywhere, so the reported R^2 is test R^2.

`learned_weight_fraction` = (trainable readout params) / (trainable params)
    geometry arm -> 0.0        weights arm -> 1.0        full arm -> arch share
This is the x-axis of the prediction: it varies WITHIN each architecture,
so the correlation is not degenerate over three points.

Output: outputs/abl_03_mismatch.txt + _plot.png
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abl_common import (                                   # noqa: E402
    ARCHS, OUT_DIR, SUBSTRATES, Report, apply_substrate_noise, build_model,
    cache_path, effective_budget, evaluate, header, inconclusive_text,
    learning_ok, load_model, make_regression_data, mean_std, r2, save_model,
    selftest, selftest_cross_substrate, set_diffusion_defaults, set_ideal,
    set_noise, train_model, try_plots,
)

ARMS = ("geometry", "weights", "full")
DEFAULTS = dict(seeds=3, epochs=60, budget=None, reps=3)

# The two damage axes have the same MEANING on both substrates -- one hits the
# geometry that was learned, one hits the readout that was learned -- but
# different physical knobs and units:
#
#   substrate   geometry axis                 weight axis
#   circuit     mismatch  (h gain, squarer)   drift   (memristor conductance)
#   diffusion   position_resolution [m]       strength_noise [relative]
#
# On the diffusion cell the injected strengths ARE the readout weights, so a
# relative error on them is the exact analogue of conductance drift.
AXES = {
    "circuit": dict(
        geom_key="mismatch", weight_key="drift",
        geom_grid=(0.0, 0.01, 0.02, 0.05, 0.10),
        weight_grid=(0.0, 0.0005, 0.001, 0.005, 0.020),
        geom_label="mismatch (gain/width, additive)",
        weight_label="memristor drift (multiplicative on W)",
    ),
    "diffusion": dict(
        geom_key="position_resolution", weight_key="strength_noise",
        geom_grid=(0.0, 2e-5, 5e-5, 1e-4, 3e-4),
        weight_grid=(0.0, 0.0005, 0.001, 0.005, 0.020),
        geom_label="positioner resolution [m] (additive, geometry)",
        weight_label="injected-strength noise (multiplicative on W)",
    ),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ablation C: mismatch sensitivity")
    p.add_argument("--task", default="4d", choices=["2d", "4d"])
    p.add_argument("--substrate", default="circuit", choices=list(SUBSTRATES),
                   help="physical medium; must match abl_01 for cache hits")
    p.add_argument("--archs", default="G,M,W")
    p.add_argument("--arms", default="geometry,weights,full")
    p.add_argument("--seeds", type=int, default=None, help="default 3")
    p.add_argument("--epochs", type=int, default=None, help="default 60")
    p.add_argument("--budget", type=int, default=None,
                   help="EXACT FD probes per epoch (must match abl_01 for cache "
                        "hits). Default: AUTO, same rule as abl_01")
    p.add_argument("--budget-mode", default="equal_probes",
                   choices=["equal_probes", "equal_sweeps"],
                   help="measurement protocol; must match abl_01 for cache hits")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--reps", type=int, default=None,
                   help="noise realizations averaged per grid cell (default 3)")
    p.add_argument("--probe-batch", type=int, default=None,
                   help="mini-batch per FD loss measurement (default 96)")
    p.add_argument("--quick", action="store_true",
                   help="1 seed, 40 epochs, budget 200, 1 rep")
    return p.parse_args(argv)


def train_or_load(arch, arm, seed, task, X_tr, y_tr, X_te, y_te,
                  arms, epochs, budget, lr, probe_batch, rep, substrate="circuit",
                  budget_mode="equal_probes"):
    """
    Reuse ablation A's cached model when available, else train it.

    The cache key contains the substrate, the effective per-arm probe budget and
    the epoch count, so a --quick smoke run, a different medium, or a different
    measurement protocol can never be silently reused as a real result.  Run
    abl_01 and abl_03 with the same --substrate --budget-mode --seeds --epochs
    for the cache to hit.  On a miss the model is retrained identically.
    """
    bd, _sizes = effective_budget(arch, X_tr, arms, budget,
                                  substrate=substrate, mode=budget_mode)
    b_eff = int(bd[arm])
    path = cache_path(arch, arm, "none", seed, task, epochs, b_eff, substrate)
    if os.path.exists(path):
        model = load_model(arch, arm, "none", seed, task, epochs, b_eff, X_tr,
                           substrate=substrate)
        if model is not None:
            rep(f"  [{arch}/{arm} seed={seed}] loaded from cache "
                f"(e={epochs}, budget={b_eff})")
            return model

    rep(f"  [{arch}/{arm} seed={seed}] cache miss -> training "
        f"(budget={b_eff} probes/epoch)")
    model = build_model(arch, X_tr, seed=seed, norm_mode="none",
                        substrate=substrate)
    train_model(model, X_tr, y_tr, X_te, y_te, arm=arm, epochs=epochs,
                lr=lr, eps_rel=1e-3, budget=b_eff,
                update_mode="sequential", seed=seed, verbose=False,
                probe_batch=probe_batch)
    save_model(model, arch, arm, "none", seed, task, epochs, b_eff,
               substrate=substrate,
               extra={"budget": b_eff, "epochs": epochs})
    return model


def main(argv=None):
    args = parse_args(argv)
    if args.seeds is None:
        args.seeds = 1 if args.quick else DEFAULTS["seeds"]
    if args.epochs is None:
        args.epochs = 40 if args.quick else DEFAULTS["epochs"]
    if args.reps is None:
        args.reps = 1 if args.quick else DEFAULTS["reps"]
    if args.probe_batch is None:
        args.probe_batch = 96
    if args.quick and args.budget is None:
        args.budget = 200
    if args.substrate == "diffusion":
        set_diffusion_defaults(level="none")

    ax = AXES[args.substrate]
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    sub_tag = "" if args.substrate == "circuit" else "_" + args.substrate
    rep = Report(f"abl_03_mismatch{sub_tag}.txt")
    for line in header("Ablation C - damage sensitivity of geometry vs weights",
                       f"substrate={args.substrate} task={args.task} "
                       f"archs={archs} arms={arms} seeds={args.seeds} "
                       f"reps={args.reps}"):
        rep(line)
    rep(f"  geometry axis: {ax['geom_label']}  {ax['geom_grid']}")
    rep(f"  weight   axis: {ax['weight_label']}  {ax['weight_grid']}")

    rep("\n--- Part 0: forward-chain self-test ---")
    selftest(verbose=True)
    selftest_cross_substrate(verbose=True)
    set_ideal()
    if args.quick:
        rep("")
        rep("  " + "!" * 60)
        rep("  ! SMOKE RUN (--quick): 1 seed, few epochs, 1 noise realization.")
        rep("  ! Correlations below are not meaningful at this scale.")
        rep("  " + "!" * 60)

    X_tr, y_tr, X_te, y_te = make_regression_data(args.task)
    rep(f"\n--- Data ({args.task}) --- train={X_tr.shape} test={X_te.shape}")

    # ---------------------------------------------------------------------------------------
    # Part 1 -?models
    # ---------------------------------------------------------------------------------------
    rep("\n--- Part 1: models ---")
    models = {}
    meta = {}
    for a in archs:
        for arm in arms:
            for s in range(args.seeds):
                m = train_or_load(a, arm, s, args.task, X_tr, y_tr, X_te, y_te,
                                  arms, args.epochs, args.budget, args.lr,
                                  args.probe_batch, rep, substrate=args.substrate,
                                  budget_mode=args.budget_mode)
                models[(a, arm, s)] = m

                b = m.breakdown()
                n_arm = int(m.index_set(arm).size)
                n_w = 0 if arm == "geometry" else int(m.index_set("weights").size)
                lwf = n_w / max(n_arm, 1)
                meta[(a, arm, s)] = {
                    "readout_share_arch": b["readout_share"],
                    "learned_weight_fraction": lwf,
                    "n_trainable": n_arm,
                }

    rep("")
    rep(f"  {'arch':<5} {'arm':<9} {'trainable':>10} {'arch readout share':>19} "
        f"{'learned weight frac':>20}")
    rep("  " + "-" * 68)
    for a in archs:
        for arm in arms:
            k = (a, arm, 0)
            rep(f"  {a:<5} {arm:<9} {meta[k]['n_trainable']:>10d} "
                f"{meta[k]['readout_share_arch']*100:>18.1f}% "
                f"{meta[k]['learned_weight_fraction']*100:>19.0f}%")

    # ---------------------------------------------------------------------------------------
    # Part 2 -?noise grid
    # ---------------------------------------------------------------------------------------
    rep("\n--- Part 2: two-axis damage grid, test R^2, mean of "
        f"{args.reps} noise realization(s) ---")
    t0 = time.time()

    geom_grid = ax["geom_grid"]
    weight_grid = ax["weight_grid"]
    curve_geo = {arm: {gv: [] for gv in geom_grid} for arm in arms}
    curve_wgt = {arm: {wv: [] for wv in weight_grid} for arm in arms}
    grid = {}

    def damage(model, gv, wv):
        apply_substrate_noise(model, args.substrate,
                              **{ax["geom_key"]: gv, ax["weight_key"]: wv})

    for a in archs:
        for arm in arms:
            for s in range(args.seeds):
                m = models[(a, arm, s)]
                for gv in geom_grid:
                    for wv in weight_grid:
                        damage(m, gv, wv)
                        mu, _sd = evaluate(m, X_te, y_te, reps=args.reps)
                        grid[(a, arm, s, gv, wv)] = mu
                        if wv == 0.0:
                            curve_geo[arm][gv].append(mu)
                        if gv == 0.0:
                            curve_wgt[arm][wv].append(mu)
                # restore the ideal medium before moving on
                damage(m, 0.0, 0.0)
    if args.substrate == "circuit":
        set_ideal()
    rep(f"  {len(grid)} grid cells evaluated in {time.time()-t0:.0f}s")

    MISMATCH_GRID, DRIFT_GRID = geom_grid, weight_grid
    curve_mis, curve_dft = curve_geo, curve_wgt

    # ---------------------------------------------------------------------------------------
    # Part 3 -?sensitivities
    # ---------------------------------------------------------------------------------------
    mis_max = max(MISMATCH_GRID)
    dft_max = max(DRIFT_GRID)

    rep("\n" + "=" * 66)
    rep("  Part 3a: sensitivity per arm (averaged over architectures and seeds)")
    rep("=" * 66)
    rep(f"  {'arm':<9} {'R2 clean':>9} {'R2@mis':>9} {'drop':>8} {'rel':>7} "
        f"{'R2@drift':>9} {'drop':>8} {'rel':>7}    "
        f"(mis={mis_max}, drift={dft_max})")
    rep("  " + "-" * 84)

    arm_stats = {}
    for arm in arms:
        clean = float(np.mean(curve_mis[arm][0.0])) if curve_mis[arm][0.0] else np.nan
        r_mis = float(np.mean(curve_mis[arm][mis_max]))
        r_dft = float(np.mean(curve_dft[arm][dft_max]))
        d_mis = clean - r_mis
        d_dft = clean - r_dft
        arm_stats[arm] = dict(
            clean=clean, r_mis=r_mis, r_dft=r_dft,
            drop_mis=d_mis, drop_dft=d_dft,
            rel_mis=d_mis / max(abs(clean), 1e-9),
            rel_dft=d_dft / max(abs(clean), 1e-9),
        )
        st = arm_stats[arm]
        rep(f"  {arm:<9} {st['clean']:>+9.4f} {st['r_mis']:>+9.4f} "
            f"{st['drop_mis']:>8.4f} {st['rel_mis']:>7.3f} "
            f"{st['r_dft']:>+9.4f} {st['drop_dft']:>8.4f} {st['rel_dft']:>7.3f}")

    # arm-level paired comparison, within architecture
    rep("\n  H-check at arm level (drift damage, weights arm vs geometry arm):")
    wins = 0
    total = 0
    for a in archs:
        g = float(np.mean([grid[(a, "geometry", s, 0.0, dft_max)]
                           for s in range(args.seeds)]))
        g0 = float(np.mean([grid[(a, "geometry", s, 0.0, 0.0)]
                            for s in range(args.seeds)]))
        w = float(np.mean([grid[(a, "weights", s, 0.0, dft_max)]
                           for s in range(args.seeds)]))
        w0 = float(np.mean([grid[(a, "weights", s, 0.0, 0.0)]
                            for s in range(args.seeds)]))
        dg, dw = g0 - g, w0 - w
        total += 1
        wins += int(dw > dg)
        rep(f"    {a}: drift drop  geometry={dg:+.4f}  weights={dw:+.4f}  "
            f"-> {'weights more sensitive' if dw > dg else 'geometry more sensitive'}")
    rep(f"    {wins}/{total} architectures agree with H1")

    # ---------------------------------------------------------------------------------------
    # Part 3b -?the actual prediction: correlation
    # ---------------------------------------------------------------------------------------
    rep("\n" + "=" * 66)
    rep("  Part 3b: does damage follow where the learned capacity sits?")
    rep("=" * 66)

    xs_lwf, xs_share = [], []
    ys_dft_abs, ys_dft_rel, ys_mis_abs, ys_mis_rel = [], [], [], []
    for a in archs:
        for arm in arms:
            for s in range(args.seeds):
                clean = grid[(a, arm, s, 0.0, 0.0)]
                r_mis = grid[(a, arm, s, mis_max, 0.0)]
                r_dft = grid[(a, arm, s, 0.0, dft_max)]
                xs_lwf.append(meta[(a, arm, s)]["learned_weight_fraction"])
                xs_share.append(meta[(a, arm, s)]["readout_share_arch"])
                ys_dft_abs.append(clean - r_dft)
                ys_mis_abs.append(clean - r_mis)
                ys_dft_rel.append((clean - r_dft) / max(abs(clean), 1e-9))
                ys_mis_rel.append((clean - r_mis) / max(abs(clean), 1e-9))

    def corr(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.std() < 1e-12 or y.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    rep(f"  n = {len(xs_lwf)} models "
        f"({len(archs)} archs x {len(arms)} arms x {args.seeds} seeds)")
    rep("")
    rep("  H1 (drift): damage should GROW with the learned weight fraction")
    rep(f"    corr(learned weight frac, drift drop abs) = "
        f"{corr(xs_lwf, ys_dft_abs):+.3f}")
    rep(f"    corr(learned weight frac, drift drop rel) = "
        f"{corr(xs_lwf, ys_dft_rel):+.3f}")
    rep("  H2 (mismatch): damage should FALL with the learned weight fraction")
    rep(f"    corr(learned weight frac, mismatch drop abs) = "
        f"{corr(xs_lwf, ys_mis_abs):+.3f}")
    rep(f"    corr(learned weight frac, mismatch drop rel) = "
        f"{corr(xs_lwf, ys_mis_rel):+.3f}")
    rep("")
    rep("  Secondary x-axis (architectural readout share; only "
        f"{len(set(np.round(xs_share, 4)))} distinct values, low power):")
    rep(f"    corr(arch readout share, drift drop rel)    = "
        f"{corr(xs_share, ys_dft_rel):+.3f}")
    rep(f"    corr(arch readout share, mismatch drop rel) = "
        f"{corr(xs_share, ys_mis_rel):+.3f}")

    r_h1 = corr(xs_lwf, ys_dft_rel)
    r_h2 = corr(xs_lwf, ys_mis_rel)

    rep("\n" + "=" * 66)
    rep("  VERDICT")
    rep("=" * 66)
    clean_vals = [grid[(a, arm, s, 0.0, 0.0)]
                  for a in archs for arm in arms for s in range(args.seeds)]
    if not learning_ok(clean_vals):
        rep("  " + "!" * 62)
        rep("  ! " + inconclusive_text(clean_vals))
        rep("  ! H1/H2 below cannot be tested on models that never fit the task.")
        rep("  " + "!" * 62)
    ok1 = np.isfinite(r_h1) and r_h1 > 0.3
    ok2 = np.isfinite(r_h2) and r_h2 < -0.3
    rep(f"  H1 (drift hits learned readout weights):     "
        f"{'SUPPORTED' if ok1 else 'NOT SUPPORTED'}  (r={r_h1:+.3f})")
    rep(f"  H2 (mismatch hits learned geometry):         "
        f"{'SUPPORTED' if ok2 else 'NOT SUPPORTED'}  (r={r_h2:+.3f})")
    if ok1 and ok2:
        rep("\n  Both hold. Consequence for hardware: the precision budget should go")
        rep("  to the quantity that layer actually learns with; drift-hardened")
        rep("  memristors matter for readout-dominated layers, matched voltage")
        rep("  references matter for geometry-dominated ones.")
    elif ok1 and not ok2:
        rep("\n  Only H1 holds: multiplicative drift on the readout is the dominant")
        rep("  non-ideality. Geometry mismatch matters less than expected, so the")
        rep("  'coordinates are cheap to store' argument is weaker than claimed.")
    elif ok2 and not ok1:
        rep("\n  Only H2 holds: geometry mismatch dominates. Note this also means the")
        rep("  '7.8e-7 digital-physical gap' figure is not informative about the")
        rep("  quantity that actually limits accuracy.")
    else:
        rep("\n  Neither prediction holds at this scale. Either the trained models are")
        rep("  too small for the effect to appear, or the effect is not there -?report")
        rep("  it as a negative result and scale up particles/epochs before concluding.")

    # Save the report BEFORE plotting: an early pipe close (e.g. piping through
    # Select-Object -First N) kills the process on its next print.
    _saved = rep.save()
    print(f"\nReport saved: {_saved}")

    # ---------------------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------------------
    plt = try_plots()
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        colors = {"geometry": "#2196F3", "weights": "#F44336", "full": "#4CAF50"}

        ax = axes[0]
        for arm in arms:
            ys = [np.mean(curve_mis[arm][mv]) for mv in MISMATCH_GRID]
            ax.plot(MISMATCH_GRID, ys, "-o", color=colors[arm], label=arm)
        ax.set_xlabel("mismatch (gain / width, additive)")
        ax.set_ylabel("test R2")
        ax.set_title("Mismatch sweep (drift=0)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

        ax = axes[1]
        for arm in arms:
            ys = [np.mean(curve_dft[arm][dv]) for dv in DRIFT_GRID]
            ax.plot(DRIFT_GRID, ys, "-o", color=colors[arm], label=arm)
        ax.set_xlabel("memristor drift (multiplicative on W)")
        ax.set_ylabel("test R2")
        ax.set_title("Drift sweep (mismatch=0)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

        ax = axes[2]
        ax.scatter(xs_lwf, ys_dft_rel, s=28, c="#F44336", label="drift (H1)")
        ax.scatter(xs_lwf, ys_mis_rel, s=28, c="#2196F3", marker="^",
                   label="mismatch (H2)")
        ax.set_xlabel("learned weight fraction")
        ax.set_ylabel("relative R2 drop")
        ax.set_title(f"H1 r={r_h1:+.2f}   H2 r={r_h2:+.2f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

        fig.suptitle("Ablation C: which non-ideality damages which parameters",
                     fontsize=11)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "abl_03_mismatch_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot: {p}")


if __name__ == "__main__":
    main()
