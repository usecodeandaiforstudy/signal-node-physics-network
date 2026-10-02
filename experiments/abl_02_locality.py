"""
Ablation B -?Is particle independence real?  Locality vs the PU coupling
=======================================================================
Project: singal-node-physics-network

CLAIM UNDER TEST
----------------
README.md, line 16:
    "Proves parameter-group independence -?RBF particle output contributions
     are mutually independent, naturally supporting multi-channel hardware
     parallel training"

and training/himal_nd.py's own framing that per-particle updates are local.

The claim is asserted by experiments/train_01_grouped_fd.py, but that script
tests SEQUENTIAL grouped updates (Gauss-Seidel) and therefore cannot detect
coupling at all -?a sequential sweep is consistent with a fully coupled model.
Worse, train_01 runs with use_pu=False while the classification path runs with
PU on, and PU normalisation

        z_hat_i = z_i / sum_j z_j        (simulator/layer_nd.py:209-215)

makes every particle's activation an explicit function of every other
particle's parameters.  This script measures that instead of assuming it.

THREE MEASUREMENTS
------------------
1. ACTIVATION COUPLING (the literal README claim)
   C[i, j] = || d z_i / d theta_j ||, normalised per row by || d z_i / d theta_i ||
   -> "how much does particle i's activation move when particle j's parameters
       move, relative to when its own parameters move".
   Under a non-coupling normalisation the off-diagonal must be exactly 0;
   under PU it must be O(1).  Computed by central differences on the real
   physical forward chain, per normalisation mode.

2. PARALLEL EQUIVALENCE (the hardware claim)
   Train the same model twice from the same init:
       sequential (Gauss-Seidel): probe + update one coordinate at a time
       parallel   (Jacobi):       probe all coordinates from the frozen
                                  epoch-start state, then apply all at once
   Both spend exactly 2*|selected| forward passes per epoch.  "Parallel" is
   what K independent hardware channels actually do.  The gap
       Delta = R2_sequential - R2_parallel
   is the price of locality.  It is compared across normalisation modes.

3. NORMALISATION MODES
       "pu"    : z / sum(z)                     (production, coupling)
       "none"  : raw z                          (no coupling, unbounded scale)
       "fixed" : z / S, S frozen at init        (scale control, no coupling)
   The point of "fixed" is to test whether PU's apparent robustness is worth
   the locality it destroys.

Output: outputs/abl_02_locality.txt + _coupling.png + _parallel.png
"""

import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abl_common import (                                   # noqa: E402
    ARCHS, NORM_MODES, OUT_DIR, SUBSTRATES, Report, build_model,
    effective_budget, header, inconclusive_text, layer_forward, learning_ok,
    make_regression_data, mean_std, r2, selftest, selftest_cross_substrate,
    set_diffusion_defaults, set_ideal, train_model, try_plots,
)


# ---------------------------------------------------------------------------------------------
#  Measurement 1: activation coupling spectrum
# ---------------------------------------------------------------------------------------------

def particle_block_indices(layer, j, block):
    """
    Flat indices of particle j's parameter block inside ONE layer's vector.

    Layout (simulator/layer_nd.py:124-152):
        [ centers (D*N, C-order of a (D,N) array) | h (N) | W_gauss | W_act ]

    centers is (D, N) and .flatten() is C-order, so the element for dimension d
    of particle j sits at index d*N + j  --  NOT j*D + d.  Getting this wrong
    makes the "particle block" a scattered set of unrelated coordinates and
    destroys the entire measurement (note: experiments/train_01_grouped_fd.py
    uses the j*D+d convention, which does not match the (D,N) layout).
    """
    D, N, V = layer.dim, layer.n_particles, layer.n_vars
    n_c, n_h = D * N, N
    if block == "geometry":
        idx = [d * N + j for d in range(D)] + [n_c + j]
    elif block == "weights":
        base = n_c + n_h
        idx = [base + k * N + j for k in range(V)]
    else:
        raise ValueError(f"unknown block: {block}")
    return np.asarray(idx, dtype=np.int64)


def activation_coupling(layer, X, block="geometry", norm_mode="none",
                        scales=None, delta=0.05, seed=0, substrate="circuit"):
    """
    Returns (C, ratio) where
        C[i, j]     = RMS_batch | d z_i / d (unit direction in block j) |
        ratio[i, j] = C[i, j] / C[i, i]
    i.e. ratio[i, j] is the fraction of particle i's own-parameter sensitivity
    that particle j's parameters can reproduce.
    """
    rng = np.random.default_rng(seed)
    N = layer.n_particles
    C = np.zeros((N, N), dtype=np.float64)

    for j in range(N):
        idx = particle_block_indices(layer, j, block)
        direction = rng.normal(size=idx.size)
        direction /= (np.linalg.norm(direction) + 1e-12)

        lp = copy.deepcopy(layer)
        lm = copy.deepcopy(layer)
        vp = lp.get_params_vector()
        vm = lm.get_params_vector()
        vp[idx] += delta * direction
        vm[idx] -= delta * direction
        lp.set_params_from_vector(vp)
        lm.set_params_from_vector(vm)

        zp = layer_forward(lp, X, norm_mode, scales, substrate=substrate,
                           want_z=True)["z_g"]
        zm = layer_forward(lm, X, norm_mode, scales, substrate=substrate,
                           want_z=True)["z_g"]
        dz = (zp - zm) / (2.0 * delta)                 # (B, N)
        C[:, j] = np.sqrt(np.mean(dz ** 2, axis=0))

    diag = np.diag(C).copy()
    ratio = C / np.maximum(diag[:, None], 1e-12)
    return C, ratio


def offdiag_stats(ratio):
    N = ratio.shape[0]
    mask = ~np.eye(N, dtype=bool)
    vals = ratio[mask]
    return float(np.mean(vals)), float(np.max(vals))


# ---------------------------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------------------------

DEFAULTS = dict(seeds=3, epochs=60, budget=None)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ablation B: particle locality")
    p.add_argument("--task", default="4d", choices=["2d", "4d"])
    p.add_argument("--arch", default="M", help="architecture for measurements 2/3")
    p.add_argument("--substrate", default="circuit", choices=list(SUBSTRATES),
                   help="physical medium the whole ablation runs on")
    p.add_argument("--seeds", type=int, default=None, help="default 3")
    p.add_argument("--epochs", type=int, default=None, help="default 60")
    p.add_argument("--budget", type=int, default=None,
                   help="EXACT FD probes per epoch. Default: AUTO = 2 full "
                        "sweeps of the largest arm (see effective_budget)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eps-rel", type=float, default=1e-3)
    p.add_argument("--probe-batch", type=int, default=None,
                   help="mini-batch per FD loss measurement (default 96)")
    p.add_argument("--probe", type=int, default=32,
                   help="probe batch size for the coupling measurement")
    p.add_argument("--quick", action="store_true",
                   help="1 seed, 30 epochs, budget 150")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seeds is None:
        args.seeds = 1 if args.quick else DEFAULTS["seeds"]
    if args.epochs is None:
        args.epochs = 30 if args.quick else DEFAULTS["epochs"]
    if args.probe_batch is None:
        args.probe_batch = 96
    if args.quick and args.budget is None:
        args.budget = 150
    if args.substrate == "diffusion":
        set_diffusion_defaults(level="none")

    sub_tag = "" if args.substrate == "circuit" else "_" + args.substrate
    rep = Report(f"abl_02_locality{sub_tag}.txt")
    budget_txt = "AUTO (2 sweeps of the largest arm)" if args.budget is None \
        else str(args.budget)
    for line in header("Ablation B - Particle independence vs normalisation",
                       f"substrate={args.substrate} task={args.task} "
                       f"arch={args.arch} seeds={args.seeds} "
                       f"epochs={args.epochs} budget={budget_txt}"):
        rep(line)

    rep("\n--- Part 0: forward-chain self-test ---")
    selftest(verbose=True)
    selftest_cross_substrate(verbose=True)
    set_ideal()
    if args.quick:
        rep("")
        rep("  " + "!" * 60)
        rep("  ! SMOKE RUN (--quick): 1 seed, few epochs, small budget.")
        rep("  ! M1 (coupling) is exact regardless of the budget; M2/M3 are not.")
        rep("  " + "!" * 60)

    X_tr, y_tr, X_te, y_te = make_regression_data(args.task)
    probe = X_tr[:args.probe].copy()
    budget_by_arm, arm_sizes = effective_budget(args.arch, X_tr, ["full"],
                                                args.budget,
                                                substrate=args.substrate)
    budget_eff = int(budget_by_arm["full"])
    rep("")
    rep(f"  arch {args.arch} ({ARCHS[args.arch]['label']}): full arm = "
        f"{arm_sizes['full']} params -> {budget_eff} probes/epoch "
        f"({2*budget_eff} forward passes), "
        f"{budget_eff/max(arm_sizes['full'],1):.2f} sweeps/epoch")

    # ---------------------------------------------------------------------------------------
    # Measurement 1 -?activation coupling
    # ---------------------------------------------------------------------------------------
    rep("\n" + "=" * 66)
    rep("  M1: ACTIVATION COUPLING  (the literal README claim)")
    rep("=" * 66)
    rep("  ratio[i,j] = how much particle i's activation responds to particle j's")
    rep("  parameters, as a fraction of its response to its own parameters.")
    rep("  Non-coupling architectures MUST give exactly 0.00 off-diagonal.")
    rep("")
    rep(f"  {'norm':<7} {'layer':<6} {'N':>4} {'block':<9} "
        f"{'mean off-diag':>14} {'max off-diag':>13} {'verdict':>9}")
    rep("  " + "-" * 72)

    coupling_store = {}
    for norm in NORM_MODES:
        model = build_model(args.arch, X_tr, seed=0, norm_mode=norm,
                            substrate=args.substrate)
        for li, layer in enumerate(model.layers):
            scales = model.scales[li] if norm == "fixed" else None
            for block in ("geometry", "weights"):
                _C, ratio = activation_coupling(
                    layer, probe, block=block, norm_mode=norm,
                    scales=scales, delta=0.05, seed=7,
                    substrate=args.substrate,
                )
                mu, mx = offdiag_stats(ratio)
                verdict = "LOCAL" if mu < 1e-6 else "COUPLED"
                rep(f"  {norm:<7} L{li+1:<5} {layer.n_particles:>4d} {block:<9} "
                    f"{mu:>14.3e} {mx:>13.3e} {verdict:>9}")
                if block == "geometry":
                    coupling_store[(norm, li)] = ratio

    rep("")
    rep("  Read this table as follows:")
    rep("   * off-diagonal == 0 means particle i's activation depends ONLY on its")
    rep("     own parameters -> the README independence claim holds literally.")
    rep("   * off-diagonal > 0 means the normalisation mixes particles, so a")
    rep("     'parallel independent channel' story needs re-qualification.")
    rep("   * the weights block never enters z at all, so it is 0 by construction;")
    rep("     it is printed as a validity check on the measurement.")

    # ---------------------------------------------------------------------------------------
    # Measurements 2 & 3 -?parallel equivalence across norm modes
    # ---------------------------------------------------------------------------------------
    rep("\n" + "=" * 66)
    rep("  M2/M3: PARALLEL EQUIVALENCE  (the hardware training claim)")
    rep("=" * 66)
    rep("  sequential = Gauss-Seidel (probe+update one coordinate at a time)")
    rep("  parallel   = Jacobi (probe all from the frozen epoch-start state,")
    rep("               then apply every update at once) == K parallel channels")
    rep("  Both modes spend exactly 2*|selected| forward passes per epoch.")
    rep("")

    table = {}
    finals = {}
    t_start = time.time()
    for norm in NORM_MODES:
        for mode in ("sequential", "parallel"):
            vals = []
            for s in range(args.seeds):
                model = build_model(args.arch, X_tr, seed=s, norm_mode=norm,
                                    substrate=args.substrate)
                train_model(
                    model, X_tr, y_tr, X_te, y_te,
                    arm="full", epochs=args.epochs, lr=args.lr,
                    eps_rel=args.eps_rel, budget=budget_eff,
                    update_mode=mode, seed=s, verbose=False,
                    probe_batch=args.probe_batch,
                )
                vals.append(r2(y_te, model.forward(X_te)))
                finals[(norm, mode, s)] = model.get_params()
            table[(norm, mode)] = vals

    rep(f"  {'norm':<7} {'update':<11} {'R2 mean':>9} {'+-std':>8} "
        f"{'gap vs sequential':>18} {'||dtheta||/||theta||':>21}")
    rep("  " + "-" * 78)
    for norm in NORM_MODES:
        seq_mu = mean_std(table[(norm, "sequential")])[0]
        for mode in ("sequential", "parallel"):
            mu, sd = mean_std(table[(norm, mode)])
            gap = seq_mu - mu
            dists = []
            for s in range(args.seeds):
                ps = finals[(norm, "sequential", s)]
                pp = finals[(norm, "parallel", s)]
                dists.append(float(np.linalg.norm(ps - pp) /
                                   max(np.linalg.norm(ps), 1e-12)))
            rep(f"  {norm:<7} {mode:<11} {mu:>+9.4f} {sd:>8.4f} "
                f"{gap:>+18.4f} {np.mean(dists):>21.4f}")

    rep("\n" + "=" * 66)
    rep("  VERDICT")
    rep("=" * 66)

    gaps = {norm: mean_std(table[(norm, "sequential")])[0]
            - mean_std(table[(norm, "parallel")])[0] for norm in NORM_MODES}
    for norm in NORM_MODES:
        rep(f"  price of locality, norm={norm:<6}: Delta R2 = {gaps[norm]:+.4f}")

    all_r2 = [v for norm in NORM_MODES for mode in ("sequential", "parallel")
              for v in table[(norm, mode)]]
    if not learning_ok(all_r2):
        rep("")
        rep("  " + "!" * 62)
        rep("  ! M2/M3 " + inconclusive_text(all_r2))
        rep("  ! M1 above is unaffected: it is a property of the forward chain,")
        rep("  ! not of training, and is exact at any budget.")
        rep("  " + "!" * 62)
    elif "pu" in gaps and "none" in gaps:
        if gaps["none"] < gaps["pu"]:
            rep("\n  As predicted: removing the PU coupling REDUCES the penalty for")
            rep("  fully parallel (Jacobi) updates. PU buys bounded activations at")
            rep("  the cost of exactly the locality the hardware story relies on.")
        else:
            rep("\n  Prediction NOT confirmed: the PU coupling did not make parallel")
            rep("  updates worse than the uncoupled normalisation here. The")
            rep("  'independence' narrative is then not the bottleneck -?look at the")
            rep("  shared-residual term instead (every particle's gradient needs the")
            rep("  same batch residual, which is a global scalar but still global).")
    if "fixed" in gaps:
        rep(f"\n  norm=fixed (frozen scale, strictly non-coupling): Delta = "
            f"{gaps['fixed']:+.4f}  <- the configuration to use if you want both")
        rep("  bounded activations and a defensible parallel-channel claim.")

    rep(f"\n  Wall time: {time.time()-t_start:.0f}s")

    # Save the report BEFORE plotting: an early pipe close (e.g. piping through
    # Select-Object -First N) kills the process on its next print.
    path = rep.save()
    print(f"\nReport saved: {path}")

    # ---------------------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------------------
    plt = try_plots()
    if plt is not None:
        ncol = len(NORM_MODES)
        fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.4), squeeze=False)
        for ax, norm in zip(axes[0], NORM_MODES):
            ratio = coupling_store.get((norm, 0))
            if ratio is None:
                continue
            im = ax.imshow(np.clip(ratio, 0, 1), cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"norm={norm}  (layer 1)\n"
                         f"mean off-diag={offdiag_stats(ratio)[0]:.2e}", fontsize=9)
            ax.set_xlabel("particle j moving")
            ax.set_ylabel("particle i observed")
            plt.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle("Ablation B / M1: activation coupling  "
                     "ratio[i,j] = |dz_i/dtheta_j| / |dz_i/dtheta_i|", fontsize=11)
        plt.tight_layout()
        p1 = os.path.join(OUT_DIR, "abl_02_coupling.png")
        plt.savefig(p1, dpi=150)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
        x = np.arange(len(NORM_MODES))
        w = 0.36
        seq = [mean_std(table[(n, "sequential")])[0] for n in NORM_MODES]
        par = [mean_std(table[(n, "parallel")])[0] for n in NORM_MODES]
        seq_e = [mean_std(table[(n, "sequential")])[1] for n in NORM_MODES]
        par_e = [mean_std(table[(n, "parallel")])[1] for n in NORM_MODES]
        ax.bar(x - w / 2, seq, w, yerr=seq_e, capsize=4, label="sequential (Gauss-Seidel)",
               color="#2196F3", edgecolor="black")
        ax.bar(x + w / 2, par, w, yerr=par_e, capsize=4, label="parallel (Jacobi)",
               color="#F44336", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(NORM_MODES)
        ax.set_ylabel("test R2")
        ax.set_title(f"Ablation B / M2: price of parallel updates "
                     f"({args.substrate}, arch {args.arch}, budget {budget_eff})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
        plt.tight_layout()
        p2 = os.path.join(OUT_DIR, "abl_02_parallel.png")
        plt.savefig(p2, dpi=150)
        plt.close()
        rep(f"  Plots: {p1}\n         {p2}")

    path = rep.save()
    print(f"\nReport saved: {path}")


if __name__ == "__main__":
    main()
