"""
Ablation A -?Where does the capacity live?  Geometry vs readout matrix
=====================================================================
Project: singal-node-physics-network

CLAIM UNDER TEST
----------------
The architecture's thesis is "nodes replace matrices": the network's capacity
is carried by the movable particle coordinates (centers + widths), with the
memristor readout as a mere linear labelling of the resulting field.

The production trainer cannot test this, because HiMALND optimises centers,
widths and readout weights jointly (training/himal_nd.py:679-692).  There is
no "train the nodes only" path anywhere in the repository.

PROTOCOL
--------
Fix the architecture, then train three disjoint parameter subsets:

    geometry : centers + widths          (the "node coordinates")
    weights  : W_gauss                   (the memristor readout)
    full     : both                      (reference)

and compare them across three architectures chosen so that the *parameter
share* of geometry varies while the PARTICLE COUNT IS HELD FIXED:

    G : out_dims=(2, 2, 1)   contracting  -> geometry-dominated
    M : out_dims=(4, 4, 2)   balanced
    W : out_dims=(16, 8, 4)  expanding    -> matrix-dominated

Key control -?EQUAL ANALOG MEASUREMENT BUDGET.  Every arm probes at most
`--budget` coordinates per epoch (2 forward passes each), so an arm with 4x
the parameters does NOT get 4x the measurements.  This is the only comparison
that means anything for hardware, and it is the difference between "more
parameters help" and "more stored analog quantities help".

REPORTED
--------
  * test R^2 per (arch, arm): mean +/- std over seeds
  * `retained` = R^2_arm / R^2_full  (fraction of the full model's skill kept)
  * the exact geometry/readout parameter split per arch
  * correlation between geometry share and the geometry arm's retained skill

CAVEATS (stated up front, not hidden)
------------------------------------
  * Random-uniform centers with data-scaled widths are used instead of k-means
    init, so the ablation does not silently depend on clustering quality.
    k-means init favours the geometry arm and is a known improvement
    (reg_01/reg_06; Wurzberger & Schwenker, Entropy 2024).
  * Adam is used on the finite-difference gradients.  With plain SGD a single
    global lr gives centers (scale ~2), widths (~1) and weights (~0.3) very
    different effective step sizes, which would confound the arms.
  * Reported R^2 is test R^2.  There is no early stopping and no model
    selection anywhere in this script, so this is not test-set leakage.

Output: outputs/abl_01_capacity_split.txt + _plot.png
Also caches trained parameter vectors for abl_03 (~/outputs/abl_cache).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abl_common import (                                   # noqa: E402
    ARCHS, OUT_DIR, SUBSTRATES, Report, build_model, cache_path, curve_at,
    effective_budget, header, inconclusive_text, learning_ok,
    make_regression_data, mean_std, merge_curves, nanmean_safe, r2,
    readout_calibration_cost, save_model, selftest, selftest_cross_substrate,
    set_diffusion_defaults, set_ideal, train_model, try_plots,
)

ARMS = ("geometry", "weights", "hybrid", "full")
DEFAULTS = dict(seeds=3, epochs=60, budget=None, probe_batch=96)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ablation A: capacity split")
    p.add_argument("--task", default="4d", choices=["2d", "4d"])
    p.add_argument("--substrate", default="circuit", choices=list(SUBSTRATES),
                   help="physical medium the whole ablation runs on")
    p.add_argument("--archs", default="G,M,W")
    p.add_argument("--arms", default="geometry,weights,hybrid,full",
                   help="hybrid = probe the geometry, CALIBRATE the source "
                        "strengths in closed form (one-time offline cost, "
                        "reported separately)")
    p.add_argument("--seeds", type=int, default=None, help="default 3")
    p.add_argument("--epochs", type=int, default=None, help="default 60")
    p.add_argument("--budget", type=int, default=None,
                   help="EXACT FD probes per epoch, identical for every arm. "
                        "Default: AUTO = 2 full sweeps of each architecture's "
                        "largest arm (arms cycle when this exceeds their size)")
    p.add_argument("--budget-mode", default="equal_probes",
                   choices=["equal_probes", "equal_sweeps"],
                   help="equal_probes: same measurement cost per arm per epoch, "
                        "which favours the SMALLER arm. equal_sweeps: each arm "
                        "completes one full sweep of its own parameters per "
                        "epoch, i.e. equal coverage at unequal cost. Run BOTH: "
                        "an arm that wins one and loses the other is merely more "
                        "efficient per measurement.")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eps-rel", type=float, default=1e-3)
    p.add_argument("--probe-batch", type=int, default=None,
                   help="mini-batch used for each FD loss measurement "
                        "(default 96, 0 = full batch)")
    p.add_argument("--quick", action="store_true",
                   help="1 seed, 40 epochs, budget 200 (smoke run)")
    p.add_argument("--attribute", dest="attribute", action="store_true",
                   default=True,
                   help="run Phase 2 (freeze-and-finetune attribution)")
    p.add_argument("--no-attribute", dest="attribute", action="store_false",
                   help="skip Phase 2")
    p.add_argument("--no-curve", dest="curve", action="store_false", default=True,
                   help="skip the R^2(measurements) trajectory recording")
    p.add_argument("--milestones", default="3000,10000,25000,50000,100000",
                   help="measurement counts reported in the scaling table")
    p.add_argument("--tag", default=None,
                   help="suffix for the report/plot filenames, so running a "
                        "SUBSET of architectures (e.g. a share-axis sweep) cannot "
                        "overwrite the full report")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # None == "not specified": resolved here so --quick can fill in small values
    # without ever overriding an explicit argument (even one equal to a default)
    if args.seeds is None:
        args.seeds = 1 if args.quick else DEFAULTS["seeds"]
    if args.epochs is None:
        args.epochs = 40 if args.quick else DEFAULTS["epochs"]
    if args.probe_batch is None:
        args.probe_batch = DEFAULTS["probe_batch"]
    if args.quick and args.budget is None:
        args.budget = 200
    if args.substrate == "diffusion":
        set_diffusion_defaults(level="none")

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in archs:
        if a not in ARCHS:
            raise SystemExit(f"unknown arch '{a}', choose from {list(ARCHS)}")

    mode_tag = "" if args.budget_mode == "equal_probes" else "_" + args.budget_mode
    sub_tag = "" if args.substrate == "circuit" else "_" + args.substrate
    tag = f"_{args.tag}" if args.tag else ""
    rep = Report(f"abl_01_capacity_split{sub_tag}{mode_tag}{tag}.txt")
    budget_txt = "AUTO (2 sweeps of the largest arm)" if args.budget is None \
        else str(args.budget)
    for line in header("Ablation A - Capacity split: geometry vs readout matrix",
                       f"substrate={args.substrate} task={args.task} archs={archs} "
                       f"arms={arms} seeds={args.seeds} "
                       f"epochs={args.epochs} budget={budget_txt}"):
        rep(line)

    rep("\n--- Part 0: forward-chain self-test ---")
    selftest(verbose=True)
    selftest_cross_substrate(verbose=True)
    rep("  both substrates compute the same function from the same parameters,")
    rep("  so the numbers below are produced by real physics on either medium")
    set_ideal()
    if args.quick:
        rep("")
        rep("  " + "!" * 60)
        rep("  ! SMOKE RUN (--quick): 1 seed, few epochs, small budget.")
        rep("  ! These numbers only prove the pipeline runs. They are NOT results.")
        rep("  " + "!" * 60)

    X_tr, y_tr, X_te, y_te = make_regression_data(args.task)
    rep(f"\n--- Data ({args.task}) ---")
    rep(f"  train={X_tr.shape} test={X_te.shape}  y std={float(np.std(y_tr)):.3f}")

    # ---- architecture census + equal-budget derivation ----
    rep("\n--- Architecture census (single channel, scalar residual readout) ---")
    census = {}
    budgets = {}
    for a in archs:
        m = build_model(a, X_tr, seed=0, substrate=args.substrate)
        b = m.breakdown()
        census[a] = b
        bd, sizes = effective_budget(a, X_tr, arms, args.budget,
                                     substrate=args.substrate,
                                     mode=args.budget_mode)
        budgets[a] = bd
        sweeps = {k: round(bd[k] / max(v, 1), 2) for k, v in sizes.items()}
        rep(f"  {a} ({ARCHS[a]['label']:<19}) {m.describe()}")
        rep(f"      arm sizes {sizes} -> probes/epoch {bd}, sweeps/epoch {sweeps}")
    rep("  NOTE: the number of epochs is identical everywhere; the protocol is")
    rep(f"        {args.budget_mode}. equal_probes gives equal measurement cost")
    rep("        and therefore MORE sweeps to the smaller arm; equal_sweeps gives")
    rep("        equal coverage at unequal cost. An arm that wins one and loses")
    rep("        the other is more efficient per measurement, not a better")
    rep("        representation -- which is why both protocols should be run.")

    # ---- training ----
    results = {a: {arm: [] for arm in arms} for a in archs}
    history = {a: {} for a in archs}
    curves = {a: {arm: [] for arm in arms} for a in archs}
    trained = {}
    t_start = time.time()

    for a in archs:
        for arm in arms:
            r2s = []
            for s in range(args.seeds):
                t0 = time.time()
                model = build_model(a, X_tr, seed=s, substrate=args.substrate)
                hist = train_model(
                    model, X_tr, y_tr, X_te, y_te,
                    arm=arm, epochs=args.epochs, lr=args.lr,
                    eps_rel=args.eps_rel, budget=budgets[a][arm],
                    update_mode="sequential", seed=s, verbose=False,
                    probe_batch=args.probe_batch, substrate=args.substrate,
                    eval_every=(1 if args.curve else 0),
                    eval_X=X_te, eval_y=y_te,
                    # honest accounting: hybrid_once pays for its calibration on
                    # the device ONCE; hybrid re-calibrates every epoch and is
                    # therefore only defensible as twin-assisted (zero physical
                    # measurement cost), which is how it is counted here.
                    calib_physical=(arm == "hybrid_once"),
                )
                te_r2 = r2(y_te, model.forward(X_te))
                r2s.append(te_r2)
                history[a][(arm, s)] = hist
                curves[a][arm].append(hist["curve"])
                trained[(a, arm, s)] = model.get_params()
                save_model(model, a, arm, "none", s, args.task,
                           args.epochs, budgets[a][arm], substrate=args.substrate,
                           extra={"test_r2": te_r2, "fwd": hist["fwd"],
                                  "budget": budgets[a][arm],
                                  "epochs": args.epochs})
                rep(f"  [{a}/{arm} seed={s}] test R2={te_r2:+.4f} "
                    f"probes/ep={budgets[a][arm]} fwd={hist['fwd']} "
                    f"calib={hist.get('calib_cost', 0)} "
                    f"({time.time()-t0:.1f}s)")
            results[a][arm] = r2s

    all_r2 = [v for a in archs for arm in arms for v in results[a][arm]]
    learned = learning_ok(all_r2)
    rep("")
    if not learned:
        rep("  " + "!" * 62)
        rep("  ! " + inconclusive_text(all_r2))
        rep("  " + "!" * 62)

    # ---- summary ----
    rep("\n" + "=" * 66)
    rep("  RESULTS -?test R^2 at equal analog measurement budget")
    rep("=" * 66)
    rep(f"  {'arch':<5} {'capacity split':<26} {'arm':<9} {'params':>7} "
        f"{'geom%':>6} {'probes/ep':>9} {'sweeps':>7} {'R2 mean':>9} {'+-std':>7} "
        f"{'retained':>9}")
    rep("  " + "-" * 112)

    summary = {}
    for a in archs:
        b = census[a]
        full_mu = mean_std(results[a]["full"])[0] if "full" in results[a] else np.nan
        for arm in arms:
            mu, sd = mean_std(results[a][arm])
            # the ratio is only meaningful when the reference model actually fits
            valid = abs(full_mu) >= 0.05
            retained = (mu / full_mu) if valid else float("nan")
            summary[(a, arm)] = dict(r2=mu, sd=sd, retained=retained)
            ret_txt = f"{retained:>9.2f}" if valid else f"{'n/a':>9}"
            sw = budgets[a][arm] / max(b["geometry"] if arm in ("geometry", "hybrid")
                                       else (b["readout"] if arm == "weights"
                                        else b["total"]), 1)
            rep(f"  {a:<5} {ARCHS[a]['label']:<26} {arm:<9} {b['total']:>7d} "
                f"{b['geometry_share']*100:>5.0f}% {budgets[a][arm]:>9d} "
                f"{sw:>7.2f} {mu:>+9.4f} {sd:>7.4f} {ret_txt}")
    rep("  ('retained' = R2_arm / R2_full, shown only when R2_full >= 0.05;")
    rep("   otherwise n/a, because a ratio against a non-fitting reference is noise)")
    if "hybrid" in arms:
        probe_model = build_model(archs[0], X_tr, seed=0, substrate=args.substrate)
        cc = readout_calibration_cost(probe_model, X_tr.shape[0])
        ep_cost = 2 * budgets[archs[0]]["geometry"]
        rep("")
        rep("  'hybrid' = geometry probed by FD (same budget as the geometry arm)")
        rep("             + source strengths CALIBRATED in closed form.")
        rep("  W_ij IS the excitation strength of particle j, so the calibration is")
        rep("  an offline source-strength setting, not a numerical shortcut.")
        rep(f"  Cost accounting for arch {archs[0]}: the calibration is ONE-TIME")
        rep(f"    physical route : {cc} measurement-equivalents "
            f"(n_train x sum_k N_k)")
        rep(f"    digital twin   : 0 physical measurements")
        rep(f"    per-epoch FD   : {ep_cost} forward passes/epoch x "
            f"{args.epochs} epochs = {ep_cost*args.epochs}")
        rep("  Compare the one-time cost against the TOTAL FD cost before")
        rep("  concluding that calibrating is cheaper -- on this problem it is not;")
        rep("  it is cheaper only when the epoch count or the particle count is")
        rep("  large.  What it does buy unconditionally is that W is then EXACT.")

    # ---- scaling: R^2 as a function of physical measurements consumed ----
    milestones = [float(t) for t in str(args.milestones).split(",") if t.strip()]
    have_curve = args.curve and any(curves[a][arm] for a in archs for arm in arms)
    if have_curve:
        rep("\n" + "=" * 66)
        rep("  SCALING: test R^2 vs physical measurements consumed")
        rep("=" * 66)
        rep("  A point estimate at one budget is fragile; the RATE (R^2 bought per")
        rep("  measurement) is what a hardware designer needs and is the more")
        rep("  publishable form of the result.  One run yields the whole curve.")
        rep("")
        rep("  Cost model of the x-axis:")
        rep("    geometry / weights / full : 2 x budget forward passes per epoch")
        rep("    hybrid_once               : same, plus ONE physical calibration")
        rep("    hybrid                    : same, calibration assumed")
        rep("                                twin-supplied (zero measurement cost)")
        rep("                                -- physically refreshing it every")
        rep("                                epoch would cost far more than the")
        rep("                                FD probing itself and is not a")
        rep("                                defensible scheme")
        rep("")
        rep(f"  {'arch':<5} {'arm':<13} "
            + " ".join(f"{int(m):>11d}" for m in milestones))
        rep("  " + "-" * (19 + 12 * len(milestones)))
        for a in archs:
            for arm in arms:
                mu, sd, n = merge_curves(curves[a][arm], milestones)
                cells = []
                for j in range(len(milestones)):
                    cells.append(f"{'--':>11}" if n[j] == 0
                                 else f"{mu[j]:>+11.4f}")
                rep(f"  {a:<5} {arm:<13} " + " ".join(cells))
        rep("  ('--' = that run had not yet consumed that many measurements;")
        rep("   milestones are test R^2, mean over seeds)")

    # ---- share axis: does anything vary MONOTONICALLY with the geometry share? ----
    # Only meaningful when several architectures spanning the share axis are run.
    # This is the cross-check that keeps the parameter-count rule of section 5 and
    # the calibration rule of section 6 from being single-point claims, and W is
    # the decisive architecture because it is the only one where geometry has
    # FEWER parameters than the readout.
    if have_curve and len(archs) >= 3 and milestones:
        rep("\n" + "=" * 74)
        rep("  SHARE AXIS: does the outcome vary monotonically with geometry share?")
        rep("=" * 74)
        rep("  Sorted by share.  'margin' is the calibrated arm minus the best search")
        rep("  arm, at a LOW milestone so the comparison is not saturated.")
        rep("")
        m_lo = milestones[0]
        share_x, marg_x, geo_ret_x = [], [], []
        rep(f"  {'arch':<5} {'share':>6} {'geom':>6} {'read':>6} {'count says':>11} "
            f"{'actually':>9} {'margin@'+str(int(m_lo)):>13} {'geo kept':>9}")
        rep("  " + "-" * 78)
        order = sorted(archs, key=lambda a: -census[a]["geometry_share"])
        for a in order:
            sh = census[a]["geometry_share"]
            # NOTE: `sizes` (from effective_budget) is keyed by ARM and is
            # overwritten by each loop iteration, so per-architecture parameter
            # counts must come from `census`.  Using sizes[a] here is what made
            # this section crash with KeyError on the last architecture.
            sg = int(census[a]["geometry"])
            sr = int(census[a]["readout"])
            pred = "geometry" if sg < sr else "readout"
            if ("geometry" in arms) and ("weights" in arms):
                g = summary[(a, "geometry")]["r2"]
                w = summary[(a, "weights")]["r2"]
                act = "geometry" if g > w else "readout"
                # same display rule as the main table: a ratio is meaningless
                # when the denominator is near zero
                geo_ret = (g / w) if w > 0.05 else float("nan")
            else:
                act, geo_ret = "--", float("nan")
            if "hybrid" in arms:
                mh, _s, _n = merge_curves(curves[a]["hybrid"], [m_lo])
                mb = max(merge_curves(curves[a][x], [m_lo])[0][0]
                         for x in arms if x in ("geometry", "weights", "full"))
                marg = (float(mh[0]) - float(mb)
                        if np.isfinite(mh[0]) and np.isfinite(mb) else float("nan"))
            else:
                marg = float("nan")
            share_x.append(sh)
            marg_x.append(marg)
            geo_ret_x.append(geo_ret)
            rep(f"  {a:<5} {sh*100:>5.0f}% {sg:>6d} {sr:>6d} {pred:>11} {act:>9} "
                f"{marg:>+13.4f} "
                f"{('--' if not np.isfinite(geo_ret) else f'{geo_ret:9.2f}'):>9}")
        rep("")
        rep(f"  {'count says':>11} = which arm has FEWER parameters")
        rep(f"  {'actually':>11} = which arm ends with the higher R^2")
        rep("")
        hits = sum(1 for a in order
                   if ("geometry" in arms) and ("weights" in arms)
                   and (("geometry" if census[a]["geometry"] < census[a]["readout"]
                         else "readout")
                        == ("geometry"
                            if summary[(a, "geometry")]["r2"]
                            > summary[(a, "weights")]["r2"] else "readout")))
        rep(f"  parameter-count rule: {hits}/{len(order)} architectures")
        rep("  (the rule is protocol-dependent where it fails: an arm with fewer")
        rep("   parameters completes more sweeps of its own space per probe, which")
        rep("   equal_probes rewards and equal_sweeps does not)")
        rep("")
        rep("  READOUT PARAMS vs TRAINING SAMPLES -- the variable that appears to")
        rep("  control the calibration, more so than the geometry share:")
        n_tr = int(X_tr.shape[0])
        for a in order:
            ratio = census[a]["readout"] / max(n_tr, 1)
            if "hybrid" in arms:
                best = max(summary[(a, x)]["r2"] for x in arms
                           if x in ("geometry", "weights", "full"))
                verdict = "WINS" if summary[(a, "hybrid")]["r2"] > best else "LOSES"
            else:
                verdict = "--"
            rep(f"    {a:<5} readout {census[a]['readout']:>5d} / {n_tr} samples"
                f" = {ratio:>5.2f}x   hybrid vs best search: {verdict}")
        for nm, xs, ys in (("geometry-arm retained R2", share_x, geo_ret_x),
                           ("calibration margin", share_x, marg_x)):
            v = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
            if len(v) >= 2 and np.std([x for x, _ in v]) > 1e-12:
                r = float(np.corrcoef([x for x, _ in v], [y for _, y in v])[0, 1])
                rep(f"  corr(geometry share, {nm}) = {r:+.3f}  over {len(v)} archs")
        rep("")
        rep("  A correlation over 5 points is weak evidence on its own; what it can")
        rep("  do is falsify a MONOTONE claim.  Read the ordering, not the number.")

    # ---- verdict ----
    rep("\n" + "=" * 66)
    rep("  VERDICT -?Phase 1 (each arm trained from scratch)")
    rep("=" * 66)

    geo_wins = []
    have_pair = ("geometry" in arms) and ("weights" in arms)
    if not have_pair:
        rep("  (need both the 'geometry' and 'weights' arms for a verdict)")
    for a in (archs if have_pair else []):
        g = summary[(a, "geometry")]["r2"]
        w = summary[(a, "weights")]["r2"]
        geo_wins.append(g > w)
        rep(f"  {a}: geometry R2={g:+.4f} vs weights R2={w:+.4f} "
            f"-> {'GEOMETRY' if g > w else 'WEIGHTS'} carries more skill")

    if not learned:
        rep("\n  " + inconclusive_text(all_r2))
    elif not have_pair:
        pass
    elif all(geo_wins):
        rep("\n  Node coordinates carry more task-relevant capacity per unit of")
        rep("  analog measurement than the readout matrix, in all 3 architectures.")
        rep("  This is the result the 'nodes replace matrices' thesis predicts.")
    elif not any(geo_wins):
        rep("\n  The readout matrix dominates in all 3 architectures: under this")
        rep("  budget the geometry is a feature map, not the carrier of capacity.")
    else:
        rep("\n  The winner FLIPS with the architecture. Capacity follows whichever")
        rep("  parameter set the architecture allocates more stored quantities to,")
        rep("  i.e. parameter SHARE predicts the outcome better than parameter TYPE.")

    # ---- Phase 2: attribution from the trained full model ----
    # Phase 1 trains each subset from scratch, which handicaps an arm that has
    # to fit through a random frozen partner.  Phase 2 removes that confound:
    # start from the TRAINED full model (equally good for both), then fine-tune
    # only one subset.  The R^2 that survives tells you which half was actually
    # carrying the fit.
    attr_rows = []
    if args.attribute and ("full" in arms) and have_pair:
        rep("\n" + "=" * 66)
        rep("  PHASE 2 -?attribution from the trained full model")
        rep("=" * 66)
        rep("  Start from the trained 'full' params, then fine-tune ONE subset")
        rep("  with the same probe budget. R^2 retained = capacity carried by")
        rep("  the FROZEN half (if freezing it costs nothing, it was redundant).")
        rep("")
        rep(f"  {'arch':<5} {'fine-tune':<10} {'frozen':<10} {'R2 start':>9} "
            f"{'R2 after':>9} {'delta':>8} {'kept':>7}")
        rep("  " + "-" * 64)

        # Phase 2 starts from an already-trained model, so it needs only a
        # short fine-tune: a quarter of the Phase-1 schedule.
        attr_epochs = max(args.epochs // 4, 15)
        for a in archs:
            for s in range(args.seeds):
                base_vec = trained[(a, "full", s)]
                for arm in ("geometry", "weights"):
                    m2 = build_model(a, X_tr, seed=s, substrate=args.substrate)
                    m2.set_params(base_vec)
                    r_start = r2(y_te, m2.forward(X_te))
                    train_model(
                        m2, X_tr, y_tr, X_te, y_te, arm=arm,
                        epochs=attr_epochs, lr=args.lr, eps_rel=args.eps_rel,
                        budget=budgets[a][arm], update_mode="sequential",
                        seed=s, verbose=False, probe_batch=args.probe_batch,
                    )
                    r_after = r2(y_te, m2.forward(X_te))
                    frozen = "weights" if arm == "geometry" else "geometry"
                    # ratio only meaningful when the full model actually fits;
                    # below R2=0.05 the ratio is numerical noise
                    kept = (r_after / r_start) if r_start > 0.05 else float("nan")
                    attr_rows.append(dict(arch=a, seed=s, finetune=arm,
                                          frozen=frozen, r_start=r_start,
                                          r_after=r_after, kept=kept))
                    rep(f"  {a:<5} {arm:<10} {frozen:<10} {r_start:>+9.4f} "
                        f"{r_after:>+9.4f} {r_after-r_start:>+8.4f} {kept:>7.3f}")
            rep("")

        # aggregate -- use the ABSOLUTE gain as the primary metric: in a smoke
        # run the starting R^2 can be near zero, which makes a ratio meaningless
        rep(f"  {'arch':<5} {'freeze geom':>12} {'freeze wts':>11} "
            f"{'mean delta (geom-frozen)':>25} {'mean delta (wts-frozen)':>24} "
            f"{'verdict':>28}")
        rep("  " + "-" * 104)
        attr_verdict = []
        for a in archs:
            d_g = [r["r_after"] - r["r_start"] for r in attr_rows
                   if r["arch"] == a and r["frozen"] == "geometry"]
            d_w = [r["r_after"] - r["r_start"] for r in attr_rows
                   if r["arch"] == a and r["frozen"] == "weights"]
            k_g = [r["kept"] for r in attr_rows
                   if r["arch"] == a and r["frozen"] == "geometry"]
            k_w = [r["kept"] for r in attr_rows
                   if r["arch"] == a and r["frozen"] == "weights"]
            mg = float(np.nanmean(d_g)) if d_g else np.nan
            mw = float(np.nanmean(d_w)) if d_w else np.nan
            # d_w: gain from re-fitting GEOMETRY with weights frozen
            # d_g: gain from re-fitting WEIGHTS with geometry frozen
            if mw > mg:
                v = "geometry carries the fit"
            else:
                v = "weights carry the fit"
            if not learned:
                v = "(inconclusive: models did not learn)"
            attr_verdict.append((a, mg, mw, mw > mg))
            rep(f"  {a:<5} {nanmean_safe(k_g):>12.3f} {nanmean_safe(k_w):>11.3f} "
                f"{mg:>25.4f} {mw:>24.4f} {v:>28}")
        rep("")
        rep("  'freeze geom' / 'freeze wts' = R^2 retained by re-fitting the OTHER")
        rep("  half (nan when the full model's R^2 was below 0.05, i.e. the ratio")
        rep("  is not interpretable -- read the delta columns then).")
        rep("  'mean delta' = absolute R^2 gained by re-fitting that half; the")
        rep("  larger one is the half that still had room to improve.")
        n_geo = sum(1 for _a, _g, _w, wins in attr_verdict if wins)
        if learned:
            rep(f"  -> geometry carries the fit in {n_geo}/{len(attr_verdict)} "
                f"architectures, weights in "
                f"{len(attr_verdict)-n_geo}/{len(attr_verdict)}")
        else:
            rep("  -> " + inconclusive_text(all_r2))
        rep("")
        rep("  Phase 1 vs Phase 2 disagreeing is itself informative: it separates")
        rep("  'which parameters can LEARN the task from scratch' (Phase 1) from")
        rep("  'which parameters HOLD the solution once found' (Phase 2).")

    # ---- correlation: geometry share vs geometry-arm retained skill ----
    xs = np.array([census[a]["geometry_share"] for a in archs])
    if len(archs) >= 2 and "geometry" in arms:
        try:
            ys = np.array([summary[(a, "geometry")]["retained"] for a in archs])
            if np.std(xs) > 1e-9 and np.std(ys) > 1e-9:
                r_g = float(np.corrcoef(xs, ys)[0, 1])
                rep(f"\n  corr(geometry share, geometry-arm retained R2) = {r_g:+.2f} "
                    f"over {len(archs)} architectures")
        except Exception as exc:
            rep(f"  [warn] correlation skipped: {exc}")

    rep(f"\n  Total wall time: {time.time()-t_start:.0f}s")
    rep(f"  Substrate: {args.substrate}")
    rep(f"  Cache: {os.path.dirname(cache_path('x','y','z',0,'w',0,0))} "
        f"(keyed by task/substrate/arch/arm/seed/epochs/budget; reused by abl_03)")

    # Save the report BEFORE plotting: if a downstream pipe closes early
    # (e.g. `python ... | Select-Object -First 20`) the process is killed on its
    # next print, and results must not be lost to a cosmetic failure.
    path = rep.save()
    print(f"\nReport saved: {path}")

    # ---- plot ----
    plt = try_plots()
    if plt is not None:
        fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4.2),
                                 squeeze=False)
        for ax, a in zip(axes[0], archs):
            mus = [summary[(a, arm)]["r2"] for arm in arms]
            sds = [summary[(a, arm)]["sd"] for arm in arms]
            palette = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
            colors = [palette[i % len(palette)] for i in range(len(arms))]
            ax.bar(range(len(arms)), mus, yerr=sds, capsize=4,
                   color=colors, edgecolor="black")
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(arms, fontsize=9)
            ax.set_title(f"{a}: {ARCHS[a]['label']}\n"
                         f"geom {census[a]['geometry_share']*100:.0f}% / "
                         f"readout {census[a]['readout_share']*100:.0f}%", fontsize=9)
            ax.grid(True, alpha=0.25, axis="y")
            if ax is axes[0][0]:
                ax.set_ylabel("test R2")
        fig.suptitle(f"Ablation A: capacity split at equal measurement budget "
                     f"({args.substrate}, task={args.task}, budget={budget_txt})",
                     fontsize=11)
        plt.tight_layout()
        plot_path = os.path.join(
            OUT_DIR, f"abl_01_capacity_split{sub_tag}{mode_tag}{tag}_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot: {plot_path}")

        if have_curve:
            all_m = [m for a in archs for arm in arms
                     for c in curves[a][arm] for m, _ in c]
            if all_m:
                grid = np.unique(np.round(np.logspace(
                    np.log10(max(min(all_m), 1.0)), np.log10(max(all_m)), 40)))
                palette = {"geometry": "#2196F3", "weights": "#F44336",
                           "hybrid": "#4CAF50", "hybrid_once": "#9C27B0",
                           "full": "#FF9800"}
                fig, axes = plt.subplots(1, len(archs),
                                         figsize=(5.2 * len(archs), 4.4),
                                         squeeze=False)
                for axx, a in zip(axes[0], archs):
                    for arm in arms:
                        mu, sd, n = merge_curves(curves[a][arm], grid)
                        col = palette.get(arm)
                        axx.plot(grid, mu, "-", color=col, linewidth=1.8, label=arm)
                        axx.fill_between(grid, mu - sd, mu + sd, color=col,
                                         alpha=0.15, linewidth=0)
                    axx.set_xscale("log")
                    axx.set_xlabel("physical measurements")
                    axx.set_ylabel("test R2")
                    axx.set_title(f"{a}: {ARCHS[a]['label']}\n"
                                  f"geom {census[a]['geometry_share']*100:.0f}%",
                                  fontsize=9)
                    axx.grid(True, alpha=0.25)
                    axx.legend(fontsize=7)
                fig.suptitle("Scaling: test R2 vs physical measurements consumed "
                             f"({args.substrate}, {args.budget_mode}, "
                             f"{args.seeds} seeds)", fontsize=11)
                plt.tight_layout()
                sp = os.path.join(
                    OUT_DIR,
                    f"abl_01_scaling{sub_tag}{mode_tag}{tag}_plot.png")
                plt.savefig(sp, dpi=150)
                plt.close()
                print(f"  Scaling plot: {sp}")


if __name__ == "__main__":
    main()
