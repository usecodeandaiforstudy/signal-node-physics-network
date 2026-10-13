"""
Ablation F -- Does the headline rule hold on OTHER tasks?
=========================================================
Project: singal-node-physics-network

WHY THIS EXISTS
---------------
Every result in FINDINGS.md is measured on one 4D synthetic regression.  The
headline claim -- "calibrate the source strengths in closed form, spend the
measurements on the geometry, for 15-30x fewer measurements" -- therefore has an
untested boundary: is it a property of the method, or of that one task?

This script sweeps a family of SELF-CONTAINED toy tasks (no downloads, no
external data, reproducible offline) that differ deliberately in the ways that
could matter: dimensionality, irrelevant dimensions, task TYPE, target
smoothness, and noise structure.

    kind     character
    2d       D=2 smooth                          baseline
    4d       D=4 smooth + product term           the task used everywhere else
    10d      D=10, 8 pure-noise dims             dimensionality
    moons    D=2 two-class boundary, +-1         different TASK TYPE
    steps    D=4 DISCONTINUOUS target            adversarial: a Gaussian basis
                                                 cannot fit a square wave, so the
                                                 linear readout is the bottleneck
    hetero   D=4 HETEROSCEDASTIC noise           adversarial: the calibration IS
                                                 an ordinary least-squares fit, so
                                                 non-uniform noise misspecifies it
    shift    D=4 train [-2,2], test [-3,3]       adversarial: the gate validates
                                                 IN-DISTRIBUTION, so it cannot see
                                                 a shift.  This is the test for the
                                                 one counterexample that would
                                                 matter -- a cell where the gate
                                                 ACCEPTS and the arm still loses.

The last two are adversaries of the rule, not illustrations of it.  A sweep that
only contained friendly tasks would be circular.

PROTOCOL
--------
Two architectures spanning the geometry-share range (G = 69%, F2 = 81%), four
arms (geometry / readout / hybrid / full), equal measurement cost per epoch,
identical seeds and epoch count for every task.  Nothing about the trainer is
task-specific, so any difference between tasks is a difference in the task.

THE CROSS-TASK METRIC
---------------------
Absolute R^2 is not comparable between tasks (a discontinuous target simply has a
lower ceiling).  The reported quantity is therefore the RATE:

    speedup = (measurements every arm has spent at the frontier)
              / (measurements the calibrated arm needed to reach the accuracy the
                 best search arm reaches at the frontier)

i.e. "how much earlier does calibrating get you where searching ends up".  A
speedup of 1 means calibrating is not better; `never` means the calibrated arm
does not reach the search arm's final accuracy inside the tested budget.

Output: outputs/abl_06_task_sweep.txt + _plot.png
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abl_common import (                                   # noqa: E402
    ARCHS, OUT_DIR, Report, build_model, cache_path, curve_at, effective_budget,
    header, load_model, make_regression_data, merge_curves, r2, save_model,
    selftest, set_diffusion_defaults, set_ideal, task_description, train_model,
    try_plots,
)

TASKS = ("2d", "4d", "10d", "moons", "steps", "hetero", "shift")
ARCH_LIST = ("G", "F2")
SEARCH_ARMS = ("geometry", "weights", "full")
ARMS = SEARCH_ARMS + ("hybrid",)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ablation F: task sweep")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--archs", default=",".join(ARCH_LIST))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--substrate", default="diffusion",
                   help="diffusion is ~2x faster per forward and agrees with the "
                        "circuit to 2e-08 on identical parameters")
    p.add_argument("--low-budget", type=int, default=3000,
                   help="measurement count for the early-budget comparison")
    p.add_argument("--probe-batch", type=int, default=96)
    p.add_argument("--tag", default=None,
                   help="suffix for the report filename, so a partial rerun does "
                        "not overwrite the full sweep")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


def _load_or_train(a, arm, s, tk, X_tr, y_tr, X_te, y_te, budget, args):
    """
    One (task, arch, arm, seed) cell, cached on disk.

    The cached payload carries the MEASUREMENT CURVE and the gate counters, not
    just the parameters, because the milestone tables of this sweep are built from
    the curves -- caching only the model would make a restart produce a report
    with missing tables instead of a resumable one.
    """
    path = cache_path(a, arm, "none", s, tk, args.epochs, budget, args.substrate)
    if os.path.exists(path):
        try:
            d = np.load(path)
            curve = list(zip(d["curve_m"].tolist(), d["curve_r2"].tolist()))
            model = load_model(a, arm, "none", s, tk, args.epochs, budget, X_tr,
                               substrate=args.substrate)
            if model is not None:
                hist = {"curve": curve,
                        "fwd": int(d["fwd"]) if "fwd" in d else 0,
                        "n_calibrations": int(d["n_cal"]) if "n_cal" in d else 0,
                        "calib_rejected": int(d["n_rej"]) if "n_rej" in d else 0,
                        "calib_reject_gap": [], "_from_cache": True}
                return model, hist
        except Exception:
            pass                       # unreadable cache entry -> retrain

    model = build_model(a, X_tr, seed=s, substrate=args.substrate)
    hist = train_model(
        model, X_tr, y_tr, X_te, y_te, arm=arm,
        epochs=args.epochs, budget=budget, seed=s,
        probe_batch=args.probe_batch, substrate=args.substrate,
        eval_every=1, eval_X=X_te, eval_y=y_te,
    )
    try:
        save_model(model, a, arm, "none", s, tk, args.epochs, budget,
                   substrate=args.substrate,
                   extra={"curve_m": np.array([m for m, _ in hist["curve"]],
                                              dtype=np.int64),
                          "curve_r2": np.array([v for _, v in hist["curve"]],
                                               dtype=np.float64),
                          "fwd": hist["fwd"],
                          "n_cal": hist["n_calibrations"],
                          "n_rej": hist["calib_rejected"]})
    except Exception:
        pass                           # a failed cache write must not kill the run
    return model, hist


def main():
    args = parse_args()
    if args.quick:
        if args.seeds == 2:
            args.seeds = 1
        if args.epochs == 40:
            args.epochs = 12
    tasks = [t for t in str(args.tasks).split(",") if t.strip()]
    archs = [a for a in str(args.archs).split(",") if a.strip()]

    set_diffusion_defaults(level="none")
    set_ideal()

    rep = Report(f"abl_06_task_sweep{('_' + args.tag) if args.tag else ''}.txt")
    for line in header("Ablation F - Does the calibration rule hold on other tasks?",
                       f"tasks={tasks} archs={archs} seeds={args.seeds} "
                       f"epochs={args.epochs} substrate={args.substrate}"):
        rep(line)

    rep("\n--- Part 0: self-test ---")
    selftest(verbose=True)
    rep("  All tasks are synthetic and self-contained: no downloads, no external")
    rep("  datasets, reproducible offline.  Same 256/128 split, same seeds, same")
    rep("  trainer and same epoch count for every task and architecture.")

    # ---------------- task characterisation ----------------
    rep("\n" + "=" * 78)
    rep("  Part 1: the tasks, and how hard they are on their own terms")
    rep("=" * 78)
    rep(f"  {'task':<8} {'D':>3} {'y std':>8} {'|y| max':>9} {'ridge on x (R2)':>17} "
        f"{'character'}")
    rep("  " + "-" * 96)
    data = {}
    for tk in tasks:
        X_tr, y_tr, X_te, y_te = make_regression_data(tk)
        data[tk] = (X_tr, y_tr, X_te, y_te)
        # a linear readout of the RAW INPUT is the honesty check on task hardness:
        # if this is already near 1 the task is linearly trivial, and if it is far
        # below 1 the nonlinearity is doing the work.
        A = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
        w, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
        Ate = np.hstack([X_te, np.ones((X_te.shape[0], 1))])
        lin_r2 = r2(y_te, Ate @ w)
        rep(f"  {tk:<8} {X_tr.shape[1]:>3} {y_tr.std():>8.3f} "
            f"{np.abs(y_tr).max():>9.3f} {lin_r2:>+17.4f}  {task_description(tk)}")
    rep("")
    rep("  The last column matters: a task whose plain linear readout is already")
    rep("  near 1 cannot discriminate between the arms, and one far below 1 is")
    rep("  where a nonlinear basis has to earn its keep.")

    # ---------------- training ----------------
    # CHECKPOINTED.  This is the longest run in the repository (seven tasks x two
    # architectures x four arms x seeds, ~100 min at 3 seeds) and it used to reuse
    # nothing, so any interruption threw the whole sweep away -- which happened
    # twice.  Every (task, arch, arm, seed) cell is now cached, INCLUDING its
    # measurement curve, so the milestone tables survive a restart.  The cache key
    # carries the task name, so cells of different tasks can never collide.
    curves = {}          # (task, arch, arm) -> list of curves
    hists = {}           # (task, arch, arm) -> list of history dicts
    n_loaded, n_trained = 0, 0
    t0 = time.time()
    for tk in tasks:
        X_tr, y_tr, X_te, y_te = data[tk]
        for a in archs:
            bd, sizes = effective_budget(a, X_tr, list(ARMS), None,
                                        substrate=args.substrate)
            budget = int(bd["full"])
            for arm in ARMS:
                cs, hs = [], []
                for s in range(args.seeds):
                    model, hist = _load_or_train(
                        a, arm, s, tk, X_tr, y_tr, X_te, y_te, budget, args)
                    if hist.get("_from_cache"):
                        n_loaded += 1
                    else:
                        n_trained += 1
                    cs.append(hist["curve"])
                    hs.append(hist)
                curves[(tk, a, arm)] = cs
                hists[(tk, a, arm)] = hs
            rep(f"  [{tk}/{a}] {len(ARMS)} arms x {args.seeds} seeds "
                f"({n_loaded} reused, {n_trained} trained; "
                f"{time.time()-t0:.0f}s elapsed)")
    rep("")
    rep(f"  {len(curves)} cells in {time.time()-t0:.0f}s "
        f"({n_loaded} reused from cache, {n_trained} trained)")

    # ---------------- the do-no-harm guard ----------------
    rep("\n" + "=" * 78)
    rep("  Part 1b: the do-no-harm guard (calibration deployed only if it helps)")
    rep("=" * 78)
    rep("  A calibration is rejected when it does not beat the strengths already in")
    rep("  place on a held-out slice of the training data.  Rejections are the")
    rep("  guard doing its job: they mark cells where an unregularised solve would")
    rep("  have overfitted (see section 6.5 of FINDINGS.md).")
    rep("")
    rep(f"  {'task':<8} {'arch':<5} {'calibrations':>13} {'rejected':>9} "
        f"{'mean gap':>10}")
    rep("  " + "-" * 50)
    for tk in tasks:
        for a in archs:
            hs = hists.get((tk, a, "hybrid"), [])
            n_cal = sum(h.get("n_calibrations", 0) for h in hs)
            n_rej = sum(h.get("calib_rejected", 0) for h in hs)
            gaps = [g for h in hs for g in h.get("calib_reject_gap", [])]
            gap_s = f"{np.mean(gaps):+.4f}" if gaps else "--"
            rep(f"  {tk:<8} {a:<5} {n_cal:>13d} {n_rej:>9d} {gap_s:>10}")

    # ---------------- per-task frontier table ----------------
    rep("\n" + "=" * 78)
    rep("  Part 2: accuracy at the COMMON MEASUREMENT FRONTIER")
    rep("=" * 78)
    rep("  Every arm spends the same budget, so the frontier is the same for all.")
    rep("")
    rep(f"  {'task':<8} {'arch':<5} {'frontier':>9} {'geometry':>10} {'readout':>10} "
        f"{'full':>10} {'hybrid':>10}")
    rep("  " + "-" * 70)
    frontier_vals = {}
    for tk in tasks:
        for a in archs:
            ms = [c[-1][0] for arm in ARMS for c in curves[(tk, a, arm)] if c]
            fr = min(ms)
            vals = {}
            for arm in ARMS:
                mu, _s, _n = merge_curves(curves[(tk, a, arm)], [fr])
                vals[arm] = float(mu[0]) if np.isfinite(mu[0]) else float("nan")
            frontier_vals[(tk, a)] = (fr, vals)
            rep(f"  {tk:<8} {a:<5} {fr:>9d} {vals['geometry']:>+10.4f} "
                f"{vals['weights']:>+10.4f} {vals['full']:>+10.4f} "
                f"{vals['hybrid']:>+10.4f}")

    # ---------------- the cross-task rate metric ----------------
    rep("\n" + "=" * 78)
    rep("  Part 3: THE RATE -- how much earlier does calibrating get there?")
    rep("=" * 78)
    rep("  speedup = frontier / (measurements the calibrated arm needed to reach")
    rep("            the accuracy the best search arm reaches at the frontier)")
    rep("")
    rep(f"  {'task':<8} {'arch':<5} {'best search':>12} {'hybrid':>10} "
        f"{'hybrid wins':>12} {'speedup':>10} {'@' + str(args.low_budget):>12}")
    rep("  " + "-" * 76)
    speedups = {}
    for tk in tasks:
        for a in archs:
            fr, vals = frontier_vals[(tk, a)]
            best = max(vals["geometry"], vals["weights"], vals["full"])
            hyb = vals["hybrid"]
            wins = np.isfinite(hyb) and hyb > best
            # earliest measurement count at which the hybrid reaches `best`
            reach = None
            for c in curves[(tk, a, "hybrid")]:
                for m, v in c:
                    if np.isfinite(v) and v >= best:
                        reach = m if reach is None else min(reach, m)
            sp = (fr / reach) if reach else float("nan")
            speedups[(tk, a)] = sp
            lo_h, _s, _n = merge_curves(curves[(tk, a, "hybrid")],
                                        [args.low_budget])
            lo_b = max(
                merge_curves(curves[(tk, a, arm)], [args.low_budget])[0][0]
                for arm in SEARCH_ARMS)
            lo = (f"{lo_h[0]:+.3f} vs {lo_b:+.3f}"
                  if np.isfinite(lo_h[0]) and np.isfinite(lo_b) else "--")
            sp_s = f"{sp:.1f}x" if np.isfinite(sp) else "never"
            rep(f"  {tk:<8} {a:<5} {best:>+12.4f} {hyb:>+10.4f} "
                f"{'YES' if wins else 'NO':>12} {sp_s:>10} {lo:>12}")

    # ---------------- verdict ----------------
    rep("\n" + "=" * 78)
    rep("  VERDICT: is the rule a property of the method or of one task?")
    rep("=" * 78)
    n_cells = len(speedups)
    n_wins = sum(1 for (tk, a) in speedups
                 if np.isfinite(frontier_vals[(tk, a)][1]["hybrid"])
                 and frontier_vals[(tk, a)][1]["hybrid"]
                 > max(frontier_vals[(tk, a)][1][x] for x in SEARCH_ARMS))
    fin = [v for v in speedups.values() if np.isfinite(v)]
    rep(f"  The calibrated arm beats the best search arm in {n_wins}/{n_cells} "
        f"task x architecture cells.")
    if fin:
        rep(f"  Speedup where it wins: median {np.median(fin):.1f}x, "
            f"range {min(fin):.1f}x to {max(fin):.1f}x")
    per_task = {}
    for tk in tasks:
        vals = [speedups[(tk, a)] for a in archs if np.isfinite(speedups[(tk, a)])]
        per_task[tk] = float(np.mean(vals)) if vals else float("nan")
    rep("")
    rep("  Mean speedup per task:")
    for tk in sorted(per_task, key=lambda k: -(per_task[k] if np.isfinite(per_task[k]) else -1)):
        v = per_task[tk]
        rep(f"    {tk:<8} {'never' if not np.isfinite(v) else f'{v:6.1f}x'}   "
            f"{task_description(tk)}")
    rep("")
    rep("  The two ADVERSARIAL tasks are the test of the claim:")
    adv = [per_task.get(t, float("nan")) for t in ("steps", "hetero", "shift")]
    oth = [per_task.get(t, float("nan")) for t in ("2d", "4d", "10d", "moons")]
    oth_f = [v for v in oth if np.isfinite(v)]
    adv_f = [v for v in adv if np.isfinite(v)]
    rep(f"    friendly tasks   (2d/4d/10d/moons): "
        f"{'n/a' if not oth_f else f'mean {np.mean(oth_f):.1f}x'}")
    rep(f"    adversarial tasks (steps/hetero/shift)  : "
        f"{'n/a' if not adv_f else f'mean {np.mean(adv_f):.1f}x'}")
    if n_wins == n_cells:
        rep("")
        rep("  The rule holds on EVERY task and architecture tested, including the")
        rep("  two designed to break it (a discontinuous target and heteroscedastic")
        rep("  noise).  Within this family the result is a property of the method,")
        rep("  not of the 4D task that produced it.")
    elif n_wins == 0:
        rep("")
        rep("  The rule fails everywhere outside the task it was found on, so the")
        rep("  headline must be restated as task-specific.")
    else:
        rep("")
        rep("  The rule is TASK-DEPENDENT.  Section 6 must state the conditions")
        rep("  under which calibrating wins instead of recommending it outright.")

    path = rep.save()
    print(f"\nReport saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/')}")

    plt = try_plots()
    if plt is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        cols = {"geometry": "#2196F3", "weights": "#F44336",
                "full": "#FF9800", "hybrid": "#4CAF50"}
        for axx, tk in zip(axes.ravel(), tasks):
            for a in archs:
                for arm in ARMS:
                    all_m = [m for c in curves[(tk, a, arm)] for m, _ in c]
                    if not all_m:
                        continue
                    grid = np.unique(np.linspace(min(all_m), max(all_m), 25))
                    mu, sd, _n = merge_curves(curves[(tk, a, arm)], grid)
                    axx.plot(grid, mu, "-", color=cols[arm], linewidth=1.4,
                             alpha=0.9 if a == archs[0] else 0.45,
                             label=f"{arm} ({a})" if len(archs) <= 2 else arm)
            axx.set_xlabel("physical measurements")
            axx.set_ylabel("test R2")
            axx.set_title(f"{tk}: {task_description(tk)}", fontsize=8)
            axx.grid(True, alpha=0.25)
            axx.legend(fontsize=6)
        for axx in axes.ravel()[len(tasks):]:
            axx.axis("off")
        fig.suptitle("Ablation F: does 'calibrate the strengths, measure the "
                     "geometry' survive a change of task?", fontsize=12)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "abl_06_task_sweep_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
