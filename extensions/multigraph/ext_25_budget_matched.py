"""
ext_25 -- the budget-matched variant: FIX the budget, vary only N
=================================================================
Project: singal-node-physics-network, extension `multigraph`

WHY THIS EXISTS
---------------
ext_24 tried to answer the base's section 11 tail -- "does the hybrid advantage at
3 000 measurements hold for a larger particle budget?" -- and could not, because of a
confound it priced but did not remove: `effective_budget` derives the per-epoch budget
from the architecture, so raising N raised the budget too (`600 -> 1200 -> 2500`), which
moved the hybrid curve's FIRST recorded point (`1200 -> 2400 -> 5000`) past the `3000`
grid point.  At N=208 the comparison was therefore **undefined, not bad**.

**The fix is one line of design: pass an explicit budget and vary only N.**  Then the
first curve point is the same for every architecture and the low-budget comparison is
defined everywhere.

THE RISK THIS DESIGN INTRODUCES, AND HOW IT IS HANDLED
------------------------------------------------------
A FIXED budget can be too small for a larger network: `train_model` spends `budget` probes
per epoch and each probe costs 2 forward passes, so a full sweep of an N-particle layer
needs `2*(D+2)*N` forward passes.  At budget 600 an N=208 layer manages 0.24 sweeps per
epoch.  The base version's own section 7 warns about exactly this: at insufficient probes
per epoch its comparison was reported **INCONCLUSIVE** because nothing reached R^2 >= 0.05,
so the cells carried no information.

**So every cell here is checked against the learning floor, and cells that fail it are
reported as uninformative rather than counted as wins.**  A hybrid that "wins" because the
search arms could not train is not evidence, and that failure mode is the one this design
makes possible.

WHAT IS MEASURED
----------------
  P1  N in {52, 104, 208} x a FIXED per-epoch budget in {600, 2400}, one task, one seed.
      For each cell: the frontier values of the base's four arms, the `@3000` comparison
      (now defined for every N at a fixed budget), and whether the cell clears the floor.
  P2  the same reading the base uses, quoted from its own report: the R^2 at 3 000
      measurements, hybrid against best search.

SCOPE: task 4d, 1 seed, 40 epochs, single-layer matrix-free architectures added to
`abl_common.ARCHS` AT RUNTIME (the base file is not edited).  Toy, as everywhere.

Output: outputs/ext_25_budget_matched.txt + _plot.png
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)

import abl_common as ac                                  # noqa: E402

from abl_common import (                                 # noqa: E402
    Report, build_model, cache_path, header, load_model, make_regression_data,
    merge_curves, save_model, set_diffusion_defaults, set_ideal, train_model,
    try_plots,
)
from extensions.multigraph.ext_24_particle_budget import (   # noqa: E402
    ARMS, EPOCHS, LOW_BUDGET, SCALE_ARCHS, SUBSTRATE, frontier_and_rate,
    install_scale_archs,
)

BUDGETS = (600, 2400)
FLOOR = 0.05          # abl_common.learning_ok's own bar


def run_cell_fixed(arch, arm, seed, task, X_tr, y_tr, X_te, y_te, budget,
                   epochs=EPOCHS, substrate=SUBSTRATE, probe_batch=96):
    """
    `ext_24.run_cell` with the budget PASSED IN rather than derived from the arch.

    That single change is the whole point of this script: `effective_budget` makes the
    per-epoch budget a function of N, so N could not be varied alone.  The cache key is
    still the base's own `cache_path(...)`, now with the explicit budget, so these cells
    occupy their own slots and can never collide with the base's own sweeps.
    """
    path = cache_path(arch, arm, "none", seed, task, epochs, budget, substrate)
    if os.path.exists(path):
        try:
            d = np.load(path)
            model = load_model(arch, arm, "none", seed, task, epochs, budget, X_tr,
                               substrate=substrate)
            if model is not None:
                return dict(curve=list(zip(d["curve_m"].tolist(),
                                           d["curve_r2"].tolist())),
                            from_cache=True, budget=budget,
                            n_cal=int(d["n_cal"]) if "n_cal" in d else 0,
                            n_rej=int(d["n_rej"]) if "n_rej" in d else 0)
        except Exception:
            pass
    model = build_model(arch, X_tr, seed=seed, substrate=substrate)
    hist = train_model(model, X_tr, y_tr, X_te, y_te, arm=arm, epochs=epochs,
                       budget=int(budget), seed=seed, probe_batch=probe_batch,
                       substrate=substrate, eval_every=1, eval_X=X_te, eval_y=y_te)
    try:
        save_model(model, arch, arm, "none", seed, task, epochs, int(budget),
                   substrate=substrate,
                   extra={"curve_m": np.array([m for m, _ in hist["curve"]],
                                              dtype=np.int64),
                          "curve_r2": np.array([v for _, v in hist["curve"]],
                                               dtype=np.float64),
                          "fwd": hist["fwd"],
                          "n_cal": hist["n_calibrations"],
                          "n_rej": hist["calib_rejected"]})
    except Exception:
        pass
    return dict(curve=hist["curve"], from_cache=False, budget=int(budget),
                n_cal=hist["n_calibrations"], n_rej=hist["calib_rejected"])


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_25: budget-matched particle scaling")
    ap.add_argument("--task", default="4d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--archs", default="S52,S104,S208")
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    # The device layer draws from the GLOBAL np.random, so without this the realisation
    # differs every run and a borderline assertion can flip: ext_06's C2 margin was 133x
    # one run and 9.73x the next, against a 10x bar.  This is defect 58 in CONTRIBUTING.md,
    # and the fix is the same one the other thirty scripts already carry.
    np.random.seed(0)
    set_ideal()
    set_diffusion_defaults(level="none")
    added = install_scale_archs()
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    archs = [a for a in args.archs.split(",") if a.strip()]

    rep = Report("ext_25_budget_matched.txt")
    for line in header("ext_25 - fix the budget, vary only N",
                       f"task={args.task} seed={args.seed} epochs={EPOCHS} "
                       f"budgets={budgets} archs={archs}"):
        rep(line)

    rep("")
    rep("  WHY: ext_24 could not answer section 11's tail because `effective_budget`")
    rep("  derives the per-epoch budget from the architecture, so raising N raised the")
    rep("  budget too (600 -> 1200 -> 2500) and moved the hybrid curve's FIRST point")
    rep("  (1200 -> 2400 -> 5000) past the 3000 grid point.  At N=208 the low-budget")
    rep("  comparison was UNDEFINED, not bad.  Passing an explicit budget fixes it:")
    rep("  the first point is then identical for every N.")
    rep("")
    rep("  AND IT INTRODUCES A RISK: a fixed budget can be too small for a bigger network.")
    rep("  `budget` is probes per epoch and each probe costs 2 forward passes, so a full")
    rep("  sweep of an N-particle layer needs 2*(D+2)*N forward passes -- at budget 600 an")
    rep("  N=208 layer manages 0.24 sweeps/epoch.  Section 7 of the base reports exactly")
    rep(f"  this situation as INCONCLUSIVE.  So every cell is checked against the {FLOOR}")
    rep("  learning floor and a cell that fails it is NOT counted as a win.")
    rep(f"\n  runtime archs added: {added if added else '(already installed)'}")

    X_tr, y_tr, X_te, y_te = make_regression_data(args.task, seed=42)
    t0 = time.time()

    rep("\n" + "=" * 100)
    rep("  P1  N x FIXED budget, one task, one seed")
    rep("=" * 100)
    rep("")
    rep(f"  {'budget':>7} {'arch':<6} {'N':>4} {'sweeps/ep':>10} {'frontier':>9} "
        f"{'best search':>12} {'hybrid':>10} {'win':>5} {'@3000 hybrid':>13} "
        f"{'@3000 best':>11} {'@3000 win':>10} {'1st m':>7} {'floor':>7}")
    rep("  " + "-" * 118)
    out = {}
    sweeps_per_epoch = {}
    for budget in budgets:
        for arch in archs:
            N = SCALE_ARCHS[arch]["particles"][0]
            curves = {}
            for arm in ARMS:
                info = run_cell_fixed(arch, arm, args.seed, args.task, X_tr, y_tr,
                                      X_te, y_te, budget)
                curves[arm] = info["curve"]
                print(f"  [P1] b={budget:<5} {arch}/{arm:<9} cache={info['from_cache']} "
                      f"pts={len(info['curve'])}", flush=True)
            r = frontier_and_rate(curves, low_budget=LOW_BUDGET)
            r["N"] = N
            r["budget"] = budget
            r["first_m"] = int(curves["hybrid"][0][0]) if curves["hybrid"] else -1
            # a full sweep of an N-particle layer, in forward passes
            n_params = (X_tr.shape[1] + 2) * N
            r["sweeps"] = budget * 2.0 / max(n_params, 1)
            r["floor_ok"] = r["best"] > FLOOR
            out[(budget, arch)] = r
            sweeps_per_epoch[(budget, arch)] = r["sweeps"]
            rep(f"  {budget:>7} {arch:<6} {N:>4} {r['sweeps']:>10.2f} "
                f"{r['frontier']:>9} {r['best']:>+12.4f} {r['hybrid']:>+10.4f} "
                f"{'YES' if r['win'] else 'no':>5} {r['low_hybrid']:>+13.4f} "
                f"{r['low_best']:>+11.4f} {'YES' if r['low_win'] else 'no':>10} "
                f"{r['first_m']:>7} {'ok' if r['floor_ok'] else 'LOW':>7}")
        rep("")

    rep("  'sweeps/ep' = forward passes spent per epoch / forward passes a full sweep of")
    rep("  that layer needs.  'floor' is whether the BEST search arm clears the learning")
    rep("  floor: a row marked LOW is uninformative and is not counted as a win.")

    # ---- verdict ---------------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  VERDICT: with the budget fixed, does the low-budget advantage survive N?")
    rep("=" * 100)
    rep("")
    for budget in budgets:
        rows = [out[(budget, a)] for a in archs]
        rep(f"  budget {budget}:")
        rep(f"    first curve point for every N: "
            f"{sorted(set(r['first_m'] for r in rows))}  "
            f"-> @{LOW_BUDGET} "
            f"{'DEFINED for all N' if all(r['first_m'] < LOW_BUDGET for r in rows) else 'undefined somewhere'}")
        rep(f"    sweeps/epoch: "
            + ", ".join(f"N={r['N']} {r['sweeps']:.2f}" for r in rows))
        rep(f"    cells clearing the floor: "
            f"{sum(int(r['floor_ok']) for r in rows)}/{len(rows)}")
        rep(f"    @{LOW_BUDGET} hybrid leads at: "
            + ", ".join(f"N={r['N']} {'YES' if r['low_win'] else 'NO'}" for r in rows))
        rep(f"    at the frontier it leads at: "
            + ", ".join(f"N={r['N']} {'YES' if r['win'] else 'NO'}" for r in rows))
        rep("")
    # the verdict, on the cells that can carry one
    pairs = []
    for budget in budgets:
        usable = [a for a in archs if out[(budget, a)]["floor_ok"]]
        if len(usable) >= 2:
            lo = out[(budget, usable[0])]
            hi = out[(budget, usable[-1])]
            pairs.append((budget, lo, hi))
    if not pairs:
        rep("  NO BUDGET LEFT A CELL ABOVE THE FLOOR AT MORE THAN ONE N, so this design")
        rep("  also cannot answer the question -- and now for the reason section 7 names:")
        rep("  too few probes per epoch.  Raising the fixed budget is the only move left.")
    else:
        for budget, lo, hi in pairs:
            rep(f"  budget {budget}: within the cells that clear the floor, the @{LOW_BUDGET}")
            rep(f"    margin goes N={lo['N']} {lo['low_hybrid']-lo['low_best']:+.4f} -> "
                f"N={hi['N']} {hi['low_hybrid']-hi['low_best']:+.4f}")
        rep("")
        allwin = all(out[(b, a)]["low_win"] for b, _lo, _hi in pairs for a in archs
                     if out[(b, a)]["floor_ok"])
        if allwin:
            rep("  THE ADVANTAGE SURVIVES, on every cell that can carry a verdict.  So")
            rep("  ext_24's apparent collapse at N=104 was an artefact of the budget moving")
            rep("  with N -- which is what the confound it priced would predict, and this")
            rep("  design removes it.")
        else:
            rep("  THE ADVANTAGE DOES NOT SURVIVE at every usable cell, and the cells where")
            rep("  it fails are named above.  This is the answer ext_24 could not give, and")
            rep("  it is a real one because the budget is now held fixed.")
    rep("")
    rep("  SCOPE: one task, one seed, single-layer matrix-free architectures, and 40")
    rep("  epochs.  The scaled family is not the base's own (whose families are")
    rep("  three-layer at the same total particle count), so this bounds the claim for")
    rep("  THIS family and the base's tail still needs its own budget-matched run.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

    path = rep.save()
    print(f"\nReport saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, '/')}")

    plt = try_plots()
    if plt is not None:
        __import__("warnings").filterwarnings(
            "ignore", message=".*does not have a glyph.*")
        plt.rcParams["axes.unicode_minus"] = False
        try:
            import logging as _lg
            _lg.getLogger("matplotlib").setLevel(_lg.ERROR)
        except Exception:
            pass
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
        for budget in budgets:
            xs = [out[(budget, a)]["N"] for a in archs]
            ax[0].plot(xs, [out[(budget, a)]["low_hybrid"] for a in archs], "o-",
                       label=f"hybrid @{LOW_BUDGET}, b={budget}")
            ax[0].plot(xs, [out[(budget, a)]["low_best"] for a in archs], "s--",
                       label=f"best search @{LOW_BUDGET}, b={budget}")
        ax[0].set_xscale("log", base=2)
        ax[0].set_xticks([out[(budgets[0], a)]["N"] for a in archs])
        ax[0].set_xticklabels([str(out[(budgets[0], a)]["N"]) for a in archs])
        ax[0].set_xlabel("particles N (budget FIXED per series)")
        ax[0].set_ylabel("R^2 at the low budget")
        ax[0].set_title("the comparison ext_24 could not make")
        ax[0].grid(True, alpha=0.25)
        ax[0].legend(fontsize=7)
        x = np.arange(len(archs))
        for i, budget in enumerate(budgets):
            ax[1].bar(x + (i - 0.5) * 0.4,
                      [out[(budget, a)]["low_hybrid"] - out[(budget, a)]["low_best"]
                       for a in archs], 0.4, label=f"budget {budget}")
        ax[1].axhline(0.0, color="k", lw=0.9)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([f"N={out[(budgets[0], a)]['N']}" for a in archs])
        ax[1].set_ylabel("hybrid - best search, at the low budget")
        ax[1].set_title("the margin, budget held fixed")
        ax[1].grid(True, axis="y", alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_25_budget_matched_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, '/')}")

    # The assert encodes what the measurement supports: every (N, budget) cell produced
    # four curves, a frontier and a low-budget comparison, and the floor was evaluated.
    # It cannot encode "the advantage survives", because that is the result.
    ok = all(len(out[(b, a)]) > 0 and out[(b, a)]["frontier"] > 0
             for b in budgets for a in archs)
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
