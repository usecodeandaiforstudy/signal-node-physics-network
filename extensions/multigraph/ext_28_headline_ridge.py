"""
ext_28 -- is the BASE's own headline also an artefact of the fixed ridge scale?
=============================================================================
Project: singal-node-physics-network, extension `multigraph`

THE QUESTION
------------
`ext_27` (section 12.12) found that a low-budget comparison is decided by the ridge scale:
at `lam="auto"` the hybrid's `@3000` margin is positive at every particle count in two
families, while the base's default `closed_form_lam=1e-3` is the ONLY value on a five-point
grid that collapses it.  And:

> `abl_06_task_sweep.py` never passes `closed_form_lam`, so **every cell of the base's task
> sweep -- and therefore `9/14`, `8/14` and `6/14` alike -- was measured at a FIXED
> `1e-3`.**

This asks whether the base's own headline moves under `auto`.  It is the only measurement
in this audit that could change the base version's central number.

WHAT IS RE-RUN, AND WHAT IS REUSED
-----------------------------------
`closed_form_lam` reaches only the HYBRID arm -- the calibration.  The other three arms
(geometry, weights, full) never call `solve_readout_closed_form`, so their curves are
independent of it and are REUSED from the base's own cache.  Only the hybrid arm is
re-trained, which is 14 cells x 9 seeds rather than 56 x 9.

**NOTHING IS WRITTEN TO THE BASE'S CACHE.**  `cache_path(...)` has no lambda in its key, so
saving a `lam="auto"` result there would OVERWRITE the base's `1e-3` cell for the same
(task, arch, arm, seed) -- destroying the evidence the base's published numbers rest on.
The re-runs are therefore computed and discarded, and the cache is opened READ-ONLY.

WHAT IS REPORTED
----------------
  P0  FAITHFULNESS: the base's cached curves, read back through this script's arithmetic,
      must reproduce the published `_s9` Part 2 row cell by cell.  Without it a difference
      under `auto` could be this script rather than the ridge scale.
  P1  the 14 cells at `lam=1e-3` and at `lam="auto"`, both criteria: the base's own
      `max(geometry, weights, full)`, and the corrected `max(weights, full)` of section
      12.7 -- because 12.7 showed the first can credit the hybrid for beating a component
      of itself.

SCOPE: the base's seven tasks and two architectures, 9 seeds (matching `_s9`), 40 epochs,
the diffusion substrate, and the base's own `effective_budget`.  Toy scale throughout.

Output: outputs/ext_28_headline_ridge.txt + _plot.png
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

from abl_common import (                                 # noqa: E402
    Report, build_model, cache_path, effective_budget, header, load_model,
    make_regression_data, merge_curves, set_diffusion_defaults, set_ideal,
    train_model, try_plots,
)

TASKS = ("2d", "4d", "10d", "moons", "steps", "hetero", "shift")
ARCHS = ("G", "F2")
ARMS = ("geometry", "weights", "full", "hybrid")
SEARCH = ("geometry", "weights", "full")
SEARCH_CORRECTED = ("weights", "full")
EPOCHS = 40
SUBSTRATE = "diffusion"
LOW_BUDGET = 3000
SEEDS = 9
LAM_BASE = 1e-3
LAM_AUTO = "auto"


def cached_curves(task, arch, seed, budget, X_ref):
    """
    The base's own cached curves, READ ONLY, for the three arms that do not calibrate.

    Returns None for the hybrid: its cached curve is the `lam=1e-3` one and is read back
    only in P0, never reused for the `auto` comparison.
    """
    out = {}
    for arm in SEARCH:
        p = cache_path(arch, arm, "none", seed, task, EPOCHS, budget, SUBSTRATE)
        if not os.path.exists(p):
            return None
        try:
            d = np.load(p)
            out[arm] = list(zip(d["curve_m"].tolist(), d["curve_r2"].tolist()))
        except Exception:
            return None
    return out


def cached_hybrid(task, arch, seed, budget, X_ref):
    """The base's `lam=1e-3` hybrid cell, for P0 only."""
    p = cache_path(arch, "hybrid", "none", seed, task, EPOCHS, budget, SUBSTRATE)
    if not os.path.exists(p):
        return None
    try:
        d = np.load(p)
        return list(zip(d["curve_m"].tolist(), d["curve_r2"].tolist()))
    except Exception:
        return None


def train_hybrid(arch, X_tr, y_tr, X_te, y_te, budget, seed, lam):
    """
    Re-train ONLY the hybrid arm at the requested ridge scale.  NOT CACHED -- see above.
    """
    model = build_model(arch, X_tr, seed=seed, substrate=SUBSTRATE)
    hist = train_model(model, X_tr, y_tr, X_te, y_te, arm="hybrid", epochs=EPOCHS,
                       budget=int(budget), seed=seed, probe_batch=96,
                       substrate=SUBSTRATE, eval_every=1, eval_X=X_te, eval_y=y_te,
                       closed_form_lam=lam)
    return hist["curve"]


def cell(curves, criterion=SEARCH, low=LOW_BUDGET):
    """The base's Part 2/Part 3 arithmetic, with the comparison set selectable."""
    ms = [c[-1][0] for c in curves.values() if c]
    if not ms:
        return None
    fr = int(min(ms))
    vals = {}
    for arm, c in curves.items():
        mu, _s, _n = merge_curves([c], [fr])
        vals[arm] = float(mu[0]) if np.isfinite(mu[0]) else float("nan")
    best = max(vals[a] for a in criterion)
    which = max(criterion, key=lambda a: vals[a])
    lo = {}
    for arm, c in curves.items():
        mu, _s, _n = merge_curves([c], [low])
        lo[arm] = float(mu[0]) if np.isfinite(mu[0]) else float("nan")
    lo_best = max(lo[a] for a in criterion)
    return dict(frontier=fr, vals=vals, best=best, which=which,
                hybrid=vals["hybrid"], win=vals["hybrid"] > best,
                margin=vals["hybrid"] - best,
                low_hybrid=lo["hybrid"], low_best=lo_best,
                low_win=lo["hybrid"] > lo_best, low_margin=lo["hybrid"] - lo_best)


def _aggregate(agg, seeds, criterion):
    """The base's own Part 2/Part 3 aggregation, on per-seed curves."""
    ms = [agg[arm][s][-1][0] for arm in ARMS for s in range(seeds)]
    fr = int(min(ms))
    vals, lo = {}, {}
    for arm in ARMS:
        hi, li = [], []
        for s in range(seeds):
            mu, _a, _b = merge_curves([agg[arm][s]], [fr])
            hi.append(float(mu[0]))
            mu2, _a2, _b2 = merge_curves([agg[arm][s]], [LOW_BUDGET])
            li.append(float(mu2[0]))
        vals[arm] = float(np.mean(hi))
        lo[arm] = float(np.mean(li))
    best = max(vals[a] for a in criterion)
    which = max(criterion, key=lambda a: vals[a])
    lo_best = max(lo[a] for a in criterion)
    return dict(frontier=fr, best=best, which=which, hybrid=vals["hybrid"],
                margin=vals["hybrid"] - best, win=vals["hybrid"] > best,
                low_hybrid=lo["hybrid"], low_best=lo_best,
                low_margin=lo["hybrid"] - lo_best, low_win=lo["hybrid"] > lo_best,
                vals=vals)


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_28: the base headline under lam=auto")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert
    tasks = [t for t in args.tasks.split(",") if t.strip()]

    set_ideal()
    set_diffusion_defaults(level="none")
    rep = Report("ext_28_headline_ridge.txt")
    for line in header("ext_28 - is the BASE's headline an artefact of lam=1e-3?",
                       f"tasks={len(tasks)} archs={list(ARCHS)} seeds={args.seeds} "
                       f"epochs={EPOCHS} substrate={SUBSTRATE}"):
        rep(line)

    rep("")
    rep("  `ext_27` found a low-budget comparison decided by the ridge scale, and that")
    rep("  `abl_06_task_sweep.py` NEVER passes `closed_form_lam` -- so the base's whole")
    rep("  task sweep, and `9/14` / `8/14` / `6/14` with it, was measured at a fixed 1e-3.")
    rep("")
    rep("  Only the HYBRID arm touches that parameter; the other three never call")
    rep("  `solve_readout_closed_form`, so their curves are independent of it and are")
    rep("  REUSED from the base's cache.  Only 14 x seeds hybrid cells are re-trained.")
    rep("")
    rep("  **NOTHING IS WRITTEN TO THE BASE'S CACHE.**  Its key carries no lambda, so")
    rep("  saving an `auto` cell would overwrite the base's 1e-3 cell for the same")
    rep("  (task, arch, arm, seed) and destroy the evidence its numbers rest on.")

    t0 = time.time()

    # ---- load the cached arms and reproduce the published row (P0) -------
    data = {}
    for tk in tasks:
        X_tr, y_tr, X_te, y_te = make_regression_data(tk, seed=42)
        for a in ARCHS:
            bd, _sz = effective_budget(a, X_tr, list(ARMS), None, substrate=SUBSTRATE)
            budget = int(bd["full"])
            data[(tk, a)] = dict(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                                 budget=budget)

    rep("\n" + "=" * 100)
    rep("  P0  faithfulness: the base's cached curves through this script's arithmetic")
    rep("=" * 100)
    rep("")
    # P0 compares against a NINE-SEED published mean, so a different seed count makes it
    # fail for a reason that has nothing to do with the arithmetic.  Stated up front
    # rather than discovered by watching 15 mismatches: an earlier smoke test ran with
    # --seeds 1 and looked like a broken runner.
    if args.seeds != 9:
        rep(f"  NOTE: P0 checks against the published NINE-SEED row, and this run uses")
        rep(f"  {args.seeds} seed(s), so mismatches below are EXPECTED and say nothing")
        rep("  about the arithmetic.  Run with --seeds 9 for a real P0.")
        rep("")
    rep(f"  {'cell':<12} {'arm':<9} {'this script':>12} {'published _s9':>14} {'agree':>7}")
    rep("  " + "-" * 58)
    ok_arms, tot_arms = 0, 0
    for tk in tasks:
        for a in ARCHS:
            d = data[(tk, a)]
            pub = published_row(os.path.join(ROOT, "outputs", "abl_06_task_sweep_s9.txt"),
                                tk, a)
            for arm in ARMS:
                # ALL FOUR, including the hybrid: it is the arm this script re-runs, so
                # the cached 1e-3 cell it replaces must be verified first.  An earlier
                # version checked only the three search arms.
                cs = []
                for s in range(args.seeds):
                    p = cache_path(a, arm, "none", s, tk, EPOCHS, d["budget"], SUBSTRATE)
                    if not os.path.exists(p):
                        cs = None
                        break
                    try:
                        z = np.load(p)
                        cs.append(list(zip(z["curve_m"].tolist(),
                                           z["curve_r2"].tolist())))
                    except Exception:
                        cs = None
                        break
                if cs is None or arm not in pub:
                    continue
                mu, _s, _n = merge_curves(cs, [min(c[-1][0] for c in cs)])
                mine = float(mu[0])
                tot_arms += 1
                agree = abs(mine - pub[arm]) < 5e-4
                ok_arms += int(agree)
                if not agree:
                    rep(f"  {tk+'/'+a:<12} {arm:<9} {mine:>+12.4f} {pub[arm]:>+14.4f} "
                        f"{'NO':>7}")
    rep(f"  arms checked: {tot_arms}, reproduced: {ok_arms}")
    rep("  (mismatches are listed individually; silence above means none)")

    # ---- P1: the hybrid at both ridge scales -----------------------------
    rep("\n" + "=" * 100)
    rep("  P1  the 14 cells at lam=1e-3 and lam=auto (hybrid re-trained, 3 arms reused)")
    rep("=" * 100)
    rep("")
    rep(f"  {'cell':<12} {'lam':>6} {'best search':>12} {'hybrid':>10} {'margin':>9} "
        f"{'win':>5} | {'excl geom':>10} {'margin':>9} {'win':>5}")
    rep("  " + "-" * 96)
    res = {}
    for tk in tasks:
        for a in ARCHS:
            d = data[(tk, a)]
            for lam, tag in ((LAM_BASE, "1e-3"), (LAM_AUTO, "auto")):
                if lam == LAM_BASE:
                    # THE BASE'S OWN CELLS represent the base's own lambda -- verified by
                    # P0 above -- so re-training this side would cost half the run for the
                    # same numbers, and would substitute my run for the published one.
                    hyb = []
                    for s in range(args.seeds):
                        p = cache_path(a, "hybrid", "none", s, tk, EPOCHS,
                                       d["budget"], SUBSTRATE)
                        z = np.load(p)
                        hyb.append(list(zip(z["curve_m"].tolist(),
                                            z["curve_r2"].tolist())))
                else:
                    hyb = []
                    for s in range(args.seeds):
                        hyb.append(train_hybrid(a, d["X_tr"], d["y_tr"], d["X_te"],
                                                d["y_te"], d["budget"], s, lam))
                agg = {}
                for arm in SEARCH:
                    cs = []
                    for s in range(args.seeds):
                        z = np.load(cache_path(a, arm, "none", s, tk, EPOCHS,
                                               d["budget"], SUBSTRATE))
                        cs.append(list(zip(z["curve_m"].tolist(),
                                           z["curve_r2"].tolist())))
                    agg[arm] = cs
                agg["hybrid"] = hyb
                # THE BASE'S OWN AGGREGATION, which is what `abl_06` does:
                #   vals[arm] = mean across seeds of that arm's value AT THE FRONTIER
                #   best      = max over the comparison set OF THOSE MEANS
                #   margin    = vals["hybrid"] - best
                # An earlier version took max PER SEED and then averaged the margins,
                # which is a different quantity: it makes `best search` systematically
                # higher (2d/G 0.6982 against the published 0.6692), and while it also
                # happened to give 8/14, a coinciding count is not the same criterion.
                m = _aggregate(agg, args.seeds, SEARCH)
                mc = _aggregate(agg, args.seeds, SEARCH_CORRECTED)
                res[(tk, a, tag)] = (m, mc)
                rep(f"  {tk+'/'+a:<12} {tag:>6} {m['best']:>+12.4f} "
                    f"{m['hybrid']:>+10.4f} {m['margin']:>+9.4f} "
                    f"{'YES' if m['win'] else 'no':>5} | {mc['best']:>+10.4f} "
                    f"{mc['margin']:>+9.4f} {'YES' if mc['win'] else 'no':>5}")
                print(f"  [P1] {tk}/{a} lam={tag} margin {m['margin']:+.4f}", flush=True)
        rep("")

    # ---- verdict ---------------------------------------------------------
    rep("=" * 100)
    rep("  VERDICT: does the base's headline move under lam=auto?")
    rep("=" * 100)
    rep("")
    for crit, key in (("base criterion", "win"), ("geometry excluded", None)):
        for tag in ("1e-3", "auto"):
            n = sum(int(res[k][0]["win"] if key else res[k][1]["win"])
                    for k in res if k[2] == tag)
            tot = sum(1 for k in res if k[2] == tag)
            rep(f"  {crit:<20} lam={tag:<5}: {n}/{tot} cells won")
        rep("")
    for tag in ("1e-3", "auto"):
        flip = [k[:2] for k in res if k[2] == tag
                and res[k][0]["win"] != res[(k[0], k[1], "auto" if tag == "1e-3" else "1e-3")][0]["win"]]
        rep(f"  cells whose base-criterion verdict differs between the two lambdas: "
            f"{len(set(flip))}")
        for c in sorted(set(flip)):
            a1 = res[(c[0], c[1], "1e-3")][0]
            a2 = res[(c[0], c[1], "auto")][0]
            rep(f"    {c[0]}/{c[1]}: margin {a1['margin']:+.4f} -> {a2['margin']:+.4f}")
    rep("")
    n_base = sum(int(res[k][0]["win"]) for k in res if k[2] == "1e-3")
    n_auto = sum(int(res[k][0]["win"]) for k in res if k[2] == "auto")
    rep(f"  THE HEADLINE: {n_base}/14 at the base's own lam, {n_auto}/14 at auto.")
    rep("")
    rep("  SCOPE: the base's seven tasks and two architectures, the diffusion substrate,")
    rep("  the base's own effective_budget, 40 epochs, and the seed count printed above.")
    rep("  The lambda grid of ext_27 was five points; `auto` is one of them, and it was")
    rep("  the one that worked in both families there.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

    path = rep.save()
    print(f"\nReport saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

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
        cells = sorted({(k[0], k[1]) for k in res})
        x = np.arange(len(cells))
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 4.4))
        ax.bar(x - 0.2, [res[(t, a, "1e-3")][0]["margin"] for t, a in cells], 0.4,
               label="lam=1e-3 (the base's default)", color="#2b7bba")
        ax.bar(x + 0.2, [res[(t, a, "auto")][0]["margin"] for t, a in cells], 0.4,
               label="lam=auto", color="#c44e52")
        ax.axhline(0.0, color="k", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}/{a}" for t, a in cells], rotation=45, ha="right",
                           fontsize=8)
        ax.set_ylabel("hybrid - best search, at the frontier")
        ax.set_title("the base's 14 cells under two ridge scales")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_28_headline_ridge_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

    if do_assert and not (ok_arms == tot_arms and tot_arms > 0):
        raise SystemExit(1)
    return 0



def published_row(path, task, arch):
    import re as _re
    row = _re.compile(r"^\s+(\w+)\s+(G|F2)\s+(\d+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s+"
                      r"([-+][\d.]+)\s+([-+][\d.]+)\s*$")
    out, in_p2 = {}, False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "Part 2: accuracy at the COMMON MEASUREMENT FRONTIER" in line:
                in_p2 = True
                continue
            if in_p2 and out and line.startswith("===="):
                break
            if in_p2:
                m = row.match(line.rstrip("\n"))
                if m and m.group(1) == task and m.group(2) == arch:
                    out = {"geometry": float(m.group(4)), "weights": float(m.group(5)),
                           "full": float(m.group(6)), "hybrid": float(m.group(7))}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
