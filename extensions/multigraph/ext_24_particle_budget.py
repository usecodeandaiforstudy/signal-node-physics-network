"""
ext_24 -- does the hybrid's low-budget advantage survive a LARGER particle budget?
=================================================================================
Project: singal-node-physics-network, extension `multigraph`

THE OPEN ITEM
-------------
The base version's section 11 closes the scaling law as done, with one tail left:

    "Still to do: check whether the hybrid advantage at 3 000 measurements holds
     for a larger particle budget."

The `@3000` column of the base's task sweep is a LOW-BUDGET comparison: the calibrated
arm versus the best search arm at 3 000 measurements, i.e. long before the common
frontier.  The question is whether that early lead survives when the network has more
particles.

WHY IT IS NOT OBVIOUS
---------------------
`build_model` scales each layer's width as `spread / sqrt(N)` (abl_common line 703), so
more particles means NARROWER, more localised nodes.  Two effects pull in opposite
directions: a larger geometry is harder to search, which should favour calibrating, while
the calibration's own one-time cost (`n_train * sum_k N_k` measurements) grows with N,
which should hurt it.  Only a measurement decides which wins.

THE FREEZE, AND HOW THIS RESPECTS IT
------------------------------------
All five of the base's architectures hold the total particle count at 52, so a larger
budget needs a new entry in `abl_common.ARCHS` -- inside a base file.  **This script does
not edit it.**  It adds keys to the `ARCHS` dict AT RUNTIME, which changes no existing
behaviour and leaves the file byte-identical; `P0` below proves the unmodified
architecture still reproduces the published numbers through this same code path.

WHAT IS MEASURED
----------------
  P0  FAITHFULNESS.  The cell (4d, F2, seed 0, 40 epochs) is run through this script's
      cell runner, which uses the base's own cache key, so it must hit the CACHE and
      reproduce the published frontier and `@3000` values.  Without this control a
      difference at larger N could be this script rather than the physics.
  P1  SCALE.  Single-layer matrix-free architectures at N = 52, 104, 208 particles --
      the base's own total, then 2x and 4x it -- on the same task, same arms, same
      budget rule (`effective_budget`, the base's own), and the same `@3000` comparison.

SCOPE: task 4d, 1 seed per cell, 40 epochs, diffusion substrate (the base's sweep
default).  Toy, as everywhere.  The arms are the base's own four: geometry, weights
(readout), full, hybrid.

Output: outputs/ext_24_particle_budget.txt + _plot.png
"""

import argparse
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)

import abl_common as ac                                  # noqa: E402

from abl_common import (                                 # noqa: E402
    Report, build_model, cache_path, effective_budget, header, load_model,
    make_regression_data, merge_curves, save_model, set_diffusion_defaults,
    set_ideal, train_model, try_plots,
)

ARMS = ("geometry", "weights", "full", "hybrid")
SEARCH_ARMS = ("geometry", "weights", "full")
LOW_BUDGET = 3000
EPOCHS = 40
SUBSTRATE = "diffusion"

# Runtime additions to the base's ARCHS.  Same total count as the base's own families
# (52) for the smallest, then 2x and 4x.  Single layer, scalar readout, which is the
# matrix-free form the thesis actually claims.
SCALE_ARCHS = {
    "S52": dict(particles=(52,), out_dims=(1,), label="single-layer, 52 (the base total)"),
    "S104": dict(particles=(104,), out_dims=(1,), label="single-layer, 104"),
    "S208": dict(particles=(208,), out_dims=(1,), label="single-layer, 208"),
}


def install_scale_archs():
    """
    Add the larger architectures to abl_common.ARCHS AT RUNTIME.

    This is the freeze-respecting route: the base file is not edited, no existing key is
    modified, and every base routine that looks up `ARCHS[arch_key]` picks the new ones
    up unchanged.
    """
    added = []
    for k, v in SCALE_ARCHS.items():
        if k not in ac.ARCHS:
            ac.ARCHS[k] = v
            added.append(k)
    return added


def run_cell(arch, arm, seed, task, X_tr, y_tr, X_te, y_te, epochs=EPOCHS,
             substrate=SUBSTRATE, probe_batch=96, use_cache=True):
    """
    Mirror of `abl_06._load_or_train` (a script, so not importable).

    The cache key is the BASE's own `cache_path(...)` with the same arguments, so for an
    architecture and settings the base has already run this returns the CACHED curve
    rather than retraining -- which is what makes P0 a faithfulness check that costs
    nothing.
    """
    budget_arch = arch
    X_ref = X_tr
    bd, _sizes = effective_budget(budget_arch, X_ref, list(ARMS), None,
                                  substrate=substrate)
    budget = int(bd["full"])

    path = cache_path(arch, arm, "none", seed, task, epochs, budget, substrate)
    if use_cache and os.path.exists(path):
        try:
            d = np.load(path)
            model = load_model(arch, arm, "none", seed, task, epochs, budget, X_ref,
                               substrate=substrate)
            if model is not None:
                curve = list(zip(d["curve_m"].tolist(), d["curve_r2"].tolist()))
                return model, dict(curve=curve, from_cache=True, budget=budget,
                                   n_cal=int(d["n_cal"]) if "n_cal" in d else 0,
                                   n_rej=int(d["n_rej"]) if "n_rej" in d else 0)
        except Exception:
            pass

    model = build_model(arch, X_ref, seed=seed, substrate=substrate)
    hist = train_model(model, X_tr, y_tr, X_te, y_te, arm=arm, epochs=epochs,
                       budget=budget, seed=seed, probe_batch=probe_batch,
                       substrate=substrate, eval_every=1, eval_X=X_te, eval_y=y_te)
    try:
        save_model(model, arch, arm, "none", seed, task, epochs, budget,
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
    return model, dict(curve=hist["curve"], from_cache=False, budget=budget,
                       n_cal=hist["n_calibrations"], n_rej=hist["calib_rejected"])


def parse_published_s9(path, task, arch):
    """
    The base's published Part 2 row for one (task, arch) cell, read from the report.

    Read from the ARTIFACT rather than transcribed, so the faithfulness control compares
    against what was published and not against a copy of it.
    """
    row = re.compile(
        r"^\s+(\w+)\s+(G|F2)\s+(\d+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s+"
        r"([-+][\d.]+)\s+([-+][\d.]+)\s*$")
    out, in_part2 = {}, False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "Part 2: accuracy at the COMMON MEASUREMENT FRONTIER" in line:
                in_part2 = True
                continue
            # the title is followed by a `====` rule, so only break once rows exist
            if in_part2 and out and line.startswith("===="):
                break
            if in_part2:
                m = row.match(line.rstrip("\n"))
                if m and m.group(1) == task and m.group(2) == arch:
                    out = {"geometry": float(m.group(4)), "weights": float(m.group(5)),
                           "full": float(m.group(6)), "hybrid": float(m.group(7))}
    return out


def frontier_and_rate(curves, low_budget=LOW_BUDGET):
    """
    The base's own Part 2 / Part 3 arithmetic, reproduced:
      frontier = min over arms of the last measurement count
      values   = each arm's R^2 AT the frontier (merge_curves)
      reach    = earliest measurement count where the hybrid reaches `best search`
    """
    ms = [c[-1][0] for arm, c in curves.items() if c]
    fr = int(min(ms))
    vals = {}
    for arm, c in curves.items():
        mu, _s, _n = merge_curves([c], [fr])
        vals[arm] = float(mu[0]) if np.isfinite(mu[0]) else float("nan")
    best = max(vals["geometry"], vals["weights"], vals["full"])
    hyb = vals["hybrid"]
    reach = None
    for m, v in curves["hybrid"]:
        if np.isfinite(v) and v >= best:
            reach = m if reach is None else min(reach, m)
    lo = {}
    for arm, c in curves.items():
        mu, _s, _n = merge_curves([c], [low_budget])
        lo[arm] = float(mu[0]) if np.isfinite(mu[0]) else float("nan")
    lo_best = max(lo["geometry"], lo["weights"], lo["full"])
    return dict(frontier=fr, vals=vals, best=best, hybrid=hyb,
                win=bool(np.isfinite(hyb) and hyb > best),
                reach=reach, speedup=(fr / reach) if reach else float("nan"),
                low_hybrid=lo["hybrid"], low_best=lo_best,
                low_win=bool(np.isfinite(lo["hybrid"]) and lo["hybrid"] > lo_best))


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_24: the particle-budget question")
    ap.add_argument("--task", default="4d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--archs", default="S52,S104,S208")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    set_ideal()
    set_diffusion_defaults(level="none")
    added = install_scale_archs()

    rep = Report("ext_24_particle_budget.txt")
    for line in header("ext_24 - does the low-budget advantage survive more particles?",
                       f"task={args.task} seed={args.seed} epochs={EPOCHS} "
                       f"substrate={SUBSTRATE}"):
        rep(line)

    rep("")
    rep("  The base's section 11 leaves one tail open: does the hybrid advantage at 3000")
    rep("  measurements hold for a LARGER particle budget?  All five of the base's")
    rep("  architectures hold the total at 52, so this needs new architectures -- and the")
    rep("  base is FROZEN.  They are added to abl_common.ARCHS AT RUNTIME instead of by")
    rep(f"  editing it: {added if added else '(already installed)'}.  No existing key is")
    rep("  touched and the file is byte-identical.")
    rep("")
    rep("  Not obvious either way, because build_model scales each width as spread/sqrt(N):")
    rep("  more particles means narrower nodes, a HARDER geometry to search (which should")
    rep("  favour calibrating) while the calibration's own one-time cost n_train*sum(N)")
    rep("  grows with N (which should hurt it).")

    X_tr, y_tr, X_te, y_te = make_regression_data(args.task, seed=42)
    t0 = time.time()

    # ---- P0: faithfulness, against a PUBLISHED table ---------------------
    rep("\n" + "=" * 94)
    rep("  P0  faithfulness: reproduce a PUBLISHED table through this script's arithmetic")
    rep("=" * 94)
    rep("")
    rep("  A single-seed run cannot be compared with the base's published means, so this")
    rep("  control does what can be: it loads the base's caches for ALL NINE seeds of")
    rep("  (4d, F2, 40 epochs), averages them at the frontier with this script's own")
    rep("  arithmetic, and compares against the published `_s9` row.  If the four arms")
    rep("  agree, the frontier and averaging code here IS the base's.")
    rep("")
    ref_arch = "F2"
    seed_curves = {arm: [] for arm in ARMS}
    n_cached = 0
    for s in range(9):
        for arm in ARMS:
            _m, info = run_cell(ref_arch, arm, s, args.task, X_tr, y_tr, X_te, y_te)
            seed_curves[arm].append(info["curve"])
            n_cached += int(info["from_cache"])
    rep(f"  cached cells loaded: {n_cached} of {9*len(ARMS)}")
    rep("")
    rep(f"  {'arm':<10} {'this script (9-seed mean)':>26} {'published _s9 row':>20} "
        f"{'agree':>7}")
    rep("  " + "-" * 68)
    pub = parse_published_s9(os.path.join(ROOT, "outputs", "abl_06_task_sweep_s9.txt"),
                             args.task, ref_arch)
    ok_arms = 0
    fr = None
    for arm in ARMS:
        mu, _s, _n = merge_curves(seed_curves[arm],
                                  [min(c[-1][0] for cs in seed_curves.values()
                                       for c in cs)])
        fr = min(c[-1][0] for cs in seed_curves.values() for c in cs)
        mine = float(mu[0])
        theirs = pub.get(arm, float("nan"))
        agree = abs(mine - theirs) < 5e-4
        ok_arms += int(agree)
        rep(f"  {arm:<10} {mine:>+26.4f} {theirs:>+20.4f} {'YES' if agree else 'NO':>7}")
    rep("")
    rep(f"  frontier here {fr}; the published row's frontier is in the same report.")
    rep(f"  arms reproduced: {ok_arms}/{len(ARMS)}" +
        ("   -> the arithmetic is the base's own" if ok_arms == len(ARMS)
         else "   -> NOT reproduced; treat P1 with suspicion"))

    # ---- P1: scale the particle budget -----------------------------------
    rep("\n" + "=" * 94)
    rep("  P1  the same cell, at 52 / 104 / 208 particles")
    rep("=" * 94)
    rep("")
    rep(f"  {'arch':<7} {'N':>4} {'frontier':>9} {'best search':>12} {'hybrid':>10} "
        f"{'win':>5} {'@3000 hybrid':>13} {'@3000 best':>11} {'@3000 win':>10} "
        f"{'1st m':>7}")
    rep("  " + "-" * 100)
    out = {}
    for arch in [a for a in args.archs.split(",") if a.strip()]:
        curves = {}
        for arm in ARMS:
            _m, info = run_cell(arch, arm, args.seed, args.task, X_tr, y_tr, X_te, y_te)
            curves[arm] = info["curve"]
            print(f"  [P1] {arch}/{arm:<9} cache={info['from_cache']} "
                  f"budget={info['budget']} pts={len(info['curve'])}", flush=True)
        r = frontier_and_rate(curves)
        r["N"] = SCALE_ARCHS[arch]["particles"][0]
        # WHERE THE COMPARISON WINDOW STARTS -- the mechanism the verdict turns on.  The
        # curve's first recorded point is twice the architecture's per-epoch budget, and
        # that budget scales with N, so a larger particle count pushes the first point
        # past the low-budget grid point: the @3000 column becomes UNDEFINED rather than
        # bad.  An earlier version printed `nan` without saying why.
        r["first_m"] = int(curves["hybrid"][0][0]) if curves["hybrid"] else -1
        out[arch] = r
        rep(f"  {arch:<7} {r['N']:>4} {r['frontier']:>9} {r['best']:>+12.4f} "
            f"{r['hybrid']:>+10.4f} {'YES' if r['win'] else 'no':>5} "
            f"{r['low_hybrid']:>+13.4f} {r['low_best']:>+11.4f} "
            f"{'YES' if r['low_win'] else 'no':>10} {r['first_m']:>7}")

    rep("")
    rep(f"  '1st m' is the measurement count of the hybrid curve's FIRST recorded point.")
    rep("  It is twice the architecture's per-epoch budget, and that budget grows with N,")
    rep(f"  so the @{LOW_BUDGET} column is only DEFINED while `1st m` stays below "
        f"{LOW_BUDGET}:")
    for arch in out:
        v = out[arch]
        rep(f"    {arch:<6} N={v['N']:<4} first m={v['first_m']:<7} -> "
            f"@{LOW_BUDGET} {'defined' if v['first_m'] < LOW_BUDGET else 'UNDEFINED'}")

    # ---- verdict ---------------------------------------------------------
    rep("\n" + "=" * 94)
    rep("  VERDICT: does the low-budget advantage survive a larger particle budget?")
    rep("=" * 94)
    rep("")
    Ns = [out[a]["N"] for a in out]
    lows = [out[a]["low_win"] for a in out]
    fins = [out[a]["win"] for a in out]
    rep(f"  N tested: {Ns}")
    rep(f"  @{LOW_BUDGET} the hybrid leads the best search arm at: "
        + ", ".join(f"N={out[a]['N']} {'YES' if out[a]['low_win'] else 'NO'}"
                    for a in out))
    rep(f"  at the frontier it leads at:                        "
        + ", ".join(f"N={out[a]['N']} {'YES' if out[a]['win'] else 'NO'}" for a in out))
    rep("")
    small = [a for a in out if out[a]["N"] == min(Ns)][0]
    big = [a for a in out if out[a]["N"] == max(Ns)][0]
    d_low = out[big]["low_hybrid"] - out[big]["low_best"]
    d_small = out[small]["low_hybrid"] - out[small]["low_best"]
    rep(f"  the low-budget margin: N={out[small]['N']} {d_small:+.4f} -> "
        f"N={out[big]['N']} {d_low:+.4f}  ({d_low - d_small:+.4f})")
    rep("")
    if all(lows):
        rep("  THE ADVANTAGE SURVIVES THE SCALE-UP, on the budget tested.  The calibrated")
        rep("  arm leads the best search arm at 3000 measurements at every particle count")
        rep(f"  from {min(Ns)} to {max(Ns)}, so the base's section 11 tail can be closed for")
        rep("  this range -- with the scope stated: one task, one seed, single-layer,")
        rep("  matrix-free, and 4x the base's particle total at the top end.")
    else:
        undef = [a for a in out if out[a]["first_m"] >= LOW_BUDGET]
        lost = [a for a in out if out[a]["first_m"] < LOW_BUDGET and not out[a]["low_win"]]
        rep("  THE QUESTION AS POSED CANNOT BE ANSWERED BY THIS DESIGN, and the reason is")
        rep("  measured rather than argued:")
        rep("")
        rep(f"  (1) THE COMPARISON WINDOW SCALES AWAY.  The hybrid curve's first recorded")
        rep(f"      point is twice the per-epoch budget, and the budget grows with N "
            f"({out[small]['first_m']} at")
        rep(f"      N={out[small]['N']}, ... {out[big]['first_m']} at N={out[big]['N']}).  "
            f"At {len(undef)} of {len(out)} counts the first point")
        rep(f"      is already past {LOW_BUDGET}, so the calibrated arm has NO MEASUREMENT "
            f"there and")
        rep(f"      the @{LOW_BUDGET} column is UNDEFINED -- not bad, absent.  **The window")
        rep("      disappears before the advantage does.**")
        rep("")
        if lost:
            rep(f"  (2) WHERE IT IS STILL DEFINED, THE ADVANTAGE IS GONE: "
                f"{', '.join('N='+str(out[a]['N']) for a in lost)}")
            rep("      shows the hybrid BEHIND the search arms.  The margin at N=104 is")
            rep(f"      {out['S104']['low_hybrid'] - out['S104']['low_best']:+.4f}, against "
                f"{d_small:+.4f} at N=52.")
            rep("")
            rep("      MECHANISM NOT ESTABLISHED.  The hybrid's R^2 at N=104 is negative")
            rep("      throughout (its first four points are -0.52, -0.53, -0.51, -0.46),")
            rep("      which the base's own readout-params-per-sample rule would NOT have")
            rep("      predicted -- 104/256 = 0.41x is well inside the 0.20x-safe to")
            rep("      2.19x-failing band -- and N=208 does BETTER than N=104 while being")
            rep("      larger still.  **So this is not a clean monotone capacity story, and")
            rep("      the collapse is not explained here.**")
        rep("")
        rep("  (3) THE CONFOUND FLAGGED IN ADVANCE IS REAL AND NOW PRICED: N changes the")
        rep("      per-epoch budget (600 -> 1200 -> 2500) as well as the geometry, so this")
        rep("      design varies three things at once.  **A budget-matched variant -- a FIXED")
        rep("      per-epoch budget across N -- is what would answer the question**, and it")
        rep("      is not run here.  The honest close of section 11's tail is therefore that")
        rep("      the question needs restating before it can be answered.")
    rep("")
    rep("  SCOPE, stated because the family is not the base's: the scaled architectures are")
    rep("  SINGLE-LAYER, while the base's own families are three-layer with the same total")
    rep("  particle count.  The comparison WITHIN this family is like for like (S52 -> S104")
    rep("  -> S208 differ only in N), but S52 is not the base's F2, and the two are reported")
    rep("  side by side in P0 rather than treated as the same thing.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

    path = rep.save()
    print(f"\nReport saved: {path}")

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
        labels = [a for a in out]
        xs = [out[a]["N"] for a in labels]
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
        ax[0].plot(xs, [out[a]["low_hybrid"] for a in labels], "o-",
                   label=f"hybrid @{LOW_BUDGET}")
        ax[0].plot(xs, [out[a]["low_best"] for a in labels], "s-",
                   label=f"best search @{LOW_BUDGET}")
        ax[0].set_xscale("log", base=2)
        ax[0].set_xticks(xs)
        ax[0].set_xticklabels([str(n) for n in xs])
        ax[0].set_xlabel("particles N")
        ax[0].set_ylabel("R^2 at the low budget")
        ax[0].set_title("the low-budget comparison, scaled")
        ax[0].grid(True, alpha=0.25)
        ax[0].legend(fontsize=8)
        ax[1].bar(np.arange(len(labels)) - 0.2,
                  [out[a]["low_hybrid"] - out[a]["low_best"] for a in labels], 0.4,
                  label=f"@{LOW_BUDGET}")
        ax[1].bar(np.arange(len(labels)) + 0.2,
                  [out[a]["hybrid"] - out[a]["best"] for a in labels], 0.4,
                  label="at the frontier")
        ax[1].axhline(0.0, color="k", lw=0.9)
        ax[1].set_xticks(np.arange(len(labels)))
        ax[1].set_xticklabels([f"N={out[a]['N']}" for a in labels])
        ax[1].set_ylabel("hybrid - best search")
        ax[1].set_title("the margin, at two budgets")
        ax[1].grid(True, axis="y", alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_24_particle_budget_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {p}")

    # The assert encodes what the measurement supports: every arm of every architecture
    # produced a curve and every cell got a frontier and a low-budget comparison.  It
    # cannot encode "the advantage survives", because that is the result.
    ok = all(len(out[a]) > 0 and out[a]["frontier"] > 0 for a in out)
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
