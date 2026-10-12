"""
ext_33 -- does the do-no-harm gate have STATISTICAL POWER at its own boundary?
=============================================================================
Project: singal-node-physics-network, extension `multigraph`

THE OPEN ITEM, IN THE BASE'S OWN WORDS (section 6.11)
-----------------------------------------------------
> "The gate guarantees that deploying a calibration never leaves the arm worse than not
>  calibrating.  The strongest missing evidence against the whole rule would be a cell
>  where the gate **accepts** and the arm still loses.  Three designs were tried ... the
>  third attempt **did not answer its own question** (a first-order failure is detectable
>  at any slice size, so it does not probe the gate's statistical power -- **that needs a
>  task sitting near the decision boundary, which was not constructed**)."

**That task is constructed here.**  It is the last item on the base version's list that is
real science rather than a documentation choice, and it probes the strongest guarantee in
the project: the rule "calibrate the source strengths, spend every measurement on the
geometry" rests on it.

THE GATE, EXACTLY (`abl_common.solve_readout_closed_form`)
----------------------------------------------------------
    val[::max(int(guard_holdout), 2)] = True      # a STRIDE, not a random subset
    fit on the complement; score the fitted W on `val`; restore; score the in-place W
    keep  <=>  r_new > r_base   AND   r_new > guard_floor

So the decision is a **strict inequality on a finite sample** -- which is precisely the
kind of decision that luck can carry at the boundary, and `guard_holdout` is a direct knob
on how many samples decide it (2/4/8/16 -> 128/64/32/16 of 256).

THE KNOB THAT MOVES THE BOUNDARY
--------------------------------
`make_regression_data(kind, noise=...)` exposes the noise level, and noise is what shrinks
the calibration's held-out advantage toward zero.  `kind="noise"` (y independent of x) is
the far end, where section 6.11 already saw the gate refuse outright; the boundary is
somewhere between the default `0.10` and there.

WHAT IS MEASURED
----------------
  P1  the ACCEPT RATE across a noise sweep, to locate the boundary -- the noise at which
      the gate's decision is genuinely uncertain rather than 0/40 or 40/40.
  P2  the POWER TEST at the boundary: `guard_holdout` in {2,4,8,16}.  If the accept rate
      moves with the slice size there, the decision is slice-size-dependent and luck can
      carry it; if it does not, the gate is robust where it is hardest to be.
  P3  THE COUNTEREXAMPLE SEARCH -- the thing section 6.11 never found.  `train_model`'s
      own documentation says arm="hybrid" and arm="geometry" spend an IDENTICAL physical
      measurement budget and differ only by the calibration, so `hybrid < geometry`
      isolates the calibration's effect.  Cells where the gate ACCEPTED and the arm still
      ended BELOW its own geometry arm are counted; the base notes that its earlier
      "harm?" column conflated this with losing to a different arm, so the comparison here
      is the one the guarantee is actually about.

SCOPE: task 4d with a swept noise level, one architecture, the diffusion substrate and the
base's own `effective_budget`, 40 epochs, 3 seeds.  No base file is edited.

Output: outputs/ext_33_gate_power.txt + _plot.png
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
    Report, build_model, effective_budget, header, make_regression_data,
    set_diffusion_defaults, set_ideal, train_model, try_plots,
)

ARCH = "G"
TASK = "4d"
SUBSTRATE = "diffusion"
EPOCHS = 40
ARMS = ("geometry", "weights", "full", "hybrid")
NOISES = (0.10, 0.40, 0.80, 1.50, 3.00)
HOLDOUTS = (2, 4, 8, 16)
SEEDS = 3


# The repository's OWN convention for the extension's model caches: `.gitignore`
# line 63 ignores `outputs/ext_cache/` and says so ("The extension keeps its OWN
# model cache").  An earlier version wrote to `extensions/multigraph/ext_33_cache/`
# instead -- NOT ignored, and `git check-ignore` reported the opposite, so the
# location was verified with `git status` rather than with an exit code.
CACHE = os.path.join(ROOT, "outputs", "ext_cache", "ext_33")


def run(noise, holdout, seed, arm, task=TASK, epochs=EPOCHS):
    """
    One cell.  Returns the final test R^2 and the gate's counters.

    CACHED, because an earlier version of this script trained for 35 minutes and then
    died in its own reporting step on a key-type bug, losing all of it -- the same
    lesson `abl_06` records ("any interruption threw the whole sweep away -- which
    happened twice").  The cache key carries every argument that changes the cell.
    """
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, f"{task}_{arm}_n{noise}_h{holdout}_s{seed}_e{epochs}.npz")
    if os.path.exists(cf):
        try:
            z = np.load(cf)
            return dict(r2=float(z["r2"]), n_cal=int(z["n_cal"]), n_rej=int(z["n_rej"]),
                        n_acc=int(z["n_cal"]) - int(z["n_rej"]),
                        acc_rate=(int(z["n_cal"]) - int(z["n_rej"])) / int(z["n_cal"])
                        if int(z["n_cal"]) else float("nan"),
                        budget=int(z["budget"]), gaps=list(z["gaps"]))
        except Exception:
            pass
    X_tr, y_tr, X_te, y_te = make_regression_data(task, noise=noise, seed=42)
    bd, _sz = effective_budget(ARCH, X_tr, list(ARMS), None, substrate=SUBSTRATE)
    budget = int(bd["full"])
    model = build_model(ARCH, X_tr, seed=seed, substrate=SUBSTRATE)
    hist = train_model(model, X_tr, y_tr, X_te, y_te, arm=arm, epochs=epochs,
                       budget=budget, seed=seed, probe_batch=96,
                       substrate=SUBSTRATE, eval_every=1, eval_X=X_te, eval_y=y_te,
                       closed_form_guard=True, closed_form_guard_floor=0.0,
                       closed_form_guard_holdout=holdout)
    r2_final = float(hist["curve"][-1][1]) if hist["curve"] else float("nan")
    n_cal = int(hist["n_calibrations"])
    n_rej = int(hist["calib_rejected"])
    out = dict(r2=r2_final, n_cal=n_cal, n_rej=n_rej, n_acc=n_cal - n_rej,
               acc_rate=(n_cal - n_rej) / n_cal if n_cal else float("nan"),
               budget=budget,
               gaps=[float(g) for g in hist.get("calib_reject_gap", [])])
    try:
        np.savez(cf, r2=out["r2"], n_cal=out["n_cal"], n_rej=out["n_rej"],
                 budget=out["budget"], gaps=np.asarray(out["gaps"], dtype=np.float64))
    except Exception:
        pass
    return out


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_33: the gate's statistical power")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
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
    rep = Report("ext_33_gate_power.txt")
    for line in header("ext_33 - does the do-no-harm gate have power at its boundary?",
                       f"task={TASK} arch={ARCH} noises={list(NOISES)} "
                       f"holdouts={list(HOLDOUTS)} seeds={args.seeds} "
                       f"epochs={args.epochs} substrate={SUBSTRATE}"):
        rep(line)

    rep("")
    rep("  Section 6.11 tried three counterexamples and all failed, and it says why the")
    rep("  third one could not settle the question: a FIRST-ORDER failure is detectable at")
    rep("  any slice size, so it never probed the gate's STATISTICAL POWER.  That needs a")
    rep("  task near the decision boundary, and the base says it was never constructed.")
    rep("")
    rep("  THE GATE: val = every `guard_holdout`-th training sample (a STRIDE); fit the")
    rep("  strengths on the complement; keep iff the fitted strengths BEAT the in-place")
    rep("  ones on that stride, strictly.  A strict inequality on a finite sample is")
    rep("  exactly what luck can carry at the boundary, and `guard_holdout` is a direct")
    rep("  knob on how many samples decide it.")

    t0 = time.time()
    runs = {}

    # ---- P1: locate the boundary ----------------------------------------
    rep("\n" + "=" * 100)
    rep("  P1  the accept rate across a NOISE sweep (holdout=4)")
    rep("=" * 100)
    rep("")
    rep(f"  {'noise':>7} {'accepted/total':>16} {'accept rate':>12} "
        f"{'hybrid R2':>11} {'geometry R2':>12} {'hybrid - geom':>14} "
        f"{'mean reject gap':>17}")
    rep("  " + "-" * 96)
    p1 = {}
    for noise in NOISES:
        acc = tot = 0
        hy, ge, gaps = [], [], []
        for s in range(args.seeds):
            h = run(noise, 4, s, "hybrid", epochs=args.epochs)
            g = run(noise, 4, s, "geometry", epochs=args.epochs)
            runs[(noise, 4, s, "hybrid")] = h
            runs[(noise, 4, s, "geometry")] = g
            acc += h["n_acc"]
            tot += h["n_cal"]
            hy.append(h["r2"])
            ge.append(g["r2"])
            gaps += h["gaps"]
            print(f"  [P1] noise={noise} seed={s}: accepted {h['n_acc']}/{h['n_cal']} "
                  f"hybrid {h['r2']:+.4f} geom {g['r2']:+.4f}", flush=True)
        p1[noise] = dict(acc=acc, tot=tot, rate=acc / tot if tot else float("nan"),
                         hy=float(np.mean(hy)), ge=float(np.mean(ge)),
                         gap=float(np.mean(gaps)) if gaps else float("nan"))
        rep(f"  {noise:>7.2f} {f'{acc}/{tot}':>16} {p1[noise]['rate']:>12.3f} "
            f"{p1[noise]['hy']:>+11.4f} {p1[noise]['ge']:>+12.4f} "
            f"{p1[noise]['hy'] - p1[noise]['ge']:>+14.4f} {p1[noise]['gap']:>+17.4f}")
    rep("")
    finite = {n: v for n, v in p1.items() if np.isfinite(v["rate"])}
    if finite:
        boundary = min(finite, key=lambda n: abs(finite[n]["rate"] - 0.5))
        rep(f"  THE BOUNDARY, by the accept rate closest to 0.5: noise = {boundary} "
            f"(rate {finite[boundary]['rate']:.3f})")
    else:
        boundary = NOISES[len(NOISES) // 2]
        rep(f"  no finite accept rate; boundary defaults to noise = {boundary}")

    # ---- P2: the power test ---------------------------------------------
    near = [n for n in NOISES if abs(n - boundary) <= 0.45] or [boundary]
    rep("\n" + "=" * 100)
    rep("  P2  THE POWER TEST: does the accept DECISION depend on the slice size?")
    rep("=" * 100)
    rep("")
    rep(f"  noise levels used: {near}  (the boundary and its neighbours)")
    rep("")
    rep(f"  {'noise':>7} {'holdout':>8} {'val samples':>12} {'accepted/total':>16} "
        f"{'accept rate':>12} {'hybrid R2':>11} {'hybrid - geom':>14}")
    rep("  " + "-" * 86)
    p2 = {}
    for noise in near:
        for ho in HOLDOUTS:
            acc = tot = 0
            hy, ge = [], []
            for s in range(args.seeds):
                key = (noise, ho, s, "hybrid")
                h = runs.get(key) or run(noise, ho, s, "hybrid", epochs=args.epochs)
                runs[key] = h
                g = runs.get((noise, ho, s, "geometry")) or \
                    runs.get((noise, 4, s, "geometry")) or \
                    run(noise, ho, s, "geometry", epochs=args.epochs)
                runs[(noise, ho, s, "geometry")] = g
                acc += h["n_acc"]
                tot += h["n_cal"]
                hy.append(h["r2"])
                ge.append(g["r2"])
            p2[(noise, ho)] = dict(acc=acc, tot=tot,
                                   rate=acc / tot if tot else float("nan"),
                                   hy=float(np.mean(hy)), ge=float(np.mean(ge)))
            rep(f"  {noise:>7.2f} {ho:>8} {256 // ho:>12} {f'{acc}/{tot}':>16} "
                f"{p2[(noise, ho)]['rate']:>12.3f} {p2[(noise, ho)]['hy']:>+11.4f} "
                f"{p2[(noise, ho)]['hy'] - p2[(noise, ho)]['ge']:>+14.4f}")
            print(f"  [P2] noise={noise} holdout={ho}: {acc}/{tot}", flush=True)
        rep("")

    # ---- P3: the counterexample search ----------------------------------
    rep("=" * 100)
    rep("  P3  THE COUNTEREXAMPLE SEARCH: gate ACCEPTED, and the arm still lost")
    rep("=" * 100)
    rep("")
    rep("  The guarantee is 'deploying a calibration never leaves the arm worse than NOT")
    rep("  calibrating'.  `arm='hybrid'` and `arm='geometry'` spend an identical physical")
    rep("  measurement budget and differ ONLY by the calibration (train_model's own")
    rep("  documentation), so `hybrid < geometry` isolates the calibration's effect.")
    rep("")
    # THE GRID IS THE UNION OF P1 AND P2, not just the boundary pair.  An earlier
    # version restricted P3 to the noise levels P1's heuristic called "the boundary"
    # (rate closest to 0.5), which EXCLUDED the one cell P1 had already shown with a
    # NEGATIVE margin -- i.e. the counterexample candidate was in the table above and
    # the search did not look at it.
    # A LIST, not a dict: the previous version built a dict whose keys were floats from
    # P1 and tuples from P2, and `sorted()` on it raised TypeError AFTER all the training
    # was done.  Mixing key types is the bug; not sorting is the fix.
    grid = [((n, 4), v) for n, v in p1.items()]
    grid += [((n, ho), v) for (n, ho), v in p2.items() if ho != 4]
    grid.sort(key=lambda kv: (kv[0][0], kv[0][1]))
    rep(f"  {'noise':>7} {'holdout':>8} {'gate accepted':>14} {'hybrid':>10} "
        f"{'geometry':>10} {'hybrid - geom':>14} {'VERDICT':>22}")
    rep("  " + "-" * 92)
    n_accept_lose = 0
    n_cells = 0
    harmed = []
    for (noise, ho), v in grid:
        accepted = v["acc"] > 0
        lost = v["hy"] < v["ge"]
        bad = accepted and lost
        n_cells += 1
        n_accept_lose += int(bad)
        tag = ("ACCEPTED AND LOST" if bad else
               ("accepted, no harm" if accepted else "refused"))
        if bad:
            harmed.append((noise, ho, v["acc"], v["hy"] - v["ge"]))
        rep(f"  {noise:>7.2f} {ho:>8} {v['acc']:>14} {v['hy']:>+10.4f} "
            f"{v['ge']:>+10.4f} {v['hy'] - v['ge']:>+14.4f} {tag:>22}")
    rep("")

    # ---- verdict ---------------------------------------------------------
    rep("=" * 100)
    rep("  VERDICT")
    rep("=" * 100)
    rep("")
    rates = [v["rate"] for v in p2.values() if np.isfinite(v["rate"])]
    spread = (max(rates) - min(rates)) if rates else float("nan")
    rep(f"  accept rates across the slice sizes at the boundary: "
        f"{[round(r, 3) for r in rates]}")
    rep(f"  spread = {spread:.3f}")
    rep("")
    # the confound, stated where the verdict is drawn rather than in a footnote
    rep("  THE CONFOUND IN THIS COMPARISON, stated because it bounds the verdict: the")
    rep("  hybrid and geometry arms are separate trainings, matched on data, seed and")
    rep("  initialisation, but the calibration CHANGES THE TRAJECTORY -- it rewrites")
    rep("  W_gauss mid-run, so the finite differences that follow see a different loss")
    rep("  surface.  So `hybrid < geometry` is the calibration\'s effect PLUS that")
    rep("  divergence, and a single cell cannot separate them.  The base flagged the same")
    rep("  conflation in its own section 6.11 and this design narrows it without closing it.")
    rep("")
    if harmed:
        rep("  CELLS WITH A NEGATIVE MARGIN AND AN ACCEPTED CALIBRATION:")
        for noise, ho, acc, d in harmed:
            rep(f"    noise={noise} holdout={ho}: accepted {acc}, hybrid - geometry {d:+.4f}")
        rep("")
    if n_accept_lose:
        rep(f"  **THE COUNTEREXAMPLE EXISTS: {n_accept_lose} of {n_cells} boundary cells")
        rep("  had the gate ACCEPT a calibration and the arm still end BELOW its own")
        rep("  geometry arm.**  That is the evidence section 6.11 searched for and did not")
        rep("  find, and it is the guarantee -- not the stronger 'beats the best search")
        rep("  arm' claim -- that it bears on.")
    else:
        rep(f"  NO COUNTEREXAMPLE in {n_cells} boundary cells: wherever the gate accepted,")
        rep("  the arm matched or beat its own geometry arm.  **So the guarantee held even")
        rep("  where it was hardest to hold** -- at a boundary the base itself never")
        rep("  constructed, which is stronger evidence for it than the three designs in")
        rep("  section 6.11, each of which failed for its own reason.")
    rep("")
    if np.isfinite(spread) and spread > 0.2:
        rep(f"  AND THE POWER IS SLICE-SIZE-DEPENDENT: the accept rate moves by "
            f"{spread:.3f}")
        rep("  across guard_holdout.  So the gate's decision at the boundary IS partly a")
        rep("  matter of how many samples decide it -- which is the property section 6.11")
        rep("  said it had never tested.")
    else:
        rep(f"  AND THE GATE IS SLICE-SIZE-STABLE at this boundary (spread {spread:.3f}).")
        rep("  Whatever decides the accept/reject there, it is not the sample count alone.")
    rep("")
    rep("  SCOPE: ONE task whose noise level was swept, one architecture, diffusion")
    rep("  substrate, the base's own effective_budget, 40 epochs and the seeds above.")
    rep("  A single boundary is not the boundary: a different task could sit elsewhere and")
    rep("  behave differently, and nothing here bounds that.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

    path = rep.save()
    print(f"\nReport saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")}")

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
        ns = list(p1)
        ax[0].plot(ns, [p1[n]["rate"] for n in ns], "o-")
        ax[0].axhline(0.5, color="k", ls="--", lw=0.9)
        ax[0].set_xlabel("task noise level")
        ax[0].set_ylabel("gate accept rate")
        ax[0].set_title("P1: locating the boundary")
        ax[0].grid(True, alpha=0.25)
        for n in near:
            ax[1].plot(HOLDOUTS, [p2[(n, h)]["rate"] for h in HOLDOUTS], "o-",
                       label=f"noise={n}")
        ax[1].set_xscale("log", base=2)
        ax[1].set_xticks(list(HOLDOUTS))
        ax[1].set_xticklabels([str(h) for h in HOLDOUTS])
        ax[1].set_xlabel("guard_holdout (larger = smaller validation slice)")
        ax[1].set_ylabel("gate accept rate")
        ax[1].set_title("P2: is the decision slice-size-dependent?")
        ax[1].grid(True, alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_33_gate_power_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")}")

    ok = all(np.isfinite(v["hy"]) for v in p1.values()) and len(p2) == len(near) * len(HOLDOUTS)
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
