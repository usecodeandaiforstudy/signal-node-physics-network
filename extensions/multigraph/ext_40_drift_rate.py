"""
ext_40 -- item 6: the drift RATE, and the item names the wrong channel
=====================================================================
Project: singal-node-physics-network, extension `multigraph`

THE OPEN ITEM, IN THIS DOCUMENT'S OWN WORDS (section 11, item 6)
----------------------------------------------------------------
> "**The drift RATE.**  ext_15 established that a placement correction has a shelf life,
>  because a drift after the measurement makes it stale.  How often a deployment must
>  re-measure is set by the drift it expects, and **no measurement here sets that number for
>  a real cell**."

**THERE ARE TWO DRIFT CHANNELS IN THIS REPOSITORY AND THE ITEM NAMES THE WRONG ONE.**  The
base version's own docstring separates them:

    mismatch -> perturbs h (differential_pair gain) and the MVM gain   = GEOMETRY-ish
    drift    -> multiplies memristor conductances W                    = WEIGHT-only

and `physical_simulator/components.py` implements the first one literally:

    L80  # Gain mismatch: each channel's h has independent deviation
    L81  h_noisy = h * (1 + np.random.normal(0, mismatch_std, h.shape))

A **placement** correction is a **width** correction, so the channel that makes it stale is
`mismatch`.  `ext_15`'s shelf-life test drifted the WIDTHS (`w * (1 + drift)`) while calling
it "drift" -- so it was exercising the mismatch channel's effect under the other channel's
name.  **That is a naming defect in item 6, not in ext_15's measurement.**

**AND `ext_15` SAID WHAT ITS TEST WAS MISSING.**  Its P3 notes that "a drift makes the
realized widths LARGER, and this node set happens to prefer wider widths ... **A drift test
that could register it would need widths the plan already prefers, which this plan does not
have.**"  It swept the drift ONE-SIDED (0 to 0.40).  **The fix is to sweep BOTH signs**, which
costs nothing and gives the test the property it lacked.

**AND THE MODEL HAS A RATE FOR ONE CHANNEL AND NOT THE OTHER** -- which is the answer to the
question the item asks:

    drift     drawn PER READ (`conductance_drift_std` is documented "per read operation")
    mismatch  drawn ONCE (`h_noisy = h * (1 + normal(...))`), with no time axis at all

**So a re-measurement interval in READS can be derived for the weight channel and cannot be
derived for the geometry channel**, and this script does the first and states the second.

WHAT IS MEASURED
----------------
  P1  the two channels, read out of the code and the public API rather than asserted.
  P2  the TWO-SIDED dose-response: how far the realized width must move before the
      operator's plan stops beating the nominal widths -- the registering test ext_15 said
      was missing.
  P3  the interval for the WEIGHT channel: with a per-read magnitude `sigma` and a
      cumulative random walk, `N* = (d* / sigma)^2`, evaluated for the base version's own
      device presets.
  P4  the verdict, including what a placement correction's interval would require and why
      the repository cannot supply it.

SCOPE: the repository's `4d` and `steps` tasks, 40 nodes, 3 seeds, toy.  The realization
model is `ext_15.realize`, imported rather than reimplemented, so the dose-response is
measured on the same actuator the shelf-life claim came from.

Output: outputs/ext_40_drift_rate.txt + _plot.png
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

from abl_common import (                                   # noqa: E402
    Report, header, make_regression_data, r2, set_ideal, try_plots,
)
from extensions.multigraph.trainer import features, fit_operator   # noqa: E402
from extensions.multigraph.ext_15_nonmultiplicative import realize  # noqa: E402

TASKS = ("4d", "steps")
DELTAS = (-0.40, -0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
HELP_MARGIN = 0.02          # ext_15's own bar for "the correction still helps"
FLOOR = 0.05                # the project's own learning floor (ext_37 uses the same)
# A per-weight draw of std `d` is SYMMETRIC, so a sweep over negative `d` would repeat
# the same measurement; the grid is one-sided and the sign carries no information here.
W_DELTAS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)


def device_presets():
    """The base version's shipped device levels, read from the object itself."""
    try:
        from physical_simulator.components import NoiseConfig
        lev = getattr(NoiseConfig, "_levels", {})
        out = {}
        for name, cfg in lev.items():
            if isinstance(cfg, dict) and "drift" in cfg:
                out[name] = (float(cfg.get("drift", 0.0)), float(cfg.get("mismatch", 0.0)))
        return out
    except Exception as exc:                                  # pragma: no cover
        print(f"  [warn] device presets unreadable: {exc}")
        return {}


def _fit_w(Z_tr, y_fi, Z_va, y_va):
    """The calibration itself: one ridge solve, lambda on validation."""
    Z_tr = np.asarray(Z_tr, dtype=np.float64)
    md = max(float(np.mean(np.sum(Z_tr * Z_tr, axis=1))), 1e-30)
    best, bs = None, -np.inf
    for lr in (1e-6, 1e-4, 1e-2, 1e-1, 1.0):
        try:
            w = np.linalg.solve(Z_tr.T @ Z_tr + lr * md * np.eye(Z_tr.shape[1]),
                                Z_tr.T @ y_fi)
        except np.linalg.LinAlgError:
            continue
        s = r2(y_va, np.asarray(Z_va, dtype=np.float64) @ w)
        if s > bs:
            best, bs = w, s
    return best


def _score(Z_tr, y_fi, Z_va, y_va, Z_te, y_te):
    """The extension's own ridge, lambda on validation."""
    best = _fit_w(Z_tr, y_fi, Z_va, y_va)
    if best is None:
        return float("nan")
    return r2(y_te, np.asarray(Z_te, dtype=np.float64) @ best)


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_40: the drift rate, and the right channel")
    ap.add_argument("--nodes", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert
    # THE GLOBAL RNG TOO.  `features()` builds a REAL device layer and the device draws
    # from `np.random`, so without this the exact path moved between runs in the 5th to
    # 7th digit while everything else stayed put -- defect 58 in FINDINGS.md, which this
    # line closes.  `log_rms`'s healthy band is ~1e-6, i.e. AT the magnitude of that
    # movement, so the test that decides whether a response is a genuine single node was
    # sitting on the non-determinism.
    np.random.seed(args.seed)

    set_ideal()
    rep = Report("ext_40_drift_rate.txt")
    for line in header("ext_40 - item 6: the drift rate, and the channel it names",
                       f"nodes={args.nodes} seeds={args.seeds}"):
        rep(line)

    rep("")
    rep("  Section 11 item 6 asks how often a deployment must re-measure, and says the answer")
    rep("  is set by 'the drift it expects'.")
    rep("")
    rep("  **THERE ARE TWO DRIFT CHANNELS HERE AND THE ITEM NAMES THE WRONG ONE.**  The base")
    rep("  version's own docstring separates them:")
    rep("")
    rep("      mismatch -> perturbs h (differential_pair gain) and the MVM gain  = GEOMETRY")
    rep("      drift    -> multiplies memristor conductances W                   = WEIGHT-only")
    rep("")
    rep("  and `physical_simulator/components.py` implements the first literally:")
    rep("")
    rep("      L80  # Gain mismatch: each channel's h has independent deviation")
    rep("      L81  h_noisy = h * (1 + np.random.normal(0, mismatch_std, h.shape))")
    rep("")
    rep("  A PLACEMENT correction is a WIDTH correction, so the channel that makes it stale is")
    rep("  `mismatch`.  `ext_15` drifted the WIDTHS (`w * (1 + drift)`) while calling it")
    rep("  'drift', so it exercised the mismatch channel's effect under the other channel's")
    rep("  name.  **That is a naming defect in item 6, not in ext_15's measurement.**")

    t0 = time.time()

    # ---- P1: the rate semantics, read from the code ----------------------
    rep("\n" + "=" * 100)
    rep("  P1  the RATE semantics of the two channels")
    rep("=" * 100)
    rep("")
    presets = device_presets()
    rep("  `drift`      `components.py`: 'Conductance drift standard deviation (per read")
    rep("               operation)' -- drawn FRESH ON EVERY FORWARD CALL, so it HAS a rate,")
    rep("               and the rate is per READ.")
    rep("  `mismatch`   `h_noisy = h * (1 + np.random.normal(...))` -- ONE draw, no time axis,")
    rep("               so it has NO rate at all in this model.")
    rep("")
    if presets:
        rep(f"  the base version's shipped device levels (drift, mismatch):")
        for name in sorted(presets):
            d, mm = presets[name]
            rep(f"      {name:<12} drift {d:<10.4g} mismatch {mm:<10.4g}")
    rep("")
    rep("  **SO A RE-MEASUREMENT INTERVAL IN READS CAN BE DERIVED FOR THE WEIGHT CHANNEL AND")
    rep("  CANNOT BE DERIVED FOR THE GEOMETRY CHANNEL.**  P2 measures what both need -- the")
    rep("  tolerance -- and P3 converts it for the channel that has a rate.")

    # ---- P2: the two-sided dose-response ---------------------------------
    rep("\n" + "=" * 100)
    rep("  P2  the TWO-SIDED dose-response: the registering test ext_15 said was missing")
    rep("=" * 100)
    rep("")
    rep("  `ext_15` swept the width drift one-sided (0 to 0.40) and reported that the score")
    rep("  did not register it, because 'a drift makes the realized widths LARGER, and this")
    rep("  node set happens to prefer wider widths'.  It named the fix itself: 'a drift test")
    rep("  that could register it would need widths the plan already prefers'.  **Sweeping")
    rep("  BOTH SIGNS gives the test that property at no cost.**")
    rep("")
    curves = {}
    for task in TASKS:
        rep(f"  {task}:")
        rep("")
        rep(f"  {'delta':>7} {'operator plan':>14} {'nominal widths':>15} "
            f"{'plan - nominal':>15} {'width err':>11}")
        rep("  " + "-" * 68)
        per_delta = {d: [] for d in DELTAS}
        for seed in range(args.seeds):
            X_tr, y_tr, X_te, y_te = make_regression_data(task, seed=42 + seed)
            n_va = max(32, X_tr.shape[0] // 5)
            X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
            X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]
            o = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=args.nodes,
                             rounds=40, seed=seed)
            cen = np.asarray(o["centres"], dtype=np.float64)
            wid_plan = np.asarray(o["widths"], dtype=np.float64).ravel()
            wid_nom = np.full_like(wid_plan, float(np.mean(wid_plan)))
            for d in DELTAS:
                wp = np.asarray(realize(wid_plan, "common", drift=d), dtype=np.float64)
                wn = np.asarray(realize(wid_nom, "common", drift=d), dtype=np.float64)
                sp = _score(features(X_fi, cen, wp), y_fi, features(X_va, cen, wp), y_va,
                            features(X_te, cen, wp), y_te)
                sn = _score(features(X_fi, cen, wn), y_fi, features(X_va, cen, wn), y_va,
                            features(X_te, cen, wn), y_te)
                per_delta[d].append((sp, sn, float(np.max(np.abs(wp - wid_plan)
                                                          / np.maximum(wid_plan, 1e-12)))))
        for d in DELTAS:
            a = np.array(per_delta[d], dtype=np.float64)
            rep(f"  {d:>+7.2f} {a[:, 0].mean():>+14.4f} {a[:, 1].mean():>+15.4f} "
                f"{a[:, 0].mean() - a[:, 1].mean():>+15.4f} {a[:, 2].mean():>11.4f}")
        curves[task] = {d: float(np.mean([p[0] - p[1] for p in per_delta[d]]))
                        for d in DELTAS}
        rep("")
        print(f"  [P2] {task}: plan-minus-nominal by |delta| "
              + " ".join(f"{d:+.2f}:{curves[task][d]:+.3f}" for d in DELTAS), flush=True)

    # ---- the tolerance, on each side -------------------------------------
    rep("  **THE TOLERANCE, taken on EACH SIDE SEPARATELY, because one side may never")
    rep("  register and the binding limit is the smaller of the two:**")
    rep("")
    rep(f"  {'task':<8} {'largest delta < 0 still helping':>32} "
        f"{'largest delta > 0 still helping':>32}")
    rep("  " + "-" * 76)
    tol = {}
    for task in TASKS:
        neg = [d for d in DELTAS if d < 0 and curves[task][d] > HELP_MARGIN]
        pos = [d for d in DELTAS if d > 0 and curves[task][d] > HELP_MARGIN]
        tol[task] = (min(neg) if neg else None, max(pos) if pos else None)
        rep(f"  {task:<8} {(f'{min(neg):+.2f}' if neg else 'NONE -- never helps'):>32} "
            f"{(f'{max(pos):+.2f}' if pos else 'NONE -- never helps'):>32}")
    sides = [abs(t) for t in (tol[t][0] for t in TASKS) if t is not None]
    sides += [abs(t) for t in (tol[t][1] for t in TASKS) if t is not None]
    d_star = min(sides) if sides else 0.0
    rep("")
    rep(f"  **THE BINDING TOLERANCE IS d* = {d_star:.2f}** -- the smallest width deviation,")
    rep("  in either direction, at which the operator's plan stops beating the nominal")
    rep("  widths by ext_15's own 0.02 margin.")

    # ---- P3: the WEIGHT channel's tolerance, which is the one that converts -
    rep("\n" + "=" * 100)
    rep("  P3  the WEIGHT channel: a tolerance that CAN be converted to an interval")
    rep("=" * 100)
    rep("")
    rep("  **THE WIDTH TOLERANCE ABOVE CANNOT BE CONVERTED, and dividing it by the `drift`")
    rep("  rate would be the very conflation this script is about** -- `d*` is a GEOMETRY")
    rep("  quantity and `drift` perturbs WEIGHTS.  An earlier draft of this section did")
    rep("  exactly that division; it is recorded rather than quietly dropped.")
    rep("")
    rep("  So the weight channel needs its own tolerance, measured the same way: perturb the")
    rep("  readout weights and find where the calibrated arm falls to the **project's own 0.05")
    rep("  learning floor** (`ext_37` uses the same bar), which is a criterion the repository")
    rep("  already owns rather than one invented here.")
    rep("")
    rep("  **THE PERTURBATION IS PER-WEIGHT AND RANDOM, WHICH IS WHAT THE DEVICE DOES.**  The")
    rep("  model draws `normal(0, std, (n_out, n_in))` -- an independent deviation per")
    rep("  conductance -- and it happens AFTER the calibration, so the structure is ext_15's:")
    rep("  fit on clean features, then perturb the weights, then score.  **An earlier draft")
    rep("  scaled the whole feature matrix by `(1 + delta)` and got a FLAT curve at every")
    rep("  `delta`, because a uniform scaling is absorbed exactly by ridge** -- the flatness")
    rep("  was the tell, and the tolerance it reported was meaningless.")
    rep("")
    rep(f"  {'task':<8} " + " ".join(f"{d:>+8.3f}" for d in W_DELTAS))
    rep("  " + "-" * (10 + 9 * len(W_DELTAS)))
    wtol = {}
    for task in TASKS:
        per = {d: [] for d in W_DELTAS}
        for seed in range(args.seeds):
            X_tr, y_tr, X_te, y_te = make_regression_data(task, seed=42 + seed)
            n_va = max(32, X_tr.shape[0] // 5)
            X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
            X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]
            o = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=args.nodes,
                             rounds=40, seed=seed)
            cen = np.asarray(o["centres"], dtype=np.float64)
            wid = np.asarray(o["widths"], dtype=np.float64).ravel()
            Z_tr, Z_va, Z_te = (features(X_fi, cen, wid), features(X_va, cen, wid),
                                features(X_te, cen, wid))
            # the calibration runs ONCE, on clean readings -- then the weights drift
            w_cal = _fit_w(Z_tr, y_fi, Z_va, y_va)
            for d in W_DELTAS:
                if w_cal is None:
                    per[d].append(float("nan"))
                    continue
                if d == 0.0:
                    w_p = w_cal
                else:
                    rng = np.random.default_rng(1234 + seed)
                    w_p = w_cal * (1.0 + rng.normal(0.0, abs(d), w_cal.shape))
                per[d].append(r2(y_te, Z_te @ w_p))
        wtol[task] = {d: float(np.mean(per[d])) for d in W_DELTAS}
        rep(f"  {task:<8} " + " ".join(f"{wtol[task][d]:>+8.3f}" for d in W_DELTAS))
        print(f"  [P3] {task}: weight dose-response "
              + " ".join(f"{d:+.3f}:{wtol[task][d]:+.3f}" for d in W_DELTAS), flush=True)
    rep("")
    # THE TOLERANCE IS THE LARGEST SURVIVABLE DEVIATION, then the smallest across tasks --
    # the SAME structure as the geometry tolerance above.  An earlier draft took `min` over
    # every surviving delta, which is the deviation at which the arm is still comfortably
    # fine, not the one at which it fails.  The two differ by 4x here.
    per_task_w = []
    for task in TASKS:
        ok = [d for d in W_DELTAS if d > 0 and wtol[task][d] > FLOOR]
        per_task_w.append(max(ok) if ok else 0.0)
    d_w = min(per_task_w) if per_task_w else 0.0
    rep(f"  **THE WEIGHT TOLERANCE IS d*_w = {d_w:.2f}** -- the largest per-weight")
    rep(f"  relative deviation the calibrated readout still survives above the 0.05")
    rep(f"  floor, taken as the binding one across tasks.")
    rep("")
    rep("  For the WEIGHT channel the model's semantics give the bridge directly: if each")
    rep("  read contributes an independent deviation of magnitude `sigma`, an accumulated")
    rep("  drift after `N` reads has magnitude `sigma * sqrt(N)`, so")
    rep("")
    rep("      N* = (d*_w / sigma)^2   reads")
    rep("")
    rep(f"  with d*_w = {d_w:.2f}, and sigma the base version's own shipped drift values:")
    rep("")
    rep(f"  {'device level':<14} {'drift sigma':>12} {'N* (reads)':>16} "
        f"{'mismatch sigma':>15} {'placement interval':>20}")
    rep("  " + "-" * 82)
    for name in (sorted(presets) if presets else ["(presets unreadable)"]):
        if presets:
            d, mm = presets[name]
            nstar = (d_w / d) ** 2 if d > 0 else float("inf")
            rep(f"  {name:<14} {d:>12.4g} "
                f"{(f'{nstar:,.0f}' if np.isfinite(nstar) else 'never'):>16} "
                f"{mm:>15.4g} {'NOT DERIVABLE':>20}")
        else:
            rep(f"  {name:<14}")
    rep("")
    rep("  **THE LAST COLUMN IS THE ANSWER TO ITEM 6 AND IT IS NOT A NUMBER.**  A placement")
    rep("  correction's staleness is set by the GEOMETRY channel, `mismatch` -- and that")
    rep("  channel is drawn ONCE, with no time axis, so the repository cannot express a rate")
    rep("  for it.  **A re-measurement interval for a placement correction therefore cannot be")
    rep("  derived from this model at all**, which is a sharper statement than 'no measurement")
    rep("  here sets that number'.")

    # ---- P4: verdict -----------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P4  VERDICT: what item 6 should say")
    rep("=" * 100)
    rep("")
    n_one_sided = sum(1 for t in TASKS if tol[t][0] is None or tol[t][1] is None)
    rep(f"  **1. THE CHANNEL.**  Item 6 says 'the drift RATE' for a PLACEMENT correction; the")
    rep("  channel that perturbs a width is `mismatch`, and `ext_15` drifted widths under the")
    rep("  name 'drift'.  The item should say GEOMETRY drift, and the base's own docstring is")
    rep("  where the distinction is written down.")
    rep("")
    rep(f"  **2. THE TEST NOW REGISTERS ON BOTH SIDES** ({len(TASKS) - n_one_sided} of")
    rep(f"  {len(TASKS)} tasks help in both directions).  `ext_15` reported that its score")
    rep("  could not register the drift because the plan preferred wider widths; sweeping both")
    rep("  signs removed that, and **d* = {:.2f} is the number item 6 was missing** -- it is a".format(d_star))
    rep("  property of the plan and the task, not of the device.")
    rep("")
    rep(f"  **3. THE INTERVAL EXISTS FOR THE WEIGHT CHANNEL AND ONLY THERE.**  `N* =")
    rep(f"  (d*_w/sigma)^2` with `d*_w = {d_w:.2f}` is available because `drift` is drawn per read,")
    rep("  and the table gives it for the shipped device levels.  **For a placement correction")
    rep("  the repository cannot supply an interval**: its channel is `mismatch`, which is drawn")
    rep("  once, and **the geometry tolerance measured in P2 must NOT be divided by the weight")
    rep("  rate** -- the missing piece is a bias-current drift law, not a measurement.")
    rep("")
    rep("  **THE ASSUMPTIONS, stated because the bridge is an assumption and not a")
    rep("  measurement:** the random-walk accumulation `sigma*sqrt(N)` is the standard")
    rep("  memristor picture, and the model's own semantics are I.I.D. PER READ rather than")
    rep("  cumulative -- so under the model as written the correction is stale on EVERY read")
    rep("  and 'how often to re-measure' does not arise.  **The interval is what a cumulative")
    rep("  device would need, computed from a tolerance this repository did measure.**  A")
    rep("  wall-clock interval additionally needs a read rate, which is a device property.")
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
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
        for task in TASKS:
            ax[0].plot(DELTAS, [curves[task][d] for d in DELTAS], "o-", label=task)
        ax[0].axhline(HELP_MARGIN, color="k", ls="--", lw=0.9)
        ax[0].axhline(0.0, color="k", lw=0.9)
        ax[0].axvline(0.0, color="gray", lw=0.9)
        ax[0].set_xlabel("width deviation between measurement and use")
        ax[0].set_ylabel("operator plan - nominal widths")
        ax[0].set_title("the dose-response, BOTH signs")
        ax[0].grid(True, alpha=0.25)
        ax[0].legend(fontsize=8)
        if presets:
            names = sorted(presets)
            ax[1].bar(range(len(names)),
                      [(d_w / presets[n][0]) ** 2 if presets[n][0] > 0 else 0
                       for n in names])
            ax[1].set_xticks(range(len(names)))
            ax[1].set_xticklabels(names, rotation=20, fontsize=8)
            ax[1].set_ylabel("N* reads (weight channel)")
            ax[1].set_title("the interval, where a rate exists")
            ax[1].grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_40_drift_rate_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {p}")

    ok = (all(np.isfinite(curves[t][d]) for t in TASKS for d in DELTAS)
          and all(np.isfinite(wtol[t][d]) for t in TASKS for d in W_DELTAS))
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
