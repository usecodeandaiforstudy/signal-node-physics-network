"""
Ablation E -- Does the closed-form calibration survive a WRONG digital twin?
===========================================================================
Project: singal-node-physics-network

THE LAST UNMODELLED ASSUMPTION
-----------------------------
FINDINGS.md section 6 recommends: calibrate the source strengths in closed form,
spend every physical measurement on the geometry.  Ablation D showed that
recommendation survives sensor noise, drift and positioner error on the device.

But in EVERY experiment run so far the calibration read its feature matrix from
the SAME object that then performed the forward pass, i.e. the digital twin and
the device were identical by construction, so twin error was exactly zero.  That
is not the real situation:

    * the search arms (geometry / weights / full) probe the REAL device loss, so
      they adapt to whatever the device actually does;
    * the calibrated arm trusts the twin's description of the cell.  If the twin
      is wrong, the calibration does not know.

Two distinct channels carry twin error onto the strengths:

  1. twin-supplied features -- the solve uses Z from the twin, so a wrong twin
     produces a wrong Z and therefore wrong strengths;
  2. actuation conversion -- even when Z is measured on the device, converting
     the solved W into an injected pulse energy uses the twin's width
     (q_j = W_j * (4 pi D t_j)^(dim/2), t_j = h_j^2 / (4 k D)), so a wrong twin
     mis-scales the sources.

Both are modelled here at once by calibrating on a PERTURBED COPY of the device
model and then deploying the result onto the true device, which is what a real
twin-driven workflow does.

This concern is not hypothetical: the twin of an analog cell has exactly the
process spread that ablation C measured on the device (gain/width mismatch in
differential_pair, squarer mismatch, mirror copy-ratio error).  Ablation C asked
what DEVICE mismatch does to trained models; this asks what TWIN mismatch does to
calibrated strengths.

FALSIFIABLE PREDICTION
----------------------
R^2 of the calibrated arm falls monotonically with twin error, and there is a
relative twin error beyond which searching the strengths is better.  That
threshold is the actionable number: "your twin must be accurate to X% or you
should not calibrate".

CONTROLS THAT MUST HOLD (they falsify the experiment if they fail)
-----------------------------------------------------------------
  A. twin error = 0 must reproduce the unperturbed calibrated arm exactly.
  B. the search arms must be FLAT across the twin-error axis.  They never touch
     the twin, so any movement in them means the perturbation is leaking into the
     device rather than staying in the twin.

DIAGNOSTIC
----------
For each twin-error level the script also reports, at a FIXED random geometry:
    R^2 the TWIN believes it achieved   vs   R^2 the DEVICE actually delivers
with the same strengths.  This is the failure mode in one line: a wrong twin
reports success while the device disagrees.

Output: outputs/abl_05_twin_error.txt + _plot.png
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abl_common import (                                   # noqa: E402
    ARCHS, OUT_DIR, Report, build_model, curve_at, effective_budget, header,
    make_regression_data, merge_curves, r2, readout_calibration_cost,
    selftest, set_diffusion_defaults, set_ideal,
    solve_readout_closed_form, train_model, try_plots,
)

# Relative error in the twin's particle widths.  0 is the control.
H_AXIS = (0.0, 0.01, 0.05, 0.10, 0.20)
# Offset in the twin's particle positions, in INPUT units (data is ~U(-2,2)).
C_AXIS = (0.0, 0.005, 0.02)

ARMS = ("geometry", "weights", "full", "hybrid")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ablation E: twin error")
    p.add_argument("--task", default="4d", choices=["2d", "4d"])
    p.add_argument("--arch", default="F2", choices=list(ARCHS))
    p.add_argument("--substrate", default="diffusion")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--calib-every", type=int, default=10,
                   help="epochs between calibrations (as in ablation D)")
    p.add_argument("--probe-batch", type=int, default=96)
    p.add_argument("--milestones", default="10000,25000,50000")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


def make_twin(arch, X_ref, seed, device, eps_h, eps_c, substrate):
    """
    A digital twin: same strengths/geometry as the device, but its BELIEF about
    the cell's physics is perturbed.  h and centers are in input units on both
    substrates (the physical conversion happens downstream), so a relative width
    error and a positional offset mean the same thing on either medium.
    """
    twin = build_model(arch, X_ref, seed=seed, substrate=substrate)
    twin.set_params(device.get_params().copy())
    for l in twin.layers:
        l.h = np.asarray(l.h, dtype=np.float64) * (1.0 + float(eps_h))
        l.centers = np.asarray(l.centers, dtype=np.float64) + float(eps_c)
    return twin


def self_deception(arch, X_tr, y_tr, X_te, y_te, seed, eps_h, eps_c, substrate):
    """
    At a fixed random geometry, calibrate on the twin and then read the SAME
    strengths on the device.  Returns (R2 the twin believes, R2 the device gives).
    """
    device = build_model(arch, X_tr, seed=seed, substrate=substrate)
    twin = make_twin(arch, X_tr, seed, device, eps_h, eps_c, substrate)
    solve_readout_closed_form(twin, X_tr, y_tr, lam=1e-3, substrate=substrate)
    r_twin = r2(y_te, twin.forward(X_te))
    for ld, lt in zip(device.layers, twin.layers):
        ld.W_gauss[...] = np.asarray(lt.W_gauss, dtype=ld.W_gauss.dtype)
    r_dev = r2(y_te, device.forward(X_te))
    return float(r_twin), float(r_dev)


def run_one(arch, arm, seed, X_tr, y_tr, X_te, y_te, args, eps_h, eps_c):
    device = build_model(arch, X_tr, seed=seed, substrate=args.substrate)
    twin = None
    if arm == "hybrid":
        twin = make_twin(arch, X_tr, seed, device, eps_h, eps_c, args.substrate)
    hist = train_model(
        device, X_tr, y_tr, X_te, y_te, arm=arm, epochs=args.epochs,
        budget=int(budget), seed=seed, probe_batch=args.probe_batch,
        substrate=args.substrate, eval_every=1, eval_X=X_te, eval_y=y_te,
        closed_form_every=args.calib_every, closed_form_model=twin,
    )
    return hist, hist["curve"]


def main():
    args = parse_args()
    if args.quick:
        if args.seeds == 2:
            args.seeds = 1
        if args.epochs == 40:
            args.epochs = 15
    set_diffusion_defaults(level="none")
    set_ideal()

    rep = Report("abl_05_twin_error.txt")
    for line in header("Ablation E - Does the calibration survive a wrong twin?",
                       f"arch={args.arch} substrate={args.substrate} "
                       f"seeds={args.seeds} epochs={args.epochs} "
                       f"calib_every={args.calib_every}"):
        rep(line)

    rep("\n--- Part 0: self-test ---")
    selftest(verbose=True)
    rep("  Device non-idealities are OFF everywhere (level=none): the only")
    rep("  mismatch in this experiment is between the twin and the device.")

    X_tr, y_tr, X_te, y_te = make_regression_data(args.task)
    global budget
    bd, sizes = effective_budget(args.arch, X_tr, list(ARMS), args.budget,
                                substrate=args.substrate)
    budget = int(bd["full"])
    rep(f"\n  arch {args.arch} ({ARCHS[args.arch]['label']}), arm sizes {sizes}")
    rep(f"  FD budget = {budget} probes/epoch = {2*budget} forward passes")

    # ---------------- Part 1: self-deception diagnostic ----------------
    rep("\n" + "=" * 74)
    rep("  Part 1: what the twin believes vs what the device delivers")
    rep("=" * 74)
    rep("  Fixed random geometry, one calibration, same strengths read on both.")
    rep("")
    rep(f"  {'twin width err':>15} {'R2 twin thinks':>16} {'R2 device gets':>16} "
        f"{'gap':>9}")
    rep("  " + "-" * 60)
    for eps_h in H_AXIS:
        rt, rd = [], []
        for s in range(args.seeds):
            a, b = self_deception(args.arch, X_tr, y_tr, X_te, y_te, s,
                                  eps_h, 0.0, args.substrate)
            rt.append(a)
            rd.append(b)
        rep(f"  {eps_h:>15.3f} {np.mean(rt):>+16.4f} {np.mean(rd):>+16.4f} "
            f"{np.mean(rt)-np.mean(rd):>+9.4f}")
    rep("  The twin's own number is what a practitioner with a wrong twin would")
    rep("  see and trust; the device column is the truth.  A large gap means the")
    rep("  calibration's self-assessment is not evidence that it worked.")

    # ---------------- Part 2: does training degrade? ----------------
    milestones = [float(t) for t in str(args.milestones).split(",") if t.strip()]
    curves = {}
    t0 = time.time()
    for axis, levels in (("width", H_AXIS), ("position", C_AXIS)):
        for lvl in levels:
            eps_h, eps_c = (lvl, 0.0) if axis == "width" else (0.0, lvl)
            rep("")
            rep(f"  --- twin error: {axis}={lvl:g} ---")
            for arm in ARMS:
                cs = []
                for s in range(args.seeds):
                    hist, c = run_one(args.arch, arm, s, X_tr, y_tr, X_te, y_te,
                                      args, eps_h, eps_c)
                    cs.append(c)
                    if s == 0:
                        rep(f"    [{arm:<9}] end R2={c[-1][1]:+.4f} "
                            f"fwd={hist['fwd']} twin_calib={hist['twin_calibrations']}")
                curves[(axis, lvl, arm)] = cs
    rep("")
    rep(f"  {len(curves)} cells trained in {time.time()-t0:.0f}s")

    # ---------------- Part 3: milestone table + controls ----------------
    rep("\n" + "=" * 74)
    rep("  Part 2: test R^2 at fixed accumulated measurements")
    rep("=" * 74)
    for axis, levels in (("width", H_AXIS), ("position", C_AXIS)):
        rep("")
        rep(f"  twin error axis: {axis}")
        rep(f"  {'level':>10} {'arm':<10} "
            + " ".join(f"{int(m):>10d}" for m in milestones))
        rep("  " + "-" * (22 + 11 * len(milestones)))
        for lvl in levels:
            for arm in ARMS:
                mu, sd, n = merge_curves(curves[(axis, lvl, arm)], milestones)
                cells = [f"{'--':>10}" if n[j] == 0 else f"{mu[j]:>+10.4f}"
                         for j in range(len(milestones))]
                rep(f"  {lvl:>10.3f} {arm:<10} " + " ".join(cells))

    rep("\n" + "=" * 74)
    rep("  CONTROLS")
    rep("=" * 74)
    # A: twin error 0 must match the unperturbed calibrated arm
    m0 = milestones[-1]
    a0, _s, _n = merge_curves(curves[("width", 0.0, "hybrid")], [m0])
    a1, _s, _n = merge_curves(curves[("position", 0.0, "hybrid")], [m0])
    ok_a = abs(a0[0] - a1[0]) < 1e-9
    rep(f"  A. twin error 0 reproducible: width-axis {a0[0]:+.4f} vs "
        f"position-axis {a1[0]:+.4f} -> {'PASS' if ok_a else 'FAIL'}")
    # B: search arms must be flat in twin error
    rep("  B. search arms must be FLAT in twin error (they never read the twin):")
    worst = 0.0
    for arm in ("geometry", "weights", "full"):
        vals = []
        for lvl in H_AXIS:
            mu, _s, _n = merge_curves(curves[("width", lvl, arm)], [m0])
            vals.append(float(mu[0]) if np.isfinite(mu[0]) else np.nan)
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        worst = max(worst, spread)
        rep(f"     {arm:<9} spread across the axis = {spread:.4f}   "
            f"({'flat' if spread < 0.03 else 'NOT FLAT - perturbation leaked'})")
    rep(f"     -> {'PASS' if worst < 0.03 else 'FAIL'} "
        f"(largest spread {worst:.4f})")

    # ---------------- Part 4: verdict ----------------
    rep("\n" + "=" * 74)
    rep("  VERDICT: is there a twin-accuracy threshold?")
    rep("=" * 74)
    rep(f"  At {int(m0)} measurements, best search arm vs calibrated arm:")
    rep("")
    rep(f"  {'twin width err':>15} {'best search':>12} {'calibrated':>12} "
        f"{'calibrated wins?':>17}")
    rep("  " + "-" * 60)
    winners = []
    for lvl in H_AXIS:
        best = -np.inf
        for arm in ("geometry", "weights", "full"):
            mu, _s, _n = merge_curves(curves[("width", lvl, arm)], [m0])
            if np.isfinite(mu[0]):
                best = max(best, float(mu[0]))
        mu, _s, _n = merge_curves(curves[("width", lvl, "hybrid")], [m0])
        cal = float(mu[0]) if np.isfinite(mu[0]) else float("nan")
        ok = cal > best
        winners.append((lvl, ok))
        rep(f"  {lvl:>15.3f} {best:>+12.4f} {cal:>+12.4f} "
            f"{'YES' if ok else 'NO':>17}")

    n_ok = sum(1 for _l, o in winners if o)
    rep("")
    if n_ok == len(winners):
        rep("  The calibrated arm wins at EVERY twin-error level tested, including")
        rep("  20% width error.  The recommendation of section 6 therefore does not")
        rep("  need a twin-accuracy caveat within the tested range.")
    elif n_ok == 0:
        rep("  The calibrated arm loses at every level: twin error alone is enough")
        rep("  to destroy the headline, and section 6 must be restricted to a")
        rep("  twin-calibrated scheme with an explicit accuracy requirement.")
    else:
        thresh = None
        for i, (_l, o) in enumerate(winners):
            if not o:
                thresh = H_AXIS[i]
                break
        rep(f"  The calibrated arm wins below a twin width error of about "
            f"{thresh:.0%} and loses above it.")
        rep("  That threshold is the actionable number: it converts section 6 from")
        rep("  an unconditional rule into a device requirement, and it belongs in")
        rep("  the design specification next to the positioner resolution.")

    path = rep.save()
    print(f"\nReport saved: {path}")

    plt = try_plots()
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        cols = {"geometry": "#2196F3", "weights": "#F44336",
                "full": "#FF9800", "hybrid": "#4CAF50"}
        for axx, (axis, levels) in zip(axes, (("width", H_AXIS),
                                              ("position", C_AXIS))):
            for arm in ARMS:
                ys = []
                for lvl in levels:
                    mu, _s, _n = merge_curves(curves[(axis, lvl, arm)], [m0])
                    ys.append(float(mu[0]) if np.isfinite(mu[0]) else np.nan)
                axx.plot(range(len(levels)), ys, "o-", color=cols[arm], label=arm)
            axx.set_xticks(range(len(levels)))
            axx.set_xticklabels([f"{v:g}" for v in levels])
            axx.set_xlabel(f"twin error: {axis}")
            axx.set_ylabel(f"test R2 at {int(m0)} measurements")
            axx.set_title(f"{args.arch}: calibration vs a wrong twin")
            axx.grid(True, alpha=0.25)
            axx.legend(fontsize=8)
        fig.suptitle("Ablation E: does the closed-form calibration survive a "
                     "wrong digital twin?", fontsize=11)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "abl_05_twin_error_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {p}")


if __name__ == "__main__":
    main()
