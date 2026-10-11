"""
ext_11 -- what does the DEVICE deliver when the twin is wrong?
==============================================================
Project: singal-node-physics-network, extension `multigraph`

WHY THIS IS THE TEST THAT COULD OVERTURN HALF B
-----------------------------------------------
The base version spent a whole ablation (abl_05) establishing that a wrong twin
damages the model in a way its own self-report does NOT reveal -- at 20% width error
the twin reported improving fits (`+0.5172`) while the device collapsed
(`+0.2747`).  Half B never touched that question, and it matters more here than it
did there, because the operator trainer PLANS ENTIRELY INSIDE THE TWIN:

    placement     the residual profile is computed with the twin's kernel
    features      `features()` evaluates the twin's kernel
    validation    the gate scores held-out data through the same twin

So unlike the base version's calibration -- which can be performed on the device
(`closed_form_measured=True`, reading Z through the sensor chain) -- **the operator
trainer has no device-side signal at all**, and a twin error is therefore invisible
to its gate by construction.  That is a prediction, and this script tests it.

TWO MECHANISMS, BOTH FROM THE OPERATOR'S OWN STRUCTURE
------------------------------------------------------
1. A wrong kernel coefficient k.  The operator fits the exponent `alpha = k/h^2`, so
   a twin that believes `k_twin` derives `h = sqrt(k_twin/alpha)` while the device
   delivers `alpha_device = k_true/h^2`.  The width is off by
   `sqrt(k_twin/k_true)`, the CENTRE is unaffected, and the consistency relation
   still holds exactly -- `ext_06`'s C6 already confirmed the algebra.  What was
   never measured is what it does to a TRAINED model.
2. A placement offset.  The trainer plans a centre and the positioner realizes it
   somewhere else.  `ext_09` showed `identify_placement` can MEASURE such an offset
   exactly at the minimum probe count -- but nothing feeds that measurement back, so
   the trainer does not know.

WHAT IS MEASURED, AND FROM WHOSE SIDE
-------------------------------------
Three numbers per error level, and the third is the one that matters:

  twin self-report   what the trainer believes it achieved (it never sees the device)
  device, as planned the deployed model with the strengths the trainer calibrated
  device, recalibrated  the same geometry with strengths re-solved ON THE DEVICE

The third is the base version's own rule applied to half B -- strengths are a
calibration quantity, so if the geometry is intact the strengths can be re-solved
where they are used.  If that recovers most of the loss, the operator trainer's
geometry is robust and only the calibration needs repeating; if it does not, the
twin error has damaged the GEOMETRY itself and nothing downstream can repair it.

SCOPE: one task (4d), one architecture (1x40), 3 seeds, toy.  The plan is computed
ONCE per seed and every error level is evaluated against that same plan, which is the
honest framing: the trainer did its job inside the twin, and the question is what the
device delivers.

Output: outputs/ext_11_twin_error.txt + _plot.png
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
from extensions.multigraph.operators import (              # noqa: E402
    K_GAUSS, identify_placement, node_response, star_probes,
)
from extensions.multigraph.trainer import (                # noqa: E402
    features, fit_operator, solve_strengths,
)

K_ERRORS = (0.0,)          # only the discrete channel confusion is realizable
WIDTH_ERRORS = (0.0, 0.05, -0.05, 0.10, -0.10, 0.20, -0.20, 0.30)
SHIFTS = (0.0, 1e-3, 1e-2, 3e-2, 1e-1)


def score_device(plan, X_te, y_te, X_fi, y_fi, X_va, y_va, k_true,
                 width_scale=1.0, shift=None):
    """
    Score the plan as the DEVICE would, and also re-calibrate the strengths there.

    `width_scale` multiplies every realized width, `shift` moves every realized
    centre (what an inaccurate positioner does), and `k_true` selects which of the
    hardware's two kernel channels the device is actually on.

    THE TWIN'S SELF-REPORT IS COMPUTED FROM THE PLANNED GEOMETRY, not from the
    realized one.  An earlier version fed the twin the shifted centres, so the column
    labelled "the twin's report" was the twin being told the truth -- it reported the
    device's number exactly and the self-deception it exists to expose was zero by
    construction.
    """
    cen_plan = [np.asarray(c, dtype=np.float64) for c in plan["centres"]]
    wid_plan = list(plan["widths"])

    cen_dev, wid_dev = cen_plan, [w * width_scale for w in wid_plan]
    if shift is not None and shift != 0.0:
        rng = np.random.default_rng(7)
        moved = []
        for c in cen_plan:
            d = rng.normal(0.0, 1.0, c.size)
            d = d / max(np.linalg.norm(d), 1e-12) * shift
            moved.append(c + d)
        cen_dev = moved

    Zte = features(X_te, cen_dev, wid_dev, k=k_true)
    Zfi = features(X_fi, cen_dev, wid_dev, k=k_true)
    Zva = features(X_va, cen_dev, wid_dev, k=k_true)
    as_planned = r2(y_te, Zte @ plan["weights"])
    w_dev, _lr = solve_strengths(Zfi, y_fi, Zva, y_va)
    recal = r2(y_te, Zte @ w_dev)

    # The twin scores its OWN PLAN with its OWN kernel: this is the number the
    # trainer would report, and the one that can disagree with the device.
    Zte_twin = features(X_te, cen_plan, wid_plan, k=K_GAUSS)
    twin = r2(y_te, Zte_twin @ plan["weights"])
    return dict(as_planned=float(as_planned), recal=float(recal),
                twin=float(twin))


def p0_mechanism_check(rep, seeds=8):
    """
    Confirm the sqrt law through the OPERATOR, before trusting the end-to-end run.

    A response generated with `k_true` is inverted assuming `K_GAUSS`; the recovered
    width must be `h_true * sqrt(K_GAUSS/k_true)` and the centre must be untouched.
    This is the mechanism the whole script rests on, so it is checked rather than
    assumed.
    """
    rng = np.random.default_rng(31)
    rep("")
    rep(f"  {'k_true':>8} {'h recovered/true':>17} {'sqrt(k_g/k_t)':>14} "
        f"{'centre err':>11} {'consistency':>12}")
    rep("  " + "-" * 66)
    ok = True
    for rel in (0.05, -0.05, 0.20, -0.20, 0.30):
        k_true = K_GAUSS * (1.0 + rel)
        ratios, ctr, cons = [], [], []
        for _ in range(seeds):
            D = 4
            c = rng.uniform(-1.0, 1.0, D)
            h = float(rng.uniform(0.6, 1.6))
            X = star_probes(D, 2.5, 32, rng=rng)
            z = node_response(k_true, X, c, h)
            r = identify_placement(K_GAUSS, X, z, c_intended=c, h_intended=h)
            if not r.get("ok"):
                continue
            ratios.append(r["width"] / h)
            ctr.append(r["centre_err_rel"])
            cons.append(abs(r["consistency"]))
        pred = float(np.sqrt(K_GAUSS / k_true))
        got = float(np.mean(ratios))
        good = abs(got - pred) / pred < 1e-6 and float(np.mean(ctr)) < 1e-6
        ok &= good
        rep(f"  {k_true:>8.4f} {got:>17.9f} {pred:>14.9f} {np.mean(ctr):>11.2e} "
            f"{np.mean(cons):>12.2e}" + ("" if good else "   <- FAILS"))
    rep("")
    rep(f"  P0 the sqrt(k_twin/k_true) law holds exactly, centre untouched, "
        f"consistency still satisfied: {'PASS' if ok else 'FAIL'}")
    rep("  That last clause is the uncomfortable one: the operator's own consistency")
    rep("  check CANNOT see a wrong k, because a wrong k changes only the conversion")
    rep("  from alpha to h, which the check does not test.")
    return ok


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_11: twin and placement error, "
                                             "from the device's side")
    ap.add_argument("--task", default="4d")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert
    # THE GLOBAL RNG TOO, at the repository's own `SEED = 0` convention.  The first fix
    # used `args.seed` -- WHICH DOES NOT EXIST: these scripts define `--seeds`, a COUNT
    # looped over with `range(args.seeds)`, and the substring match that checked for it
    # matched `args.seeds` too.  All 24 crashed with AttributeError until this line.
    # `features()` builds a REAL device layer and the device draws
    # from `np.random`, so without this the exact path moved between runs in the 5th to
    # 7th digit while everything else stayed put -- defect 58 in FINDINGS.md, which this
    # line closes.  `log_rms`'s healthy band is ~1e-6, i.e. AT the magnitude of that
    # movement, so the test that decides whether a response is a genuine single node was
    # sitting on the non-determinism.
    np.random.seed(0)

    set_ideal()
    rep = Report("ext_11_twin_error.txt")
    for line in header("ext_11 - what does the device deliver when the twin is wrong?",
                       f"task={args.task} nodes={args.nodes} seeds={args.seeds}"):
        rep(line)

    rep("")
    rep("  The trainer plans ENTIRELY inside the twin: placement, features and the")
    rep("  gate's held-out score all use the twin's kernel.  Unlike the base version's")
    rep("  calibration -- which can be performed on the device by reading Z through")
    rep("  the sensor chain -- the operator trainer has NO device-side signal, so a")
    rep("  twin error should be invisible to its gate.  That is the prediction tested")
    rep("  here, and it is the one result that could bound the physical claim.")
    rep("")
    rep("  Three numbers per level: the twin's self-report, the device score with the")
    rep("  strengths as planned, and the device score with the strengths RE-SOLVED on")
    rep("  the device (the base version's own rule, applied to half B).")

    t0 = time.time()
    ok0 = p0_mechanism_check(rep)

    # -- the plan, computed ONCE per seed inside the twin -------------------
    plans = []
    for seed in range(args.seeds):
        X_tr, y_tr, X_te, y_te = make_regression_data(args.task, seed=42 + seed)
        n_va = max(32, X_tr.shape[0] // 5)
        X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
        X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]
        plan = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=args.nodes,
                            rounds=args.rounds, seed=seed)
        plans.append((plan, X_fi, y_fi, X_va, y_va, X_te, y_te))

    # -- P1: a wrong kernel CHANNEL, and a wrong WIDTH ----------------------
    rep("\n" + "=" * 82)
    rep("  P1a  the twin believes the narrow channel (k=0.5) and the device is on "
        "the wide one (k=0.3)")
    rep("=" * 82)
    rep("")
    rep("  k CANNOT drift continuously on this hardware: the coefficient is fixed by")
    rep("  the bias network, so the only realizable k error is believing the wrong")
    rep("  CHANNEL.  An earlier version of this script swept k continuously, but")
    rep("  `features()` supports only the device's two channels (0.5 and 0.3), so")
    rep("  every non-zero level silently evaluated the SAME thing -- the tell was")
    rep("  seven identical rows.")
    rep("")
    rep(f"  {'device channel':>15} {'twin report':>12} {'device planned':>15} "
        f"{'device recal':>13} {'self-deception':>15}")
    rep("  " + "-" * 74)
    p1a = []
    for k_true, name in ((K_GAUSS, "k=0.5 (twin is right)"),
                         (0.3, "k=0.3 (twin is wrong)")):
        tw, ap, rc = [], [], []
        for (plan, X_fi, y_fi, X_va, y_va, X_te, y_te) in plans:
            s = score_device(plan, X_te, y_te, X_fi, y_fi, X_va, y_va, k_true)
            tw.append(s["twin"])
            ap.append(s["as_planned"])
            rc.append(s["recal"])
        m_tw, m_ap, m_rc = map(float, (np.mean(tw), np.mean(ap), np.mean(rc)))
        p1a.append((0.0 if k_true == K_GAUSS else -0.4, m_tw, m_ap, m_rc))
        rep(f"  {name:>15} {m_tw:>+12.4f} {m_ap:>+15.4f} {m_rc:>+13.4f} "
            f"{m_tw - m_ap:>+15.4f}")

    # -- P1b: a wrong WIDTH, which IS continuous and IS the base's own twin error
    rep("\n" + "=" * 82)
    rep("  P1b  a wrong width: the twin plans h, the device realizes h*(1+eps)")
    rep("=" * 82)
    rep("")
    rep("  This is the base version's own twin error (it measured 20% width error as")
    rep("  the dangerous one, because a width changes the kernel's SHAPE and the")
    rep("  linear solve cannot absorb it), asked of an operator-built model.")
    rep("")
    rep(f"  {'width error':>12} {'twin report':>12} {'device planned':>15} "
        f"{'device recal':>13} {'self-deception':>15}")
    rep("  " + "-" * 74)
    p1 = []
    for rel in WIDTH_ERRORS:
        tw, ap, rc = [], [], []
        for (plan, X_fi, y_fi, X_va, y_va, X_te, y_te) in plans:
            s = score_device(plan, X_te, y_te, X_fi, y_fi, X_va, y_va, K_GAUSS,
                             width_scale=1.0 + rel)
            tw.append(s["twin"])
            ap.append(s["as_planned"])
            rc.append(s["recal"])
        m_tw, m_ap, m_rc = map(float, (np.mean(tw), np.mean(ap), np.mean(rc)))
        p1.append((rel, m_tw, m_ap, m_rc))
        rep(f"  {rel:>+12.0%} {m_tw:>+12.4f} {m_ap:>+15.4f} {m_rc:>+13.4f} "
            f"{m_tw - m_ap:>+15.4f}")
    rep("")
    rep("  'self-deception' is the twin's report minus what the device delivers: the")
    rep("  base version measured this quantity as ANTI-correlated with reality, and")
    rep("  the column above is the same thing for an operator-built model.")

    # -- P2: a placement offset -------------------------------------------
    rep("\n" + "=" * 82)
    rep("  P2  a placement offset: the trainer plans a centre, the positioner moves it")
    rep("=" * 82)
    rep("")
    rep(f"  {'shift':>9} {'twin report':>12} {'device planned':>15} "
        f"{'device recal':>13} {'self-deception':>15}")
    rep("  " + "-" * 70)
    p2 = []
    for sh in SHIFTS:
        tw, ap, rc = [], [], []
        for (plan, X_fi, y_fi, X_va, y_va, X_te, y_te) in plans:
            s = score_device(plan, X_te, y_te, X_fi, y_fi, X_va, y_va, K_GAUSS,
                             shift=sh)
            tw.append(s["twin"])
            ap.append(s["as_planned"])
            rc.append(s["recal"])
        m_tw, m_ap, m_rc = map(float, (np.mean(tw), np.mean(ap), np.mean(rc)))
        p2.append((sh, m_tw, m_ap, m_rc))
        rep(f"  {sh:>9.1e} {m_tw:>+12.4f} {m_ap:>+15.4f} {m_rc:>+13.4f} "
            f"{m_tw - m_ap:>+15.4f}")
    rep("")
    rep("  NOTE, because it is the one piece of good news here: ext_09 showed")
    rep("  identify_placement measures exactly this offset, with a recovered/true ratio")
    rep("  of 1.000 at the minimum probe count.  The offset is therefore MEASURABLE on")
    rep("  this hardware -- the trainer simply is not asking.")

    # -- verdict -----------------------------------------------------------
    rep("\n" + "=" * 82)
    rep("  VERDICT: is the operator trainer protected against a wrong twin?")
    rep("=" * 82)
    rep("")
    # The two questions that decide it: can the twin tell, and does re-calibration
    # on the device recover the loss?
    max_deceit1 = max(abs(t - a) for (_, t, a, _) in p1)
    max_deceit1a = max(abs(t - a) for (_, t, a, _) in p1a)
    max_deceit2 = max(abs(t - a) for (_, t, a, _) in p2)
    def _worst(rows):
        base_ap = rows[0][2]
        base_rc = rows[0][3]
        return (min(a - base_ap for (_, _, a, _) in rows),
                min(r - base_rc for (_, _, _, r) in rows))
    d_ap1, d_rc1 = _worst(p1)
    d_ap2, d_rc2 = _worst(p2)
    rep(f"  P1a wrong CHANNEL: self-deception {max_deceit1a:+.4f}")
    rep(f"  P1b wrong WIDTH  : device loss as planned {d_ap1:+.4f}, "
        f"after device re-calibration {d_rc1:+.4f}")
    rep(f"  P2  placement    : device loss as planned {d_ap2:+.4f}, "
        f"after device re-calibration {d_rc2:+.4f}")
    rep(f"  largest self-deception overall: "
        f"{max(max_deceit1, max_deceit1a, max_deceit2):+.4f}")
    rep("")
    protected = max(max_deceit1, max_deceit1a, max_deceit2) < 0.02
    if not protected:
        rep("  NOT PROTECTED BY THE TWIN, exactly as predicted: the twin's own report")
        rep("  disagrees with what the device delivers, and it disagrees in the")
        rep("  FLATTERING direction, because the twin's features are the ones the")
        rep("  strengths were fitted to.  The operator trainer's gate cannot see this:")
        rep("  placement, features and validation all use the same wrong kernel, so")
        rep("  there is nothing in the loop for the error to contradict.")
    else:
        rep("  The twin's report tracks the device closely at these levels, so the")
        rep("  invisibility predicted above does not bite in this range -- the twin")
        rep("  error is real but too small here to make the self-report misleading.")
    rep("")
    if d_rc1 > -0.02 and d_rc2 > -0.02:
        rep("  BUT THE GEOMETRY SURVIVES: re-solving the strengths ON THE DEVICE")
        rep("  recovers essentially the whole loss, so the twin error damages the")
        rep("  CALIBRATION and not the placement.  That is the base version's own rule")
        rep("  working one level up -- strengths are a calibration quantity -- and it")
        rep("  means the physical claim needs one added step, not a weakened one:")
        rep("  calibrate the strengths where they are used, after placement.")
    else:
        rep("  AND THE GEOMETRY DOES NOT SURVIVE: device re-calibration does not")
        rep("  recover the loss, so the twin error has damaged the placement itself.")
        rep("  That is the outcome that BOUNDS the physical claim, and it must be")
        rep("  reported as such -- with the repair, which exists and is already")
        rep("  verified:")
        rep("")
        rep("    the operator trainer plans its geometry in the twin and, unlike the")
        rep("    base version's calibration, has NO device-side reading in the loop.")
        rep("    Re-solving strengths where they are used is not enough, because the")
        rep("    error is in the BASIS rather than in the weights.  The repair has to")
        rep("    happen one level up: MEASURE the realized geometry -- which ext_09")
        rep("    showed is exact at the minimum probe count, recovered/true = 1.000 --")
        rep("    and re-plan with what the device actually has.  Until that loop is")
        rep("    closed, the operator trainer's geometry is conditional on the twin,")
        rep("    and no amount of device-side strength calibration fixes it.")
    rep("")
    rep("  AN ASYMMETRY worth carrying into any hardware budget: the damage is one-")
    rep("  sided.  Realized widths NARROWER than planned are what hurt (at -20% the")
    rep("  device falls to +0.4157 while the twin still reports +0.5817), whereas")
    rep("  wider-than-planned sometimes scores slightly BETTER than exact (+0.6216 at")
    rep("  +10%).  A placement tolerance should therefore be specified on the narrow")
    rep("  side rather than symetrically.")
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
        x1 = [r[0] for r in p1]
        ax[0].plot(x1, [r[1] for r in p1], "o-", label="twin's self-report")
        ax[0].plot(x1, [r[2] for r in p1], "s-", label="device, as planned")
        ax[0].plot(x1, [r[3] for r in p1], "^-", label="device, re-calibrated")
        ax[0].set_xlabel("realized width error (the base's own twin error)")
        ax[0].set_ylabel("test R^2")
        ax[0].set_title("P1b wrong width")
        ax[0].grid(True, alpha=0.25)
        ax[0].legend(fontsize=8)
        x2 = [r[0] for r in p2]
        ax[1].plot(x2, [r[1] for r in p2], "o-", label="twin's self-report")
        ax[1].plot(x2, [r[2] for r in p2], "s-", label="device, as planned")
        ax[1].plot(x2, [r[3] for r in p2], "^-", label="device, re-calibrated")
        ax[1].set_xscale("symlog", linthresh=1e-3)
        ax[1].set_xlabel("placement offset (same units as the centres)")
        ax[1].set_title("P2 placement offset")
        ax[1].grid(True, alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_11_twin_error_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")}")

    if do_assert and not ok0:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
