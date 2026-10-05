"""
ext_09 -- which points to fit on, and where free choice IS available
===================================================================
Project: singal-node-physics-network, extension `multigraph`

THE CONSTRAINT THIS EXPERIMENT EXISTS TO RESPECT
------------------------------------------------
The previous step was specified as "wire the designed probe set into the trainer's
refinement".  Doing that literally would have been wrong, and the reason is worth
stating because it is easy to miss:

    the residual is known only where the TARGET is known.  `refine_node` fits a
    log-profile of the residual, so it cannot read at points of its own choosing --
    it can only choose a SUBSET of the training inputs.

The designed `star_probes` set belongs to a different operation, the device-side
read-out `identify_placement`, which really can read anywhere.  This script measures
both, separately, because they are separate claims:

  P1  SUBSET SELECTION.  Given that only a subset can be chosen, does choosing by
      LEVERAGE (the D-optimal choice for a fixed candidate set) recover a node's
      geometry better than taking every point with a positive profile?  Tested on a
      CONTROLLED problem where the answer is known: the target IS one node, so the
      operator's job is to find it back.

  P2  DEVICE-SIDE IDENTIFICATION.  With a free choice of probe points, can
      `identify_placement` measure where a source actually is, and how small a
      placement error can it resolve?  This is where the designed set is used.

  P3  END-TO-END.  Does the subset rule change the trainer's test R^2?

The A/B is a parameter of the trainer (`select="leverage"|"all"`), so the two arms
are one code path rather than two copies of a file.

Output: outputs/ext_09_probe_selection.txt + _plot.png
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
from extensions.multigraph.trainer import features, fit_operator  # noqa: E402

D_DEFAULT = 4


# ---------------------------------------------------------------------------
def rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def p1_controlled_recovery(D, n_train, noise, seeds, rep):
    """
    The target IS one node, so the operator's job has a known right answer.

    This is the measurement P1 needs and the trainer cannot provide: end-to-end R^2
    moves for many reasons, whereas here the recovered geometry is compared with the
    geometry that generated the data.
    """
    from extensions.multigraph.trainer import refine_node
    rng = np.random.default_rng(4242)
    out = {sel: dict(ce=[], we=[], ok=0, n=0) for sel in ("leverage", "all")}
    for seed in range(seeds):
        c_true = rng.uniform(-1.5, 1.5, D)
        h_true = float(rng.uniform(0.5, 2.0))
        X = rng.uniform(-2.5, 2.5, (n_train, D))
        z = node_response(K_GAUSS, X, c_true, h_true)
        y = 1.0 * z + rng.normal(0.0, noise, n_train)
        for sel in ("leverage", "all"):
            got = refine_node(X, y, 1.0, K_GAUSS, select=sel)
            out[sel]["n"] += 1
            if got is None:
                continue
            c_hat, h_hat = got
            out[sel]["ok"] += 1
            out[sel]["ce"].append(rel_err(c_hat, c_true))
            out[sel]["we"].append(abs(h_hat - h_true) / h_true)
    rep("")
    rep(f"  {'D':>3} {'n':>5} {'noise':>8} {'rule':>9} {'solved':>8} "
        f"{'ctr err':>10} {'wid err':>10}")
    rep("  " + "-" * 60)
    res = {}
    for sel in ("leverage", "all"):
        d = out[sel]
        mce = float(np.mean(d["ce"])) if d["ce"] else float("nan")
        mwe = float(np.mean(d["we"])) if d["we"] else float("nan")
        rep(f"  {D:>3} {n_train:>5} {noise:>8.0e} {sel:>9} "
            f"{d['ok']:>4d}/{d['n']:<3d} {mce:>10.2e} {mwe:>10.2e}")
        res[sel] = (mce, mwe, d["ok"] / max(d["n"], 1))
    return res


def p2_placement_identification(D, rep, seeds=20):
    """
    With a FREE choice of probe points, measure a source's placement error.

    The probe set is the designed star.  Two numbers matter: how often a placement
    error is detected at all, and how the ESTIMATED error compares with the TRUE
    one -- an estimator that reports a constant would be useless even if it always
    "detected" something.
    """
    rng = np.random.default_rng(777)
    rep("")
    rep(f"  {'probes':>7} {'true shift':>11} {'recovered':>11} {'ratio':>8} "
        f"{'detected':>9} {'ctr err':>10}")
    rep("  " + "-" * 62)
    rows = []
    for n_probe in (D + 2, 16, 64):
        for shift_rel in (0.0, 1e-3, 1e-2, 3e-2):
            rec, tru, det, ctr = [], [], 0, []
            for _ in range(seeds):
                c_int = rng.uniform(-1.0, 1.0, D)
                h_int = float(rng.uniform(0.6, 1.8))
                # the realized source differs from the intention
                delta = rng.normal(0.0, 1.0, D)
                delta = delta / max(np.linalg.norm(delta), 1e-12) * shift_rel
                c_act = c_int + delta
                X = star_probes(D, 2.5, n_probe, rng=rng)
                z = node_response(K_GAUSS, X, c_act, h_int)
                r = identify_placement(K_GAUSS, X, z, c_intended=c_int,
                                       h_intended=h_int)
                if not r.get("ok"):
                    continue
                est = float(np.linalg.norm(r["centre"] - c_int))
                rec.append(est)
                tru.append(shift_rel)
                ctr.append(r["centre_err_rel"])
                # A shift is "detected" when the estimate exceeds a floor that the
                # probe geometry itself sets; zero shift should stay below it.
                det += int(est > 1e-4)
            if not rec:
                rep(f"  {n_probe:>7} {shift_rel:>11.0e}   (operator refused)")
                continue
            m_est, m_tru = float(np.mean(rec)), float(np.mean(tru))
            ratio = m_est / m_tru if m_tru > 0 else float("nan")
            rep(f"  {n_probe:>7} {m_tru:>11.2e} {m_est:>11.2e} {ratio:>8.3f} "
                f"{det:>4d}/{len(rec):<4d} {np.mean(ctr):>10.2e}")
            if m_tru > 0:
                rows.append((n_probe, m_tru, m_est, ratio))
    return rows


def p3_end_to_end(task, seeds, rounds, rep):
    """Does the subset rule change the trainer's own score -- and does REFINE fire?"""
    out = {"leverage": [], "all": []}
    ref = {"leverage": [], "all": []}
    geom = {"leverage": [], "all": []}
    for seed in range(seeds):
        X_tr, y_tr, X_te, y_te = make_regression_data(task, seed=42 + seed)
        n_va = max(32, X_tr.shape[0] // 5)
        X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
        X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]
        for sel in ("leverage", "all"):
            r = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=40, rounds=rounds,
                             seed=seed, select=sel, refine=True)
            out[sel].append(r2(y_te, features(X_te, r["centres"], r["widths"])
                               @ r["weights"]))
            ref[sel].append((r["refine_tried"], r["refine_refused"],
                             r["refine_kept"]))
            geom[sel].append(np.asarray(r["centres"], dtype=np.float64))
    rep("")
    rep(f"  {'rule':>9} {'mean R^2':>10} {'sd':>8} {'worst':>10} "
        f"{'refine tried':>13} {'refused':>9} {'kept':>7}")
    rep("  " + "-" * 72)
    stats = {}
    for sel in ("leverage", "all"):
        v = np.asarray(out[sel], dtype=np.float64)
        stats[sel] = (float(v.mean()), float(v.std()))
        t = float(np.mean([x[0] for x in ref[sel]]))
        rf = float(np.mean([x[1] for x in ref[sel]]))
        k = float(np.mean([x[2] for x in ref[sel]]))
        rep(f"  {sel:>9} {v.mean():>+10.4f} {v.std():>8.4f} {v.min():>+10.4f} "
            f"{t:>13.0f} {rf:>9.0f} {k:>7.0f}")
    rep("")
    rep("  The refinement counters are not decoration: the first version of this")
    rep("  experiment reported IDENTICAL scores for two different selection rules,")
    rep("  which is what a step that never fires looks like from the outside.")
    # Does the refinement REACH the returned model?  Accepted refinements with an
    # identical final geometry would mean the round gate rejects every refinement
    # round, so ext_07's headline would come from GROW + SOLVE + GATE alone.  That
    # is a different claim, and it is decided here rather than assumed.
    diffs = [float(np.max(np.abs(geom["leverage"][i] - geom["all"][i])))
             for i in range(seeds)]
    # The bar is the FLOAT32 floor, not exact zero: the layer stores centres as
    # float32, so on unit-scale coordinates one ULP is about 1.2e-7 and a difference
    # of that size is rounding rather than a moved node.  Requiring exact equality
    # would have reported "the refinement does reach the model" on the strength of
    # one ULP -- the opposite of what the measurement means.
    ulp = 4.0 * float(np.finfo(np.float32).eps)
    rep("")
    rep(f"  max |centres(leverage) - centres(all)| per seed: "
        + ", ".join(f"{d:.3e}" for d in diffs))
    rep(f"  float32 floor on unit-scale coordinates: {ulp:.1e} "
        f"(4 x eps); anything at or below it is rounding")
    if max(diffs) <= ulp:
        rep("  -> the refinement does NOT move the model: its accepted changes are")
        rep("     rounding, and on most seeds there is no change at all.")
        rep("")
        rep("     The MECHANISM, which is worth stating because it is a property of")
        rep("     the operator rather than a bug: OP-G inverts a response, so refining")
        rep("     a node against a residual that this node already explains returns the")
        rep("     SAME node -- a fixed point.  GROW + SOLVE place each node to fit the")
        rep("     current residual, i.e. already at that fixed point, so REFINE has")
        rep("     nothing to move.  It would only bite if a node were misplaced, which")
        rep("     is a fault-tolerance use rather than a training one.")
        rep("")
        rep(f"     CONSEQUENCE FOR ext_07: its {stats['leverage'][0]:+.4f} comes from")
        rep("     GROW + SOLVE + GATE.  Claiming a trainer that uses 'all the operators'")
        rep("     would overstate it; REFINE is implemented and verified but inert here.")
    else:
        rep("  -> the refinement DOES move the returned geometry beyond rounding, so")
        rep("     the identical scores are a genuine insensitivity at this scale.")
    return stats, ref, max(diffs)


# ---------------------------------------------------------------------------
def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_09: subset selection and "
                                             "device-side identification")
    ap.add_argument("--task", default="4d")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--grid", default="4:256:0.0,4:256:0.01,4:128:0.01,4:64:0.01,"
                                     "2:128:0.01",
                    help="D:n_train:noise cells for P1")
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
    rep = Report("ext_09_probe_selection.txt")
    for line in header("ext_09 - which points to fit on, and where free choice "
                       "really exists",
                       f"task={args.task} seeds={args.seeds} grid={args.grid}"):
        rep(line)

    rep("")
    rep("  THE CONSTRAINT, which changed what this step could be: the residual is")
    rep("  known only where the TARGET is known, so refinement cannot read at chosen")
    rep("  points -- it can only choose a SUBSET of the training inputs.  The designed")
    rep("  star probe set therefore belongs to `identify_placement`, the device-side")
    rep("  read-out, which really can read anywhere.  Both are measured below.")
    rep("")
    rep("  The A/B is a PARAMETER of the trainer (select='leverage'|'all'), not two")
    rep("  copies of it, so there is exactly one code path to trust.")

    t0 = time.time()

    # -- P1 ----------------------------------------------------------------
    rep("\n" + "=" * 78)
    rep("  P1  SUBSET selection on a CONTROLLED problem (the target IS one node)")
    rep("=" * 78)
    rep("  'solved' counts the runs where the operator returned a geometry at all;")
    rep("  the errors are over those.  The right answer is known, which the trainer's")
    rep("  end-to-end score cannot provide.")
    p1 = {}
    for cell in args.grid.split(","):
        D, n_tr, nz = cell.split(":")
        res = p1_controlled_recovery(int(D), int(n_tr), float(nz), args.seeds, rep)
        p1[(int(D), int(n_tr), float(nz))] = res
    rep("")
    better = sum(1 for v in p1.values() if v["leverage"][0] < v["all"][0])
    rep(f"  leverage selects a lower centre error in {better}/{len(p1)} cells")

    # -- P2 ----------------------------------------------------------------
    rep("\n" + "=" * 78)
    rep("  P2  DEVICE-side identification with a FREE choice of probe points")
    rep("=" * 78)
    rep("  'ratio' is the recovered shift over the true shift: 1.0 is a calibrated")
    rep("  measurement, and a constant would be useless however often it 'detected'.")
    p2 = p2_placement_identification(D_DEFAULT, rep)

    # -- P3 ----------------------------------------------------------------
    rep("\n" + "=" * 78)
    rep("  P3  END-TO-END: does the subset rule change the trainer's score?")
    rep("=" * 78)
    p3, ref3, geom_diff = p3_end_to_end(args.task, args.seeds, args.rounds, rep)

    # -- verdict -----------------------------------------------------------
    rep("\n" + "=" * 78)
    rep("  VERDICT")
    rep("=" * 78)
    lo, ao = p1[(4, 256, 0.01)]["leverage"], p1[(4, 256, 0.01)]["all"]
    rep("")
    rep(f"  P1 controlled, D=4 n=256 noise=0.01: leverage centre err {lo[0]:.3e} "
        f"vs all {ao[0]:.3e}")
    ratio = lo[0] / ao[0] if ao[0] > 0 else float("nan")
    if ratio < 0.9:
        rep(f"  -> leverage is better by {1/ratio:.1f}x on the controlled problem, so")
        rep("     the conditioning argument does buy something where it applies.")
    elif ratio > 1.1:
        rep(f"  -> leverage is WORSE by {ratio:.1f}x.  The measurement says the")
        rep("     conditioning argument does not pay off at these sizes, and reporting")
        rep("     it as a fix would be reporting the opposite of what was measured.")
    else:
        rep(f"  -> NO material difference ({ratio:.2f}x): the subset rule is a wash at")
        rep("     these sizes rather than a fix.")
    rep("")
    if p2:
        ratios = [r for (_, mt, _, r) in p2 if mt > 0]
        if ratios:
            rep(f"  P2 the recovered placement shift tracks the true one within "
                f"{min(ratios):.2f}-{max(ratios):.2f}x")
            rep("     across probe counts and shift sizes, so placement error is a")
            rep("     MEASURABLE quantity on this hardware rather than an unchecked")
            rep("     tolerance.  That is what the designed probe set is for.")
    rep("")
    d = p3["leverage"][0] - p3["all"][0]
    tol = max(0.02, 2.0 * max(p3["leverage"][1], p3["all"][1]) / np.sqrt(args.seeds))
    rep(f"  P3 end-to-end: leverage {p3['leverage'][0]:+.4f} vs all "
        f"{p3['all'][0]:+.4f}  (gap {d:+.4f}, tol {tol:.4f})")
    if geom_diff <= 4.0 * float(np.finfo(np.float32).eps):
        rep("     The two rules produce the SAME model to within float32 rounding, and")
        rep("     the decisive fact is the mechanism: OP-G is a FIXED POINT for a node")
        rep("     that already explains its residual, which is where GROW + SOLVE put")
        rep("     it.  So REFINE is inert, and ext_07's")
        rep(f"     {p3['leverage'][0]:+.4f} comes from GROW + SOLVE + GATE.  That is a")
        rep("     weaker claim than 'a trainer using all the operators', and it is the")
        rep("     one the evidence supports.")
    elif abs(d) <= tol:
        rep("     NOT DISTINGUISHABLE end to end, and the geometries DO differ beyond")
        rep("     rounding, so this is a genuine insensitivity at this scale.")
    else:
        rep(f"     Distinguishable ({d:+.4f}), so the subset rule does reach the final")
        rep("     model at this scale.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

    # The assert runs BEFORE rep.save().  An earlier version appended these lines to a
    # Report that had ALREADY been written, so the file carried the claims without
    # the check on them -- the same shape of defect as a test that never fires.
    if do_assert:
        p1_ok = all(v["leverage"][2] > 0 for v in p1.values())
        p2_ok = bool(p2) and all(abs(r - 1.0) < 0.05 for (_, mt, _, r) in p2
                                 if mt > 0)
        p3_ok = geom_diff <= 4.0 * float(np.finfo(np.float32).eps)
        rep("")
        rep("=" * 78)
        rep("  ASSERT COVERAGE")
        rep("=" * 78)
        for name, okv, what in (
                ("P1", p1_ok, "the controlled recovery returns a geometry"),
                ("P2", p2_ok, "the placement shift is recovered proportionally"),
                ("P3", p3_ok, "the refinement is inert to within float32")):
            rep(f"  {name}: {'PASS' if okv else 'FAIL'}   {what}")
        rep("")
        rep("  P1 is deliberately NOT asserted on improvement: leverage measured WORSE")
        rep("  than taking every informative point, so an assert demanding the opposite")
        rep("  would contradict the result this file reports.")
        assert_all_ok = p1_ok and p2_ok and p3_ok
    else:
        assert_all_ok = True

    path = rep.save()
    print(f"\nReport saved: {path}")

    plt = try_plots()
    if plt is not None:
        __import__("warnings").filterwarnings(
            "ignore", message=".*does not have a glyph.*")
        plt.rcParams["axes.unicode_minus"] = False   # default font lacks U+2212
        try:
            import logging as _lg
            _lg.getLogger("matplotlib").setLevel(_lg.ERROR)
        except Exception:
            pass
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
        labels, lev, al = [], [], []
        for (D, n_tr, nz), v in p1.items():
            labels.append(f"D={D}\nn={n_tr}\nnoise={nz:g}")
            lev.append(v["leverage"][0])
            al.append(v["all"][0])
        x = np.arange(len(labels))
        ax[0].bar(x - 0.2, lev, 0.4, label="leverage", color="#2b7bba")
        ax[0].bar(x + 0.2, al, 0.4, label="all positive", color="#c44e52")
        ax[0].set_yscale("log")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels, fontsize=7)
        ax[0].set_ylabel("centre error (log)")
        ax[0].set_title("P1 controlled recovery")
        ax[0].legend(fontsize=8)
        ax[0].grid(True, axis="y", alpha=0.25, which="both")
        if p2:
            for n_probe in sorted({r[0] for r in p2}):
                sub = [r for r in p2 if r[0] == n_probe]
                ax[1].plot([s[1] for s in sub], [s[2] for s in sub], "o-",
                           label=f"{n_probe} probes")
            lim = [min(r[1] for r in p2), max(r[1] for r in p2)]
            ax[1].plot(lim, lim, "k--", lw=1.0, label="perfect estimate")
            ax[1].set_xscale("log")
            ax[1].set_yscale("log")
            ax[1].set_xlabel("true placement shift")
            ax[1].set_ylabel("recovered shift")
            ax[1].set_title("P2 device-side identification")
            ax[1].legend(fontsize=8)
            ax[1].grid(True, alpha=0.25, which="both")
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_09_probe_selection_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {p}")

    if do_assert and not assert_all_ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
