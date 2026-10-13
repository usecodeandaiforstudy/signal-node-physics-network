"""
ext_16 -- is the `steps` failure the BASIS, the PLACEMENT, or both?
==================================================================
Project: singal-node-physics-network, extension `multigraph`

THE QUESTION, AND WHY IT NEEDS ITS OWN EXPERIMENT
-------------------------------------------------
ext_13 (12 seeds) measured that on the discontinuous task `steps` the operator
placement is WORSE than random geometry (-0.0297 against +0.0573).  That is a
statement about the placement.  It is NOT the same as "the operators failed", and
it is very much not the same as "the network failed", and the three have to be
separated by measurement rather than by wording:

    the DEVICE      produces Gaussian nodes; it does not care what the target is
    the NETWORK     40 Gaussian nodes with a linear readout
    the PLACEMENT   which 40 nodes, chosen by the operator trainer
    the STRENGTHS   the readout weights, solved in closed form

`steps` is `sign(sin(3*x1)) + 0.5*cos(2*x2)`, a SQUARE WAVE in x1.  A finite sum of
Gaussians is smooth, so a basis limitation is expected -- and if that is the whole
story, no placement rule can fix it and the operator's deficit is second order.  If
instead a good placement CAN fit `steps`, then the operator's deficit is the finding
and it is a real defect.

WHAT IS MEASURED
----------------
  A  BEST-FOUND on `steps`: restarts of a wide-pool constructor, validation-picked,
     exactly as ext_14 built it.  If best-found is also ~0, the BASIS is the binding
     constraint and the placement question is moot.  If best-found is clearly
     positive, the placement is the finding.
  B  THE MECHANISM, measured rather than asserted: OP-C proposes nodes where the
     residual is large, and on a jump the residual is largest AT the jump -- a place
     a Gaussian should not go.  The tell is the WIDTH distribution: approximating a
     step with smooth bumps forces very NARROW nodes, so the greedy widths on `steps`
     should collapse relative to `4d` and relative to the random arm's widths.
  C  THE CONTROL: the same three arms on `4d`, where the basis CAN fit the target,
     so a width collapse there would mean the statistic is measuring something else.

SCOPE: 1x40, task in {steps, 4d}, 3 seeds, 16 restarts.  Toy, as everywhere.

Output: outputs/ext_16_steps_diagnosis.txt + _plot.png
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
from extensions.multigraph.trainer import (                # noqa: E402
    features, fit_operator, solve_strengths,
)

TASKS = [("steps", "DISCONTINUOUS: sign(sin(3x1)) + 0.5cos(2x2)"),
         ("4d", "the smooth control: sin(3x1) + cos(2x2) + 0.5*x3*x4")]


def random_geometry(X_fi, y_fi, X_va, y_va, X_te, y_te, n_nodes, seed, draws=8):
    """
    The null hypothesis, averaged over several draws of its OWN seed.

    A single draw is not a baseline on `steps`: ext_10/13 and ext_16 use the same
    code with different RNG streams and got +0.0573 and -0.6521 for the SAME arm --
    the null's own seed moves it by 0.7, which is larger than any effect being
    compared against it.  Averaging, and reporting the spread, is the minimum needed
    to make the comparison meaningful.
    """
    vals, widths = [], None
    for d in range(draws):
        rng = np.random.default_rng(9000 + 137 * d + seed)
        lo, hi = X_fi.min(axis=0), X_fi.max(axis=0)
        span = float(np.mean(hi - lo))
        cen = [rng.uniform(lo, hi) for _ in range(n_nodes)]
        wid = [float(rng.uniform(0.25, 1.5) * span / np.sqrt(n_nodes))
               for _ in range(n_nodes)]
        widths = wid
        Zfi = features(X_fi, cen, wid)
        w, _lr = solve_strengths(Zfi, y_fi, features(X_va, cen, wid), y_va)
        vals.append(r2(y_te, features(X_te, cen, wid) @ w))
    return float(np.mean(vals)), float(np.std(vals)), widths


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_16: basis or placement?")
    ap.add_argument("--seeds", type=int, default=12,
                    help="12, to match the seed count at which ext_13's `steps` "
                         "verdict was settled; 3 reproduces a superseded number")
    ap.add_argument("--nodes", type=int, default=40)
    ap.add_argument("--restarts", type=int, default=16)
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
    rep = Report("ext_16_steps_diagnosis.txt")
    for line in header("ext_16 - is the steps failure the basis or the placement?",
                       f"nodes={args.nodes} seeds={args.seeds} "
                       f"restarts={args.restarts}"):
        rep(line)

    rep("")
    rep("  The three levels have to be separated by measurement, not by wording:")
    rep("    DEVICE     produces Gaussian nodes; it does not know the target")
    rep("    NETWORK    40 Gaussian nodes with a linear readout")
    rep("    PLACEMENT  which 40 nodes, chosen by the operator trainer")
    rep("    STRENGTHS  the readout weights, solved in closed form")
    rep("")
    rep("  A finite sum of Gaussians is SMOOTH, and `steps` is a square wave, so a")
    rep("  basis limitation is expected.  The question is whether it is the whole")
    rep("  story: if best-found is also ~0 then no placement can fix `steps` and the")
    rep("  operator's deficit there is second order; if best-found is clearly")
    rep("  positive, the placement deficit is the finding.")

    t0 = time.time()
    out = {}
    for task, desc in TASKS:
        rep("\n" + "=" * 88)
        rep(f"  {task}: {desc}")
        rep("=" * 88)
        rep("")
        rows = []
        for seed in range(args.seeds):
            X_tr, y_tr, X_te, y_te = make_regression_data(task, seed=42 + seed)
            n_va = max(32, X_tr.shape[0] // 5)
            X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
            X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]

            g_out = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=args.nodes,
                                 rounds=args.rounds, seed=seed)
            g = r2(y_te, features(X_te, g_out["centres"], g_out["widths"])
                   @ g_out["weights"])
            rnd, rnd_sd, rnd_wid = random_geometry(X_fi, y_fi, X_va, y_va, X_te,
                                                  y_te, args.nodes, seed)
            # best-found: restarts of the SAME configuration as the operator arm.
            # An earlier version restarted with a WIDE pool (n_random=32) while the
            # operator arm ran at 4, so the comparison was between two constructors
            # rather than a bound -- and on `steps` it produced the absurdity of a
            # "best-found" BELOW the operator arm.  Same configuration, same data
            # seed, different search seed.
            pool = []
            for r in range(args.restarts):
                o = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=args.nodes,
                                 rounds=args.rounds, seed=1000 + 17 * r + seed)
                tv = r2(y_te, features(X_te, o["centres"], o["widths"])
                        @ o["weights"])
                vv = r2(y_va, features(X_va, o["centres"], o["widths"])
                        @ o["weights"])
                pool.append((vv, tv, o["widths"], o["centres"]))
            # the operator arm's own run is IN the pool, so best-found dominates it
            pool.append((r2(y_va, features(X_va, g_out["centres"],
                                           g_out["widths"]) @ g_out["weights"]),
                         g, g_out["widths"], g_out["centres"]))
            pool.sort(key=lambda kv: -kv[0])
            bf, bf_wid = pool[0][1], pool[0][2]
            rows.append(dict(g=g, rnd=rnd, rnd_sd=rnd_sd, bf=bf,
                             g_wid=np.asarray(g_out["widths"], dtype=float),
                             r_wid=np.asarray(rnd_wid, dtype=float),
                             b_wid=np.asarray(bf_wid, dtype=float)))
            print(f"  [ran ] {task:<6} seed {seed}  OP {g:+.4f}  RND {rnd:+.4f}  "
                  f"BEST-FOUND {bf:+.4f}", flush=True)

        g = np.array([r["g"] for r in rows])
        rn = np.array([r["rnd"] for r in rows])
        bf = np.array([r["bf"] for r in rows])
        gw = np.concatenate([r["g_wid"] for r in rows])
        rw = np.concatenate([r["r_wid"] for r in rows])
        bw = np.concatenate([r["b_wid"] for r in rows])
        out[task] = dict(g=g, rnd=rn, bf=bf, gw=gw, rw=rw, bw=bw,
                         rnd_sd=np.array([r["rnd_sd"] for r in rows]))

        rep(f"  {'arm':<26} {'test R^2':>10}")
        rep("  " + "-" * 38)
        rep(f"  {'OPERATOR (the trainer)':<26} {g.mean():>+10.4f}")
        rep(f"  {'RANDOM geometry':<26} {rn.mean():>+10.4f}"
            f"   (own-draw sd {out[task]['rnd_sd'].mean():.4f})")
        rep(f"  {'BEST-FOUND (restarts)':<26} {bf.mean():>+10.4f}")
        rep("")
        rep(f"  {'width statistic (mean over nodes)':<34} {'value':>10}")
        rep("  " + "-" * 46)
        rep(f"  {'OPERATOR widths':<34} {gw.mean():>10.4f}")
        rep(f"  {'RANDOM widths':<34} {rw.mean():>10.4f}")
        rep(f"  {'BEST-FOUND widths':<34} {bw.mean():>10.4f}")
        rep(f"  {'OPERATOR / RANDOM width ratio':<34} "
            f"{gw.mean()/rw.mean():>10.4f}")
        rep(f"  {'OPERATOR width coefficient of var':<34} "
            f"{gw.std()/gw.mean():>10.4f}")

    # -- verdict ------------------------------------------------------------
    rep("\n" + "=" * 88)
    rep("  VERDICT: which level failed on `steps`?")
    rep("=" * 88)
    rep("")
    st, sm = out["steps"], out["4d"]
    rep(f"  steps  OPERATOR {st['g'].mean():+.4f}  RANDOM {st['rnd'].mean():+.4f}  "
        f"BEST-FOUND {st['bf'].mean():+.4f}")
    rep(f"  4d     OPERATOR {sm['g'].mean():+.4f}  RANDOM {sm['rnd'].mean():+.4f}  "
        f"BEST-FOUND {sm['bf'].mean():+.4f}")
    rep("")
    bf_ok = st["bf"].mean() > 0.20
    rep(f"  can ANY placement of {args.nodes} Gaussian nodes fit `steps`?  "
        f"best-found {st['bf'].mean():+.4f} -> "
        f"{'YES' if bf_ok else 'NO'}")
    rep("")
    if not bf_ok:
        rep("  THE BASIS IS THE BINDING CONSTRAINT, NOT THE OPERATORS.  Restarts of")
        rep("  the same constructor -- a strictly stronger search than a single greedy")
        rep("  run -- also cannot fit a square wave with smooth bumps, so on `steps`")
        rep("  the network fails for a reason no placement rule can address.  The")
        rep("  operator's deficit against random on that task is therefore SECOND")
        rep("  ORDER: a real effect, in a cell where the basis was the wall.")
    else:
        rep("  THE PLACEMENT IS THE FINDING: a good placement CAN fit `steps`, so the")
        rep("  operator's deficit there is a defect in the rule rather than a wall in")
        rep("  the basis.")
    rep("")
    rep("  THE MECHANISM, measured: OP-C places nodes where the residual is large, and")
    rep("  a jump puts its largest residual AT the jump -- where a Gaussian should not")
    rep("  go.  Approximating a step with smooth bumps forces NARROW nodes, so the")
    rep("  tell is the width distribution:")
    rep("")
    for task in ("steps", "4d"):
        z = out[task]
        rep(f"    {task:<6} operator widths {z['gw'].mean():.4f} vs random "
            f"{z['rw'].mean():.4f}  -> ratio "
            f"{z['gw'].mean()/z['rw'].mean():.3f}")
    rep("")
    r_st = st["gw"].mean() / st["rw"].mean()
    r_sm = sm["gw"].mean() / sm["rw"].mean()
    if r_st < r_sm - 0.10:
        rep("  The operator's nodes ARE narrower on `steps` than on the smooth control")
        rep("  (ratio " + f"{r_st:.3f} against {r_sm:.3f}" + "), which is what chasing a")
        rep("  discontinuity looks like.  The control is what makes this readable: on a")
        rep("  target the basis CAN fit, the same statistic does not collapse.")
    else:
        rep(f"  The width statistic does NOT separate the two tasks "
            f"({r_st:.3f} against {r_sm:.3f}),")
        rep("  so the 'chasing the jump with narrow nodes' explanation is NOT supported")
        rep("  by this measurement and should not be repeated as though it were.")
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
        labels = [t for t, _d in TASKS]
        x = np.arange(len(labels))
        ax[0].bar(x - 0.27, [out[t]["g"].mean() for t in labels], 0.27,
                  label="OPERATOR", color="#2b7bba")
        ax[0].bar(x, [out[t]["rnd"].mean() for t in labels], 0.27,
                  label="RANDOM", color="#999999")
        ax[0].bar(x + 0.27, [out[t]["bf"].mean() for t in labels], 0.27,
                  label="BEST-FOUND", color="#c44e52")
        ax[0].axhline(0.0, color="k", lw=0.8)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels)
        ax[0].set_ylabel("test R^2")
        ax[0].set_title("can any placement fit it?")
        ax[0].grid(True, axis="y", alpha=0.25)
        ax[0].legend(fontsize=8)
        ax[1].bar(x - 0.2, [out[t]["gw"].mean() for t in labels], 0.2,
                  label="operator widths", color="#2b7bba")
        ax[1].bar(x, [out[t]["rw"].mean() for t in labels], 0.2,
                  label="random widths", color="#999999")
        ax[1].bar(x + 0.2, [out[t]["bw"].mean() for t in labels], 0.2,
                  label="best-found widths", color="#c44e52")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels)
        ax[1].set_ylabel("mean node width")
        ax[1].set_title("are the nodes collapsing to chase a jump?")
        ax[1].grid(True, axis="y", alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_16_steps_diagnosis_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, '/')}")

    # The assert encodes what the measurement supports: a good placement CAN fit
    # `steps` (best-found clearly above zero), so the deficit is the rule.  At 3
    # seeds with a mismatched baseline this file first concluded the opposite --
    # a red guard for the wrong claim is as useless as one that can never pass.
    if do_assert and not bf_ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
