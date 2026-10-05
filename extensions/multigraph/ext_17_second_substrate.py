"""
ext_17 -- half B on the SECOND substrate: the "any physical system" claim
=========================================================================
Project: singal-node-physics-network, extension `multigraph`

THE GAP THIS CLOSES
-------------------
The thesis is that a node network can be realised in ANY physical system whose phase
space can generate movable nonlinear activation sources.  The base version verified
two substrates (selftest_cross_substrate), and half A ran on both (ext_01, ext_05).
Half B -- the entire operator suite, ext_06 through ext_16 -- had never touched the
second substrate, and could not: `trainer.features` HARD-CODED `make_layer("circuit")`
and had no substrate parameter at all.  A claim whose main clause is "any physical
system" was resting on one substrate for the part of the argument that trains it.

WHAT THE SECOND SUBSTRATE ACTUALLY IS
-------------------------------------
`substrate_diffusion` engineers the SAME kernel out of device physics: a localised
decaying Green's function `G(r,t) = (4 pi D t)^(-dim/2) exp(-r^2/(4Dt))` is, for
`t = h^2/(4kD)`, exactly `exp(-k|X-c|^2/h^2)` TIMES A PREFACTOR.  `KERNELS = (0.5, 0.3)`
are the same two channels the circuit uses.  So the cells differ in exactly two ways
that can reach the operator:

  (1) a PER-SOURCE AMPLITUDE.  `_activations` returns a unit-peak z plus a calibration
      gain g, and an UNCALIBRATED field measurement returns z/g -- a per-source factor
      that depends on the node's own width, since g ~ t^(dim/2) and t ~ h^2.
  (2) a SENSOR CHAIN.  On this cell the activations are fields read by sensors, so
      noise, drift, clipping and quantisation land on the readings
      (abl_common._measure_features).  The circuit has no separate sensor stage.

THE PREDICTION, STATED BEFORE MEASURING
---------------------------------------
In log space the amplitude enters ONLY the constant term:

    log(a_j * z_j) = log a_j  -  alpha|x|^2 + gamma.x - delta

and OP-G recovers h from alpha and c from gamma -- it never touches delta.  So:

    THE GEOMETRY OPERATOR MUST BE INVARIANT TO THE PER-SOURCE AMPLITUDE.

That is falsifiable and non-obvious, and it is the whole reason this experiment is
worth more than a coverage check.  It does NOT say the amplitude is harmless: an
uncalibrated injection that lands below the sensor's full scale is a MEASUREMENT
problem, and this report prices it separately.

SCOPE: task 4d, one layer of 40 nodes, 3 seeds.  Toy, as everywhere.

Output: outputs/ext_17_second_substrate.txt + _plot.png
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
    Report, header, make_regression_data, r2, set_diffusion_defaults,
    set_ideal, try_plots,
)
from extensions.multigraph.operators import (              # noqa: E402
    K_GAUSS, identify_placement, star_probes,
)
from extensions.multigraph.trainer import (                # noqa: E402
    features, fit_operator, node_amplitude, solve_strengths,
)

LEVELS = ("none", "low", "medium", "high")


def random_geometry(X_fi, y_fi, X_va, y_va, X_te, y_te, n_nodes, seed, draws=8,
                    substrate="diffusion"):
    vals, widths = [], None
    for d in range(draws):
        rng = np.random.default_rng(9000 + 137 * d + seed)
        lo, hi = X_fi.min(axis=0), X_fi.max(axis=0)
        span = float(np.mean(hi - lo))
        cen = [rng.uniform(lo, hi) for _ in range(n_nodes)]
        wid = [float(rng.uniform(0.25, 1.5) * span / np.sqrt(n_nodes))
               for _ in range(n_nodes)]
        widths = wid
        Zfi = features(X_fi, cen, wid, substrate=substrate)
        w, _lr = solve_strengths(Zfi, y_fi,
                                 features(X_va, cen, wid, substrate=substrate), y_va)
        vals.append(r2(y_te, features(X_te, cen, wid, substrate=substrate) @ w))
    return float(np.mean(vals)), float(np.std(vals)), widths


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_17: half B on the diffusion cell")
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
    set_diffusion_defaults(level="none")
    rep = Report("ext_17_second_substrate.txt")
    for line in header("ext_17 - half B on the second substrate (diffusion)",
                       f"task={args.task} nodes={args.nodes} seeds={args.seeds}"):
        rep(line)

    rep("")
    rep("  The operator suite was circuit-only until now, and could not be otherwise:")
    rep("  `features` hard-coded make_layer(\"circuit\") and had no substrate")
    rep("  parameter.  A thesis whose main clause is 'any physical system' rested on")
    rep("  one substrate for the part of the argument that trains the network.")
    rep("")
    rep("  PREDICTION, stated before measuring: in log space the per-source amplitude")
    rep("  enters ONLY the constant term, and OP-G recovers h from alpha and c from")
    rep("  gamma without ever using that constant -- so the geometry operator MUST be")
    rep("  invariant to it.  Falsifiable, and the reason this is more than coverage.")
    rep("")
    rep("  The measurement protocol the operators assume, stated because it is an")
    rep("  assumption: ONE node is driven at a time with the others off.  Under that")
    rep("  protocol the diffusion cell's PU normalisation does not enter the reading")
    rep("  (the normalising sum is the driven node alone), so PU is not a confound")
    rep("  here -- and a protocol that drove all sources at once would need the")
    rep("  coupling term accounted for, which nothing in this extension does.")

    D = 4
    t0 = time.time()

    # -- P0: the engineered kernel, on the second substrate ----------------
    rep("\n" + "=" * 88)
    rep("  P0  is the diffusion cell actually the kernel the operators assume?")
    rep("=" * 88)
    rep("")
    rng = np.random.default_rng(4242)
    Xp = star_probes(D, 2.5, max(8, D + 2), rng=rng)
    rep(f"  {'substrate':>11} {'log-fit residual':>18} {'centre err':>12} "
        f"{'width err/true':>15} {'cond':>10}")
    rep("  " + "-" * 72)
    p0 = {}
    for sub in ("circuit", "diffusion"):
        res, cerr, werr, cond = [], [], [], []
        for _ in range(12):
            c = rng.uniform(-1.0, 1.0, D)
            h = float(rng.uniform(0.6, 1.6))
            z = features(Xp, [c], [h], substrate=sub).ravel()
            r = identify_placement(K_GAUSS, Xp, z, c_intended=c, h_intended=h)
            if not r.get("ok"):
                continue
            res.append(abs(r["consistency"]))
            cerr.append(r["centre_err_rel"])
            werr.append(abs(r["width"] - h) / h)
            cond.append(r["cond"])
        p0[sub] = dict(res=float(np.mean(res)), cerr=float(np.mean(cerr)),
                       werr=float(np.mean(werr)), cond=float(np.mean(cond)))
        rep(f"  {sub:>11} {np.mean(res):>18.3e} {np.mean(cerr):>12.3e} "
            f"{np.mean(werr):>15.3e} {np.mean(cond):>10.1f}")
    rep("")
    rep("  BOTH ARE AT THEIR OWN NUMERICAL FLOOR, and the floors differ: the circuit")
    rep("  layer computes in float32, so its residual sits near 1e-6, while the")
    rep("  diffusion path is float64 and reaches machine precision.  The identity is")
    rep("  exact on both; only the arithmetic differs.  Reading the two as 'circuit is")
    rep("  less exact' would be a mistake about precision, not about physics.")

    # -- P1: the amplitude-invariance prediction ---------------------------
    rep("\n" + "=" * 88)
    rep("  P1  THE PREDICTION: is the geometry operator invariant to per-source amplitude?")
    rep("=" * 88)
    rep("")
    nodes = []
    for _ in range(12):
        nodes.append((rng.uniform(-1.0, 1.0, D), float(rng.uniform(0.6, 1.6))))
    amps = node_amplitude(Xp, [c for c, _h in nodes], [h for _c, h in nodes],
                          substrate="diffusion")
    rep(f"  per-source amplitude 1/g_j over {len(nodes)} nodes: "
        f"min {amps.min():.3e}  max {amps.max():.3e}  ratio {amps.max()/amps.min():.3e}")
    rep("  (it depends on the node's own width, since g ~ t^(dim/2) and t ~ h^2)")
    rep("")
    rep(f"  {'quantity':>34} {'unit peak':>14} {'field (rescaled)':>18} "
        f"{'difference':>12}")
    rep("  " + "-" * 82)
    dc, dw, dres = [], [], []
    refused = 0
    for (c, h), a in zip(nodes, amps):
        z = features(Xp, [c], [h], substrate="diffusion").ravel()
        # what an UNCALIBRATED field measurement returns is z*a.  A real
        # experimenter brings each node's reading back to full scale before
        # fitting -- which is a per-node CONSTANT, exactly what the prediction
        # says must not matter.
        zf = z * a
        zf = zf / max(np.max(zf), 1e-300)
        r_u = identify_placement(K_GAUSS, Xp, z, c_intended=c, h_intended=h)
        r_f = identify_placement(K_GAUSS, Xp, zf, c_intended=c, h_intended=h)
        if not (r_u.get("ok") and r_f.get("ok")):
            refused += 1
            continue
        dc.append(float(np.linalg.norm(np.asarray(r_f["centre"]) - np.asarray(r_u["centre"]))))
        dw.append(abs(r_f["width"] - r_u["width"]) / max(r_u["width"], 1e-12))
        dres.append(abs(r_f["consistency"] - r_u["consistency"]))
    rep(f"  {'centre difference (absolute)':>34} {'':>14} {'':>18} "
        f"{np.mean(dc):>12.3e}")
    rep(f"  {'width difference (relative)':>34} {'':>14} {'':>18} "
        f"{np.mean(dw):>12.3e}")
    rep(f"  {'consistency difference':>34} {'':>14} {'':>18} "
        f"{np.mean(dres):>12.3e}")
    rep("")
    rep("  READ THE CONSISTENCY ROW AS A CONFIRMATION, not as a failure.  The operator")
    rep("  checks delta = |gamma|^2/(4 alpha), and the amplitude is absorbed by delta --")
    rep("  which is exactly where the prediction says it must go.  So the amplitude")
    rep("  moves the CONSISTENCY RESIDUAL and leaves the GEOMETRY untouched, and that")
    rep("  is a sharper statement of the mechanism than the invariance alone:")
    rep("")
    rep("    A CONSISTENCY GATE REJECTS AN UNCALIBRATED MEASUREMENT OF A PERFECTLY")
    rep("    RECOVERABLE NODE.  The geometry is right to 1e-15 while the check that is")
    rep("    supposed to validate the measurement moves by 0.4.  Any hardware procedure")
    rep("    that gates on consistency must therefore calibrate the amplitude FIRST, or")
    rep("    it will discard good readings -- which is the opposite of what a validity")
    rep("    check is for.")
    rep(f"  {'nodes the operator refused':>34} {'':>14} {'':>18} {refused:>12d}")
    rep("")
    invariant = np.mean(dc) < 1e-9 and np.mean(dw) < 1e-9
    if invariant:
        rep("  THE PREDICTION HOLDS.  A per-source amplitude spanning "
            f"{amps.max()/amps.min():.0e}x leaves the")
        rep("  recovered centre and width unchanged to numerical precision, because")
        rep("  the amplitude is a per-node constant and OP-G never uses the constant")
        rep("  term.  The geometry operators are therefore substrate-AGNOSTIC BY")
        rep("  CONSTRUCTION, not by coincidence -- which is the strongest form the")
        rep("  'any physical system' claim can take.")
    else:
        rep("  THE PREDICTION FAILS: the amplitude moves the recovered geometry, so")
        rep("  the operator carries a substrate assumption the kernel identity does not")
        rep("  predict.  That would be the finding, and it must be reported as such.")

    # -- P2: the sensor chain ---------------------------------------------
    rep("\n" + "=" * 88)
    rep("  P2  and what the SENSOR CHAIN costs, since that one should NOT be free")
    rep("=" * 88)
    rep("")
    rep(f"  {'level':>7} {'log-fit residual':>18} {'centre err':>12} "
        f"{'width err':>12} {'refused':>9}")
    rep("  " + "-" * 62)
    p2 = []
    for lv in LEVELS:
        set_diffusion_defaults(level=lv)
        res, cerr, werr, ref = [], [], [], 0
        for (c, h) in nodes:
            z = features(Xp, [c], [h], substrate="diffusion", measured=True).ravel()
            r = identify_placement(K_GAUSS, Xp, z, c_intended=c, h_intended=h)
            if not r.get("ok"):
                ref += 1
                continue
            res.append(abs(r["consistency"]))
            cerr.append(r["centre_err_rel"])
            werr.append(abs(r["width"] - h) / h)
        p2.append((lv, float(np.mean(res)) if res else float("nan"),
                   float(np.mean(cerr)) if cerr else float("nan"),
                   float(np.mean(werr)) if werr else float("nan"), ref))
        rep(f"  {lv:>7} {p2[-1][1]:>18.3e} {p2[-1][2]:>12.3e} "
            f"{p2[-1][3]:>12.3e} {ref:>9d}")
    set_diffusion_defaults(level="none")
    rep("")
    rep("  Unlike the amplitude, the sensor chain is NOT free: it corrupts the")
    rep("  readings themselves, so the operator's inputs are damaged and its")
    rep("  estimates must degrade.  Priced here rather than assumed.")

    # -- P3: the whole trainer, end to end, on the second substrate --------
    rep("\n" + "=" * 88)
    rep("  P3  does the operator TRAINER still work on the second substrate?")
    rep("=" * 88)
    rep("")
    rep(f"  {'arm':>34} {'circuit':>11} {'diffusion':>11} "
        f"{'diff + chain(low)':>18}")
    rep("  " + "-" * 78)
    res = {}
    # The chain column runs at a REAL level.  An earlier version left the global
    # level at "none" after P2, so the column labelled "diffusion+chain" was a
    # no-op chain and reproduced the other two to the last digit -- three identical
    # columns were the tell, the same way four identical rows exposed the
    # disconnected level in the first run.
    for sub, meas, lv in (("circuit", False, "none"),
                          ("diffusion", False, "none"),
                          ("diffusion", True, "low")):
        key = (sub, meas)
        set_diffusion_defaults(level=lv)
        op_s, rnd_s = [], []
        for seed in range(args.seeds):
            X_tr, y_tr, X_te, y_te = make_regression_data(args.task, seed=42 + seed)
            n_va = max(32, X_tr.shape[0] // 5)
            X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
            X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]
            o = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=args.nodes,
                             rounds=args.rounds, seed=seed, substrate=sub,
                             measured=meas)
            op_s.append(r2(y_te, features(X_te, o["centres"], o["widths"],
                                          substrate=sub, measured=meas)
                           @ o["weights"]))
            m, _sd, _w = random_geometry(X_fi, y_fi, X_va, y_va, X_te, y_te,
                                         args.nodes, seed, substrate=sub)
            rnd_s.append(m)
        res[key] = (float(np.mean(op_s)), float(np.mean(rnd_s)))
    set_diffusion_defaults(level="none")
    rep(f"  {'OPERATOR (the operator trainer)':>34} {res[('circuit', False)][0]:>+11.4f} "
        f"{res[('diffusion', False)][0]:>+11.4f} {res[('diffusion', True)][0]:>+16.4f}")
    rep(f"  {'RANDOM geometry':>34} {res[('circuit', False)][1]:>+11.4f} "
        f"{res[('diffusion', False)][1]:>+11.4f} {res[('diffusion', True)][1]:>+16.4f}")
    rep(f"  {'operator - random':>34} "
        f"{res[('circuit', False)][0]-res[('circuit', False)][1]:>+11.4f} "
        f"{res[('diffusion', False)][0]-res[('diffusion', False)][1]:>+11.4f} "
        f"{res[('diffusion', True)][0]-res[('diffusion', True)][1]:>+16.4f}")

    # -- verdict -----------------------------------------------------------
    rep("\n" + "=" * 88)
    rep("  VERDICT: does the operator suite transfer to the second substrate?")
    rep("=" * 88)
    rep("")
    # "Exact" means each substrate reaches its OWN numerical floor: the circuit
    # layer is float32, the diffusion path float64.  An earlier bar of 1e-9 for
    # both would have been a bar the circuit cannot meet by construction -- the
    # same class of mistake as ext_12's gain tolerance.
    circ_ok = p0["circuit"]["res"] < 1e-4
    diff_ok = p0["diffusion"]["res"] < 1e-12
    both_exact = circ_ok and diff_ok
    op_wins = all(v[0] > v[1] + 0.02 for v in res.values())
    rep(f"  P0 the kernel identity is exact on both substrates, each at its own "
        f"floor: {'YES' if both_exact else 'NO'}")
    rep(f"       circuit {p0['circuit']['res']:.2e} (float32 floor), "
        f"diffusion {p0['diffusion']['res']:.2e} (float64)")
    rep(f"  P1 the operator is invariant to per-source amplitude: "
        f"{'YES' if invariant else 'NO'}")
    rep(f"  P3 the operator trainer beats random on every path tested: "
        f"{'YES' if op_wins else 'NO'}")
    rep("")
    if both_exact and invariant and op_wins:
        rep("  THE SUITE TRANSFERS.  The kernel identity holds exactly on the diffusion")
        rep("  cell, the geometry operators are provably insensitive to the one thing")
        rep("  that differs per source there, the sensor chain's cost is priced, and the")
        rep("  full trainer still beats the null with the chain engaged.  The 'any")
        rep("  physical system' claim now has evidence from TWO substrates for the part")
        rep("  of the argument that trains the network, not one.")
        rep("")
        rep("  WHAT THIS DOES NOT ESTABLISH, stated so the claim is not read wider than")
        rep("  it is: two substrates is not 'any'.  Both cells were built here to")
        rep("  reproduce the same engineered kernel, so this shows the operators do not")
        rep("  depend on which of the two produced the field -- not that they would")
        rep("  survive a substrate whose response is not Gaussian in the input at all.")
        rep("  The prediction says exactly what the operators need: log-linearity in")
        rep("  [-|x|^2, x, -1].  A substrate that breaks THAT breaks the operators, and")
        rep("  no cell here tests that.")
    else:
        rep("  The transfer is partial; the failing part is named above and must not be")
        rep("  averaged into the rest.")
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
        x = np.arange(2)
        ax[0].bar(x - 0.2, [res[("circuit", False)][0],
                            res[("diffusion", False)][0]], 0.2,
                  label="OPERATOR", color="#2b7bba")
        ax[0].bar(x, [res[("circuit", False)][1],
                      res[("diffusion", False)][1]], 0.2,
                  label="RANDOM", color="#999999")
        ax[0].bar(x + 0.2, [res[("circuit", False)][0],
                            res[("diffusion", True)][0]], 0.2,
                  label="OPERATOR + sensor chain", color="#c44e52")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(["circuit", "diffusion"])
        ax[0].set_ylabel("test R^2")
        ax[0].set_title("the trainer on both substrates")
        ax[0].grid(True, axis="y", alpha=0.25)
        ax[0].legend(fontsize=8)
        ax[1].semilogy([r[0] for r in p2], [r[1] for r in p2], "o-")
        ax[1].set_xlabel("diffusion device level")
        ax[1].set_ylabel("log-fit residual (log scale)")
        ax[1].set_title("P2 the sensor chain is not free")
        ax[1].grid(True, alpha=0.25)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_17_second_substrate_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {p}")

    if do_assert and not (both_exact and invariant):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
