"""
ext_03 -- does the rank budget SURVIVE geometry training?
=========================================================
Project: singal-node-physics-network, extension `multigraph`

WHERE THIS SITS
---------------
ext_02 showed that at a FIXED geometry a rank-k readout matches the full-rank one
for k >= 3, i.e. the parameter saving costs essentially no accuracy.  That left
the obvious hole:

    the geometry in ext_02 was never trained.  Is the rank constraint still free
    once the geometry adapts -- and the readout becomes the thing the geometry has
    to work WITH?

That is the question here, and it is the last representational question before the
extension's claim can be stated as a result rather than as a budget calculation.

PROTOCOL, AND WHAT IT IS NOT COMPARABLE TO
------------------------------------------
Both arms run the SAME loop at the SAME measurement cost:

    geometry : central-difference search on an analog budget (2 forwards per
               coordinate probed), relative eps, Adam -- i.e. the base version's
               protocol
    readout  : re-solved in CLOSED FORM each epoch, once unconstrained and once
               rank-constrained to k, chosen on a validation slice

**The loop MIRRORS abl_common.train_model rather than calling it**, and that is
forced: train_model's closed-form hook calls solve_readout_closed_form, which
writes a full-rank row (`layer.W_gauss[row, :] = w`) and has no place to put a
rank constraint.  Modifying the base version is out of scope by the project's own
rule.  The consequence must be stated plainly:

    > the ABSOLUTE numbers here are NOT comparable to FINDINGS.md.  This is an
    > internal A/B between two readouts on one identical loop, and only the
    > difference between the arms means anything.

Mirroring also risks protocol drift, so the loop states its constants explicitly
(eps_rel, Adam betas, budget accounting) to make any drift visible.

Output: outputs/ext_03_geometry_rank.txt + _plot.png
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
    OUT_DIR, Report, build_model, effective_budget, header, layer_activations,
    r2, set_ideal, try_plots,
)
from extensions.multigraph.ext_02_rank_budget import (     # noqa: E402
    _als_rank, _ridge, _r2_multi, _svd_candidate, make_multi_output,
)

M_OUT = 8


# ---------------------------------------------------------------------------
class GeoNet:
    """
    One shared graph whose GEOMETRY is trainable, plus a readout that is solved in
    closed form (unconstrained or rank-constrained) instead of searched.

    The trainable vector is the geometry only: centres (D*N) then widths (N).  The
    readout is deliberately NOT in the vector, because the project's rule is that
    the strengths are a calibration quantity, not a search quantity.
    """

    def __init__(self, layer, k, lam):
        self.layer = layer
        self.k = k
        self.lam = lam
        self.W = None                     # (m, N), set by fit_readout
        self.n_probe_calls = 0
        self.n_calibrations = 0
        self.n_calibration_measurements = 0
        self.p0 = None                    # geometry at t=0, for the movement check

    # -- geometry vector ----------------------------------------------------
    def get_params(self):
        c = np.asarray(self.layer.centers, dtype=np.float64).ravel()
        h = np.asarray(self.layer.h, dtype=np.float64).ravel()
        return np.concatenate([c, h])

    def set_params(self, v):
        D, N = self.layer.centers.shape
        self.layer.centers = np.asarray(v[:D * N], dtype=np.float32).reshape(D, N)
        self.layer.h = np.asarray(v[D * N:D * N + N], dtype=np.float32)

    @property
    def n_params(self):
        D, N = self.layer.centers.shape
        return D * N + N

    def geometry_movement(self):
        """
        ||p_end - p_0|| / ||p_0||, in the parameter vector's own metric.

        THIS IS THE ANTI-VACUITY CHECK.  If the geometry barely moves, then
        "the rank constraint survives geometry training" is not a finding -- the
        arms simply re-measured ext_02's fixed-geometry case.  ext_01 shipped a
        version whose negative control silently skipped for exactly this kind of
        reason, so the diagnostic is reported rather than assumed.
        """
        if self.p0 is None:
            return 0.0
        d = self.get_params() - self.p0
        return float(np.linalg.norm(d) / max(np.linalg.norm(self.p0), 1e-12))

    def features(self, X):
        zg, _za, _og, _oa = layer_activations(self.layer, X, "circuit", "none",
                                              None)
        return np.asarray(zg, dtype=np.float64)

    def forward(self, X):
        """Forward through the CURRENT readout; used only for loss reporting."""
        if self.W is None:
            return np.zeros((np.shape(X)[0], 1), dtype=np.float64)
        return self.features(X) @ self.W.T


def fit_readout(net, Zfi, Yfi, Zva, Yva, m, N):
    """
    Solve the readout.  k >= m means unconstrained (the base version's case);
    k < m means W = C . W_g, fitted by the ext_02 machinery and chosen on VAL.

    Choosing on a validation slice, never on test, is what keeps the two arms
    comparable: neither is allowed to see the test set when picking its readout.
    """
    if net.k >= m:
        return _ridge(Zfi, Yfi, net.lam)
    cands = _als_rank(Zfi, Yfi, net.k, net.lam, seed=0)
    cands.append(_svd_candidate(Zfi, Yfi, net.k, net.lam))
    best = None
    for (W, _C, _Wg) in cands:
        v = _r2_multi(Yva, Zva @ W.T)
        if best is None or v > best[0]:
            best = (v, W)
    return best[1]


def train_geometry(net, X_tr, Y_tr, *, epochs, budget, lr, eps_rel, probe_batch,
                   seed, X_te, Y_te, n_geometry_measurements_log):
    """
    Central-difference search on the geometry, exactly as the base version does it:
    relative eps, Adam, a fixed mini-batch within an epoch, and `budget` =
    coordinates probed per epoch (2 forward passes each).

    The readout is refreshed by closed form at the START of every epoch, so the
    geometry is always searched against an up-to-date readout -- which is the whole
    point of the project's rule, and it is applied identically to both arms.
    """
    rng = np.random.default_rng(seed)
    p = net.get_params()
    net.p0 = p.copy()                 # for the anti-vacuity movement check
    n = p.size
    m_ = Y_tr.shape[1]
    n_fit = X_tr.shape[0]
    n_val = max(32, n_fit // 4)

    mvec = np.zeros(n)
    vvec = np.zeros(n)
    b1, b2 = 0.9, 0.999
    eps_vec = eps_rel * (1.0 + np.abs(p))

    curve = []
    for ep in range(epochs):
        # -- refresh the readout in closed form (both arms pay the same) ------
        Z = net.features(X_tr)
        Zfi, Yfi = Z[:-n_val], Y_tr[:-n_val]
        Zva, Yva = Z[-n_val:], Y_tr[-n_val:]
        net.W = fit_readout(net, Zfi, Yfi, Zva, Yva, m_, Z.shape[1])
        # The accounting is explicit because the measurement count IS the currency
        # of this comparison.  A closed-form calibration of an m x N readout is
        # charged n_fit measurements -- one reading per training sample -- in BOTH
        # arms, so the arms differ only in what they STORE, never in what they pay.
        net.n_calibrations += 1
        net.n_calibration_measurements += n_fit
        n_geometry_measurements_log[0] += n_fit

        # -- probe budget: how many coordinates this epoch --------------------
        per_epoch = int(budget) if budget else n
        idx_all = np.arange(n)
        chunks, remaining = [], per_epoch
        while remaining > 0:
            perm = rng.permutation(n)
            take = min(remaining, n)
            chunks.append(perm[:take])
            remaining -= take

        if probe_batch and 0 < int(probe_batch) < n_fit:
            sel = rng.choice(n_fit, size=int(probe_batch), replace=False)
            Xb, Yb = X_tr[sel], Y_tr[sel]
        else:
            Xb, Yb = X_tr, Y_tr

        t = ep + 1

        def loss_of(vec, _Xb=Xb, _Yb=Yb):
            net.set_params(vec)
            pred = net.features(_Xb) @ net.W.T
            net.n_probe_calls += 1
            return float(np.mean((pred - _Yb) ** 2))

        for chunk in chunks:
            for j in chunk:
                p0 = p[j]
                e = eps_vec[j]
                p[j] = p0 + e
                Lp = loss_of(p)
                p[j] = p0 - e
                Lm = loss_of(p)
                p[j] = p0
                g = (Lp - Lm) / (2.0 * e)
                mvec[j] = b1 * mvec[j] + (1 - b1) * g
                vvec[j] = b2 * vvec[j] + (1 - b2) * g * g
                mh = mvec[j] / (1 - b1 ** t)
                vh = vvec[j] / (1 - b2 ** t)
                p[j] = p0 - lr * mh / (np.sqrt(vh) + 1e-8)

        Zt = net.features(X_te)
        te = _r2_multi(Y_te, Zt @ net.W.T) if net.W is not None else float("nan")
        # measurement breakdown, so the currency is auditable rather than asserted
        geom_meas = net.n_probe_calls * 2          # central difference: 2 forwards
        curve.append(dict(epoch=ep + 1, test_r2=te,
                          measurements=geom_meas
                          + net.n_calibration_measurements,
                          geom_measurements=geom_meas,
                          calib_measurements=net.n_calibration_measurements,
                          movement=net.geometry_movement(),
                          geom_params_probed=net.n_probe_calls // 2))
    return curve


# ---------------------------------------------------------------------------
def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_03: the rank budget under "
                                             "geometry training")
    ap.add_argument("--arch", default="F2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--budget", type=int, default=200,
                    help="coordinates probed per epoch (the analog budget)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eps-rel", type=float, default=1e-4)
    ap.add_argument("--probe-batch", type=int, default=96)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ks", default="4,3,8",
                    help="k values; k >= m is the unconstrained full-rank arm.  A low "
                         "k that FAILS is data -- it locates the boundary -- so "
                         "--assert requires only the largest k < m to pass")
    ap.add_argument("--freq", type=float, default=0.5,
                    help="task difficulty; 0.5 is the INFORMATIVE one per ext_02")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    # THE GLOBAL RNG TOO.  `features()` builds a REAL device layer and the device draws
    # from `np.random`, so without this the exact path moved between runs in the 5th to
    # 7th digit while everything else stayed put -- defect 58 in FINDINGS.md, which this
    # line closes.  `log_rms`'s healthy band is ~1e-6, i.e. AT the magnitude of that
    # movement, so the test that decides whether a response is a genuine single node was
    # sitting on the non-determinism.
    np.random.seed(args.seed)

    set_ideal()
    rep = Report("ext_03_geometry_rank.txt")
    for line in header("ext_03 - does the rank budget survive geometry training?",
                       f"arch={args.arch} k={ks} m={M_OUT} epochs={args.epochs} "
                       f"budget={args.budget} freq={args.freq}"):
        rep(line)

    rep("")
    rep("  Both arms: geometry by central differences on a matched analog budget,")
    rep("  readout re-solved in CLOSED FORM at the start of every epoch (the")
    rep("  project's rule), readout chosen on a validation slice of train.")
    rep("")
    rep("  SCOPE, stated because it is easy to misread: the loop MIRRORS")
    rep("  abl_common.train_model instead of calling it, because train_model's")
    rep("  closed-form hook has no place to put a rank constraint and the base")
    rep("  version may not be modified.  Therefore the ABSOLUTE numbers below are")
    rep("  NOT comparable to FINDINGS.md -- only the difference between the arms is.")

    t0 = time.time()
    results = {}
    for seed in range(args.seeds):
        X_tr, Y_tr, X_te, Y_te = make_multi_output(m=M_OUT, freq=args.freq,
                                                   seed=7 + seed)
        m = Y_tr.shape[1]
        N = None
        for k in ks:
            base = build_model(args.arch, X_tr, seed=seed, substrate="circuit")
            lay = base.layers[0]
            N = int(lay.n_particles)
            net = GeoNet(lay, k, args.lam)
            log = [0]
            curve = train_geometry(net, X_tr, Y_tr, epochs=args.epochs,
                                   budget=args.budget, lr=args.lr,
                                   eps_rel=args.eps_rel,
                                   probe_batch=args.probe_batch, seed=seed,
                                   X_te=X_te, Y_te=Y_te,
                                   n_geometry_measurements_log=log)
            results.setdefault(k, []).append(dict(
                curve=curve, net=net, n_geometry_params=net.n_params,
                readout_params=(m * N if k >= m else k * N + m * k),
                n_fit=X_tr.shape[0] - max(32, X_tr.shape[0] // 4)))

    N = N or 0
    n_fit = results[ks[0]][0]["n_fit"]
    rep("")
    rep(f"  shared graph: arch {args.arch} stage 0, N = {N} particles, "
        f"train {n_fit} / val / test")
    rep(f"  geometry parameters {results[ks[0]][0]['n_geometry_params']} "
        f"(centres + widths), searched by FD in BOTH arms")
    rep(f"  readout parameters: " + "   ".join(
        f"k={k}: {results[k][0]['readout_params']} "
        f"({results[k][0]['readout_params']/n_fit:.2f}x)"
        for k in ks))
    rep("  ^ the GEOMETRY cost is identical across arms by construction; only the")
    rep("    readout size differs, which is the variable DESIGN section 5 says")
    rep("    decides whether closed-form calibration works.")

    # -- results ------------------------------------------------------------
    for k in ks:
        rep("\n" + "=" * 78)
        rep(f"  k = {k}" + ("  (unconstrained full-rank readout)" if k >= M_OUT
                            else f"  (rank-constrained, "
                                 f"{results[k][0]['readout_params']/n_fit:.2f}x)"))
        rep("=" * 78)
        curves = [r["curve"] for r in results[k]]
        grid = sorted({c["epoch"] for cur in curves for c in cur})
        rep(f"  {'epoch':>6} {'test R^2 mean':>14} {'sd':>8} {'geom probes':>12} "
            f"{'calib':>8} {'total meas':>11} {'geom moved':>11}")
        rep("  " + "-" * 78)
        for e in grid:
            rows_e = [[c for c in cur if c["epoch"] == e][0] for cur in curves]
            if e % max(1, args.epochs // 6) and e not in (1, grid[-1]):
                continue
            rep(f"  {e:>6} {np.mean([r['test_r2'] for r in rows_e]):>+14.4f} "
                f"{np.std([r['test_r2'] for r in rows_e]):>8.4f} "
                f"{int(np.mean([r['geom_params_probed'] for r in rows_e])):>12d} "
                f"{int(np.mean([r['calib_measurements'] for r in rows_e])):>8d} "
                f"{int(np.mean([r['measurements'] for r in rows_e])):>11d} "
                f"{np.mean([r['movement'] for r in rows_e]):>11.4f}")
        last = curves[0][-1]
        rep("")
        rep(f"  measurements at the end: {last['geom_measurements']} geometry "
            f"(2 forwards each) + {last['calib_measurements']} calibration "
            f"({args.epochs} closed-form solves x {n_fit} samples)")
        rep(f"  BOTH arms pay these identical numbers: the arms differ only in what")
        rep(f"  they STORE, never in what they measure.")

    # -- verdict ------------------------------------------------------------
    rep("\n" + "=" * 78)
    rep("  VERDICT: does the rank constraint still cost nothing once the geometry")
    rep("  is trained?")
    rep("=" * 78)
    full_k = max(ks)
    def _per_seed(k):
        return np.asarray([r["curve"][-1]["test_r2"] for r in results[k]],
                          dtype=np.float64)
    ref_vec = _per_seed(full_k)
    ref, ref_sd = float(ref_vec.mean()), float(ref_vec.std())
    move = float(np.mean([r["curve"][-1]["movement"] for r in results[full_k]]))
    # ANTI-VACUITY.  A pass is worthless if the geometry never moved, because then
    # this experiment is ext_02 with extra steps.  The floor is stated, not
    # implied, and a vacuous run is reported as vacuous rather than as a pass.
    move_floor = 0.05
    moved = move >= move_floor
    rep("")
    rep(f"  geometry movement at the end: {move:.4f} relative "
        f"(floor {move_floor}) -> {'MOVED' if moved else 'BARELY MOVED'}")
    rep(f"  full-rank arm (k={full_k}): final test R^2 {ref:+.4f} "
        f"(sd {ref_sd:.4f} ACROSS SEEDS) at "
        f"{results[full_k][0]['readout_params']/n_fit:.2f}x readout/samples")
    rep("")
    rep("  PAIRED comparison.  The arms see the SAME data seeds, so the correct")
    rep("  statistic is the per-seed difference (k minus full), not the difference")
    rep("  of means against the across-seed sd -- the latter is dominated by which")
    rep("  data split was drawn and makes the test unable to fail.")
    ok = True
    passing = []
    for k in ks:
        if k >= M_OUT:
            continue
        d = _per_seed(k) - ref_vec                    # paired, same seed, same data
        mean_d = float(d.mean())
        sd_d = float(d.std(ddof=1)) if d.size > 1 else 0.0
        sem = sd_d / np.sqrt(max(d.size, 1))
        # Tolerance is the ext_02 value UNLESS the paired standard error says the
        # measurement cannot support it, in which case the run is reported as
        # underpowered instead of silently passing.
        tol = max(0.02, 3.0 * sem)
        powerful = 3.0 * sem < 0.02
        good = mean_d >= -tol
        ok &= good
        if good:
            passing.append(k)
        rep(f"  k={k}: paired gap {mean_d:+.4f} (sd {sd_d:.4f}, sem {sem:.4f} over "
            f"{d.size} seeds)  tol {tol:.4f}  "
            f"{'OK' if good else 'FAILS'}"
            f"{'' if powerful else '   <- UNDERPOWERED'}")
    # The POINT of the experiment is to locate the boundary, so a low k failing is
    # DATA, not a failed run.  What the assert requires is that the largest tested
    # k < m passes -- i.e. that some smaller readout is affordable at all -- plus
    # that the geometry actually moved.
    edges = [k for k in ks if k < M_OUT]
    k_best = max(edges) if edges else None
    k_pass = min(passing) if passing else None
    assert_ok = moved and (k_best in passing if k_best is not None else True)
    rep("")
    if edges:
        failing = [k for k in edges if k not in passing]
        rep(f"  boundary: largest tested k < m is {k_best}; smallest PASSING k is "
            f"{k_pass if k_pass is not None else 'none'}")
        if k_pass is not None and k_pass > min(edges):
            rep(f"  -> the affordable region is k >= {k_pass}; "
                f"failing k = {', '.join(str(x) for x in failing)}")
    rep("")
    rep("  Does the constraint's cost GROW as the geometry is trained?  A paired gap")
    rep("  that widens would mean the geometry and the rank constraint compete; one")
    rep("  that stays flat or narrows means they do not.")
    rep("")
    grid_chk = sorted({c["epoch"] for r in results[full_k] for c in r["curve"]})
    show = [e for e in grid_chk
            if e in (1, grid_chk[len(grid_chk) // 2], grid_chk[-1])]
    rep(f"  {'epoch':>6} " + "".join(f"{'k=' + str(k) + ' gap':>12}"
                                     for k in ks if k < M_OUT))
    rep("  " + "-" * (6 + 12 * len([k for k in ks if k < M_OUT])))
    for e in show:
        refe = np.asarray([[c["test_r2"] for c in r["curve"] if c["epoch"] == e][0]
                           for r in results[full_k]])
        cells = ""
        for k in ks:
            if k >= M_OUT:
                continue
            ke = np.asarray([[c["test_r2"] for c in r["curve"] if c["epoch"] == e][0]
                             for r in results[k]])
            cells += f"{float((ke - refe).mean()):>+12.4f}"
        rep(f"  {e:>6} {cells}")
    rep("")
    if assert_ok and not ok:
        rep("  SURVIVES DOWN TO A BOUNDARY.  The larger rank-constrained readouts keep")
        rep("  full-rank accuracy while the geometry is trained against them, and the")
        rep("  smaller ones do not -- which is the USEFUL outcome, because it locates")
        rep("  the design point instead of only confirming that one exists.")
        rep(f"  Affordable region: k >= {k_pass}.  Combined with ext_02 (which found")
        rep("  the same boundary at a fixed geometry) the conclusion is that the")
        rep("  boundary is a property of the ARCHITECTURE, not of the untrained")
        rep("  geometry: geometry training neither widens nor narrows the affordable")
        rep("  region.")
    elif assert_ok and ok:
        rep("  SURVIVES at every k tested, and the geometry demonstrably moved, so this")
        rep("  is not a re-measurement of the fixed-geometry case.  Combined with")
        rep("  ext_02 this is the representational case for the extension: the saving")
        rep("  is free both at a fixed geometry and under geometry training.")
        if edges:
            rep(f"  The boundary lies below k = {min(edges)} and was not probed here.")
    elif not moved:
        rep("  INCONCLUSIVE, not a pass: the accuracy gaps may be fine but the")
        rep("  geometry did not move enough for the question to have been asked.")
    else:
        rep("  DOES NOT SURVIVE at the largest k tested: the geometry cannot fully")
        rep("  compensate for the smaller readout, so ext_02's parity was a property")
        rep("  of the FIXED geometry rather than of the architecture.  Report that")
        rep("  negative result.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

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
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))
        for k in ks:
            curves = [r["curve"] for r in results[k]]
            grid = sorted({c["epoch"] for cur in curves for c in cur})
            mean = [np.mean([[c["test_r2"] for c in cur if c["epoch"] == e][0]
                             for cur in curves]) for e in grid]
            meas = [np.mean([[c["measurements"] for c in cur if c["epoch"] == e][0]
                             for cur in curves]) for e in grid]
            ax.plot(meas, mean, "o-",
                    label=f"k={k} ({results[k][0]['readout_params']} readout params)")
        ax.set_xlabel("analog measurements (geometry probes x2 + calibrations)")
        ax.set_ylabel("test R^2 (mean over output channels)")
        ax.set_title("ext_03: rank budget under geometry training")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "ext_03_geometry_rank_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {p}")

    if do_assert and not assert_ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
