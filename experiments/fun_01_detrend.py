"""
fun_01_detrend.py -- subtract a local mean before the nodes.  One instance of the pipeline.

WHAT THIS DOES
--------------
    stage 1   m(x) = sum_j w_j phi_j(u)(x)      fitted so that m approximates G_sigma * u
              r(x) = u(x) - lam * m(x)          the residual, a field again
    stage 2   phi_k(r) -> a result             via `fun_common.use_field`
    stage 3   ridge to +-1 one-vs-rest, lambda chosen on a per-class validation split

`fixed` replaces stage 1's fitted `m` with the blur itself (`m = G_sigma * u`, no fitted
weights).  That comparison is the point of the file: it separates "a fitted first stage"
from "subtracting a local mean".

WHY A BLURRED TARGET
--------------------
Fitting stage 1 to reconstruct `u` is kernel interpolation: with 100 nodes whose centres
are patches of the same images it reached R^2 = 0.999 and left a residual with 0.6% of the
variance.  A detrend wants exactly the opposite -- explain only the smooth part -- so the
target is `G_sigma * u` and `sigma` is the knob that defines what counts as trend.

WHAT IS REPORTED
----------------
  acc0          clean accuracy at the chosen lambda
  slope         accuracy change per 0.1 of additive offset (positive = flatter)
  trend_frac    var(m)/var(u): how much of the image the trend carries
  std(r)        residual amplitude in pixel units, compared against one grey level

    python experiments/fun_01_detrend.py --seeds 1 --n-test 100 --verbose
    python experiments/fun_01_detrend.py --seeds 5 --n-test 200 --sigma 2

RECORDED RESULT (5 seeds, 200 test images per class, sigma=2, 100 nodes per bank)
    raw 46.54 | fixed 50.94 | learned 51.34
    slope goes from -2.55 to about 0 when the subtraction is applied
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "extensions", "multigraph"), ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fun_common import (BW, QUANT, N_CLASSES, SHIPPED_BASELINE, NodeBank,   # noqa: E402
                        blur, centre_field, choose_centres_bandwidths,
                        clean_features, draw_split, fit_readout, fit_trend,
                        gamma_map, illum, load_gray, paired, patchify,
                        reduce_mean, split_fit_val, state, trend_field)

OFFSETS = (-0.08, -0.04, 0.0, 0.04, 0.08)
GAMMAS = (0.80, 0.90, 1.00, 1.10, 1.30)


def rows_of(field):
    P = patchify(field[..., None], 3)
    return P.reshape(-1, P.shape[-1])


def slope_from(accs, dev):
    """Least-squares slope of accuracy against |deviation from identity|."""
    x = np.abs(np.asarray(dev, dtype=np.float64))
    y = np.asarray(accs, dtype=np.float64)
    den = float(((x - x.mean()) ** 2).sum())
    if den <= 0:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).sum() / den)


def endpoint(imgs, lam, sigma, bank_a, w, mode):
    """The field the second bank is applied to, at this lambda."""
    c = centre_field(imgs)
    if mode == "raw":
        return c
    m = (centre_field(blur(imgs, sigma)) if mode == "fixed"
         else trend_field(bank_a, w, imgs))
    return c - lam * m


def run_cell(mode, lam, sigma, n1, n2, seed, gray, labels, per_class, n_test,
             verbose=False):
    tr, te = draw_split(gray, labels, seed, n_train=per_class, n_test=n_test)
    fit_idx, va_idx = split_fit_val(tr, labels, 0.8, seed=seed)
    Xf, Xv, Xt = gray[fit_idx], gray[va_idx], gray[te]
    yf, yv, yt = labels[fit_idx], labels[va_idx], labels[te]

    # bank A: nodes on the raw field.  Only the `learned` arm fits weights on it.
    rng_a = np.random.default_rng(7000 + 11 * seed)
    C_a, h_a = choose_centres_bandwidths(rows_of(Xf), n1, rng_a)
    bank_a = NodeBank(C_a, h_a)
    if mode == "learned":
        w, r2, eps = fit_trend(bank_a, Xf, Xv, sigma)
    else:
        w, r2, eps = None, float("nan"), 0.0

    Rf = endpoint(Xf, lam, sigma, bank_a, w, mode)
    Rv = endpoint(Xv, lam, sigma, bank_a, w, mode)
    Rt = endpoint(Xt, lam, sigma, bank_a, w, mode)

    std_r = float(np.std(Rt))
    if std_r < 1e-6:
        # `fixed` with sigma=0 makes m exactly u, so r is identically zero and every
        # patch is zero.  That cell is degenerate by construction; it gets no score.
        if verbose:
            print("      %-7s lam %.2f  DEGENERATE (residual identically zero)" % (mode, lam))
        return dict(acc0=None, slope=None, trend=float("nan"), std_r=std_r, cols=0,
                    r2=r2)

    # bank B: nodes on the residual field, drawn from the residual's own distribution.
    rng_b = np.random.default_rng(9100 + 11 * seed)
    C_b, h_b = choose_centres_bandwidths(rows_of(Rf), n2, rng_b)
    bank_b = NodeBank(C_b, h_b)

    def feats(imgs):
        return reduce_mean(bank_b.field(endpoint(imgs, lam, sigma, bank_a, w, mode)))

    F = {"tr": feats(Xf), "va": feats(Xv), "te": feats(Xt)}
    Z, kept, total, idx = clean_features(F)
    sd = np.maximum(F["tr"].std(axis=0)[idx], 1e-12)
    W = fit_readout(Z["tr"], yf, Z["va"], yv)
    acc0 = float((np.argmax((F["te"][:, idx] / sd) @ W, axis=1) == yt).mean())

    def acc_of(images):
        Fq = feats(images)
        return float((np.argmax((Fq[:, idx] / sd) @ W, axis=1) == yt).mean())

    trend_frac = float(np.nan)
    if mode != "raw":
        m = (centre_field(blur(Xt, sigma)) if mode == "fixed"
             else trend_field(bank_a, w, Xt))
        trend_frac = float(np.var(m) / (np.var(centre_field(Xt)) + 1e-12))

    off = [acc_of(illum(Xt, 1.0, b)[0]) for b in OFFSETS]
    gam = [acc_of(gamma_map(Xt, g)) for g in GAMMAS]
    slope = slope_from(off, np.abs(np.asarray(OFFSETS)))
    slope_g = slope_from(gam, np.abs(np.asarray(GAMMAS) - 1.0))

    flag = "  <-- AT THE QUANTISATION FLOOR" if std_r < QUANT else ""
    if verbose:
        print("      %-7s lam %.2f  acc0 %.4f  trend %.3f  std(r) %.4f  "
              "slope %+6.2f  slope_gam %+6.2f  cols %3d%s"
              % (mode, lam, acc0, trend_frac, std_r, 100.0 * slope * 0.1,
                 100.0 * slope_g * 0.1, total, flag))
    return dict(acc0=acc0, slope=slope, slope_g=slope_g, trend=trend_frac,
                std_r=std_r, cols=int(total), r2=r2, kept=int(kept))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-test", type=int, default=100,
                    help="test images per class; the repo default is 30, which gives "
                         "150 images in total and about +-4 points of binomial noise")
    ap.add_argument("--per-class", type=int, default=100,
                    help="training images per class; 300 reproduces the data-scale cell")
    ap.add_argument("--n1", type=int, default=100, help="nodes in bank A (stage 1)")
    ap.add_argument("--n2", type=int, default=100, help="nodes in bank B (stage 2)")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="blur scale in pixels; 0 means stage 1 interpolates, not trends")
    ap.add_argument("--lambdas", type=str, default="0,1")
    ap.add_argument("--arms", type=str, default="raw,fixed,learned")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    modes = [m.strip() for m in args.arms.split(",") if m.strip()]
    lams = [float(x) for x in args.lambdas.split(",") if x.strip()]
    gray, labels = load_gray()
    print("=" * 78)
    print("  fun_01_detrend -- subtract a local mean before the nodes")
    print("=" * 78)
    print("  stage 1  m = sum_j w_j phi_j(u) fitted to G_sigma*u ;  r = u - lam*m")
    print("  stage 2  %d nodes on r, positional mean" % args.n2)
    print("  stage 3  ridge, lambda per class on a validation split")
    print("  sigma %.1f px, %d train images per class, %d test images per class"
          % (args.sigma, args.per_class, args.n_test))
    print("  reference: cifar_02 published 36.0%% (single seeded draw, no interval)"
          % ())

    res = {}
    for s in range(max(1, int(args.seeds))):
        for mode in modes:
            for lam in lams:
                out = run_cell(mode, lam, args.sigma, args.n1, args.n2, s, gray,
                               labels, args.per_class, args.n_test, args.verbose)
                res.setdefault((mode, lam), {})[s] = out
        if args.seeds <= 1:
            print()

    def mean(key, field):
        v = [r[field] for r in res[key].values() if r.get(field) is not None]
        return float(np.mean(v)) if v else float("nan")

    print("\n  %-8s %-6s %-9s %-9s %-17s %s"
          % ("arm", "lam", "acc0", "trend", "slope/0.1 (off, gam)", "cols"))
    for mode in modes:
        for lam in lams:
            a = 100.0 * mean((mode, lam), "acc0")
            if np.isnan(a):
                print("  %-8s %-6.2f %-9s" % (mode, lam, "degenerate"))
                continue
            print("  %-8s %-6.2f %8.2f%% %9.3f     %+6.2f  %+6.2f     %d"
                  % (mode, lam, a, mean((mode, lam), "trend"),
                     100.0 * 0.1 * mean((mode, lam), "slope"),
                     100.0 * 0.1 * mean((mode, lam), "slope_g"),
                     int(mean((mode, lam), "cols"))))

    print("\n  PAIRED, lam=%.2f against lam=%.2f within each arm:"
          % (lams[-1], lams[0]))
    for mode in modes:
        seeds = sorted(set(res[(mode, lams[-1])]) & set(res[(mode, lams[0])]))
        a = [res[(mode, lams[-1])][s]["acc0"] for s in seeds]
        b = [res[(mode, lams[0])][s]["acc0"] for s in seeds]
        if any(v is None for v in a + b):
            print("    %-8s contains a degenerate cell, not compared" % mode)
            continue
        m, half = paired(a, b)
        print("    %-8s %+7.2f points  %-24s %s"
              % (mode, m,
                 "one seed" if not np.isfinite(half) else
                 "CI[%+7.2f,%+7.2f]" % (m - half, m + half), state(m, half)))

    print("\n  NOTES")
    print("    `fixed` needs no fitted weights in stage 1; it is the comparison that")
    print("    separates 'a fitted first stage' from 'subtracting a local mean'.")
    print("    A residual below one grey level (%.4f) is quantisation noise: the" % QUANT)
    print("    acc0 at that cell is not measuring a trend.")
    print("    sigma = 0 with the `fixed` arm is degenerate by construction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
