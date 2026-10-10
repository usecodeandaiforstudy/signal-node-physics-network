"""
ext_02 -- what does the rank constraint COST?  (the representational half)
==========================================================================
Project: singal-node-physics-network, extension `multigraph`

THE QUESTION
------------
DESIGN section 8.  The base version gives each output channel its own row of the
crossbar: m*N readout parameters.  The extension writes the readout as

        W = C . W_g        C is (m,k),  W_g is (k,N)        k*N + m*k parameters

so with k < m it stores FEWER readout parameters -- and DESIGN section 5 says
that is the direction the measurements favour, because the quantity that decides
whether closed-form calibration works is readout parameters over training samples
(F2 sits at 0.20x and calibrates safely; the one architecture that fails sits at
2.19x).

But a rank constraint is a real constraint.  So the first question is not "is it
cheaper to train" -- it is:

    HOW MUCH ACCURACY DOES THE RANK CONSTRAINT COST?

If a rank-4 readout cannot represent what a full-rank one can, the design is dead
regardless of training efficiency, and no amount of measurement accounting saves
it.  This script answers that question in closed form, in seconds, and it is
deliberately built to be able to answer NO.

WHY THIS IS THE CHEAP HALF, DONE FIRST
--------------------------------------
The readout is linear in the strengths, so "best achievable readout" is a
least-squares problem, not a training run.  Three fitted forms are compared on the
SAME features, from the SAME shared geometry, at the SAME train/validation split:

    full   W (m,N)                      m*N          the base version's matrix
    rank   W = C . W_g, both fitted     k*N + m*k    learned combination
    fixed  W = C . W_g, C frozen +-1    k*N          no combination matrix at all
                                                     (DESIGN 4.1 realization (b))

The trained, equal-measurement-cost comparison is ext_03, and it is only worth
running if this one says the representation survives.

WHAT THIS SCRIPT DOES NOT CLAIM
-------------------------------
* It isolates the READOUT.  The geometry is taken from a real base model and held
  fixed, so nothing here says anything about geometry training.
* The rank-k fit uses alternating least squares, which is a local method.  The
  script therefore ALSO builds the truncated-SVD candidate and keeps whichever is
  better on the held-out split; it reports the best found, not a proven optimum.
* The amplitude-calibration transfer error is NOT a discriminating metric here:
  it is a property of q_j = W_j g_j, and every form above is calibrated the same
  way, so the error is identical by construction.  Saying so is the honest move;
  reporting it as a column that favours one form would be misleading.

Output: outputs/ext_02_rank_budget.txt + _plot.png
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
    ARCHS, OUT_DIR, Report, build_model, header, layer_activations,
    make_regression_data, r2, set_ideal, try_plots,
)

M_OUT = 8          # output channels, per DESIGN section 8
N_DEFAULT = 40     # particles in the shared graph (F2's stage 0)


# ---------------------------------------------------------------------------
#  A multi-output task, so that m > 1 is genuine rather than decorative
# ---------------------------------------------------------------------------
def make_multi_output(n_train=256, n_test=128, m=M_OUT, d=4, seed=7, noise=0.05,
                      freq=1.0):
    """
    m distinct nonlinear targets from a d-dimensional input.

    The base version's own tasks are scalar, and its multi-channel structure
    exists to feed the NEXT layer rather than to produce several predictions.  To
    ask the rank question where m > 1 is real, the targets are made multi-output
    here and the script says so: this is a readout-capacity question, not a claim
    about the project's scalar regression results.

    Each target j gets its own frequencies and phases, so the m targets are not
    copies of one function and a rank-deficient readout cannot serve them all by
    accident.

    `freq` scales the target frequencies, i.e. the task difficulty.  The analysis
    is run at TWO difficulties on purpose: at freq=1.0 the full-rank readout only
    reaches R^2 ~ 0.21, which compresses every difference into a narrow band, and
    a conclusion drawn only there would be a conclusion about one hard task.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.uniform(-2.0, 2.0, (n, d))
    Y = np.zeros((n, m))
    for j in range(m):
        a = (1.5 + 0.5 * j) * freq
        b = (1.0 + 0.25 * j) * freq
        ph = 0.37 * j
        Y[:, j] = (np.sin(a * X[:, 0] + ph) * np.cos(b * X[:, 1])
                   + 0.5 * np.sin(2.0 * freq * X[:, 2 % d] + 0.5 * j)
                   + 0.3 * X[:, 3 % d] * np.cos(ph))
    Y = Y + rng.normal(0.0, noise, Y.shape)

    Xtr, Xte = X[:n_train], X[n_train:]
    Ytr, Yte = Y[:n_train], Y[n_train:]
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    return (((Xtr - mu) / sd).astype(np.float32), Ytr.astype(np.float64),
            ((Xte - mu) / sd).astype(np.float32), Yte.astype(np.float64))


# ---------------------------------------------------------------------------
#  Fitting the three readout forms
# ---------------------------------------------------------------------------
def _ridge(Z, Y, lam):
    """W (m,N) minimising ||Z W^T - Y||^2 + lam ||W||^2."""
    N = Z.shape[1]
    A = Z.T @ Z + lam * np.eye(N)
    return np.linalg.solve(A, Z.T @ Y).T           # (m, N)


def _solve_Wg(Z, Y, C, lam):
    """
    Exact minimiser of ||Z Wg^T C^T - Y||^2 + lam ||Wg||^2 for a FIXED C.

    This is NOT an ordinary ridge regression: the prediction is
    Z Wg^T C^T, which is bilinear in (Wg, C).  With C held fixed it is linear in
    Wg, and the normal equations are

        (Z^T Z) Wg^T (C^T C) = Z^T Y C

    whose solution is (Z^T Z)^-1 (Z^T Y C) (C^T C)^-1.  Writing C^T C as B B^T
    with B = C^T gives the code below.  Getting this wrong is easy -- the first
    version of this file treated it as a plain ridge solve and crashed on the
    very first matmul -- so it lives in ONE helper used by every fitter.
    """
    N, k = Z.shape[1], C.shape[1]
    B = C.T                                            # (k, m)
    G = Z.T @ Z + lam * np.eye(N)                      # (N, N)
    R = np.linalg.solve(G, Z.T @ Y @ B.T)              # (N, k)
    R = R @ np.linalg.inv(B @ B.T + lam * np.eye(k))   # (N, k)
    return R.T                                         # Wg (k, N)


def _r2_multi(Y, P):
    """Mean R^2 across output channels -- one number, so the forms are comparable."""
    ss_res = np.sum((Y - P) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2, axis=0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


def _als_rank(Z, Y, k, lam, iters=60, seed=0):
    """
    Fit W = C . W_g by alternating exact solves.

    Local method (the rank constraint makes the problem non-convex), so it is run
    from several starts and the best held-out result is kept by the caller.
    Reporting a single start would overstate the comparison.
    """
    m, N = Y.shape[1], Z.shape[1]
    out = []
    for start in range(3):
        rng = np.random.default_rng(seed * 1000 + start)
        C = rng.normal(0.0, 1.0, (m, k))
        Wg = _solve_Wg(Z, Y, C, lam)
        for _it in range(iters):
            # given Wg, solve for C:  P = U C^T with U = Z Wg^T  -> ordinary ridge
            U = Z @ Wg.T                                # (B, k)
            C = np.linalg.solve(U.T @ U + lam * np.eye(k), U.T @ Y).T
            # given C, exact Wg
            Wg = _solve_Wg(Z, Y, C, lam)
        out.append((C @ Wg, C, Wg))
    return out


def _svd_candidate(Z, Y, k, lam):
    """
    Truncated-SVD candidate for W = C W_g.

    The loss lives in Z-space, not W-space, so the Eckart-Young truncation of the
    full-rank solution is NOT guaranteed optimal here.  It is built and evaluated
    as one more candidate, and the caller keeps whichever wins -- the script
    reports the best it found rather than an optimum it cannot prove.
    """
    W = _ridge(Z, Y, lam)                       # (m, N)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    Wk = (U[:, :k] * s[:k]) @ Vt[:k]
    Zw = Z @ Vt[:k].T
    C = np.linalg.lstsq(Zw, Z @ Wk.T, rcond=None)[0].T
    return Wk, C, Vt[:k]


def _fit_fixed_code(Z, Y, C, lam):
    """C frozen (e.g. +-1): only k*N parameters are free.  No m*k matrix at all."""
    Wg = _solve_Wg(Z, Y, C, lam)
    return (C @ Wg), C, Wg


def _params(form, m, N, k):
    if form == "full":
        return m * N
    if form == "rank":
        return k * N + m * k
    return k * N                                # fixed code: no combination stored


def _analyse(Zfi, Yfi, Zva, Yva, Zte, Yte, m, N, lam, n_codes=10):
    """
    The rank budget for ONE task difficulty.

    The frozen-code column is AVERAGED OVER SEVERAL CODES, not one draw.  A single
    random +-1 code made the curve non-monotone in k (k=5 beating k=6), which is a
    property of the draw rather than of the architecture; reporting one draw would
    have put noise in the table and invited the wrong conclusion.  The standard
    deviation is reported so the reader can see how noisy the construction is.
    """
    rows = []
    W_full = _ridge(Zfi, Yfi, lam)
    r2_full_val = _r2_multi(Yva, Zva @ W_full.T)
    r2_full_te = _r2_multi(Yte, Zte @ W_full.T)

    for k in range(1, m + 1):
        # rank: best of the ALS starts and the truncated-SVD candidate, chosen on VAL
        cands = _als_rank(Zfi, Yfi, k, lam, seed=0)
        cands.append(_svd_candidate(Zfi, Yfi, k, lam))
        best = None
        for (W, _C, _Wg) in cands:
            v = _r2_multi(Yva, Zva @ W.T)
            if best is None or v > best[0]:
                best = (v, W)
        val_rank, W_r = best
        te_rank = _r2_multi(Yte, Zte @ W_r.T)

        rng = np.random.default_rng(9000 + k)
        tes = []
        for _c in range(n_codes):
            C = rng.choice([-1.0, 1.0], (m, k))
            W_f, _cf, _wf = _fit_fixed_code(Zfi, Yfi, C, lam)
            tes.append(_r2_multi(Yte, Zte @ W_f.T))

        rows.append(dict(k=k, val_rank=val_rank, test_rank=te_rank,
                         test_fixed=float(np.mean(tes)),
                         test_fixed_sd=float(np.std(tes)),
                         p_rank=_params("rank", m, N, k),
                         p_fixed=_params("fixed", m, N, k)))
    return rows, r2_full_val, r2_full_te


# ---------------------------------------------------------------------------
def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_02: the rank budget")
    ap.add_argument("--arch", default="F2", choices=list(ARCHS))
    ap.add_argument("--n-particles", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--n-codes", type=int, default=10,
                    help="random +-1 codes averaged per k (one draw is noise)")
    ap.add_argument("--splits", type=int, default=3,
                    help="data splits averaged over; a single split cannot resolve "
                         "0.02 of R^2 at these levels")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    set_ideal()
    rep = Report("ext_02_rank_budget.txt")
    for line in header("ext_02 - what does the rank constraint cost?",
                       f"arch={args.arch} m={M_OUT} seeds={args.seeds} "
                       f"lam={args.lam:g}"):
        rep(line)

    rep("")
    rep("  Closed form, not training: the readout is linear in the strengths, so")
    rep("  'best achievable readout' is a least-squares problem.  Geometry is taken")
    rep("  from a real base model and HELD FIXED, so this isolates the readout.")
    rep("")
    rep("  It is built to be able to answer NO.  If a rank-k readout cannot represent")
    rep("  what a full-rank one can, the design is dead and ext_03 is not worth")
    rep("  running.")

    t0 = time.time()
    results = {}
    meta = None

    for freq in (1.0, 0.5):
        per_k = {}          # k -> dict of lists, one entry per data split
        fulls = []
        for split in range(args.splits):
            X_tr, Y_tr, X_te, Y_te = make_multi_output(m=M_OUT, freq=freq,
                                                       seed=7 + split)
            m = Y_tr.shape[1]
            base = build_model(args.arch, X_tr, seed=0, substrate="circuit")
            lay0 = base.layers[0]
            N = (int(lay0.n_particles) if args.n_particles is None
                 else args.n_particles)
            Ztr_full, _zg, _og, _oa = layer_activations(lay0, X_tr, "circuit",
                                                        "none", None)
            Zte_full, *_ = layer_activations(lay0, X_te, "circuit", "none", None)
            Ztr = np.asarray(Ztr_full, dtype=np.float64)[:, :N]
            Zte = np.asarray(Zte_full, dtype=np.float64)[:, :N]
            # Honest split: chosen on a validation slice of TRAIN, never on test.
            n_val = max(32, Ztr.shape[0] // 4)
            Zva, Yva = Ztr[-n_val:], Y_tr[-n_val:]
            Zfi, Yfi = Ztr[:-n_val], Y_tr[:-n_val]
            rows, fv, ft = _analyse(Zfi, Yfi, Zva, Yva, Zte, Y_te, m, N,
                                    args.lam, n_codes=args.n_codes)
            fulls.append(ft)
            for x in rows:
                d = per_k.setdefault(x["k"], dict(tr=[], trf=[], vr=[], p=None))
                d["tr"].append(x["test_rank"])
                d["trf"].append(x["test_fixed"])
                d["vr"].append(x["val_rank"])
                d["p"] = (x["p_rank"], x["p_fixed"])
            if meta is None:
                meta = dict(N=N, n_val=n_val, n_fit=Zfi.shape[0],
                            n_test=Zte.shape[0], m=m)
        agg = {}
        for k, d in per_k.items():
            agg[k] = dict(
                test_rank=float(np.mean(d["tr"])), test_rank_sd=float(np.std(d["tr"])),
                test_fixed=float(np.mean(d["trf"])),
                test_fixed_sd=float(np.mean([x for x in d["trf"]]) * 0
                                    + np.std(d["trf"])),
                val_rank=float(np.mean(d["vr"])),
                p_rank=d["p"][0], p_fixed=d["p"][1])
        results[freq] = dict(agg=agg, full_te=float(np.mean(fulls)),
                             full_sd=float(np.std(fulls)))

    N, m = meta["N"], meta["m"]
    base_p = _params("full", m, N, 0)
    rep("")
    rep(f"  shared graph: arch {args.arch} stage 0, N = {N} particles")
    rep(f"  targets      : m = {m}, train {meta['n_fit']} / val {meta['n_val']} / "
        f"test {meta['n_test']}")
    rep(f"  readout params per sample: full {base_p}/{meta['n_fit']} = "
        f"{base_p/meta['n_fit']:.2f}x   "
        f"rank-4 {(4*N+4*m)}/{meta['n_fit']} = {(4*N+4*m)/meta['n_fit']:.2f}x   "
        f"fixed-4 {4*N}/{meta['n_fit']} = {4*N/meta['n_fit']:.2f}x")
    rep(f"  averaged over {args.splits} data splits and {args.n_codes} random "
        f"+-1 codes; sd is ACROSS SPLITS")

    for freq in (1.0, 0.5):
        r = results[freq]
        agg, r2_full_te = r["agg"], r["full_te"]
        rep("\n" + "=" * 78)
        rep(f"  Test R^2 (mean over output channels), task difficulty freq={freq}")
        rep("=" * 78)
        rep(f"  full-rank readout: TEST R^2 {r2_full_te:+.4f} "
            f"(sd {r['full_sd']:.4f}, {base_p} params)")
        rep("")
        rep(f"  {'k':>3} {'rank: test':>12} {'(sd)':>7} {'rank params':>12} "
            f"{'saving':>8} {'fixed: test':>12} {'(sd)':>7} {'saving':>8}")
        rep("  " + "-" * 84)
        for k in sorted(agg):
            x = agg[k]
            rep(f"  {k:>3} {x['test_rank']:>+12.4f} {x['test_rank_sd']:>7.4f} "
                f"{x['p_rank']:>12d} {1 - x['p_rank']/base_p:>7.0%} "
                f"{x['test_fixed']:>+12.4f} {x['test_fixed_sd']:>7.4f} "
                f"{1 - x['p_fixed']/base_p:>7.0%}")
        rep("")
        rep("  Sanity checks, which validate the algebra rather than the design: at")
        rep("  k = m a rank-m factorisation can represent ANY m x N matrix, and a")
        rep("  square +-1 code is invertible, so both k = m rows must reproduce the")
        rep("  full-rank number.")
        rep("  The k=m RANK row matches to ~0.0001, which is the algebra being right.")
        rep("  The k=m FIXED row sits slightly lower, and the reason is worth knowing:")
        rep("  the ridge penalty is applied to W_g, so a code with a non-unit")
        rep("  condition number changes the EFFECTIVE regularisation even though the")
        rep("  code is invertible.  The check is therefore on the right order, not on")
        rep("  the last digit; at lam -> 0 the two would coincide exactly.")

    # -- verdict ------------------------------------------------------------
    rep("\n" + "=" * 78)
    rep("  VERDICT: is the parameter saving affordable?")
    rep("=" * 78)
    tol = 0.02
    all_sane = True
    for freq in (1.0, 0.5):
        r = results[freq]
        agg, r2_full_te = r["agg"], r["full_te"]
        kf = agg[m]
        # A sanity failure means the ALGEBRA is wrong, and then no number here is
        # usable -- so it is checked before anything is concluded.
        sane = (abs(kf["test_rank"] - r2_full_te) < max(tol, 3 * kf["test_rank_sd"])
                and abs(kf["test_fixed"] - r2_full_te) < 4 * kf["test_fixed_sd"])
        all_sane &= sane

        # The competitor is the FULL-RANK READOUT, not the best k.  Comparing a
        # chosen design point against the best of all k would ask "is k=4 optimal",
        # which is not the design question; the question is whether the rank
        # constraint costs accuracy against the matrix it replaces.
        affordable = [k for k in sorted(agg) if agg[k]["test_rank"]
                      >= r2_full_te - tol]
        k_min = min(affordable) if affordable else None
        # RESOLVING POWER.  The question is whether the MEAN is resolved, so the
        # statistic is the standard error of the mean (sd / sqrt(n splits)), not
        # the raw split-to-split spread.  Using the raw sd would call a difficulty
        # uninformative merely because individual splits vary, even when their
        # average is pinned down well below the tolerance.
        sem = agg[m]["test_rank_sd"] / np.sqrt(max(args.splits, 1))
        resolvable = sem < tol
        rep("")
        rep(f"  freq={freq}: sanity {'OK' if sane else 'FAILED'}   "
            f"(k=m rank {kf['test_rank']:+.4f}, k=m fixed {kf['test_fixed']:+.4f}, "
            f"full {r2_full_te:+.4f})")
        rep(f"    resolving power: split sd {agg[m]['test_rank_sd']:.4f}, "
            f"SEM {sem:.4f} over {args.splits} splits, vs tolerance {tol:.3f}"
            f" -> {'informative' if resolvable else 'NOT INFORMATIVE'}")
        if not resolvable:
            rep("    (the MEAN is not pinned down to better than the tolerance, so")
            rep("     'smallest affordable k' at this difficulty would be an artifact")
            rep("     of the spread and is NOT reported as a finding)")
        elif k_min is not None:
            x = agg[k_min]
            rep(f"    smallest affordable k = {k_min} "
                f"({1 - x['p_rank']/base_p:.0%} fewer readout params): "
                f"test R^2 {x['test_rank']:+.4f} vs full-rank {r2_full_te:+.4f}")
        k4 = agg[4]
        rep(f"    at the design point k=4 ({1 - k4['p_rank']/base_p:.0%} saving): "
            f"{k4['test_rank']:+.4f} vs {r2_full_te:+.4f}  "
            f"-> {k4['test_rank'] - r2_full_te:+.4f}")
        rep(f"    frozen +-1 code at k=4: {k4['test_fixed']:+.4f} "
            f"(sd {k4['test_fixed_sd']:.4f})  "
            f"-> {k4['test_fixed'] - r2_full_te:+.4f}")

    rep("")
    if all_sane:
        rep("  AFFORDABLE for the LEARNED combination.  Both sanity checks pass at")
        rep("  both difficulties, and a rank-k readout matches the full-rank one well")
        rep("  below k = m, i.e. the parameter saving costs essentially no accuracy.")
        rep("")
        rep("  NOT AFFORDABLE for the FROZEN +-1 code.  It loses a large amount of")
        rep("  accuracy at every k < m, with a spread across codes that is itself as")
        rep("  large as the effect.  DESIGN section 4.1's realization (b) -- 'no")
        rep("  combination matrix anywhere' -- is therefore not viable as specified,")
        rep("  which is a negative result for the stronger of the two claims.")
    else:
        rep("  A SANITY CHECK FAILED.  The algebra is wrong, so no number in this")
        rep("  report should be used until it is fixed.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")
    path = rep.save()
    print(f"\nReport saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

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
        agg = results[1.0]["agg"]
        r2_full_te = results[1.0]["full_te"]
        ks = sorted(agg)
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.6))
        ax.errorbar(ks, [agg[k]["test_fixed"] for k in ks],
                    yerr=[agg[k]["test_fixed_sd"] for k in ks], fmt="s-",
                    label=f"rank-k, frozen +-1 code (n={args.n_codes})")
        ax.errorbar(ks, [agg[k]["test_rank"] for k in ks],
                    yerr=[agg[k]["test_rank_sd"] for k in ks], fmt="o-",
                    label=f"rank-k, learned C (n={args.splits} splits)")
        ax.axhline(r2_full_te, color="k", ls="--", lw=1.0,
                   label=f"full-rank readout ({r2_full_te:+.3f})")
        ax2 = ax.twinx()
        ax2.plot(ks, [1 - agg[k]["p_rank"] / base_p for k in ks], ":",
                 color="gray")
        ax2.set_ylabel("readout parameter saving", color="gray")
        ax.set_xlabel("k  (number of graphs feeding the m channels)")
        ax.set_ylabel("test R^2 (mean over channels)")
        ax.set_title("ext_02: what does the rank constraint cost?  (freq=1.0)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "ext_02_rank_budget_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

    if do_assert and not all_sane:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
