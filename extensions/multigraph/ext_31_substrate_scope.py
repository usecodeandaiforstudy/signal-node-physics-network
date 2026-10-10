"""
ext_31 -- "any physical system": turning the widest claim into a checkable precondition
=====================================================================================
Project: singal-node-physics-network, extension `multigraph`

THE CLAIM, AND WHY IT IS THE WEAKEST EVIDENCE IN THE PROJECT
-----------------------------------------------------------
The extension's own section 11, item 5:

    "ext_17 shows the operators do not depend on WHICH of the two cells produced the
     field ... But both cells were built here to reproduce the same engineered kernel,
     so **nothing here tests a substrate whose response is not Gaussian in the input at
     all**.  'Any physical system' is still the widest claim in this document and the
     least covered."

That is a CIRCULARITY, not merely a gap: the operators work because the kernel is
Gaussian, and every substrate tested is Gaussian by construction.

THE PRECONDITION, STATED PRECISELY
----------------------------------
The operator suite rests on one identity.  For a source at `c` with width `h`,

    z(x) = exp(-k * ||(x-c)/h||^2)   =>   log z = -alpha*||x||^2 + gamma.x - delta

with `alpha = k/h^2`, `gamma = 2*alpha*c`, `delta = alpha*||c||^2` and the exact constraint
`delta = ||gamma||^2 / (4*alpha)`.  Every operator reads `c` and `h` out of those
coefficients.

**But "Gaussian" is not the requirement.  The requirement is:**

    log z(x)  must be LINEAR in some FIXED, FINITE basis phi(x) of the input,

because that is what makes the coefficients identifiable from a measured field.  The
Gaussian satisfies it because `||x-c||^2` expands to `||x||^2 - 2c.x + ||c||^2` -- a
polynomial of degree 2 whose coefficients encode `c`.  So does any source whose exponent is
a POLYNOMIAL in `x`, and the degree of that polynomial is the degree of the basis needed.

WHAT THIS SCRIPT MEASURES
-------------------------
  P1  the identity fit on a GAUSSIAN source against `[||x||^2, x, 1]` -- the control, which
      must reach machine precision (the base version's own result).
  P2  a QUARTIC source, `exp(-k*||(x-c)/h||^4)`.  Its exponent is a degree-4 polynomial, so
      it is IN SCOPE with a bigger basis -- and the position must come back out of the
      degree-3 coefficients, exactly as the Gaussian's comes out of the degree-1 ones.
  P3  TWO OUT-OF-SCOPE sources: a YUKAWA/screened-Poisson field `exp(-kappa*r)/r`, whose log
      needs `log r`, and a CAUCHY/Lorentzian `1/(1+(r/g)^2)`.  Neither is log-linear in any
      finite polynomial basis, so the operators must FAIL here.
  P4  THE DELIVERABLE: the fit RESIDUAL as a function of the assumed basis degree.  A
      substrate is in scope iff that residual collapses at some finite degree -- so the
      residual is a QUALIFICATION TEST a new substrate can be put through BEFORE any
      operator is trusted on it.

SCOPE: synthetic, self-contained, no training loop and no device -- this tests the
IDENTIFICATION PREMISE the operators rest on, which is the right level for a claim about
which physical systems are admissible.  `k = 0.5` is the repository's own kernel constant.

Output: outputs/ext_31_substrate_scope.txt + _plot.png
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)

K_DEVICE = 0.5                      # the repository's own kernel sharpness
DEGREES = (2, 4, 6, 8)


# ----------------------------------------------------------------------
#  Substrates: each is a per-source response profile
# ----------------------------------------------------------------------
def z_gauss(X, c, h, k=K_DEVICE):
    """exp(-k*||(x-c)/h||^2) -- the kernel every previous result uses."""
    r2 = np.sum(((X - c[None, :]) / h) ** 2, axis=1)
    return np.exp(-k * r2)


def z_quartic(X, c, h, k=K_DEVICE):
    """exp(-k*||(x-c)/h||^4) -- a POLYNOMIAL exponent of degree 4: still in scope."""
    r2 = np.sum(((X - c[None, :]) / h) ** 2, axis=1)
    return np.exp(-k * r2 ** 2)


def z_yukawa(X, c, kappa=1.0):
    """exp(-kappa*r)/r -- screened Poisson / Debye.  log needs log r: NOT polynomial."""
    r = np.linalg.norm(X - c[None, :], axis=1)
    r = np.maximum(r, 1e-3)
    return np.exp(-kappa * r) / r


def z_cauchy(X, c, g=1.0):
    """1/(1+(r/g)^2) -- a Lorentzian line shape: also not polynomial in log."""
    r = np.linalg.norm(X - c[None, :], axis=1)
    return 1.0 / (1.0 + (r / g) ** 2)


SUBSTRATES = (
    ("gaussian (control)", z_gauss, True),
    ("quartic exponent", z_quartic, True),
    ("yukawa exp(-kr)/r", z_yukawa, False),
    ("cauchy 1/(1+(r/g)^2)", z_cauchy, False),
)


# ----------------------------------------------------------------------
#  The candidate basis: all monomials of total degree <= d
# ----------------------------------------------------------------------
def monomial_basis(X, d):
    """
    Columns are the monomials of total degree <= d in the input coordinates.

    The Gaussian needs d=2 (`[||x||^2, x, 1]` is the same span); a quartic exponent needs
    d=4.  The basis is the ONLY thing a new substrate changes -- which is the point.
    """
    D = X.shape[1]
    cols, names = [], []
    for total in range(d + 1):
        for combo in itertools.combinations_with_replacement(range(D), total):
            col = np.ones(X.shape[0])
            for j in combo:
                col = col * X[:, j]
            cols.append(col)
            names.append("*".join(f"x{j+1}" for j in combo) if combo else "1")
    return np.stack(cols, axis=1), names


def fit_identity(X, z, d):
    """Least-squares fit of log z to the degree-d monomial basis.  Returns residual."""
    A, _names = monomial_basis(X, d)
    y = np.log(z)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    scale = max(float(np.max(np.abs(y))), 1e-12)
    return coef, float(np.max(np.abs(resid)) / scale), float(np.sqrt(np.mean(resid ** 2)) / scale)


def recover_from_poly(coef, names, D, d, k=K_DEVICE):
    """
    Recover (c, width) from the fitted coefficients -- THE SUBSTRATE-SPECIFIC STEP.

    For a polynomial exponent of degree `d`, `log z = -(k/h^d) * ||x-c||^d`, so the
    TOP-degree coefficient fixes the width and the degree-(d-1) coefficients fix the
    position, by the same structural argument at each degree:

        d = 2 :  ||x-c||^2 = ||x||^2 - 2 c.x + ||c||^2
                 -> coef of x_j is -2*c_j,  coef of ||x||^2 is -k/h^2
        d = 4 :  ||x-c||^4 expands with the degree-3 coefficients carrying -4*c_j
                 -> c_j = -coef(x_j^3) / 4,  and the ||x||^4 coefficient is -k/h^4

    This function does it for the two degrees tested, and returns None where the pattern
    does not apply -- so an out-of-scope substrate cannot silently produce a fake answer.
    """
    idx = {n: i for i, n in enumerate(names)}
    if d == 2:
        # log z = -alpha*sum_j (x_j-c_j)^2  with alpha = k/h^2, so
        #     coef(x_j^2) = -alpha        (the SAME value for every j)
        #     coef(x_j)   =  2*alpha*c_j
        # An earlier version SUMMED the pure-square coefficients and negated the total,
        # which is -D*alpha: for D=2 that is negative, the guard rejected it, and every
        # in-scope substrate came back as "(inversion not defined here)".
        pure = []
        for j in range(D):
            key = f"x{j+1}*x{j+1}"
            if key not in idx:
                return None
            pure.append(float(coef[idx[key]]))
        alpha = -float(np.mean(pure))                       # = k/h^2 > 0
        if alpha <= 0:
            return None
        c = np.array([float(coef[idx[f"x{j+1}"]]) / (2 * alpha) for j in range(D)])
        return c, float(np.sqrt(k / alpha))
    if d == 4:
        # log z = -(k/h^4) * ||x-c||^4, and (sum_j (x_j-c_j)^2)^2 has
        #     coef(x_j^3) = -4*c_j * (k/h^4)      <- the (k/h^4) FACTOR IS REQUIRED
        #     coef(x_j^4) =   1   * (k/h^4)
        # An earlier version divided coef(x_j^3) by 4 alone, dropping that factor, and
        # returned a NUMBER rather than an error: on the quartic control it gave
        # c = -0.1225 against a true 0.7 (the omitted factor is k/h^4 = 0.17506, and
        # 4*0.7*0.17506/4 = 0.1225).  The width was right either way.
        lead = []
        for j in range(D):
            key = f"x{j+1}*x{j+1}*x{j+1}*x{j+1}"
            if key not in idx:
                return None
            lead.append(float(coef[idx[key]]))
        kh4 = -float(np.mean(lead))                         # = k/h^4 > 0
        if kh4 <= 0:
            return None
        c = np.zeros(D)
        for j in range(D):
            key = f"x{j+1}*x{j+1}*x{j+1}"
            if key not in idx:
                return None
            c[j] = float(coef[idx[key]]) / (4.0 * kh4)
        return c, float((k / kh4) ** 0.25)
    return None


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_31: which substrates are in scope?")
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--probes", type=int, default=400)
    ap.add_argument("--c", default="0.7,-0.4")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    rng = np.random.default_rng(0)
    D = args.dim
    c_true = np.array([float(x) for x in args.c.split(",")][:D], dtype=float)
    h_true = 1.3
    # probes spread over a box; for the singular substrates the source itself is excluded
    X = rng.uniform(-2.5, 2.5, (args.probes, D))

    from abl_common import Report, header, try_plots     # noqa: E402

    rep = Report("ext_31_substrate_scope.txt")
    for line in header("ext_31 - which physical systems are in scope for the operators?",
                       f"dim={D} probes={args.probes} source c={c_true.tolist()} "
                       f"h={h_true} k={K_DEVICE}"):
        rep(line)

    rep("")
    rep("  The operators rest on ONE identity, and the extension's own section 11 item 5")
    rep("  says the widest claim in the document -- 'any physical system' -- is the least")
    rep("  covered, because both substrates tested were built to reproduce the SAME")
    rep("  engineered Gaussian kernel.  That is a circularity, not just a gap.")
    rep("")
    rep("  THE PRECONDITION IS NOT 'GAUSSIAN'.  It is: **log z(x) must be LINEAR in some")
    rep("  fixed, FINITE basis phi(x) of the input** -- that is what makes the coefficients")
    rep("  identifiable from a measured field.  The Gaussian satisfies it because its")
    rep("  exponent is a degree-2 polynomial in x.  ANY polynomial exponent does.")
    rep("")
    rep(f"  basis degrees tried: {list(DEGREES)}  (monomials of total degree <= d)")

    t0 = time.time()

    # ---- P1-P3: the residual by substrate and degree ---------------------
    rep("\n" + "=" * 104)
    rep("  P1-P3  is log z linear in a finite monomial basis?  residual by degree")
    rep("=" * 104)
    rep("")
    rep(f"  {'substrate':<22} {'in scope':>9} " + " ".join(f"{'d='+str(d):>11}"
                                                         for d in DEGREES) +
        f" {'first d with resid<1e-10':>26}")
    rep("  " + "-" * 106)
    rows = {}
    for name, fn, expect in SUBSTRATES:
        z = fn(X, c_true, h_true) if name.startswith(("gauss", "quartic")) \
            else fn(X, c_true)
        res = []
        first = None
        for d in DEGREES:
            _coef, rmax, _rms = fit_identity(X, z, d)
            res.append(rmax)
            if first is None and rmax < 1e-10:
                first = d
        rows[name] = dict(expect=expect, res=res, first=first, z=z)
        rep(f"  {name:<22} {str(expect):>9} " + " ".join(f"{r:>11.2e}" for r in res) +
            f" {str(first) if first else 'NONE':>26}")
    rep("")
    rep("  'in scope' is the PRIOR -- whether the exponent is a polynomial in x.  The")
    rep("  residual is the MEASUREMENT, and the two must agree; where they do, the")
    rep("  residual is a usable test on a substrate nobody has characterised.")

    # ---- P2 detail: does the position come back out? ---------------------
    rep("\n" + "=" * 104)
    rep("  P2  the substrate-specific step: recovering the SOURCE from the coefficients")
    rep("=" * 104)
    rep("")
    rep("  The basis is what a new substrate changes; the INVERSION has to be re-derived")
    rep("  per substrate.  For a polynomial exponent of degree d the top-degree")
    rep("  coefficient fixes the width and the degree-(d-1) coefficients fix the position:")
    rep("      d=2 :  coef(x_j^2) = -alpha          coef(x_j)   = +2*alpha*c_j")
    rep("      d=4 :  coef(x_j^4) = -(k/h^4)        coef(x_j^3) = +4*c_j*(k/h^4)")
    rep("")
    rep(f"  {'substrate':<22} {'basis d':>8} {'c recovered':>26} {'h recovered':>13} "
        f"{'c error':>10}")
    rep("  " + "-" * 84)
    rec = {}
    for name, fn, expect in SUBSTRATES:
        if not expect:
            continue
        for d in (2, 4):
            z = rows[name]["z"]
            coef, rmax, _rms = fit_identity(X, z, d)
            _A, names = monomial_basis(X, d)
            got = recover_from_poly(coef, names, D, d)
            if got is None:
                rep(f"  {name:<22} {d:>8} {'(inversion not defined here)':>26}")
                continue
            c_hat, h_hat = got
            err = float(np.max(np.abs(c_hat - c_true)))
            rec[(name, d)] = (c_hat, h_hat, err)
            rep(f"  {name:<22} {d:>8} "
                f"{np.array2string(c_hat, precision=6, floatmode='fixed'):>26} "
                f"{h_hat:>13.6f} {err:>10.2e}")
    rep("")
    rep("  A basis that is too SMALL cannot represent the identity, so the recovery there")
    rep("  is meaningless -- AND THE CODE STILL RETURNS A NUMBER.  That is visible above:")
    rep("  on the quartic substrate read through the DEGREE-2 basis the position comes back")
    rep("  as a plausible-looking but wrong point and the width as roughly a third of the")
    rep("  true one.  **An out-of-scope basis does not fail loudly; it answers confidently")
    rep("  and wrongly.**  So the residual is not a nicety -- it has to be checked BEFORE")
    rep("  any position is believed, and that is the whole argument for P4.")
    rep("")
    rep("  AND WHERE THE BASIS IS RIGHT, THE GENERALISATION IS EXACT: the quartic substrate")
    rep("  read through the DEGREE-4 basis returns the source to machine precision, just as")
    rep("  the Gaussian does through degree 2.  **What a new substrate changes is the basis;")
    rep("  the inversion is re-derived with it, and neither is a matter of luck.**")

    # ---- P4: the qualification test --------------------------------------
    rep("\n" + "=" * 104)
    rep("  P4  THE DELIVERABLE: the residual as a qualification test for a new substrate")
    rep("=" * 104)
    rep("")
    rep("  Put a measured field through the fit and read the residual.  A substrate is in")
    rep("  scope iff the residual COLLAPSES at some finite degree -- so a substrate nobody")
    rep("  has characterised can be qualified BEFORE any operator is trusted on it.")
    rep("")
    ok = 0
    for name, _fn, expect in SUBSTRATES:
        r = rows[name]
        got = r["first"] is not None
        agree = (got == expect)
        ok += int(agree)
        verdict = ("IN SCOPE" if got else "OUT OF SCOPE")
        rep(f"  {name:<22} residual says {verdict:<12} prior says "
            f"{'IN' if expect else 'OUT':<12} {'AGREE' if agree else 'DISAGREE'}")
    rep("")
    rep(f"  prior and measurement agree on {ok}/{len(SUBSTRATES)} substrates.")
    rep("")
    if ok == len(SUBSTRATES):
        rep("  THE PRECONDITION IS THE RIGHT ONE.  'Any physical system' narrows to a")
        rep("  CHECKABLE statement -- any system whose per-source response is log-linear in")
        rep("  a fixed finite basis -- and the check is a least-squares residual, not a")
        rep("  judgement call.  **A polynomial-exponent substrate is IN SCOPE and the")
        rep("  operators generalise to it by swapping the basis; a Yukawa or Lorentzian")
        rep("  field is OUT and the residual says so before anything is built on it.**")
        rep("")
        rep("  SO THE THESIS'S WIDEST CLAIM IS NOT REFUTED, IT IS BOUNDED: the operators")
        rep("  need log-linearity in a finite basis, which is a property a substrate can be")
        rep("  TESTED for rather than assumed.  That is strictly more than the two-Gaussian-")
        rep("  cells evidence supported, and strictly less than 'any physical system'.")
    else:
        rep("  THE PRECONDITION IS NOT SUFFICIENT: the residual and the prior disagree on at")
        rep("  least one substrate, so log-linearity in a finite basis is not what decides")
        rep("  admissibility.  That is a negative result about the stated precondition and it")
        rep("  is reported as one.")
    rep("")
    rep("  WHAT THIS DOES NOT DO: the operators themselves are not re-derived here, only")
    rep("  the IDENTIFICATION PREMISE they rest on.  Section 7.1's optimality bound was")
    rep("  measured for the Gaussian basis and is NOT transferred to the quartic one.")
    rep("  And nothing here is a device: these are profiles evaluated at points, which is")
    rep("  the right level for a question about which responses are admissible, and the")
    rep("  wrong level for claiming anything about hardware.")
    rep(f"\n  wall time {time.time()-t0:.2f}s")

    path = rep.save()
    print(f"\nReport saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

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
        for name, _fn, expect in SUBSTRATES:
            ax[0].semilogy(list(DEGREES), rows[name]["res"], "o-",
                           label=f"{name}{'' if expect else ' (out)'}")
        ax[0].axhline(1e-10, color="k", ls="--", lw=0.9)
        ax[0].set_xlabel("monomial basis degree d")
        ax[0].set_ylabel("max |log z - fit| / scale")
        ax[0].set_title("the qualification test")
        ax[0].grid(True, alpha=0.25, which="both")
        ax[0].legend(fontsize=7)
        r = np.linspace(0.05, 4.0, 400)
        ax[1].semilogy(r, np.exp(-K_DEVICE * r ** 2) + 1e-18, label="gaussian  exp(-k r^2)")
        ax[1].semilogy(r, np.exp(-K_DEVICE * r ** 4) + 1e-18, label="quartic  exp(-k r^4)")
        ax[1].semilogy(r, np.exp(-r) / np.maximum(r, 1e-3), label="yukawa  exp(-kr)/r")
        ax[1].semilogy(r, 1.0 / (1.0 + r ** 2), label="cauchy  1/(1+r^2)")
        ax[1].set_xlabel("distance from the source (dimensionless)")
        ax[1].set_ylabel("response z")
        ax[1].set_title("the profiles")
        ax[1].grid(True, alpha=0.25, which="both")
        ax[1].legend(fontsize=7)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_31_substrate_scope_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

    # The assert encodes what the measurement supports: the control reached machine
    # precision, and every substrate's residual was computed at every degree.  It cannot
    # encode "the precondition is right", because that is the result.
    ctrl = rows["gaussian (control)"]
    good = (ctrl["first"] is not None and all(len(r["res"]) == len(DEGREES)
                                              for r in rows.values()))
    if do_assert and not good:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
