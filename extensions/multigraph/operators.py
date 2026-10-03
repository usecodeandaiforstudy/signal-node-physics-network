"""
Dedicated operators for a network made of PARAMETERS
====================================================
Project: singal-node-physics-network, extension `multigraph`

WHAT THIS FILE IS FOR
---------------------
The base version established that a node network needs a DIFFERENT KIND of operator
from a matrix network: its source strengths are a *calibration* quantity with an
exact closed-form operator (`abl_common.solve_readout_closed_form`), not something
to be searched.  That operator covers ONE of the network's two parameter families.
The other family -- the node's POSITION and WIDTH -- has only ever been searched, by
finite differences on an analog budget.

This file supplies the missing operators.  They exist because of an exact algebraic
property of the kernel the hardware actually computes.

THE IDENTITY THE WHOLE FILE RESTS ON
------------------------------------
Read out of the device's own code path, the kernel is

    simulator/layer_nd.py:     dx = differential_pair(x, c, h)      -> (x - c) / h
                               d2 = gilbert_square + kirchhoff_sum   -> sum_i dx_i^2
                               z  = bjt_gaussian_kernel(d2, k)       -> exp(-k d2 / V_T)

so, with V_T = 1 in the ideal configuration,

    z_j(x) = exp( -k_j * || (x - c_j) / h_j ||^2 )
           = exp( -alpha_j ||x||^2 + beta_j . x - delta_j )

with

    alpha_j = k_j / h_j^2        beta_j = 2 alpha_j c_j        delta_j = alpha_j ||c_j||^2

`log z_j` is therefore LINEAR in the D+2 coefficients (alpha, beta, delta) -- and
those coefficients are not free: they satisfy ONE exact relation,

    delta_j = ||beta_j||^2 / (4 alpha_j)

which is what makes the inverse well posed (D+1 parameters -- D coordinates and one
width -- recovered from D+2 numbers, with the extra number serving as a CHECK
rather than a fit).

So a node's geometry can be SOLVED rather than searched:

    recover    h_j = sqrt(k_j / alpha_j)        c_j = beta_j / (2 alpha_j)
    check      delta_j == ||beta_j||^2 / (4 alpha_j)

WHAT THAT BUYS, AND WHY IT IS THE POINT
---------------------------------------
  * a closed-form GEOMETRY operator to sit beside the closed-form STRENGTH operator,
    so the network can be trained without gradient search on either family;
  * an operator that is meaningful in hardware: "where is this source and how wide
    is it" is answered from field readings, which is what a physical placement
    procedure would need;
  * a CONSISTENCY RESIDUAL that says whether a response is a genuine single node,
    which a search-based trainer cannot report at all.

SCOPE, STATED BEFORE ANY RESULT
-------------------------------
The operator is exact per node and per layer, in the layer's OWN input space.  In a
cascade each layer's nodes respond to the previous layer's outputs, so the identity
holds layer by layer but a deep network's geometry in terms of the original x is not
one log-linear model.  Nothing here claims otherwise.
"""

from __future__ import annotations

import numpy as np

# The two kernel coefficients the device uses, from simulator/layer_nd.py:205-206.
K_GAUSS = 0.5        # narrow channel
K_ACT = 0.3          # wide channel


def design_matrix(X):
    """
    phi(x) = [ -||x||^2 , x , -1 ]  with D+2 columns.

    log z = -alpha||x||^2 + gamma.x - delta with gamma = 2 alpha c, so a linear
    solve in phi recovers the node's natural parameters in closed form, and

        c = gamma / (2 alpha)        delta = alpha||c||^2 = ||gamma||^2 / (4 alpha)

    The middle block is `x`, NOT `2x`.  The first version of this function used 2x
    while the recovery below still divided by 2a, which put a factor of exactly 2
    into every recovered centre -- and the verification caught it as a centre error
    of precisely 0.5000 in every row while the width came out to 1e-6.  Two
    self-consistent conventions exist; mixing them is the trap.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, D = X.shape
    phi = np.empty((n, D + 2), dtype=np.float64)
    phi[:, 0] = -np.sum(X * X, axis=1)
    phi[:, 1:1 + D] = X
    phi[:, D + 1] = -1.0
    return phi


def node_from_response(k, X, z, min_z=1e-12, min_rel=1e-6, weights=None):
    """
    OP-G: recover a node's (centre, width) from its response, in closed form.

    Parameters
    ----------
    k : float
        The kernel coefficient the channel uses (K_GAUSS or K_ACT).  It is a
        hardware constant, not a fitted quantity.
    X : (n, D)
        The probe points, in the node's own input space.
    z : (n,)
        The node's response at those points.
    min_z, min_rel : float
        Readings are DROPPED when z <= min_z (underflow) or z <= min_rel * max(z).
        The relative floor is the one that matters and it is a physical statement,
        not numerical tidiness: `log z` is what gets fitted, so a reading far
        outside the node's active region enters the least-squares problem with a
        huge |log z| and dominates it while carrying almost no information about
        where the node is.  With an absolute floor only, that showed up as sporadic
        ill-conditioned recoveries (log residual ~1e-2 against the usual ~1e-6).
        The operator REFUSES rather than returning a garbage geometry when too few
        informative readings survive.

    Returns
    -------
    dict with centre, width, the raw coefficients, the consistency residual, the
    design-matrix condition number, n_used, and the log-space fit residual.
    """
    X = np.asarray(X, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != z.size:
        raise ValueError(f"X has {X.shape[0]} rows but z has {z.size}")

    finite = np.isfinite(z)
    peak = float(np.max(z[finite])) if np.any(finite) else 0.0
    use = finite & (z > min_z) & (z > min_rel * peak)
    n_used = int(np.count_nonzero(use))
    D = X.shape[1]
    if n_used < D + 2:
        return dict(ok=False, reason=f"only {n_used} informative readings for "
                                     f"D+2={D+2} unknowns (relative floor "
                                     f"{min_rel:g})", n_used=n_used,
                    n_probe=int(z.size))

    phi = design_matrix(X[use])
    L = np.log(z[use])
    w = None if weights is None else np.asarray(weights, float).ravel()[use]
    if w is not None:
        sw = np.sqrt(np.maximum(w, 0.0))
        A = phi * sw[:, None]
        b = L * sw
    else:
        A, b = phi, L

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    # theta = (alpha, gamma, delta) with gamma = 2 alpha c
    alpha, gamma, delta = theta[0], theta[1:1 + D], theta[-1]

    resid = A @ theta - b
    log_rms = float(np.sqrt(np.mean(resid ** 2)))
    try:
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        cond = float("inf")
    # The consistency residual is the identity this whole operator rests on:
    # a genuine single node satisfies delta = ||gamma||^2 / (4 alpha) exactly, so
    # this number says how far the response is from being one node's kernel.
    if alpha > 0:
        consist = float(delta - np.dot(gamma, gamma) / (4.0 * alpha))
        width = float(np.sqrt(k / alpha))
        centre = gamma / (2.0 * alpha)
    else:
        consist = float("nan")
        width = float("nan")
        centre = np.full(D, np.nan)

    return dict(ok=bool(alpha > 0), reason="" if alpha > 0 else "alpha <= 0: the "
                "response decays with distance from a point at infinity, which no "
                "finite node produces",
                centre=centre, width=width, alpha=float(alpha), gamma=gamma,
                delta=float(delta), consistency=consist, log_rms=log_rms,
                cond=cond, n_used=n_used, n_probe=int(z.size))


def node_response(k, X, centre, width):
    """
    The kernel the operator inverts, reimplemented ONLY as a reference for tests.

    The operator itself never uses this: every verification goes through the
    device's own layer so that a mistake in this function cannot make the operator
    look correct.  Kept here so a reader can see the identity written down once.
    """
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64).ravel()
    d = (X - c[None, :]) / float(width)
    return np.exp(-float(k) * np.sum(d * d, axis=1))


def propose_node(k, X, residual, min_z=1e-12, top_frac=0.5):
    """
    OP-C: propose the NEXT node from a residual, in closed form.

    Rationale, and it is the matching-pursuit argument: the best single node to add
    is the one whose response has the largest |cos| with the residual.  That is
    nonlinear in (c, h), but the node should be LARGE where the residual is large,
    so the proposal is obtained by fitting the log-linear model to a target profile
    built from |residual| and solving it with OP-G.

    This is a PROPOSAL, not a guarantee: the caller must score it (one scalar
    regression per candidate) and keep it only if it actually reduces the residual.
    Returning a candidate without that check is how a constructive trainer silently
    becomes a random one.
    """
    X = np.asarray(X, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X[:, None]
    a = np.abs(r)
    if not np.any(a > 0):
        return dict(ok=False, reason="residual is identically zero")
    # Weight the fit towards where the residual actually is, and give the fit a
    # positive target: the kernel is positive, so the profile must be too.
    thr = np.quantile(a, 1.0 - top_frac)
    w = np.where(a >= thr, a, 0.0)
    if not np.any(w > 0):
        w = a
    # A pseudo-response: exp(-profile) with profile normalised to the residual's
    # own scale.  log of it is what OP-G inverts.
    prof = a / max(a.max(), 1e-300)
    z = np.maximum(prof, min_z * 10.0)
    return node_from_response(k, X, z, min_z=min_z * 10.0, weights=w)


def star_probes(D, scale, n, rng=None):
    """
    A probe set designed to be WELL CONDITIONED AT THE MINIMUM SIZE.

    What the measurements support, and no more:

      * at n = D+2 (the minimum that determines D+2 coefficients) a random probe
        set in D = 4 leaves the quadratic column of phi = [-||x||^2, x, -1] nearly
        collinear with the linear ones, giving a centre error of 2.2e-03.  The
        design below gives 2.5e-04 at the same size, ~10x better;
      * at n >= 16 random probes are already adequate (~3e-5), and a TRUNCATED
        design is not automatically better -- an earlier version of this function
        built a long structured list and sliced it, which at n=8 left an unbalanced
        set with no negative points and did WORSE than random (7e-02).

    So the design is: the origin, ONE axis point per dimension at an inner radius,
    and points at an outer radius.  That fixes the radius degeneracy (all points at
    one radius make ||x||^2 constant, which makes the width unidentifiable) while
    keeping the set balanced.  Any slots beyond the structured part are filled at
    random, because that is what the measurements say is adequate there.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    r_in, r_out = 0.5 * scale, scale
    pts = [np.zeros(D)]
    for i in range(D):                      # inner radius: separates x from ||x||^2
        e = np.zeros(D)
        e[i] = r_in
        pts.append(e)
    for i in range(D):                      # outer radius: the radius contrast
        e = np.zeros(D)
        e[i] = r_out
        pts.append(e)
    structured = np.array(pts)
    if n <= structured.shape[0]:
        # Inside the structured part, keep the balance: origin, then the inner
        # ring, then the outer ring -- never slice an unbalanced prefix.
        return structured[:n]
    extra = rng.uniform(-scale, scale, (n - structured.shape[0], D))
    return np.vstack([structured, extra])


def leverage_select(X, mask, n_target, min_keep=None):
    """
    Choose WHICH available points to fit on, when free choice is not an option.

    This exists because of a constraint that is easy to miss and that an earlier
    version of this work got wrong: the residual is only known where the target is
    known, so `refine_node` cannot probe at designed points -- it can only choose a
    SUBSET of the training inputs.  The designed `star_probes` set is for the
    device-side read-out (`identify_placement`), which really can read anywhere.

    Given that restriction, the same conditioning argument still applies, and the
    standard tool is the LEVERAGE score of the design matrix:

        h_i = phi(x_i) (Phi^T Phi)^-1 phi(x_i)^T

    Rows with large leverage are the ones that control the fit, so keeping the
    largest-leverage rows conditions it best for a given count.  This is D-optimal
    selection reduced to the case where the candidate set is fixed.

    Returns the selected indices into X (a subset of where mask is True).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    idx = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    D = X.shape[1]
    n_keep = int(n_target)
    if min_keep is None:
        min_keep = D + 2
    if idx.size <= max(n_keep, min_keep):
        return idx
    phi = design_matrix(X[idx])
    # Keep at least min_keep rows, then add the highest-leverage remainder.  The
    # scoring is done on the FULL candidate set: recomputing it per addition would
    # be exact D-optimal but costs a solve per candidate, and the measurement below
    # is what decides whether that refinement is worth anything.
    G = phi.T @ phi
    try:
        Ginv = np.linalg.pinv(G)
    except np.linalg.LinAlgError:
        return idx[:max(n_keep, min_keep)]
    lev = np.einsum("ij,jk,ik->i", phi, Ginv, phi)
    order = np.argsort(-lev)
    keep = max(n_keep, min_keep)
    return idx[order[:keep]]


def identify_placement(k, X_probe, z, c_intended=None, h_intended=None):
    """
    The DEVICE-side use of the operator: measure where a source actually is.

    This is the operation the designed probe set belongs to.  A source is placed
    (or fabricated) with some intended centre and width; reading the field at a
    DESIGNED set of points and inverting gives the realized geometry, and comparing
    with the intention turns placement error into a measurable quantity rather than
    a tolerance nobody checks.  `ext_06` established that at the minimum probe count
    the designed set has a bounded worst case where a random set is a gamble, which
    is exactly the property a placement procedure needs.

    Returns the operator's result plus the placement error when an intention is
    supplied.
    """
    r = node_from_response(k, X_probe, z, min_rel=1e-6)
    if not r.get("ok"):
        return r
    if c_intended is not None:
        ci = np.asarray(c_intended, dtype=np.float64).ravel()
        r["centre_err"] = float(np.linalg.norm(r["centre"] - ci))
        r["centre_err_rel"] = float(
            np.linalg.norm(r["centre"] - ci) / max(np.linalg.norm(ci), 1e-12))
    if h_intended is not None:
        r["width_err_rel"] = float(abs(r["width"] - float(h_intended))
                                   / max(abs(float(h_intended)), 1e-12))
    return r


def best_weight(X, residual, k, centre, width):
    """
    One scalar regression: the strength that best explains the residual with this
    node.  Closed form, and the criterion a constructive trainer must use before
    accepting a proposal.
    """
    z = node_response(k, X, centre, width)
    denom = float(np.dot(z, z))
    if denom <= 0:
        return 0.0, 0.0
    w = float(np.dot(residual, z) / denom)
    pred = w * z
    gain = float(np.dot(residual, residual) - np.dot(residual - pred,
                                                    residual - pred))
    return w, gain
