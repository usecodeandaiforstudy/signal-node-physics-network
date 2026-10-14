"""
Constructive trainer for a network made of parameters
=====================================================
Project: singal-node-physics-network, extension `multigraph`

WHAT THIS IS
------------
A training procedure for a node network that uses ONLY operators built for this
architecture.  No gradients, no finite differences, no generic optimiser:

    OP-C   propose the next node from the residual      (closed form, log-linear)
    OP-G   re-solve an existing node's geometry          (closed form, log-linear)
    strength solve   set all weights at once             (closed form, ridge)

The base version established half of this idea -- source strengths are a
CALIBRATION quantity with an exact operator, not a search quantity.  `operators.py`
supplied the other half for the geometry.  This file composes them into a trainer
and is therefore the thing the project was missing: a way to train a
parameter-composed network with operators designed for it.

THE LOOP
--------
    1. GROW     place nodes one at a time from the current residual, choosing by
                the closed-form score `best_weight` (a proposal that does not
                reduce the residual must be rejected, or the trainer becomes a
                random one)
    2. SOLVE    set every strength at once by ridge over the whole feature matrix,
                with the ridge chosen on validation -- greedy weights are far from
                optimal and this step is what makes placement robustness matter
                less
    3. REFINE   for each node, re-solve its geometry from the residual via OP-G
    4. GATE     keep the round only if it improves the HELD-OUT score

Steps 2-4 repeat; step 1 only runs in the first round, so the node COUNT is a
budget rather than a search.

WHY THE GATE IS NOT OPTIONAL
----------------------------
Every step above fits something to a residual, and a residual fit can improve
in-sample while making the model worse.  The base version learned this the hard
way (its calibration gate was scoring in-sample fits; the fix was a symmetric
out-of-sample check).  The same discipline is applied here: a round is kept only
on held-out evidence.
"""

from __future__ import annotations

import numpy as np

from extensions.multigraph.operators import (
    K_GAUSS, node_from_response, propose_node, star_probes,
)

LAM_RELS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


# ---------------------------------------------------------------------------
#  features, through the DEVICE
# ---------------------------------------------------------------------------
def _build_layer(substrate, X, centres, widths):
    """One layer of the requested substrate carrying exactly these node parameters."""
    import sys
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    if os.path.join(root, "experiments") not in sys.path:
        sys.path.insert(0, os.path.join(root, "experiments"))
        sys.path.insert(0, root)
    from abl_common import make_cell, make_layer

    D = np.asarray(X).shape[1]
    n = len(centres)
    # The CELL must come from abl_common.make_cell: a diffusion layer built with
    # cell=None creates its own default substrate and silently IGNORES
    # set_diffusion_defaults(level=...), so a device-level sweep would report the
    # same numbers at every level.  ext_17's first run did exactly that -- four
    # identical rows were the tell -- and the fix is to thread the real cell.
    return make_layer(
        substrate,
        np.asarray(centres, dtype=np.float32).T.reshape(D, n),
        np.asarray(widths, dtype=np.float32).reshape(n),
        np.ones((1, n), dtype=np.float32),
        np.zeros((1, n), dtype=np.float32),
        use_pu=False,
        cell=make_cell(substrate),
    )


def features(X, centres, widths, k=K_GAUSS, substrate="circuit", measured=False):
    """
    The node responses at X, read from a real layer of the requested SUBSTRATE.

    The trainer never evaluates its own kernel formula in place of the device's:
    `operators.node_response` exists only as the reference the verification test
    checks against.  If the two ever disagree, that is a finding, not a detail.

    `substrate` was HARD-CODED to "circuit" until ext_17.  Half B -- the whole
    operator suite -- had therefore never touched the diffusion cell, on a thesis
    whose core claim is "any physical system whose phase space can generate movable
    nonlinear activation sources".  That is a gap in the argument, not plumbing.

    `measured` puts the diffusion feature matrix through the cell's SENSOR CHAIN
    (noise, drift, clipping, quantisation) via abl_common._measure_features, which
    is what a calibration performed on hardware actually sees.  The circuit
    substrate has no separate sensor stage and is returned unchanged -- the same
    asymmetry abl_common documents.
    """
    import sys
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    if os.path.join(root, "experiments") not in sys.path:
        sys.path.insert(0, os.path.join(root, "experiments"))
        sys.path.insert(0, root)
    from abl_common import _measure_features, layer_activations

    lay = _build_layer(substrate, X, centres, widths)
    zg, za, _og, _oa = layer_activations(lay, np.asarray(X, dtype=np.float32),
                                         substrate, "none", None)
    Z = np.asarray(zg if k == K_GAUSS else za, dtype=np.float64)
    Z = Z if Z.ndim == 2 else Z.reshape(-1, 1)
    if measured:
        Z = np.asarray(_measure_features(lay, Z, substrate), dtype=np.float64)
    return Z


def node_amplitude(X, centres, widths, k=K_GAUSS, substrate="diffusion"):
    """
    The per-source amplitude a device measurement imposes: `1/g_j` per node.

    On the diffusion cell `_activations` returns a UNIT-PEAK z together with a
    calibration gain g, and a field measurement with the injection UNCALIBRATED
    returns `z/g` -- so the readout then carries a per-source factor that depends on
    the node's own width, because `g ~ t^(dim/2)` and `t ~ h^2`.  On the circuit
    substrate there is no such factor and this returns ones.

    It exists because this is the one place the two substrates genuinely differ in
    what the OPERATOR sees, and ext_17 uses it to test whether the geometry operator
    is invariant to it -- which the kernel identity predicts it must be, since the
    amplitude enters the log-response ONLY through the constant term.
    """
    if substrate != "diffusion":
        return np.ones(len(centres), dtype=np.float64)
    from substrate_diffusion import KERNELS

    lay = _build_layer(substrate, X, centres, widths)
    _z, g, _t = lay._activations(np.asarray(X, dtype=np.float64),
                                 KERNELS[0] if k == K_GAUSS else KERNELS[1])
    return 1.0 / np.maximum(np.asarray(g, dtype=np.float64), 1e-300)


# ---------------------------------------------------------------------------
#  the strength operator
# ---------------------------------------------------------------------------
def solve_strengths(Z, y, Zva=None, yva=None, lam_rels=LAM_RELS):
    """
    Ridge over the whole feature matrix, with the ridge chosen on validation.

    The relative-ridge-with-validation-selection rule is not a detail: ext_04
    produced a non-monotone curve and a spurious "the rank constraint wins" purely
    from a fixed absolute lambda, and the base version had already learned that
    lesson.  Without a validation slice the smallest candidate is taken.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    best = None
    for lr in lam_rels:
        lam = lr * float(np.mean(np.diag(Z.T @ Z)))
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
        score = (r2(yva, Zva @ w) if (Zva is not None and yva is not None)
                 else -float(np.sum((Z @ w - y) ** 2)))
        if best is None or score > best[0]:
            best = (score, w, lr)
    return best[1], best[2]


def r2(y, p):
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    ss = np.sum((y - p) ** 2)
    st = np.sum((y - y.mean()) ** 2) + 1e-300
    return float(1.0 - ss / st)


# ---------------------------------------------------------------------------
#  OP-C: propose the next node
# ---------------------------------------------------------------------------
def propose_candidates(X, residual, k, rng, n_random=4):
    """
    Several proposals for one node, all cheap, to be scored by the caller.

    A single proposal is not enough in practice: the log-profile fit is the
    principled one, but it can land where the residual's magnitude is large and its
    SIGN structure makes it useless.  Candidates are drawn from four sources and the
    caller keeps whichever actually reduces the residual -- a proposal that is never
    scored is how a constructive trainer quietly degrades into a random one.

    `n_random` is exposed because ext_14 asks whether the POOL WIDTH is what limits
    the placement: if widening it barely helps, the limitation is elsewhere.  It was
    hard-coded, which made that question unaskable.
    """
    cands = []

    # (a) the log-linear proposal: where is the residual big, and at what scale
    pr = propose_node(k, X, residual, top_frac=0.5)
    if pr.get("ok"):
        cands.append((pr["centre"], pr["width"]))

    # (b) centre on the largest-|residual| point, width from the local spread
    a = np.abs(residual)
    i = int(np.argmax(a))
    spread = float(np.std(X, axis=0).mean())
    cands.append((np.asarray(X[i], dtype=np.float64), max(spread * 0.5, 1e-3)))

    # (c) the residual-weighted centroid, at a couple of scales
    wgt = a / max(a.sum(), 1e-300)
    cen = (wgt[:, None] * X).sum(axis=0)
    for mult in (0.5, 1.0):
        cands.append((cen, max(spread * mult, 1e-3)))

    # (d) random draws, so the search is not trapped by a bad residual profile
    lo, hi = X.min(axis=0), X.max(axis=0)
    for _ in range(n_random):
        cands.append((rng.uniform(lo, hi), float(rng.uniform(0.25, 1.5) * spread)))

    return cands


# ---------------------------------------------------------------------------
#  OP-G: refine an existing node
# ---------------------------------------------------------------------------
def refine_node(X_tr, residual, w, k, select="leverage"):
    """
    Re-solve ONE node's geometry from the residual, in closed form.

    Given the node's current strength w, the node should look like residual/w, so the
    log-linear inverse of `operators` applies directly to the residual profile.
    Returns None when the residual has no positive structure for this node, and the
    caller then keeps the geometry it had -- which is why a refusal here is safe
    rather than fatal.

    WHICH POINTS IT FITS ON, and the constraint that decides it: the residual is
    known only where the TARGET is known, so this cannot probe at designed points --
    it can only choose a subset of the training inputs.  It therefore selects by
    LEVERAGE, which is the D-optimal choice when the candidate set is fixed.  An
    earlier version simply took every point with a positive profile, which let
    far-away points with large |log profile| dominate the fit.  `star_probes` is not
    used here on purpose: it belongs to `identify_placement`, the device-side
    operation that really can read anywhere.
    """
    from extensions.multigraph.operators import leverage_select

    y = np.asarray(residual, dtype=np.float64).ravel()
    if abs(w) < 1e-12:
        return None
    prof = y / w
    if np.all(prof <= 0):
        return None
    # OP-G inverts a RESPONSE, and a response is positive, so the negative part of
    # the profile is dropped rather than rectified: rectifying would invent
    # structure that the node cannot represent anyway.
    # ORDER MATTERS, and getting it wrong made the leverage rule 7x WORSE in the
    # controlled test: the relative floor that says which readings are informative
    # must be applied BEFORE selecting, because leverage on Phi prefers the EXTREMES
    # of the design space -- which for a node are exactly the points where its
    # response is negligible.  Selecting first let leverage pick points that the
    # operator then discarded, leaving a worse set than no selection at all.
    pos = prof > 0
    if not np.any(pos):
        return None
    floor = 1e-6 * float(prof[pos].max())      # OP-G's own default min_rel
    keep = prof > floor
    if int(np.count_nonzero(keep)) < X_tr.shape[1] + 2:
        return None
    sel = (leverage_select(X_tr, keep, n_target=max(4 * X_tr.shape[1], 32))
           if select == "leverage" else np.flatnonzero(keep))
    Xs = X_tr[sel]
    r = node_from_response(k, Xs, prof[sel])
    if not r.get("ok"):
        return None
    width = float(r["width"])
    if not np.isfinite(width) or width <= 0:
        return None
    return np.asarray(r["centre"], dtype=np.float64), width


# ---------------------------------------------------------------------------
#  the trainer
# ---------------------------------------------------------------------------
def fit_cascade(X_tr, y_tr, X_va, y_va, *, layer_sizes, k=K_GAUSS, rounds=3,
                seed=0, select="leverage", refine=False):
    """
    Build a CASCADE of operator-trained layers, stagewise.

    The base version's architecture is a wire-direct cascade with a scalar residual
    readout: `h_0 = X`, `out_k = W_k z_k(h_{k-1})`, `h_k = out_k`, and the prediction
    is the SUM of the layers' scalar outputs.  Extending the operator trainer to it
    is the architectural half of `ext_10`, and the extension is the same idea the
    base version already applies to the strengths, one level up:

        for each layer:  take the residual the layers so far cannot explain,
                         train ONE operator-built layer on it,
                         feed that layer's output forward as the next input.

    Each layer is trained by `fit_operator` in its OWN input space, so nothing about
    the operators changes -- what is new is only that they compose across depth.

    Greedy by construction, and that is stated rather than hidden: layer k is fitted
    to the residual left by layers 1..k-1 and is never revisited, whereas the
    finite-difference arm optimises every layer's geometry jointly.  That makes the
    comparison in `ext_10` a comparison of a greedy constructor against a joint
    search, which is the honest description of the two methods.
    """
    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64).ravel()
    X_va = np.asarray(X_va, dtype=np.float64)
    y_va = np.asarray(y_va, dtype=np.float64).ravel()

    h_tr, h_va = X_tr, X_va
    pred_tr = np.zeros_like(y_tr)
    pred_va = np.zeros_like(y_va)
    layers = []
    total_evals = 0
    for li, n_nodes in enumerate(layer_sizes):
        r_tr = y_tr - pred_tr
        r_va = y_va - pred_va
        out = fit_operator(h_tr, r_tr, h_va, r_va, n_nodes=n_nodes, k=k,
                           rounds=rounds, seed=seed + 100 * li, select=select,
                           refine=refine)
        layers.append(out)
        total_evals += out["n_evals"]
        # the layer's own scalar output, which is what the cascade feeds forward
        z_tr = features(h_tr, out["centres"], out["widths"]) @ out["weights"]
        z_va = features(h_va, out["centres"], out["widths"]) @ out["weights"]
        total_evals += 2
        pred_tr = pred_tr + z_tr
        pred_va = pred_va + z_va
        h_tr = z_tr.reshape(-1, 1)
        h_va = z_va.reshape(-1, 1)
    return dict(layers=layers, n_evals=total_evals,
                val_r2=r2(y_va, pred_va))


def cascade_predict(layers, X):
    """Forward a cascade built by `fit_cascade`."""
    h = np.asarray(X, dtype=np.float64)
    pred = None
    for out in layers:
        z = features(h, out["centres"], out["widths"]) @ out["weights"]
        pred = z if pred is None else pred + z
        h = z.reshape(-1, 1)
    return pred


def _fit_once(X_tr, y_tr, X_va, y_va, *, n_nodes, k=K_GAUSS, rounds=4,
              seed=0, verbose=False, select="leverage",
              refine=False, n_random=4, substrate="circuit", measured=False):
    """
    Train with the operators only.

    THE LOOP, and which steps actually do work:

        GROW    OP-C proposes nodes from the residual, each scored before acceptance
        SOLVE   all strengths at once, ridge chosen on validation
        GATE    keep a round only on held-out evidence
        REFINE  OP-G per node -- **off by default, and that is the measured answer**

    `refine` defaults to False because ext_09 measured it to be INERT for training:
    OP-G inverts a response, so refining a node against a residual that this node
    already explains returns the SAME node -- a fixed point.  GROW + SOLVE place each
    node to fit the current residual, i.e. already at that fixed point, so REFINE has
    nothing to move.  Measured: with two different selection rules the returned
    geometries agreed to `9.7e-08` (one float32 ULP) on one seed and BIT-IDENTICALLY
    on the other two, while end-to-end R^2 was identical.

    Turning it off does not change the score, which is the point: the option remains
    because OP-G is verified as an inverse and is genuinely useful for fault
    tolerance -- `identify_placement` measures a mis-placed source exactly -- but
    claiming it as a training step would overstate the method.

    Returns centres, widths, weights, the held-out history, the MEASURED evaluation
    count, and the refinement counters (all zero when `refine` is off).
    """
    evals = [0]

    def feats(X, cs, ws):
        evals[0] += 1
        return features(X, cs, ws, k=k, substrate=substrate, measured=measured)

    rng = np.random.default_rng(seed)
    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64).ravel()
    X_va = np.asarray(X_va, dtype=np.float64)
    y_va = np.asarray(y_va, dtype=np.float64).ravel()

    centres, widths, w = [], [], np.zeros(0)
    hist = []

    def score(cs, ws, ws_w):
        if len(cs) == 0:
            return -np.inf
        Zv = feats(X_va, cs, ws)
        return r2(y_va, Zv @ ws_w)

    # -- 1. GROW -----------------------------------------------------------
    for j in range(n_nodes):
        Z = feats(X_tr, centres, widths) if centres else np.zeros((X_tr.shape[0], 0))
        pred = Z @ w if centres else np.zeros_like(y_tr)
        res = y_tr - pred
        cands = propose_candidates(X_tr, res, k, rng, n_random=n_random)
        best = None
        for (c, h) in cands:
            # Scored in the DEVICE's own features, not in node_response, so the
            # comparison uses the same numbers the model will actually see.
            z = feats(X_tr, [c], [h]).ravel()
            den = float(np.dot(z, z))
            if den <= 0:
                continue
            wj = float(np.dot(res, z)) / den
            g = float(np.dot(res, res) - np.sum((res - wj * z) ** 2))
            if best is None or g > best[0]:
                best = (g, c, h, wj)
        if best is None:
            break
        _, c, h, _wj = best
        centres.append(np.asarray(c, dtype=np.float64))
        widths.append(float(h))
        # re-solve ALL strengths rather than appending the greedy one: this is the
        # strength operator doing its job, and it is what makes the placement
        # order matter much less
        w, _lr = solve_strengths(feats(X_tr, centres, widths), y_tr,
                                 feats(X_va, centres, widths), y_va)
        if verbose:
            print(f"    node {j+1}/{n_nodes}: val R2 {score(centres, widths, w):+.4f}",
                  flush=True)

    best_state = (list(centres), list(widths), np.array(w))
    hist.append(score(centres, widths, w))

    # -- 2-4. SOLVE / GATE, and REFINE when asked for ----------------------
    # The counters exist because the end-to-end score CANNOT tell whether REFINE did
    # anything: two different selection rules once reported identical numbers, which
    # is what an inert step looks like from the outside.
    refine_tried, refine_none, refine_kept = 0, 0, 0
    for rd in range(rounds):
        if refine:
            # refine each node's geometry from the residual
            Z = feats(X_tr, centres, widths)
            res = y_tr - Z @ w
            for i in range(len(centres)):
                refine_tried += 1
                got = refine_node(X_tr, res, w[i], k, select=select)
                if got is None:
                    refine_none += 1
                    continue
                c_new, h_new = got
                c_old, h_old = centres[i], widths[i]
                centres[i], widths[i] = c_new, h_new
                Zt = feats(X_tr, centres, widths)
                w_new, _lr = solve_strengths(Zt, y_tr,
                                             feats(X_va, centres, widths), y_va)
                if score(centres, widths, w_new) > score(centres, widths, w):
                    w = w_new
                    res = y_tr - Zt @ w
                    refine_kept += 1
                else:                                 # keep the old geometry
                    centres[i], widths[i] = c_old, h_old
        # one more joint solve, then the gate
        Zt = feats(X_tr, centres, widths)
        w_new, _lr = solve_strengths(Zt, y_tr, feats(X_va, centres, widths), y_va)
        s_new = score(centres, widths, w_new)
        hist.append(s_new)
        # GATE: keep the round only on held-out evidence.  Without this the loop
        # can walk uphill in-sample and downhill in reality.
        if s_new >= hist[-2]:
            w = w_new
            best_state = (list(centres), list(widths), np.array(w))
        else:
            centres, widths, w = (list(best_state[0]), list(best_state[1]),
                                  np.array(best_state[2]))
            break

    return dict(centres=best_state[0], widths=best_state[1], weights=best_state[2],
                history=hist, n_evals=evals[0], refine_tried=refine_tried,
                refine_refused=refine_none, refine_kept=refine_kept)


def fit_operator(X_tr, y_tr, X_va, y_va, *, n_nodes, k=K_GAUSS, rounds=4,
                 seed=0, verbose=False, select="leverage",
                 refine=False, n_random=4, substrate="circuit", measured=False,
                 restarts=1):
    """
    The operator trainer, with the restart-and-select procedure.

    ext_14 identified this repair -- run M restarts and keep the one with the best
    HELD-OUT score -- from the placement bound, and ext_16 then measured what it is
    worth: on the discontinuous task a single greedy run lands `0.234` below what
    restarts of its OWN configuration find, against `0.031` on the smooth control.  It
    had been identified, measured and left uninstalled; this is it installed.

    `restarts=1` runs exactly the single greedy construction every earlier result used,
    with the SAME seed, so those numbers stay reproducible bit for bit.  `restarts=M`
    runs M independent constructions -- same configuration, different search seed -- and
    keeps the best.

    SELECTION IS ON VALIDATION, NEVER ON THE TEST SET.  That is not a detail: ext_14's
    first version took the maximum test score over its restarts as the baseline, which
    is the error this project warns about and which inflated the baseline by `+0.0577`.

    The selection evaluations are charged to `n_evals`, because the cost unit here is
    forward evaluations and a restart that is not paid for is not a method.

    UNSAFE ABOVE 1, AND THAT IS MEASURED RATHER THAN CAUTION.  ext_19 found the repair
    reaches the ceiling at `M = 4` on `steps` (`+0.2045`, exactly ext_16's best-found)
    and selected `-7.5180` at `M = 16`, and `-0.4235` on the SMOOTH control at `M = 2`.
    ext_21 then measured three candidate guards and none works: on `steps` the selected
    restart scores `-7.5182` while its OWN POOL'S MEAN is `-0.3404`, i.e. selection picks
    worse than a random member of the pool, and the winner's curse explains only
    `+0.1657` of that.  The cause is upstream -- the proposal family wins only `42.5%`
    of grow steps on `steps` against an equal-sized random pool where chance is `~50%`,
    so the restarts differ by noise and a 51-sample split cannot rank noise.

    `restarts=1` is the default for that reason.  Raising it is a research knob, not a
    fix, and it should not be turned on for a task whose proposals carry no signal.
    """
    m = max(1, int(restarts))
    runs = []
    total_evals = 0
    for r in range(m):
        out = _fit_once(X_tr, y_tr, X_va, y_va, n_nodes=n_nodes, k=k, rounds=rounds,
                        seed=seed + 1000 * r, verbose=verbose, select=select,
                        refine=refine, n_random=n_random, substrate=substrate,
                        measured=measured)
        total_evals += out["n_evals"]
        Zva = features(X_va, out["centres"], out["widths"], k=k,
                       substrate=substrate, measured=measured)
        total_evals += 1                      # scoring a restart is a forward pass
        runs.append((r2(y_va, Zva @ out["weights"]), out))
    runs.sort(key=lambda kv: -kv[0])
    best = dict(runs[0][1])
    best.update(n_evals=total_evals, restarts=m,
                restart_scores=[float(s) for s, _o in runs],
                seed=seed)
    return best
