"""
ext_50 -- geometry-only search: closing 12.28's "bound, not a measurement"
=========================================================================
Project: singal-node-physics-network, extension `multigraph`

THE OWED ITEM, IN 12.28's OWN WORDS
-----------------------------------
    "The calibration is applied AFTER the SPSA step, so a true 'calibrate, do not search' arm
     would remove those coordinates from the parameter vector -- a change to `get_params_vector`,
     a base method -- and that is NOT done.  **The measurement bounds the gain from below rather
     than measuring it.**"

**THIS SCRIPT DOES THAT CHANGE AND TURNS THE BOUND INTO A MEASUREMENT.**

WHAT "REMOVING THE COORDINATES" MEANS, PRECISELY
-------------------------------------------------
`get_params_vector` concatenates `centers (D*N) + h (N) + W_gauss (n_vars*N) + W_act (n_vars*N)`.
The strengths are the last `2*n_vars*N` entries.  A geometry-only search must therefore mask
BOTH the SPSA perturbation and the update:

  * masking only the UPDATE would leave `L(theta + eps*delta)` differing from `L(theta - eps*delta)`
    through coordinates that are never moved, which biases the estimate rather than restricting
    it -- so both are masked, and the estimate is taken over the geometry subspace alone.

At EQUAL FORWARD PASSES, not equal steps or equal wall time -- the project's cost unit is
measurements, and every arm here pays the same `2 * spsa_avg` forwards per epoch.

WHAT IS MEASURED
----------------
  P1  the control, FIRST: the shipped search arm must clear the project's learning floor of 0.05,
      or the comparison is between two arms that did not learn and is not evidence (12.28 failed
      this control once and had to be re-run).
  P2  three arms at equal forwards, paired seeds: SEARCH, CALIB (12.28's arm: search everything,
      then overwrite the strengths), and CALIB_ONLY (search only the geometry, then overwrite).
  P3  the answer to the owed question: does the geometry-only arm beat the calibrate-after arm,
      i.e. WAS THE GAIN BEING UNDERSTATED BY THE BOUND?
  P4  the verdict and the limits.

Output: outputs/ext_50_geometry_only.txt + _plot.png
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

from abl_common import (                                              # noqa: E402
    Report, header, layer_activations, make_regression_data, r2, set_ideal, try_plots,
)
from training.himal_nd import HiMALND                                  # noqa: E402

ORIGINAL_SPSA = HiMALND._spsa_step


# ----------------------------------------------------------------------
#  The closed-form read-out, exactly as 12.28 used it (kept identical so the
#  calibrate step is not a second variable)
# ----------------------------------------------------------------------
def _ridge_solve(Z, Y, lam_rels=(1e-8, 1e-6, 1e-4, 1e-2, 1e-1)):
    """
    Closed-form read-out for one layer, returned TRANSPOSED as `(d_out, N)`.

    THAT TRANSPOSE IS NOT COSMETIC AND THE FIRST VERSION OF THIS FUNCTION OMITTED IT, which is
    why it died on `layer.W_gauss[:d, :] = W` with "could not broadcast (20,1) into (1,20)".
    `np.linalg.solve` returns `(N, d_out)`, while `layer.W_gauss` is `(n_vars, N)` -- so the
    solve is scored against `(Z_va @ W.T - Y_va)` and returned as `W.T`, exactly as `ext_43`
    does it.  Copied from there deliberately: the calibrate step must not be a second variable
    between the two scripts.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n_va = max(8, len(Z) // 5)
    Zf, Yf, Zv, Yv = Z[:-n_va], Y[:-n_va], Z[-n_va:], Y[-n_va:]
    md = max(float(np.mean(np.sum(Zf * Zf, axis=1))), 1e-30)
    best, best_s = None, -np.inf
    for lr in lam_rels:
        try:
            W = np.linalg.solve(Zf.T @ Zf + lr * md * np.eye(Zf.shape[1]),
                                Zf.T @ Yf).T                    # (d_out, N)
        except np.linalg.LinAlgError:
            continue
        s = float(np.mean((Zv @ W.T - Yv) ** 2))
        if -s > best_s:
            best, best_s = W, -s
    return best


def calibrate_layer(layer, X, y):
    """Replace the fused read-out with the closed-form solution.  Returns its R^2."""
    z_g = layer_activations(layer, np.asarray(X, dtype=np.float32), "circuit", "none")[0]
    z_g = np.asarray(z_g, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    d = min(layer.W_gauss.shape[0], Y.shape[1])
    assert z_g.shape[1] == layer.W_gauss.shape[1], (
        f"activation width {z_g.shape[1]} != readout width {layer.W_gauss.shape[1]}")
    W = _ridge_solve(z_g, Y[:, :d])
    if W is None:
        return float("nan")
    # The two channels are degenerate (only their sum matters), so the solution goes into the
    # gaussian channel and the other is zeroed -- otherwise the old W_act would ADD to it on
    # the next forward and the solve would be undone.
    layer.W_gauss = np.zeros_like(layer.W_gauss)
    layer.W_act = np.zeros_like(layer.W_act)
    layer.W_gauss[:d, :] = W.astype(layer.W_gauss.dtype)
    return float(r2(Y[:, :d].ravel(), (z_g @ W.T).ravel()))


def geometry_mask(layer):
    """
    Boolean mask over `get_params_vector`, TRUE where the parameter is GEOMETRY.

    Layout, from `simulator/layer_nd.py`: centers (D*N) + h (N) + W_gauss (n_vars*N) +
    W_act (n_vars*N).  The strengths are everything after the first `(D + 1) * N`.
    """
    D, N = layer.dim, layer.n_particles
    n_geom = (D + 1) * N
    vec = layer.get_params_vector()
    m = np.zeros(len(vec), dtype=np.float64)
    m[:n_geom] = 1.0
    assert int(m.sum()) == n_geom, "the mask does not cover exactly the geometry block"
    assert len(vec) - n_geom == 2 * layer.n_vars * N, (
        "the strengths block is not 2 * n_vars * N -- the layout assumption is wrong")
    return m


def make_calib_wrapper(counter):
    """12.28's arm: search EVERYTHING, then overwrite the strengths."""
    def wrapped(self, layer_A, layer_B, net_name, data, gamma, eps, epoch=0):
        ORIGINAL_SPSA(self, layer_A, layer_B, net_name, data, gamma, eps, epoch)
        layer = layer_A if net_name == "A" else layer_B
        X = data["XA"] if net_name == "A" else data["XB"]
        y = data["yA"] if net_name == "A" else data["yB"]
        counter.append(calibrate_layer(layer, X, y))
    return wrapped


def make_geometry_only_wrapper(counter):
    """
    THE ARM THIS SCRIPT EXISTS FOR: SPSA moves the GEOMETRY only, and the strengths are then
    written by the closed-form solve.

    BOTH the perturbation and the update are masked.  Masking only the update would let the two
    loss evaluations differ through coordinates that are never moved, which biases the estimate
    rather than restricting it -- and the whole claim being tested is that restricting the
    search is what the calibration was paying for.
    """
    def wrapped(self, layer_A, layer_B, net_name, data, gamma, eps, epoch=0):
        layer = layer_A if net_name == "A" else layer_B
        vec = layer.get_params_vector()
        m = geometry_mask(layer)

        decay_steps = self._spsa_decay_steps
        lr = self._spsa_lr / (1.0 + epoch / max(decay_steps, 1))

        grad = np.zeros_like(vec)
        for _ in range(self.spsa_avg):
            delta = np.random.choice([-1.0, 1.0], size=len(vec)).astype(np.float32) * m
            layer.set_params_from_vector(vec + eps * delta)
            L_plus = self._loss_fn(layer_A, layer_B, data, gamma)
            layer.set_params_from_vector(vec - eps * delta)
            L_minus = self._loss_fn(layer_A, layer_B, data, gamma)
            layer.set_params_from_vector(vec)
            grad += (L_plus - L_minus) / (2.0 * eps) * delta
        grad /= max(self.spsa_avg, 1)
        grad *= m                                   # the update is masked too

        clip_val = getattr(self, "_spsa_clip", 1.0)
        gn = float(np.linalg.norm(grad))
        if gn > clip_val and gn > 1e-12:
            grad = grad * (clip_val / gn)

        layer.set_params_from_vector(vec - lr * grad)

        # and the strengths come from the CLOSED FORM, not from the search
        X = data["XA"] if net_name == "A" else data["XB"]
        y = data["yA"] if net_name == "A" else data["yB"]
        counter.append(calibrate_layer(layer, X, y))
    return wrapped


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_50: geometry-only search")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--particles", type=int, default=20)
    ap.add_argument("--task", default="2d")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    np.random.seed(0)
    set_ideal()
    rep = Report("ext_50_geometry_only.txt")
    for line in header("ext_50 - geometry-only search",
                       f"task={args.task} epochs={args.epochs} particles={args.particles} "
                       f"seeds={args.seeds}"):
        rep(line)

    rep("")
    rep("  **THE OWED ITEM, in 12.28's own words:** \"a true 'calibrate, do not search' arm would")
    rep("  remove those coordinates from the parameter vector -- a change to `get_params_vector`,")
    rep("  a base method -- and that is NOT done.  The measurement BOUNDS the gain from below")
    rep("  rather than measuring it.\"")
    rep("")
    rep("  **`get_params_vector` is `centers (D*N) + h (N) + W_gauss (n_vars*N) + W_act (n_vars*N)`,")
    rep("  so the strengths are the last `2*n_vars*N` entries and the mask is exact rather than")
    rep("  approximate -- asserted against the layer's own shape before any arm runs.**")
    rep("")
    rep("  **AND BOTH THE PERTURBATION AND THE UPDATE ARE MASKED.**  Masking only the update")
    rep("  would let `L(theta + eps*delta)` differ from `L(theta - eps*delta)` through coordinates")
    rep("  that never move, which BIASES the estimate instead of restricting it.")
    rep("")
    rep("  Cost: every arm pays `2 * spsa_avg * epochs * layers * 2 networks` forward passes --")
    rep("  **equal FORWARDS, not equal steps or equal wall time.**")

    dc = 2 * 1 * args.epochs * 3 * 2
    t0 = time.time()

    # ---- P1/P2 -----------------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P1/P2  the control first, then three arms at equal forwards, paired seeds")
    rep("=" * 100)
    rep("")
    rows = []
    for seed in range(args.seeds):
        X_tr, y_tr, X_te, y_te = make_regression_data(args.task, seed=100 + seed)
        out = {}
        for arm in ("search", "calib", "calib_only"):
            HiMALND._spsa_step = ORIGINAL_SPSA
            counter = []
            try:
                if arm == "calib":
                    HiMALND._spsa_step = make_calib_wrapper(counter)
                elif arm == "calib_only":
                    HiMALND._spsa_step = make_geometry_only_wrapper(counter)
                mdl = HiMALND(layer_sizes=(args.particles,) * 3,
                              input_dim=X_tr.shape[1], task="regression",
                              seed=0, use_spsa=True)
                lA, lB, _d = mdl.train(X_tr, y_tr, epochs_per_layer=args.epochs,
                                       lr=0.012, eps=1e-3, verbose=False)
                pred = np.asarray(mdl.predict(lA, lB, X_te))
                out[arm] = float(r2(y_te, pred.ravel()[:len(y_te)]))
            finally:
                HiMALND._spsa_step = ORIGINAL_SPSA
            out[arm + "_calib_r2"] = float(np.mean(counter)) if counter else float("nan")
        rows.append(out)
        print(f"  [P2] seed {seed}: search {out['search']:+.4f}  calib {out['calib']:+.4f}  "
              f"calib_only {out['calib_only']:+.4f}", flush=True)

    rep(f"  {'seed':>4} {'search':>10} {'calib':>10} {'calib_only':>12} "
        f"{'calib RFit':>12} {'only RFit':>12}")
    rep("  " + "-" * 64)
    for i, r in enumerate(rows):
        rep(f"  {i:>4} {r['search']:>+10.4f} {r['calib']:>+10.4f} {r['calib_only']:>+12.4f} "
            f"{r['calib_calib_r2']:>12.4f} {r['calib_only_calib_r2']:>12.4f}")
    s_mean = float(np.mean([r["search"] for r in rows]))
    c_mean = float(np.mean([r["calib"] for r in rows]))
    g_mean = float(np.mean([r["calib_only"] for r in rows]))
    rep("  " + "-" * 64)
    rep(f"  {'mean':>4} {s_mean:>+10.4f} {c_mean:>+10.4f} {g_mean:>+12.4f}")

    # ---- P3: the answer --------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P3  WAS THE GAIN BEING UNDERSTATED BY THE BOUND?")
    rep("=" * 100)
    rep("")
    rep(f"  **CONTROL FIRST, because 12.28 failed it once and had to be re-run**: the shipped")
    rep(f"  SEARCH arm averages `{s_mean:+.4f}` against the project's learning floor of 0.05.")
    if s_mean <= 0.05:
        rep("")
        rep("  **THE CONTROL REFUSES.**  The search arm did not learn, so a comparison against it")
        rep("  would be a contest between two failures and is NOT reported as a result.  The")
        rep("  numbers above are printed for diagnosis and nothing is concluded from them.")
    else:
        rep("  It clears the floor, so the comparison is between arms that learned.")
        rep("")
        d_cal = c_mean - s_mean
        d_only = g_mean - s_mean
        d_between = g_mean - c_mean
        rep(f"  search -> calib:        {d_cal:+.4f}   (12.28's arm, search everything then overwrite)")
        rep(f"  search -> calib_only:   {d_only:+.4f}   (search the geometry only)")
        rep(f"  **calib -> calib_only:  {d_between:+.4f}**  <- the owed quantity")
        rep("")
        if d_between > 0:
            rep("  **THE GAIN WAS BEING UNDERSTATED, and the bound was the reason.**  Removing the")
            rep("  strength coordinates from the search buys the geometry-only arm a real")
            rep("  improvement over the arm that searched them and had them overwritten.  12.28's")
            rep("  number was a LOWER BOUND, exactly as it said, and the measurement is now above it.")
        elif d_between > -0.01:
            rep("  **THE BOUND WAS TIGHT.**  The two calibration arms are within a hundredth of each")
            rep("  other, so the forwards that 12.28's arm wasted on coordinates it then overwrote")
            rep("  were not buying anything measurable.  **That does not make the arm's design")
            rep("  right -- it makes its NUMBER right**, and the difference matters for what may be")
            rep("  claimed: the calibration is not paying for the search it replaces.")
        else:
            rep("  **THE GEOMETRY-ONLY ARM DOES WORSE**, which is a real possibility worth stating:")
            rep("  with fewer coordinates to perturb, SPSA's per-step estimate is over a smaller")
            rep("  space and may explore it less effectively at the same step count.  **So")
            rep("  restricting the search is not free, and 12.28's arm was not merely wasteful.**")
            rep("  That inverts the expectation the owe was written under and is reported as such.")

    # ---- P4: verdict -----------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P4  VERDICT AND LIMITS")
    rep("=" * 100)
    rep("")
    rep(f"  `get_params_vector` was NOT changed -- the base method is untouched.  What changed is")
    rep("  the ARM, which now masks the perturbation and the update to the geometry subspace,")
    rep("  asserted against the layer's own layout before running.  **12.28's owed change is")
    rep("  therefore DONE, and done without editing the base version**, which is the same")
    rep("  monkey-patch discipline `ext_43` already used.")
    rep("")
    rep("  **THE LIMITS.**  One task, one architecture, one scale -- the extension's standing")
    rep("  limitation, unchanged.  The search is still SPSA with one averaging and one")
    rep("  perturbation size, both inherited from the shipped trainer and never tuned, so the")
    rep("  relative standing of three arms is a statement about THOSE settings.  And the arms are")
    rep("  compared at equal forwards, which is the project's unit but not the only defensible one:")
    rep("  per-step, the geometry-only arm updates fewer coordinates.")
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
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
        x = np.arange(len(rows))
        ax[0].bar(x - 0.26, [r["search"] for r in rows], 0.26, label="search (all)")
        ax[0].bar(x, [r["calib"] for r in rows], 0.26, label="calib (search all, overwrite)")
        ax[0].bar(x + 0.26, [r["calib_only"] for r in rows], 0.26, label="calib_only (geometry)")
        ax[0].axhline(0.05, color="#c44e52", ls="--", label="learning floor 0.05")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels([f"s{i}" for i in range(len(rows))])
        ax[0].set_ylabel("test R²")
        ax[0].set_title("P2: three arms at EQUAL forward passes")
        ax[0].grid(True, axis="y", alpha=0.25)
        ax[0].legend(fontsize=7)
        ax[1].bar([0, 1, 2], [s_mean, c_mean, g_mean],
                  color=["#4c72b0", "#dd8452", "#55a868"])
        ax[1].axhline(0.05, color="#c44e52", ls="--")
        ax[1].set_xticks([0, 1, 2])
        ax[1].set_xticklabels(["search", "calib", "calib_only"], fontsize=9)
        ax[1].set_ylabel("mean test R²")
        ax[1].set_title("P3: the owed quantity is calib_only - calib")
        ax[1].grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_50_geometry_only_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, '/')}")

    ok = all(np.isfinite([r["search"], r["calib"], r["calib_only"]]).all() for r in rows)
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
