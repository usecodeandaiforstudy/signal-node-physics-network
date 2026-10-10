"""
ext_05 -- the rank budget on the DIFFUSION substrate
====================================================
Project: singal-node-physics-network, extension `multigraph`

THE GAP THIS CLOSES
-------------------
ext_02, ext_03 and ext_04 all measured the CIRCUIT cell.  The base version's
second substrate is the one where its most interesting warnings were measured:

  * the amplitude calibration is MANDATORY there -- uncalibrated transfer error
    9.98 against 5.92e-08 calibrated, eight orders of magnitude, because each
    source's contribution is scaled by its own Green's-function prefactor g_j;
  * the calibration is performed through a sensor chain (noise, ambient drift,
    full-scale clipping, quantisation), and the base version's robustness study
    lives on that chain, not on the circuit cell;
  * the "self-report is anti-correlated with reality" finding is a property of
    this cell.

So "the extension was validated" is not a statement that transfers for free.

THREE QUESTIONS
---------------
 P1  Does the k >= 3 boundary transfer?  Fixed geometry, ideal device, calibrated
     -- i.e. ext_02 repeated on the other medium.

 P2  Does the boundary survive the SENSOR CHAIN?  A rank-k readout fits fewer
     parameters from readings that pass through noise, drift and quantisation, so
     the ranking could change with device level.  Levels from the base version's
     own presets.

 P3  Does the rank constraint change the COST OF FORGETTING the amplitude
     calibration?  The uncalibrated error is a per-SOURCE (per-column) rescaling
     of the feature matrix, and a rank-k readout has fewer degrees of freedom to
     absorb a column-wise distortion -- so this could go either way, which is why
     it is measured rather than argued.

HOW THE PHYSICS IS EXERCISED, AND WHY THAT MATTERS
--------------------------------------------------
Nothing here re-derives the cell's physics.  The sensor chain comes from
abl_common._measure_features (the base version's own path, read-only), and the
uncalibrated deployment is obtained by setting the layer's `calibrate` flag, which
is what `_measure_channel` uses to decide whether the injected strength is
W_ij * g_j or W_ij.  Re-deriving either would risk measuring my own algebra
instead of the device's.

Output: outputs/ext_05_diffusion_rank.txt + _plot.png
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
    Report, _measure_features, build_model, header, layer_activations,
    layer_forward, set_diffusion_defaults, set_ideal, try_plots,
)
from extensions.multigraph.ext_02_rank_budget import (     # noqa: E402
    _als_rank, _ridge, _r2_multi, _svd_candidate, make_multi_output,
)
from extensions.multigraph.ext_04_calibration_samples import (   # noqa: E402
    LAM_RELS, _lam_abs,
)

M_OUT = 8
N_PARTICLES = 40
N_VAL = 64
LEVELS = ("none", "low", "medium", "high")


def fit_readout(Zfi, Yfi, Zva, Yva, k, m, lam_rels=LAM_RELS):
    """
    Rank-k (or full-rank) readout, ridge chosen on VALIDATION per arm.

    The validation-chosen RELATIVE ridge is not a detail: ext_04 first produced a
    non-monotone curve -- and a spurious "the rank constraint wins" -- purely from
    a fixed absolute lambda, and the base version had already learned that lesson.
    Using it here keeps this run comparable to ext_02's boundary.
    """
    best = None
    for lr in lam_rels:
        lam = _lam_abs(Zfi, lr)
        val, W = None, None
        if k >= m:
            W = _ridge(Zfi, Yfi, lam)
            val = _r2_multi(Yva, Zva @ W.T)
        else:
            cands = _als_rank(Zfi, Yfi, k, lam, seed=0)
            cands.append(_svd_candidate(Zfi, Yfi, k, lam))
            for (Wc, _C, _Wg) in cands:
                v = _r2_multi(Yva, Zva @ Wc.T)
                if val is None or v > val:
                    val, W = v, Wc
        if best is None or val > best[0]:
            best = (val, W, lr)
    return best[1], best[2]


def deploy_and_score(layer, W, X, Y, substrate, calibrate, norm_mode="none"):
    """
    Put W into the layer and read the PHYSICAL output, through the base version's
    own forward path and sensor chain.

    Digital reconstruction (Z @ W.T) is deliberately NOT used here: the whole point
    of this experiment is that the diffusion cell's reading differs from the
    digital one by the per-source prefactor and by the sensor chain, so scoring the
    digital expression would measure nothing about the device.
    """
    layer.W_gauss = np.asarray(W, dtype=np.float32)
    layer.W_act = np.zeros_like(layer.W_gauss)
    if hasattr(layer, "calibrate"):
        layer.calibrate = bool(calibrate)
    r = layer_forward(layer, X, norm_mode, None, substrate=substrate)
    pred = np.asarray(r["out_g"], dtype=np.float64) + \
        np.asarray(r["out_a"], dtype=np.float64)
    return _r2_multi(Y, pred)


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_05: the rank budget on the "
                                             "diffusion substrate")
    ap.add_argument("--arch", default="F2")
    ap.add_argument("--ks", default="8,4,3,2")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    rep = Report("ext_05_diffusion_rank.txt")
    for line in header("ext_05 - the rank budget on the diffusion substrate",
                       f"arch={args.arch} k={ks} splits={args.splits} "
                       f"levels={'/'.join(LEVELS)} freq={args.freq}"):
        rep(line)

    rep("")
    rep("  Fixed geometry (ext_02's regime), readout fitted in closed form with the")
    rep("  ridge chosen on validation per arm, then DEPLOYED through the layer's own")
    rep("  physical forward.  Nothing here re-derives the cell: the sensor chain is")
    rep("  abl_common._measure_features and the uncalibrated case is the layer's own")
    rep("  `calibrate` flag, so this measures the device rather than my algebra.")

    t0 = time.time()
    m = M_OUT
    # results[level][cal][k] = list of test R^2 over splits
    results = {lv: {c: {k: [] for k in ks} for c in ("cal", "uncal")}
               for lv in LEVELS}
    lam_seen = {k: [] for k in ks}

    for split in range(args.splits):
        X_tr, Y_tr, X_te, Y_te = make_multi_output(m=M_OUT, freq=args.freq,
                                                   seed=7 + split)
        # The GEOMETRY does not depend on the device level, so it is drawn once per
        # split and reused: only the readings change with the level, which is what
        # makes the level sweep a comparison rather than a re-draw.
        set_diffusion_defaults(level="none")
        set_ideal()
        base_g = build_model(args.arch, X_tr, seed=split, substrate="diffusion")
        lay_g = base_g.layers[0]
        Ztr_full, _zg, _og, _oa = layer_activations(lay_g, X_tr, "diffusion",
                                                    "none", None)
        Zte_full, *_ = layer_activations(lay_g, X_te, "diffusion", "none", None)
        Ztr = np.asarray(Ztr_full, dtype=np.float64)[:, :N_PARTICLES]
        Zte = np.asarray(Zte_full, dtype=np.float64)[:, :N_PARTICLES]
        Zva, Yva = Ztr[-N_VAL:], Y_tr[-N_VAL:]
        Zfi, Yfi = Ztr[:-N_VAL], Y_tr[:-N_VAL]

        Ws = {}
        for k in ks:
            W, lr = fit_readout(Zfi, Yfi, Zva, Yva, k, m)
            Ws[k] = W
            lam_seen[k].append(lr)

        for lv in LEVELS:
            set_diffusion_defaults(level=lv)
            set_ideal()
            base = build_model(args.arch, X_tr, seed=split, substrate="diffusion")
            lay = base.layers[0]
            # APPLY THE SENSOR CHAIN TO THE CALIBRATION READINGS.  This is the
            # point of P2 and the first version of this script got it wrong: the
            # sensor chain (noise, drift, clipping, quantisation) is applied to Z
            # AT CALIBRATION TIME, not inside the forward, so fitting on the
            # digital Z left the chain unexercised and made the "low" level come
            # out bit-identical to "none".  _measure_features is the base
            # version's own path for exactly this.
            Ztr_l = _measure_features(
                lay, np.asarray(layer_activations(lay, X_tr, "diffusion", "none",
                                                  None)[0],
                                dtype=np.float64)[:, :N_PARTICLES], "diffusion")
            Ztr_l = np.asarray(Ztr_l, dtype=np.float64)
            Zva_l, Yva_l = Ztr_l[-N_VAL:], Y_tr[-N_VAL:]
            Zfi_l, Yfi_l = Ztr_l[:-N_VAL], Y_tr[:-N_VAL]
            for k in ks:
                W_l, _lr = fit_readout(Zfi_l, Yfi_l, Zva_l, Yva_l, k, m)
                for cal in ("cal", "uncal"):
                    results[lv][cal][k].append(
                        deploy_and_score(lay, W_l, X_te, Y_te, "diffusion",
                                         calibrate=(cal == "cal")))

    # -- P1 / P2: the boundary, by device level -----------------------------
    k_full = max(ks)
    rep("\n" + "=" * 78)
    rep("  P1/P2  test R^2 by device level (calibrated deployment)")
    rep("=" * 78)
    rep(f"  {'level':>7} " + "".join(f"{'k=' + str(k):>11}" for k in ks)
        + f"{'gap k=3':>10}")
    rep("  " + "-" * (7 + 11 * len(ks) + 10))
    boundary = {}
    for lv in LEVELS:
        row = {k: float(np.mean(results[lv]["cal"][k])) for k in ks}
        gap = row[3] - row[k_full] if 3 in row else float("nan")
        rep(f"  {lv:>7} " + "".join(f"{row[k]:>+11.4f}" for k in ks)
            + f"{gap:>+10.4f}")
        # Boundary = the smallest k that stays within 0.02 of the full-rank arm at
        # this level.  Same tolerance as ext_02/ext_03 so the numbers are
        # comparable rather than freshly chosen.
        ok = [k for k in sorted(ks) if row[k] >= row[k_full] - 0.02]
        boundary[lv] = min([k for k in ok if k < k_full], default=None)
    rep("")
    rep("  boundary (smallest k within 0.02 of full rank) per level: "
        + "  ".join(f"{lv}={boundary[lv]}" for lv in LEVELS))

    # -- P3: the cost of forgetting the calibration -------------------------
    rep("\n" + "=" * 78)
    rep("  P3  cost of forgetting the amplitude calibration, per rank")
    rep("=" * 78)
    rep("  The base version's number for this is 9.98 uncalibrated against")
    rep("  5.92e-08 calibrated in transfer error.  Here it is asked of each rank:")
    rep("  an uncalibrated reading is a per-SOURCE rescaling of the features, and a")
    rep("  rank-k readout has fewer degrees of freedom to absorb column distortion.")
    rep("")
    rep(f"  {'level':>7} " + "".join(f"{'k=' + str(k) + ' dR2':>12}" for k in ks))
    rep("  " + "-" * (7 + 12 * len(ks)))
    worst = {}
    for lv in LEVELS:
        cells = ""
        for k in ks:
            d = (float(np.mean(results[lv]["uncal"][k]))
                 - float(np.mean(results[lv]["cal"][k])))
            cells += f"{d:>+12.4f}"
            worst.setdefault(k, []).append(d)
        rep(f"  {lv:>7} {cells}")
    rep("")
    rep("  mean damage by rank (more negative = the rank suffers more):")
    order = sorted(ks, key=lambda k: float(np.mean(worst[k])))
    for k in order:
        rep(f"    k={k}: {float(np.mean(worst[k])):+.4f}")
    full_damage = float(np.mean(worst[k_full]))
    rank_damage = float(np.mean(worst[3])) if 3 in worst else float("nan")
    spread = float(np.std([np.mean(worst[k]) for k in ks]))
    rep("")
    if abs(rank_damage - full_damage) < 0.05 * abs(full_damage):
        rep(f"  The damage is the SAME for every rank to within "
            f"{abs(rank_damage - full_damage):.2f} R^2 out of a "
            f"{abs(full_damage):.0f} collapse, i.e. {100*abs(rank_damage-full_damage)/abs(full_damage):.2f}%.")
        rep("  So the rank constraint neither helps nor hurts here: forgetting the")
        rep("  amplitude calibration destroys the readout whatever its rank, and the")
        rep("  ordering between ranks is far too small to carry a claim.  The earlier")
        rep("  text in this file called that ordering a 'cost' -- it is not; it is")
        rep("  noise on a collapse.  What the table DOES establish is that the base")
        rep("  version's mandatory-calibration finding is rank-independent, so the")
        rep("  amplitude calibration cannot be traded away for parameters.")
    elif rank_damage > full_damage:
        rep(f"  The rank-3 readout loses MORE ({rank_damage:+.4f}) than the full-rank")
        rep(f"  one ({full_damage:+.4f}), so the rank constraint does not buy")
        rep("  robustness to a missing calibration.")
    else:
        rep(f"  The rank-3 readout LOSES LESS ({rank_damage:+.4f} against "
            f"{full_damage:+.4f}), the direction the degrees-of-freedom argument")
        rep("  predicts.  Check it against the across-rank spread before reporting it")
        rep(f"  as a robustness benefit (spread here: {spread:.4f}).")

    # -- verdict ------------------------------------------------------------
    rep("\n" + "=" * 78)
    rep("  VERDICT")
    rep("=" * 78)
    circuit_boundary = 3      # measured in ext_02/ext_03 on the circuit cell
    transfers = all(boundary[lv] == circuit_boundary for lv in LEVELS)
    rep("")
    rep(f"  circuit-cell boundary (ext_02/ext_03): k={circuit_boundary}")
    rep(f"  diffusion-cell boundary by level       : "
        + ", ".join(f"{lv}={boundary[lv]}" for lv in LEVELS))
    rep("")
    if transfers:
        rep(f"  TRANSFERS.  The boundary is k={circuit_boundary} on the diffusion cell")
        rep("  at EVERY device level, including the base version's own presets, so it")
        rep("  is a property of the architecture and not of one medium or one idealised")
        rep("  device.  That is the statement the extension needed and did not have.")
    else:
        rep("  DOES NOT TRANSFER unchanged.  The boundary moves with the substrate or")
        rep("  the device level, which means the extension's design point has to be")
        rep("  chosen per medium rather than once.  Report the table, not a single k.")
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
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))
        x = np.arange(len(LEVELS))
        for k in ks:
            ax.plot(x, [float(np.mean(results[lv]["cal"][k])) for lv in LEVELS],
                    "o-", label=f"k={k} calibrated")
            ax.plot(x, [float(np.mean(results[lv]["uncal"][k])) for lv in LEVELS],
                    "x:", alpha=0.6, label=f"k={k} uncalibrated")
        ax.set_xticks(x)
        ax.set_xticklabels(LEVELS)
        ax.set_xlabel("diffusion cell device level")
        ax.set_ylabel("test R^2 (mean over output channels)")
        ax.set_title("ext_05: rank budget on the diffusion substrate")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_05_diffusion_rank_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

    if do_assert and not transfers:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
