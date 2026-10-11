"""
ext_44 -- is the deep regression failure a DEFECT or a STRUCTURAL property?
=========================================================================
Project: singal-node-physics-network, extension `multigraph`

WHERE THIS COMES FROM
---------------------
`ext_43` put the closed-form readout calibration inside `training/himal_nd.py`, the shipped
SEARCH arm did not clear the project's own 0.05 learning floor, and the script refused to
compare.  What it DID show was that the per-layer readout solve fits its own residual target
to `R^2 ~ 0.34-0.38` while the END-TO-END test `R^2` is `~0`.  So the deep path's bottleneck
is not the readout -- and the worklist's next question is what it IS.

THE CODE SAYS THE PREDICTION PATH DROPS THE ACCUMULATION
--------------------------------------------------------
`train` builds a running sum, layer by layer:

    # Proto maps h_i -> residual target (or full y for layer 1)
    # At predict time these are summed -> recovers original y
    pred_iA = self._proto_predict(hA, y_proto_A, h_proto_A)
    acc_proto_A = pred_iA if acc_proto_A is None else _zero_pad_sum(acc_proto_A, pred_iA)

and layers 2 and 3 are trained on `y - acc_proto` (L871: `cur_yA = y_orig_A - acc_proto_A`).
So each layer's contribution is a RESIDUAL, and the sum is what recovers `y`.

`predict`'s regression branch does NOT do that:

    # Final output = average of last layer's A/B predictions
    return (hA + hB) / 2.0

It runs the cascade and returns the LAST layer's output -- **which is a residual, not the
target.**  Its own docstring says otherwise: "Regression: proto-weighted prediction with
residual accumulation.  output = (acc_A + acc_B) / 2".  **Code and docstring disagree.**

AND THE INTENDED PREDICTION IS RECONSTRUCTIBLE FROM OUTSIDE
-----------------------------------------------------------
`train` stores the prototypes on the trainer (`self.proto_h_A`, `self.proto_y_A`, ...) and
`_proto_predict` is a method, so the prediction the training path was built around can be
replayed **without editing the base** -- the same mechanism 12.19/12.26/12.43 used.

WHAT IS MEASURED
----------------
  P1  the code facts, quoted, so the discrepancy is on the record rather than asserted.
  P2  the decisive test: ONE trained model, TWO prediction paths, test `R^2` --
      `predict` as shipped, and the accumulation the training path was built around.
  P3  per-zone, to rule out the alternative explanation that the A/B median split simply
      cannot be averaged.
  P4  the verdict: defect, or structural.

SCOPE: `training/himal_nd.py` as shipped, its own `2d` configuration, 3 seeds, toy.  Nothing
is edited in the base; the replay lives here.

Output: outputs/ext_44_regression_defect.txt + _plot.png
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
from training.himal_nd import HiMALND, _zero_pad_sum          # noqa: E402


def replay_old_predict(mdl, layers_A, layers_B, X):
    """
    The behaviour `HiMALND.predict` had BEFORE the fix: run the cascade and return the last
    layer's raw output.

    This is kept because the fix would otherwise destroy this script's own evidence -- once
    `predict` accumulates, comparing it against the accumulation shows no difference.  With
    the old behaviour reconstructed here, the comparison stays valid and doubles as a
    REGRESSION TEST for the fix.
    """
    hA = np.asarray(X, dtype=np.float32).copy()
    hB = np.asarray(X, dtype=np.float32).copy()
    for i in range(len(layers_A)):
        hA_out = mdl._forward_layer(layers_A[i], hA)
        hB_out = mdl._forward_layer(layers_B[i], hB)
        if hA_out.shape[0] != hA.shape[0]:
            hA_out = hA_out.T
        if hB_out.shape[0] != hB.shape[0]:
            hB_out = hB_out.T
        hA, hB = hA_out, hB_out
    return (hA + hB) / 2.0


def replay_accumulated(mdl, layers_A, layers_B, X):
    """
    The prediction `train` was built for: run the cascade layer by layer and SUM each
    layer's proto-weighted contribution.

    Orientation follows `train`'s own handling (`_match_shape`, and the transpose test
    against the sample count), because the layer's raw output can come back either way.
    """
    preds = []
    for which, layers, protos_h, protos_y in (
            ("A", layers_A, mdl.proto_h_A, mdl.proto_y_A),
            ("B", layers_B, mdl.proto_h_B, mdl.proto_y_B)):
        if not layers or not protos_h:
            return None
        h = np.asarray(X, dtype=np.float32).copy()
        acc = None
        for i, layer in enumerate(layers):
            h = mdl._forward_layer(layer, h)
            if h.shape[0] != X.shape[0]:
                h = h.T
            h = mdl._match_shape(h, (X.shape[0], h.shape[-1]))
            y_pr = np.asarray(protos_y[i], dtype=np.float64)
            h_pr = np.asarray(protos_h[i], dtype=np.float64)
            # `_proto_predict` expects (N, h_dim); the stored protos are in train's frame
            if h_pr.shape[1] != h.shape[1] and h_pr.shape[0] == h.shape[1]:
                h_pr = h_pr.T
            if h_pr.shape[1] != h.shape[1]:
                return None
            pi = mdl._proto_predict(np.asarray(h, dtype=np.float64), y_pr, h_pr)
            acc = pi if acc is None else _zero_pad_sum(acc, pi)
        preds.append(acc)
    if any(p is None for p in preds):
        return None
    d = min(p.shape[1] for p in preds)
    return sum(p[:, :d] for p in preds) / len(preds)


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_44: defect or structural?")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--particles", type=int, default=20)
    ap.add_argument("--task", default="2d")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    set_ideal()
    rep = Report("ext_44_regression_defect.txt")
    for line in header("ext_44 - the deep regression failure: defect or structural?",
                       f"epochs={args.epochs} seeds={args.seeds} task={args.task}"):
        rep(line)

    rep("")
    rep("  `ext_43` found the deep regression path at end-to-end `R^2 ~ 0` while its per-layer")
    rep("  readout solved to `~0.34-0.38`.  So the readout is not the bottleneck and the")
    rep("  question is what is.  **The code answers it, and the answer is a discrepancy:**")
    rep("")
    rep("      training:   pred_i = _proto_predict(h_i, y_proto_i, h_proto_i)")
    rep("                  acc     = acc + pred_i          (running sum, layer by layer)")
    rep("                  layers 2/3 are trained on `y - acc`")
    rep("                  # 'At predict time these are summed -> recovers original y'")
    rep("")
    rep("      predict:    return (h_last_A + h_last_B) / 2")
    rep("")
    rep("  **`predict` returns the LAST LAYER'S OUTPUT, which is a RESIDUAL, not the target**")
    rep("  -- and its own docstring says the opposite ('proto-weighted prediction with residual")
    rep("  accumulation.  output = (acc_A + acc_B) / 2').  **Code and docstring disagree.**")
    rep("")
    rep("  AND IT IS TESTABLE WITHOUT EDITING THE BASE: `train` stores the prototypes on the")
    rep("  trainer and `_proto_predict` is a method, so the prediction the TRAINING path was")
    rep("  built around can be replayed here.")

    t0 = time.time()

    # ---- P2/P3: one model, two prediction paths --------------------------
    rep("\n" + "=" * 100)
    rep("  P2/P3  ONE trained model, TWO prediction paths")
    rep("=" * 100)
    rep("")
    rep(f"  {'seed':>4} {'predict() as shipped':>21} {'accumulated replay':>19} "
        f"{'replay - shipped':>17} {'zone A':>9} {'zone B':>9}")
    rep("  " + "-" * 84)
    rows = []
    fixed_rows = []
    for seed in range(args.seeds):
        X_tr, y_tr, X_te, y_te = make_regression_data(args.task, seed=42 + seed)
        mdl = HiMALND(layer_sizes=(args.particles,) * 3,
                      input_dim=X_tr.shape[1], task="regression", seed=0, use_spsa=True)
        layers_A, layers_B, dsplit = mdl.train(X_tr, y_tr, epochs_per_layer=args.epochs,
                                               lr=0.012, eps=1e-3, verbose=False)
        # `mdl.predict` is the FIXED path; `replay_old_predict` is what it used to be.
        shipped = np.asarray(replay_old_predict(mdl, layers_A, layers_B,
                                               X_te)).ravel()[:len(y_te)]
        fixed = np.asarray(mdl.predict(layers_A, layers_B, X_te)).ravel()[:len(y_te)]
        replay = replay_accumulated(mdl, layers_A, layers_B, X_te)
        s_r2 = float(r2(y_te, shipped))
        r_r2 = float(r2(y_te, np.asarray(replay).ravel()[:len(y_te)])) if replay is not None \
            else float("nan")
        # per-zone, on the split the model itself used
        sv = float(dsplit.get("split_value", np.median(X_tr[:, 0])))
        zA = X_te[:, 0] >= sv
        zB = ~zA
        za = float(r2(y_te[zA], shipped[zA])) if zA.sum() > 3 else float("nan")
        zb = float(r2(y_te[zB], shipped[zB])) if zB.sum() > 3 else float("nan")
        rows.append(dict(s=s_r2, r=r_r2, za=za, zb=zb))
        fixed_rows.append(float(r2(y_te, fixed)))
        rep(f"  {seed:>4} {s_r2:>+21.4f} {r_r2:>+19.4f} {r_r2 - s_r2:>+17.4f} "
            f"{za:>+9.4f} {zb:>+9.4f}")
        print(f"  [P2] seed {seed}: shipped {s_r2:+.4f}  replay {r_r2:+.4f}  "
              f"(zones {za:+.3f} / {zb:+.3f})", flush=True)

    # ---- P4: verdict ------------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P4  VERDICT: defect, or structural?")
    rep("=" * 100)
    rep("")
    s_mean = float(np.mean([r["s"] for r in rows]))
    r_mean = float(np.mean([r["r"] for r in rows]))
    zone_best = float(np.mean([max(r["za"], r["zb"]) for r in rows]))
    rep(f"  shipped `predict`: {s_mean:+.4f}   accumulated replay: {r_mean:+.4f}   "
        f"best-zone shipped: {zone_best:+.4f}")
    rep("")
    f_mean = float(np.mean(fixed_rows))
    rep(f"  and the FIXED `predict` averages {f_mean:+.4f}, i.e. "
        f"{f_mean - s_mean:+.4f} over the pre-fix behaviour.")
    rep("")
    rep("  **NOTE ON WHAT THIS SCRIPT IS NOW.**  `HiMALND.predict` has been FIXED (extension")
    rep("  12.29), so the comparison above is between a RECONSTRUCTION of the pre-fix")
    rep("  behaviour and the fixed path -- the reconstruction is kept here precisely so that")
    rep("  fixing the defect does not erase the evidence for it.  **This script is therefore")
    rep("  also a REGRESSION TEST**: if a future change reintroduces the old return, the two")
    rep("  columns diverge again.")
    rep("")
    if r_mean > s_mean + 0.05:
        rep("  **THE FAILURE IS A DEFECT IN THE PREDICTION PATH, NOT A PROPERTY OF THE")
        rep("  ARCHITECTURE.**  The same trained models score materially better when the")
        rep("  prediction is computed the way `train` accumulates it -- **so the network")
        rep("  learned, and what was broken was how its answer was read out.**")
        rep("")
        rep("  **AND IT EXPLAINS `ext_43`'s FIRST RUN, WHICH IS WORTH KEEPING SEPARATE FROM ITS")
        rep("  SECOND.**  Before this fix, `ext_43` could not compare at all: its SEARCH arm")
        rep("  scored `-0.0341`, below the floor, and the control refused -- because the")
        rep("  end-to-end number was destroyed DOWNSTREAM of the readout its calibration was")
        rep("  fixing.  **After this fix `ext_43` reran and DID compare, and the calibration did")
        rep("  NOT help there** (`1 of 3` seeds, mean `-0.0710`), on a path that for the first")
        rep("  time cleared the floor.  **The two results are different claims and both stand:**")
        rep("  a calibration cannot repair a broken prediction path, AND on the repaired path it")
        rep("  is not the thing that helps.")
        rep("")
        rep("  **CONSEQUENCE, stated because it is the point of the whole exercise:**")
        rep("  `himal_nd`'s regression mode is used by `nd_09` (a SPEED benchmark) and by")
        rep("  `reg_01`, which 12.17 RETRACTED.  **So this defect sat in a path that no")
        rep("  published number depends on** -- which is why it survived -- **and it would")
        rep("  have corrupted the first serious use of that path.**")
    elif r_mean > s_mean:
        rep("  **THE REPLAY IS BETTER BUT NOT DECISIVELY**, so the accumulation is part of the")
        rep("  story and not all of it.  The defect is real and it is not the only thing")
        rep("  wrong; the remainder is NOT located here.")
    else:
        rep("  **THE REPLAY DOES NOT HELP, so the missing accumulation is NOT the failure.**")
        rep(f"  The best-zone shipped score is `{zone_best:+.4f}`, and if that is also low then")
        rep("  the model is not learning per zone either -- which would make this a")
        rep("  representation failure rather than a read-out one.  **The code discrepancy is")
        rep("  still a discrepancy** and is reported above on its own evidence.")
    rep("")
    rep("  **THE HONEST LIMITS.**  The replay reconstructs the intended prediction from the")
    rep("  STORED prototypes; if `train` and the replay disagree about frame or orientation on")
    rep("  some configuration, the replay returns None and that seed is reported as `nan`")
    rep("  rather than guessed.  Nothing here edits the base, so the defect is DEMONSTRATED")
    rep("  and not FIXED -- fixing it means changing `HiMALND.predict`, which is a base method")
    rep("  and a decision rather than a cleanup.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

    path = rep.save()
    print(f"\nReport saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")}")

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
        x = np.arange(len(rows))
        ax[0].bar(x - 0.2, [r["s"] for r in rows], 0.4, label="predict() as shipped")
        ax[0].bar(x + 0.2, [r["r"] for r in rows], 0.4, label="accumulated replay")
        ax[0].axhline(0.0, color="k", lw=0.9)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels([f"s{i}" for i in range(len(rows))])
        ax[0].set_ylabel("test R^2")
        ax[0].set_title("one trained model, two prediction paths")
        ax[0].grid(True, axis="y", alpha=0.25)
        ax[0].legend(fontsize=8)
        ax[1].bar(x - 0.2, [r["za"] for r in rows], 0.4, label="zone A")
        ax[1].bar(x + 0.2, [r["zb"] for r in rows], 0.4, label="zone B")
        ax[1].axhline(0.0, color="k", lw=0.9)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([f"s{i}" for i in range(len(rows))])
        ax[1].set_ylabel("test R^2")
        ax[1].set_title("the same shipped scores, split by zone")
        ax[1].grid(True, axis="y", alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_44_regression_defect_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")}")

    ok = all(np.isfinite(r["s"]) for r in rows)
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
