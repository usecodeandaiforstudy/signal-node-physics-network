"""
ext_32 -- the "5x better" direct-residual claim, tested as a CONTROLLED comparison
=================================================================================
Project: singal-node-physics-network, extension `multigraph`

THE CLAIM, AND WHERE IT COMES FROM
----------------------------------
`THEORY.md` section 7.3 reports that forming the residual target for layers 2..3 from
each layer's RAW scalar output is **~5x better** than from the proto-weighted prediction,
and says that finding "changes the design principle".  `README.md` carries the number as

    | Regression Depth | 2D nonlinear function | R^2 Gain | +0.064 (direct residual) vs +0.013 (Proto) |

so the metric is the **depth gain** -- what layers 2 and 3 add on top of layer 1 -- and
`0.064 / 0.013 = 4.9`, which is the 5x.

The base version's own status (`FINDINGS.md` section 9, `THEORY.md` section 7.3) says the
comparison **has not been reproduced in this codebase**: a short check agreed in
*direction* but "nowhere near that magnitude", which is why `residual_mode` defaults to
`"proto"`.

WHY THE ORIGINAL COMPARISON COULD NOT SETTLE IT
-----------------------------------------------
The two numbers come from **two different programs**, and the differences are not one:

| | `reg_01_proto_baseline.py` (the `+0.013` side) | `reg_02_direct_residual.py` (the `+0.064` side) |
|---|---|---|
| residual mode | proto-weighted | direct |
| kernel | **Cauchy, anisotropic h, multi-scale k** | **pure Gaussian** |
| architecture | `HiMALND`, dual A/B network | **separate single network, no A/B split** |
| trainer | **`HiMALND` with TWO METHODS MONKEY-PATCHED at runtime** | its own hand-written loop |
| particles | 20/15/10, 80 epochs | its own `CONFIGS` |

`reg_01` replaces the training loop (`train_one_layer = make_train_one_layer(...)`) and the
forward (`_forward_layer = lambda self, l, X: l.fused_output(X)`) before training, so it is
**not running the shipped trainer**; and `reg_02` does not import `HiMALND` at all.  **At
least four things change between the two numbers and the 5x is attributed to one of them.**

THE CONTROLLED TEST, WHICH THE SWITCH NOW MAKES POSSIBLE
-------------------------------------------------------
`HiMALND` now takes `residual_mode`, and that parameter reaches exactly one branch
(`training/himal_nd.py:845`).  So the shipped trainer can compare the two schemes with
**everything else held fixed** -- which is what this script does:

  * SHIPPED `HiMALND`, no monkey-patching of any kind;
  * `reg_01`'s own configuration, so the proto side is comparable to the published one;
  * the ONLY difference between the two arms is `residual_mode`.

MEASURED BOTH WAYS, because the base's two inference paths disagree
------------------------------------------------------------------
`reg_01` computes its depth curve by SUMMING the per-layer proto predictions, and its own
source comment at that point reads "wire-direct accumulation **broken in old path**" --
referring to `HiMALND.predict`, which walks the layers wire-direct and returns the LAST
layer's output with no accumulation at all, while the training design assumes the protos
are summed ("At predict time these are summed -> recovers original y",
`training/himal_nd.py:889`).  **So both paths are reported here**: the proto-summed depth
curve (the claim's own metric) and `HiMALND.predict` (the shipped inference).

SCOPE: one synthetic task (the two scripts' shared one), the diffusion-free circuit layer
path, `reg_01`'s architecture, and the seed count printed in the report.  No base file is
edited; nothing is monkey-patched.

Output: outputs/ext_32_residual_mode.txt + _plot.png
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from physical_simulator import NoiseConfig              # noqa: E402
NoiseConfig.set_level("none")

from training.himal_nd import HiMALND, _zero_pad_sum    # noqa: E402

# reg_01's own configuration, so the proto arm is comparable with the published one
LAYER_SIZES = (20, 15, 10)
OUTPUT_DIMS = (4, 3, 2)
EPOCHS, LR, EPS = 80, 0.02, 0.01
GAMMA = 0.04
MODES = ("proto", "direct")


def make_data(n=400, noise=0.15, seed=42):
    """The function BOTH scripts use: y = sin(3x1) + cos(2x2) + 0.5*x1*x2 + noise."""
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, (n, 2)).astype(np.float32)
    y = (np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 1]
         + np.random.normal(0, noise, n)).astype(np.float32)
    idx = np.random.permutation(n)
    n_te = int(round(0.3 * n))
    te, tr = idx[:n_te], idx[n_te:]
    return X[tr], y[tr], X[te], y[te]


def r2(pred, true):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    mse = float(np.mean((pred - true) ** 2))
    return 1.0 - mse / max(float(np.var(true)), 1e-12)


def proto_sum_predict(trainer, lA, lB, X, n_layers):
    """
    `reg_01`'s own depth evaluation, reproduced: sum the per-layer proto predictions.

    This is the CLAIM'S metric.  It is written here rather than called from `reg_01`
    because that script is a program, not a module -- and because reproducing it is the
    point: if the number moves, the reader should be able to see which recipe moved it.
    """
    hA, hB = np.asarray(X, dtype=np.float32).copy(), np.asarray(X, dtype=np.float32).copy()
    acc_A = acc_B = None
    for i in range(n_layers):
        hA_out = trainer._forward_layer(lA[i], hA)
        hB_out = trainer._forward_layer(lB[i], hB)
        if hA_out.shape[0] != hA.shape[0]:
            hA_out = hA_out.T
        if hB_out.shape[0] != hB.shape[0]:
            hB_out = hB_out.T
        pa = trainer._proto_predict(hA_out, trainer.proto_y_A[i], trainer.proto_h_A[i])
        pb = trainer._proto_predict(hB_out, trainer.proto_y_B[i], trainer.proto_h_B[i])
        if acc_A is None:
            acc_A, acc_B = pa.copy(), pb.copy()
        else:
            acc_A = _zero_pad_sum(acc_A, pa)
            acc_B = _zero_pad_sum(acc_B, pb)
        hA, hB = hA_out, hB_out
    return ((acc_A + acc_B) / 2.0).ravel()


def run_once(mode, data, seed, epochs=EPOCHS):
    X_tr, y_tr, X_te, y_te = data
    # `HiMALND` HAS NO SEED PARAMETER AND NEVER SEEDS THE GLOBAL RNG: `_init_layer`
    # draws its centres and W_gauss from `np.random.uniform`, so two calls with identical
    # arguments give different networks depending on what ran before.  Without this line
    # the two arms get DIFFERENT INITIALISATIONS and the comparison is not controlled --
    # which is visible in the depth curve, because layer 1 is trained identically under
    # both modes and its R^2 must therefore match.
    np.random.seed(seed)
    trainer = HiMALND(
        layer_sizes=LAYER_SIZES, output_dims=OUTPUT_DIMS, input_dim=2,
        init_range=(-3.0, 3.0), alignment_weight=GAMMA, n_protos=10,
        temperature=0.15, task="regression", use_spsa=False,
        residual_mode=mode,
    )
    t0 = time.time()
    lA, lB, _split = trainer.train(X_tr, y_tr, split_method="median_x",
                                   epochs_per_layer=epochs, lr=LR, eps=EPS,
                                   verbose=False)
    curve = [r2(proto_sum_predict(trainer, lA, lB, X_te, k), y_te) for k in (1, 2, 3)]
    # The SHIPPED inference path.  Under reg_01's own output_dims=(4,3,2) the last layer
    # emits TWO columns while the target is scalar, so `HiMALND.predict` cannot be scored
    # against y at all -- recorded as NaN with the shapes, because that is a finding and
    # not an error to paper over.
    raw = np.asarray(trainer.predict(lA, lB, X_te))
    yv = np.asarray(y_te).ravel()
    if raw.size == yv.size:
        ship, ship_note = r2(raw.ravel(), yv), f"shape {raw.shape}"
    else:
        ship, ship_note = float("nan"), f"shape {raw.shape} vs y {yv.shape} -- NOT SCORABLE"
    return dict(mode=mode, curve=curve, gain=curve[2] - curve[0], shipped=ship,
                ship_note=ship_note, secs=time.time() - t0)


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_32: the direct-residual claim")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    from abl_common import Report, header, try_plots     # noqa: E402

    rep = Report("ext_32_residual_mode.txt")
    for line in header("ext_32 - the '5x better' direct-residual claim, controlled",
                       f"layer_sizes={LAYER_SIZES} output_dims={OUTPUT_DIMS} "
                       f"epochs={args.epochs} seeds={args.seeds} modes={list(MODES)}"):
        rep(line)

    rep("")
    rep("  THE CLAIM: forming the layer-2/3 residual target from each layer's RAW scalar")
    rep("  output is ~5x better than from the proto-weighted prediction.  README carries")
    rep("  it as '+0.064 (direct residual) vs +0.013 (Proto)' on a 2D nonlinear")
    rep("  regression -- so the metric is the DEPTH GAIN, and 0.064/0.013 = 4.9.")
    rep("")
    rep("  AND THE BASE SAYS IT HAS NOT BEEN REPRODUCED: 'a short check gave the right")
    rep("  direction but nowhere near that magnitude', which is why the shipped default")
    rep("  is still 'proto'.")
    rep("")
    rep("  WHY THE ORIGINAL NUMBERS COULD NOT SETTLE IT -- they come from TWO PROGRAMS:")
    rep("")
    rep("    reg_01_proto_baseline.py  (+0.013)  |  reg_02_direct_residual.py  (+0.064)")
    rep("    ------------------------------------ | ------------------------------------")
    rep("    residual mode: proto-weighted       |  direct")
    rep("    kernel: Cauchy, anisotropic h,      |  pure Gaussian")
    rep("            multi-scale k                |")
    rep("    HiMALND, dual A/B network           |  SEPARATE single net, no A/B split")
    rep("    trainer MONKEY-PATCHED:             |  its own hand-written loop; does not")
    rep("      train_one_layer replaced          |  import HiMALND at all")
    rep("      _forward_layer replaced           |")
    rep("")
    rep("  **At least four things change between the two numbers, and the 5x is attributed")
    rep("  to one of them.**  reg_01 does not run the shipped trainer; reg_02 does not run")
    rep("  it either, or anything derived from it.")
    rep("")
    rep("  THIS SCRIPT: the SHIPPED HiMALND, nothing monkey-patched, reg_01's own")
    rep("  configuration, and residual_mode as the ONLY difference between the two arms.")

    data = make_data()
    t0 = time.time()
    runs = {m: [] for m in MODES}
    for m in MODES:
        for s in range(args.seeds):
            r = run_once(m, data, s, epochs=args.epochs)
            runs[m].append(r)
            print(f"  [{m}] seed {s}: depth curve "
                  f"{['%+.4f' % v for v in r['curve']]} gain {r['gain']:+.4f} "
                  f"({r['secs']:.0f}s)", flush=True)

    rep("\n" + "=" * 100)
    rep("  P1  the depth curve under each mode, mean over seeds")
    rep("=" * 100)
    rep("")
    rep(f"  {'mode':<9} {'R2 (L1 only)':>13} {'R2 (L1+L2)':>12} {'R2 (L1+L2+L3)':>14} "
        f"{'DEPTH GAIN':>12} {'HiMALND.predict':>16}")
    rep("  " + "-" * 82)
    agg = {}
    for m in MODES:
        c = np.array([r["curve"] for r in runs[m]])
        g = np.array([r["gain"] for r in runs[m]])
        sh = np.array([r["shipped"] for r in runs[m]])
        agg[m] = dict(l1=float(c[:, 0].mean()), l2=float(c[:, 1].mean()),
                      l3=float(c[:, 2].mean()), gain=float(g.mean()),
                      gain_sd=float(g.std(ddof=1)) if len(g) > 1 else float("nan"),
                      shipped=float(sh.mean()))
        rep(f"  {m:<9} {agg[m]['l1']:>+13.4f} {agg[m]['l2']:>+12.4f} "
            f"{agg[m]['l3']:>+14.4f} {agg[m]['gain']:>+12.4f} "
            f"{agg[m]['shipped']:>+16.4f}")
    rep("")
    # THE BUILT-IN CONTROL.  residual_mode branches only for layers 2+, so layer 1 is
    # trained identically under both modes and its R^2 MUST agree seed by seed.  If it
    # does not, the two arms saw different initialisations and nothing below is a
    # controlled comparison.  An earlier version of this script had exactly that defect
    # and reported an L1 gap of +0.117 between the arms.
    same = all(abs(runs["proto"][s]["curve"][0] - runs["direct"][s]["curve"][0]) < 1e-9
               for s in range(args.seeds))
    l1p = float(np.mean([r["curve"][0] for r in runs["proto"]]))
    l1d = float(np.mean([r["curve"][0] for r in runs["direct"]]))
    rep(f"  CONTROL: layer 1 is trained identically under both modes, so its R^2 must")
    rep(f"  match seed by seed.  proto {l1p:+.4f} vs direct {l1d:+.4f}, "
        f"identical: {'YES' if same else 'NO -- THE COMPARISON IS INVALID'}")
    rep("")
    gp, gd = agg["proto"]["gain"], agg["direct"]["gain"]
    rep(f"  depth gain: proto {gp:+.4f} (sd {agg['proto']['gain_sd']:.4f}), "
        f"direct {gd:+.4f} (sd {agg['direct']['gain_sd']:.4f})")
    rep(f"  direct - proto = {gd - gp:+.4f}")
    rep("  the claim's published pair was proto +0.013 / direct +0.064, a ratio of 4.9")
    rep(f"  this run's ratio: "
        + (f"{gd / gp:.2f}x" if abs(gp) > 1e-9 else "undefined (proto gain is ~0)"))

    # ---- which inference path, and does it matter ------------------------
    rep("\n" + "=" * 100)
    rep("  P2  the two inference paths DISAGREE, and the base already knew")
    rep("=" * 100)
    rep("")
    rep("  reg_01's own source comment where it builds the depth curve reads:")
    rep("      'Proto-weighted vs direct-h (L1 only -- wire-direct accumulation broken")
    rep("       in old path)'")
    rep("  'the old path' is `HiMALND.predict`, which walks the layers wire-direct and")
    rep("  returns the LAST layer's output with NO accumulation -- while the training")
    rep("  design assumes the protos are summed (training/himal_nd.py:889: 'At predict")
    rep("  time these are summed -> recovers original y').")
    rep("")
    rep(f"  {'mode':<9} {'proto-summed (the claim)':>26} {'HiMALND.predict':>18} {'gap':>10}")
    rep("  " + "-" * 68)
    for m in MODES:
        sh = agg[m]["shipped"]
        rep(f"  {m:<9} {agg[m]['l3']:>+26.4f} "
            + (f"{sh:>+18.4f} {agg[m]['l3'] - sh:>+10.4f}" if np.isfinite(sh)
               else f"{'NOT SCORABLE':>18} {'--':>10}"))
    rep("")
    note = runs[MODES[0]][0].get("ship_note", "")
    rep(f"  HiMALND.predict returned {note}")
    if not np.isfinite(agg['proto']['shipped']):
        rep("")
        rep("  **AND THAT IS WORSE THAN A DISAGREEMENT.**  Under reg_01's OWN published")
        rep("  configuration -- output_dims ending at 2 -- the shipped inference path emits")
        rep("  two columns against a scalar target, so it cannot be scored at all.  So the")
        rep("  published depth numbers were necessarily computed through the hand-written")
        rep("  proto sum, and the trainer's own `predict` was never a candidate.")
    rep("")
    rep("  **So there are TWO live inference paths in the shipped trainer and they return")
    rep("  different things.**  This script reports both and claims neither; which one is")
    rep("  correct is a question about the base version's design, and it is a separate one")
    rep("  from the 5x.")

    # ---- verdict ---------------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  VERDICT: does the 5x reproduce under control?")
    rep("=" * 100)
    rep("")
    if gd > gp and gp > 0:
        ratio = gd / gp
        rep(f"  DIRECTION CONFIRMED: direct ({gd:+.4f}) beats proto ({gp:+.4f}) at the")
        rep(f"  same depth and the same everything else, ratio {ratio:.2f}x against the")
        rep("  published 4.9x.")
        if ratio >= 3.0:
            rep("")
            rep("  AND THE MAGNITUDE IS IN THE PUBLISHED BALLPARK, so the claim survives a")
            rep("  controlled test -- it was measured through two uncontrolled programs, and")
            rep("  it holds anyway.")
        else:
            rep("")
            rep(f"  **BUT NOT AT THE PUBLISHED MAGNITUDE**: {ratio:.2f}x against 4.9x.  So the")
            rep("  base's own note ('right direction, nowhere near that magnitude') is what a")
            rep("  controlled run reproduces, and the published 5x is a property of the two")
            rep("  programs rather than of the residual mode.")
    elif gd > gp:
        rep("  the direct gain is larger, but the proto gain is ~0 or negative, so the")
        rep("  RATIO is not a usable statistic here.  Direction only.")
    else:
        rep("  NOT CONFIRMED: with everything else held fixed, the direct residual does not")
        rep("  produce the larger depth gain.  **That is a controlled refutation of the")
        rep("  claim as stated**, and it means the published pair measured the difference")
        rep("  between two programs rather than between two residual schemes.")
    rep("")
    rep("  THE HONEST LIMIT: this is one task, one architecture, one epoch count and the")
    rep("  seed count above.  The published pair used a different kernel family and a")
    rep("  different network topology, so a 5x could still exist THERE -- what is")
    rep("  refuted is that the residual mode ALONE produces it.")
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
        for m in MODES:
            c = np.array([r["curve"] for r in runs[m]])
            ax[0].errorbar([1, 2, 3], c.mean(axis=0), yerr=c.std(axis=0, ddof=1),
                           marker="o", capsize=4, label=f"residual_mode={m}")
        ax[0].set_xticks([1, 2, 3])
        ax[0].set_xticklabels(["L1", "L1+L2", "L1+L2+L3"])
        ax[0].set_xlabel("depth")
        ax[0].set_ylabel("R^2 (proto-summed, the claim's metric)")
        ax[0].set_title("the depth curve, controlled")
        ax[0].grid(True, alpha=0.25)
        ax[0].legend(fontsize=8)
        x = np.arange(len(MODES))
        ax[1].bar(x - 0.2, [agg[m]["gain"] for m in MODES], 0.4, label="depth gain")
        ax[1].bar(x + 0.2, [0.013, 0.064][:len(MODES)], 0.4,
                  label="published pair (uncontrolled)")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([f"residual_mode={m}" for m in MODES], fontsize=8)
        ax[1].set_ylabel("R^2 gain from depth")
        ax[1].set_title("controlled vs published")
        ax[1].grid(True, axis="y", alpha=0.25)
        ax[1].legend(fontsize=7)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_32_residual_mode_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {p}")

    ok = (all(len(runs[m]) == args.seeds for m in MODES)
          and all(np.isfinite(r["curve"]).all() for m in MODES for r in runs[m])
          # the control must HOLD, or the verdict is not admissible
          and all(abs(runs["proto"][s]["curve"][0] - runs["direct"][s]["curve"][0]) < 1e-9
                  for s in range(args.seeds)))
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
