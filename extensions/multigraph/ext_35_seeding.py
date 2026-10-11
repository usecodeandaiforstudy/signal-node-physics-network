"""
ext_35 -- does the trainer's new `seed` actually make it reproducible?
=====================================================================
Project: singal-node-physics-network, extension `multigraph`

WHY THIS EXISTS
---------------
`BASE_CORRECTIONS.md` entry 0 recorded that `HiMALND` took **no `seed` parameter and never
seeded the global RNG**, so two calls with identical arguments produced **different
networks** depending on what had run before -- and therefore every number published from
that class was one draw from an unseeded process.

The fix adds `seed` to `__init__` and calls `np.random.seed(self.seed)` at the top of
`train`.  That is sufficient **because the class has exactly two randomness sources and
both are reached from `train`**:

    `_init_layer`   -- centres, W_gauss, W_act          (three `np.random.uniform` calls)
    `_spsa_step`    -- the SPSA perturbation vector      (one `np.random.choice`)

**This script is the test entry 0 named: two runs, identical arguments, identical result.**
It checks three things, and the second and third matter as much as the first:

  P1  IDENTICAL ARGUMENTS -> IDENTICAL RESULT.  Bit-identical, on the trained layers
      themselves and on the depth curve, not merely "close".
  P2  A DIFFERENT SEED -> A DIFFERENT RESULT.  A "seed" that changes nothing would pass
      P1 while being useless, so the seed has to be shown to DO something.
  P3  THE DRAW IS NOT THE DOMINANT EFFECT.  The spread across seeds is reported next to
      the quantity the published numbers quote, so a reader can see how much of a
      published number was the draw -- which is the question entry 0 leaves behind.

SCOPE: the task and architecture `reg_01`/`reg_02` use, at a REDUCED epoch count for P1
and P2 (determinism does not need convergence, and the test should be cheap enough to run
in CI), and at the published configuration for P3.

Output: outputs/ext_35_seeding.txt
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

# reg_01's own configuration, so P3 speaks to the published pair
LAYER_SIZES = (20, 15, 10)
OUTPUT_DIMS = (4, 3, 2)
EPOCHS, LR, EPS = 80, 0.02, 0.01
GAMMA = 0.04
QUICK_EPOCHS = 3


def make_data(n=400, noise=0.15, seed=42):
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
    return 1.0 - float(np.mean((pred - true) ** 2)) / max(float(np.var(true)), 1e-12)


def depth_curve(trainer, lA, lB, X, y, n_layers=3):
    """The metric the published pair quotes: the per-depth proto-summed R^2."""
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
    return float(r2(((acc_A + acc_B) / 2.0).ravel(), y))


def run(data, seed, epochs, mode="proto"):
    X_tr, y_tr, X_te, y_te = data
    t = HiMALND(layer_sizes=LAYER_SIZES, output_dims=OUTPUT_DIMS, input_dim=2,
                init_range=(-3.0, 3.0), alignment_weight=GAMMA, n_protos=10,
                temperature=0.15, task="regression", use_spsa=False,
                residual_mode=mode, seed=seed)
    lA, lB, _s = t.train(X_tr, y_tr, split_method="median_x",
                         epochs_per_layer=epochs, lr=LR, eps=EPS, verbose=False)
    # the raw parameter vectors, so P1 can compare BIT-IDENTICALLY
    vec = np.concatenate([l.get_params_vector().ravel() for l in (lA + lB)])
    curve = [depth_curve(t, lA, lB, X_te, y_te, k) for k in (1, 2, 3)]
    return vec, curve


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_35: is the trainer reproducible now?")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--quick-epochs", type=int, default=QUICK_EPOCHS)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    from abl_common import Report, header     # noqa: E402

    rep = Report("ext_35_seeding.txt")
    for line in header("ext_35 - does `seed` make HiMALND reproducible?",
                       f"layer_sizes={LAYER_SIZES} quick_epochs={args.quick_epochs} "
                       f"seeds={args.seeds}"):
        rep(line)

    rep("")
    rep("  `BASE_CORRECTIONS.md` entry 0: HiMALND took no `seed` and never seeded the")
    rep("  global RNG, so two calls with identical arguments gave different networks.")
    rep("  The fix seeds once at the top of `train`, which is sufficient because the")
    rep("  class has exactly two randomness sources and both hang off `train`:")
    rep("      `_init_layer`  -- centres, W_gauss, W_act   (3x np.random.uniform)")
    rep("      `_spsa_step`   -- the SPSA perturbation      (1x np.random.choice)")

    data = make_data()
    t0 = time.time()

    # ---- P1: identical arguments -> identical result --------------------
    rep("\n" + "=" * 100)
    rep("  P1  two runs, identical arguments, identical result")
    rep("=" * 100)
    rep("")
    v1, c1 = run(data, 0, args.quick_epochs)
    v2, c2 = run(data, 0, args.quick_epochs)
    same_vec = bool(np.array_equal(v1, v2))
    same_curve = all(abs(a - b) < 1e-15 for a, b in zip(c1, c2))
    rep(f"  parameters returned  : {v1.size} floats")
    rep(f"  bit-identical        : **{same_vec}**")
    rep(f"  depth curve identical : **{same_curve}**   "
        f"{['%+.6f' % x for x in c1]}")
    rep("")
    rep("  (Before the fix this was False, and how different the two runs were depended")
    rep("  on what had run earlier in the process -- which is why the old numbers were")
    rep("  single draws rather than measurements.)")

    # ---- P2: a different seed must DO something -------------------------
    rep("\n" + "=" * 100)
    rep("  P2  a different seed must change the result, or the parameter is useless")
    rep("=" * 100)
    rep("")
    v3, c3 = run(data, 1, args.quick_epochs)
    diff = bool(not np.array_equal(v1, v3))
    rep(f"  seed 0 vs seed 1     : parameters differ = **{diff}**")
    rep(f"  depth curve seed 0   : {['%+.6f' % x for x in c1]}")
    rep(f"  depth curve seed 1   : {['%+.6f' % x for x in c3]}")
    rep("")
    rep("  A seed that changed nothing would pass P1 while doing no work, so this is the")
    rep("  check that the parameter is load-bearing.")

    # ---- P3: how much of a published number was the draw? ---------------
    rep("\n" + "=" * 100)
    rep("  P3  at the PUBLISHED configuration: how much of the depth gain is the draw?")
    rep("=" * 100)
    rep("")
    rep("  This is the question entry 0 leaves behind.  `reg_01`'s +0.013 and `reg_02`'s")
    rep("  +0.064 were single unseeded draws, so the honest report of any re-run is a")
    rep("  spread and not a point.")
    rep("")
    rep("  **THE SPREAD IS ALREADY MEASURED, in `ext_32`** -- the same configuration,")
    rep("  seeds 0-2, both modes, and it reports `proto +0.0069 (sd 0.0045)` against")
    rep("  `direct -0.0284 (sd 0.0152)`.  So this part runs ONE confirmation per mode")
    rep("  rather than repeating a 50-minute sweep: its job is to show that the number")
    rep("  is now reproducible AT A STATED SEED, which is what entry 0 asked for.")
    rep("")
    rep(f"  {'mode':<9} {'seed':>5} {'R2 (L1)':>10} {'R2 (L1+L2)':>12} "
        f"{'R2 (L1+L2+L3)':>14} {'DEPTH GAIN':>12}")
    rep("  " + "-" * 66)
    p3 = {}
    for mode in ("proto", "direct"):
        for s in range(args.seeds):
            _v, c = run(data, s, EPOCHS, mode=mode)
            p3[(mode, s)] = c
            rep(f"  {mode:<9} {s:>5} {c[0]:>+10.4f} {c[1]:>+12.4f} {c[2]:>+14.4f} "
                f"{c[2] - c[0]:>+12.4f}")
            print(f"  [P3] {mode} seed {s}: gain {c[2]-c[0]:+.4f}", flush=True)
        g = [p3[(mode, s)][2] - p3[(mode, s)][0] for s in range(args.seeds)]
        rep(f"  {mode:<9} {'mean':>5} {'':>10} {'':>12} {'':>14} {np.mean(g):>+12.4f}"
            f"   (sd {np.std(g, ddof=1) if len(g) > 1 else float('nan'):.4f}, "
            f"n={len(g)})")
        rep("")

    # ---- verdict --------------------------------------------------------
    rep("=" * 100)
    rep("  VERDICT")
    rep("=" * 100)
    rep("")
    if same_vec and same_curve and diff:
        rep("  **THE FIX WORKS: identical arguments give a bit-identical network, and a")
        rep("  different seed gives a different one.**  Entry 0's test is met on both")
        rep("  sides, so numbers from this class can now be reported with the seed that")
        rep("  produced them.")
    elif not diff:
        rep("  **P1 passes but P2 FAILS: the seed changes nothing**, so the parameter is")
        rep("  decorative and the reproducibility would be an accident of determinism")
        rep("  rather than of the seed.  Must be fixed before any number is re-run.")
    else:
        rep("  **P1 FAILS: the trainer is still not reproducible.**  Two runs with")
        rep("  identical arguments differ, so a random source is being missed -- the")
        rep("  candidates are any draw outside `_init_layer` and `_spsa_step`.")
    rep("")
    rep("  WHAT THIS DOES NOT DO: it does not re-run every script that quotes a number")
    rep("  from this class.  It establishes that they CAN be re-run reproducibly, and")
    rep("  P3 reports what one of them should report -- a spread, not a point.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

    path = rep.save()
    print(f"\nReport saved: {os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")}")

    ok = bool(same_vec and same_curve and diff)
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
