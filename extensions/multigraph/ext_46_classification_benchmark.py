"""
ext_46 -- `HiMALND`'s CLASSIFICATION mode, benchmarked against the project's own null
====================================================================================
Project: singal-node-physics-network, extension `multigraph`

THE ITEM, AND A CLAIM OF MINE THAT HAD TO BE CORRECTED FIRST
------------------------------------------------------------
The worklist said the classification path "is exactly what the CIFAR line uses".  **That is
false and the correction changes what this script can claim**: `grep`-ing the CIFAR family
for `himal` returns NOTHING -- that line is `RBFLayerND` trained by PyTorch autograd and
exported to a physical layer, a DIFFERENT architecture.  So nothing measured here says
anything about the project's image demonstration.

What is true: `HiMALND`'s classification mode is used by `nd_01`, `nd_03`, `reg_03`, `reg_05`
and `legacy_suite`, and `nd_01` DOES report an accuracy -- against a **v1 ResidualRBFNetwork** and
an MLP, i.e. against **sibling architectures**.  **What it has never been compared against is
this project's OWN standard**: the operator's placement, the closed-form solve, and the
random-geometry null -- the three things that, applied to regression, showed the deep path to
be dominated (12.30).

THE COSTS ARE THE SAME TWO UNITS AND ARE NEVER ADDED
----------------------------------------------------
  deep       SPSA "2 * spsa_avg forward passes" per epoch per network -> analytic
  shallow    `fit_operator` returns `n_evals`, and the closed-form solve costs
             `n_train * N` READINGS (`readout_calibration_cost`) -- a second unit.

WHAT IS MEASURED
----------------
  P1  accuracy for the deep classification arm, the operator arm, and the null, on the two
      classification tasks the repository can build without downloading anything.
  P2  the operator arm swept across its own cost, so the comparison is read at equal
      forwards rather than at equal settings.
  P3  the verdict, with the scope stated first because it is the part that is easiest to
      overclaim.
  P4  the limits.

SCOPE: `HiMALND(task="classification")` at the repository's own 2-D configuration, the
base's `moons` task and `nd_01`'s own 4-D moons, 3 seeds, toy.  **This is NOT the CIFAR
path.**  Nothing is edited in the base.

Output: outputs/ext_46_classification_benchmark.txt + _plot.png
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
    Report, header, make_regression_data, set_ideal, try_plots,
)
from extensions.multigraph.trainer import (                 # noqa: E402
    features, fit_operator, solve_strengths,
)
from training.himal_nd import HiMALND                        # noqa: E402

NODE_GRID = (10, 20, 40, 80)
ROUND_GRID = (2, 4)


def accuracy(y_true, scores):
    """argmax over the vote matrix against the labels."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 1:
        return float(np.mean((s > 0) == (np.asarray(y_true).ravel() > 0)))
    return float(np.mean(np.argmax(s, axis=1) == np.asarray(y_true).ravel().astype(int)))


def repo_4d_moons(n=800, noise=0.12, seed=42):
    """`nd_01_baseline.generate_4d_moons`, reproduced: 2-D moons in 4-D, TWO classes."""
    from sklearn.datasets import make_moons
    X_2d, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    rng = np.random.RandomState(seed + 1)
    return (np.column_stack([X_2d, rng.normal(0, 0.3, (n, 2))]).astype(np.float32), y)


def deep_cost(epochs, layers=3, networks=2, spsa_avg=1):
    return 2 * int(spsa_avg) * int(epochs) * int(layers) * int(networks)


def task_data(name, seed):
    """Both tasks come back as (X_tr, y_tr, X_te, y_te) with 0/1 labels."""
    if name == "moons":
        X_tr, y_tr, X_te, y_te = make_regression_data("moons", seed=42 + seed)
        return X_tr, (y_tr > 0).astype(int), X_te, (y_te > 0).astype(int)
    X, y = repo_4d_moons(seed=42 + seed)
    n_te = len(y) // 5
    return X[:-n_te], y[:-n_te].astype(int), X[-n_te:], y[-n_te:].astype(int)


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_46: benchmark HiMALND classification")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--particles", type=int, default=20)
    ap.add_argument("--task", default="moons", choices=("moons", "moons4d"))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert
    # THE GLOBAL RNG TOO, at the repository's own `SEED = 0` convention.  The first fix
    # used `args.seed` -- WHICH DOES NOT EXIST: these scripts define `--seeds`, a COUNT
    # looped over with `range(args.seeds)`, and the substring match that checked for it
    # matched `args.seeds` too.  All 24 crashed with AttributeError until this line.
    # `features()` builds a REAL device layer and the device draws
    # from `np.random`, so without this the exact path moved between runs in the 5th to
    # 7th digit while everything else stayed put -- defect 58 in FINDINGS.md, which this
    # line closes.  `log_rms`'s healthy band is ~1e-6, i.e. AT the magnitude of that
    # movement, so the test that decides whether a response is a genuine single node was
    # sitting on the non-determinism.
    np.random.seed(0)

    set_ideal()
    rep = Report("ext_46_classification_benchmark.txt")
    for line in header("ext_46 - HiMALND classification, benchmarked",
                       f"task={args.task} epochs={args.epochs} seeds={args.seeds}"):
        rep(line)

    rep("")
    rep("  **SCOPE FIRST, BECAUSE IT IS THE EASIEST THING TO OVERCLAIM HERE.**  A worklist")
    rep("  note of mine said the classification path 'is exactly what the CIFAR line uses'.")
    rep("  **That is false.**  Searching the CIFAR family for `himal` returns NOTHING: that")
    rep("  line is `RBFLayerND` trained by PyTorch autograd and exported to a physical layer,")
    rep("  a DIFFERENT architecture.  **So nothing measured here says anything about the")
    rep("  project's image demonstration**, and the note is corrected rather than left to be")
    rep("  quoted.")
    rep("")
    rep("  What IS true: this mode is used by `nd_01`, `nd_03`, `reg_03`, `reg_05` and")
    rep("  `legacy_suite`, and `nd_01` does report an accuracy -- against a v1 ResidualRBFNetwork")
    rep("  and an MLP, i.e. against SIBLING ARCHITECTURES.  **What it has never been compared")
    rep("  against is this project's own standard**: the operator's placement, the closed-form")
    rep("  solve, and the random-geometry null -- the three things that, applied to")
    rep("  regression, showed the deep path to be dominated (12.30).")

    t0 = time.time()
    dc = deep_cost(args.epochs)
    rep("")
    rep(f"  the deep arm's cost, analytic from SPSA's documented '2 * spsa_avg forward")
    rep(f"  passes': 2 * {args.epochs} * 3 layers * 2 networks = **{dc:,} forwards**.")

    # ---- P1/P2 -----------------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P1/P2  accuracy, and the operator arm swept across its own cost")
    rep("=" * 100)
    rep("")
    rows = []
    for seed in range(args.seeds):
        X_tr, y_tr, X_te, y_te = task_data(args.task, seed)
        n_va = max(32, X_tr.shape[0] // 5)
        X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
        X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]

        # the deep classification arm
        mdl = HiMALND(layer_sizes=(args.particles,) * 3, input_dim=X_tr.shape[1],
                      task="classification", n_classes=2, seed=0, use_spsa=True)
        lA, lB, _d = mdl.train(X_tr, y_tr, epochs_per_layer=args.epochs,
                               lr=0.012, eps=1e-3, verbose=False)
        votes = np.asarray(mdl.predict(lA, lB, X_te))
        deep_acc = accuracy(y_te, votes[:len(y_te)])

        # the operator arm and the null, both CLOSED FORM: ridge on node features to +-1,
        # then the sign.  That is the project's own method applied to classification.
        tgt = np.where(y_fi > 0, 1.0, -1.0)
        shallow = {}
        for n_nodes in NODE_GRID:
            for rounds in ROUND_GRID:
                o = fit_operator(X_fi, y_fi.astype(float), X_va, y_va.astype(float),
                                 n_nodes=n_nodes, rounds=rounds, seed=seed)
                Z_tr = features(X_fi, o["centres"], o["widths"])
                Z_va = features(X_va, o["centres"], o["widths"])
                Z_te = features(X_te, o["centres"], o["widths"])
                w, _lr = solve_strengths(Z_tr, tgt, Z_va,
                                         np.where(y_va > 0, 1.0, -1.0))
                shallow[(n_nodes, rounds)] = (
                    accuracy(y_te, Z_te @ w), int(o.get("n_evals", 0)),
                    len(y_fi) * n_nodes)
        rng = np.random.default_rng(seed)
        cen = rng.uniform(X_fi.min(axis=0), X_fi.max(axis=0),
                          (args.particles, X_fi.shape[1]))
        wid = np.full(args.particles, float(np.mean(X_fi.max(axis=0) - X_fi.min(axis=0))
                                            / np.sqrt(args.particles)))
        Zt, Zv = features(X_fi, cen, wid), features(X_va, cen, wid)
        w, _lr = solve_strengths(Zt, tgt, Zv, np.where(y_va > 0, 1.0, -1.0))
        rows.append(dict(deep=deep_acc, shallow=shallow,
                         null=accuracy(y_te, features(X_te, cen, wid) @ w)))
        print(f"  [P2] seed {seed}: deep {deep_acc:.4f}  null {rows[-1]['null']:.4f}  "
              f"shallow best {max(v[0] for v in shallow.values()):.4f}", flush=True)

    d_mean = float(np.mean([r["deep"] for r in rows]))
    n_mean = float(np.mean([r["null"] for r in rows]))
    rep(f"  {'arm':<30} {'forwards':>10} {'calib readings':>16} {'accuracy':>10}")
    rep("  " + "-" * 70)
    rep(f"  {'deep (himal_nd, classification)':<30} {dc:>10,} {0:>16,} {d_mean:>10.4f}")
    rep(f"  {'RANDOM geometry + solve':<30} {0:>10,} {len(y_fi)*args.particles:>16,} "
        f"{n_mean:>10.4f}")
    rep("")
    sweep = {}
    for key in sorted({k for r in rows for k in r["shallow"]},
                      key=lambda k: rows[0]["shallow"][k][1]):
        v = float(np.mean([r["shallow"][key][0] for r in rows]))
        nv, cm = rows[0]["shallow"][key][1], rows[0]["shallow"][key][2]
        sweep[nv] = v
        mark = "   <-- the deep arm's budget" if abs(nv - dc) / max(dc, 1) < 0.3 else ""
        rep(f"  N={key[0]:<3} rounds={key[1]:<3} {nv:>14,} {cm:>16,} {v:>10.4f}{mark}")

    # ---- P3: verdict ------------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P3  VERDICT: does the classification path beat the project's own standard?")
    rep("=" * 100)
    rep("")
    at_or_under = [(nv, v) for nv, v in sweep.items() if nv <= dc]
    best = max(at_or_under, key=lambda t: t[1]) if at_or_under else (0, float("nan"))
    rep(f"  deep arm:                {d_mean:.4f} accuracy at {dc:,} forwards")
    rep(f"  RANDOM geometry + solve: {n_mean:.4f} accuracy at 0 forwards")
    rep(f"  operator at or under dc: {best[1]:.4f} accuracy at {best[0]:,} forwards")
    rep("")
    beats_null = d_mean > n_mean
    beats_op = d_mean >= best[1]
    if not beats_null:
        rep("  **THE DEEP CLASSIFICATION ARM DOES NOT BEAT THE NULL EITHER.**  Random geometry")
        rep("  plus a closed-form solve -- ZERO forwards, and the class labels used only as")
        rep("  +-1 regression targets -- is at least as accurate.  **That is the same result")
        rep("  12.30 found for regression, arriving through a completely different read-out**, so")
        rep("  it is no longer a property of that one prediction path.")
    elif not beats_op:
        rep("  **THE DEEP ARM BEATS THE NULL BUT NOT THE OPERATOR**, which is the reading the")
        rep("  regression case gave too: the placement is what earns its cost, and depth is not")
        rep("  adding to it on these tasks.")
    else:
        rep("  **THE DEEP CLASSIFICATION ARM BEATS BOTH**, unlike the regression path -- the")
        rep("  first case in this extension where a deep arm wins on the project's own standard.")
        rep("  **AND THE SCOPE NOTE AT THE TOP STILL APPLIES**: it is not the CIFAR path, so it")
        rep("  is a result about this architecture and not about the image demonstration.")
    rep("")
    rep("  **AND THE HONEST ASYMMETRY IN THE NULL.**  The closed-form arms read the classes as")
    rep("  +-1 and take a sign, which is a LEAST-SQUARES classifier -- not a margin method,")
    rep("  and not what the deep arm's prototype voting does.  **Each arm is scored the way it")
    rep("  is built to be scored**, which is the fair comparison and also the one that gives")
    rep("  the closed-form side a handicap on tasks where least squares is misspecified.")
    rep("")
    rep("  **THE LIMITS.**  Toy scale and two 2-class tasks the repository can build without")
    rep("  downloading anything; the deep arm ships with hyper-parameters inherited from a")
    rep("  pipeline test and never tuned; `n_evals` is the shallow arm's own accounting while")
    rep("  the deep arm's is analytic.  **And the CIFAR line is untouched and unmeasured")
    rep("  here**, which is the largest caveat this section carries.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

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
        xs = sorted(sweep)
        ax[0].semilogx([x for x in xs], [sweep[x] for x in xs], "o-",
                       label="operator + solve")
        ax[0].axhline(d_mean, color="#c44e52", ls="--", label=f"deep {d_mean:.3f}")
        ax[0].axhline(n_mean, color="gray", ls=":", label=f"random geometry {n_mean:.3f}")
        ax[0].axvline(dc, color="k", lw=0.8)
        ax[0].set_xlabel("forward evaluations")
        ax[0].set_ylabel("test accuracy")
        ax[0].set_title("the operator arm swept across its own cost")
        ax[0].grid(True, alpha=0.25, which="both")
        ax[0].legend(fontsize=7)
        x = np.arange(len(rows) + 2)
        ax[1].bar(x, [r["deep"] for r in rows] + [d_mean, n_mean])
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([f"s{i}" for i in range(len(rows))] + ["deep\nmean", "null"],
                              fontsize=8)
        ax[1].set_ylabel("test accuracy")
        ax[1].set_title("the deep classification arm, with its references")
        ax[1].grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_46_classification_benchmark_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

    ok = all(np.isfinite(r["deep"]) for r in rows)
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
