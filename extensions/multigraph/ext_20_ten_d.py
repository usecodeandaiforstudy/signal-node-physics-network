"""
ext_20 -- 10d: test it properly, or state plainly that it is out of reach
========================================================================
Project: singal-node-physics-network, extension `multigraph`

THE LAST SILENT CELL
--------------------
`ext_10` excluded both 10d cells because no arm cleared `abl_common.learning_ok`'s
floor -- a CAPACITY result, since 40 nodes over a 10-dimensional input cannot fit the
task.  Every other cell in this extension is either positive or negative; 10d is the
one place the results are **silent**, and silence is not evidence either way.  The item
was written as "test it properly, or drop it", so this is the test: raise the node
count until the cells discriminate, and see which way they go when they do.

WHAT WOULD MAKE IT INTERESTING EITHER WAY
-----------------------------------------
  * if the operator arm wins once the cell can discriminate, the dimension axis has no
    hole and `ext_10`'s exclusion was purely a capacity artefact;
  * if it LOSES at a node count where every arm learns, that is a real negative on a
    fifth dimension and belongs in the same list as `steps` and `noise`;
  * if no affordable node count discriminates, then "out of reach at this scale" is the
    honest answer and it should be stated rather than left as an open cell.

SCOPE: task `10d`, node counts in {40, 120, 360}, one layer, 3 seeds.  The arms are the
same three as everywhere -- OPERATOR, RANDOM geometry (the null), and the base version's
own FD trainer cost-matched to the operator's measured evaluations -- so the numbers are
comparable to `ext_10` and `ext_13`.

Output: outputs/ext_20_ten_d.txt + _plot.png
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
    Report, header, set_ideal, try_plots,
)
from extensions.multigraph.ext_10_generalization import run_cell   # noqa: E402

NODE_COUNTS = (40, 120, 360)
FLOOR = 0.05          # abl_common.learning_ok's own bar


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_20: 10d properly, or not at all")
    ap.add_argument("--task", default="10d")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    # The device layer draws from the GLOBAL np.random, so without this the realisation
    # differs every run and a borderline assertion can flip: ext_06's C2 margin was 133x
    # one run and 9.73x the next, against a 10x bar.  This is defect 58 in CONTRIBUTING.md,
    # and the fix is the same one the other thirty scripts already carry.
    np.random.seed(0)
    set_ideal()
    rep = Report("ext_20_ten_d.txt")
    for line in header("ext_20 - 10d: tested properly, at last",
                       f"task={args.task} nodes={NODE_COUNTS} seeds={args.seeds}"):
        rep(line)

    rep("")
    rep("  `ext_10` excluded both 10d cells because no arm cleared the learning floor --")
    rep("  a CAPACITY result at 40 nodes over a 10-dimensional input.  Silence is not")
    rep("  evidence either way, so the item was written as 'test it properly, or drop")
    rep("  it'.  This raises the node count until the cells discriminate and reports")
    rep("  which way they go when they do.")
    rep("")
    rep("  Same three arms as ext_10 and ext_13 -- OPERATOR, RANDOM geometry as the")
    rep("  null, and the base version's own FD trainer cost-matched to the operator's")
    rep("  MEASURED evaluation count -- so the numbers stay comparable across the")
    rep("  extension.")

    t0 = time.time()
    table = {}
    for n in NODE_COUNTS:
        rows = []
        for seed in range(args.seeds):
            d = run_cell(args.task, [n], seed, args.rounds)
            rows.append(d)
            print(f"  [ran ] 10d N={n:<4} seed {seed}  OP {d['op']:+.4f}  "
                  f"RND {d['rnd']:+.4f}  FD {d['fd']:+.4f}  "
                  f"evals {d.get('op_evals', float('nan')):.0f}", flush=True)
        table[n] = rows

    rep("\n" + "=" * 88)
    rep("  Test R^2 on 10d, by node count (mean over seeds)")
    rep("=" * 88)
    rep(f"  {'nodes':>6} {'OPERATOR':>11} {'RANDOM':>10} {'FD':>10} "
        f"{'op-random':>11} {'best arm':>10} {'discriminates':>14}")
    rep("  " + "-" * 76)
    summary = {}
    for n in NODE_COUNTS:
        rows = table[n]
        op = np.array([r["op"] for r in rows])
        rn = np.array([r["rnd"] for r in rows])
        fd = np.array([r["fd"] for r in rows])
        best = float(max(op.mean(), rn.mean(), fd.mean()))
        disc = best > FLOOR
        summary[n] = dict(op=op, rnd=rn, fd=fd, best=best, disc=disc)
        rep(f"  {n:>6} {op.mean():>+11.4f} {rn.mean():>+10.4f} {fd.mean():>+10.4f} "
            f"{(op - rn).mean():>+11.4f} {best:>+10.4f} "
            f"{'yes' if disc else 'NO -- silent':>14}")
    rep("")
    rep(f"  the learning floor is {FLOOR}; a cell below it cannot say which placement")
    rep("  rule is better, only that none of them fits.")

    # -- verdict ------------------------------------------------------------
    rep("\n" + "=" * 88)
    rep("  VERDICT: is 10d a hole in the dimension axis, or out of reach?")
    rep("=" * 88)
    rep("")
    live = [n for n in NODE_COUNTS if summary[n]["disc"]]
    if not live:
        rep(f"  OUT OF REACH AT THIS SCALE, and now stated rather than left open: at")
        rep(f"  {NODE_COUNTS[-1]} nodes -- {NODE_COUNTS[-1]//10} per input dimension -- the")
        rep("  best arm still does not clear the floor.  The dimension axis therefore has")
        rep("  a hole above 4d, and the honest statement is that nothing here is")
        rep("  evidence about high-dimensional inputs, in either direction.")
    else:
        n = live[0]
        z = summary[n]
        d_r = float((z["op"] - z["rnd"]).mean())
        tol = max(0.02, 2.0 * max(float(z["op"].std()), float(z["rnd"].std()))
                  / np.sqrt(len(z["op"])))
        rep(f"  THE CELL DISCRIMINATES at N={n}: best arm {z['best']:+.4f}, above the")
        rep(f"  {FLOOR} floor.  So `ext_10`'s exclusion was a CAPACITY artefact, not a")
        rep("  property of the dimension.")
        rep("")
        rep(f"  and at that node count the operator arm is "
            f"{z['op'].mean():+.4f} against random {z['rnd'].mean():+.4f} "
            f"({d_r:+.4f}, tol {tol:.3f})")
        if d_r > tol:
            rep("  -> the operators WIN here too, so the dimension axis has no hole once")
            rep("     the cell can discriminate.")
        else:
            rep("  -> the operators do NOT resolve a win here.  On a cell that CAN")
            rep("     discriminate, that is a real negative on a fifth dimension and it")
            rep("     belongs in the same list as `steps` and `noise` -- not filed as")
            rep("     'silent'.")
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
        x = np.arange(len(NODE_COUNTS))
        ax[0].bar(x - 0.27, [summary[n]["op"].mean() for n in NODE_COUNTS], 0.27,
                  label="OPERATOR", color="#2b7bba")
        ax[0].bar(x, [summary[n]["rnd"].mean() for n in NODE_COUNTS], 0.27,
                  label="RANDOM", color="#999999")
        ax[0].bar(x + 0.27, [summary[n]["fd"].mean() for n in NODE_COUNTS], 0.27,
                  label="FD", color="#c44e52")
        ax[0].axhline(FLOOR, color="k", ls=":", lw=1.0, label=f"floor {FLOOR}")
        ax[0].axhline(0.0, color="k", lw=0.8)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels([f"N={n}" for n in NODE_COUNTS])
        ax[0].set_ylabel("test R^2 on 10d")
        ax[0].set_title("does the cell ever discriminate?")
        ax[0].grid(True, axis="y", alpha=0.25)
        ax[0].legend(fontsize=8)
        ax[1].plot(NODE_COUNTS, [summary[n]["op"].mean() for n in NODE_COUNTS],
                   "o-", label="OPERATOR")
        ax[1].plot(NODE_COUNTS, [summary[n]["rnd"].mean() for n in NODE_COUNTS],
                   "s-", label="RANDOM")
        ax[1].axhline(FLOOR, color="k", ls=":", lw=1.0, label=f"floor {FLOOR}")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("nodes")
        ax[1].set_ylabel("test R^2")
        ax[1].set_title("capacity vs dimension")
        ax[1].grid(True, alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_20_ten_d_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")}")

    # The assert encodes what the measurement supports: the sweep RAN and produced a
    # verdict on every node count.  It cannot encode "10d wins" or "10d is out of
    # reach", because either is a result.
    if do_assert and not all(len(table[n]) == args.seeds for n in NODE_COUNTS):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
