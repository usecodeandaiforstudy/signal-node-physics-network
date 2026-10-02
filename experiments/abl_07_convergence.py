"""
Ablation G -- Three loose ends on the calibration rule
======================================================
Project: singal-node-physics-network

Three questions that FINDINGS.md section 11 leaves open.  They share a script
because they share the harness and each is cheap; they are independent findings.

Part A -- EPOCHS x BUDGET: is "the readout wins" just slower convergence?
    Section 5 measures the capacity split at a fixed epoch count, and section 11
    records that the geometry arm roughly DOUBLES between 40 and 80 epochs.  If
    the gap simply closes with more training then the result is about convergence
    rate, not about where capacity lives -- a materially different claim.  The
    grid separates the two: at each budget, does the gap shrink toward zero as
    epochs grow (convergence) or plateau at a positive value (asymptote)?

Part B -- CALIBRATION SCHEDULE: how often does it need to be attempted?
    The gate accepts only 0-4 of 80 attempts, so "recalibrate every epoch" is
    mostly wasted effort.  Sweep the attempt period and check whether the outcome
    moves.  This is a cost question, not a correctness one.

Part C -- GATE SAMPLE SIZE: how large must the validation slice be?
    The gate decides on a held-out slice (1/4 of the training data = 64 samples).
    A small slice has a noisy score, so a WORSE calibration can win by luck.  This
    is the real remaining hole in "the calibrated arm is never worse than not
    calibrating", and unlike distribution shift it is testable: shrink the slice
    and see whether harmful calibrations start getting through on the task known
    to be pathological (10d).  TWO designed counterexamples were tried before this
    one and both failed -- `shift` (the calibration won out of distribution) and
    `noise` (the gate refused to fit pure noise) -- so this checks the gate's
    statistical power rather than guessing at another task.

Output: outputs/abl_07_convergence.txt + _plot.png
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abl_common import (                                   # noqa: E402
    ARCHS, OUT_DIR, Report, build_model, effective_budget, header, r2,
    make_regression_data, merge_curves, selftest, set_diffusion_defaults,
    set_ideal, task_description, train_model, try_plots,
)

EPOCH_GRID = (20, 40, 80, 160)
BUDGET_GRID = (300, 600, 1200)
SCHEDULE_GRID = (1, 5, 10, 10 ** 9)      # 10**9 == calibrate once, at epoch 0
HOLDOUT_GRID = (4, 8, 16)                # 1/4, 1/8, 1/16 of 256 -> 64, 32, 16 samples


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ablation G: convergence, schedule, gate power")
    p.add_argument("--part", default="abc", help="which parts to run: any of a,b,c")
    p.add_argument("--arch", default="F2", choices=list(ARCHS))
    p.add_argument("--arch-c", default="10d_arch_placeholder", help=argparse.SUPPRESS)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--substrate", default="diffusion")
    p.add_argument("--probe-batch", type=int, default=96)
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


def train(arch, arm, X, Y, Xte, Yte, *, epochs, budget, seed, args, task="4d",
          calib_every=1, guard_holdout=4, substrate=None):
    model = build_model(arch, X, seed=seed, substrate=substrate or args.substrate)
    hist = train_model(
        model, X, Y, Xte, Yte, arm=arm, epochs=epochs, budget=budget, seed=seed,
        probe_batch=args.probe_batch, substrate=substrate or args.substrate,
        closed_form_every=calib_every, closed_form_guard_holdout=guard_holdout,
    )
    return float(r2(Yte, model.forward(Xte))), hist


def main():
    global EPOCH_GRID, BUDGET_GRID, SCHEDULE_GRID, HOLDOUT_GRID
    args = parse_args()
    if args.quick:
        if args.seeds == 2:
            args.seeds = 1
        # --quick must mean TINY, not just fewer seeds: part A is a 3x4 grid whose
        # largest single cell is 160 epochs at budget 1200, which kept a "quick"
        # run near 25 minutes and made it useless as a CI step.
        EPOCH_GRID = (20, 40)
        BUDGET_GRID = (300,)
        SCHEDULE_GRID = (1, 10 ** 9)
        HOLDOUT_GRID = (4,)
    parts = set(str(args.part).lower())
    set_diffusion_defaults(level="none")
    set_ideal()

    rep = Report("abl_07_convergence.txt")
    for line in header("Ablation G - convergence, schedule, gate sample size",
                       f"parts={sorted(parts)} arch={args.arch} "
                       f"seeds={args.seeds} substrate={args.substrate}"):
        rep(line)
    rep("\n--- self-test ---")
    selftest(verbose=True)

    X4, Y4, X4te, Y4te = make_regression_data("4d")
    t0 = time.time()

    # ---------------- Part A: epochs x budget ----------------
    if "a" in parts:
        rep("\n" + "=" * 78)
        rep("  Part A: EPOCHS x BUDGET -- convergence gap or asymptotic gap?")
        rep("=" * 78)
        rep(f"  arch {args.arch}, arms geometry vs readout, 1 seed per cell so the")
        rep("  grid fits in the compute budget; the question is the SHAPE, not the")
        rep("  precision of any single cell.")
        rep("")
        rep(f"  {'budget':>7} {'epochs':>7} {'geometry':>10} {'readout':>10} "
            f"{'gap':>9} {'geo fwd':>10}")
        rep("  " + "-" * 60)
        grid = {}
        for b in BUDGET_GRID:
            for ep in EPOCH_GRID:
                g, hg = train(args.arch, "geometry", X4, Y4, X4te, Y4te,
                              epochs=ep, budget=b, seed=0, args=args)
                w, hw = train(args.arch, "weights", X4, Y4, X4te, Y4te,
                              epochs=ep, budget=b, seed=0, args=args)
                grid[(b, ep)] = (g, w)
                rep(f"  {b:>7d} {ep:>7d} {g:>+10.4f} {w:>+10.4f} {w-g:>+9.4f} "
                    f"{hg['fwd']:>10d}")
            rep("")
        rep("  Reading it: if the gap shrinks toward 0 as epochs grow at FIXED")
        rep("  budget, the result is about convergence rate.  If it plateaus at a")
        rep("  positive value, the readout genuinely carries more of the fit.")
        rep("")
        rep(f"  {'budget':>7} {'gap@20':>9} {'gap@40':>9} {'gap@80':>9} "
            f"{'gap@160':>9} {'trend':>12}")
        rep("  " + "-" * 62)
        for b in BUDGET_GRID:
            gaps = [grid[(b, ep)][1] - grid[(b, ep)][0] for ep in EPOCH_GRID]
            # "closing" = the gap at the last epoch is under half the first
            trend = ("closing" if gaps[-1] < 0.5 * gaps[0]
                     else "asymptotic" if gaps[-1] > 0.8 * gaps[0] else "partial")
            rep(f"  {b:>7d} " + " ".join(f"{g:>+9.4f}" for g in gaps)
                + f" {trend:>12}")

    # ---------------- Part B: calibration schedule ----------------
    if "b" in parts:
        rep("\n" + "=" * 78)
        rep("  Part B: CALIBRATION SCHEDULE -- how often must it be attempted?")
        rep("=" * 78)
        bd, _sz = effective_budget(args.arch, X4, ["geometry", "weights", "full"],
                                   None, substrate=args.substrate)
        budget = int(bd["full"])
        rep(f"  arch {args.arch}, arm hybrid, budget {budget} probes/epoch, "
            f"{args.seeds} seeds")
        rep("")
        rep(f"  {'attempt every':>14} {'R2':>10} {'+-std':>8} {'attempts':>9} "
            f"{'accepted':>9} {'fwd':>9}")
        rep("  " + "-" * 64)
        for ce in SCHEDULE_GRID:
            vals, acc, att, fwd = [], 0, 0, 0
            for s in range(args.seeds):
                r, h = train(args.arch, "hybrid", X4, Y4, X4te, Y4te, epochs=40,
                             budget=budget, seed=s, args=args, calib_every=ce)
                vals.append(r)
                acc += h["n_calibrations"] - h["calib_rejected"]
                att += h["n_calibrations"]
                fwd += h["fwd"]
            lab = "once (ep 0)" if ce > 10 ** 6 else f"{ce} epoch(s)"
            rep(f"  {lab:>14} {np.mean(vals):>+10.4f} {np.std(vals):>8.4f} "
                f"{att:>9d} {acc:>9d} {fwd:>9d}")
        rep("")
        rep("  If the R^2 column is flat across the schedule, the attempt period can")
        rep("  be lengthened for free -- the gate, not the schedule, is what decides.")

    # ---------------- Part C: gate sample size ----------------
    if "c" in parts:
        rep("\n" + "=" * 78)
        rep("  Part C: GATE SAMPLE SIZE -- how big must the validation slice be?")
        rep("=" * 78)
        rep("  Shrinking the slice makes the gate's score noisier, so a WORSE")
        rep("  calibration can win by luck.  Run on `10d`, the task where the")
        rep("  calibration is known to be harmful, so any acceptance is a mistake.")
        rep("")
        X10, Y10, X10te, Y10te = make_regression_data("10d")
        bd10, _ = effective_budget(args.arch, X10, ["geometry", "weights", "full"],
                                   None, substrate=args.substrate)
        budget10 = int(bd10["full"])
        rep(f"  arch {args.arch}, budget {budget10} probes/epoch, "
            f"{args.seeds} seeds, 40 epochs")
        rep("")
        rep(f"  {'holdout':>8} {'val samples':>12} {'best search':>12} "
            f"{'hybrid':>10} {'accepted':>9} {'harm?':>7}")
        rep("  " + "-" * 64)
        for ho in HOLDOUT_GRID:
            n_val = len(range(0, X10.shape[0], ho))
            vals_h, acc, att = [], 0, 0
            for s in range(args.seeds):
                r, h = train(args.arch, "hybrid", X10, Y10, X10te, Y10te,
                             epochs=40, budget=budget10, seed=s, args=args,
                             guard_holdout=ho)
                vals_h.append(r)
                acc += h["n_calibrations"] - h["calib_rejected"]
                att += h["n_calibrations"]
            best = max(
                np.mean([train(args.arch, arm, X10, Y10, X10te, Y10te, epochs=40,
                               budget=budget10, seed=s, args=args)[0]
                         for s in range(args.seeds)])
                for arm in ("geometry", "weights", "full"))
            mh = float(np.mean(vals_h))
            rep(f"  {ho:>8d} {n_val:>12d} {best:>+12.4f} {mh:>+10.4f} "
                f"{acc:>9d} {'YES' if mh < best - 0.02 else 'no':>7}"
                + (f"   (of {att} attempts)" if att else ""))
        rep("")
        rep("  'harm' means the calibrated arm ends MEANINGFULLY below the best")
        rep("  search arm on a task where every accepted calibration is a mistake.")
        rep("  At holdout=4 the gate should reject everything (80/80); the question")
        rep("  is how far it can be shrunk before it starts letting harm through.")

    rep("")
    rep(f"  total wall time {time.time()-t0:.0f}s")
    path = rep.save()
    print(f"\nReport saved: {path}")

    plt = try_plots()
    if plt is not None and "a" in parts:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.6))
        for b in BUDGET_GRID:
            gaps = [grid[(b, ep)][1] - grid[(b, ep)][0] for ep in EPOCH_GRID]
            ax.plot(EPOCH_GRID, gaps, "o-", label=f"budget {b}")
        ax.set_xscale("log")
        ax.set_xlabel("epochs")
        ax.set_ylabel("readout R2 - geometry R2")
        ax.set_title(f"{args.arch}: is the capacity gap closing or asymptotic?")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "abl_07_convergence_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {p}")


if __name__ == "__main__":
    main()
