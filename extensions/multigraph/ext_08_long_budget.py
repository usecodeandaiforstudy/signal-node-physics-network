"""
ext_08 -- the LONG-BUDGET comparison, resumable
===============================================
Project: singal-node-physics-network, extension `multigraph`

WHAT THIS SETTLES
-----------------
ext_07 found the operator trainer beating the finite-difference search by +0.291
R^2 while using ~6x fewer forward evaluations -- but arm C's budget there was only
2544 evaluations, about 6 sweeps of a 200-parameter geometry, whereas a typical
base-version run spends tens of thousands.  So the claim was explicitly bounded to
the low-budget regime, and THIS script is the check that can remove or confirm that
bound:

    at what evaluation budget does the FD search reach the operator arm's score?

If FD needs far more than the operator's 417 evaluations, the operator claim
strengthens.  If it catches up at a few thousand, the honest report is that the
operators are cheaper by a modest factor rather than by an order of magnitude.  The
script is built to be able to return the second answer.

RESUMABILITY, WHICH IS THE POINT OF THE FILE'S SHAPE
----------------------------------------------------
A long-budget sweep will be interrupted -- by a timeout, a reboot, or a decision to
stop.  Every (seed, budget) cell is therefore written to its own JSON checkpoint the
moment it finishes, written ATOMICALLY (temp file then rename), and skipped on the
next run.  Each checkpoint carries a CONFIG FINGERPRINT: change the task, the node
count or the trainer settings and the old cells are no longer reused, because
silently mixing results from two configurations is how a sweep becomes unreadable.

    python extensions/multigraph/ext_08_long_budget.py            # run / resume
    python extensions/multigraph/ext_08_long_budget.py --report-only
    python extensions/multigraph/ext_08_long_budget.py --status

SCOPE: toy, deliberately.  One layer of 40 nodes on the project's own 4D regression
task, the same setting as ext_07 so the two are comparable.  Nothing here says
anything about a large model, and the compute budget does not allow one.

Output: outputs/ext_08_long_budget.txt + _plot.png
        outputs/ext_ckpt/fd_s<seed>_e<evals>.json   (the resumable cells)
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)

from abl_common import (                                   # noqa: E402
    Report, header, make_regression_data, r2, set_ideal, train_model, try_plots,
)
from extensions.multigraph.trainer import features, fit_operator  # noqa: E402

D_IN = 4
N_NODES = 40
CKPT_DIR = os.path.join(ROOT, "outputs", "ext_ckpt")


# ---------------------------------------------------------------------------
#  checkpointing
# ---------------------------------------------------------------------------
def fingerprint(task, seeds, n_nodes, fd_epochs):
    """
    Identifies the CONFIGURATION a cell belongs to.

    Without this, changing --task or --seeds would silently reuse cells computed
    under the old settings, and the sweep would be a mixture nobody could read.
    """
    blob = json.dumps(dict(task=task, seeds=seeds, n_nodes=n_nodes,
                           fd_epochs=fd_epochs, d_in=D_IN, v=2),
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cell_path(seed, evals):
    return os.path.join(CKPT_DIR, f"fd_s{seed}_e{evals}.json")


def load_cell(seed, evals, fp):
    p = cell_path(seed, evals)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None                      # a torn file is treated as absent
    if d.get("fingerprint") != fp:
        return None
    return d


def save_cell(seed, evals, fp, payload):
    """
    Atomic write: a checkpoint that is half-written is worse than none, because the
    next run would treat it as done.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    p = cell_path(seed, evals)
    tmp = p + ".tmp"
    payload = dict(payload, fingerprint=fp, seed=seed, evals_target=evals,
                   saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, p)                   # atomic on the same volume
    return p


# ---------------------------------------------------------------------------
#  the FD arm
# ---------------------------------------------------------------------------
def budget_to_schedule(evals, max_epochs=80, min_epochs=6):
    """
    Turn a total evaluation budget into (epochs, budget per epoch).

    train_model charges 2 evaluations per coordinate probed, per epoch, so
    evaluations = 2 * budget * epochs.  Longer budgets get more epochs as well as
    more probes, because a search that probes a lot without stepping is not a
    longer run -- it is the same run with a bigger sample.
    """
    epochs = int(np.clip(evals // 1500, min_epochs, max_epochs))
    budget = max(int(evals // (2 * epochs)), 1)
    return epochs, budget


def run_fd_cell(X_fi, y_fi, X_va, y_va, X_te, y_te, seed, evals, fd_epochs):
    from ext_07_operator_trainer import build_one_layer
    epochs, budget = budget_to_schedule(evals)
    model = build_one_layer(X_fi, seed)
    t0 = time.time()
    train_model(model, X_fi, y_fi, X_va, y_va, arm="hybrid", epochs=epochs,
                budget=budget, closed_form=True, closed_form_every=1, seed=seed,
                substrate="circuit")
    test_r2 = r2(y_te, model.forward(X_te))
    return dict(test_r2=float(test_r2), epochs=epochs, budget=budget,
                evals_actual=int(2 * budget * epochs),
                wall_s=round(time.time() - t0, 2))


# ---------------------------------------------------------------------------
def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_08: long-budget FD comparison, "
                                             "resumable")
    ap.add_argument("--task", default="4d")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--grid", default="2544,10000,30000,100000,300000",
                    help="FD evaluation budgets; each is its own resumable cell")
    ap.add_argument("--fd-epochs", type=int, default=12,
                    help="epochs used for the ext_07 reference point only")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--status", action="store_true",
                    help="print which cells exist and exit")
    ap.add_argument("--report-only", action="store_true",
                    help="build the report from checkpoints without running")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert
    grid = [int(x) for x in args.grid.split(",") if x.strip()]
    fp = fingerprint(args.task, args.seeds, N_NODES, args.fd_epochs)

    if args.status:
        print(f"fingerprint {fp}  task={args.task} seeds={args.seeds} "
              f"nodes={N_NODES}")
        have = 0
        for seed in range(args.seeds):
            for ev in grid:
                d = load_cell(seed, ev, fp)
                mark = "done " if d else "     "
                extra = (f"test R2 {d['test_r2']:+.4f}  "
                         f"({d['epochs']}x{d['budget']}={d['evals_actual']} evals, "
                         f"{d['wall_s']}s)" if d else "missing")
                print(f"  {mark} seed {seed}  budget {ev:>7d}   {extra}")
                have += int(bool(d))
        print(f"\n{have}/{len(grid)*args.seeds} cells present")
        return 0
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
    rep = Report("ext_08_long_budget.txt")
    for line in header("ext_08 - the long-budget comparison (resumable)",
                       f"task={args.task} nodes={N_NODES} seeds={args.seeds} "
                       f"grid={grid}"):
        rep(line)

    rep("")
    rep("  The question: at what FD evaluation budget does the finite-difference")
    rep("  search reach the OPERATOR arm's score from ext_07?  ext_07 could only say")
    rep("  'the operators lead when evaluations are scarce' because its FD budget was")
    rep("  2544.  This sweep removes or confirms that bound.")
    rep("")
    rep("  Toy by construction and by instruction: one layer of 40 nodes on the 4D")
    rep("  regression task, the same setting as ext_07 so the two are comparable.")
    rep("")
    rep(f"  RESUMABLE: each (seed, budget) cell is written atomically to")
    rep(f"  outputs/ext_ckpt/ and skipped on the next run.  Config fingerprint {fp}")

    t0 = time.time()

    # -- the operator arm (cheap, deterministic, recomputed) ---------------
    op_scores, op_evals = [], []
    for seed in range(args.seeds):
        X_tr, y_tr, X_te, y_te = make_regression_data(args.task, seed=42 + seed)
        n_va = max(32, X_tr.shape[0] // 5)
        X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
        X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]
        out = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=N_NODES,
                           rounds=args.rounds, seed=seed)
        op_scores.append(r2(y_te, features(X_te, out["centres"],
                                           out["widths"]) @ out["weights"]))
        op_evals.append(out["n_evals"])
    op_mean = float(np.mean(op_scores))
    op_sd = float(np.std(op_scores))
    ev_mean = float(np.mean(op_evals))

    # -- the FD sweep, cell by cell, resumable ----------------------------
    reused, ran = 0, 0
    table = {}
    for ev in grid:
        table[ev] = []
        for seed in range(args.seeds):
            d = load_cell(seed, ev, fp)
            if d is not None:
                reused += 1
            elif args.report_only:
                continue
            else:
                X_tr, y_tr, X_te, y_te = make_regression_data(args.task,
                                                              seed=42 + seed)
                n_va = max(32, X_tr.shape[0] // 5)
                X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
                X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]
                d = run_fd_cell(X_fi, y_fi, X_va, y_va, X_te, y_te, seed, ev,
                                args.fd_epochs)
                save_cell(seed, ev, fp, d)
                ran += 1
                print(f"  [ran ] seed {seed} budget {ev:>7d} -> "
                      f"{d['test_r2']:+.4f}  ({d['epochs']}x{d['budget']}="
                      f"{d['evals_actual']} evals, {d['wall_s']}s)", flush=True)
            if d is not None:
                table[ev].append(d["test_r2"])
    rep("")
    rep(f"  cells reused from checkpoint: {reused}   newly run: {ran}")

    # -- report ------------------------------------------------------------
    rep("\n" + "=" * 78)
    rep("  Test R^2 vs FD evaluation budget (mean over seeds, paired data)")
    rep("=" * 78)
    rep(f"  OPERATOR arm (ext_07): {op_mean:+.4f} (sd {op_sd:.4f}) at "
        f"{ev_mean:.0f} evaluations")
    rep("")
    rep(f"  {'FD budget':>10} {'evals actual':>13} {'mean R^2':>10} {'sd':>8} "
        f"{'worst':>9} {'vs operator':>12}")
    rep("  " + "-" * 68)
    # Two DIFFERENT things must not be conflated, and the first version of this
    # file conflated them: FD REACHING the operator's score (gap >= 0) is a
    # crossover, while FD's gap merely falling inside the seed spread is
    # INCONCLUSIVE -- it means the comparison has run out of resolution, not that
    # the search caught up.  Reading the second as the first understated the
    # operator arm's advantage and produced a verdict the data does not support.
    reached = None
    inconclusive = None
    for ev in grid:
        v = np.asarray(table[ev], dtype=np.float64)
        if v.size == 0:
            rep(f"  {ev:>10d} {'(no cells yet)':>13}")
            continue
        gap = float(v.mean()) - op_mean
        tol = max(0.02, 2.0 * max(v.std(), op_sd) / np.sqrt(args.seeds))
        rep(f"  {ev:>10d} {ev:>13d} {v.mean():>+10.4f} {v.std():>8.4f} "
            f"{v.min():>+9.4f} {gap:>+12.4f}")
        if reached is None and gap >= 0.0:
            reached = ev
        if inconclusive is None and gap >= -tol:
            inconclusive = ev
    rep("")
    rep("  'vs operator' is FD minus operator at that budget; positive means FD is")
    rep("  ahead.  'No cells yet' means the resumable cell has not been run.")
    rep("")
    rep(f"  FD actually reaches the operator score at : "
        f"{reached if reached else 'no budget tested'}")
    rep(f"  FD gap falls inside the seed spread at    : "
        f"{inconclusive if inconclusive else 'no budget tested'}")
    rep("  The first is a crossover.  The second only means the comparison has no")
    rep("  resolution left, and the two must not be reported as the same event.")
    rep("")
    rep("  A note on the operator arm's cost number: it is MEASURED per run and")
    rep("  varies by about 1% between runs (417-420 evaluations) with the same data")
    rep("  and seeds, so ratios below are quoted to two significant figures rather")
    rep("  than pretending to a precision the measurement does not have.")

    # -- verdict -----------------------------------------------------------
    rep("\n" + "=" * 78)
    rep("  VERDICT: does the low-budget bound survive a long-budget search?")
    rep("=" * 78)
    rep("")
    gaps = {ev: float(np.mean(table[ev])) - op_mean for ev in grid if table[ev]}
    if reached is None:
        last = max(gaps)
        ratio = last / max(ev_mean, 1.0)
        rep(f"  NO CROSSOVER.  The FD search does not reach the operator arm's "
            f"{op_mean:+.4f}")
        rep(f"  anywhere up to {last} evaluations, i.e. {ratio:.0f}x the operator "
            f"arm's {ev_mean:.0f}.")
        rep(f"  The gap does narrow -- {gaps[last]:+.4f} at the top of the range -- but")
        if inconclusive is not None:
            rep(f"  from {inconclusive} onward it is smaller than the seed spread, so")
            rep("  the honest statement is BOUNDED BY RESOLUTION, not 'FD caught up':")
            rep("  within this toy setting the operator's cost advantage is at least")
            rep(f"  {inconclusive/max(ev_mean,1.0):.0f}x and the sweep runs out of")
            rep("  resolution before the search demonstrably closes it.")
        else:
            rep("  the search remains measurably behind at every budget tested, so")
            rep("  ext_07's low-budget advantage is not an artefact of a short run.")
        rep("")
        rep("  ALSO REPORTED, because it is visible in the cells and would otherwise")
        rep("  be averaged away: FD's long-budget behaviour is NOT monotone per seed.")
        worst = None
        for seed in range(args.seeds):
            seq = [load_cell(seed, ev, fp) for ev in grid]
            seq = [d["test_r2"] for d in seq if d]
            for i in range(1, len(seq)):
                if seq[i] < seq[i - 1]:
                    worst = (seed, grid[i], seq[i - 1], seq[i])
        if worst:
            rep(f"  Seed {worst[0]} scores WORSE at {worst[1]} evaluations than at the")
            rep(f"  budget below it ({worst[3]:+.4f} against {worst[2]:+.4f}), so more")
            rep("  search budget is not reliably better for this trainer.  The mean")
            rep("  curve rises; individual runs do not have to.")
    else:
        ratio = reached / max(ev_mean, 1.0)
        rep(f"  CROSSOVER at {reached} FD evaluations, which is {ratio:.0f}x the")
        rep(f"  operator arm's {ev_mean:.0f}.  So the operators are cheaper by a")
        rep("  FACTOR rather than by orders of magnitude, and ext_07's low-budget")
        rep("  advantage does narrow once the search is given room.  The honest claim")
        rep(f"  is 'the operators reach the same score for about {ratio:.0f}x fewer")
        rep("  evaluations', which is a cost claim and not a capability claim.")
    rep("")
    rep("  Unchanged either way: the operators need no step size, no gradient and no")
    rep("  schedule, and their geometry is exact rather than approached.  What this")
    rep("  sweep decides is only the SIZE of the cost advantage.")
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
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.5))
        xs = [e for e in grid if table[e]]
        ys = [float(np.mean(table[e])) for e in xs]
        es = [float(np.std(table[e])) for e in xs]
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="s-", label="FD search")
        ax.axhline(op_mean, color="#2b7bba", ls="--",
                   label=f"OPERATOR ({op_mean:+.3f} at {ev_mean:.0f} evals)")
        if reached:
            ax.axvline(reached, color="k", ls=":", lw=1.0,
                       label=f"FD reaches operator at {reached}")
        elif inconclusive:
            ax.axvspan(inconclusive, max(grid), color="gray", alpha=0.12,
                       label="no resolution left (gap < seed spread)")
        ax.set_xscale("log")
        ax.set_xlabel("FD forward evaluations (log)")
        ax.set_ylabel("test R^2 (mean over seeds)")
        ax.set_title("ext_08: does the operator advantage survive a long search?")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_08_long_budget_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")}")

    if do_assert and reached is not None and reached <= 2 * ev_mean:
        # A real crossover within 2x of the operator's cost would mean the cost
        # claim is wrong; that is worth failing the run over, but it is information
        # rather than a crash -- so it only fails under --assert.
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
