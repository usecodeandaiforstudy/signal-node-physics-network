"""
ext_26 -- the budget-matched run with the BASE's OWN family shape
=================================================================
Project: singal-node-physics-network, extension `multigraph`

WHAT THIS CLOSES
----------------
ext_25 removed ext_24's confound -- a fixed per-epoch budget, varying only N -- and gave
a NEGATIVE answer for a single-layer matrix-free family: the hybrid's low-budget advantage
goes from `+0.4060` at 52 particles to `-0.0240` at 208.  But it said, in its own scope
note, that **the family it scaled is not the base's own**: the base's families are
THREE-layer at the same total particle count.

So the base version's section 11 tail is only half answered.  This runs the same design on
**the base's own shape** -- the `F2` configuration, which is three-layer, front-loaded and
matrix-free -- scaled by keeping its proportions:

    T52   (40,  8,  4)   <-- IDENTICAL to the base's own F2
    T104  (80, 16,  8)
    T208  (160, 32, 16)

**`T52` is the base's `F2` exactly**, so it doubles as an anchor: whatever the design does
at the base's own size can be read against the base's published behaviour there.

WHY THE ANSWER COULD DIFFER FROM ext_25's
-----------------------------------------
A three-layer front-loaded family stores its parameters differently from a single layer of
the same total: layer 1 carries `(D+2)*N_1` while layers 2-3 carry `3*N_k` each, so the
geometry/readout split and the calibration's one-time cost `n_train * sum_k N_k` both
differ.  The base's own controlling quantity is readout-params-per-sample, and this family
changes how that quantity grows with the total.  **So a different answer here would be a
result, not an inconsistency.**

WHAT IS MEASURED
----------------
The same table as ext_25 -- the four arms' frontier values, the `@3000` comparison with
the budget held fixed, the curve's first point (which must be identical across N for the
comparison to be defined), and the learning-floor check, so a cell where the search arms
could not train is reported as uninformative rather than counted as a win.

SCOPE: task 4d, 1 seed, 40 epochs, budgets {600, 2400}, bases added to `abl_common.ARCHS`
AT RUNTIME.  The base file is not edited.

Output: outputs/ext_26_base_family.txt + _plot.png
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

import abl_common as ac                                  # noqa: E402

from abl_common import (                                 # noqa: E402
    Report, header, make_regression_data, set_diffusion_defaults, set_ideal,
    try_plots,
)
from extensions.multigraph.ext_24_particle_budget import (   # noqa: E402
    ARMS, EPOCHS, LOW_BUDGET, SUBSTRATE, frontier_and_rate,
)
from extensions.multigraph.ext_25_budget_matched import (    # noqa: E402
    FLOOR, run_cell_fixed,
)

# The BASE's own shape (F2), scaled by proportion.  T52 == abl_common.ARCHS["F2"].
BASE_FAMILY = {
    "T52": dict(particles=(40, 8, 4), out_dims=(1, 1, 1),
                label="base family (== the base's F2), 52 particles"),
    "T104": dict(particles=(80, 16, 8), out_dims=(1, 1, 1),
                 label="base family scaled 2x, 104 particles"),
    "T208": dict(particles=(160, 32, 16), out_dims=(1, 1, 1),
                 label="base family scaled 4x, 208 particles"),
}
BUDGETS = (600, 2400)


def install_base_family():
    """
    Runtime additions again -- the base file stays byte-identical.

    `T52` is checked against the base's own F2 spec before being installed, so a typo here
    cannot silently make the anchor something other than the base's architecture.
    """
    f2 = ac.ARCHS["F2"]
    assert tuple(BASE_FAMILY["T52"]["particles"]) == tuple(f2["particles"]), (
        "T52 must BE the base's F2; the specs differ")
    assert tuple(BASE_FAMILY["T52"]["out_dims"]) == tuple(f2["out_dims"])
    added = []
    for k, v in BASE_FAMILY.items():
        if k not in ac.ARCHS:
            ac.ARCHS[k] = v
            added.append(k)
    return added


def layer_params(particles, dim_in):
    """Forward passes a full sweep needs: sum_k (D_k + 2) * N_k, D_1 = input dim."""
    total, d = 0, dim_in
    for N in particles:
        total += (d + 2) * N
        d = 1                      # the matrix-free cascade carries one scalar
    return total


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_26: the base family, budget fixed")
    ap.add_argument("--task", default="4d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--archs", default="T52,T104,T208")
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    set_ideal()
    set_diffusion_defaults(level="none")
    added = install_base_family()
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    archs = [a for a in args.archs.split(",") if a.strip()]
    X_tr, y_tr, X_te, y_te = make_regression_data(args.task, seed=42)
    dim_in = X_tr.shape[1]

    rep = Report("ext_26_base_family.txt")
    for line in header("ext_26 - the base's own family, budget held fixed",
                       f"task={args.task} seed={args.seed} epochs={EPOCHS} "
                       f"budgets={budgets} archs={archs}"):
        rep(line)

    rep("")
    rep("  ext_25 removed ext_24's confound and answered NEGATIVELY -- but it said in its")
    rep("  own scope note that the family it scaled is NOT the base's own.  The base's")
    rep("  families are three-layer at the same total particle count, and a three-layer")
    rep("  front-loaded family stores its parameters differently, so the base's own")
    rep("  controlling quantity (readout params per sample) grows differently with N.")
    rep("")
    rep("  This scales the BASE's shape by proportion:")
    for k in archs:
        rep(f"    {k:<5} particles={BASE_FAMILY[k]['particles']}  "
            f"out_dims={BASE_FAMILY[k]['out_dims']}")
    rep("")
    rep("  T52 is IDENTICAL to the base's own F2 (asserted at install time), so it doubles")
    rep("  as an anchor at the base's own size.")
    rep(f"\n  runtime archs added: {added if added else '(already installed)'}")

    t0 = time.time()
    rep("\n" + "=" * 104)
    rep("  P1  the base's family, N x FIXED budget")
    rep("=" * 104)
    rep("")
    rep(f"  {'budget':>7} {'arch':<6} {'N':>4} {'params':>7} {'sweeps/ep':>10} "
        f"{'frontier':>9} {'best search':>12} {'hybrid':>10} {'win':>5} "
        f"{'@3000 hybrid':>13} {'@3000 best':>11} {'@3000 win':>10} {'1st m':>7} "
        f"{'floor':>7}")
    rep("  " + "-" * 122)
    out = {}
    for budget in budgets:
        for arch in archs:
            particles = BASE_FAMILY[arch]["particles"]
            N = sum(particles)
            curves = {}
            for arm in ARMS:
                info = run_cell_fixed(arch, arm, args.seed, args.task, X_tr, y_tr,
                                      X_te, y_te, budget)
                curves[arm] = info["curve"]
                print(f"  [P1] b={budget:<5} {arch}/{arm:<9} cache={info['from_cache']} "
                      f"pts={len(info['curve'])}", flush=True)
            r = frontier_and_rate(curves, low_budget=LOW_BUDGET)
            r.update(N=N, budget=budget, particles=particles,
                     first_m=int(curves["hybrid"][0][0]) if curves["hybrid"] else -1,
                     params=layer_params(particles, dim_in))
            r["sweeps"] = budget * 2.0 / max(r["params"], 1)
            r["floor_ok"] = r["best"] > FLOOR
            out[(budget, arch)] = r
            rep(f"  {budget:>7} {arch:<6} {N:>4} {r['params']:>7} {r['sweeps']:>10.2f} "
                f"{r['frontier']:>9} {r['best']:>+12.4f} {r['hybrid']:>+10.4f} "
                f"{'YES' if r['win'] else 'no':>5} {r['low_hybrid']:>+13.4f} "
                f"{r['low_best']:>+11.4f} {'YES' if r['low_win'] else 'no':>10} "
                f"{r['first_m']:>7} {'ok' if r['floor_ok'] else 'LOW':>7}")
        rep("")

    # ---- verdict ---------------------------------------------------------
    rep("\n" + "=" * 104)
    rep("  VERDICT: does the base's own family behave like the single-layer one?")
    rep("=" * 104)
    rep("")
    for budget in budgets:
        rows = [out[(budget, a)] for a in archs]
        rep(f"  budget {budget}:")
        rep(f"    first curve point across N: "
            f"{sorted(set(r['first_m'] for r in rows))}  -> @{LOW_BUDGET} "
            f"{'DEFINED for all N' if all(r['first_m'] < LOW_BUDGET for r in rows) else 'undefined somewhere'}")
        rep(f"    sweeps/epoch:  " + ", ".join(f"N={r['N']} {r['sweeps']:.2f}"
                                               for r in rows))
        rep(f"    clearing the floor: {sum(int(r['floor_ok']) for r in rows)}/{len(rows)}")
        usable = [r for r in rows if r["floor_ok"] and r["first_m"] < LOW_BUDGET]
        if len(usable) >= 2:
            margin = [r["low_hybrid"] - r["low_best"] for r in usable]
            rep(f"    @{LOW_BUDGET} margin across the usable cells: "
                + ", ".join(f"N={r['N']} {m:+.4f}" for r, m in zip(usable, margin)))
            rep(f"    -> {'SURVIVES' if all(m > 0 for m in margin) else 'DOES NOT SURVIVE'}"
                f" on this family at this budget")
        else:
            rep(f"    only {len(usable)} usable cell(s) -- no within-budget verdict")
        rep("")
    rep("  'USABLE' means the cell clears the learning floor AND its first curve point is")
    rep(f"  below {LOW_BUDGET}; a cell failing either carries no verdict.")
    rep("")
    base52 = out[(budgets[0], archs[0])]
    rep(f"  THE ANCHOR: {archs[0]} IS the base's F2, run here at budget {budgets[0]} with")
    rep(f"  the budget held fixed.  Its frontier best-search is {base52['best']:+.4f} and")
    rep(f"  hybrid {base52['hybrid']:+.4f}; compare with the base's own F2 rows under")
    rep("  effective_budget, which is a DIFFERENT budget and therefore not the same cell.")
    rep("")
    rep("  SCOPE: one task, one seed, 40 epochs.  The family is now the base's own SHAPE,")
    rep("  scaled -- not the base's own sizes, which are fixed at 52, so this is as close")
    rep("  to the base's own tail as one task and one seed allow.  A multi-seed version")
    rep("  would be needed before any of it is quoted as the base's result.")
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
        for budget in budgets:
            xs = [out[(budget, a)]["N"] for a in archs]
            ax[0].plot(xs, [out[(budget, a)]["low_hybrid"] for a in archs], "o-",
                       label=f"hybrid @{LOW_BUDGET}, b={budget}")
            ax[0].plot(xs, [out[(budget, a)]["low_best"] for a in archs], "s--",
                       label=f"best search @{LOW_BUDGET}, b={budget}")
        ax[0].set_xscale("log", base=2)
        ax[0].set_xticks([out[(budgets[0], a)]["N"] for a in archs])
        ax[0].set_xticklabels([str(out[(budgets[0], a)]["N"]) for a in archs])
        ax[0].set_xlabel("total particles (base family shape, budget FIXED)")
        ax[0].set_ylabel("R^2 at the low budget")
        ax[0].set_title("the BASE's own family, scaled")
        ax[0].grid(True, alpha=0.25)
        ax[0].legend(fontsize=7)
        x = np.arange(len(archs))
        for i, budget in enumerate(budgets):
            ax[1].bar(x + (i - 0.5) * 0.4,
                      [out[(budget, a)]["low_hybrid"] - out[(budget, a)]["low_best"]
                       for a in archs], 0.4, label=f"budget {budget}")
        ax[1].axhline(0.0, color="k", lw=0.9)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([f"N={out[(budgets[0], a)]['N']}" for a in archs])
        ax[1].set_ylabel("hybrid - best search, at the low budget")
        ax[1].set_title("the margin, base family, budget fixed")
        ax[1].grid(True, axis="y", alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_26_base_family_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}")

    # The assert encodes what the measurement supports: the anchor really is the base's
    # F2 spec, and every cell produced four curves, a frontier and a floor verdict.  It
    # cannot encode the answer, because that is the result.
    ok = (tuple(ac.ARCHS["T52"]["particles"]) == tuple(ac.ARCHS["F2"]["particles"])
          and all(out[(b, a)]["frontier"] > 0 for b in budgets for a in archs))
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
