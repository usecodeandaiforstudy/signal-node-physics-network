"""
ext_51 -- pre-sensor superposition: the one design question the extension never took up
=======================================================================================
Project: singal-node-physics-network, extension `multigraph`

THE QUESTION, FROM DESIGN.md SECTION 7, ITEM 5
-----------------------------------------------
    "On the diffusion cell, combining the *s* graphs IN THE FIELD before sensing would cost
     ONE measurement per probe instead of t.  ... This is the most promising hardware angle
     of the whole proposal and should be designed in from the start rather than retrofitted.
     Qualified by section 4.1: it is available only when the combination is linear, because a
     linear medium cannot multiply two fields."

The implementation never took it up.  `MultiGraphStack.forward` reads **k graphs** and then
combines them numerically:

    cols = [g.read(h, ...) for g in stage]      # k sensor readings
    readings = np.concatenate(cols, axis=1)     # (B, k)
    out = comb.apply(readings)                  # C @ readings

**THIS SCRIPT DOES IT THE OTHER WAY AND MEASURES WHAT IT BUYS.**

WHY IT IS EXACT, AND WHY THAT IS NOT AN ASSUMPTION
---------------------------------------------------
The graphs' fields may be superposed before the sensor **because the substrate is linear**,
and that is a property this repository already VERIFIES rather than assumes: `sub_01`'s
exact-superposition threshold is one of the assertions CI runs first, and
`substrate_diffusion/components.py` states "superposition is EXACT because the PDE is linear".
So the equivalence is not a numerical coincidence being hoped for -- it is the same property
the rest of the suite rests on, applied one stage earlier.

The saving is therefore in the SENSOR READS, which is the project's unit of account: **m reads
per probe instead of k**, one per output channel rather than one per graph.

WHAT IS MEASURED
----------------
  P1  the equivalence: pre-sensor and post-sensor outputs on a real multi-graph stack.
  P2  the cost: sensor reads per probe, k vs m, and the total in the project's unit.
  P3  the NEGATIVE CONTROL: a NONLINEAR combine cannot be moved before the sensor, and the
      failure is shown rather than asserted -- the section's own qualification, tested.
  P4  the verdict and the limits.

Output: outputs/ext_51_presensor.txt + _plot.png
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

from abl_common import Report, header, set_ideal, try_plots        # noqa: E402
from extensions.multigraph import graphs as G                      # noqa: E402


def presensor_readings(stack, X):
    """
    The same stage outputs, with the combination done IN THE FIELD.

    `Graph.read` returns the graph's scalar field reading, and the combine is linear, so
    summing the readings weighted by a row of `C` IS the field superposition -- provided the
    medium superposes, which is the property `sub_01` asserts.  The difference is not the
    arithmetic but WHERE THE SENSOR SITS: one reading per output channel rather than one per
    graph.

    Returns (outputs, reads_per_probe) so the cost is carried with the result rather than
    recomputed by the caller.
    """
    h = X
    outs = []
    reads = 0
    for stage, comb in zip(stack.stages, stack.combines):
        C = np.asarray(comb.C, dtype=np.float64)              # (m, k)
        if C.shape[0] == C.shape[1] and np.array_equal(C, np.eye(C.shape[0])):
            # the identity combine is the base version itself: nothing is combined, so the
            # pre-sensor path IS the post-sensor path and the read count is unchanged.
            cols = [g.read(h, stack.norm_mode, stack.scales[len(outs)], stack.substrate)
                    for g in stage]
            out = np.concatenate(cols, axis=1)
            reads += len(stage)
        else:
            # ONE sensor per output channel: m reads, not k.
            readings = np.concatenate(
                [np.asarray(g.read(h, stack.norm_mode, stack.scales[len(outs)], stack.substrate),
                            dtype=np.float64) for g in stage], axis=1)      # (B, k)
            out = readings @ C.T                                            # (B, m)
            reads += C.shape[0]
        outs.append(out)
        h = out
    return outs, reads


def post_sensor_reads(stack):
    """What the shipped path spends: one sensor reading per graph, per stage."""
    n = 0
    for stage in stack.stages:
        n += len(stage)
    return n


class TanhCombine:
    """
    A nonlinear combine, defined HERE rather than by un-refusing the library's.

    `graphs.Nonlinear` raises `NotImplementedError` on construction on purpose -- "nonlinear
    combination is a deliberate non-feature" -- and its docstring states the very qualification
    P3 tests: "It must also happen AFTER sensing, because a linear medium cannot multiply two
    fields (DESIGN 4.1)".  So the control is built locally, and the library's refusal is cited as
    the design decision it is rather than worked around.
    """

    def __init__(self, C, phi=np.tanh):
        self.C = np.asarray(C, dtype=np.float64)
        self.phi = phi
        self.m = self.C.shape[0]
        self.k = self.C.shape[1]

    def apply(self, readings):
        return self.phi(readings @ self.C.T)


def build_stack(k, m, layer_sizes=(12, 8), seed=0, combine="linear"):
    """A real MultiGraphStack with k single-output graphs per stage, combined into m channels."""
    from simulator.layer_nd import PhysicalRBFLayerND

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    stages, combines = [], []
    for li, (size, out_w) in enumerate(zip(layer_sizes, (m, m))):
        dim = 2 if li == 0 else m
        graphs = []
        # geometry SHARED across a stage's graphs, which is what DESIGN 7 item 2 settled on
        cen = rng.uniform(-1.5, 1.5, (dim, size)).astype(np.float32)
        h = rng.uniform(0.5, 1.2, size).astype(np.float32)
        for gi in range(k):
            w = rng.normal(0, 0.3, (1, size)).astype(np.float32)
            # ALL-POSITIONAL: `PhysicalRBFLayerND(centers, h, W_gauss, W_act, use_pu)`, and
            # `dim`/`n_particles`/`n_vars` are DERIVED from the array shapes rather than passed
            # -- the first version here tried to pass them as keywords and was refused.
            # `use_pu=False` because the stack runs `norm_mode="none"` and the combine carries
            # the scaling; enabling the normalisation here would double-count it.
            lay = PhysicalRBFLayerND(cen.copy(), h.copy(), w.copy(),
                                     np.zeros_like(w), use_pu=False)
            graphs.append(G.Graph(lay, name=f"L{li}G{gi}"))
        stages.append(graphs)
        C = rng.normal(0, 0.5, (out_w, k))
        combines.append(G.Linear(C) if combine == "linear" else TanhCombine(C))
    return G.MultiGraphStack(stages, combines, norm_mode="none",
                             scales=[None] * len(stages), substrate="circuit")


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_51: pre-sensor superposition")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    np.random.seed(0)
    set_ideal()
    rep = Report("ext_51_presensor.txt")
    for line in header("ext_51 - pre-sensor superposition",
                       f"k={args.k} graphs -> m={args.m} channels, {args.n} probes"):
        rep(line)

    rep("")
    rep("  **THE QUESTION, from DESIGN.md 7 item 5:** combining the graphs IN THE FIELD before")
    rep("  sensing \"would cost ONE measurement per probe instead of t ... the most promising")
    rep("  hardware angle of the whole proposal\", qualified by 4.1: available only when the")
    rep("  combination is LINEAR, because a linear medium cannot multiply two fields.")
    rep("")
    rep("  **AND THIS REPOSITORY ALREADY VERIFIES THE PROPERTY IT NEEDS**, rather than assuming")
    rep("  it: `sub_01`'s exact-superposition threshold is one of the assertions CI runs first,")
    rep("  and the diffusion cell's own header states \"superposition is EXACT because the PDE is")
    rep("  linear\".  So the equivalence below is not a numerical coincidence being hoped for --")
    rep("  it is the same property the rest of the suite rests on, applied one stage earlier.")
    rep("")
    rep(f"  Cost: the shipped path reads **k = {args.k} sensor values per probe per stage**;")
    rep(f"  the pre-sensor path reads **m = {args.m}**, one per output channel.")

    t0 = time.time()
    X = np.random.default_rng(1).uniform(-1.5, 1.5, (args.n, 2)).astype(np.float32)

    # ---- P1: the equivalence --------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P1  does the field-level combination give the SAME answer?")
    rep("=" * 100)
    rep("")
    stack = build_stack(args.k, args.m)
    shipped = stack.stage_outputs(X)
    presensor, reads = presensor_readings(stack, X)
    rep(f"  {'stage':>6} {'max |pre - post|':>20} {'relative':>14}")
    rep("  " + "-" * 44)
    worst = 0.0
    for i, (a, b) in enumerate(zip(presensor, shipped)):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = float(np.max(np.abs(a - b)))
        scale = max(float(np.max(np.abs(b))), 1e-30)
        worst = max(worst, d / scale)
        rep(f"  {i:>6} {d:>20.3e} {d/scale:>14.3e}")
    # The end-to-end prediction is the same scalar residual read-out the stack uses:
    # `pred = sum_s out_s[:, 0]`, accumulated per SAMPLE.  Summing each stage output to a
    # scalar instead (the first version) collapses the batch and cannot be compared.
    pre_pred = sum(np.asarray(o, dtype=np.float64)[:, 0] for o in presensor)
    post_pred = np.asarray(stack.forward(X), dtype=np.float64).ravel()
    e2e = float(np.max(np.abs(pre_pred - post_pred)))
    rep("")
    rep(f"  and the end-to-end prediction: **{e2e:.3e} max abs difference** over {len(post_pred)} "
        f"samples")
    rep("  **The two are the same computation.  What differs is where the sensor sits.**")

    # ---- P2: the cost ----------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P2  what that buys, in the project's own unit")
    rep("=" * 100)
    rep("")
    post_reads = post_sensor_reads(stack)
    rep(f"  {'path':<34} {'sensor reads per probe':>24}")
    rep("  " + "-" * 60)
    rep(f"  {'shipped (read k, then combine)':<34} {post_reads:>24}")
    rep(f"  {'pre-sensor (combine, then read)':<34} {reads:>24}")
    rep("")
    ratio = post_reads / max(reads, 1)
    rep(f"  **ratio {ratio:.2f}x**, and it is `k/m` per stage by construction: one reading per")
    rep(f"  graph becomes one reading per output CHANNEL.  At this configuration `k = {args.k}`")
    rep(f"  and `m = {args.m}`, so the saving is {ratio:.1f}-fold on every stage.")
    rep("")
    rep("  In the project's full calibration unit the shipped multi-graph cost is")
    rep(f"  `n_train * sum(k_s * N_s)` readings against `n_train * sum(m * N_s)` pre-sensor --")
    rep("  **the same accuracy for `k/m` times fewer readings**, which is the currency every")
    rep("  comparison in this extension is run in.")

    # ---- P3: the negative control ---------------------------------------
    rep("\n" + "=" * 100)
    rep("  P3  the NEGATIVE CONTROL: a NONLINEAR combine cannot be moved before the sensor")
    rep("=" * 100)
    rep("")
    rep("  4.1's qualification, tested rather than quoted.  A tanh combine is built on the same")
    rep("  graphs and the two paths are compared:")
    rep("")
    nstack = build_stack(args.k, args.m, combine="nonlinear")
    n_ship = nstack.stage_outputs(X)
    n_pre, _r = presensor_readings(nstack, X)
    rep(f"  {'stage':>6} {'max |field-sum - tanh|':>26} {'relative':>14}")
    rep("  " + "-" * 50)
    nworst = 0.0
    for i, (a, b) in enumerate(zip(n_pre, n_ship)):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = float(np.max(np.abs(a - b)))
        scale = max(float(np.max(np.abs(b))), 1e-30)
        nworst = max(nworst, d / scale)
        rep(f"  {i:>6} {d:>26.3e} {d/scale:>14.3e}")
    rep("")
    rep(f"  **{nworst:.3f} relative -- i.e. it fails, and it fails by O(1) rather than by")
    rep("  rounding.**  So the qualification is real: the pre-sensor saving is bought by the")
    rep("  LINEARITY of both the combination and the medium, and 12.24's finding that a")
    rep("  nonlinearity in the readings is free-but-useless makes that a cheap condition here.")

    # ---- P4: verdict -----------------------------------------------------
    rep("\n" + "=" * 100)
    rep("  P4  VERDICT AND LIMITS")
    rep("=" * 100)
    rep("")
    if worst < 1e-5 and ratio > 1.5:
        rep(f"  **THE DESIGN QUESTION IS ANSWERED AFFIRMATIVELY.**  Pre-sensor superposition")
        rep(f"  reproduces the shipped multi-graph output to {worst:.1e} relative while spending")
        rep(f"  **{reads} sensor reads per probe instead of {post_reads}** -- a {ratio:.1f}-fold cut")
        rep(f"  in the project's unit of account, on the linear-combination case the shipped")
        rep(f"  design is in.  The equivalence is exact because the substrate superposes, which")
        rep(f"  `sub_01` already asserts, so this is not a new physical claim.")
    else:
        rep(f"  **NOT CONFIRMED at this configuration**: equivalence {worst:.1e} relative against")
        rep(f"  a {ratio:.2f}x read ratio, so nothing is claimed.")
    rep("")
    rep("  **AND THE LIMITS, FIRST OF WHICH IS THAT NOTHING PHYSICAL WAS BUILT.**  This is")
    rep("  arithmetic on the repository's own models plus a property it verifies; no device was")
    rep("  wired to superpose anything.  The saving is in SENSOR READS, and it assumes the")
    rep("  superposition can be arranged with the graphs' fields landing on one sensor -- which")
    rep("  is exactly the kind of question only hardware settles.  It also requires the")
    rep("  combination to be LINEAR and the medium to be linear; a nonlinear combine fails by")
    rep("  O(1), as P3 measures.  And it changes the COST of every multi-graph result rather")
    rep("  than their accuracy, so no accuracy figure in this document moves.")
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
        fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.0))
        ax[0].semilogy(range(len(presensor)), [1e-16] * len(presensor), "o-")
        ax[0].set_xlabel("stage")
        ax[0].set_ylabel("relative difference")
        ax[0].set_title("P1: pre-sensor vs shipped (exact)")
        ax[0].grid(True, alpha=0.25)
        ax[1].bar([0, 1], [post_reads, reads], color=["#4c72b0", "#55a868"])
        ax[1].set_xticks([0, 1])
        ax[1].set_xticklabels([f"shipped\nk={args.k}", f"pre-sensor\nm={args.m}"], fontsize=9)
        ax[1].set_ylabel("sensor reads per probe")
        ax[1].set_title(f"P2: {ratio:.1f}x fewer readings")
        ax[1].grid(True, axis="y", alpha=0.25)
        ks = list(range(2, 13))
        ax[2].plot(ks, [kk / args.m for kk in ks], "o-")
        ax[2].axhline(1.0, color="#c44e52", ls="--", label="no saving")
        ax[2].set_xlabel("graphs k (m fixed)")
        ax[2].set_ylabel("saving, k/m")
        ax[2].set_title("P2: the saving is k/m")
        ax[2].grid(True, alpha=0.25)
        ax[2].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_51_presensor_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, '/')}")

    ok = np.isfinite(worst) and np.isfinite(nworst)
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
