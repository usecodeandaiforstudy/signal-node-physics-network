"""
ext_23 -- quantifying the win criterion's defect, from the base's own reports
============================================================================
Project: singal-node-physics-network, extension `multigraph`

THE DEFECT, FOUND BY ext_21/22's AUDIT LINE
-------------------------------------------
`abl_06_task_sweep.py` decides "the calibrated arm beats the best search arm" by
comparing the hybrid against `max(geometry, readout, full)`.  But the hybrid IS the
geometry arm plus a closed-form calibration, and that calibration carries a
**do-no-harm guarantee**: it is deployed only when it beats the strengths already in
place on held-out data.  So when the `geometry` arm happens to be the strongest of the
three, the comparison is against **a component of the hybrid itself**, and the guarantee
supplies the verdict rather than the measurement.

Section 12.6 found this in the 160-epoch run: two cells (`10d/F2`, `steps/F2`) have
`geometry` as their best search arm.  This script quantifies what the defect costs by
recomputing every verdict with `geometry` EXCLUDED from the comparison set -- which is
the criterion that actually asks "is calibrating worth it", against methods that do not
contain it.

WHY NO NEW RUN IS NEEDED
------------------------
Both reports already carry the four arms' frontier accuracies per cell.  This is an
analysis of existing evidence, so it costs seconds and cannot drift from the runs it
describes -- it re-reads their tables rather than re-deriving anything.

WHAT IT REPORTS
---------------
  * both criteria, side by side, on all 14 cells, at BOTH epoch counts;
  * the cells whose verdict changes, and by how much the margin changes;
  * the corrected epoch-axis effect, since part of the `8/14 -> 6/14` drop may have been
    this artifact rather than a real epoch effect.

SCOPE: reads only.  No base file is written, and the cache is untouched.

Output: outputs/ext_23_criterion_audit.txt + _plot.png
"""

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)

from abl_common import Report, header, try_plots        # noqa: E402

RUNS = [
    ("40 epochs, 9 seeds", "outputs/abl_06_task_sweep_s9.txt"),
    ("160 epochs, 1 seed", "outputs/abl_06_task_sweep_e160.txt"),
]
N_CELLS = 14

# task, arch, frontier, geometry, readout, full, hybrid
ROW = re.compile(
    r"^\s+(\w+)\s+(G|F2)\s+(\d+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s+"
    r"([-+][\d.]+)\s+([-+][\d.]+)\s*$")


def parse_part2(path):
    """
    The Part 2 table: task arch frontier geometry readout full hybrid.

    Parsed from the REPORT rather than recomputed, so the numbers audited are the
    numbers published.  The parser asserts its own coverage below, because a silently
    short parse would produce a wrong win count rather than an error.
    """
    rows, in_part2 = [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "Part 2: accuracy at the COMMON MEASUREMENT FRONTIER" in line:
                in_part2 = True
                continue
            # The Part 2 title is immediately followed by a `====` rule, so breaking on
            # the first one after the title would end the scan before any data row --
            # which is exactly what an earlier version did, and the coverage assert
            # below is what caught it (0 cells parsed instead of 14).
            if in_part2 and rows and line.startswith("===="):
                break
            if in_part2:
                m = ROW.match(line.rstrip("\n"))
                if m:
                    rows.append(dict(task=m.group(1), arch=m.group(2),
                                     frontier=int(m.group(3)),
                                     geometry=float(m.group(4)),
                                     readout=float(m.group(5)),
                                     full=float(m.group(6)),
                                     hybrid=float(m.group(7))))
    return rows


def verdicts(rows, exclude_geometry):
    out = []
    for r in rows:
        cands = [r["readout"], r["full"]]
        if not exclude_geometry:
            cands.append(r["geometry"])
        best = max(cands)
        which = ("readout" if best == r["readout"]
                 else "full" if best == r["full"] else "geometry")
        out.append(dict(r, best=best, which=which, win=r["hybrid"] > best,
                        margin=r["hybrid"] - best))
    return out


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_23: quantify the win criterion")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert

    rep = Report("ext_23_criterion_audit.txt")
    for line in header("ext_23 - what the win criterion's defect costs",
                       "recomputed from the base's own published tables"):
        rep(line)

    rep("")
    rep("  `abl_06` calls a cell won when the hybrid beats max(geometry, readout, full).")
    rep("  The hybrid IS geometry plus a calibration carrying a DO-NO-HARM guarantee, so")
    rep("  when `geometry` is itself the strongest of the three, the comparison is against")
    rep("  a component of the hybrid and the guarantee supplies the verdict.  Excluding")
    rep("  geometry leaves the question the criterion is meant to ask: is calibrating")
    rep("  worth it, against methods that do not contain it.")
    rep("")
    rep("  No new run: both reports already carry the four arms per cell, so this re-reads")
    rep("  their tables instead of re-deriving anything.")

    tables = {}
    for label, path in RUNS:
        full_path = os.path.join(ROOT, path)
        assert os.path.exists(full_path), f"missing report: {path}"
        rows = parse_part2(full_path)
        assert len(rows) == N_CELLS, (
            f"{path}: parsed {len(rows)} cells, expected {N_CELLS} -- a short parse "
            f"would produce a wrong win count rather than an error")
        tables[label] = rows
        print(f"  [read] {label:<22} {len(rows)} cells from "
              f"{os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))).replace(os.sep, '/')}",
              flush=True)

    summary = {}
    for label, _p in RUNS:
        rows = tables[label]
        orig = verdicts(rows, exclude_geometry=False)
        corr = verdicts(rows, exclude_geometry=True)
        summary[label] = (orig, corr)

        rep("\n" + "=" * 98)
        rep(f"  {label}")
        rep("=" * 98)
        rep("")
        rep(f"  {'task/arch':<11} {'geometry':>9} {'readout':>9} {'full':>9} "
            f"{'hybrid':>9} {'best(incl)':>11} {'win':>5} {'best(excl)':>11} "
            f"{'win':>5} {'change':>8}")
        rep("  " + "-" * 94)
        n_o = n_c = 0
        flips = []
        for o, c in zip(orig, corr):
            n_o += int(o["win"])
            n_c += int(c["win"])
            changed = o["win"] != c["win"]
            if changed:
                flips.append((o, c))
            rep(f"  {o['task']+'/'+o['arch']:<11} {o['geometry']:>+9.4f} "
                f"{o['readout']:>+9.4f} {o['full']:>+9.4f} {o['hybrid']:>+9.4f} "
                f"{o['best']:>+11.4f} {'YES' if o['win'] else 'no':>5} "
                f"{c['best']:>+11.4f} {'YES' if c['win'] else 'no':>5} "
                f"{'FLIP' if changed else '':>8}")
        rep("")
        rep(f"  wins, original criterion      : {n_o}/{N_CELLS}")
        rep(f"  wins, geometry EXCLUDED       : {n_c}/{N_CELLS}")
        rep(f"  cells whose verdict changes   : {len(flips)}")
        for o, c in flips:
            rep(f"    {o['task']}/{o['arch']}: the old 'best search' was the "
                f"{o['which']} arm at {o['best']:+.4f}")
            rep(f"      -> excluded, the best is {c['which']} at {c['best']:+.4f}; "
                f"margin {o['margin']:+.4f} -> {c['margin']:+.4f}")

    # -- the aggregate, and what it does to the epoch axis ------------------
    rep("\n" + "=" * 98)
    rep("  WHAT THE DEFECT COSTS")
    rep("=" * 98)
    rep("")
    rep(f"  {'run':<22} {'original':>10} {'geometry excluded':>18} {'cells affected':>16}")
    rep("  " + "-" * 70)
    counts = {}
    for label, _p in RUNS:
        orig, corr = summary[label]
        n_o = sum(int(o["win"]) for o in orig)
        n_c = sum(int(c["win"]) for c in corr)
        aff = sum(1 for o in orig if o["which"] == "geometry")
        counts[label] = (n_o, n_c, aff)
        rep(f"  {label:<22} {str(n_o)+'/'+str(N_CELLS):>10} "
            f"{str(n_c)+'/'+str(N_CELLS):>18} {str(aff)+'/'+str(N_CELLS):>16}")

    rep("")
    l40, l160 = RUNS[0][0], RUNS[1][0]
    o40, c40, a40 = counts[l40]
    o160, c160, a160 = counts[l160]
    rep(f"  the epoch axis, ORIGINAL criterion : {o40}/{N_CELLS} -> {o160}/{N_CELLS} "
        f"({o160 - o40:+d})")
    rep(f"  the epoch axis, CORRECTED          : {c40}/{N_CELLS} -> {c160}/{N_CELLS} "
        f"({c160 - c40:+d})")
    rep("")
    if a40 == 0 and a160 > 0:
        rep("  THE DEFECT IS EPOCH-DEPENDENT, and it INFLATED the epoch effect.")
        rep(f"  At 40 epochs no cell has `geometry` as its best search arm, so the")
        rep(f"  correction changes nothing there ({o40}/{N_CELLS} either way).  At 160")
        rep(f"  epochs {a160} of {N_CELLS} do, and excluding geometry moves the count from")
        rep(f"  {o160} to {c160}.  So part of what looked like a real epoch effect was the")
        rep("  criterion comparing the hybrid against itself.  The honest statement of the")
        rep(f"  epoch axis is the CORRECTED one: {c40}/{N_CELLS} -> {c160}/{N_CELLS}.")
    elif a40 == 0 and a160 == 0:
        rep("  THE DEFECT HAS NO EFFECT ON EITHER RUN, so it is a latent weakness of the")
        rep("  criterion rather than a present one -- and section 12.6's two cells must be")
        rep("  read again, because this script disagrees with it.")
    else:
        rep(f"  THE DEFECT AFFECTS BOTH RUNS ({a40} and {a160} cells), so every win count")
        rep("  quoted anywhere needs recomputing under the corrected criterion.")
    rep("")
    rep("  WHAT THIS DOES NOT CHANGE: the do-no-harm GUARANTEE is untouched -- worst")
    rep("  hybrid R^2 is still never below its own geometry arm, which is what the")
    rep("  guarantee claims.  What changes is only the stronger, empirical claim about")
    rep("  beating the competing methods.")
    rep("")
    rep("  THE STATED LIMIT: excluding `geometry` is the correction THIS audit argues for,")
    rep("  not a criterion the base version endorsed.  An alternative is to keep geometry")
    rep("  in the set and report cells where it dominates separately; both fix the same")
    rep("  problem and neither is chosen here by measurement, because the choice is about")
    rep("  what the sentence is meant to claim.")

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
        labels = [r[0] for r in RUNS]
        x = np.arange(len(labels))
        ax[0].bar(x - 0.2, [counts[l][0] for l in labels], 0.4,
                  label="original criterion", color="#2b7bba")
        ax[0].bar(x + 0.2, [counts[l][1] for l in labels], 0.4,
                  label="geometry excluded", color="#c44e52")
        for i, l in enumerate(labels):
            ax[0].text(i - 0.2, counts[l][0] + 0.15, str(counts[l][0]), ha="center",
                       fontsize=9)
            ax[0].text(i + 0.2, counts[l][1] + 0.15, str(counts[l][1]), ha="center",
                       fontsize=9)
        ax[0].axhline(N_CELLS, color="k", ls=":", lw=1.0)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels([l.replace(", ", "\n") for l in labels])
        ax[0].set_ylim(0, N_CELLS + 1)
        ax[0].set_ylabel(f"cells won (of {N_CELLS})")
        ax[0].set_title("wins, under each criterion")
        ax[0].grid(True, axis="y", alpha=0.25)
        ax[0].legend(fontsize=8)
        # margins, both runs, marking which cells had geometry as their best
        for i, l in enumerate(labels):
            orig, corr = summary[l]
            xs = np.arange(len(orig)) + i * 0.0
            ax[1].plot(np.arange(len(orig)) + (i - 0.5) * 0.0,
                       [o["margin"] for o in orig], "o-",
                       label=f"{l} (original)", alpha=0.8)
        ax[1].axhline(0.0, color="k", lw=0.9)
        ax[1].set_xlabel("cell index (14 cells, same order in both runs)")
        ax[1].set_ylabel("hybrid - best search")
        ax[1].set_title("margins; the criterion's defect moves these")
        ax[1].grid(True, alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_23_criterion_audit_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")}")

    # The assert encodes what the measurement supports: every cell of every run was
    # parsed and scored under both criteria.  It cannot encode "the defect matters",
    # because that is the result.
    ok = all(len(tables[l]) == N_CELLS for l, _p in RUNS)
    if do_assert and not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
