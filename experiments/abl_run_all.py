"""
abl_run_all.py - reproduce the evidence behind FINDINGS.md
==========================================================
Project: singal-node-physics-network

FINDINGS.md cites a specific report for every claim.  This runner regenerates that
report set in dependency order, so "reproduce the results" is one command instead
of fifteen.

    S (sub_01) substrate verification : is the kernel physics?     ~4 s
    A (abl_01) capacity split         : where does the capacity live?
    B (abl_02) locality               : is particle independence real?
    C (abl_03) mismatch               : which non-ideality damages what?
    D (abl_04) device robustness      : does calibration survive the device?
    E (abl_05) twin error             : does it survive a WRONG digital twin?
    F (abl_06) task sweep             : does the rule hold on other tasks?
    G (abl_07) convergence            : epochs x budget, schedule, gate power

Order matters for cost, not for correctness: C reuses A's cached models, so A runs
first.  Everything else is independent.

The CANONICAL settings per script (below) are the ones that produced the reports
FINDINGS cites, so a default run REPRODUCES them rather than producing new variants.
A full pass is expensive -- roughly 6-7 hours, dominated by C (2 h) and F (1.7 h) --
which is why `--quick` exists for CI.

`--quick` runs every stage with tiny settings purely to prove the pipeline still
works end to end (~3 min).  It is a REGRESSION guard, not a reproduction: the
numbers it prints are not the numbers in FINDINGS.  The sharp guard is
`sub_01_diffusion_verify.py --assert`, which enforces the physical thresholds that
sections 2, 3 and 8 rest on and exits non-zero if one breaks.

Child stdio is inherited rather than piped on purpose: under the DSH sandbox piped
stdio can fail with EPERM, and this runner does not need to parse the children's
output - each script writes its own report under outputs/.

Usage:
    python experiments/abl_run_all.py                 # reproduce everything (~6.5 h)
    python experiments/abl_run_all.py --quick          # smoke, for CI (~3 min)
    python experiments/abl_run_all.py --assert-only    # just the threshold checks
    python experiments/abl_run_all.py --only 4         # a single stage
    python experiments/abl_run_all.py --list           # show the plan
"""

import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

# (index, tag, script, description, canonical args, quick args, takes --substrate)
STAGES = [
    (0, "S", "sub_01_diffusion_verify.py",
     "Substrate verification: kernel vs PDE, superposition, cost model",
     ["--assert"], ["--assert"], False),
    (1, "A", "abl_01_capacity_split.py",
     "Capacity split: geometry (nodes) vs readout at equal measurement cost",
     ["--archs", "G,F,F2", "--arms", "geometry,weights,hybrid,hybrid_once,full",
      "--seeds", "3", "--epochs", "80", "--no-attribute",
      "--budget-mode", "equal_probes"],
     ["--quick"], True),
    (2, "B", "abl_02_locality.py",
     "Locality: measured activation coupling + the price of parallel updates",
     # M2/M3 need >= 1 sweep/epoch: at the default budget for arch M the full arm
     # gets 150 probes against 444 parameters (0.34 sweeps) and the script reports
     # the comparison as INCONCLUSIVE.  These are the settings that make section 7's
     # numbers reproducible.
     ["--seeds", "1", "--epochs", "40", "--budget", "600"], ["--quick"], True),
    (3, "C", "abl_03_mismatch_sensitivity.py",
     "Damage sensitivity: which non-ideality hits which parameters",
     ["--seeds", "3", "--epochs", "60", "--reps", "3"], ["--quick"], True),
    (4, "D", "abl_04_calibration_robustness.py",
     "Calibration vs device non-idealities (sensor noise, positioner)",
     ["--seeds", "3", "--epochs", "40", "--calib-every", "10"], ["--quick"], True),
    (5, "E", "abl_05_twin_error.py",
     "Calibration vs a WRONG digital twin",
     ["--seeds", "3", "--epochs", "40", "--calib-every", "10"], ["--quick"], True),
    (6, "F", "abl_06_task_sweep.py",
     "Task sweep: does the rule hold on other tasks?  (cache-resumable)",
     ["--seeds", "3", "--epochs", "40", "--tag", "s3"],
     ["--quick", "--tasks", "2d"], True),
    (7, "G", "abl_07_convergence.py",
     "Convergence grid, calibration schedule, gate sample size",
     ["--seeds", "1"], ["--quick"], True),
]

assert_only_stage = 0


def main():
    # Line-buffer our own output.  Without this, `python abl_run_all.py > log`
    # block-buffers the parent's stage results, so a run that is interrupted (or
    # merely watched) shows the children's output but none of the summary -- which
    # is exactly what happened the first time this runner was used in anger.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny settings everywhere: a pipeline smoke test, NOT a "
                         "reproduction (its numbers are not the ones in FINDINGS)")
    ap.add_argument("--assert-only", action="store_true",
                    help="run only the stage that enforces thresholds (fast; this "
                         "is the guard CI should use)")
    ap.add_argument("--only", type=int, default=None,
                    choices=[s[0] for s in STAGES], help="run a single stage")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--substrate", default="circuit",
                    choices=["circuit", "diffusion"],
                    help="medium for the stages that take one.  abl_03 reuses "
                         "abl_01's cache, so substrate/seeds/epochs/budget must "
                         "match between them for the cache to hit")
    args = ap.parse_args()

    if args.list:
        print("  idx  tag  script                              quick?  substrate?")
        for idx, tag, fname, desc, canon, quick, takes_sub in STAGES:
            print(f"  {idx:>3}  {tag:>3}  {fname:<35} {'yes' if quick else 'no ':>6}"
                  f"  {'yes' if takes_sub else 'no':>10}")
            print(f"           {desc}")
            print(f"           canonical: {' '.join(canon)}")
        return 0

    mode = "QUICK (smoke)" if args.quick else "CANONICAL (reproduction)"
    if args.assert_only:
        mode = "ASSERT ONLY"
    print("=" * 70)
    print("  Evidence reproduction for FINDINGS.md")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}   {mode}"
          f"   substrate={args.substrate}")
    print("=" * 70)

    status = {}
    t_all = time.time()

    for idx, tag, fname, desc, canon, quick, takes_sub in STAGES:
        if args.assert_only and idx != assert_only_stage:
            continue
        if args.only is not None and args.only != idx:
            continue

        path = os.path.join(BASE, fname)
        if not os.path.exists(path):
            print(f"  [skip] missing {fname}")
            status[tag] = "MISSING"
            continue

        extra = quick if args.quick else canon
        cmd = [sys.executable, path] + list(extra)
        if takes_sub and not args.quick:
            cmd += ["--substrate", args.substrate]

        print(f"\n{'-'*70}\n  [{tag}] {desc}\n  $ {' '.join(cmd)}\n{'-'*70}")
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        status[tag] = "OK" if rc == 0 else f"FAILED(rc={rc})"
        print(f"  [{tag}] {status[tag]} in {time.time()-t0:.0f}s")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for idx, tag, fname, desc, canon, quick, takes_sub in STAGES:
        if tag in status:
            print(f"  {tag}: {status[tag]:<14} {desc}")

    failed = [t for t, s in status.items() if s != "OK"]
    print("\n  Reports land in outputs/ ; the tracked evidence set is")
    print("  outputs/abl_*.txt|png and outputs/sub_01_* (see .gitignore).")
    print(f"  Substrate: {args.substrate}     Total: {time.time()-t_all:.0f}s")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
