# Contributing

This repository is a measurement record before it is a codebase, and the conventions below
exist because breaking them has already produced wrong numbers here. They are cheap to follow
and each one is traceable to a specific failure in `FINDINGS.md` §9 or the extension's defect
list.

## The one rule

**A number may not be written down until it has been measured, and it may not be cited until an
artifact exists that a reader can open.** Six claims on the README were stated before they were
measured and are now retracted with their reasoning; the status box at the top of the README is
the result.

Concretely:

* **Every script writes an artifact** to `outputs/`, and the artifact is tracked in git if a
  document cites it. `.gitignore` un-ignores exactly those; adding a citation means adding the
  rule.
* **Every script takes `--assert`** and exits non-zero when its own thresholds are violated.
  Without it a script only *prints* its numbers, so a rewrite can break a claim and still exit
  0 — which is why CI runs `sub_01_diffusion_verify.py --assert` before anything else.
* **Every script is listed in the reproduce block** of the document that reports it, next to an
  artifact row. That block is the contract; a script that is not in it is not reproducible from
  the document.

## Layout, and the one structural rule

```
experiments/            the base version's scripts (abl_*, sub_01, nd_*, cifar_*, ...)
extensions/multigraph/  the extension: its own scripts, its own FINDINGS.md
training/ simulator/ physical_simulator/ substrate_diffusion/ network/   the library
rbf_network/ forward_training/   the v1 tree, kept as provenance
```

**Do not modify the base version to add new work.** New results go in `extensions/`, with their
own document. The base is edited only when a measurement **overturns one of its own
conclusions** — §9 of `FINDINGS.md` is where those edits are recorded, and each one says what
was believed, what was measured, and what replaced it.

## Reproducibility

* **Seed everything that draws.** `np.random.seed(...)` for the global generator **and** a
  seeded `default_rng` for local use. The global one is not optional: `features()` builds a real
  device layer and the device draws from `np.random`, so a script that seeds only its own
  generator still moves between runs — that is defect 58, and 24 scripts had it.
* **`torch.manual_seed(...)` too**, for anything that trains. Data-split seeding is not model
  seeding; the cifar line seeded only the split and every number it published was a single draw
  from an unseeded process, which moved those numbers by up to 6.6 points when fixed.
* **Record which seed produced a report.** A number whose seed is not in the artifact beside it
  is not reproducible by the person reading it.
* **Never leave a stale report on disk.** Delete the artifact before a run that writes it, or a
  syntax error leaves the previous run's numbers looking like this run's. That mistake is in the
  defect list twice.

## Measurements, and how to report them

* **State the cost in the project's unit: measurements.** Forward evaluations and read-out
  calibrations are two different units and are never added.
* **Report spreads, not points.** A null's seed-to-seed spread has been larger than the effect
  being claimed; more seeds *estimate* a spread rather than shrinking it, so "we ran more seeds"
  is not an answer by itself.
* **Ship the control.** A learning-floor control, a null arm, or a paired comparison belongs in
  the script, and a comparison of two arms that both failed to learn is not a result.
* **A negative result is a result.** The deep architecture being beaten by a zero-forward
  baseline, the calibration not helping the deep path, and 55 of 116 deployed calibrations
  leaving the arm worse are all in the documents, with their numbers.

## When something is wrong

* **Correct it where it is stated**, and record the correction in §9 with the measurement that
  forced it. A retraction is more valuable than a quietly edited paragraph.
* **Add the defect to the list** with what it cost. The extension's list is at the end of its
  `FINDINGS.md`; the base's is §9.
* **Distinguish a limitation from a gap.** "We did not measure this" and "we measured this and
  it is out of reach" are different statements, and the second is worth more.

## Practical

```bash
pip install -r requirements.txt
python experiments/sub_01_diffusion_verify.py --assert     # ~4 s, the cheap gate
python extensions/multigraph/ext_01_baseline.py --assert   # the extension's correctness anchor
```

Both are what CI runs. The rest of the reproduce block takes hours in total and is meant to be
run per-claim rather than wholesale.

## Scope

The repository is a **toy-scale** study. Every result is on small tasks, on simulated devices,
and with hyper-parameters that were not tuned. **Nothing here has been fabricated in hardware.**
Contributions that extend the substrate coverage, replace a simulation with a measurement, or
falsify a claim are the ones this project is built to accept.
