# Contributing

This repository is a measurement record before it is a codebase, and the conventions below exist
because breaking them has already produced wrong numbers here. Each one is stated with the
failure it prevents, because the failure is the reason for the rule.

## The one rule

**A number may not be written down until it has been measured, and it may not be cited until an
artifact exists that a reader can open.** This is the rule the others serve.

Concretely:

* **Every script writes an artifact** to `outputs/`, and the artifact is tracked in git if a
  document cites it. `.gitignore` un-ignores exactly those; adding a citation means adding the
  rule.
* **Every script takes `--assert`** and exits non-zero when its own thresholds are violated.
  Without it a script only *prints* its numbers, so a rewrite can break a claim and still exit 0 —
  which is why CI runs `sub_01_diffusion_verify.py --assert` before anything else.
* **Every script is listed in the reproduce block** of the document that reports it, next to an
  artifact row. That block is the contract; a script that is not in it is not reproducible from
  the document.
* **A number a document cites must exist in that document's own text.** Every scientific-notation
  figure in the front-page documents is checked against `FINDINGS.md` and the extension's for
  exactly this reason: writing a number from memory, or rounding one, has produced figures that
  appeared nowhere else.

## Layout, and the one structural rule

```
experiments/            the base version's scripts (abl_*, sub_01, nd_*, cifar_*, ...)
extensions/multigraph/  the extension: its own scripts, its own FINDINGS.md
training/ simulator/ physical_simulator/ substrate_diffusion/ network/   the library
rbf_network/ forward_training/   the v1 tree, kept as provenance
```

**Do not modify the base version to add new work.** New results go in `extensions/`, with their
own document. The base is edited only when a measurement **overturns one of its own conclusions**,
and then the conclusion itself is corrected where it is stated — not annotated with a note saying
it used to be different. A reader has no use for the history; what they need is the version that
survived, with its number.

## Reproducibility

* **Seed everything that draws.** `np.random.seed(...)` for the global generator **and** a seeded
  `default_rng` for local use. The global one is not optional: `features()` builds a real device
  layer and the device draws from `np.random`, so a script that seeds only its own generator still
  moves between runs — 24 scripts here had exactly that defect.
* **`torch.manual_seed(...)` too**, for anything that trains. Data-split seeding is not model
  seeding: the CIFAR line seeded only the split, and every number it published was therefore a
  single draw from an unseeded process. Fixing it moved those numbers by up to 6.6 points, in both
  directions, so the reported differences between its architectures were smaller than their own
  seed sensitivity.
* **Record which seed produced a report.** A number whose seed is not in the artifact beside it is
  not reproducible by the person reading it.
* **Never leave a stale report on disk.** Delete the artifact before a run that writes it, or a
  syntax error leaves the previous run's numbers looking like this run's.

## Measurements, and how to report them

* **State the cost in the project's unit: measurements.** Forward evaluations and read-out
  calibrations are two different units and are never added.
* **Report spreads, not points.** A null's seed-to-seed spread has been larger than the effect
  being claimed; more seeds *estimate* a spread rather than shrinking it, so "we ran more seeds"
  is not an answer by itself.
* **Ship the control.** A learning-floor control, a null arm, or a paired comparison belongs in
  the script, and a comparison of two arms that both failed to learn is not a result.
* **A negative result is a result.** The deep architecture being beaten by a zero-forward
  baseline, the calibration not helping the deep path, and 55 of 116 deployed calibrations leaving
  the arm worse are all in the documents, with their numbers.

## When something is wrong

* **Correct it where it is stated.** Replace the wrong conclusion with the measured one, at the
  place a reader will meet it. Do not add a note explaining what the text used to say: a document
  that narrates its own edits makes the reader reconstruct the current state, and the current
  state is the only thing they came for.
* **Say what the measurement cost.** A defect is worth recording with its consequence — "a stale
  report survived a syntax error and was read as this run's numbers" is worth more than "reports
  should be deleted first".
* **Distinguish a limitation from a gap.** "We did not measure this" and "we measured this and it
  is out of reach" are different statements, and the second is worth more.
* **Verify a claim before writing it down, and verify it a second way before publishing it.**
  Several conclusions here were reached by one route and did not survive a second: a comparison
  run through two different programs rather than one trainer with one switch, a measurement taken
  at a budget where the arm being measured had not started, an index convention that mixed two
  strides in one list. Each was caught by re-measuring, not by re-reading.

## Practical

```bash
pip install -r requirements.txt
python experiments/sub_01_diffusion_verify.py --assert     # ~4 s, the cheap gate
python extensions/multigraph/ext_01_baseline.py --assert   # the extension's correctness anchor
```

Both are what CI runs. The rest of the reproduce block takes hours in total and is meant to be run
per-claim rather than wholesale.

## Scope

The repository is a **toy-scale** study. Every result is on small tasks, on simulated devices, and
with hyper-parameters that were not tuned. **Nothing here has been fabricated in hardware.**
Contributions that extend the substrate coverage, replace a simulation with a measurement, or
falsify a claim are the ones this project is built to accept.
