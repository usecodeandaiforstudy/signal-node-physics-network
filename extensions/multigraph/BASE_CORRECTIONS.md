# Corrections the base version's text needs, from the extension's audit

**What this is.**  The extension's `FINDINGS.md` section 12 measures the base version's
claims and finds that several of them are wrong, stale, or justified by a reason the
measurement does not support.  This file collects them as a **work list**: for each one,
the exact location, the text as it stands, what the measurement says, and the replacement.

**What this is not.**  **No base file is edited by this audit, and this file does not edit
one either.**  The base's four conclusions documents and its trainer are frozen as far as
this extension is concerned; the extension's rule is to add alongside, not to rewrite.  So
this is a *decision for the repository's owner*, not a change that has been made: apply
it, reject it, or leave the audit where it is.  Every entry is checkable line by line.

**Status vocabulary** — chosen so that "wrong" and "superseded" are not confused:

| status | means |
|---|---|
| **FALSE** | measured to be wrong as stated |
| **STALE** | was true when written, superseded since |
| **UNJUSTIFIED** | the claim may hold, but the reason given for it does not |
| **UNEXECUTED** | an instruction the base gives itself that was never carried out |
| **CLOSED** | an open item that now has an answer |
| **DEFECT** | a code or reproducibility problem, not a wording one |

**One entry is deliberately first because it is not about wording at all.**

---

## 0. `training/himal_nd.py` cannot be seeded, so its published numbers are single draws

**Status: DEFECT.**  This is the only entry that affects work already published rather
than how it is described.

`HiMALND` takes **no `seed` parameter**, and the file contains **no seeding of any kind**
(`np.random.seed`, `default_rng`, `RandomState`, `random_state` — none).  `_init_layer`
draws its centres and `W_gauss` from the global `np.random.uniform`, so two calls with
identical arguments produce **different networks** depending on what ran before.

**Consequence.**  Every number published from this trainer — `reg_01`, `reg_02`,
`nd_0*`, and the residual-mode comparison — is **one draw from an unseeded process**.
That is also the plainest explanation of the base's own note that a short check of the
direct-residual claim "gave the right direction but nowhere near that magnitude": a short
check is a different draw.

**Suggested change.**  Add `seed` to `__init__`, call `np.random.seed(seed)` at the top of
`train`, and re-run whichever numbers are quoted externally.  **Until then, any number
from this trainer should be reported with its draw count, and no two of its arms can be
compared without seeding matched.**

Evidence: `FINDINGS.md` (extension) 12.17; `extensions/multigraph/ext_32_residual_mode.py`.

---

## 1. `FINDINGS.md` line 1179 — section 11, item 2: the comparison IS constructible

**Status: FALSE.**

As it stands:

> **2. Equal-parameter-count comparison.**  Not constructible in this family (geometry
> costs `D+1` per particle against `1` for a scalar readout), so the parameter-count rule
> of section 5 can only be tested by varying the ratio.

**Measured.**  A layer with input dim `D`, output width `V` and `N` particles stores
`(D+1)*N` geometry numbers and `V*N` readout numbers, so its geometry share is
`(D+1)/(D+1+V)` — **`V` is a continuous knob on the share.**  Holding the total fixed is
then an integer problem, solved exactly with two layers: `T = (D+1+V)*N1 + 3*N2` with
`N1 = T // (D+1+V)` and `N2` the remainder over 3.  Five points were built at **the same
single-channel total of 312** across a share sweep of 83% to 26%.

**Suggested replacement.**  Replace the item with the construction and what it bought:

> **2. Equal-parameter-count comparison.**  **Constructible, contrary to an earlier
> version of this item.**  A layer's geometry share is `(D+1)/(D+1+V)`, so the output
> width `V` is a knob on the share; the total is then held fixed exactly by a two-layer
> construction.  At one fixed total the search arms move by up to `0.319` across the share
> axis while the calibrated arm moves by `0.091`, and the calibration's margin **grows**
> as the allocation worsens — **so the calibration substitutes for a good allocation.**
> The parameter-count rule of section 5 therefore holds for the calibrated arm and not
> for the search arms.

Evidence: `FINDINGS.md` (extension) 12.15; `outputs/ext_30_equal_params.txt`.

---

## 2. `FINDINGS.md` lines 1193-1195 — section 11, item 5: right concern, wrong reason

**Status: UNJUSTIFIED.**

As it stands:

> **5. Reservoir variant.**  On an excitable medium the only trainable part is the
> readout, which is section 6's recommendation taken to its limit — but it requires
> giving up movable centres and per-particle widths.

**Measured.**  With the geometry never trained — no probes, no gradient, one closed-form
solve — best-of-R redraws **beats the geometry arm, which spends the entire probe budget,
in 9 of 14 cells**, and loses to search-plus-solve in 13 of 14.  In about half the cells
the redraw *pick* is **worse than not picking at all**, and doubling the validation slice
helps (`+0.0204` mean).  So the movable centres are not earning their keep because a
frozen geometry is too weak; the binding constraint is **selecting among instances**.

**Suggested replacement.**

> **5. Reservoir variant.**  Tested (`FINDINGS.md` 12.14).  A frozen geometry with a single
> closed-form solve beats the geometry arm, at zero probe cost, in most cells — so "only
> the readout is trainable" is not by itself disqualifying.  What it loses to is search
> PLUS solve, and the reason is **model selection**: a reservoir's one knob is redrawing
> the geometry, and the redraw has to be picked, which on a quarter-of-training slice is
> worse than not picking in about half the cells.  **The gap is a selector, not movable
> centres** — and the same selector failure appears in the operator suite (12.21 / ext_21).

Evidence: `FINDINGS.md` (extension) 12.14; `outputs/ext_29_reservoir.txt`.

---

## 3. `FINDINGS.md` lines 1201-1204 — section 11, item 6(b): the counterexample exists

**Status: STALE.**  The base's own section 6.7 recorded it; the index never caught up.

As it stands, in item 6:

> (b) the gate validates in-distribution, and although the `shift` task showed the
> advantage transfers, a task where the gate *accepts* and the arm still loses has still
> not been found — that remains the strongest missing counterexample

**Measured.**  Section 6.7 already reports it: on `hetero`/G one calibration was
**accepted**, it **improved** the arm, and the arm **still lost to `full`** — 1 accepted
of 120 cells.  The two sentences have been in the same document since the 3-seed sweep.

**Suggested replacement.**

> (b) the counterexample to the *stronger* claim exists and is recorded in section 6.7:
> `hetero`/G, where one calibration was accepted, improved the arm, and the arm still lost
> to `full`.  **It is not a counterexample to the guarantee**, which is about not
> calibrating rather than about other arms.  On the guarantee itself, an extension audit
> (12.18) has now located the gate's **decision boundary** — where the accept rate is
> slice-size-dependent by `0.48` — and found **one cell of eleven with the counterexample's
> shape** at that boundary, without yet isolating the calibration from the trajectory
> divergence that a second training run introduces.

Evidence: `FINDINGS.md` (extension) 12.1a and 12.18; `outputs/ext_33_gate_power.txt`.

---

## 4. `FINDINGS.md` lines 1208-1212 — section 11, item 7 is stale and self-contradicting

**Status: STALE.**

As it stands:

> **7. Task sweep coverage.**  Seven tasks, two architectures, 2 seeds.  The decisive
> cells are the nine that are won; the five losses all reduce to the geometry arm, so
> they bound the risk but do not discriminate between the mechanisms.

**Measured.**  It is **3 seeds**, and the same section's item 3 says the seed upgrade is
done.  The counts are also different from those quoted: at 9 seeds the headline is
**8 of 14**, not nine.

**Suggested replacement.**  Delete the item, or restate it as what is genuinely still
narrow: *the losses bound the risk without discriminating between mechanisms* — which is
a limitation, not a coverage gap, and item 3 already covers the seeds.

Evidence: `FINDINGS.md` (extension) 12.1a and 12.7.

---

## 5. `FINDINGS.md` lines 1176-1178 — section 11, item 1's open question, answered

**Status: CLOSED**, with an answer that is more specific than yes or no.

As it stands:

> Still to do: check whether the hybrid advantage at 3 000 measurements holds for a
> larger particle budget.

**Measured.**  The question needed restating before it could be asked, and then it was
answered: the comparison's **window moves with the architecture**, because `effective_budget`
derives the per-epoch budget from the network size.  With the budget **held fixed**, two
independent families both say the low-budget advantage **does not survive** four times the
particles — **at the default ridge scale**.  Changing only that scale (`closed_form_lam="auto"`)
**reverses it**: the margin is positive at every size and rises with it.

**Suggested replacement.**  Replace the tail with the restated question and its answer:

> Settled (extension 12.9-12.13), and the answer is conditional on one constant.  At the
> default `closed_form_lam=1e-3` the low-budget margin goes from `+0.41` at 52 particles
> to `-0.02` at 208, in two independent families, with the budget held fixed.  At
> `lam="auto"` it is **positive at every size and rises to `+0.60`**.  **`1e-3` is the only
> value on a five-point grid that collapses it.**  The question as originally phrased could
> not be asked, because the comparison window itself scales with the architecture.

Evidence: `FINDINGS.md` (extension) 12.9-12.13; `outputs/ext_25_budget_matched.txt`,
`outputs/ext_26_base_family.txt`, `outputs/ext_27_nonmonotone.txt`.

---

## 6. `FINDINGS.md` line 1136 — section 9: `train_01` should be superseded, not regenerated

**Status: UNEXECUTED**, and the instruction as written would close nothing.

As it stands, about `experiments/train_01_grouped_fd.py` and `train_02_grouped_spsa.py`:

> Their previously reported numbers must be regenerated.

**Measured.**  Both were regenerated.  `train_02`'s numbers genuinely vary with `K` and it
**passes** — grouped SPSA beats exact per-parameter FD at the same forward budget, and at
`K=2` converges 21x sooner.  **`train_01`'s regeneration is vacuous**: the grouping is by
consecutive particle blocks, so the flattened parameter visit order is identical for every
`K` and the results are bit-identical **by construction**.  Its PASS verdict therefore
tests the loop nesting, not particle independence — and the extension adds a **second,
independent reason** why it cannot, one that survives the index-convention fix.

**Suggested replacement.**

> `train_02`'s numbers were regenerated and support the claim (grouped SPSA matches or
> beats exact FD at equal budget; see extension 12.8).  **`train_01` should be marked
> SUPERSEDED rather than regenerated**: its grouping preserves the parameter visit order,
> so every `K` is bit-identical by construction and the verdict does not test what it
> claims.  The independence claim is measured in section 7 and the front-page line about
> `train_01` should point there instead.

Evidence: `FINDINGS.md` (extension) 12.8; `outputs/abl_06_task_sweep_s9.txt` (unaffected).

---

## 7. `FINDINGS.md` lines 901-903 — section 6.11: the boundary task now exists

**Status: CLOSED**, and the closing result is the one thing that was missing.

As it stands:

> the third attempt **did not answer its own question** (a first-order failure is
> detectable at any slice size, so it does not probe the gate's statistical power — that
> needs a task sitting near the decision boundary, which was not constructed)

**Measured.**  The task is constructed (extension 12.18).  **Noise cannot construct the
boundary** — raising it drives the accept rate to `0/120`, first-order like `10d`.  **The
validation-slice size can**: the accept rate moves from `0.025` to `0.508` as the slice
shrinks from 128 samples to 16 (**spread 0.483**), which is the property the paragraph
above says was never tested.  The base's own `10d` sweep saw no movement because that is a
property of `10d`, not of the slice size.  At that boundary, **1 of 11 cells** has the
counterexample's shape (the gate accepted, and the arm ended below its own geometry arm).

**AND THE COUNTEREXAMPLE IS FOUND, one subsection later (12.19).**  Wrapping
`solve_readout_closed_form` — which `train_model` calls as a bare module-global name, so
**no base file is edited** — yields the device's **test** `R^2` immediately before and after
each deployment inside a single run.  **55 of 116 deployed calibrations left it lower than
it was immediately before.**  That is the counterexample, with the trajectory divergence
removed by construction rather than by matching.

Two qualifications travel with it, and neither is optional.  **Every cell still ends with a
positive margin**, so the dips are **recovered** and the per-run recommendation is
unaffected.  And the harmful range reaches down to `-0.0003`, so **part of the 55 is
numerically indistinguishable from no change** — the guarantee is about the sign, and
lumping `-0.0003` together with `-0.0502` would exaggerate it.

**The cell 12.18 flagged is exonerated**: its three deployments all helped or were neutral,
so its negative margin was the divergence 12.18 named and could not exclude.

**Suggested replacement.**  Add after the paragraph:

> **Closed by extension 12.18 and 12.19.**  A task near the decision boundary was
> constructed: **the knob is the validation-slice size, not the task's noise**, and at the
> boundary the accept rate is slice-size-dependent by `0.48`.  The isolation 12.18 said was
> still missing was then supplied, and it **finds the counterexample**: 55 of 116 deployed
> calibrations left the device's test `R^2` lower than immediately before, on the same
> network at the same moment.  **Every cell still ends ahead**, so the dips are recovered;
> and the cell 12.18 flagged is **exonerated**, its three deployments having done no harm.

Evidence: `FINDINGS.md` (extension) 12.18 and 12.19; `outputs/ext_33_gate_power.txt`,
`outputs/ext_34_deployment_harm.txt`.

---

## 8. `THEORY.md` section 7.3, `FINDINGS.md` line 1143 and `training/himal_nd.py` lines 129-137 — the "5x" is refuted, not merely unreproduced

**Status: FALSE as stated; the default it justifies is right for a better reason.**

As it stands in the code:

> `"direct"` -- original y minus the accumulated RAW scalar output `h_m[:, 0]` of the
> previous layers (THEORY.md section 7.3 reports this as ~5x better, but that comparison
> has not been reproduced in this codebase, so `"proto"` remains the default rather than
> switching on an unverified claim)

**Measured.**  The published pair `+0.064` (direct) against `+0.013` (proto) comes from
**two different programs**: `reg_01` monkey-patches two of `HiMALND`'s methods at runtime
and uses a different kernel family, and `reg_02` does not import `HiMALND` at all.  With
the shipped trainer, matched initialisation and `residual_mode` as the **only** difference,
the direct arm's depth gain is **negative** (`-0.0284`) against proto's `+0.0069` — a ratio
of `-4.12x` against the claimed `4.9x`.  **The mechanism**: `direct` fits layers 2-3 to the
residual of the raw output while the inference path sums the **proto** predictions, so the
deeper layers are trained against a residual the inference never reconstructs.  It is a
**mismatched** target, not a better one.

**Suggested replacement** (the comment, and the matching note in `THEORY.md` 7.3):

> `"direct"` -- forms the layer-2/3 residual from the raw scalar output `h_m[:, 0]`.
> **Measured and rejected** (extension 12.17): with matched initialisation its depth gain
> is negative where `"proto"`'s is positive, because the raw output is not what the
> inference path sums.  `"proto"` is the default not because the alternative is unverified
> but because the alternative is **measured to be harmful** at the configuration the
> original claim was made about.

Evidence: `FINDINGS.md` (extension) 12.17; `outputs/ext_32_residual_mode.txt`.

---

## 9. `THEORY.md` lines 647-654 (Method C) — an instruction the base gives itself that was never executed

**Status: UNEXECUTED.**

As it stands:

> This method is **not implemented**.  The functions that would realise it ... are defined
> and re-exported, but there is **no call site anywhere in the repository**, and no
> experiment trains with them.  The same applies to the receiver-ADC path ... **Either
> implement and measure this method, or remove it from the feature list.**

**Verified, with one precision.**  The functions do call **each other**: `analog_gradient_mac`
is called from `analog_weight_gradient`, and `analog_adc` from `analog_loss_readout` — but
**neither enclosing function is called anywhere outside `components.py` and `__init__.py`**.
So the accurate statement is *an internally-connected island with no entry point*, not *no
call site anywhere*; the conclusion is the same and a reader checking the wording literally
would wrongly think the note stale.

**Suggested change.**  Execute the either/or.  If removing: drop Method C from the feature
list and the "four tiers" phrasing in **`README.md` line 42** (already annotated) and
**line 46** (NOT annotated — see the next entry).  If implementing: it is a project, not a
paragraph, and the note should say so.

Evidence: `FINDINGS.md` (extension) notes on `analog_*` reachability; verified by
repository-wide grep.

---

## 10. `README.md` line 46 — the one feature bullet with no inline correction

**Status: STALE / uncorrected.**

As it stands:

> - 🧠 **Physically Deployable**: Every operation (distance, Gaussian kernel, MVM,
>   **gradient estimation**) has a corresponding analog circuit; fully analog inference
>   with zero digital dependency

**Why it is listed.**  The status box (line 21) and five of the six feature bullets carry
inline corrections; **this one does not.**  It still asserts that gradient estimation has a
corresponding analog circuit, which is the claim entry 9 shows is dead code.

**Suggested change.**  Add the same inline note the neighbours carry:

> ... MVM, gradient estimation) has a corresponding analog circuit ... *(the gradient chain
> is dead code — see the status box and `FINDINGS.md` §9)*

Evidence: this file's entry 9; `README.md` lines 21, 42, 49 for the pattern already used.

---

## 11. `FINDINGS.md` section 11's preamble — the index is still missing items

**Status: STALE, and it is the meta-entry.**

The section's own preamble says it was rebuilt because "this is the index a reader uses to
choose the next experiment, and **choosing from a stale index is how the wrong experiment
gets run**".  Three genuinely-open items were still absent from it, and are picked up by
this file instead:

| missing from section 11's index | where the base says it | this file |
|---|---|---|
| Method C: "either implement and measure this method, or remove it" | `THEORY.md` Method C, `FINDINGS.md` §9 | entry 9 |
| the "5x" direct-residual claim is unreproduced | `THEORY.md` 7.3, `himal_nd.py` 129-137 | entry 8 |
| a task near the gate's decision boundary "was not constructed" | `FINDINGS.md` 6.11 | entry 7 |

**Suggested change.**  Add them to section 11 so the next reader does not have to find them
by grep.

---

## Suggested order of work

1. **Entry 0** (the unseeded trainer) — it is the only one that affects published numbers
   rather than their description, and it is a code change with a clear test (two runs,
   identical arguments, identical result).
2. **Entry 9** (Method C's either/or) — it is the largest gap between what the project
   advertises and what exists, and both branches are cheap.
3. **Entries 1, 2, 3, 4, 8, 10, 11** — text, each with the replacement written above.
4. **Entries 5 and 7** — closing an open item.  **Entry 7 is fully closed and its result is
   the most consequential of the twelve**: the counterexample the base searched for through
   three designs was found (55 of 116 deployments), and the one cell that looked like a
   counterexample was exonerated.  **The replacement text above carries both halves**, and
   applying only the first would replace an understatement with an overstatement.

## What this file does not ask for

It does not ask for a new experiment.  Every correction above is backed by a measurement
that already exists, and **only one entry has work genuinely unfinished: entry 0**, which
needs a code change and a re-run of whichever numbers are quoted externally.  The other
eleven are text, and the two that were open questions — entries 5 and 7 — have been
answered rather than left as measurements to make.

It also does not ask the base to adopt the extension's conclusions wholesale.  Where the
audit found the base right — the do-no-harm guarantee surviving a boundary the base never
built, the headline surviving the ridge scale at `8/14` either way, `train_02` passing on
regeneration — those are recorded in `FINDINGS.md` section 12 and require no change here.
