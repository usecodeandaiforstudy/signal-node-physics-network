# Operator-only training of a parameter network: extension findings

**Status:** the extension is implemented and measured.  It is a **separate** result
set from the base version: the base `FINDINGS.md` is not extended, and nothing here
changes a base conclusion.

**Scope rule, honoured throughout:** every extension script depends on the base
version READ-ONLY (`substrate_diffusion`, `experiments/abl_common.py`).  No base
file is modified in any commit that touches `extensions/`.

**Read this before quoting a number.**  The extension has two halves that do NOT
share a task:

* **half A (ext_01-05)** uses a synthetic multi-output target with the geometry
  held fixed, to isolate the readout.  Its numbers are comparable only to each
  other.
* **half B (ext_06-08)** uses the project's OWN 4D regression task.  ext_07's
  finite-difference arm is the base version's own trainer, so that comparison is
  authoritative; ext_08 extends it.

Neither half is comparable to the base version's headline numbers, because neither
reproduces the base's full setup.

---

## 0. The two halves, and what each establishes

**Half A -- parameter economy.**  Writing the readout as `W = C . W_g` with
`k < m` stores fewer parameters, and the base version measured readout
parameters per training sample as the quantity that decides whether closed-form
calibration works.

> **Supported:** with `m = 8`, `k >= 3` reduces the readout from 320 parameters
> (1.67x per sample) to 192 (1.00x) or 144 (0.75x) at NO measurable accuracy cost
> -- both substrates, all four device levels, geometry fixed and geometry trained.

**Half B -- the operators.**  The base version established that source strengths are
a *calibration* quantity with an exact operator rather than a search quantity.  Half
B supplies the same thing for the geometry, and then a trainer that uses only
operators.

> **Supported:** a node's centre and width can be SOLVED in closed form from its
> response (ext_06); a network can be trained **without gradients, finite
> differences or a schedule** by using those operators (ext_07, and see 5.1 for
> which of its steps actually carry the result); the cost advantage survives a
> 300,000-evaluation search (ext_08); the operator's device-side use -- measuring
> where a source actually sits -- is exact at the minimum probe count (ext_09); the
> advantage **generalises to a second task, to both a wider and a deeper
> architecture, to the base version's own adversary tasks, and to the second
> SUBSTRATE** (ext_10, ext_13, ext_17); the placement rule is **bounded** and nearly
> optimal at the operating point (ext_14); and a wrong twin, invisible to the
> trainer's own gate, is **repaired by measuring the placement and inverting it PER
> NODE** (ext_11, ext_12, ext_15).
>
> **Not supported, and this list is the one to read against the claims:** `steps`,
> where the single greedy run lands `0.234` below its own restarts (5.11); `10d`,
> where no arm learns and the results are therefore silent (5.5); pure `noise`, where
> this method is the most confidently wrong (5.8); and "any physical system", where
> two substrates built to the same kernel is not "any" (5.12).

---

## 1. The correctness anchor (ext_01)

`ext_01` asserts that the extension reproduces the base version in the degenerate
configuration (`k = m`, shared centres *and* widths, `C = I`).  It passes, and the
anchor is what makes every half-A number usable:

| test | result |
|---|---|
| T1 degenerate identity | **bit-identical** where a stage has `n_vars == 1` (both substrates); float32 agreement (`~1.8e-07`) where `n_vars > 1` |
| T2 `C.W` algebra | correct to float32 (`~3.5e-07`) with per-stage random `C` |
| T3 negative control | a permuted wiring IS rejected, executed 4x (not skipped) |
| T4 base guards | `selftest`, `selftest_cross_substrate` pass |
| T5 cache isolation | base arch keys and task strings are rejected |
| parameter bookkeeping at `k = m` | matches the base **exactly** (G: 204/92, F2: 224/52) |

---

## 2. Half A -- the rank budget is essentially free

| experiment | question | answer |
|---|---|---|
| `ext_02` | fixed geometry, circuit cell | boundary at **k = 3**; `k=4` gap `+0.0003`, `k=3` gap `-0.002`; `k=2` fails at `-0.035` |
| `ext_03` | geometry TRAINED against the readout | same boundary; `k=4` gap `-0.0002` (sem 0.0004), `k=3` `-0.0027` (sem 0.0014), `k=2` `-0.0357` FAILS |
| `ext_04` | calibration size swept 16 -> 192 samples (20x -> 1.67x) | `k=4` and `k=3` within `+-0.002` of full rank at **every** size |
| `ext_05` | diffusion cell, device levels none/low/medium/high | boundary **k = 3 at every level**; the `k=3` gap is `-0.0034 ... -0.0023` while the level itself costs the full-rank arm **0.136** |

Two things follow that the extension did not have before:

* **The boundary is a property of the architecture, not of a medium or a device.**
  It is `k = 3` on the circuit cell, on the diffusion cell, at all four of the base
  version's device presets, and whether or not the geometry is trained.
* **Geometry training and the rank constraint do not compete.**  The paired gap is
  flat over 60 epochs (`k=4`: `+0.0000 -> -0.0002`; `k=3`: `-0.0024 -> -0.0027`;
  `k=2`: `-0.0311 -> -0.0357`).  If they competed, the gap would widen as the
  geometry moved; it does not.

`ext_03` also carries an **anti-vacuity check**, because "it survived geometry
training" is meaningless if the geometry never moved: movement is reported
(`0.0699` relative against a floor of `0.05` declared in advance) together with the
gain it produced (`+0.052` R^2 over 60 epochs).

---

## 3. Half A does NOT deliver

These are as much the result as section 2 is.  Each was measured, and three of them
are directions a reader would reasonably expect the idea to pay off in.

**3.1 No measurement saving.**  Under the base version's own cost model
(`abl_common.readout_calibration_cost`) a calibration costs `n_train * sum_k N_k` --
one reading per **source** per sample -- which is independent of `m` and of the
readout's rank, because solving for `W` needs the feature matrix `Z` and `Z` costs
`N` readings per sample however low the rank is.  The extension's win is a
**parameter** win, not a measurement win.

**3.2 The frozen `+-1` code is not viable.**  `ext_02`: at `k = 4` it costs
`-0.111` R^2 (freq 1.0) to `-0.432` (freq 0.5), with a spread across codes as large
as the effect.  This refutes design-memo section 4.1's realization **(b)** -- "no
combination matrix anywhere" -- which was the stronger of the two claims.  The
surviving realization is **(a)**: fold the combination into the injected strengths,
keep the base crossbar, and claim "matrix-free in notation", not "matrix-free
device".

**3.3 No sample-efficiency win.**  `ext_04` swept the calibration set from 16 to 192
samples and the rank-constrained readout never beat the full-rank one by more than
the run-to-run spread.  The saving is real but **neutral**.

**3.4 No robustness to a missing amplitude calibration.**  The base version's
finding that the amplitude calibration is mandatory is **rank-independent**:
`ext_05` measured the uncalibrated collapse at about `-220` R^2, and the spread
across ranks is 0.05 of that, i.e. **0.02%**.

---

## 4. Half B -- the geometry operator (ext_06)

### 4.1 The identity it rests on

Read from the device's own code path, the kernel is

```
z_j(x) = exp( -k_j || (x - c_j) / h_j ||^2 )
```

so with `alpha_j = k_j / h_j^2`, `gamma_j = 2 alpha_j c_j`, `delta_j = alpha_j ||c_j||^2`:

```
log z_j = -alpha_j ||x||^2 + gamma_j . x - delta_j
```

**linear in three coefficients that satisfy ONE exact relation**,
`delta = ||gamma||^2 / (4 alpha)`.  Hence

```
h_j = sqrt(k_j / alpha_j)        c_j = gamma_j / (2 alpha_j)
```

so a node's geometry can be **solved** from its response, with the extra coefficient
serving as a CHECK rather than a fit.  This is the geometry counterpart of the
strength calibration, and it is what makes operator-only training possible.

### 4.2 What was verified (ext_06), all checks passing

| check | result |
|---|---|
| C1 exact recovery | centre and width relative error `< 1e-3` (typically `1e-6`) for both kernel channels, `D = 1..4`, probe counts `>= 16` |
| C2 discrimination | the consistency relation separates a genuine node from a non-node by a **20x** margin |
| C3 negative controls | a sum of two nodes **CAUGHT** (`2.5e-01`), pure noise **CAUGHT**; executed, not skipped |
| C4b probe design | at the minimum count `n = D+2`, the DESIGNED probe set has a bounded worst case (`< 1e-3`) |
| C6 identifiability | the wrong-`k` width scaling is confirmed exactly (`0.774597 = sqrt(0.3/0.5)`, `1.290994 = sqrt(0.5/0.3)`) |
| C5 noise | centre error `1.2e-03` at 1% reading noise, `4.1e-03` at 3% |

### 4.3 The three limits, all measured rather than assumed

1. **`k` and `h` are not separately identifiable.**  The operator fits the exponent,
   so a wrong assumed `k` leaves `alpha`, the centre and the consistency relation
   EXACTLY right and scales the width by `sqrt(k_assumed/k_true)`.  This was found by
   a negative control that **failed to fail**.  It is a requirement on the hardware,
   not a defect: `k` must be a known constant (`0.5` narrow, `0.3` wide, both fixed
   by the bias network).
2. **Partition-of-unity normalisation is detectable only in some regimes.**  With
   two nodes co-located at different widths the normalised profile is a genuine
   two-scale function and is caught; with the two nodes at random positions one
   usually dominates the probe region and the result is indistinguishable from that
   single node.  A consistency check on a normalised channel must be told which
   regime it is in.  This connects to the base version's own finding that
   normalisation is what breaks parameter-group independence.
3. **Probe points should be DESIGNED at the minimum count, and need not be beyond
   it.**  At `n = D+2` a random set is a gamble whose worst case varies over two
   orders of magnitude across `D = 1..4`, while the design's worst case stays below
   `1e-3`.  The design even LOSES to random at one dimension, so the honest claim is
   predictability rather than a win on every instance; from `n >= 16` random probes
   are adequate.

---

## 5. Half B -- training with operators only (ext_07)

`trainer.py` trains a node network with **no gradients, no finite differences and no
general optimiser**.  Which of its steps actually do work is measured in 5.1; the
comparison against two controls is 5.2, the accounting history is 5.3, and the
follow-up that moved the probe design to where it belongs is 5.4.

| step | operator | form |
|---|---|---|
| 1 GROW | OP-C proposes nodes from the residual, each candidate SCORED before acceptance | closed form |
| 2 SOLVE | ridge over the full feature matrix, `lambda` chosen on validation | closed form |
| 3 GATE | keep a round only on **held-out** evidence | comparison |
| (REFINE) | OP-G re-solves each node's geometry — **measured inert, off by default** | closed form |

### 5.1 What the trainer actually uses, after measurement

| step | operator | does it change the result? |
|---|---|---|
| GROW | OP-C proposes nodes, each candidate SCORED before acceptance | **yes** — this is the placement |
| SOLVE | ridge over the full feature matrix, `lambda` on validation | **yes, and it is what makes placement order matter little** |
| GATE | keep a round only on held-out evidence | **yes** — it decides what is returned |
| REFINE | OP-G re-solves each node's geometry | **no — measured inert, and OFF by default** |

`REFINE` is off by default because `ext_09` measured it to be a fixed point: OP-G
inverts a *response*, so refining a node against a residual that this node already
explains returns the SAME node, and GROW + SOLVE leave each node already there.
Turning it off does not change the score, and that is the evidence rather than the
claim.  The operator itself remains verified and is genuinely useful on the device
side (section 5.4); what is withdrawn is calling it a training step.

### 5.2 The three arms, at matched cost

Arm C's budget is sized from arm A's **MEASURED** evaluation count.  An earlier
version sized it from a nominal `nodes x candidates x probes` figure that arm A
never spends -- the proposals come from the residual profile over training inputs,
so no probe is read -- which handed arm C about six times arm A's real cost while
the report called the two matched.  Corrected, on the project's own 4D task, one
layer of 40 nodes, 3 seeds:

| arm | test R^2 | sd | forward evaluations |
|---|---|---|---|
| **A OPERATOR** | **+0.5817** | 0.0569 | **414 (measured)** |
| B RANDOM geometry (null hypothesis) | +0.2470 | 0.0626 | 0 |
| C FD SEARCH (the base's own `arm="hybrid"`) | +0.1589 | 0.1601 | 408 (budget) |

Operator beats random by `+0.3347` (tolerance 0.0723) and the cost-matched FD search
by `+0.4228` (tolerance 0.1849), the large tolerance being FD's own spread at this
budget.  FD's **worst seed is negative** (`-0.0675`): at ~400 evaluations the search
is not merely behind, it is unstable.

**Arm B is what makes the result mean anything.**  Without the null hypothesis, a
good score from A could simply mean that any placement works once the strengths are
calibrated, and the operators would be decoration.

**The regime bound, stated in the report and then tested:** arm C here is a very
short run -- 414 evaluations is about one sweep of the 200-parameter geometry, and
a typical base-version run spends tens of thousands.  So ext_07 alone could only
claim "the operators lead when evaluations are scarce".  Section 6 tests that bound
directly, and section 5.3 records what changed when the budget was corrected.

### 5.3 A note on which direction the corrections went

Three separate accounting defects were found in this comparison, and the **first two
flattered the operator arm** while the third did not:

1. arm A was charged probe readings it never takes (a fiction that made it look
   cheaper than it is) -- fixed by measuring evaluations;
2. the report printed equal costs for two arms six times apart -- fixed by deriving
   arm C's budget from the measurement;
3. the crossover rule in ext_08 conflated "the gap fell inside the seed spread" with
   "FD reached the operator", and its verdict then **understated** the operators.

Correcting 1 and 2 moved FD from `+0.2905` (over-funded) to `+0.1589` (matched),
i.e. the earlier framing was **generous to the competitor**.  That is the direction
a corrected accounting error should be checked in, and it is recorded here rather
than silently replaced.

### 5.4 Where free choice of probe points actually exists (ext_09)

The step that was going to be "wire the designed probe set into the trainer's
refinement" turned out to rest on a mistake, and the mistake is the finding:

> **the residual is known only where the TARGET is known**, so refinement cannot
> read at points of its own choosing -- it can only select a SUBSET of the training
> inputs.  `star_probes` belongs to the device-side read-out, which really can read
> anywhere.

Both halves were then measured, and they came out opposite:

| | question | answer |
|---|---|---|
| **P1** subset choice | does leverage (D-optimal for a fixed candidate set) beat taking every informative point? | **NO -- it is 7.4x WORSE** (`4.78e-01` against `6.42e-02` on the controlled problem), better in only 1 of 5 cells.  D-optimality targets exact identification; under NOISE, more points beat better-conditioned points |
| **P2** device-side read-out | can `identify_placement` measure where a source actually is? | **YES, exactly.**  The recovered placement shift tracks the true one with ratio **1.000** at 6, 16 and 64 probes and shifts from `1e-3` to `3e-2`; a zero shift recovers at `3e-15`.  Placement error is a MEASURABLE quantity, and it works at the minimum count `D+2 = 6` |
| **P3** end-to-end | does the subset rule change the trainer? | **NO, and the reason is the mechanism**: the two rules return the same geometry to within `3.1e-07` (one float32 ULP) while `refine_kept` differs (7 against 1).  OP-G is a fixed point here, so the step cannot move |

P1 is the useful kind of negative result: a plausible, principled improvement that
the measurement rejected.  P2 is the operator finding its real job.

### 5.5 Does it generalise?  A second task and a second architecture (ext_10)

Every earlier half-B result rests on one task (4D) and one architecture (one layer of
40 nodes), which was the largest remaining limitation: the advantage could have been
a property of that problem.  `ext_10` varies both -- tasks 2d/4d/10d, architectures
`1x40`, `1x80` (wider) and `2x[40,40]` (a two-layer wire-direct cascade, the base
version's own shape) -- with the same three arms in every cell, and it is resumable
like ext_08 (18 atomic cells; a warm `--report-only` rebuild takes 1 s against 14 s).

| cell | OPERATOR | RANDOM | FD | op-random | op-FD |
|---|---|---|---|---|---|
| `1x40_2d` | **+0.9825** | +0.9226 | +0.9056 | +0.0599 | +0.0769 |
| `1x40_4d` | **+0.5817** | +0.2627 | +0.1580 | +0.3190 | +0.4237 |
| `1x80_4d` | **+0.6278** | +0.2401 | -0.0229 | +0.3877 | +0.6508 |
| `2x40_4d` | **+0.5877** | +0.2987 | +0.0173 | +0.2890 | +0.5704 |
| `1x40_10d` | -0.0455 | -0.0908 | -0.0797 | *excluded* | *excluded* |
| `2x40_10d` | -0.0692 | -0.0284 | +0.0011 | *excluded* | *excluded* |

> **GENERALISES, on every cell that can discriminate.**  4 of the 6 cells clear the
> resolving-power floor and the operator placement wins **4/4 against the null
> hypothesis and 4/4 against the cost-matched FD search** -- including the wider
> layer and the deeper cascade.  The advantage is a property of the method, not of
> the single-layer 4D setting the earlier experiments used.

**The two excluded cells are a CAPACITY result and must not be read as a failure of
the operators.**  On 10d *every* arm, random geometry included, sits at or below
zero -- none of them beats predicting the mean -- so the cell cannot discriminate
between placement rules.  40 nodes over a 10-dimensional input simply is not enough
capacity.  The floor used is the base version's own (`abl_common.learning_ok` treats
`R^2 <= 0.05` as "did not learn") rather than a number invented here, and the
exclusion is reported per cell instead of being averaged away.

**What the cascade cells are and are not.**  The operator cascade is built
STAGEWISE -- each layer fitted to the residual the previous ones left, never
revisited -- while the FD arm optimises every layer JOINTLY.  So `2x40_4d` compares
greedy construction against joint search, not two runs of the same optimiser, and
that is the honest description of the two methods.

### 5.6 The test that could have overturned half B (ext_11)

The base version's most important warning is that a wrong twin damages the model in
a way the twin's own report does not reveal -- at 20% width error it reported
improving fits while the device collapsed.  Half B never touched that question, and
it matters MORE here, because the operator trainer **plans entirely inside the twin**:

    placement    the residual profile is computed with the twin's kernel
    features     `features()` evaluates the twin's kernel
    validation   the gate scores held-out data through the same twin

Unlike the base version's calibration -- which can be performed on the device by
reading Z through the sensor chain -- **the operator trainer has no device-side
signal at all**, so a twin error should be invisible to its gate.  That was the
prediction; this is the measurement, and it is the one result that bounds the
physical claim.

| realized width error | twin's self-report | device, as planned | device, re-calibrated | self-deception |
|---|---|---|---|---|
| 0% | +0.5817 | +0.5817 | +0.5817 | 0.0000 |
| +5% | +0.5817 | +0.6063 | +0.5941 | -0.0246 |
| -5% | +0.5817 | +0.5490 | +0.5546 | +0.0327 |
| +10% | +0.5817 | +0.6216 | +0.6130 | -0.0399 |
| -10% | +0.5817 | +0.5097 | +0.5212 | +0.0721 |
| +20% | +0.5817 | +0.6180 | +0.6370 | -0.0363 |
| **-20%** | +0.5817 | **+0.4157** | **+0.3927** | **+0.1660** |
| +30% | +0.5817 | +0.5581 | +0.6490 | +0.0236 |

> **NOT PROTECTED, exactly as predicted.**  The twin's report is CONSTANT at
> `+0.5817` while the device falls as low as `+0.4157`: the self-deception reaches
> **+0.166**, in the flattering direction, because the twin's features are the ones
> the strengths were fitted to.  The gate cannot see it -- placement, features and
> validation use the same wrong kernel, so there is nothing in the loop for the
> error to contradict.

> **AND THE GEOMETRY DOES NOT SURVIVE.**  Re-solving the strengths ON the device
> does not recover the loss (`-0.1660` planned against `-0.1890` re-calibrated), so
> the error is in the **basis**, not in the weights.  This is the outcome that
> bounds the physical claim -- along with the repair, which exists and is already
> verified: `ext_09`'s `identify_placement` measures a realized geometry with
> ratio `1.000` at the minimum probe count, so the loop can be closed by
> **measuring what the device actually has and re-planning with it**.  Until that
> loop exists, the operator trainer's geometry is conditional on the twin, and
> device-side strength calibration does not substitute for it.

**Two further results.**  Only two `k` errors are physically realizable -- the
coefficient is fixed by the bias network, so the twin can believe the wrong CHANNEL
but `k` cannot drift; that costs `-0.0155` and device re-calibration more than
recovers it (`+0.6484`).  And the width damage is **one-sided**: narrower-than-planned
is what hurts while wider sometimes scores better than exact (`+0.6216` at `+10%`),
so a placement tolerance should be specified on the narrow side rather than
symmetrically.

### 5.7 Closing the loop: calibrate the ACTUATOR (ext_12)

ext_11 left the physical claim bounded -- a twin error damages the basis and
device-side strength calibration cannot repair it.  `ext_12` builds the repair and
tests whether it recovers.  The distinction it turns on is not obvious and is the
whole result:

| arm | what it does with the measurement |
|---|---|
| **A** open loop | nothing -- plan, place, deploy the planned strengths |
| **B** accept | ADOPT the measured geometry as the model's geometry, re-solve strengths |
| **C** correct | estimate the ACTUATOR'S GAIN from the measurement, re-command `h/ghat`, re-solve strengths |

| realized width error | A open | B accept | **C correct** | C - A | gain estimated | gain true |
|---|---|---|---|---|---|---|
| 0% | +0.5817 | +0.5819 | +0.5819 | +0.0002 | 1.00002 | 1.00000 |
| +5% | +0.6063 | +0.5942 | **+0.5819** | -0.0245 | 1.05000 | 1.05000 |
| -5% | +0.5490 | +0.5545 | **+0.5816** | +0.0325 | 0.95001 | 0.95000 |
| +10% | +0.6216 | +0.6131 | +0.5819 | -0.0397 | 1.10008 | 1.10000 |
| -10% | +0.5097 | +0.5212 | **+0.5817** | +0.0720 | 0.90002 | 0.90000 |
| +20% | +0.6180 | +0.6370 | +0.5816 | -0.0363 | 1.20004 | 1.20000 |
| **-20%** | **+0.4157** | +0.3927 | **+0.5817** | **+0.1660** | 0.80000 | 0.80000 |
| +30% | +0.5581 | +0.6490 | +0.5818 | +0.0236 | 1.29999 | 1.30000 |

> **THE LOOP CLOSES.**  Arm C lands within `0.02` of the error-free score in
> **8 of 8** levels; at the worst level the open loop lost `0.166` and arm C loses
> `0.0000`.  The actuator's gain is recovered to `6.8e-05` relative.

> **And B is confirmed to fail**, exactly as ext_11 predicted: adopting a
> uniformly-shrunk node set makes the device slightly WORSE than the open loop at
> `-20%` (`+0.3927` against `+0.4157`), because the basis is the problem and no
> readout repairs a basis.  The two arms differ by the whole result, and running
> only B would have concluded that measurement does not help.

**So the repair is the same move the base version made one level down**: there, the
twin's prefactor `g_j` converted a strength into an injected energy and MEASURING it
replaced searching for it; here, the twin's assumption about the realized geometry
converts a plan into a command, and measuring it replaces hoping.  ext_11's bound
becomes a **procedure**:

    place  ->  measure each node at a designed probe set  ->  re-command  ->
    calibrate the strengths

**The price**, because a loop that is not priced is not a procedure: `D+2 = 6`
readings per node here, `240` in total for 40 nodes, against a strength calibration
that costs `n_train` per source.  The loop is cheap relative to what it protects.

**For a random placement offset there is nothing to pre-compensate** -- only
something to know -- so arms B and C coincide there by construction, and that is
stated rather than presented as two mechanisms.  Offsets turn out to matter little
anyway (`+0.0011` at a shift of `0.1`), consistent with the base version's finding
that a positional offset is nearly free while a width error is not.

### 5.8 A third task FAMILY: the base version's own adversaries (ext_13)

ext_10 varied the input DIMENSION and the architecture, but every task there is the
same kind of object: a smooth synthetic regression.  The base version already carries
tasks built to ATTACK its own headline rule, and its own docstring says why:

> "`steps` and `hetero` are deliberate adversaries of the headline rule.  The rule is
> 'solve the readout in closed form', which is an ordinary least-squares fit; it
> should look worst when the target is discontinuous ... and when the noise is not
> uniform ... Including them is what makes a positive sweep informative instead of
> circular."

Half B uses the same Gaussian basis and the same least-squares solve, so the same
attacks apply, and it had never been pointed at them.

| task | structure | OPERATOR | RANDOM | FD | op-random | op-FD |
|---|---|---|---|---|---|---|
| `4d` | smooth (the reference) | +0.5698 | +0.2213 | +0.1722 | +0.3486 | +0.3977 |
| `moons` | a different TASK TYPE (classification boundary) | +0.9861 | +0.9835 | +0.9846 | +0.0026 | +0.0015 |
| `steps` | **DISCONTINUOUS**: a Gaussian basis cannot fit it | **-0.0297** | **+0.0573** | -0.0138 | **-0.0869** | -0.0159 || `hetero` | **HETEROSCEDASTIC**: least squares is misspecified | **+0.2303** | +0.0730 | **-0.0428** | +0.1573 | +0.2731 |
| `shift` | train `[-2,2]` vs test `[-3,3]` | **+0.1249** | +0.0309 | +0.0520 | +0.0940 | +0.0729 |
| `noise` | `y` independent of `x` | -0.0525 | -0.0467 | -0.0100 | *excluded* | *excluded* |

> **THE ADVERSARY RESULT IS SPLIT, ONE EACH -- but read 5.11 before quoting the
> `steps` half, because that row's RANDOM column is not a usable baseline.**  `steps`
> breaks the single greedy run's PLACEMENT, and `hetero` does not break anything: the
> operator is the best of the three there and more than reverses the FD arm's
> negative score.  On the discontinuous target the operator reads `-0.0297` against a
> random arm at `+0.0573` -- except that the random arm's own draw-to-draw spread on
> `steps` is `0.3443`, larger than the `0.0869` being claimed.  The 5.11 measurement
> replaces that comparison with one that does not depend on the null at all.

> **AND THE 3-SEED READING WAS WRONG, in both directions, both against the method.**
> At 3 seeds this table said the operator was the BEST on `steps` (`+0.155` against
> `+0.073` and `-0.018`) and that it did **not** fit pure noise hardest.  At 12 seeds
> it is the **WORST** on `steps` and it **DOES** fit the noise hardest (OPERATOR
> `-0.0525` against RANDOM `-0.0467` and FD `-0.0100`: on data with no signal, this
> method is the most confidently wrong).  Both flips were forecast as "more seeds
> would settle it", so this is the settling rather than a new experiment -- and it is
> the reason that sentence is worth writing down when it is true.

**The scorecard, with the non-wins diagnosed rather than counted.**  5 of 6 tasks are
informative (the `noise` cell is excluded: no arm clears the floor).  The operator
beats random in **3/5** and matches or beats cost-matched FD in **5/5**.  The two
non-wins are diagnosed separately, and the diagnosis now distinguishes an imprecise
LOSS from an unresolved comparison -- an earlier version filed "the mean is against
the operators with a large spread" as SPREAD, which reads as "unresolved" when the
direction is in fact against the method:

* `moons` -- **CEILING**: every arm sits at `~0.98`, so there is no headroom to win in;
* `steps` -- **LOSS, imprecise**: the mean is against the operators (`-0.0869`) and the
  spread is large, so it is a loss whose size is uncertain, not a comparison that
  failed to resolve.

**The `noise` cell, asked as its own question.**  A greedy constructor keeps reducing
the residual, so on pure noise it is the arm most at risk of *fitting the noise* and
scoring below zero.  At 3 seeds the answer looked reassuring (the operator was not the
most negative); **at 12 seeds it is** -- OPERATOR `-0.0525`, RANDOM `-0.0467`, FD
`-0.0100`.  On data with no signal this method is the most confidently wrong, which is
the honest cost of a constructor that always reduces the residual.

**A prediction that FAILED, recorded because it was stated in advance.**  The design
note argued that `shift` would be SHARPER for half B than for half A: the operator
trainer plans entirely inside the twin and validates in-distribution, so it should be
structurally blind to a shift.  It is not -- the operator arm scores `+0.0895` against
`+0.0178` and `+0.0177`, four times the alternatives.  The reason is that half B's
**placement is not gate-driven**: the gate only decides whether to keep a refinement
round, and ext_09 measured that step to be inert.  So the blindness argument applies
to the base version's gate-driven *calibration* and not to a constructor that reduces
the residual directly.

### 5.9 A bound on the placement rule (ext_14)

Every half-B result rests on one component that had never been justified: the
placement rule, whose only support was that the one principled alternative tried
(leverage) measured 7.4x worse.  `ext_14` replaces that with a measured bound.

**What the bound can and cannot be.**  The optimal N-node fit is non-convex and
nobody can compute it, so the baseline is **best-found**: restarts of a wider version
of the same constructor, PICKED BY VALIDATION and reporting that run's test score --
the deployable thing.  That is one-sided and is read as one:

> best-found is an UPPER bound on achievable R^2, so the greedy-vs-best gap is a
> LOWER bound on the true gap.  A small gap rules out the failure that matters
> (greedy being badly suboptimal) at this scale; it does not prove optimality.

`capture = (G - R) / (B - R)` is the fraction of the found improvement the rule
obtains, where `R` is the random-geometry null used everywhere else.

| N | G greedy | W wide pool (32) | B best-found (val-picked) | R random | **capture** | G - B |
|---|---|---|---|---|---|---|
| 4 | +0.0982 | +0.1169 | +0.1625 | +0.0729 | **0.282** | -0.0643 |
| 8 | +0.2182 | +0.2172 | +0.2641 | +0.1429 | **0.621** | -0.0459 |
| 16 | +0.4067 | +0.4165 | +0.4833 | +0.1972 | **0.732** | -0.0766 |
| **40** | **+0.5817** | +0.5710 | +0.5930 | +0.1227 | **0.976** | **-0.0113** |

> **AT THE OPERATING POINT THE RULE IS NEARLY OPTIMAL BY THIS MEASURE.**  Every other
> half-B result runs at `N = 40`, where capture is **0.976** and the gap to
> best-found is **-0.0113**.  So this bound does NOT undercut the headline results:
> at the node count they use, a single greedy run obtains essentially everything any
> restart finds.

> **AND THE RULE IS MYOPIC AT SMALL N, measurably so.**  Capture falls to **0.282**
> at `N = 4`, where random geometry (`+0.0729`) actually BEATS greedy (`+0.0982`),
> because with very few nodes each choice locks in a configuration the remaining
> nodes cannot repair.  The trend is monotone in N: `0.28, 0.62, 0.73, 0.98`.  This
> bounds the method from below, at small N, which is worth knowing before anyone
> scales it down.

**The limitation is not the proposal pool.**  Widening it from 4 random candidates to
32 moves the score by less than `0.05` at every node count, while merely choosing a
different RESTART by validation is worth `+0.022` to `+0.067`.  So the gap is
run-to-run **variation**, not a shortage of proposals -- and that has a cheap fix
worth naming:

    run M restarts and keep the one with the best HELD-OUT score

which is the base version's gate discipline applied to the placement, costs M times a
training run that takes a fraction of a second, and captures most of the small-N gap
without touching the proposal rule at all.

**Two things recorded rather than smoothed over.**  A test-set peek would have bought
up to `+0.0577` more than the honest validation-picked baseline; it is not used, and
the size of that number is the reason the distinction is stated.  And the run's own
first version took the maximum TEST score over restarts as its baseline, which
inflates the baseline and makes the rule look worse than it is -- corrected, and the
correction moved `N = 40` capture from `0.669` to `0.976`.

### 5.10 The repair, against error shapes it was not built for (ext_15)

ext_12 closed the loop for a **common multiplicative** width error, estimating one
pooled gain `ghat = mean(h_measured / h_commanded)`.  That is one shape, and the
closed loop is now part of the physical claim -- so a repair validated against one
shape is not yet a repair.  Three shapes were applied, each breaking the pooled
estimator differently.

**P1 -- PER-NODE error (`h_j(1 + eps_j)`, independent per node).**  The shape most
likely in fabrication, where per-device spread is the norm.

| per-node error sd | open loop | pooled gain (ext_12) | **per-node inverse** | what the pooled gain reports | true mean gain |
|---|---|---|---|---|---|
| 0.05 | +0.5685 | +0.5686 | **+0.5816** | 0.9981 | 0.9981 |
| 0.10 | +0.5651 | +0.5660 | **+0.5818** | 0.9962 | 0.9961 |
| 0.20 | +0.5303 | +0.5344 | **+0.5815** | 0.9923 | 0.9923 |
| 0.30 | +0.4936 | +0.4994 | **+0.5817** | 0.9884 | 0.9884 |

> **THE POOLED ESTIMATOR FAILS, AND FAILS SILENTLY.**  Averaging independent errors
> converges to `1.0`, so it reports "the actuator is fine" while every node is
> individually wrong -- at `sd = 0.30` it moves the score by `+0.006` and the true
> mean gain is `0.9884`.  The **per-node inverse** uses the same measurement without
> averaging and returns the **error-free score at every spread tested**
> (`+0.5816 ... +0.5817`).

**So the procedure is corrected, and this supersedes ext_12's:**

    place -> measure each node -> invert PER NODE -> re-command
          -> iterate if the gain depends on the command
          -> calibrate the strengths

ext_12's pooled form is a **special case**, valid only when the error really is
common, and should be read as such rather than as the general repair.

**P2 -- WIDTH-DEPENDENT gain (`h(1+eps)(1+beta h)`).**  The gain depends on the
command, so one inverse step is not the fixed point.

| beta | pass 0 | pass 1 | pass 2 | pass 3 | width error 0 -> 3 passes |
|---|---|---|---|---|---|
| 0.00 | +0.3927 | +0.5817 | +0.5817 | +0.5817 | 0.2000 -> **0.0000** |
| 0.25 | +0.5320 | +0.5862 | +0.5812 | +0.5818 | 0.0937 -> **0.0002** |
| 0.50 | +0.5697 | +0.5755 | +0.5833 | +0.5811 | 0.0244 -> **0.0007** |
| 1.00 | +0.6332 | +0.4868 | +0.5818 | +0.5759 | 0.2430 -> **0.0087** |

**Two passes converge**: the width error is below `0.009` at every nonlinearity
tested, and one pass is visibly incomplete at `beta = 1`.  Each pass costs its own
`D+2` readings per node.

**P3 -- DRIFT between measurement and use.**  The correction goes **stale**: the width
error against the plan grows monotonically with drift (0, 0.0625, 0.125, 0.25, 0.5 at
drift 0, 0.05, 0.10, 0.20, 0.40).  But the SCORE does not register it here, and the
reason is reported rather than smoothed over: a drift makes the realized widths
LARGER, and this node set happens to prefer wider widths -- the same one-sided effect
ext_11 measured (`+0.6216` at `+10%` width).  A drift test that could register it
would need widths the plan already prefers, which this plan does not have.  What the
measurement does establish is that the correction has a **shelf life**, so any
deployment has to re-measure at a rate set by the drift it expects.

---
### 5.11 Which level fails on `steps`: the basis, the placement, or the null? (ext_16)

ext_13 left `steps` reading as "random geometry beats the operator", which invites the
conclusion that the method failed.  Three different things could be true, and only a
measurement separates them:

    the DEVICE     produces Gaussian nodes; it does not know the target
    the NETWORK    40 Gaussian nodes with a linear readout
    the PLACEMENT  which 40 nodes, chosen by the operator trainer
    the STRENGTHS  the readout weights, solved in closed form

`ext_16` measures the network's ceiling by **restarts of the operator's OWN
configuration** (same candidate pool, same data seed, different search seed), picked by
validation, with the operator's own run included in the pool so best-found dominates
it by construction.

| arm | `steps` | `4d` (smooth control) |
|---|---|---|
| OPERATOR (one greedy run) | **-0.0297** | +0.5698 |
| RANDOM geometry (mean of 8 draws) | -0.1087 | +0.1281 |
| RANDOM's OWN draw-to-draw spread | **0.3443** | 0.1531 |
| **BEST-FOUND (restarts)** | **+0.2043** | +0.6011 |
| **the operator's gap to its own restarts** | **0.234** | **0.031** |

> **THE NETWORK IS NOT BROKEN ON `steps`.**  Best-found reaches **+0.2043**, so 40
> Gaussian nodes CAN fit a square wave usefully and the basis is not the wall.  What
> fails is the **single greedy run's placement**, and it fails by **0.234** there
> against **0.031** on the smooth control -- a **7.5x larger gap exactly on the
> discontinuous task**.  The finding is a defect in the rule, not a limit of the
> substrate, the basis or the readout.

> **AND THE RANDOM BASELINE IS NOT USABLE ON `steps`.**  The null's own draw-to-draw
> spread is `0.3443` -- larger than the `0.0869` difference ext_13 reported against
> it.  ext_10/13 and ext_16 run the SAME null code with different RNG streams and get
> `+0.0573` and `-0.6521` for the same arm, which is what exposed it.  **So "random
> geometry beats the operator on `steps`" is retracted**: that comparison was inside
> the null's own noise.  The gap-to-restarts comparison above replaces it and does not
> depend on the null at all.

**A mechanism that was tested and NOT supported.**  The natural explanation -- OP-C
puts nodes where the residual is large, a jump puts its largest residual at the jump,
and approximating a step with smooth bumps forces NARROW nodes -- predicts that the
operator's widths should collapse on `steps` relative to the smooth control.  Measured
width ratios against random: **`1.247` on `steps` against `1.160` on `4d`**, i.e. no
task separation.  The explanation is therefore **not supported** and is not repeated
anywhere in this document as though it were.  The `steps` mechanism remains
**unexplained**, which is the honest state and the next thing to measure.

### 5.12 Half B on the SECOND substrate (ext_17)

The thesis says a node network can be realised in **any** physical system whose phase
space can generate movable nonlinear activation sources.  The base version verified two
substrates, and half A ran on both (ext_01, ext_05).  **Half B had never touched the
second one, and could not**: `trainer.features` hard-coded `make_layer("circuit")` and
had no substrate parameter at all.  A claim whose main clause is "any physical system"
was resting on one substrate for the part of the argument that trains the network.

`substrate_diffusion` engineers the same kernel out of device physics: a localised
decaying Green's function `(4 pi D t)^(-dim/2) exp(-r^2/4Dt)` is, for `t = h^2/(4kD)`,
exactly `exp(-k|X-c|^2/h^2)` **times a prefactor**, and `KERNELS = (0.5, 0.3)` are the
same two channels.  So exactly two things can reach the operator: a **per-source
amplitude**, and a **sensor chain** on the readings.

**THE PREDICTION, stated before measuring.**  In log space the amplitude enters only the
constant term, and OP-G recovers `h` from `alpha` and `c` from `gamma` without ever using
that constant:

    log(a_j * z_j) = log a_j  -  alpha|x|^2 + gamma.x - delta   ->   (c, h) unchanged

| measurement | value |
|---|---|
| per-source amplitude spread over 12 nodes | **10.5x** |
| recovered centre difference, unit-peak vs field | **2.36e-15** |
| recovered width difference | **3.42e-16** |
| **consistency residual difference** | **0.3975** |

> **THE PREDICTION HOLDS, at numerical precision, under a 10.5x amplitude spread.**
> The geometry operators are **substrate-agnostic by construction**, not by coincidence
> -- the strongest form the "any physical system" claim can take.

> **AND THE CONSISTENCY ROW IS A CONFIRMATION, NOT A FAILURE.**  The operator checks
> `delta = |gamma|^2/(4 alpha)`, and the amplitude is absorbed by `delta` -- exactly
> where the prediction says it must go.  So the amplitude moves the CONSISTENCY RESIDUAL
> by `0.4` while the geometry stays right to `1e-15`, which yields a practical warning:
> **a consistency gate rejects an uncalibrated measurement of a perfectly recoverable
> node.**  Any hardware procedure that gates on consistency must calibrate the amplitude
> FIRST, or it will discard good readings -- the opposite of what a validity check is for.

**P0, both substrates at their own numerical floor.**  Circuit `1.7e-06`, diffusion
`2.7e-15`.  The circuit layer computes in float32 and the diffusion path in float64, so
"both exact" means each reaches its OWN floor.  Reading the first as "less exact" would
be a mistake about precision, not about physics.

**P2, the sensor chain is NOT free**, and unlike the amplitude it should not be:

| level | log-fit residual | centre err | width err | nodes refused |
|---|---|---|---|---|
| none | 1.97e-15 | 2.30e-08 | 1.83e-08 | 0 |
| low | 5.43e-02 | 1.54e-01 | 6.13e-02 | 0 |
| medium | 6.59e-01 | 1.05e+00 | 3.52e-01 | 0 |
| high | 2.25e+00 | 4.59e+00 | 1.34e+00 | 2 |

**P3, the trainer transfers, and on the undamaged cells the two substrates agree to the
last digit**: OPERATOR `+0.5817` on the circuit and `+0.5817` on diffusion, against
random at `+0.1497` on both -- the same trajectory, because both cells deliver the same
unit-peak kernel.  With the chain engaged at `low` the operator arm reaches `+0.5952`
against `+0.1498`, so it still beats the null by `+0.445`.

> **WHAT THIS DOES NOT ESTABLISH**, stated so the claim is not read wider than it is:
> **two substrates is not "any"**.  Both cells were built here to reproduce the same
> engineered kernel, so this shows the operators do not depend on WHICH of the two
> produced the field -- not that they would survive a substrate whose response is not
> Gaussian in the input at all.  The prediction states exactly what the operators need:
> log-linearity in `[-|x|^2, x, -1]`.  A substrate that breaks THAT breaks the
> operators, and no cell here tests it.

**The measurement protocol, stated because it is an assumption**: the operators assume
**one node is driven at a time with the others off**.  Under that protocol the diffusion
cell's PU normalisation does not enter the reading -- the normalising sum is the driven
node alone -- so PU is not a confound here.  A protocol that drove all sources at once
would need the coupling term accounted for, and nothing in this extension does that.

### 5.13 Is the linearity worth its cost? (ext_18)

This is the trade the design memo says must not be left to chance:

> "That operator exists **only because the readout is linear in the strengths.**  Any
> combination that is nonlinear in the readings makes fitting the strengths a nonlinear
> problem again, and trades away that advantage.  It may buy expressiveness worth more
> than the advantage -- but **the trade must be made deliberately, and measured, not
> discovered later.**"

**The fairness problem is the design.**  A nonlinear readout has far more parameters
than a linear one on the same nodes, so the decisive control is an arm that spends the
same budget on **more nodes**.  Three arms, on the SAME 40 nodes where the node set is
shared:

    L40     ridge on z                            the current design, fully closed-form
    N(H)    c . tanh(Wz + b), W,b DRAWN,          nonlinear in the readings, outer
            c by ridge                            weights still closed form
    L(N')   ridge on z with more nodes            the budget spent on nodes instead

| task | L40 | N80 | N320 | N1280 | L80 | L160 | L320 |
|---|---|---|---|---|---|---|---|
| `4d` | +0.5761 | +0.6056 | +0.5314 | +0.5645 | +0.6405 | +0.6610 | **+0.7028** |
| `steps` | +0.1572 | +0.2232 | +0.1636 | +0.1167 | +0.2293 | +0.2936 | **+0.3171** |
| `hetero` | +0.2215 | +0.1941 | +0.1950 | +0.2153 | +0.2902 | +0.3146 | **+0.3211** |

| task | what nonlinearity buys | what nodes buy |
|---|---|---|
| `4d` | +0.0295 (H=80) | **+0.1252** (N=320) |
| `steps` | +0.0660 (H=80) | **+0.1599** (N=320) |
| `hetero` | -0.0062 | **+0.0996** (N=320) |

> **THE LINEARITY IS WORTH KEEPING**, on the evidence of its own adversaries.  On the
> discontinuous and heteroscedastic tasks -- built by the base version specifically to
> attack a linear least-squares readout -- a nonlinear combination of the same nodes
> does not beat what spending more on nodes buys, and it pays a search penalty on more
> strengths to do it.  **Nonlinearity beats the node control on 0 of 3 tasks.**

**And the cost axis makes it worse, in the project's own currency.**  Assigning a
strength on this hardware is a *calibration*, at `n_train` measurements per strength
(`abl_common.readout_calibration_cost`) -- which is exactly what the closed-form
operator saves:

| arm | strengths | closed form? | measurements |
|---|---|---|---|
| `L40` (current design) | 40 | yes, all | 10,240 |
| `N` inner at H=80 | **3200** | **NO -- searched** | **>= 819,200** |
| `L320` (node control) | 320 | yes, all | 81,920 |

The nonlinear arm's inner strengths sit **inside** the nonlinearity, so they cannot be
solved in closed form at all: they must be searched, which is the `15-30x` penalty the
operator exists to avoid, applied to `80x` more strengths than the baseline and `10x`
more than the node control.  The node control costs `8x` the measurements **and still
wins**.

**The honest limit, stated because it is what would overturn this.**  `W` and `b` were
**drawn**, not trained, and the nonlinear gain **saturates and then falls** with hidden
width (`4d`: `+0.6056` at H=80, `+0.5314` at H=320; `steps`: `+0.2232` down to
`+0.1167`) -- random directions are wasted and the ridge outer solve cannot exploit
them.  A **trained** inner layer is precisely what would fix that, so this measurement
bounds the gain of *random* nonlinearity and a trained one could overturn the
conclusion.  That asymmetry is why the cheap form went first: a small gain under it is
strong evidence, while a large one would only have motivated the full version -- and it
came out small.

**This is the answer to a question the design memo said must not be "discovered
later", and it is the first time it has been measured at all.**

### 5.14 The `steps` gap: reached at the ceiling, and the repair is NOT safe (ext_19)

ext_16 measured the one failure of this method -- on the discontinuous task a single
greedy run lands `0.234` below what restarts of its own configuration find -- and ext_14
had already identified the repair, which sat uninstalled.  `fit_operator` now takes
`restarts=M`, with `restarts=1` reproducing every earlier number (same seed, verified).

| `steps` | M=1 | **M=4** | M=8 | M=16 |
|---|---|---|---|---|
| test R^2 | +0.1551 | **+0.2045** | +0.0564 | **-7.5180** |
| what selection gave up vs the test-best restart | 0.0000 | +0.0158 | +0.1782 | **+7.7794** |

| `4d` (the smooth control) | M=1 | **M=2** | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|
| test R^2 | +0.5817 | **-0.4235** | **-0.4235** | +0.5905 | +0.5910 |

> **THE CEILING IS REACHED.**  At `M = 4` on `steps` the repair lands on `+0.2045`,
> which **is** the best-found ceiling ext_16 measured (`+0.2043`).  The gap closes, and
> the diagnosis that produced the repair was right.

> **AND THE SELECTOR IS NOT SAFE.**  It is neither monotone nor reliable: `M = 8` and
> `M = 16` on `steps` select runs at `+0.0564` and **`-7.5180`**, and `M = 2` and
> `M = 4` on the **SMOOTH CONTROL** select `-0.4235`.  Selection on held-out `R^2`
> alone gave up `+7.7794` at the worst level.  The failure is **not** a property of
> `steps`.

**The mechanism of the failure**, and it is the base version's own warning arriving
here: a restart can build a pathological geometry whose held-out `R^2` looks fine while
its weights are huge, and a Gaussian basis that decays outside its support,
**multiplied by a large weight, explodes there** -- exactly what the base version
documented on its `shift` task.  So held-out `R^2` alone cannot select among restarts:
`restarts=M` as implemented reaches the ceiling at some `M` and is **actively dangerous**
at others.

> **AND THE MECHANISM OF `steps` IS NOW PARTLY EXPLAINED.**  At every grow step, the
> best candidate the rule proposes against the best of an **equal-sized** random pool:
> the rule wins **42.5%** on `steps` (**chance is ~50%**) and **63.3%** on `4d`.  So on
> the discontinuous task the log-linear residual profile **stops being informative** --
> the proposals carry no signal, the restarts are a lottery over them, and that is why
> the repair is *unstable* there rather than merely imperfect.  On the smooth control
> the same rule wins clearly.

---

### 5.15 10d: out of reach at this scale, and now stated (ext_20)

`ext_10` excluded both 10d cells because no arm cleared `learning_ok`'s floor -- a
CAPACITY result, since 40 nodes over a 10-dimensional input cannot fit the task.  Every
other cell in this extension is positive or negative; 10d was the one **silent** cell,
and silence is not evidence either way.  Raising the node count until the cells
discriminate:

| nodes | OPERATOR | RANDOM | FD | best arm | discriminates? |
|---|---|---|---|---|---|
| 40 | -0.0455 | -0.0908 | -0.0797 | -0.0455 | NO -- silent |
| 120 | +0.0183 | -0.1666 | -0.0797 | +0.0183 | NO -- silent |
| 360 | +0.0014 | -0.0764 | -0.0797 | +0.0014 | NO -- silent |

> **OUT OF REACH AT THIS SCALE, and now stated rather than left open.**  At 360 nodes --
> `36` per input dimension -- the best arm still does not clear the `0.05` floor.  The
> dimension axis therefore has a **hole above 4d**, and the honest statement is that
> nothing in this extension is evidence about high-dimensional inputs, **in either
> direction**.

**Two things worth recording even though no cell can carry a claim.**  The operator arm
is the best of the three at every node count, and `op - random` is `+0.0452`, `+0.1849`,
`+0.0778` -- consistently positive and largest at `N = 120`.  That is the direction the
method would want, and it is **not** a result: a cell below the floor cannot say which
placement rule is better, only that none of them fits.

---

### 5.16 The guard does not exist, and the defect is upstream (ext_21)

ext_19 reached the ceiling with `restarts=M` and found the selector unsafe.  The item
that opened was a GUARD, and this is the measurement of whether one exists.

**The hypothesised mechanism was REFUTED, and so were its alternatives.**  ext_19's
hypothesis was that a large weight on a wide node makes a large constant prediction that
explodes outside the basis support.  Three statistics were measured, not one:

| statistic | `steps`: catastrophic vs good | separates? | `4d`: catastrophic vs good | separates? |
|---|---|---|---|---|
| prediction spread / target spread | 0.674 vs 0.778 | **no** | 0.694 vs 0.693 | no |
| weight norm | 7.98 vs 8.3 | no | 14 vs 12.5 | no |
| feature condition number | 535 vs 176 | yes | 13.7 vs 13 | **no** |

The catastrophic restarts are if anything **TIGHTER** in prediction spread, so the
explosion story is wrong.  The condition number separates on `steps` only -- a post-hoc
guard on the task it was measured on, which is **not adopted**.

> **AND THE REAL REASON NO GUARD COULD WORK.**  On `steps` the validation-selected
> restart scores **-7.5182** on test while the **pool's own mean is -0.3404**: selection
> is not merely noisy, it picks **worse than a random member of its own pool**.  The
> winner's curse accounts for only `+0.1657` of held-out flattery and cannot explain a
> gap of that size.

> **WHICH TIES BACK TO ext_19's MECHANISM FINDING, and the two together are the
> result:** on `steps` the proposal family wins only `42.5%` of grow steps against an
> equal-sized random pool where chance is `~50%`, so the runs differ by **NOISE rather
> than by quality**, and a `51`-sample held-out split cannot rank noise.  **A restart
> repair can only work where the proposals carry signal.**  Where they do not, more
> restarts make selection worse, and no guard on the resulting model puts back a signal
> the criterion never had.

**So `restarts=M` is NOT shipped as a fix.**  `restarts=1` stays the default and the
parameter is documented in the code as unsafe above 1.  The ceiling is reachable --
`M = 4` found it -- but a selector that cannot be trusted to find it is not a repair.

**THE DEFECT IS UPSTREAM, in the PROPOSAL FAMILY on discontinuous targets** -- not in the
selector and not in the number of restarts.  That is a sharper place to look than where
this item started, and it is the honest outcome of a guard that was measured rather than
chosen to look good.

---

### 5.17 The upstream defect is LOCATED, not repaired (ext_22)

ext_19 located the defect in the proposal family and ext_21 showed no guard could rescue
the selector.  "The proposals are bad" hides three failures -- a bad POOL, a bad
CRITERION, or both -- and `ext_19`'s own result pointed at the criterion: the rule's
candidates and RANDOM candidates scored the same **through the same criterion**, so if
the criterion does not track the outcome, two pools of different quality look identical
through it.  That was a prediction, and it is half refuted.

**What the criterion is, and what was measured against it.**

    crit      = ||res||^2 - ||res - w_j z_j||^2   training set, ONE node, before the
              strengths are re-solved -- the shipped rule
    val_after = held-out R^2 after adding the candidate and re-solving EVERY strength

| task | rank agreement `crit` vs `val_after` | mean regret | regret / pool spread |
|---|---|---|---|
| `steps` | **+0.097** | 0.0415 | **0.46** |
| `4d` | **+0.191** | 0.0513 | **0.54** |

> **MY HYPOTHESIS IS REFUTED, AND WHAT REPLACES IT IS BETTER.**  The criterion does not
> track the outcome on **either** task -- `+0.191` on the SMOOTH control against
> `+0.097` on the discontinuous one -- so this is **not** a property of discontinuous
> targets.  It is a general weakness of ranking by a one-node, training-set reduction.

> **AND THE REGRET SAYS HOW MUCH IT COSTS.**  The criterion's pick leaves about **half
> the pool's spread** on the table (`0.46` on `steps`, `0.54` on `4d`), so a better
> candidate is available at almost every step and the rule does not find it.  That is
> the mechanism behind ext_19's "random matches the rule" **that does not require the
> pool to be bad**.

**But replacing it is catastrophic, and the way it fails is the finding.**

| task | arm | test R^2 | val R^2 | gap |
|---|---|---|---|---|
| `steps` | criterion (shipped) | +0.1587 | +0.2832 | +0.1245 |
| `steps` | val_after (re-solve) | **-0.1927** | **+0.8412** | **+1.0340** |
| `4d` | criterion (shipped) | +0.5441 | +0.4846 | -0.0596 |
| `4d` | val_after (re-solve) | **-1.2580** | **+0.9098** | **+2.1679** |

Ranking ~70 candidates per step on a **51-sample** split for 40 steps is a great deal of
selection on very little data, so the new criterion **overfits the validation set far
worse than the shipped rule underfits the outcome**.  This is the risk stated before the
run, and it materialised.

> **SO THE ANSWER IS NEITHER "THE CANDIDATES" NOR "THE CRITERION" -- IT IS A
> BIAS-VARIANCE TRADE, AND THE SAMPLE COUNT IS WHAT BINDS.**  The shipped rule is a
> **low-variance** estimator (a one-node training-set reduction) that is biased; the
> replacement is closer to unbiased and far noisier.  With 51 validation samples the
> biased one wins, and it wins by a wide margin -- which makes the shipped choice
> **defensible for a reason this document had never given**: not because it is accurate,
> but because it is **stable**.

**What that means for the upstream item: it is not fixable by changing the ranking rule
at this scale.**  The binding constraint is the **sample count**, which is a property of
the toy setup (256 train / 51 validation) and not of the method.  The defect has been
**located rather than repaired**, and located somewhere the previous three experiments
did not point at.

---

## 6. The long-budget check (ext_08)

Resumable sweep: every (seed, budget) cell is written **atomically** to
`outputs/ext_ckpt/` and skipped on the next run, with a config fingerprint so that
changing the settings invalidates old cells.  Verified by rebuilding the report from
checkpoints: **293 s -> 1 s, 15 cells reused, 0 re-run.**

| FD budget | mean R^2 | sd | vs operator |
|---|---|---|---|
| 2,544 | +0.3003 | 0.0176 | -0.2814 |
| 10,000 | +0.3617 | 0.0698 | -0.2200 |
| 30,000 | +0.4582 | 0.0478 | -0.1235 |
| 100,000 | +0.5326 | 0.0368 | -0.0491 |
| 300,000 | +0.5596 | 0.1211 | -0.0221 |

> **NO CROSSOVER.**  The FD search does not reach the operator arm's `+0.5817`
> anywhere up to **300,000** evaluations, i.e. **719x** the operator's 417.  The gap
> narrows to `-0.0221`, but from 100,000 onward it is smaller than the seed spread,
> so the correct statement is **BOUNDED BY RESOLUTION**, not "FD caught up".

**Two things the sweep found that the mean curve hides.**  FD's long-budget
behaviour is **not monotone per seed**: seed 1 scores worse at 300,000 than at
100,000 (`+0.3887` against `+0.4892`), so more search budget is not reliably better
for this trainer.  And the operator arm's cost number is MEASURED, varying about 1%
between runs (417-420) on identical data, so ratios are quoted to two significant
figures.

---

## 7. Half B does NOT deliver

**7.1 No optimality claim -- now bounded, at the operating point.**  `ext_14` measures
the placement rule against best-found: capture is **0.976** at `N = 40`, the node
count every other half-B result uses, so the rule is nearly optimal *by that
measure* where it is used.  It is **myopic at small N** (capture 0.282 at `N = 4`,
where random geometry even beats it), and best-found is an upper bound on what is
achievable, so the true gap is bounded from below and never from above: a different
node family or optimiser could do better and this experiment would not see it.

**7.2 The designed probe set does not belong in the trainer, and now does not sit
there.**  This item previously read "the refinement should probe at designed points
but that is not wired in".  `ext_09` showed that framing was wrong: the residual is
known only where the target is, so refinement cannot probe freely.  The designed set
was moved to `identify_placement`, where free choice genuinely exists and where it
measures a placement error with ratio `1.000` (section 5.4).  What remains open is
smaller: making the *subset* choice better than "every informative point", which
leverage did not achieve.

**7.3 Breadth on four axes, but ONE scale, and no device.**  This item previously read
"one task, one architecture, one scale", which was true when it was written and is not
any more: `ext_10` varied the input dimension (`2d`/`4d`/`10d`) and the architecture
(`1x40`, `1x80`, `2x[40,40]`), `ext_13` varied the task family (six tasks, including the
base version's own adversaries), and `ext_17` varied the SUBSTRATE.  What is still
narrow is stated here so the four axes are not read as more than they are:

* **one SCALE** -- 40 nodes, 256 samples, and `10d` is excluded because no arm clears
  the learning floor at that node count, so the dimension axis has a hole in it;
* **every task is synthetic**, generated by `abl_common.make_regression_data`, so
  nothing here is evidence about a real dataset;
* **everything is simulation.**  The cells are the repository's own models of a circuit
  and a diffusion medium, both built to reproduce the same engineered kernel.  The
  operators have never been run against a fabricated device, and the one result that
  speaks to hardware -- the closed loop in 5.7 -- is a procedure validated against a
  *modelled* error, not a measured one.

**7.4 No comparison to the base version's absolute numbers.**  ext_07's arm C is the
base's own trainer, so the *comparison* is authoritative, but the surrounding setup
is not the base's, so ext_07's R^2 values should not be read against `FINDINGS.md`.

---

## 8. Methodological defects found in the extension's own work

Recorded because the base version records its retractions, and because almost every
one of these pointed in the **flattering** direction for the extension.

**Half A**

1. **A vacuous anchor that counted as a pass.**  `ext_01`'s first version anchored
   on F2, whose layers are all `n_vars == 1`: the degenerate case was trivial and
   the negative control **silently skipped**.  Fixed: G is the primary anchor, and a
   vacuous T3 is now a FAILURE.
2. **A parameter count that flattered nothing, which is how it was caught.**
   Counting the identity combination's entries made the extension report 55 readout
   parameters against the base's 52 **for the same model**.
3. **The wrong competitor.**  `ext_02`'s first verdict compared `k=4` against the
   best of all `k` rather than against the full-rank readout it replaces.
4. **One random draw reported as a curve.**  The frozen-code column was a single
   `+-1` code, which made the curve non-monotone in `k`.
5. **Statistics with no power, twice.**  `ext_02` used the raw split spread where
   the standard error of the mean was right; `ext_03` used an unpaired tolerance
   that worked out to `0.0693` while the effect was `0.0006`.  Fixed (SEM; per-seed
   paired differences), after which the tests can fail and `k=2` does.
6. **The one that nearly became a published claim.**  `ext_04`'s first version used
   a fixed absolute ridge across a 20x range of parameters per sample, producing
   catastrophic fits (`R^2 = -0.85`) and a non-monotone curve -- read as **a
   crossover at which the rank constraint WINS**.  It was the regulariser failing,
   and the base version had already learned that lesson.  With a
   validation-chosen RELATIVE ridge the curve is monotone and **the crossover
   disappears**.
7. **A test that did not test what it claimed.**  `ext_05`'s first version asserted
   it exercised the sensor chain while fitting on the digital `Z`; the giveaway was
   the `low` device level coming out **bit-identical** to `none`.
8. **`--assert` semantics.**  In a boundary-finding experiment a low `k` failing is
   DATA, not a failed run.

**Half B**

9. **An exact factor of 2 in the operator.**  The first version put `2x` in the
   design matrix while still dividing by `2a` in the recovery, so every centre came
   out at exactly `0.5000` relative error while the width was right to `1e-6`.  Two
   self-consistent conventions exist and mixing them is the trap.  Caught because
   the verification compares against the DEVICE's centres, not the operator's own
   formula.
10. **An absolute probe floor where a relative one was needed.**  Points far outside
    the node's active region entered the log-space fit with huge `|log z|` and
    dominated it, giving sporadic ill-conditioned recoveries (`log` residual `1e-2`
    against the usual `1e-6`).  Fixed; the operator now REFUSES when too few
    informative readings survive.
11. **A probe "design" that was worse than random.**  `star_probes` built a long
    structured list and sliced it, which at `n=8`, `D=4` left an unbalanced set with
    no negative points and did WORSE than random (`7e-02`).
12. **A negative control that failed to fail** -- turned into the identifiability
    finding of section 4.3 rather than being quietly deleted.
13. **Mean where the worst case was the claim.**  At the minimum probe count a
    random set sometimes returns a perfect-looking fit of a nearly singular system
    together with a 100x wrong centre; averaging hides exactly the failure a
    placement procedure cannot afford.
14. **An unjustified absolute threshold that flipped a check between runs.**  C2
    required the worst cell below `1e-2`, a number this file never justified, and it
    disagreed with the discrimination margin (`20x`, passing) on alternate runs.
    Removed; the check is now decided on discrimination alone.
15. **Two cost-accounting fictions, both favouring the operators.**  `ext_07`
    charged arm A "one reading per probe per candidate" for probes it never reads
    (the proposals come from the residual profile over training inputs), and gave
    arm C twelve times its intended budget while the report printed equal numbers
    for both.  Fixed in two stages: first the cost unit became forward evaluations
    with arm A's count MEASURED, and then -- because the first fix was **incomplete**
    -- arm C's budget was derived from that measurement instead of from the fictional
    nominal.  Correcting it moved FD from `+0.2905` to `+0.1589`, i.e. the earlier
    framing had been **generous to the competitor**.
16. **A criterion that conflated two different events.**  `ext_08`'s first crossover
    rule treated "the gap fell inside the seed spread" as "FD reached the operator",
    and the verdict then claimed FD had caught up and that the operators were only
    238x cheaper -- **understating the operators**, since FD is behind at every
    budget tested.  The two events are now reported separately.
17. **A data table deleted by an edit.**  Fixing the verdict above removed the whole
    table-printing loop, leaving a report with a verdict and no data.
18. **A step claimed in the loop that does nothing.**  `ext_07` was reported as
    GROW + SOLVE + **REFINE** + GATE.  `ext_09`'s counters showed `REFINE` accepted
    1 change in 40 under one rule and 7 in 67 under another, while the returned
    geometry differed by at most one float32 ULP -- and turning the step off leaves
    the score identical.  The mechanism is that OP-G is a **fixed point** for a node
    that already explains its residual.  Fixed by defaulting it off and withdrawing
    the claim, not by deleting the operator, which is useful elsewhere.
19. **A fix that was asserted but not implemented.**  Item 15's first repair
    *claimed* arm C's budget came from the measured count; the code still used the
    fictional nominal.  Found by re-reading the claim against the code rather than
    by a test, which is why the defect list is audited.
20. **Stale call sites after a signature change, twice.**  Removing dead parameters
    from `refine_node` and `propose_candidates` broke a caller in `ext_09`, and the
    check that should have caught it was run **inside one file**.  Then an
    indentation slip left a loop body outside its loop, so `refine=True` crashed.
    Both fixed, and the check now spans every file in the extension.
21. **An assert whose verdict never reached the report.**  `ext_09`'s assert block
    was appended to a `Report` that had already been saved, so the file carried the
    claims without the check on them -- the same shape as a test that never fires.
    Moved before the save, and it also now covers all three parts rather than only
    P2 being non-empty.
22. **Noisy stderr treated as harmless.**  A matplotlib glyph warning wrote ~10 KB
    per run to stderr, which is where a real traceback goes; one crash was missed
    because stderr had been discarded.  The warning is emitted through matplotlib's
    **logger**, so filtering `warnings` did not silence it -- the working fix is a
    logger level, and stderr is now 0 bytes.
23. **A capacity failure read as a method failure.**  `ext_10`'s first verdict
    counted the two 10d cells as cells where "the operator fails to beat random",
    which would have reported a task no arm can fit as evidence about the placement
    rule.  Fixed by applying the base version's own `learning_ok` floor per cell and
    excluding the cells that cannot discriminate, naming them and the reason.
24. **A tolerance stricter than the hardware.**  `ext_12`'s first verdict demanded
    the recovered actuator gain to `1e-6` while the features come from a float32
    layer, so a width carries `~1e-7` and averaging 40 of them leaves `~1e-5`.  The
    run therefore reported "the loop does not close" while **all 8 levels had in
    fact recovered**.  Fixed to a `1e-4` relative bar with the float32 reasoning
    stated -- a check that cannot be met by the device is not a check.
25. **A repeated test that looked like a sweep, found by identical rows.**  `ext_11`
    swept `k` continuously, but `features()` supports only the device's two kernel
    channels, so **seven consecutive rows were the same measurement**.  The giveaway
    was that they were numerically identical.  Fixed by splitting into the two
    physically realizable channels plus a continuous WIDTH sweep, which is the base
    version's own twin error and is realizable.
26. **A "twin self-report" that was told the truth.**  `ext_11` computed the twin's
    column from the REALIZED (shifted) geometry, so the self-deception it exists to
    expose was zero by construction.  Fixed to evaluate the twin's own plan with the
    twin's own kernel.
27. **A non-win counted as a failure, three times over.**  `ext_13`'s first verdict
    counted both unresolved tasks as places where "the adversaries bite", which would
    have reported a ceiling (`moons`: every arm at `~0.98`) and a variance problem
    (`steps`: spread exceeding the effect) as evidence about the placement rule.  It
    is the same mistake as defect 23 in a new guise, which is why the diagnosis is
    now explicit: each non-win is classified as FLOOR, CEILING, SPREAD or GENUINE,
    because the four call for completely different responses.
28. **A baseline selected on the test set.**  `ext_14`'s first version took the
    maximum TEST score over its restarts as "best-found", which selects on the test
    set -- the exact error the base version warns about -- and inflates the baseline,
    making the placement rule look worse than it is.  Corrected to pick a restart by
    VALIDATION and report that run's test score; the correction moved the `N = 40`
    capture from `0.669` to `0.976`.  The size of the difference (`+0.0577` at worst)
    is why the report now states it.
29. **An assertion that could never pass.**  `ext_14`'s bar ("capture >= 0.90 at
    every node count") is FALSE for small N, so `--assert` would have been
    permanently red -- and a guard that always fails is not a guard.  The assert now
    encodes what the measurements support (the pool is not the limitation, and every
    capture ratio is finite) and the refuted hypothesis is reported, exactly as
    `ext_09` handles its own refuted alternative.
30. **A fixed-point iteration that was not a fixed point.**  `ext_15`'s first version
    updated the command as `cmd * cmd / realized`, which coincides with the correct
    `cmd * plan / realized` on the FIRST pass (when `cmd` starts at the plan) and
    then DIVERGES.  The tell was physical: the reported width error GREW with
    iteration (0.20 -> 0.56) and a large nonlinearity collapsed the score.  Fixed,
    after which two passes converge to a width error below 0.009 at every beta.
31. **A success bar that a perfect correction could not meet.**  `ext_15` measured
    the per-node inverse as "failing" at `sd = 0.05` because it was required to beat
    the open loop by 0.02 -- but at a small spread the open loop barely suffers, so
    that margin is not available to anyone.  The measured result was that it returned
    the error-free score to four decimals.  The bar is now "recovers the error-free
    score", which is the claim actually being tested.
32. **A checkpoint fingerprint that covered the sweep extent instead of the cell.**
    ext_10's and ext_13's fingerprints included the SEED COUNT, so extending
    `--seeds 3` to `--seeds 9` invalidated every existing cell and re-ran them all --
    defeating the incremental extension that checkpointing exists for.  Fixed by
    fingerprinting what actually determines a cell's value (task, architecture,
    trainer settings), after which adding seeds or tasks is incremental by
    construction; verified by rebuilding 72 cells in 2 s.
33. **An imprecise LOSS filed as an unresolved comparison.**  `ext_13`'s diagnosis
    classified "the mean is AGAINST the operators and the spread is large" as SPREAD,
    which reads as "no resolution, direction unknown" when the direction is in fact
    against the method.  The sign is now checked first, so an imprecise loss is
    reported as a loss.
34. **Two conclusions that 12 seeds reversed, both against the method.**  At 3 seeds
    `ext_13` reported the operator arm as the BEST on `steps` and as NOT the most
    negative on pure noise.  At 12 seeds it is the WORST on `steps` and it IS the most
    negative on noise.  The 3-seed report had itself said "more seeds would settle
    it", so this is the settling rather than a new finding -- and it is the clearest
    argument in this document for writing that sentence down whenever it is true.
35. **A device level that was never connected to the layer.**  `ext_17`'s first run
    swept the diffusion sensor chain over four levels and got **four identical rows**:
    the layer was built with `cell=None`, so it created its own default substrate and
    silently ignored `set_diffusion_defaults(level=...)`.  The tell was the identical
    rows, exactly as in defects 2 and 25.  Fixed by threading
    `abl_common.make_cell(substrate)` into the layer builder; the chain then bites
    monotonically (`1.97e-15` at `none` to `2.25` at `high`).
36. **A column labelled for a test it was not running.**  `ext_17`'s end-to-end table
    had a "diffusion+chain" column, but the global level had been reset to `none`
    after the previous section, so that column was a no-op chain and reproduced the
    others to the last digit.  Three identical columns were the tell.  Fixed to run at
    a real level (`low`), where the operator arm scores `+0.5952` against random
    `+0.1498`.
37. **A bar the circuit substrate cannot meet by construction.**  `ext_17`'s first
    verdict required a log-fit residual below `1e-9` on BOTH substrates, but the
    circuit layer computes in float32 and sits near `1e-6`.  "Exact" now means each
    substrate reaches its OWN floor -- the same mistake as defect 24, in a new place.
38. **An index of open questions that stopped matching the work.**  Sections 7 and 11
    -- the two places a reader looks to choose the next experiment -- had drifted
    furthest behind.  Section 11 was two interleaved lists whose numbering collided
    (`1-4`, then `3-6`), three of its items had already been answered by later
    experiments, one was listed twice, and the **substrate gap, the largest structural
    gap this extension had, had never been listed at all**; section 7.3 still described
    the scope as "one task, one architecture, one scale" after `ext_10`, `ext_13` and
    `ext_17` had varied dimension, architecture, task family and substrate.  This one
    did not flatter the extension the way the others did -- it **hid** a gap, and a
    reader deciding from it would have run the wrong experiment.  Both sections are
    rebuilt, and the previous list's items are recorded as closed rather than deleted.
39. **Sections inserted into the middle of another section.**  `ext_16`'s and `ext_17`'s
    write-ups were added immediately after the paragraph they extended, which put
    `### 5.11` and `### 5.12` PHYSICALLY BEFORE `### 5.9` and `### 5.10`, with both
    buried inside section 5.8's discussion.  Nothing was lost, but numeric order is how
    a reader navigates, and a check of the file's own heading order caught it.  Moved to
    `5.1 ... 5.12` in order, with the joins verified.

Plus ordinary bugs, each of which produced a wrong number before it produced an
error: the bilinear solve `min ||Z W_g^T C^T - Y||` written as a plain ridge solve
(**three separate times**), a single `(k,k)` matrix applied to every stage of a
multi-stage model (**twice**), a variable read outside its branch (**twice**), a
function referring to a variable from its caller's scope, a per-seed dict list
indexed as a curve list, a `(n,2)` activation matrix flattened against `(n,)` probe
points, and an edit that left an orphaned function body.

---

## 9. Scope

* **Toy by instruction.**  Half A: 256 samples, 40 particles, `m = 8`, a synthetic
  multi-output target.  Half B: 40 nodes and 256 samples, on the project's own synthetic
  tasks -- but no longer one dimension or one architecture, since `ext_10` varied both,
  and `ext_20` later showed that the `10d` end of that axis is out of reach here.
  Nothing here is evidence about a real device, a real dataset, or a large model.
* **Two substrates, and "any" is still unproven.**  Half B now runs on the circuit cell
  and the diffusion cell (`ext_17`), and the two agree to the last digit on the
  undamaged paths.  But both were built here to reproduce the same engineered kernel, so
  what is established is that the operators do not depend on WHICH of the two produced
  the field -- not that they survive a substrate whose response is not Gaussian in the
  input.  What they need is stated exactly: log-linearity in `[-|x|^2, x, -1]`.
* **Half A is a NO-OP for the project's scalar headline.**  `m = 1` means a `1 x N`
  readout, whose rank is already 1, so there is nothing to constrain.  Half A is
  relevant only where `m > 1` is genuine -- multi-class output, or multi-channel
  intermediate layers.  Half B has no such restriction: it applies to any node
  network.
* **Half A holds the geometry fixed** in ext_02/04/05 (deliberate: it isolates the
  readout and runs in seconds).  Only ext_03 trains it, and its loop mirrors rather
  than calls the base trainer.
* **Do not read ext_04's ratio column against the base's 2.19x boundary.**  It
  reaches 20x parameters per sample and still calibrates, which does not contradict
  the base's failure at 2.19x, because the geometry there is fixed while the failing
  base arm also searched its geometry by finite differences, and the ridge is
  validation-chosen per arm.
* **The operator's `k` is a hardware constant.**  Half B's results are conditional on
  `k` being known, which on this hardware it is.

---

## 10. Artifact map

| file | what it is |
|---|---|
| `extensions/multigraph/DESIGN.md` | the design memo, including the measured correction to constraint C4 |
| `extensions/multigraph/graphs.py` | `Graph`, `Linear`, `MultiGraphStack`, cache isolation, `graphs_from_base` |
| `extensions/multigraph/operators.py` | **the dedicated operators**: OP-G node solve, OP-C node proposal, leverage subset selection, the designed probe set, `identify_placement` |
| `extensions/multigraph/trainer.py` | **the operator-only trainer**: grow, solve, gate (refine off by default, measured inert) |
| `extensions/multigraph/ext_01_baseline.py` | the correctness anchor (5 tests) |
| `extensions/multigraph/ext_02_rank_budget.py` | rank budget at a fixed geometry; refutes the frozen code |
| `extensions/multigraph/ext_03_geometry_rank.py` | the rank budget under geometry training, with paired statistics |
| `extensions/multigraph/ext_04_calibration_samples.py` | calibration-sample sweep; no crossover, no measurement saving |
| `extensions/multigraph/ext_05_diffusion_rank.py` | the second substrate, device levels, the uncalibrated collapse |
| `extensions/multigraph/ext_06_node_operator.py` | verification of the geometry operator (C1-C6) |
| `extensions/multigraph/ext_07_operator_trainer.py` | the trainer against random geometry and the cost-matched FD search |
| `extensions/multigraph/ext_08_long_budget.py` | the resumable long-budget check |
| `extensions/multigraph/ext_09_probe_selection.py` | subset selection, device-side placement identification, and the inertness of REFINE |
| `extensions/multigraph/ext_10_generalization.py` | a second task and a second architecture; the capacity exclusion |
| `extensions/multigraph/ext_11_twin_error.py` | what the device delivers when the twin is wrong; the self-deception |
| `extensions/multigraph/ext_12_closed_loop.py` | the repair: measure the placement, invert the actuator's gain, re-command |
| `extensions/multigraph/ext_13_task_family.py` | the base version's own adversary tasks; the resolution diagnosis; 12 seeds |
| `extensions/multigraph/ext_14_placement_bound.py` | the empirical bound on the placement rule, and its node-count dependence |
| `extensions/multigraph/ext_15_nonmultiplicative.py` | the repair against per-node, nonlinear and drifting error; the per-node estimator |
| `extensions/multigraph/ext_16_steps_diagnosis.py` | which level fails on `steps`; the unusable null and the retraction |
| `extensions/multigraph/ext_17_second_substrate.py` | half B on the diffusion cell; the amplitude-invariance prediction |
| `extensions/multigraph/ext_18_linearity_trade.py` | the linearity trade, measured against a more-nodes control |
| `extensions/multigraph/ext_19_restart_repair.py` | the restart repair, and the failure of held-out selection |
| `extensions/multigraph/ext_20_ten_d.py` | 10d raised until the cells discriminate, and the answer that they do not |
| `extensions/multigraph/ext_21_selector_guard.py` | the guard measured rather than chosen, and why none exists |
| `extensions/multigraph/ext_22_criterion_probe.py` | the criterion against the outcome; the bias-variance answer |
| `extensions/multigraph/trainer.py` | `features` gained `substrate` and `measured` (ext_17); `node_amplitude`; `n_random` (ext_14); `restarts=M` (ext_19) |
| `outputs/` | the reports, `ext_<NN>_<name>.txt` and a matching `_plot.png`, one pair per experiment 01-22 |
| `outputs/ext_ckpt/` | the resumable sweep cells: 15 for ext_08, 18 for ext_10, 18 for ext_13 |

Reproduce with:

```
python extensions/multigraph/ext_01_baseline.py --assert
python extensions/multigraph/ext_02_rank_budget.py --assert --splits 5
python extensions/multigraph/ext_03_geometry_rank.py --assert --epochs 60 --seeds 5 --ks 8,4,3,2
python extensions/multigraph/ext_04_calibration_samples.py --splits 4
python extensions/multigraph/ext_05_diffusion_rank.py --assert --splits 3
python extensions/multigraph/ext_06_node_operator.py --assert
python extensions/multigraph/ext_07_operator_trainer.py --assert --seeds 3 --rounds 3
python extensions/multigraph/ext_08_long_budget.py --assert        # resumes; ~5 min cold, ~1 s warm
python extensions/multigraph/ext_08_long_budget.py --status
python extensions/multigraph/ext_09_probe_selection.py --assert --seeds 3
python extensions/multigraph/ext_10_generalization.py --assert --seeds 3
python extensions/multigraph/ext_10_generalization.py --status
python extensions/multigraph/ext_11_twin_error.py --assert --seeds 3
python extensions/multigraph/ext_12_closed_loop.py --assert --seeds 3
python extensions/multigraph/ext_13_task_family.py --assert --seeds 3
python extensions/multigraph/ext_13_task_family.py --status
python extensions/multigraph/ext_14_placement_bound.py --assert --seeds 3
python extensions/multigraph/ext_15_nonmultiplicative.py --assert --seeds 3
python extensions/multigraph/ext_16_steps_diagnosis.py --assert
python extensions/multigraph/ext_17_second_substrate.py --assert
python extensions/multigraph/ext_18_linearity_trade.py --assert
python extensions/multigraph/ext_19_restart_repair.py --assert
python extensions/multigraph/ext_20_ten_d.py --assert
python extensions/multigraph/ext_21_selector_guard.py --assert
python extensions/multigraph/ext_22_criterion_probe.py --assert
```

All twenty-two exit non-zero when their own published claim fails, and every one of them
runs in seconds except ext_03 (~6 min), the cold sweeps of ext_08 (~5 min), ext_14
(~40 s), ext_16 (~95 s), ext_18 (~135 s), ext_19 (~65 s), ext_20 (~110 s), ext_21 (~35 s) and ext_22 (~80 s).

The three resumable sweeps write 51 cells in total to `outputs/ext_ckpt/`; a warm
`--report-only` rebuild of any of them takes about a second.

---

## 11. What would come next, in order of value

**This section is rewritten because it had become the least reliable part of the
document.**  It was two interleaved lists whose numbering collided (`1-4`, then `3-6`),
three of its items had already been answered by experiments added afterwards, one item
was listed twice under two different numbers, and the **SUBSTRATE gap -- the largest
structural gap this extension had -- had never been listed at all**.  Tidiness is not
the point: this is the index a reader uses to choose the next experiment, and choosing
from a stale index is how the wrong experiment gets run.

**Closed since the previous list**, recorded rather than deleted because the correction
trail is the point of this document:

| was listed as | closed by | verdict |
|---|---|---|
| "a third task family, not just a third dimension" | ext_13 | done, and it is the base version's own adversary tasks: `hetero` survives, `steps` loses, and on `noise` the operator is the most confidently wrong |
| "the closed loop against a non-multiplicative error" | ext_15 | done: the pooled estimator fails SILENTLY, the per-node inverse recovers the error-free score |
| "optimality or a bound for the placement rule" | ext_14 | done: capture `0.976` at the operating point, `0.282` at `N = 4` -- and note it had been listed TWICE, as items 3 and 6 |
| *(never listed)* | ext_17 | half B now has evidence from **two substrates**; the omission is the reason this section was rebuilt |

**Still open, ranked by what would change a conclusion:**

1. **The trade is MEASURED (5.13), and the cheap form says keep the linearity.**  A
   nonlinear readout on the same nodes gains `+0.0295`/`+0.0660`/`-0.0062` on
   `4d`/`steps`/`hetero`, while spending the budget on more nodes gains
   `+0.1252`/`+0.1599`/`+0.0996` -- and the nonlinear arm must SEARCH `3200` strengths
   against the node arm's `320` calibrated ones.  **What remains open is the one thing
   that could overturn it**: `W` and `b` were drawn, not trained, and the gain saturates
   and falls with hidden width, so a TRAINED inner layer is the untested case.  A full
   nonlinear combination with its own calibration story (item 8) is the way to settle
   it, and it is now a question with a measured prior against it rather than an open
   end.

2. **CLOSED by ext_19 (5.14) and ext_21 (5.16), and it RELOCATED the defect.**  The repair reaches the
   ceiling -- `M = 4` on `steps` lands on `+0.2045`, which is the best-found ceiling
   `ext_16` measured -- and the `steps` mechanism is now partly explained: the proposal
   family is **at chance** there (`42.5%` against an equal-sized random pool, where
   chance is ~50%) while winning clearly on `4d` (`63.3%`), so the log-linear residual
   profile carries no signal on a discontinuous target and the restarts are a lottery
   over it.  **What it opened: held-out `R^2` alone cannot select among restarts.**  The
   selector picked `-7.5180` on `steps` at `M = 16` and `-0.4235` on the SMOOTH control
   at `M = 2`, giving up `+7.7794` at the worst level, because a restart can build a
   geometry whose weights are huge and whose prediction explodes outside the basis
   support -- the base version's own `shift` warning arriving here.  The open item is
   therefore a **GUARD on the selector**, not the gap.**  ext_21 measured three
   candidate guards and **none works**: the hypothesised prediction-spread statistic
   is REFUTED, and on `steps` the validation-selected restart scores `-7.5182` while
   its own pool's mean is `-0.3404`, so selection picks worse than a random member of
   the pool.  **`restarts=M` is therefore not shipped as a fix** -- `restarts=1`
   stays the default and the code documents the parameter as unsafe above 1.  **The
   defect is upstream, and ext_22 then LOCATED it (5.17): the criterion is weak on
   BOTH tasks (`+0.097`/`+0.191` rank agreement) and leaves half the pool's spread on
   the table, but REPLACING it is catastrophic (test `-0.19`/`-1.26` against val
   `+0.84`/`+0.91`), because ranking ~70 candidates per step on a 51-sample split
   overfits it.  It is a BIAS-VARIANCE trade and the SAMPLE COUNT binds: the shipped
   rule is defensible not because it is accurate but because it is STABLE.**

3. **CLOSED by ext_20 (5.15): out of reach at this scale, now stated.**  At 360 nodes
   -- `36` per input dimension -- the best arm reaches `+0.0014` and still does not
   clear the `0.05` floor.  The dimension axis has a hole above 4d and nothing here is
   evidence about high-dimensional inputs in either direction.  Recorded alongside it,
   because it is the direction the method would want but cannot claim: the operator arm
   is the best of the three at every node count, with `op - random` at `+0.0452`,
   `+0.1849`, `+0.0778`.

4. **A genuine `m > 1` setting for half A.**  Everything there rests on a synthetic
   multi-output target, and `m = 1` makes half A a no-op.  The repository has a 4D
   classification task, and classification is where `m` channels are real.  This is the
   same KIND of gap the substrate one was: the setting does not exercise the thing the
   claim is about.

5. **Beyond two substrates.**  ext_17 shows the operators do not depend on WHICH of the
   two cells produced the field, and the prediction states exactly what they need --
   log-linearity in `[-|x|^2, x, -1]`.  But both cells were built here to reproduce the
   same engineered kernel, so nothing here tests a substrate whose response is not
   Gaussian in the input at all.  "Any physical system" is still the widest claim in
   this document and the least covered.

6. **The drift RATE.**  ext_15 established that a placement correction has a shelf life,
   because a drift after the measurement makes it stale.  How often a deployment must
   re-measure is set by the drift it expects, and no measurement here sets that number
   for a real cell.

7. **`ext_03` against the real base trainer.**  It mirrors the base's loop rather than
   calling it, because the base's closed-form hook has no place for a rank constraint.
   Wiring that hook would make the comparison authoritative instead of parallel.

8. **A non-linear combination in full, with a replacement for the closed-form
   calibration.**  Item 1 is the cheap measurement of whether the trade is worth making;
   this is the work of making it, and it is deliberately last because it forfeits the
   operator the rest of half B is built on.
