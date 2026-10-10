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
>
> **And section 12 audits the BASE version's headline from here**, because this
> extension does not modify the base: raising the base's task sweep from 3 seeds to 9
> moves it from `9/14` to **`8/14`**, with one cell flipped, one strengthened, and six
> decisive cells holding.  The base's `FINDINGS.md` still carries the 3-seed number and
> does not yet point here.

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

## 8. Scope

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

## 9. Artifact map

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
| `outputs/abl_06_task_sweep_s9.txt` | **section 12's evidence**: the BASE version's task sweep at 9 seeds, run unmodified |
| `outputs/abl_06_task_sweep_e160.txt` | **section 12.5's evidence**: the same sweep at 160 epochs, where the prediction held |
| `extensions/multigraph/ext_23_criterion_audit.py` | **section 12.7**: recomputes every verdict with `geometry` excluded |
| `extensions/multigraph/ext_24_particle_budget.py` | **section 12.9**: the particle-budget tail, via runtime ARCHS keys |
| `extensions/multigraph/ext_25_budget_matched.py` | **section 12.10**: the same question with the budget HELD FIXED |
| `extensions/multigraph/ext_26_base_family.py` | **section 12.11**: the same design on the BASE's own three-layer shape |
| `extensions/multigraph/ext_27_nonmonotone.py` | **section 12.12**: the mechanism -- one ridge-scale constant |
| `extensions/multigraph/ext_28_headline_ridge.py` | **section 12.13**: the base's headline at `lam=auto` -- it does not move |
| `extensions/multigraph/ext_29_reservoir.py` | **section 12.14**: the reservoir variant -- the constraint is selection |
| `extensions/multigraph/ext_30_equal_params.py` | **section 12.15**: the equal-parameter comparison, constructed |
| `extensions/multigraph/ext_31_substrate_scope.py` | **section 12.16**: which substrates are in scope, as a test |
| `extensions/multigraph/ext_32_residual_mode.py` | **section 12.17**: the `residual_mode` claim, controlled |
| `extensions/multigraph/ext_33_gate_power.py` | **section 12.18**: the gate at its boundary, and the counterexample's shape |
| `extensions/multigraph/ext_34_deployment_harm.py` | **section 12.19**: the deployment isolated -- no second trajectory |
| `extensions/multigraph/ext_35_seeding.py` | **section 12.20**: the seeding fix, verified -- including the check that isolates it |
| `extensions/multigraph/ext_36_trained_inner.py` | **section 12.21**: the trained inner layer, and the cost question it surfaced |
| `extensions/multigraph/ext_37_m_real.py` | **section 12.22**: a genuine `m > 1` setting -- the item names the wrong task |
| `extensions/multigraph/ext_38_third_substrate.py` | **section 12.23**: a third substrate -- an anharmonic trap, not Gaussian |
| `extensions/multigraph/ext_39_nonlinear_free.py` | **section 12.24**: item 8 -- the replacement it asks for is not needed |
| `extensions/multigraph/ext_40_drift_rate.py` | **section 12.25**: item 6 -- the drift rate, and the channel it names |
| `extensions/multigraph/ext_41_authoritative.py` | **section 12.26**: item 7 -- the blocker is not the hook |
| `extensions/multigraph/ext_42_selector_samples.py` | **section 12.27**: item 2 -- the lever the diagnostic named |
| `extensions/multigraph/ext_43_calibration_in_deep.py` | **section 12.28**: the closed-form calibration inside the deep trainer |
| `extensions/multigraph/ext_44_regression_defect.py` | **section 12.29**: the regression prediction dropped the accumulation, and the fix |
| `extensions/multigraph/ext_45_deep_benchmark.py` | **section 12.30**: the deep regression path, benchmarked -- and beaten by the null |
| `extensions/multigraph/ext_46_classification_benchmark.py` | **section 12.31**: HiMALND's classification mode -- beaten by the null too |
| `extensions/multigraph/ext_47_cifar_baseline.py` | **section 12.32**: the CIFAR line against the project's own standard -- unresolved |
| `extensions/multigraph/ext_48_lookup_table.py` | **section 12.33**: the lookup table -- the precondition it relaxes, and the defect class it exposed |
| `extensions/multigraph/ext_49_yukawa_family.py` | **section 12.34**: the h-indexed family -- reaching Yukawa, and the parameterisation lesson |
| `extensions/multigraph/ext_50_geometry_only.py` | **section 12.35**: geometry-only search -- 12.28's bound is tight, and is now a measurement |
| `extensions/multigraph/ext_51_presensor.py` | **section 12.36**: pre-sensor superposition -- DESIGN 7's last open question, answered |
| [`BASE_CORRECTIONS.md`](BASE_CORRECTIONS.md) | **the work list**: every base claim section 12 corrects, with its line number, its replacement, and its evidence |
| `outputs/e13_parallel_benchmark_report.txt` | **section 12.8**: regenerated `train_01`; identical across K by construction |
| `outputs/e14_parallel_spsa_report.txt` | **section 12.8**: regenerated `train_02`; genuinely K-dependent, and it passes |
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
python extensions/multigraph/ext_23_criterion_audit.py --assert
python extensions/multigraph/ext_24_particle_budget.py --assert
python extensions/multigraph/ext_25_budget_matched.py --assert
python extensions/multigraph/ext_26_base_family.py --assert
python extensions/multigraph/ext_27_nonmonotone.py --assert
python extensions/multigraph/ext_28_headline_ridge.py --seeds 9 --assert
python extensions/multigraph/ext_29_reservoir.py --assert
python extensions/multigraph/ext_30_equal_params.py --assert
python extensions/multigraph/ext_31_substrate_scope.py --assert
python extensions/multigraph/ext_32_residual_mode.py --seeds 3 --epochs 80 --assert
python extensions/multigraph/ext_33_gate_power.py --seeds 3 --assert
python extensions/multigraph/ext_34_deployment_harm.py --seeds 3 --assert
python extensions/multigraph/ext_35_seeding.py --assert
python extensions/multigraph/ext_36_trained_inner.py --assert
python extensions/multigraph/ext_37_m_real.py --assert
python extensions/multigraph/ext_38_third_substrate.py --assert
python extensions/multigraph/ext_39_nonlinear_free.py --assert
python extensions/multigraph/ext_40_drift_rate.py --assert
python extensions/multigraph/ext_41_authoritative.py --assert
python extensions/multigraph/ext_42_selector_samples.py --assert
python extensions/multigraph/ext_43_calibration_in_deep.py --assert
python extensions/multigraph/ext_44_regression_defect.py --assert
python extensions/multigraph/ext_45_deep_benchmark.py --assert
python extensions/multigraph/ext_46_classification_benchmark.py --assert
python extensions/multigraph/ext_47_cifar_baseline.py --assert
python extensions/multigraph/ext_48_lookup_table.py --assert
python extensions/multigraph/ext_49_yukawa_family.py --assert
python extensions/multigraph/ext_50_geometry_only.py --assert
python extensions/multigraph/ext_51_presensor.py --assert
```

All fifty-one exit non-zero when their own published claim fails, and every one of them
runs in seconds except ext_03 (~6 min), the cold sweeps of ext_08 (~5 min), ext_14
(~40 s), ext_16 (~95 s), ext_18 (~135 s), ext_19 (~65 s), ext_20 (~110 s), ext_21 (~35 s) and ext_22 (~80 s).  ext_23 is read-only and takes seconds; ext_24 is ~9 min cold and ~10 s warm; ext_25 is ~14 min cold, ext_26 ~20 min cold, ext_27 ~10 min cold, ext_28 ~36 min cold, ext_29 seconds, ext_30 ~8 min cold, ext_31 under a second and ext_32 ~50 min cold and ext_33 ~13 min cold (cached) and ext_34 ~22 min cold and ext_35 ~14 min cold and ext_36 ~4.5 min cold and ext_37 ~20 s and ext_38 under a second and ext_39 ~2 min cold and ext_40 and ext_41 a few seconds each; ext_42 ~7-15 min, ext_43/44/45 seconds each; ext_46 ~1-3 min; ext_47 ~20 s; ext_48 ~1 s; ext_49 ~1 s; ext_50 ~15 s; ext_51 ~1 s.

The three resumable sweeps write 51 cells in total to `outputs/ext_ckpt/`; a warm
`--report-only` rebuild of any of them takes about a second.

---

## 10. What remains, and what would change a conclusion

**Everything this extension set out to measure now has a measured answer.** What is left is not
a list of open experiments but three named limits, listed here so that a reader can see the shape
of the evidence and not only its results.

1. **One open design question: the third arrangement, `phi(sum_j W_j g_j)`.** The other two
   orders are settled. `sum_j W_j phi(g_j)` is nonlinear in the readings and LINEAR in the
   strengths, so the closed-form solve survives it -- and all four elementwise `phi` measured
   WORSE than the linear readout, i.e. free but useless. `W = C . W_graphs` is a linear
   per-graph combination and is half A itself. The remaining order is nonlinear in the
   STRENGTHS, it does lose the operator, and it needs its own calibration story; 12.21's
   `-0.0264` is the nearest evidence and it loses.

2. **`ext_03`'s loop is a mirror, and its fidelity is untested.** The parallel status is a
   NECESSITY rather than a shortcut -- the base trainer is scalar-output by construction and
   half A's rank question is inherently multi-output (12.26) -- so its disclaimer is the
   strongest form the comparison can take. But the mirroring has never been checked against an
   authoritative implementation, which is a real limit on the rank results.

3. **Every number here is at one scale, and no device was built.** The tasks are synthetic, the
   particle counts are in the tens, the input dimension is between 2 and 10, and the three
   substrates are models of devices rather than devices. The operators have never been run
   against a fabricated cell, and the diffusion cell's PDE verification checks the MODEL's
   kernel against physics rather than a manufactured cell against the model.

**And one thing that is not a limit:** the extension's results do not depend on the base
version's deep architecture. Where that architecture was measured it lost, and 12.30-12.31
record it.

---

## 11. What the audit of the base version found

**The results below are what the extension's audit of the base version FOUND.** The method was
to take a claim, build the measurement the claim requires, and report whichever way it came out;
the method is not reproduced here, only what it produced. Each section is an experiment with a
script, a reproduce command and a tracked artifact.

**Several came out against the base version's headline.** The substrate precondition (12.16)
narrows "any physical system" to something checkable; the direct-residual claim is refuted with
its sign reversed (12.17); the reservoir's problem is a selector rather than movable centres
(12.14); and the deep architecture is beaten by a zero-forward baseline on every task this
repository can build (12.30-12.31). **Others came out in its favour** (12.13, 12.15, 12.23).

**Where the measurements required the base version's own documents to change, they were
changed, and what qualifies its numbers is stated there** -- see `FINDINGS.md` section 9 at the
repository root.

**The subsections keep their `12.x` labels** from when this material sat under a section numbered
12, because the rest of the repository cites them by those numbers -- `THEORY.md` section 3.2
cites 12.17 and section 6 cites 12.16 and 12.23, and the base's `FINDINGS.md` cites 12.8 and
12.18-12.19. Renumbering them would break those citations for no gain.

### 12.14 The reservoir variant: item 5's concern is real, its implied reason is not (ext_29)

**The open item, in the base's own words** (section 11, item 5):

> "**Reservoir variant.**  On an excitable medium the only trainable part is the readout,
> which is section 6's recommendation taken to its limit -- but it requires giving up
> movable centres and per-particle widths."

**What was tested.**  Exactly that extreme: the geometry is NEVER trained, no probe is
spent, no gradient is taken -- one fixed random instance plus ONE closed-form stagewise
solve, the same solve the base's calibration uses.

**What a reservoir saves, and what it does not -- which is what makes the comparison
fair.**  It saves the SEARCH, not the SOLVE: the solve still costs
`n_train * sum_k N_k` measurements (the base's own `readout_calibration_cost`).  So the
reservoir has exactly one free knob -- **redrawing the geometry**, which an excitable medium
does permit, since a different device is a different random instance -- and the honest
question is at equal cost:

> **does best-of-R redraws + a solve beat searching the geometry + a solve?**

with `R = 1 + P/S`, where `P = 2 * budget * epochs` is the hybrid's probe cost.  Measured:
`S = 9 984` on every cell (all the base's families hold 52 particles, all tasks 256 samples),
`P` from `32 000` to `80 000`, so **`R` from 4 to 9**, and `R*S <= P + S` holds by
construction.

**THE RESULT.**

| comparison | cells won by best-of-R |
|---|---|
| against **`hybrid`** (search + solve) | **1/14** |
| against **`geometry`** (search only, full probe budget) | **9/14** |
| against **`readout`** (gradient-trained readout) | 7/14 |
| against **a single draw** (its own R=1) | **7/14** |

> **THE RESERVOIR DOES NOT HOLD UP AGAINST THE ARM THAT SEARCHES AND SOLVES -- BUT NOT
> FOR THE REASON ITEM 5 IMPLIES.**

**(a) It is not capacity.**  With no probes and no gradient at all, best-of-R beats the
geometry arm -- which spends the entire probe budget -- in **9 of 14** cells.  So "only the
readout is trainable" is not by itself disqualifying; what wins is search PLUS solve, and
the reservoir has only the solve.

**(b) It is SELECTION.**  A reservoir has one knob, redrawing, and **a redraw has to be
picked**.  In about half the cells **the pick is worse than not picking at all** -- worse
than the very first draw (`4d/F2` +0.2644 against +0.3143; `shift/G` +0.0793 against
+0.1317).  And P3 says what governs it:

| | |
|---|---|
| doubling the validation slice | **mean +0.0204**, better in **10 of 14** cells |
| `steps/F2` under that change | **-0.2807 -> -0.0470** |
| an ORACLE pick (best by TEST) would add | **+0.0769** on average |

The held-out slice is 64 samples, and the selected draw's held-out minus test gap reaches
**+0.41** on `shift/G` and `shift/F2`.  **The binding constraint is model selection on few
held-out samples, not the frozen geometry.**

**(c) And it is `ext_21`'s failure mode in a different setting** -- the operator selector: there, no guard worked and selection did worse than the pool mean; here,
redraw selection loses to no selection.  **Two independent designs, one cause.**

> **SO ITEM 5 IS ANSWERED: the movable centres and per-particle widths ARE earning their
> keep, but the reason is not that a frozen geometry is too weak -- it is that a family of
> instances needs a reliable way to choose among them, and on this data there is not one.**
> That also makes the item's implied remedy wrong: giving the reservoir movable centres is
> one fix, and a better selector would be another.

**THE SELECTION DISCIPLINE.**  The redraw is chosen on a held-out slice of TRAINING, never
on test.  The oracle is printed as a bound and is explicitly **not** a claim -- it is there
to say how much a perfect selector would be worth, which is what makes (b) measurable.

**SCOPE.**  Seven tasks, two architectures, 3 seeds, the diffusion substrate, the base's own
`effective_budget` and its default `closed_form_lam=1e-3`.  12.13 showed the FRONTIER verdict
is insensitive to that lambda, so it is not a live confound here; 12.12 showed the LOW-budget
comparison is not, **so no low-budget claim is made from this run**.

### 12.15 The equal-parameter comparison IS constructible, and the calibration substitutes for allocation (ext_30)

**The open item, in the base's own words** (section 11, item 2):

> "**Equal-parameter-count comparison.**  Not constructible in this family (geometry costs
> `D+1` per particle against `1` for a scalar readout), so the parameter-count rule of
> section 5 can only be tested by varying the ratio."

**IT IS CONSTRUCTIBLE, AND EXACTLY.**  A layer with input dim `D`, output width `V` and `N`
particles stores `(D+1)*N` geometry numbers and `V*N` readout numbers, so its geometry share
is `(D+1)/(D+1+V)` -- **`V` is a continuous knob on the SHARE.**  Holding the total fixed is
then an integer problem, solved exactly with two layers:

    T = (D+1+V)*N1 + 3*N2        (layer 2 has dim 1, so 3 per particle)

with `N1 = T // (D+1+V)` and `N2` the remainder over 3.  All five points below sit at **the
same single-channel total of 312**, across a share sweep of 83% to 26%:

| arch | V | particles | geometry | readout | total | share | readout/sample |
|---|---|---|---|---|---|---|---|
| `Q1` | 1 | (52,) | 260 | 52 | **312** | **83.3%** | 0.203x |
| `Q2` | 2 | (42, 6) | 222 | 90 | **312** | 71.2% | 0.352x |
| `Q4` | 4 | (34, 2) | 174 | 138 | **312** | 55.8% | 0.539x |
| `Q8` | 8 | (24,) | 120 | 192 | **312** | 38.5% | 0.750x |
| `Q16` | 16 | (14, 6) | 82 | 230 | **312** | 26.3% | 0.898x |

**The budget is FIXED across the axis**, per 12.10: `effective_budget` would otherwise move
the per-epoch budget along with the share and confound the sweep.

**THE RESULT, 3 seeds.**

| arch | share | readout/sample | geometry | readout | full | **hybrid** | **margin** |
|---|---|---|---|---|---|---|---|
| `Q1` | 83.3% | 0.203x | +0.0737 | +0.3839 | +0.2377 | **+0.5884** | +0.1988 |
| `Q2` | 71.2% | 0.352x | +0.1298 | +0.3499 | +0.3265 | **+0.5890** | +0.1979 |
| `Q4` | 55.8% | 0.539x | +0.0962 | +0.2331 | +0.1897 | **+0.6237** | +0.3498 |
| `Q8` | 38.5% | 0.750x | +0.0657 | +0.0868 | +0.0891 | **+0.5522** | **+0.4770** |
| `Q16` | 26.3% | 0.898x | +0.0113 | +0.1494 | +0.1721 | **+0.5700** | +0.3662 |

| arm | spread across the share axis |
|---|---|
| geometry | 0.0839 |
| **readout** | **0.3191** |
| full | 0.2629 |
| **hybrid** | **0.0912** |

> **(1) SECTION 11 ITEM 2 IS WRONG ABOUT CONSTRUCTIBILITY.**  The comparison exists, it is
> exact, and it is run here.  What made it look impossible is that `V` was not treated as a
> knob on the share -- the same knob the base's own section 5 box already uses when it notes
> that enabling the second channel moves F2 from an 81% share to 68%.

> **(2) THE CALIBRATED ARM IS NEARLY ALLOCATION-INDEPENDENT, AND THAT IS WHAT THE
> CALIBRATION BUYS.**  **The total is constant in every row, so every bit of spread is the
> ALLOCATION rather than the count.**  The search arms move a great deal -- the readout arm
> by **0.3191** -- while the hybrid moves by **0.0912**, and the hybrid's margin **GROWS as
> the allocation worsens**: `+0.1988, +0.1979, +0.3498, +0.4770, +0.3662` down the axis.
> **So the calibration is a SUBSTITUTE FOR A GOOD ALLOCATION: it pays most exactly where the
> search arms do worst.**  That is a stronger form of section 6's recommendation than
> "calibrating helps" -- it says what the calibration is buying.

> **(3) SO THE PARAMETER-COUNT RULE OF SECTION 5 IS NOT THE WHOLE STORY.**  At one fixed
> count the arms still move by up to 0.32, so where the parameters sit carries information
> the count does not -- **for the search arms.**  The rule survives for the calibrated arm,
> which is the configuration the base actually recommends.

**SCOPE.**  One task (4d), one fixed budget, 3 seeds, 40 epochs, one target total of 312.
A larger total or a second task could move the spreads, and **nothing here bounds them**.
The single-layer-plus-one-scalar-layer shapes are new architectures, added to `ARCHS` at
RUNTIME; the base file is not edited.

### 12.16 "Any physical system" becomes a checkable precondition (ext_31)

**The weakest evidence in the project, in the extension's own words** (section 11, item 5):

> "both cells were built here to reproduce the same engineered kernel, so **nothing here
> tests a substrate whose response is not Gaussian in the input at all**.  'Any physical
> system' is still the widest claim in this document and the least covered."

That is a **circularity, not merely a gap**: the operators work because the kernel is
Gaussian, and every substrate tested so far is Gaussian by construction.

**THE PRECONDITION IS NOT "GAUSSIAN".**  Every operator reads the source's position out of
the coefficients of `log z` in a fixed basis, so what the suite actually needs is:

> **`log z(x)` must be LINEAR in some fixed, FINITE basis `phi(x)` of the input.**

The Gaussian satisfies it because `||x-c||^2` expands to a degree-2 polynomial whose
coefficients encode `c`.  **Any polynomial exponent does**, at the degree of that
polynomial -- so a quartic-exponent source is IN SCOPE with a bigger basis, and the
precondition is testable rather than assumed.

**P1-P3 -- THE RESIDUAL BY SUBSTRATE AND BASIS DEGREE:**

| substrate | prior | d=2 | d=4 | d=6 | d=8 | first d with residual < 1e-10 |
|---|---|---|---|---|---|---|
| gaussian (control) | IN | **1.22e-15** | 1.92e-15 | 5.76e-15 | 2.82e-14 | **2** |
| quartic exponent | IN | 3.53e-01 | **8.23e-16** | 2.06e-15 | 9.32e-15 | **4** |
| yukawa `exp(-kr)/r` | OUT | 4.45e-01 | 3.29e-01 | 2.56e-01 | 2.03e-01 | **NONE** |
| cauchy `1/(1+(r/g)^2)` | OUT | 3.06e-01 | 1.59e-01 | 4.97e-02 | 2.02e-02 | **NONE** |

**The prior and the measurement agree on 4/4 substrates.**  A Yukawa field needs `log r`,
which is not a polynomial; the residual stays at `2e-01` even at degree 8 and never
collapses.  The Cauchy profile decays the same way and also never collapses -- **so the
test is not tuned to catch one thing.**

**P2 -- AND WHERE THE BASIS IS RIGHT, THE GENERALISATION IS EXACT.**

| substrate | basis d | `c` recovered | `h` recovered | error |
|---|---|---|---|---|
| gaussian (control) | **2** | **[0.700000, -0.400000]** | **1.300000** | **1.67e-15** |
| quartic exponent | 2 | [0.778201, -0.416144] | 0.496421 | **7.82e-02** |
| quartic exponent | **4** | **[0.700000, -0.400000]** | **1.300000** | **2.00e-15** |

> **AN OUT-OF-SCOPE BASIS DOES NOT FAIL LOUDLY -- IT ANSWERS CONFIDENTLY AND WRONGLY.**
> The middle row is the quartic substrate read through the degree-2 basis: a plausible
> position and a width about a third of the true one, returned without complaint.  **So the
> residual is not a nicety; it has to be checked BEFORE any position is believed.**  That
> middle row is the strongest argument in this document for the qualification test.

**THE DELIVERABLE.**  A substrate nobody has characterised can now be put through a
least-squares residual before any operator is trusted on it:

    in scope  <=>  the residual COLLAPSES at some finite basis degree

**SO THE WIDEST CLAIM IS BOUNDED RATHER THAN REFUTED.**  "Any physical system" narrows to
**any system whose per-source response is log-linear in a fixed finite basis** -- strictly
more than the two-Gaussian-cells evidence supported, and strictly less than "any physical
system".  The narrowing is a **checkable property of a substrate**, which is what the claim
was missing.

**WHAT THIS DOES NOT DO.**  The operators are not re-derived here, only the identification
premise they rest on.  7.1's optimality bound (capture `0.976` at `N = 40`) was measured for
the Gaussian basis and is **NOT transferred** to the quartic one.  And nothing here is a
device: these are profiles evaluated at probe points, which is the right level for a
question about which responses are admissible and the wrong level for hardware.

**AND THE FAILURE MODE RECURRED IN THIS VERY SCRIPT.**  The first version's two inversions
both returned **numbers rather than errors**: the degree-2 one summed the pure-square
coefficients instead of averaging them, so `alpha` came out negative and every in-scope
substrate was reported as "inversion not defined" -- loud, and caught at once; the degree-4
one **dropped the `k/h^4` factor** and returned `c = -0.1225` against a true `0.7` --
silent, and caught only by comparing against the known answer.  **The second is the same
failure mode this experiment is about.**

### 12.17 The "5x better" direct-residual claim: refuted under control, and the mechanism is a target mismatch (ext_32)

**The claim.**  `THEORY.md` 7.3 reports that forming the layer-2/3 residual target from
each layer's **raw scalar output** is **~5x better** than from the proto-weighted
prediction, and says that finding "changes the design principle".  `README.md` carries it
as `+0.064 (direct residual) vs +0.013 (Proto)` on a 2D nonlinear regression, so the metric
is the **depth gain** and `0.064/0.013 = 4.9`.  The base's own status says the comparison
**has not been reproduced** -- "a short check gave the right direction but nowhere near
that magnitude" -- which is why `residual_mode` still defaults to `"proto"`.

**WHY THE PUBLISHED PAIR COULD NOT SETTLE IT.**  The two numbers come from **two
programs**, and the differences are not one:

| | `reg_01` (the `+0.013`) | `reg_02` (the `+0.064`) |
|---|---|---|
| kernel | **Cauchy, anisotropic h, multi-scale k** | **pure Gaussian** |
| architecture | `HiMALND`, dual A/B network | **separate single net, no A/B split** |
| trainer | **`HiMALND` with TWO METHODS REPLACED AT RUNTIME** (`train_one_layer`, `_forward_layer`) | its own hand-written loop |
| particles | 20/15/10, 80 epochs | its own `CONFIGS` |

`reg_01` monkey-patches the training loop and the forward before training, so it is **not
running the shipped trainer**; `reg_02` **does not import `HiMALND` at all**.  At least four
things change between the two numbers and the 5x is attributed to one.

**AND THE TRAINER ITSELF IS UNSEEDED** -- found while debugging this script's own first
run.  `HiMALND` has **no `seed` parameter and no seeding anywhere**; `_init_layer` draws its
centres and `W_gauss` from the global `np.random.uniform`, so two calls with identical
arguments produce different networks depending on what ran before.  **The published pair is
one unseeded draw each**, which is also the plainest explanation of the base's own note
that a short check agreed only in direction.

**THE CONTROLLED TEST.**  Shipped `HiMALND`, nothing monkey-patched, `reg_01`'s own
configuration, both arms started from the same RNG state, 3 seeds, 80 epochs, and
`residual_mode` as the **only** difference:

| mode | R2 (L1) | R2 (L1+L2) | R2 (L1+L2+L3) | **depth gain** | sd |
|---|---|---|---|---|---|
| `proto` | **+0.3071** | +0.3096 | +0.3140 | **+0.0069** | 0.0045 |
| `direct` | **+0.3071** | +0.2884 | +0.2787 | **-0.0284** | 0.0152 |

**THE BUILT-IN CONTROL.**  `residual_mode` branches only for layers 2+, so **layer 1 is
trained identically under both modes and its `R^2` must agree seed by seed**: `+0.3071`
against `+0.3071`, **identical**.  That check is now an assert, because the first version of
this script lacked the seeding and reported an **L1 gap of `+0.117`** between arms that
should have been the same -- the signature of the confound, and it was visible in the table
before it was understood.

> **THE CLAIM IS REFUTED, WITH THE SIGN REVERSED.**  `direct`'s depth gain is **negative**
> (`-0.0284`), against the published `+0.064`; the ratio is `-4.12x` against the claimed
> `4.9x`.  The `proto` side is small and positive (`+0.0069` here, `+0.0138` in the
> unseeded run, published `+0.013`) -- consistent, and not exact, which is what an unseeded
> trainer predicts.

**AND THE CURVE GIVES THE MECHANISM, which matters more than the refutation.**  Under
`direct`, layers 2 and 3 make the prediction **WORSE** (`+0.3071 -> +0.2884 -> +0.2787`),
while under `proto` they make it slightly better.  The reason is in the trainer's own
comment at `training/himal_nd.py:889`: the per-layer prototypes "map `h_i` to the residual
target", and **"At predict time these are summed -> recovers original y"**.

> **`direct` trains layers 2-3 against the residual of the RAW output, while the inference
> path sums the PROTO predictions -- so the deeper layers are fitted to a residual that the
> inference never reconstructs.**  It is not a better residual target; it is a **MISMATCHED**
> one, and depth therefore hurts.

**SO THE DEFAULT IS JUSTIFIED BETTER THAN THE BASE JUSTIFIES IT.**  The shipped comment
keeps `"proto"` because the 5x "has not been reproduced ... rather than switching on an
unverified claim".  After this it is not merely unverified: **the alternative is measured to
be actively harmful at the configuration the claim was made about.**

**TWO STRUCTURAL FINDINGS ALONGSIDE IT, both about the base's inference.**  `reg_01`'s own
source comment where it builds the depth curve reads "wire-direct accumulation **broken in
old path**", referring to `HiMALND.predict`, which walks the layers wire-direct and returns
the LAST layer's output with **no accumulation** -- while the training design assumes the
protos are summed.  And under `reg_01`'s own `output_dims=(4,3,2)` that path returns
**two columns against a scalar target**, so it **cannot be scored at all**: the published
depth numbers were necessarily computed through the hand-written proto sum.  **Which of the
two inference paths is correct is a separate question about the base's design, and it is
not settled here.**

**SCOPE.**  One task, one architecture, one epoch count, 3 seeds, and the published pair's
own configuration on the proto side.  A different kernel family or topology could behave
differently -- **what is refuted is that the residual mode ALONE produces a 5x.**

### 12.18 The do-no-harm gate at its own boundary: the counterexample's SHAPE, and why it is not yet one (ext_33)

**The open item, in the base's own words** (section 6.11):

> "The strongest missing evidence against the whole rule would be a cell where the gate
> **accepts** and the arm still loses ... the third attempt **did not answer its own
> question** (a first-order failure is detectable at any slice size, so it does not probe
> the gate's statistical power -- **that needs a task sitting near the decision boundary,
> which was not constructed**)."

**That task is constructed here**, and it is the last item on the base's list that is real
science rather than a documentation choice.

**THE GATE, EXACTLY.**  `val[::guard_holdout]` is a **stride**, not a random subset; the
strengths are fitted on the complement and kept only if they **strictly** beat the in-place
ones on that stride.  A strict inequality on a finite sample is what luck can carry at a
boundary, and `guard_holdout` is a direct knob on how many samples decide it
(`2/4/8/16` -> `128/64/32/16` of 256).

**P1 -- AND THE NOISE AXIS CANNOT CONSTRUCT THE BOUNDARY.**

| noise | accepted/total | rate | hybrid | geometry | margin |
|---|---|---|---|---|---|
| 0.10 | 5/120 | 0.042 | +0.5009 | +0.2038 | +0.2971 |
| 0.40 | 3/120 | 0.025 | +0.3523 | +0.1765 | +0.1758 |
| 0.80 | 4/120 | 0.033 | +0.1524 | +0.1191 | +0.0333 |
| **1.50** | **3/120** | 0.025 | +0.0177 | +0.0460 | **-0.0283** |
| 3.00 | 0/120 | **0.000** | -0.0019 | -0.0019 | +0.0000 |

Raising the noise drives the accept rate to **0/120**, not toward 0.5 -- so noise is a
**first-order** axis, exactly like `10d`, and the base's complaint applies to it too.

**P2 -- BUT THE SLICE SIZE CAN, and that is the property section 6.11 said was untested.**

| noise | 128 | 64 | 32 | **16** |
|---|---|---|---|---|
| 0.10 | 0.033 | 0.042 | 0.092 | **0.333** |
| 0.40 | 0.042 | 0.025 | 0.133 | **0.508** |

**The accept rate moves from 0.025 to 0.508 as the validation slice shrinks (spread
0.483).**  The base swept slice size on `10d` and saw "40/40 rejected at every slice size",
concluding that failure was first-order -- **but that is a property of `10d`, not of the
slice size.**  On a task where the calibration genuinely helps, **the smaller the slice,
the more readily the gate accepts.**

**P3 -- THE COUNTEREXAMPLE'S SHAPE EXISTS, IN 1 OF 11 CELLS.**

| noise | holdout | accepted | hybrid | geometry | margin | verdict |
|---|---|---|---|---|---|---|
| 0.10 | 2/4/8/16 | 4/5/11/40 | ... | ... | +0.21..+0.34 | accepted, no harm |
| 0.40 | 2/4/8/16 | 5/3/16/61 | ... | ... | +0.18..+0.26 | accepted, no harm |
| 0.80 | 4 | 4 | +0.1524 | +0.1191 | +0.0333 | accepted, no harm |
| **1.50** | **4** | **3** | **+0.0177** | **+0.0460** | **-0.0283** | **ACCEPTED AND LOST** |
| 3.00 | 4 | 0 | -- | -- | +0.0000 | refused |

> **A CELL WHERE THE GATE ACCEPTED AND THE ARM ENDED BELOW ITS OWN NO-CALIBRATION
> BASELINE.**  That is the shape section 6.11 searched for through three designs and never
> found, at a boundary the base never built.

**AND IT IS NOT YET A COUNTEREXAMPLE TO THE GUARANTEE, which is the part that matters.**
`arm="hybrid"` and `arm="geometry"` are **separate trainings**: matched on data, seed and
initialisation, but the calibration **rewrites `W_gauss` mid-run**, so the finite
differences that follow see a different loss surface.  **So `hybrid < geometry` is the
calibration's effect PLUS that divergence**, and one cell cannot separate them -- the same
conflation the base flagged in its own 6.11 caveat.  This design narrows it (matched seeds,
1 cell of 11) **without closing it**.

> **WHAT WOULD CLOSE IT, stated precisely because it is small:** record the arm's TEST `R^2`
> immediately **before and after each deployment, inside one run**.  Same network, same
> moment, no second trajectory -- so the calibration is isolated from the divergence.  That
> needs no base file edited: `solve_readout_closed_form` can be wrapped from outside.

**AND THE HONEST POSITION OF THIS RESULT.**  The base tried three counterexamples and all
failed; this found a cell, and the cell comes with the isolation still missing.  **So the
guarantee is neither confirmed nor refuted here -- what is new is that its boundary has been
located, that the gate's decision there is slice-size-dependent by 0.48, and that a cell
with the counterexample's shape now exists to be isolated.**

**SCOPE.**  One task with a swept noise level, one architecture, the diffusion substrate,
the base's own `effective_budget`, 40 epochs, 3 seeds.  A single boundary is not the
boundary: a different task could sit elsewhere and behave differently, and nothing here
bounds that.

### 12.19 The deployment isolated: the counterexample is real, and the cell 12.18 flagged is not one (ext_34)

**What 12.18 left open.**  It found one cell of eleven where the gate accepted and the arm
ended below its own geometry arm, and said why that was not yet a counterexample: the two
arms are **separate trainings**, and the calibration rewrites `W_gauss` mid-run, so
`hybrid < geometry` is the calibration **plus** the trajectory divergence.

**THE INSTRUMENT, and why it needed no base edit.**  `train_model` calls
`solve_readout_closed_form` as a **bare module-global name** (`abl_common.py:1219`), so it
is wrapped here.  With the default `closed_form_model=None` the object that function edits
**IS the device**, and it **restores** the previous strengths on refusal and **keeps** the
fitted ones on acceptance.  So one run yields, per calibration, the device's **test** `R^2`
immediately before and after the deployment -- **same network, same moment, no second
trajectory.**  The divergence is removed by construction rather than by matching.

**AND THE INSTRUMENT VALIDATES ITSELF.**  A refusal must leave the `R^2` **unchanged**,
because the base's guard restores the strengths.  That invariant is asserted across every
logged calibration, and it holds -- so the wrapper is reading the model's real state and not
merely printing plausible numbers.

**P1 -- EVERY DEPLOYMENT, 3 seeds:**

| noise | holdout | calibrations | deployed | refused | **HARMED** | mean change | worst | best |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 4 | 120 | 5 | 115 | 1 | +0.4946 | -0.0410 | +1.1739 |
| 0.10 | 16 | 120 | 40 | 80 | **21** | +0.0636 | -0.0394 | +1.2306 |
| 0.40 | 4 | 120 | 3 | 117 | 0 | +0.7409 | +0.4099 | +1.0787 |
| 0.40 | 16 | 120 | 61 | 59 | **33** | +0.0413 | -0.0502 | +1.1486 |
| 0.80 | 4 | 120 | 4 | 116 | 0 | +0.4380 | +0.0878 | +0.8215 |
| 0.80 | 16 | 120 | 0 | 120 | 0 | -- | -- | -- |
| 1.50 | 4 | 120 | 3 | 117 | 0 | +0.1000 | +0.0235 | +0.1476 |
| 1.50 | 16 | 120 | 0 | 120 | 0 | -- | -- | -- |

> **55 OF 116 DEPLOYED CALIBRATIONS LEFT THE DEVICE'S TEST `R^2` LOWER THAN IT WAS
> IMMEDIATELY BEFORE.**  That is the counterexample section 6.11 searched for through three
> designs, and it needed no second trajectory to find -- the arm was worse after the write
> than before it, on the same network, at the same moment.

**P2 -- AND THE CELL 12.18 FLAGGED IS EXONERATED.**

| noise | holdout | end-of-run margin | deployments harmed | reading |
|---|---|---|---|---|
| 0.10 | 4 | +0.2971 | 1/5 | harmed at deployment, run still positive |
| 0.10 | 16 | +0.3361 | 21/40 | same |
| 0.40 | 16 | +0.2596 | 33/61 | same |
| **1.50** | **4** | **-0.0283** | **0/3** | **negative margin, NO harm -- the DIVERGENCE** |

> **THE ONE CELL WHOSE NEGATIVE MARGIN 12.18 COULD NOT EXPLAIN HAD THREE DEPLOYMENTS AND
> NONE OF THEM HARMED.**  So that margin is **not** the calibration: it is exactly the
> trajectory divergence 12.18 named and could not exclude.  **The counterexample exists, and
> the cell that looked like one is not one.**

**THE MECHANISM, now visible rather than argued.**  The gate judges on a **stride of the
training set**; harm here is measured on the **evaluation set**.  **A gate can therefore be
right about the data it checks and wrong about generalisation** -- which is the shape 6.11
was hunting, and it does **not** require the gate to be broken.

**AND THE GUARANTEE'S PRACTICAL CONTENT SURVIVES, which the base's framing needs.**  Every
cell in this grid ends with a **positive** margin over its geometry arm, and the asymmetry
is large:

    harmful deployments :  55   range -0.0502 to -0.0003
    helpful deployments :  61   range +1.2306 down to +0.0009

**So the dips are recovered.**  What the guarantee *says* is false per deployment; what the
*recommendation* claims is true per run.  **Reporting either without the other would
misstate it.**

**AND THE COUNT MUST NOT CARRY MORE THAN IT CAN.**  The harmful range reaches down to
**`-0.0003`**, so a substantial part of those 55 are numerically indistinguishable from no
change at all.  **The guarantee is about the sign and there are 55 negative signs; but
lumping `-0.0003` together with `-0.0502` would exaggerate it.**  The magnitude of a small
dip is not separable from the evaluation set's own sampling noise at 128 samples, and is
not claimed to be.

**SCOPE.**  Task 4d, the noise levels and slice sizes above, one architecture, the base's
own `effective_budget`, 40 epochs, 3 seeds.  The wrapper is installed for this run only and
`abl_common` is left as found.

### 12.20 The unseeded trainer is fixed, and the exposure is smaller than entry 0 estimated (ext_35)

**What was wrong.**  `HiMALND` took **no `seed` parameter and never seeded the global
RNG**, so two calls with identical arguments produced **different networks** depending on
what had run before -- and every number published from that class was one draw from an
unseeded process (12.17).

**THE FIX, and why it is only three lines.**  `seed` is added to `__init__` and
`np.random.seed(self.seed)` is called at the top of `train`.  That is **sufficient**, not a
guess: the class has **exactly two randomness sources and both are reached from `train`**:

    `_init_layer`  -- centres, W_gauss, W_act    (3x np.random.uniform)
    `_spsa_step`   -- the SPSA perturbation      (1x np.random.choice)

**AND IT IS VERIFIED, with the check that actually isolates the fix.**  Running twice in a
fresh process would pass even WITHOUT the fix, because the prior RNG state happens to match.
So the decisive test seeds the global RNG, **consumes it**, and then constructs the trainer:
the network must still be identical.  Measured:

| check | result |
|---|---|
| seed 0 twice, same prior state | identical (passes without the fix too) |
| **seed 0, DIFFERENT prior state** | **identical -- this is the fix** |
| seed 1 | differs -- so the parameter is load-bearing, not decorative |
| two runs, identical arguments (ext_35 P1) | **bit-identical, 930 floats, and the depth curve** |

**WHAT WAS AT STAKE, quantified.**  At the published configuration, seeds 0 and 1 give an
`L1` `R^2` of **`+0.1465` against `+0.2608`** -- **the draw alone spans `0.11`**.  So
`reg_01`'s `+0.013` was one sample from a distribution wider than the number itself.
**That is what "not a measurement" means here, and it is now a number rather than a
caution.**

**AND THE RE-RUN TURNS OUT TO BE UNNECESSARY, which is smaller than entry 0 estimated.**
Entry 0 said "re-run whichever numbers are quoted externally".  Surveyed:

| quoted number | source | status |
|---|---|---|
| California Housing `R^2 = 0.530` | `reg_06` | **the script seeds itself** (`np.random.seed(42)` before constructing the trainer), so it was reproducible all along |
| `+0.013` and `+0.064` | `reg_01`, `reg_02` | **already retracted** (12.17) |
| the `nd_0*` results | `nd_0*` | write no report and are quoted in no document |

**So the only externally-quoted number that was genuinely exposed is one whose script
happened to seed first.**  The defect was real -- the class could not be reproduced on
request -- but its blast radius on the published numbers is one retracted pair, not a
sweep.

**SCOPE.**  The verification uses `reg_01`'s configuration at a reduced epoch count for the
determinism checks (determinism does not need convergence) and one confirmation per mode at
the published configuration.  The spread across seeds is `ext_32`'s (12.17), not recomputed
here.

### 12.21 The trained inner layer: the untested case, tested -- and the cost question it surfaced (ext_36)

**The open item, in this document's own words** (section 11, item 1): the trade is measured
and the cheap form says keep the linearity, but "**`W` and `b` were drawn, not trained ...
a TRAINED inner layer is the untested case**".  `ext_18` called its own result a **lower
bound** for the same reason.  This runs that case.

**THE COST ATTRIBUTION WAS CHECKED BEFORE RUNNING, not assumed.**  The obvious objection is
that `W` multiplies `Z`, which is *measured*, so `W` might be digital and free.  It is not,
and `DESIGN.md`'s own table says why: its last column is "readout linear in **strengths**?",
the nonlinear rows are marked **no**, and in this architecture **a scaling coefficient IS a
source's excitation strength** (`DESIGN.md` line 82).  So `ext_18`'s charge follows the
design document's framing, and this script reuses it rather than inventing a friendlier one.
**[This check caught a false finding on the way: the first reading of `N(H)` suggested the
charge was misattributed, and `DESIGN.md` refuted that before it was written down.]**

**1 -- TRAINING THE INNER LAYER LOSES.**

| task | `L40` | `N80` drawn | `N80` trained | **training buys** | `L320` | trained - `L320` |
|---|---|---|---|---|---|---|
| `4d` | +0.5761 | +0.6056 | +0.5911 | **-0.0145** | +0.7025 | -0.1114 |
| `steps` | +0.1572 | +0.2232 | +0.1826 | **-0.0406** | +0.3118 | -0.1292 |
| `hetero` | +0.2215 | +0.1941 | +0.1699 | **-0.0242** | +0.3211 | -0.1512 |

**Average `-0.0264`, and the trained arm beats the node arm on 0 of 3 tasks.**  So the
expectation behind the open item -- that a trained inner layer would do better -- is
**wrong on this schedule**: the drawn inner layer was not a handicap that training removes,
it was the better choice.  **That is a loss, not "a real but insufficient gain", and it is
reported as one.**

**2 -- SO THE TRADE SURVIVES ON ACCURACY**, which is the one thing section 11 item 1 said
could overturn it.

**3 -- BUT NOT ON EFFICIENCY, and the two cost accountings are reported side by side
because they are different things.**

| arm | strengths | (a) calibration-style | (b) search-style | R2 @ (a) | R2 @ (b) |
|---|---|---|---|---|---|
| `L40` | 40 | 10 240 | 10 240 | +3.1083 | +3.1083 |
| `N80` drawn | 3 280 | 839 680 | **20 480** | +0.0406 | **+1.6650** |
| `N80` trained | 3 280 | 839 680 | 32 800 | +0.0375 | +0.9590 |
| `L320` | 320 | 81 920 | 81 920 | +0.5434 | +0.5434 |

> **UNDER (b) THE DRAWN ARM WINS PER MEASUREMENT.**  Its `W` is drawn and therefore free --
> only the outer values are calibrated, 80 against the node arm's 320 -- so its `R^2` per
> measurement is **three times the node arm's**.  `ext_18` could not see this because it
> charged every arm the same way.  **So the trade's conclusion is NOT
> accounting-independent: at a fixed ACCURACY the node arm wins, and under a search-style
> charge at a fixed BUDGET the drawn nonlinear arm wins.**

**Which accounting the hardware actually pays is a modelling question this script surfaces
and does not settle** -- it depends on whether the inner matrix is *programmed* (free, like
a digital weight) or *measured* (charged, like a source strength), and the design document's
statement that the nonlinear layer's strengths cannot be *calibrated* is an argument that
they are not measured.  **So "keep the linearity" is a statement about ACCURACY, not about
efficiency**, and it should be read that way.

**AND A REPRODUCIBILITY BOUND, measured rather than assumed.**  Two runs of this script with
identical arguments agreed to four decimals but **not bit for bit** (`-0.0265` against
`-0.0264`; `L320` on `4d` moved `+0.7058 -> +0.7025`).  Every draw is seeded --
`fit_operator`, `nonlinear_readout` and `trained_inner` all take a seed -- so the difference
is **float-level nondeterminism in the linear algebra**: the same draw computed slightly
differently, not a different draw.  **A conclusion that turns on the fourth decimal should
not be read from this table.**

**SCOPE.**  The node set from the operator trainer at `N = 40`, `H = 80`, 5 block-coordinate
epochs, 3 seeds, three tasks, toy.  **The training scheme is one choice among several** -- a
longer schedule or a different parameterisation could buy more, and nothing here bounds
that.

### 12.22 A genuine `m > 1` setting: the item names the wrong task, and the right one is out of reach (ext_37)

**The open item, in this document's own words** (section 11, item 4):

> "**A genuine `m > 1` setting for half A.**  Everything there rests on a synthetic
>  multi-output target, and `m = 1` makes half A a no-op.  **The repository has a 4D
>  classification task**, and classification is where `m` channels are real."

**THE ARITHMETIC, WHICH WAS CHECKED BEFORE THE TASK WAS TRUSTED.**  Half A's readout is
`W = C . W_g` with `C` of shape `m x k`, so `rank(W) <= min(m, k)`: **the constraint bites
only when `k < min(m, rank(W))`.**  Evaluated on the two tasks the repository has:

| task | classes | `m` | `rank(W)` | effective rank | `k=1` | `k=2` | `k=3` |
|---|---|---|---|---|---|---|---|
| `moons` (base, 2-class) | 2 | 2 | **1** | **1** | -1.6e-05 | -1.6e-05 | -1.6e-05 |
| 4d moons (`nd_01`, 2-class) | 2 | 2 | **1** | **1** | -8.6e-08 | -8.8e-09 | -8.8e-09 |

> **THE ITEM NAMES THE WRONG TASK, AND THE VACUITY IS STRONGER THAN EXPECTED.**  `nd_01`'s
> 4D classification task is **2-class**, so `m = 2` -- and the measured `rank(W)` is
> **1**, not 2, because the one-hot target is centred so its two channels are
> complementary.  **So even `k = 1` is already full rank there**, and the task item 4
> proposes would have produced a **vacuous test of half A, in the same way `train_01` was
> vacuous.**

**AND THE RESIDUAL GAP IS THE OPTIMISER'S, NOT THE CONSTRAINT'S.**  The gaps are not
floating-point zero -- they are `~1e-5` -- and at `k` at or above the effective rank the
rank-k model class **contains** the full-rank fit, so the only open question was whether
alternating least squares reaches it.  **The iteration sweep answers that:**

| task | gap at 60 iterations | gap at 400 iterations | shrinks |
|---|---|---|---|
| `moons` | -1.64e-05 | **-1.21e-06** | **yes, 13x** |
| 4d moons | -8.78e-09 | -7.94e-09 | yes |

**Three orders of magnitude below any effect half A is about, and shrinking with the
optimiser's budget rather than with `k`.**

**THE REPOSITORY DOES HAVE A GENUINE `m = 5` SETTING -- AND IT IS OUT OF REACH.**  Its own
cached CIFAR-10 batches are present in the working tree (gitignored, no download), and its
own CIFAR scripts use `N_CLASSES = 5`.  Run through half A's machinery at 40 nodes:

| `k` | parameters | `R^2` | gap vs full | clears the floor |
|---|---|---|---|---|
| 1 | 45 | -0.0000 | -0.0000 | **NO** |
| 2 | 90 | -0.0000 | -0.0000 | **NO** |
| 3 | 135 | -0.0000 | -0.0000 | **NO** |
| 4 | 180 | -0.0000 | -0.0000 | **NO** |
| 5 | 225 | -0.0000 | -0.0000 | **NO** |

**So that is not a weak measurement of the rank gap; it is NO measurement of it.**  12.16's
sibling result applies: `ext_20` closed the dimension axis as out of reach at this scale,
and a flat 64-dimensional feature map at 40 nodes reproduces that on the repository's own
image task.  Reaching it needs the repository's **own** CIFAR pipeline -- the `build_3d`
point cloud fed to its four-layer 3-D network -- **which is not what half A is built on**.

**SO ITEM 4 IS NOT SATISFIABLE AS STATED, and both halves are now measured**: the
repository's 2-class tasks are **vacuous**, and its 5-class one is **out of reach** from
this machinery.  **And one thing this settles about the synthetic target**: `ext_02`'s
`m = 8` was not a convenience -- it is what makes the constraint bite at `k = 3`.  **A
synthetic target with a real `m` is the CORRECT instrument for a question about rank**, and
the item's framing, that the synthetic target is the weakness, inverts that.

**SCOPE.**  The repository's own tasks and its own cached data, 3 seeds on the CIFAR part
and 1 on the algebraic part, 40 nodes, toy.  The CIFAR feature construction is a stated
choice (the `build_3d` cloud reduced to one vector per image) and it is the reason that part
fails; the repository's own pipeline is not reproduced here.

### 12.23 A third substrate, and this one is not Gaussian (ext_38)

**The open item, in this document's own words** (section 11, item 5):

> "both cells were built here to reproduce the same engineered kernel, so **nothing here
>  tests a substrate whose response is not Gaussian in the input at all**.  'Any physical
>  system' is still the widest claim in this document and the least covered."

12.16 turned that claim into a **checkable precondition** and showed on PROFILES that a
quartic exponent is in scope.  **What it did not do is run the operators on a third
MEDIUM.**  This does.

**THE SUBSTRATE: a particle in an ANHARMONIC TRAP.**  `V(r) = kappa r^4` at temperature `T`
gives equilibrium density `proportional to exp(-(kappa/kT) r^4)` -- quartic wells are
ordinary objects, the exponent is not quadratic, and **the trap's minimum and its
stiffness-over-temperature are this project's own movable centre and width.**  A medium
whose phase space generates movable nonlinear activation sources is exactly what the thesis
asks for, and here the sources are trapped particles.  It is implemented as a **layer** --
centres and widths as parameters, `forward` returning the field -- because `features()`'s
contract is that the trainer never evaluates its own kernel formula in place of the
device's, and it lives in the EXTENSION because adding a substrate to
`abl_common.make_cell` would mean editing the base.

**P1 -- THE QUALIFICATION TEST, ON THREE MEDIA.**

| substrate | d=2 | d=4 | d=6 | first collapsing degree |
|---|---|---|---|---|
| circuit (Gaussian) | **8.8e-06** | 2.3e-15 | 9.4e-15 | **4** |
| diffusion (Gaussian) | **1.7e-15** | 2.4e-15 | 9.9e-15 | **2** |
| anharmonic trap (quartic) | 8.4e-02 | **1.5e-15** | 7.3e-15 | **4** |

> **AND THE CIRCUIT CELL'S DEGREE-2 FLOOR IS THE BASE VERSION'S OWN DOCUMENTED
> PERTURBATION.**  Section 9.2 records that `bjt_gaussian_kernel` adds `sqrt(|z|)*1e-6`
> whenever `noise_std is None` -- "which is how `simulator/layer_nd.py` calls it" -- so the
> circuit cell **cannot** reach `1e-10` however good the basis is.  It is deterministic in
> `z`, so a larger basis absorbs it.  **That is why the bar for "in scope" must be each
> device's own floor rather than a fixed number**, and an absolute bar would have called
> the reference substrate out of scope.

**P2 -- THE OPERATOR TRANSFERS TO MACHINE PRECISION.**

| true `c` | recovered `c` | error | true `h` | recovered `h` | error |
|---|---|---|---|---|---|
| [0.70, -0.40, 1.10, 0.20] | [0.700000, -0.400000, 1.100000, 0.200000] | **2.0e-14** | 1.300 | 1.300000 | **6.0e-15** |
| [-1.30, 0.90, -0.20, 0.60] | [-1.300000, 0.900000, -0.200000, 0.600000] | **1.1e-14** | 1.300 | 1.300000 | **2.2e-15** |

The Gaussian operator reads the centre from the degree-1 coefficients; for a quartic
exponent the same argument runs **one degree up** -- `||x-c||^4` has degree-3 coefficients
`-4 c_j (k/h^4)` and a degree-4 leading coefficient `k/h^4`.  **That is what "swap the
basis" means concretely, and it reaches the same machine precision the Gaussian operator
reaches in its own basis.**

**P3 -- BUT EXPRESSIVENESS IS A SEPARATE QUESTION, AND THIS MEDIUM IS WORSE AT IT.**

Widths matched by **effective radius** -- a Gaussian falls to `1/e` at `h/sqrt(k)` and a
quartic at `h/k^(1/4)`, so at the same raw `h` the quartic field is narrower and its nodes
overlap less:

| seed | circuit | diffusion | anharmonic (matched) | anharmonic (raw `h`) |
|---|---|---|---|---|
| 0 | +0.5924 | +0.5924 | +0.4536 | +0.2905 |
| 1 | +0.5024 | +0.5024 | +0.4166 | +0.2774 |
| 2 | +0.6336 | +0.6336 | +0.5124 | +0.3721 |

**Matching the radius recovers a large part of the gap and not all of it.**  The anharmonic
cell covers this task less well at 40 nodes.

> **AND THE FIRST DRAFT OF THIS VERDICT CALLED THAT "THE TRANSFER DOES NOT HOLD", WHICH WAS
> A MISREADING.**  P3 measures expressiveness at fixed nodes, not the operator's transfer;
> the operator transferred.  **A different field SHAPE is a different basis, not a different
> operator** -- and the two questions are reported separately because conflating them is
> exactly the error that produced the first draft.

**SO THE WIDEST CLAIM GAINS A THIRD MEDIUM AND NOT A DISCHARGE.**  Two measurements now
support it separately: **identification transfers across kernels** (2e-14 on a quartic
field), and **calibration is substrate-independent in form** while a medium's
expressiveness is its own.  A medium whose response is not log-linear in any finite basis
-- a Yukawa field, a Lorentzian -- **remains excluded**, and 12.16 showed the test that
excludes it.

**SCOPE.**  The anharmonic cell is a medium implemented here, not a fabricated device; its
field is evaluated in closed form rather than read through a sensor chain; and the
repository's `4d` task at 40 nodes, 3 seeds, toy.

### 12.24 Item 8, and the replacement it asks for is not needed (ext_39)

> ## WHICH "COMBINATION" -- read this before the rest of 12.24
>
> **Two different arrangements have been called "the combination", and only one of them is
> this project's design.  Stated by the repository owner while this section was being
> written, and the distinction is load-bearing:**
>
> | arrangement | what it is | where it lives here |
> |---|---|---|
> | `W = C . W_graphs`, `C` of shape `m x k` | **each graph gets its OWN LINEAR scaling coefficients**, combined linearly -- `C[v,i]` is graph `i`'s coefficient into channel `v` | **THIS IS THE DESIGN.** It is half A, it is `ext_02`-`ext_05`, and its readout stays linear in the strengths |
> | `sum_j W_j phi(z_j)` | a **nonlinearity inserted into the readout path**: activate, then `phi`, then combine | 12.17/12.21/12.24 -- a HYPOTHETICAL the design memo raises and **the architecture does not use** |
> | `phi(sum_j W_j z_j)` | the nonlinearity after the sum | raised by the memo, never implemented, and it forfeits the operator |
>
> **The architecture's identity is "activate first, then combine LINEARLY"**: the
> nonlinearity lives in the SOURCE (`z_j(x)` is nonlinear in `x`) and the combination is
> linear.  **Linearity of the combination is exactly what preserves that.**  A `phi` on the
> readout path is an EXTRA nonlinearity the design does not have -- which is why 12.24's
> negative result is a **confirmation** of the architecture's choice rather than a challenge
> to it: putting nonlinearity into the readout costs accuracy, so it belongs in the source.
>
> **And it means item 8's wording pointed at a design nobody proposed.**  The phrase
> "nonlinear in the readings" comes from the design memo's own paragraph, which raises it as
> something to *consider and measure* rather than as the plan.  **That paragraph has now been
> measured, and the answer is the design's own answer: keep the combination linear.**

**The open item, in this document's own words** (section 11, item 8):

> "**A nonlinear combination in full, with a replacement for the closed-form calibration.**
>  `N(H)` keeps the outer solve closed form on purpose; the full version has to replace the
>  operator half A is about, and that is a different project."

**THE PREMISE IS TOO STRONG, AND THE ORDER OF THE OPERATIONS IS WHAT DECIDES IT.**  The
memo concludes that a combination nonlinear in the readings makes fitting the strengths a
nonlinear problem again.  That is true of

    phi( sum_j W_j g_j )      nonlinear in the STRENGTHS -- the operator IS lost

and **false** of

    sum_j W_j phi(g_j )       NONLINEAR IN THE READINGS, LINEAR IN THE STRENGTHS

because `phi(g_j)` is a fixed feature of a reading that the calibration already takes.
`DESIGN.md`'s table marks "nonlinear in the readings" as not linear in the strengths; **the
distinction between these two arrangements is exactly what the table does not draw.**

**AND `phi` IS IN THIS PROJECT'S OWN DEVICE LIBRARY.**  The base analog chain is built from
a **Gilbert squarer** and a **BJT exponential**, so `g^2` and `exp(g)` are not hypothetical
nonlinearities -- they are components the repository already models.

**P1 -- THE ALGEBRA, VERIFIED RATHER THAN ASSERTED.**

| check | value |
|---|---|
| nonlinear in the readings: `max abs(y(2g) - 2 y(g))` | **2.1300** (nonzero) |
| linear in the strengths: `max abs(W_hat - W)` | **9.99e-16** (the closed-form solve is exact) |
| features used | `phi(g_j)` for the SAME readings per sample |
| measurement cost | `n_train * N`, **identical to the linear readout** |

**P2/P3 -- THE MEASUREMENT, SAME NODES, THREE TASKS, 3 SEEDS.**

| arm | strengths | searched? | measurements | mean `R^2` |
|---|---|---|---|---|
| `L40` (linear) | 40 | no | **10 240** | **+0.3183** |
| ENL `square` (Gilbert) | 40 | no | **10 240** | **-0.0058** |
| ENL `abs` (rectifier) | 40 | no | **10 240** | +0.2286 |
| ENL `tanh` (saturation) | 40 | no | **10 240** | +0.2708 |
| ENL `softplus` (diode) | 40 | no | **10 240** | +0.2459 |
| `L320` (node control) | 320 | no | 81 920 | +0.4463 |

> **EVERY ELEMENTWISE NONLINEARITY IS WORSE THAN THE LINEAR READOUT.**  The best of the four
> is `tanh` at `-0.0475`, winning on **0 of 3** tasks.  **The first five rows cost exactly
> the same**, so elementwise nonlinearity is bought with a COMPONENT -- one element per
> source -- and not with measurements.
>
> **THE SQUARE IS CATASTROPHIC AND THAT IS A PROPERTY, NOT A BUG.**  `g^2` is not injective:
> it discards the sign of every reading, so the two halves of a node's response become
> indistinguishable.  **It is also the one element the chain actually contains**, which makes
> it the most physically available and the least useful of the four.

**P4 -- SO ITEM 8 ASKS FOR A REPLACEMENT OPERATOR AND THIS CLASS DOES NOT NEED ONE.**  And
the negative is **a stronger position for this project than a positive would have been**:
the thing the design memo feared -- that nonlinearity would force a search -- can be avoided
at **zero measurement cost**, and when it is avoided this way **it buys nothing**.  So the
linearity is not a costly constraint the architecture endures; here it is simply the right
choice, and 5.13's answer stands for a **second, independent reason**.

**AND THE STRUCTURAL RESULT IS INDEPENDENT OF THE EMPIRICAL ONE.**  P1's algebra is a fact
about the arrangement; it holds whether or not these four `phi` help.  **That is what makes
item 8's premise too strong rather than merely premature.**

**WHAT REMAINS OPEN, AND WHAT ITEM 8 SHOULD SAY.**  The third arrangement --
`phi(sum_j W_j g_j)` with its own calibration story -- is not measured here, and 12.21's
`-0.0264` (a searched inner layer, which loses) is the nearest evidence.  **The item should
be restated as that one arrangement**, not as "a full nonlinear combination".

**THE HONEST LIMITS.**  `phi` is applied to the normalised reading so the unbounded elements
are usable, and that standardisation is a training-statistic constant, stated rather than
hidden.  The arrangement assumes a per-source element **before** the summing crossbar; a
device where the nonlinearity can only sit **after** the sum is the other arrangement, and
that one does lose the operator.  Repository tasks, 40 nodes, 3 seeds, toy.

### 12.25 Item 6: the drift RATE, and the item names the wrong channel (ext_40)

**The open item, in this document's own words** (section 11, item 6):

> "**The drift RATE.**  ext_15 established that a placement correction has a shelf life,
>  because a drift after the measurement makes it stale.  How often a deployment must
>  re-measure is set by the drift it expects, and **no measurement here sets that number for
>  a real cell**."

**THERE ARE TWO DRIFT CHANNELS IN THIS REPOSITORY AND THE ITEM NAMES THE WRONG ONE.**  The
base version's own docstring separates them --

    mismatch -> perturbs h (differential_pair gain) and the MVM gain   = GEOMETRY
    drift    -> multiplies memristor conductances W                    = WEIGHT-only

-- and `physical_simulator/components.py` implements the first literally:

    L80  # Gain mismatch: each channel's h has independent deviation
    L81  h_noisy = h * (1 + np.random.normal(0, mismatch_std, h.shape))

**A PLACEMENT correction is a WIDTH correction, so the channel that makes it stale is
`mismatch`.**  `ext_15` drifted widths (`w * (1 + drift)`) while calling it "drift", so it
exercised the mismatch channel's effect under the other channel's name.  **That is a naming
defect in item 6, not in `ext_15`'s measurement.**

**AND THE TWO CHANNELS HAVE DIFFERENT RATE SEMANTICS, which is the whole answer:**

| channel | how it is drawn | rate? |
|---|---|---|
| `drift` | `conductance_drift_std`, documented **"per read operation"** -- fresh on every forward call | **yes, per READ** |
| `mismatch` | `h * (1 + normal(...))` -- **one draw, no time axis** | **no rate at all** |

**1 -- THE REGISTERING TEST `ext_15` SAID WAS MISSING.**  Its P3 notes that "a drift makes
the realized widths LARGER, and this node set happens to prefer wider widths ... **a drift
test that could register it would need widths the plan already prefers**", and it swept the
drift **one-sided** (0 to 0.40).  **Sweeping BOTH signs gives the test that property at no
cost:**

| `delta` | `4d`: plan - nominal | `steps`: plan - nominal |
|---|---|---|
| -0.30 | +0.0329 | -0.0615 |
| **-0.20** | +0.0527 | **-0.0068** |
| -0.10 | +0.0645 | +0.0391 |
| 0.00 | +0.0690 | +0.0524 |
| +0.40 | +0.0598 | +0.1545 |

> **THE BINDING GEOMETRY TOLERANCE IS `d* = 0.10`** -- the smallest width deviation, in
> either direction, at which the operator's plan stops beating the nominal widths by
> `ext_15`'s own `0.02` margin.  **And the asymmetry CONFIRMS `ext_15`'s one-sided
> explanation**: the negative side is the binding one (`-0.20` already fails on `steps`,
> `-0.40` fails on both), because the plan prefers wider widths -- which is exactly why the
> one-sided sweep could not register it.

**BUT THAT TOLERANCE CANNOT BE CONVERTED TO AN INTERVAL, AND THE FIRST DRAFT OF THIS SECTION
DID EXACTLY THAT.**  It computed `(d* / sigma_drift)^2` -- **a GEOMETRY tolerance divided by
a WEIGHT rate**, which is the very conflation this section reports in item 6.  Recorded
rather than quietly dropped.  The weight channel needs its own tolerance, measured with a
**per-weight random deviation applied AFTER the calibration** -- which is what the device
does (`normal(0, std, (n_out, n_in))`) and what `ext_15`'s structure was:

| `delta` | 0.00 | 0.05 | 0.10 | **0.20** | 0.40 | 0.80 |
|---|---|---|---|---|---|---|
| `4d` | +0.576 | +0.573 | +0.567 | **+0.547** | +0.472 | +0.190 |
| `steps` | +0.157 | +0.145 | +0.128 | **+0.084** | **-0.049** | -0.499 |

**`d*_w = 0.20`**, against the project's own `0.05` learning floor.  Then the bridge is
directly available, because this channel really is per-read:

    accumulated drift after N reads = sigma * sqrt(N)    =>    N* = (d*_w / sigma)^2

| device level | `drift` sigma | **`N*` (reads)** | `mismatch` sigma | placement interval |
|---|---|---|---|---|
| `high` | 0.005 | **1 600** | 0.05 | **NOT DERIVABLE** |
| `medium` | 0.001 | **40 000** | 0.02 | **NOT DERIVABLE** |
| `low` | 0.0005 | **160 000** | 0.01 | **NOT DERIVABLE** |

**2 -- THE LAST COLUMN IS THE ANSWER TO ITEM 6 AND IT IS NOT A NUMBER.**  A placement
correction's staleness is set by the GEOMETRY channel, and that channel is drawn **once**,
with no time axis, so **the repository cannot express a rate for it at all**.  A
re-measurement interval for a placement correction therefore **cannot be derived from this
model**, which is a sharper statement than "no measurement here sets that number".  **The
missing piece is a bias-current drift law -- a MODEL, not a measurement.**

**3 -- TWO DEFECTS OF MY OWN, both caught by their own symptoms.**  The first weight sweep
scaled the whole feature matrix by `(1 + delta)` and reported the **same R^2 at every
delta** -- because a uniform scaling is absorbed exactly by ridge.  *The flatness was the
tell.*  And the tolerance was first taken as the **smallest** surviving deviation instead of
the **largest**, which understated it 4x.  **Both are recorded because a curve that cannot
move is a stronger signal than a number that looks wrong.**

**THE ASSUMPTIONS.**  The random-walk bridge `sigma*sqrt(N)` is the standard memristor
picture and is an ASSUMPTION, not a measurement here -- and the model's own semantics are
**i.i.d. per read** rather than cumulative, so under the model as written the correction is
stale on every read and "how often to re-measure" does not arise.  **The interval is what a
cumulative device would need, computed from a tolerance this repository did measure.**  A
wall-clock interval additionally needs a read rate, which is a device property.

**SCOPE.**  Repository tasks `4d` and `steps`, 40 nodes, 3 seeds, toy; the realization model
is `ext_15.realize`, imported rather than reimplemented.

### 12.26 Item 7: the blocker is not the hook, and `ext_03`'s status is a necessity (ext_41)

**The open item, in this document's own words** (section 11, item 7):

> "**`ext_03` against the real base trainer.**  It mirrors the base's loop rather than calling
>  it, **because the base's closed-form hook has no place for a rank constraint.**  Wiring
>  that hook would make the comparison authoritative instead of parallel."

**THE STATED BLOCKER IS REAL AND IS NOT THE BINDING ONE.**  Two things are true and only the
second one blocks.

**1 -- THE HOOK CAN BE WRAPPED, AND THAT NEEDS NO BASE EDIT.**  `train_model` calls
`solve_readout_closed_form` as a **bare module global**, which is exactly the property 12.19
used to wrap the gate's deploys.  With the global replaced and a multi-output readout:

| readout | `k` | singular values (normalised) | effective rank |
|---|---|---|---|
| 1 row | full | [1.0000] | 1 |
| 3 rows | full | [1.0000, 0.4948, 0.0949] | **3** |
| 3 rows | **1** | [1.0000, 2.7e-09, 4.1e-10] | **1** |
| 3 rows | **2** | [1.0000, 0.4948, 1.7e-10] | **2** |

**So the rank projection the item says cannot be wired, can be** -- in a few lines, from the
extension, with the base untouched.  **The item's reason for leaving this undone does not
hold.**

**2 -- WHAT ACTUALLY BLOCKS IS THE TARGET, NOT THE HOOK.**  Probed with the base's own loop
on the repository's own architecture and task:

| target | shape | base loop |
|---|---|---|
| scalar (`m = 1`) | `(256,)` | **runs** |
| multi-output (`m = 2`) | `(256, 2)` | **RAISES** `IndexError: boolean index did not match indexed array along axis 0` |

**The failure's SHAPE is the finding**: a boolean mask built for a 1-D target applied to a
2-D one.  **`train_model` is scalar-output by construction**, and half A's rank question is
inherently multi-output -- `ext_02` uses `m = 8`, and at `m = 1` the constraint is vacuous,
which 12.22 measured from the other side on the repository's own 2-class tasks.

**3 -- SO `ext_03`'s PARALLEL STATUS IS A NECESSITY, NOT A SHORTCUT.**  Its own disclaimer
calls the mirroring forced and says only the difference between its arms means anything.
**That is not an apology more wiring would remove; it is the strongest form the comparison
can take.**  Making it authoritative would mean changing the base version's **trainer**, not
wiring a hook -- a different order of change from the one item 7 describes.

**AND THE TWO LIMITATIONS LINE UP EXACTLY: the loop that works is the loop where the rank
question is empty.**  Every scalar run's readout has rank 1:

| seed | readout shape | rank | test `R^2` |
|---|---|---|---|
| 0 | (1, 4) | **1** | +0.5348 |
| 1 | (1, 4) | **1** | +0.5277 |
| 2 | (1, 4) | **1** | +0.5273 |

**4 -- WHAT A BASE CHANGE WOULD HAVE TO TOUCH**, stated so the decision is informed: the
validation mask and the `r2` path in `train_model` assume a 1-D target, and the hook's
`rows=(0,)` default assumes a scalar readout.  **Two places, both in the base's trainer, and
nothing here does it.**

**5 -- AND MY OWN FIRST DRAFT OF P2 CLAIMED SOMETHING FALSE.**  It ran the wrapper through
the base's loop on a **scalar** model, showed one singular value before and one after, and
said "the singular values show the truncation taking effect".  **They show no such thing** --
a `1 x N` readout has rank 1, so there was nothing to truncate.  **The tell was the single
singular value**, and the fix was to demonstrate the mechanism where it can exist: on a
multi-output readout, called directly rather than through the loop that cannot carry one.

**THE HONEST LIMIT.**  `ext_03`'s arms differ only in the readout on one identical geometry
loop, whose constants its header states.  So the difference it measures is carried entirely
by the code it mirrors, and **the mirroring is the only place a protocol drift could hide.
That remains untested**, and it is this section's limit rather than a result.

**SCOPE.**  The repository's own `F2` architecture, its own `4d` task, its own `train_model`.
Nothing here edits the base; the wrapper is installed and removed inside one process.

### 12.27 Item 2: the lever the diagnostic named, and it works but does not pay (ext_42)

**The open item, in this document's own words** (section 11, item 2):

> "The open item is therefore a **GUARD on the selector** ... `ext_21` measured three
>  candidate guards and **none works** ... `ext_22` then LOCATED it: ... **It is a
>  BIAS-VARIANCE trade and the SAMPLE COUNT binds**: the shipped rule is defensible not
>  because it is accurate but because it is STABLE."

**ALL THREE GUARDS WERE DIFFERENT STATISTICS ON THE SAME 51 SAMPLES.**  Guards on the
prediction spread, on the conditioning, on the shrinkage -- **the same split, three
statistics.**  **The lever the diagnostic itself names, the sample count, had never been
varied anywhere in this extension.**  That is what 12.27 varies.

**AND IT IS PRICEABLE.**  More selection samples means more MEASURED probes, at the base's
own rate; scoring more candidates on a bigger set costs only compute, because they reuse the
same measured feature matrix.  **So the purchase is well defined and comparable against the
other thing the same measurements buy -- nodes.**

**P1 -- THE CONTROL, AND `ext_22` REPRODUCED.**  Enlarging the selection set with FRESH
probes, fitting set held fixed at the shipped 205 samples:

| `n_sel` | `steps` shipped | `steps` oracle | `4d` shipped | `4d` oracle |
|---|---|---|---|---|
| **51** | +0.1587 | **-0.1927** | +0.5441 | **-1.2580** |
| 102 | +0.2038 | +0.0645 | +0.5433 | +0.1772 |
| 205 | +0.2376 | +0.1339 | +0.5943 | +0.3519 |
| 410 | +0.3069 | +0.3913 | +0.5729 | +0.6346 |
| 820 | +0.2370 | **+0.3953** | +0.5634 | **+0.7020** |

> **AT 51 SAMPLES THE ORACLE IS CATASTROPHIC** -- it ranks every candidate by its held-out
> score and lands at `-1.26` on `4d` -- **which is exactly the overfitting `ext_22`
> described.**  And **the ceiling RISES with the selection set on both tasks, by `+0.5881`
> and `+1.9600`.**  **So the sample count is the lever the diagnostic named, and it works.**

**P2 -- THE PURCHASE, PRICED, AND IT IS TASK-DEPENDENT.**

| task | `n_sel` | buys nodes | shipped gain | node-arm gain | winner |
|---|---|---|---|---|---|
| `steps` | 102 | 10 | +0.0448 | **-4.7381** | selection |
| `steps` | 205 | 30 | +0.0789 | +0.0735 | selection |
| `steps` | 410 | 70 | +0.1482 | +0.0894 | selection |
| `steps` | 820 | 150 | +0.0783 | +0.1616 | NODES |
| `4d` | 102-820 | 10-150 | small | positive at every point | **NODES** |

> **COUNTING HEADS GIVES 'NODES WIN 5 OF 8', AND THAT COUNT IS NOT A RESULT.**  An earlier
> draft of the verdict stopped there.  **`steps` at 102 is a NODE-ARM BLOW-UP** -- the node
> arm gains `-4.7381`, a geometry whose prediction explodes outside the basis support, which
> is the failure mode `ext_21` named -- **so selection "winning" there says nothing about
> selection.**  And the two tasks disagree in a STABLE way: `4d` favours nodes at every
> point, `steps` at three of four.

**3 -- AND THE COMPARISON IS VARIANCE PER MEASUREMENT, NOT ONLY ACCURACY PER MEASUREMENT.**
Buying nodes buys a bigger geometry **search**, and this project has measured that the search
can produce an exploding geometry.  Buying selection samples buys a better-posed **choice**
among whatever geometries the same search produced.  **On `steps` the node arm's variance is
visible in this very table.**

**4 -- SO THE GUARD SHOULD NOT BE BUILT, and for a better reason than "it does not pay".**
Item 2's candidate repairs -- three guards, a criterion replacement, and more samples -- now
all have measured answers, and **only the samples raise anything.**  **The honest closure is
the one `ext_22` reached, now with the untried direction measured rather than assumed: the
shipped rule is defensible neither as accurate nor as improvable by guarding, but as the most
STABLE option per measurement.**

**AND ONE DEFECT OF MY OWN.**  A patch script modified the verdict text in memory, then died
on a LATER assert, and wrote the file only at the end -- **so it saved nothing while printing
`ok` for the first edit.**  The re-run's report then carried the OLD verdict and the NEW
closure **side by side**, which is how it was caught: **by reading the output rather than
trusting the patch's own success message.**  Same class as defect 54, with a new way to fail
quietly.

**THE LIMITS.**  The oracle is a CEILING on what a selector could do with a set, **not a
selector** -- it re-solves every strength for every candidate and is itself an overfitting
risk at small `n_sel`, which is exactly why it cannot be shipped.  Repository tasks, 40
nodes, 3 seeds, toy.

### 12.28 The closed-form calibration inside the deep trainer (ext_43)

**The worklist item.**  The project's largest single claim is that the source strengths
should be CALIBRATED in closed form rather than searched -- section 6 puts it at 15-30x fewer
measurements -- and neither deep path uses it: `training/himal_nd.py` runs per-parameter FD /
SPSA on EVERY parameter, and the CIFAR line runs autograd.  **So on the project's own deepest
demonstration the strengths were still searched.**

**SCOPE, CHECKED BEFORE ACTING.**  `HiMALND`'s regression path is `out = (hA + hB)/2` with
`h = z W^T`, i.e. LINEAR in the strengths, and `train`'s docstring says it is STAGEWISE
RESIDUAL -- the structure `solve_readout_closed_form` is built on.  Its classification path is
nearest-prototype voting and is not linear, which is the case `DESIGN.md`'s table already
marks "no".  **So the calibration applies to regression and not to classification.**

**AND A SIMPLIFICATION FOUND ON THE WAY.**  `_forward_layer` returns `out_g + out_a`, i.e.
`z (W_gauss + W_act)^T`.  **The two channels are DEGENERATE -- only their sum matters** -- so
the solve targets the sum and the other channel is zeroed.  Solving one and leaving the other
would add the old weights back on the next forward and silently undo the solve.

**THE INTERVENTION was a wrapper on `_spsa_step`** -- the base's loop runs unmodified and the
readout is replaced by the closed-form ridge solution of the layer's own residual target after
each update.  No base edit.  **The comparison is CONSERVATIVE**: the calibration arm still
spends the search's forward passes on those weights before overwriting them.

**THE FIRST RUN REFUSED TO CONCLUDE, and that was the control working.**

| seed | SEARCH (shipped) | CALIB (wrapped) | readout `R^2` during training |
|---|---|---|---|
| 0 | +0.0054 | -0.0008 | **0.3440** |
| 1 | -0.0486 | -0.0322 | **0.3826** |
| 2 | -0.0592 | -0.0280 | **0.3700** |

**The search arm averaged `-0.0341`, below the project's own 0.05 floor, so the script refused
to compare** -- an earlier configuration of it (`output_dims=(1,1,1)`, `eps=1e-4`, a 4-D task)
had produced "calibration better on 3 of 3, mean +0.1292", **which was a contest between two
failures and is withdrawn.**

**AND THE DIAGNOSIS WAS THE DELIVERABLE: the per-layer solve fits its own residual target to
`R^2 ~ 0.34-0.38` while the end-to-end is `~0`.  So the readout was not the bottleneck.**

**THE SECOND RUN, AFTER 12.29's FIX, IS THE FIRST REAL COMPARISON ON THIS PATH.**

| seed | SEARCH | CALIB | difference |
|---|---|---|---|
| 0 | +0.2195 | +0.2552 | +0.0357 |
| 1 | +0.2473 | +0.2457 | -0.0016 |
| 2 | +0.2204 | **-0.0265** | **-0.2469** |

**The control passes (`+0.2291 > 0.05`) and the answer is NEGATIVE: the calibration does not
help (1 of 3 seeds, mean `-0.0710`), and on seed 2 it materially hurts.**  So the readout is
not the deep path's bottleneck and **the search is earning its forwards on the GEOMETRY.**

**AND THE COST SIDE IS NOT REACHED** -- the arm that pays `n_train * N` measurements does not
even win on accuracy here.

**THE HONEST LIMIT, WHICH 12.35 HAS SINCE CLOSED.**  The calibration is applied AFTER the SPSA step, so a true "calibrate, do
not search" arm would remove those coordinates from the parameter vector -- a change to
`get_params_vector`, a base method -- and **12.35 has now made it, without editing the base:** the
arm masks the perturbation and the update to the geometry subspace, and the answer is that the
bound recorded here **was tight** (`calib -> calib_only` is `-0.0008`).  **So the sentence that
used to end this paragraph -- "the measurement bounds the gain from below rather than measuring
it" -- no longer applies**, and what it protects is unaffected either way: the calibration does
not help the deep path, and the search earns its forwards on the geometry.

**AND THE NUMBERS ABOVE PREDATE 12.29's FIX.**  This table's `-0.0710` mean comes from a run
before `HiMALND.predict` was corrected, so it is not directly comparable to 12.35's `+0.0779` for
the same arm; 12.35 shows the two runs agreeing on the calibration's own RFit to four decimals.

---

### 12.29 `himal_nd`'s regression prediction dropped the accumulation (ext_44), and it is FIXED

**How it was found.**  12.28 could not compare because the deep regression path scored below
the floor.  Its per-layer readout solved to `~0.37`, so the failure was downstream -- and the
code says where.

    training:  pred_i = _proto_predict(h_i, y_proto_i, h_proto_i)
               acc    = acc + pred_i                 # running sum, layer by layer
               # "At predict time these are summed -> recovers original y"
               # layers 2/3 are trained on `y - acc`, so each layer emits a RESIDUAL

    predict:   return (h_last_A + h_last_B) / 2      # the last layer's RAW OUTPUT

**`predict` returned a residual, not the target -- and its own docstring says the opposite**
("Regression: proto-weighted prediction with residual accumulation.  output =
`(acc_A + acc_B) / 2`").  **Code and docstring disagreed.**

**THE MEASUREMENT: one trained model, two prediction paths.**

| seed | pre-fix `predict` | accumulated | difference | zone A | zone B |
|---|---|---|---|---|---|
| 0 | +0.0054 | **+0.2195** | +0.2140 | +0.0109 | -0.0095 |
| 1 | -0.0486 | **+0.2473** | +0.2959 | -0.0237 | -0.0877 |
| 2 | -0.0592 | **+0.2204** | +0.2796 | -0.1075 | -0.0197 |
| **mean** | **-0.0341** | **+0.2291** | **+0.2632** | | |

> **A DEFECT, NOT A PROPERTY OF THE ARCHITECTURE.**  The same models score `+0.2632` better
> through the path `train` was built around -- **the network learned and the answer was read
> out wrongly** -- and **the per-zone columns rule out the alternative explanation**, since
> the A/B median split scores near zero in BOTH zones under the old return, not just on
> average.

**WHY IT SURVIVED.**  `himal_nd`'s regression path is exercised by exactly two scripts:
`nd_09_spsa_benchmark.py`, a SPEED benchmark reporting loss curves, and
`reg_01_proto_baseline.py`, which 12.17 RETRACTED.  **No published number depends on that
path** -- which is why the defect lived -- **and it would have corrupted the first serious use
of it.**

**THE FIX**, applied to `training/himal_nd.py` at the repository owner's instruction: a new
`_proto_accumulate` helper runs one network's cascade and sums each layer's proto-weighted
contribution from the prototypes `train` already stores; the regression branch of `predict`
now returns that sum; and the old return is kept as an explicit FALLBACK for a model restored
without its prototypes, with a comment saying so rather than silently returning a residual.
The classification path is untouched.

**VERIFIED THREE WAYS.**  The fixed `predict` reproduces the independent replay to `5e-06`
(float32 precision) on all three seeds; 12.28's re-run then passes its control; and `ext_44`
itself now compares the FIXED path against an IN-LINE RECONSTRUCTION of the pre-fix behaviour
-- kept precisely so that fixing the defect does not erase the evidence for it, **which makes
that script a REGRESSION TEST: if the old return is ever reintroduced, the two columns diverge
again.**

**AND IT LEAVES TWO CLAIMS THAT BOTH STAND**, which is why 12.28's two runs are reported
separately: a calibration cannot repair a broken prediction path, AND on the repaired path it
is not the thing that helps.

### 12.30 The deep regression path now works, and it is beaten by the null (ext_45)

**The item, created by 12.29's fix.**  Before it, `HiMALND.predict` returned a residual and
the regression path scored `-0.0341`.  After it, `+0.2291`.  **So the path clears the
project's own 0.05 floor for the first time -- and nothing in the repository said whether that
number is good.**  Its only two users are a speed benchmark and a retracted script, and
neither reports an accuracy.  **A number with no reference point is not a result.**

**THE COSTS ARE COUNTED, IN TWO UNITS THAT ARE NEVER ADDED.**  The project's unit is forward
evaluations and both paths are exact in it: SPSA is documented as "2 * spsa_avg forward
passes" per epoch per network, so the deep arm's `2 * 60 * 3 * 2 = 720` is analytic; and
`fit_operator` already returns `n_evals` -- "the cost unit here is forward evaluations and a
restart that is not paid for is not a method".  **The closed-form solve costs `n_train * N`
READINGS instead**, and the shallow arm pays it while the deep arm does not, so the two are
reported side by side.  **The shallow arm is SWEPT across its own cost rather than matched by
hand**, which is the honest way to compare against a budget.

**THE REPOSITORY'S OWN TWO REGRESSION TASKS, 3 SEEDS, PAIRED.**

| task | arm | forwards | calib readings | test `R^2` |
|---|---|---|---|---|
| `2d` | **deep (`himal_nd`)** | 720 | 0 | **+0.2291** |
| `2d` | **RANDOM geometry + solve** | **0** | 20 | **+0.9006** |
| `2d` | shallow, one layer `N=20` | 214 | 4 100 | +0.9585 |
| `2d` | shallow, one layer `N=40` | 411 | 8 200 | +0.9825 |
| `4d` | **deep (`himal_nd`)** | 720 | 0 | **+0.0548** |
| `4d` | **RANDOM geometry + solve** | **0** | 20 | **+0.2516** |
| `4d` | shallow, one layer `N=20` | 205 | 4 100 | +0.4479 |
| `4d` | shallow, one layer `N=40` | 408 | 8 200 | +0.5817 |

> **ON BOTH OF THE REPOSITORY'S REGRESSION TASKS THE DEEP PATH IS BEATEN BY THE NULL** --
> by `+0.67` on `2d` and by `+0.20` on `4d` -- and the null spends **ZERO forwards** and 20
> readings against the deep arm's 720 forwards, three layers and 60 particles.  **On `4d` the
> deep arm scores `+0.0548`, i.e. it barely clears the floor the shallow path exceeds five
> times over.**
>
> **AND THE NULL IS NOT A STRAW MAN.**  Random geometry plus a closed-form solve is the
> extension's own standard baseline, used in every half-B result, and it is exactly what this
> project's central claim predicts should be strong: the strengths are SOLVED, so the only
> thing left to get right is the geometry -- and on these tasks random geometry is already
> most of the way there.

**AND ONE CLAIM OF MINE THAT `4d` REFUTED, withdrawn rather than kept as a caveat.**  The
first verdict for `2d` said the deep path "exists for the harder problems in this repository
-- the 4-D and image lines", i.e. that an easy task is where it loses.  **`4d` refutes it:**
the deep arm loses to the null there too, and by a wider relative margin.  **A caveat the data
contradicts is worse than no caveat**, so the sentence is gone from the script and recorded
here.

**WHAT THE SCOPE LIMIT ACTUALLY IS.**  Toy scale; one deep configuration whose
hyper-parameters were inherited from a pipeline test and never tuned; and the repository's two
REGRESSION tasks.  **The image line is CLASSIFICATION, whose readout is nearest-prototype and
is not what this compares** -- so nothing here says anything about it either way, and that is
the one place a deep design still has room in this repository.

**THE HONEST LIMITS.**  `n_evals` is the shallow arm's own accounting and the deep arm's is
analytic from SPSA's documented cost.  Neither counts the feature-matrix evaluations inside
the closed-form solve, because those are readings rather than forwards.  3 seeds, toy.

### 12.31 `HiMALND`'s classification mode, and it is beaten by the null too (ext_46)

**SCOPE FIRST, BECAUSE A NOTE OF MINE WAS WRONG.**  The worklist said the classification path
"is exactly what the CIFAR line uses".  **That is false.**  Searching the CIFAR family for
`himal` returns NOTHING: that line is `RBFLayerND` trained by PyTorch autograd and exported to
a physical layer, a DIFFERENT architecture.  **So nothing here says anything about the
project's image demonstration**, and the note is corrected rather than left to be quoted.

What is true: `HiMALND`'s classification mode is used by `nd_01`, `nd_03`, `reg_03`, `reg_05`
and `run_all`, and `nd_01` DOES report an accuracy -- against a **v1 ResidualRBFNetwork** and
an MLP, i.e. against **sibling architectures**.  **What it had never been compared against is
this project's OWN standard**: the operator's placement, the closed-form solve, and the
random-geometry null.

**THE COSTS ARE THE SAME TWO UNITS, NEVER ADDED.**  The deep arm's 720 forwards at 60 epochs
is analytic from SPSA's documented "2 * spsa_avg forward passes"; `fit_operator` returns
`n_evals`; and the closed-form solve costs `n_train * N` READINGS.

**THE RESULT, ON THE TWO CLASSIFICATION TASKS THE REPOSITORY CAN BUILD WITHOUT DOWNLOADING.**

| task | arm | forwards | calib readings | accuracy |
|---|---|---|---|---|
| `moons` | **deep (`himal_nd` classification)** | 720 | 0 | **0.7344** |
| `moons` | **RANDOM geometry + solve** | **0** | 4 100 | **1.0000** |
| `moons` | operator + solve, `N=20` | 225 | 4 100 | 1.0000 |
| `moons4d` | **deep** | 720 | 0 | **0.7417** |
| `moons4d` | **RANDOM geometry + solve** | **0** | 10 240 | **0.9479** |
| `moons4d` | operator + solve, `N=40` | 436 | 20 480 | 0.9917 |

> **THE DEEP CLASSIFICATION ARM DOES NOT BEAT THE NULL EITHER** -- and this arrives through a
> COMPLETELY DIFFERENT READ-OUT (nearest-prototype voting against the regression path's
> accumulation), **so it is no longer a property of that one prediction path.**  Four
> task/read-out combinations now agree: `2d` regression `+0.2291` against a null's `+0.9006`,
> `4d` regression `+0.0548` against `+0.2516`, and these two.

**AND ONE OF THE TWO TASKS IS SATURATED, WHICH LIMITS WHAT IT CAN SAY.**  On `moons` the
closed-form arms reach **1.0000**, a ceiling: that task shows only that the deep arm is *not*
perfect, not how far behind it is.  **`moons4d` is the one that discriminates** -- `0.9479`
for the null against the deep arm's `0.7417`, and `0.9917` for a 40-node operator arm -- so
the claim rests on that task and `moons` is reported as a ceiling rather than as evidence of
a gap.

**AND AN ASYMMETRY IN THE NULL, STATED RATHER THAN HIDDEN.**  The closed-form arms read the
classes as `+-1` and take a sign -- a LEAST-SQUARES classifier, not a margin method, and not
what the deep arm's prototype voting does.  **Each arm is scored the way it is built to be
scored**, which is the fair comparison and also one that handicaps the closed-form side on
tasks where least squares is misspecified.

**THE LIMITS.**  Toy scale, two 2-class tasks, and a deep arm shipping with hyper-parameters
inherited from a pipeline test and never tuned.  **And the CIFAR line is untouched and
unmeasured here**, which is the largest caveat this section carries.

### 12.32 The CIFAR line against the project's own standard: UNRESOLVED (ext_47)

**Why this was the last unmeasured thing.**  12.30 and 12.31 benchmarked `HiMALND`'s
regression and classification modes and found BOTH beaten by a zero-forward null.  **The CIFAR
line is a DIFFERENT architecture** -- `RBFLayerND` trained by autograd and exported to a
physical layer -- so the project's most impressive demonstration had never been compared
against the project's own standard.

**AND ITS NUMBER WAS ALREADY IN THE REPOSITORY.**  `cifar_02` writes
`outputs/e21_regularized_report.txt`, which reports a DIGITAL TEST ACCURACY of **39.3%** on 5
classes, a train accuracy of 53.6% -- an **18.3% train-test gap** -- and a collapse to 21.3%
at the `high` physical-noise preset.  **So the reference point needed reading, not a 600-epoch
torch run**, and the file is untracked, which is why no measurement had ever cited it.

**THE BASELINE, AND THE ADAPTATION THE TASK FORCES.**  On CIFAR the layer's input is a POINT
CLOUD (`build_3d` turns each image into `(x, y, intensity)` points), and a closed-form solve
needs a vector -- so the standard becomes `z_j(image) = mean over the image's pixels of
exp(-k ||p - c_j||^2 / h_j^2)`, then a ridge to the 5-class one-hot.  Random geometry is
reported as the **MEAN over draws and not only the best**, because a cherry-picked null is not
a null; k-means is included as a standard closed-form geometry.

| arm | test accuracy |
|---|---|
| random null, mean of 5 draws | 0.3582 |
| random, best draw | 0.3756 |
| k-means centres | 0.3422 |
| **the repository's own deep pipeline (reported)** | **0.3930** |
| chance | 0.2000 |

> ### UPDATE, after the recording defects in this section were fixed
>
> Two things changed after this section was written, and **both make the non-resolution
> stronger rather than weaker.**
>
> **1. THE DEEP NUMBER MOVES BY 3.3 POINTS WHEN THE SEED IS FIXED.**  The reported `39.3%`
> came from a run whose torch initialisation was NEVER SEEDED -- all seven CIFAR scripts
> seeded only the data split, the third instance of that defect in this repository.  Re-run
> with `SEED = 0`:
>
> | | published (unseeded) | **seeded** |
> |---|---|---|
> | digital test | **39.3%** | **36.0%** |
> | digital train (peak) | 53.6% | 46.2% |
> | train-test gap | 18.3% | 11.3% |
> | physical noise `none` -> `high` | 39.3% -> 21.3% | 32.7% -> 24.0% |
>
> **The seed alone moves the number `0.033`, and the margin this section was trying to
> resolve is `+0.030`.**  So the comparison is not merely underpowered against the closed-form
> null's spread -- **the deep arm's own single-draw variation is the same size as the effect.**
> Both artifacts are kept: the unseeded one as `e21_regularized_report_UNSEEDED.txt`, because
> it is the record of what was published, and the seeded one as the report.
>
> **2. MORE SEEDS DO NOT SHRINK THE NULL'S SPREAD.**  Ten seeds instead of three:
>
> | arm | 3 seeds | **10 seeds** |
> |---|---|---|
> | random null (mean) | 0.3582 | **0.3628** |
> | random (best draw) | 0.3756 | **0.3820** |
> | k-means centres | 0.3422 | **0.3633** |
> | margin over the null's mean | +0.0348 | **+0.0302** |
> | **the null's seed-to-seed spread** | **0.1507** | **0.1507** |
>
> **The spread did not move at all.**  More seeds ESTIMATE a spread; they do not SHRINK it --
> it is a property of the estimator, not of the sample size -- so the premise behind raising
> the seed count was itself wrong.  The conclusion stands, now on ten seeds instead of three.
>
> **3. AND THE RECORDING DEFECTS ARE FIXED RATHER THAN NOTED.**  The flagship report was
> untracked, `cifar_06` HARD-CODED the result as `rbf_test=39.3`, and no CIFAR report named
> the seed that produced it.  All seven scripts now seed the model initialisation, record
> `SEED=` in their reports, and the reports the documents cite are un-ignored in
> `.gitignore`.  **The `39.3` literal is left in `cifar_06` on purpose**: changing it would
> silently alter a comparison's input, and the point of this section is that such a literal
> is exactly what should not exist.  It is now at least a literal whose provenance is
> tracked.

**AND THE VERDICT IS THAT THIS DOES NOT RESOLVE THE TWO ARMS.**  The ranking above would read
as "depth wins", and **an earlier draft of it did** -- "the depth is doing work the single-layer
closed form cannot".  **That is the error this section exists to avoid:** the null's own
**seed-to-seed spread is 0.1507** (0.3027 to 0.4400), against a margin of `+0.0348` between the
deep number and the null's mean.  **The spread is eight times the margin**, so three seeds
cannot separate these arms, and **CIFAR neither confirms nor refutes the pattern the other four
combinations found.**

**AND THE SCALE EXPLAINS WHY.**  Chance on 5 classes is `0.2000` and every arm sits between
`0.303` and `0.393` -- a few points above chance.  **Nothing is learning much on this task at
this scale**, so there is little for a comparison to separate.  **The deep pipeline's own
report says the same from the other side**: an 18.3% train-test gap, and a collapse to 21.3%
under physical noise, which is essentially chance.

**WHAT WOULD RESOLVE IT, so that this is actionable rather than a shrug:** far more than three
seeds on the closed-form side -- **the deep number is a single reported run and cannot be
re-seeded at all**, which is itself a defect in how that line records its results -- and a
scale at which the arms are not all within a few points of chance.

**AND ONE BUG OF MY OWN, of exactly the class 12.29 was about.**  The feature function's
docstring said "mean over the image's pixels" and its code averaged the wrong axis, which the
assignment rejected on the first run.  **The docstring was right and the code was not** -- the
same shape as the defect in `HiMALND.predict` -- and it is recorded because that shape has now
appeared twice in this document.

**THE LIMITS.**  The encoding runs at `IMG_SIZE=16` rather than the repository's 32, for cost;
the cloud aggregation is an adaptation this task forces; the baseline is single-layer against
the repository's four; and **the deep number is READ FROM A REPORT rather than re-run**, so it
carries that report's provenance and not a re-measurement.

### 12.33 The lookup table, and what it actually buys (ext_48)

**THE PRECONDITION IT RELAXES, AND IT IS NOT 12.16's.**  12.16 requires `log z(x)` to be LINEAR
in a fixed FINITE basis of `x`.  A table requires something DIFFERENT and WIDER: the source must
be a **FIXED function of ONE SCALAR**,

    z = T(u),   u = ||(x - c)/h||^2

because then a single 1-D table serves every particle and every width.  The table's entries are
what a calibration would MEASURE from the device, which is why this belongs in the device and
not in the trainer -- where `features()`' docstring forbids it outright.

**AND WHAT IT DOES NOT ADMIT, SAID BEFORE ANY RESULT.**  Yukawa's `exp(-r)/r` is NOT of that
form: its SHAPE depends on `h`, so one table of `u` cannot carry it and `h` would have to index
a family of tables.  **The claim is "any kernel that is a FIXED function of `u`" -- wider than
"any polynomial exponent", narrower than "any kernel".**

| what was measured | result |
|---|---|
| **P1 the error law** | the measured max error is **at most `(du)^2/8`** and approaches it from below as M grows (ratio `0.445` at M=8 to `0.987` at M=512).  **Below 1 at small M because the sample points do not sit at the worst-case midpoint of an interval** -- so it is an UPPER BOUND, which is the useful direction: a table size can be CHOSEN rather than tuned. |
| **P2 does the suite detect a table?** | **YES, 3 of 3, with NO code change.**  `node_from_response` already returns `log_rms`, the consistency residual: exact `3.9e-15`, M=512 `2.5e-05`, M=64 `1.6e-03`, M=16 `2.8e-02`. |
| **P3 tier 1 -- table in the FORWARD only** | at M=256 the end-to-end R^2 differs from exact by **`+7.5e-05`**, and one node's geometry is recovered with a centre error of **`2.4e-05`**.  The operators are untouched and still work. |
| **P4 the payoff** | `T(u) = exp(-sqrt(u))` is the LAPLACE kernel `exp(-||x-c||/h)`, whose `log z` is a CONE -- not polynomial in `x` at any degree, so 12.16 excludes it.  **The closed-form operator's own residual says so**: Gaussian `3.1e-06` against Laplace **`7.1e-02`**.  **The table computes that source**, so what it changes is which sources the FORWARD can realise -- **and it does not pretend the closed-form operator now covers them.** |

**WHY IT IS NOT ABOUT SPEED.**  A table plus interpolation measured about 2x SLOWER than
vectorised `exp` (gather-bound), so had this been a performance claim it would have failed.
**The project's cost unit is MEASUREMENTS, not seconds**, so the slowdown cannot change any
published conclusion.  **What the table buys is REACH.**

**AND A DEFECT CLASS FOUND WHILE CHECKING THIS SCRIPT'S OWN REPRODUCIBILITY.**  `ext_48` was
written with a seeded generator for its own data, and **was still not reproducible**: two runs
differed in six lines, all of them on the DEVICE path.  **The cause is that `features()` builds
a real device layer and the device draws from the GLOBAL `np.random`, which the script never
seeded.**  An audit then measured the extent: **24 of the 25 extension scripts that call
`features()` do not seed the global RNG.**

**The magnitude bounds the damage, and it was measured rather than assumed**: the movements are
`2.8e-07 <-> 5.3e-07` in a centre error, `2.0e-06 <-> 3.0e-06` in `log_rms`, and `+7.52e-05 <->
+7.59e-05` in an R^2 delta -- **the 5th to 7th digit, while the reports quote four decimals.**
**So no published conclusion changes.  But a re-run does not reproduce a report byte for byte,
and anyone diffing two runs will see differences.**  And one place makes it matter: the HEALTHY
BAND of `log_rms` is `~1e-6`, so the test that decides "is this response a genuine single node"
sits AT the magnitude of the non-determinism.  **That is the same situation as 12.32's
unresolved `sub_01` reproducibility**, and it is now on the record as the likely shared cause
rather than a separate mystery.

**THE LIMITS.**  The table is a 1-D approximation, so its error enters the FORWARD and not the
operators; a source whose `log z` is non-polynomial still needs a different recovery than the
closed-form one -- **the table makes it computable, not solvable**; and Yukawa is out, for the
reason stated first.

**TWO INTERFACE ERRORS OF MINE, BOTH CAUGHT BY READING THE CALLEE FIRST.**  `node_from_response`
takes `k` FIRST and positionally, and it recovers ONE node, so `z` is a 1-D response of length n
-- not the `(n, N)` matrix.  The first version got both wrong; the operator refused with
"X has 40 rows but z has 800", **which is the operator behaving correctly.**

### 12.34 The h-indexed family: reaching Yukawa, and the table's parameterisation (ext_49)

**WHERE 12.33 STOPPED.**  It covered any source that is a FIXED function of the single scalar
`u`, and said plainly that Yukawa's `exp(-a r)/r` is not one -- its SHAPE depends on `h` -- and
that "`h` would have to index a family of tables".  This is that family, measured.

**AND PHYSICAL ADMISSIBILITY COSTS THE ONE-PARAMETER VERSION.**  With `r = h s`, `s = sqrt(u/k)`,

    z = exp(-a h s)/(h s + r0)   =>   h z = exp(-beta s)/(s + r0/h),   beta = a h

so `h z` is a function of `(u, beta)` **ONLY WHEN `r0 = 0`** -- which is the singular form the
framework cannot sample, because it measures at the centre.  **The one-parameter family therefore
reads `inf`, and does not improve with more tables**, because the mismatch is algebraic rather
than a matter of resolution.  A finite core is not a refinement of this kernel; it changes which
family is required.

**A SECOND AND MORE GENERAL FINDING: THE TABLE'S PARAMETERISATION MUST MATCH THE KERNEL'S
SMOOTHNESS.**  The first two-parameter version still read `~2.2` and **hardly moved when the
table grew from 4 entries to 64**, which is the signature of an error that is not resolution.
The cause: it was gridded uniformly in `u`, while the kernel is `exp(-beta s)/(s + r0)` with
`s = sqrt(u/k)`, and `s` has an INFINITE derivative at `u = 0` -- so a uniform `u` grid samples
the sharpest part of the function with its sparsest spacing.  Gridded in `s` instead:

| gridded in | 4 x 8 tables | 16 x 32 tables |
|---|---|---|
| `u` | 2.219e+00 | 2.206e+00 -- **does not converge** |
| **`s`** | **4.736e-02** | **3.993e-03** -- **converges** |

**`u` is the right variable for a GAUSSIAN** -- 12.33 tabulates `exp(-u)` and converges as
`(du)^2/8` -- **and the wrong one for anything carrying a `sqrt(u)` or a `1/r`.**

**THE FORWARD IS NOW GOOD ACROSS WIDTHS**: six widths from `h = 0.25` to `3.0` all fall inside
the family, with errors from `4.0e-03` to `5.0e-02`.

**AND THE RECOVERY IS WHERE THE PRICE IS PAID, stated where the number is.**  The closed-form
operator does not apply -- measured here, not inherited: its `log_rms` on a Yukawa response is
`4.16e-01` against a Gaussian node's `~3e-06`.  A table-based fit recovers the geometry while
spending **ZERO device forwards** (960 table reads, 0 forwards), so **the project's cost claim
survives: it was always "no device forwards beyond the calibration", not "closed form".**  **But
the centre is recovered and the WIDTH IS NOT**: centre error `5.99e-02` against a width error of
`2.78e-01`, with the fit's relative MSE at `4.55e-03` -- i.e. already at the table's own error
floor.  **The width is WEAKLY IDENTIFIED for a screened source**: its response is a short-range
bump whose radius is set by `1/a`, so scaling `h` and rescaling the amplitude trade off against
each other.  A Gaussian has no such degeneracy, which is why the closed form could solve for it.
**That is a real loss of guarantee bought with the reach, and it is a different thing from the
loss of closed form.**

**AND TWO OF MY OWN ERRORS, BOTH IN THIS SCRIPT'S REPORTING.**  Its verdict printed
`min(two_rows, key=lambda r: r[2])` where `r[2]` is the u-ENTRY COUNT, so the "best" row was the
smallest table and its entry count (`2.56e+02`) was printed as an error; the print then indexed
`best2[2]` as well, so fixing the sort alone changed nothing.  And the first family gridded in
`u` for the reason above, which the report blamed on the algebra until the constant error at
every table size gave it away.

**THE LIMITS.**  The family covers `beta = a h` up to a stated maximum, so a source whose
screening weakens with width leaves it; the fit is LOCAL, starting from a guess, so it can settle
in a wrong basin where the closed form cannot; and a device must hold a family of tables rather
than one.

### 12.35 Geometry-only search: 12.28's bound is TIGHT, and it is now a measurement (ext_50)

**THE OWED ITEM, in 12.28's own words:**

    "a true 'calibrate, do not search' arm would remove those coordinates from the parameter
     vector -- a change to `get_params_vector`, a base method -- and that is NOT done.  The
     measurement BOUNDS the gain from below rather than measuring it."

**WHAT "REMOVING THE COORDINATES" MEANS, PRECISELY.**  `get_params_vector` concatenates
`centers (D*N) + h (N) + W_gauss (n_vars*N) + W_act (n_vars*N)`, so the strengths are the last
`2*n_vars*N` entries and the mask is exact -- **asserted against the layer's own shape before any
arm runs**, not assumed.  **AND BOTH THE PERTURBATION AND THE UPDATE ARE MASKED**: masking only
the update would let `L(theta + eps*delta)` differ from `L(theta - eps*delta)` through coordinates
that never move, which **biases** the estimate instead of restricting it.

**AND THE BASE IS UNTOUCHED.**  `get_params_vector` was not edited; the ARM masks, using the
same monkey-patch discipline `ext_43` already used.  So the owed change is done **without changing
the base version**, which is the extension's own rule.

**THREE ARMS AT EQUAL FORWARD PASSES, paired seeds:**

| seed | search | calib (12.28's) | **calib_only (this)** | calib RFit | only RFit |
|---|---|---|---|---|---|
| 0 | +0.2184 | +0.0688 | **+0.0655** | 0.4573 | 0.4578 |
| 1 | +0.1966 | +0.1649 | **+0.1661** | 0.3459 | 0.3461 |
| 2 | +0.2824 | -0.0000 | **-0.0004** | 0.4253 | 0.4253 |
| **mean** | **+0.2325** | **+0.0779** | **+0.0771** | | |

**CONTROL FIRST**, because 12.28 failed it once and had to be re-run: the search arm averages
`+0.2325` against the project's learning floor of `0.05`, so this is a comparison between arms
that learned.

> **THE ANSWER: `calib -> calib_only` is `-0.0008`.  THE BOUND WAS TIGHT.**
>
> The forwards that 12.28's arm spent on coordinates it then overwrote **were buying nothing
> measurable** -- below a thousandth on the mean, and the per-seed ordering is not even
> consistent.  **That does not make the arm's DESIGN right; it makes its NUMBER right**, and the
> distinction governs what may be claimed: **the calibration is not paying for the search it
> replaces**, so the calibration's own measurement cost is not offset by a search it makes
> cheaper.  The bound and the measurement agree, and the owed sentence can now be deleted rather
> than carried.

**AND A NOTE ON COMPARABILITY.**  `ext_43` reported a calibration mean of `-0.0710` and this run
reports `+0.0779` for the same arm and configuration.  **That is expected and not a
contradiction**: `ext_43`'s numbers were taken before `ext_44` fixed `HiMALND.predict`, whose
regression branch returned a residual rather than the accumulated prediction (12.29).  The
calibration arm's RFit values agree to four decimals (`0.4573/0.4578`, `0.3459/0.3461`,
`0.4253/0.4253`), which is what shows the two runs computed the same thing and differed only in
how the model's output was assembled.

**AND ONE BUG OF MINE, of the shape this list keeps recording.**  The first version's ridge solve
omitted the transpose, so it died on `layer.W_gauss[:d, :] = W` with "could not broadcast (20,1)
into (1,20)".  `np.linalg.solve` returns `(N, d_out)` while `layer.W_gauss` is `(n_vars, N)`.
**Found by reading `ext_43`'s version rather than by debugging mine** -- the two scripts must
share the calibrate step exactly, since it is not a variable under test, and the read was cheaper
than the reason would have been.

**THE LIMITS.**  One task, one architecture, one scale -- the extension's standing limitation.
The search is SPSA with one averaging and one perturbation size, both inherited from the shipped
trainer and never tuned, so the standing of three arms is a statement about THOSE settings.  And
the arms are compared at equal FORWARDS, which is the project's unit but not the only defensible
one: per-step, the geometry-only arm updates fewer coordinates.

### 12.36 Pre-sensor superposition: DESIGN 7's item 5, answered (ext_51)

**THE ONE DESIGN QUESTION THIS EXTENSION NEVER TOOK UP.**  DESIGN section 7 lists six questions to
settle before coding; five were settled by the work and the sixth read:

    "On the diffusion cell, combining the *s* graphs IN THE FIELD before sensing would cost ONE
     measurement per probe instead of t ... the most promising hardware angle of the whole
     proposal ... Qualified by section 4.1: it is available only when the combination is linear."

The shipped `MultiGraphStack.forward` reads **k graphs** and then combines them numerically.  This
does the combination first and measures what it buys.

**WHY THE EQUIVALENCE IS NOT A NEW PHYSICAL CLAIM.**  The fields may be superposed before the
sensor **because the substrate is linear**, and that is a property this repository VERIFIES rather
than assumes: `sub_01`'s exact-superposition threshold is one of the assertions CI runs first, and
the diffusion cell's own header states "superposition is EXACT because the PDE is linear".  **So
this applies the suite's own verified property one stage earlier** rather than introducing one.

| what was measured | result |
|---|---|
| **P1 the equivalence** | pre-sensor reproduces the shipped stage outputs to **`7.4e-08` relative** and the end-to-end prediction to **`1.1e-07`** -- float32 level on the circuit path, i.e. the same computation, with the sensor moved. |
| **P2 the cost** | **4 sensor reads per probe instead of 12**, at `k = 6` graphs into `m = 2` channels: **a 3.0x cut**, and the ratio is `k/m` by construction because one reading per GRAPH becomes one reading per output CHANNEL.  In the project's unit the multi-graph calibration cost is `n_train * sum(m * N_s)` where it was `n_train * sum(k_s * N_s)`. |
| **P3 the negative control** | a **tanh** combine, built locally, **fails by `0.675` relative** -- **O(1), not rounding.**  So 4.1's qualification is measured rather than quoted, and the library agrees by construction: `graphs.Nonlinear` **raises on instantiation**, calling itself "a deliberate non-feature" because it "must also happen AFTER sensing". |

**AND IT CHANGES COST, NOT ACCURACY.**  No accuracy figure in this document moves: the same
output is computed, from the same fields, by the same graphs.  What changes is how many times a
sensor has to look -- **which is the unit every comparison here is run in**, so this is the one
open design question whose answer moves the project's own headline currency.

**THE LIMITS, AND THE FIRST ONE IS THAT NOTHING PHYSICAL WAS BUILT.**  This is arithmetic on the
repository's own models plus a property it verifies; no device was wired to land two fields on one
sensor, and whether that is arrangeable is exactly the kind of question only hardware settles.
It also requires **both** the combination and the medium to be linear, which P3 measures the cost
of violating.  And it is stated at one configuration: `k = 6`, `m = 2`, two stages.

**AND FOUR INTERFACE ERRORS OF MINE IN THIS SCRIPT, ALL CAUGHT BY READING THE CALLEE FIRST.**
The parameter mask had to be asserted against the layer's own vector length rather than assumed;
`graphs.Nonlinear` refuses to be constructed, so the control was built locally rather than by
un-refusing the library; `PhysicalRBFLayerND` takes all-positional arguments and DERIVES
`dim`/`n_particles`/`n_vars` from array shapes; and the end-to-end comparison initially summed
each stage output to a scalar, collapsing the batch.  **Each was a wrong assumption about a
callee, which is the defect this list records more often than any other.**
