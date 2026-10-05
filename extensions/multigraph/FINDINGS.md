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

40. **An unsteady observation promoted to a finding.**  This extension's audit of the
    BASE version's headline reported that three of its five loss cells were GATE
    REFUSALS, on the evidence that `geometry` and `hybrid` were numerically identical
    there.  Raising the sweep from 3 to 9 seeds left that signature in ONE cell
    (`10d/F2`, 360 of 360 calibrations rejected) and removed it from the other three,
    so what had been reported as a property was a 3-seed coincidence.  Found by the
    same instrument that found defects 34 and 38 -- raising the seed count -- which is
    now the third time it has paid for itself.
41. **An analysis built on a column the base forbids quoting.**  This extension's audit
    of the base headline quoted the ablation-F `speedup` summary (friendly `31.6x`
    against adversarial `31.0x`) and built a whole subsection on it -- the rate/endpoint
    distinction -- including reading the `hetero/G` row as "a rule showing through".  The
    base version's section 6.7 says in bold that this column **must not be quoted**, and
    gives the reason: it credits a curve that reaches its target early and then DECLINES,
    which is exactly what `hetero` does.  **The base had already documented the trap and
    this audit walked into it**, because it read section 11 without reading section 6.7.
    The subsection is withdrawn and an endpoint table replaces it.
42. **A "missing" open item that the same document had already closed.**  The audit went
    looking for section 11's "strongest missing counterexample" -- a cell where the gate
    accepts and the arm still loses -- and found it in four cells of the 9-seed data.
    It was **not missing**: the base version's own section 6.7 had found it at 3 seeds
    (`hetero/G`, one accepted calibration of 120) and section 6.11 had recorded it as
    falsifying the stronger claim.  The audit read an INDEX of open questions instead of
    the findings themselves -- the same failure mode defect 38 records on the extension's
    side, and the reason both documents now carry a warning that their summary sections
    are the least maintained part of them.

43. **A flag stated more strongly than the evidence.**  Section 12.5 flagged `10d/F2`
    as possibly having a rejection counter that disagrees with the deployed model.  The
    code restores the strengths on rejection and the caller's comment matches it, and a
    2-seed re-run reproduced the counter exactly (`320 of 320`) with an arm difference of
    `0.0009`.  **It was a small numerical difference, not a bookkeeping failure**, and the
    flag should have said so.  The re-run then found the real problem next door -- the
    win criterion comparing the hybrid against an arm it is built on (12.6).
44. **A parser that read zero rows, caught by its own coverage assert.**  The criterion
    audit's first version looked for the Part 2 table but broke out of the scan on the
    `====` rule that immediately FOLLOWS the Part 2 title, so it parsed 0 cells of 14.
    The coverage assert fired on the first run, before any number was published -- which
    is the only reason this is a story about a check working rather than about a wrong
    win count.  **It is recorded because the same class of failure without the assert
    would have reported `0/14` as a finding.**
45. **The same plot-name collision, twice.**  `abl_06_task_sweep.py`'s `--tag` renames
    the REPORT but not the PLOT, and the plot it writes
    (`outputs/abl_06_task_sweep_plot.png`) is a TRACKED figure the base's section 6.5
    cites.  The 160-epoch run overwrote it, which was foreseen and handled; the
    single-cell `probe10d` run then overwrote it AGAIN, which was not -- the lesson had
    been written down and not applied.  Restored bit-identically to HEAD both times, and
    each run's own figure preserved under a tagged name.
46. **A script that wrote the file and then reported "nothing written".**  The edit
    inserting section 12.9 picked the wrong anchor, and its guard printed "reverted --
    nothing written" -- but the `open(...).write()` had already run, one line earlier,
    and the revert happened in memory afterwards.  So a wrong insertion landed in the
    document WITH a printed assurance that it had not.  Caught by the next command's
    own listing, which showed `### 12.9` at line 1001 inside section 5, and fixed by
    moving the block to the end of section 12.  **A guard that reports success without
    checking the artifact is not a guard** -- the same lesson as defects 21 and 44.
47. **The right count from the wrong quantity.**  `ext_28`'s first version aggregated the
    frontier comparison as `max` PER SEED and then averaged the margins, while `abl_06`
    takes each ARM's seed mean and then the `max` of those.  The two differ: the first
    makes `best search` systematically higher (`2d/G` 0.6982 against the published
    0.6692, `10d/F2` 0.0321 against 0.0213).  **It also happened to give 8/14 -- the same
    count as the published row -- so the number would have looked right.**  Found by
    checking the intermediate values against the published ones rather than the verdict
    count, and fixed by adopting the base's own aggregation, after which the `lam=1e-3`
    column reproduces the published cell VALUES exactly (P0, 56/56).
48. **A controlled comparison whose two arms were not initialised the same.**  `ext_32`'
    s first version ran `proto` and then `direct` without reseeding, and `HiMALND` --
    which has no seed parameter and never seeds the global RNG -- gave them DIFFERENT
    initialisations.  **The tell was in the table: layer 1 is trained identically under
    both modes, so its `R^2` should have matched, and it differed by `+0.117`.**  Fixed
    by seeding before each arm and by turning that agreement into an ASSERT, so the
    confound can no longer pass silently.  The lesson is that the base trainer's own
    irreproducibility is a hazard for anyone comparing two of its arms.
49. **A dict with mixed key types, sorted after the training was done.**  `ext_33`'s P3
    built a grid whose keys were floats from P1 and tuples from P2, and `sorted()` on it
    raised `TypeError` **after 35 minutes of training**, losing all of it and leaving the
    previous run's report in place -- so the printed table was STALE and looked fine.
    Fixed by iterating an explicit list, and by adding the cache that `abl_06`'s own
    comment already prescribes ("any interruption threw the whole sweep away -- which
    happened twice").
50. **`git check-ignore` said "ignored" and `git status` said otherwise.**  `ext_33`'
    s cache was first written to `extensions/multigraph/ext_33_cache/`, which is NOT
    ignored -- and the location also broke the repository's own documented convention
    (`.gitignore` line 63 ignores `outputs/ext_cache/` and says "The extension keeps
    its OWN model cache").  **I reported "already ignored" on the strength of
    `git check-ignore`'s exit code; the decisive test -- does `git status` show a file
    inside it? -- returned `??`, i.e. the opposite.**  Fixed by moving the cache to
    `outputs/ext_cache/ext_33` (48 cells preserved, so the re-run cost 2 seconds rather
    than 13 minutes), and recorded because the LESSON is about which check to trust:
    **an exit code is a claim and `git status` is the artefact.**
51. **A syntax error that left the PREVIOUS run's report on screen, again.**  A patch to
    `ext_36`'s verdict broke a string literal, so the script failed to compile -- and the
    console then printed the report from the earlier smoke run, which looked like a
    result.  **The same trap as defect 49, hit a second time.**  Fixed by deleting the
    report BEFORE each run, so a failure cannot masquerade as output.
52. **A verdict that contradicted its own cost table.**  `ext_36`'s first verdict said
    the node arm wins "at a fraction of the measurements" -- true under the calibration-
    style charge and FALSE under the search-style one that the same report tabulates two
    paragraphs above.  Caught by reading the finished report rather than by any assert,
    and fixed by making the sentence accounting-aware.  **A number can be right while the
    sentence around it is wrong.**
53. **An absolute tolerance below what the linear algebra can hold.**  `ext_37`'
    s assert demanded the rank-`k` gap be `< 1e-12` at `m = 2`.  It is not zero: it is
    `~1e-5`, because at `k` at or above the effective rank the model class CONTAINS the
    full-rank fit but **alternating least squares is local and does not reach it**.  The
    fix was not a looser number but a MEASUREMENT: sweeping the iteration count showed
    the gap shrinking 13x, which attributes it to the optimiser and justifies the bar.
    **The same mistake as defect 12's 1e-6 gain tolerance, in a new place.**
54. **Three self-inflicted patch failures in one experiment.**  A patch script that
    repeated a line and so ran NOTHING (its own `py_compile` passed because the target
    was untouched); a PowerShell here-string terminated early by an embedded `'@'` in
    the inserted Python; and an inserted block so convoluted -- walrus operators and
    three repeated `fit_operator` calls -- that it was replaced with a named helper
    instead.  **None reached a report, and all three cost a run.**
55. **A verdict that read expressiveness as a transfer failure, in the FIRST draft.**
    `ext_38`'s first verdict said "THE TRANSFER DOES NOT HOLD" because the anharmonic
    cell's calibration R^2 was lower than the two Gaussians'.  **That conflated two
    different questions**: the operator's transfer (which P2 measures to 2e-14) and the
    medium's ability to cover a task with 40 nodes.  Caught by re-reading the verdict
    against its own tables, and fixed by separating them -- and by matching the widths by
    EFFECTIVE RADIUS, which recovered a large part of the gap and showed the residue is
    real rather than a scaling artefact.  **The same shape as defect 52.**
56. **Three runs lost to a shape convention.**  `_build_layer` takes `centres` as
    `(N, D)` (`asarray(centres).T.reshape(D, n)`) while the layer classes it builds store
    `(D, N)`, and the new substrate was written in the other one -- so the first two runs
    died on a reshape and the third on a list-vs-array multiply.  **The convention is set
    by the callee and should have been read before the class was written**, not after two
    failures.
57. **Patch anchors that could not match, twice in one experiment, from escaping.**
    Both failed patches to `ext_39` were triple-quoted Python strings containing `\\'`
    where the target file has `\'`, so the anchor never matched and the assert fired.
    **Cost: one wasted re-run each, and the first failure was papered over because the
    FIRST patch had already succeeded and the run went ahead** -- so the report was
    regenerated with half the intended change.  The fix is to stop patching through a
    shell heredoc for anything containing escapes, and use exact-text editing instead.
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
```

All forty exit non-zero when their own published claim fails, and every one of them
runs in seconds except ext_03 (~6 min), the cold sweeps of ext_08 (~5 min), ext_14
(~40 s), ext_16 (~95 s), ext_18 (~135 s), ext_19 (~65 s), ext_20 (~110 s), ext_21 (~35 s) and ext_22 (~80 s).  ext_23 is read-only and takes seconds; ext_24 is ~9 min cold and ~10 s warm; ext_25 is ~14 min cold, ext_26 ~20 min cold, ext_27 ~10 min cold, ext_28 ~36 min cold, ext_29 seconds, ext_30 ~8 min cold, ext_31 under a second and ext_32 ~50 min cold and ext_33 ~13 min cold (cached) and ext_34 ~22 min cold and ext_35 ~14 min cold and ext_36 ~4.5 min cold and ext_37 ~20 s and ext_38 under a second and ext_39 ~2 min cold and ext_40 a few seconds.

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

**The list, ranked by what would change a conclusion, with each item's CURRENT status
stated inline.**  The heading used to read "still open", which is no longer true of it:
**items 1, 3, 4, 5, 6 and 8 have since been closed or answered** (1 by 12.21, 8 by 12.24, 3
by `ext_20`, 4 by 12.22, 5 by 12.23, 6 by 12.25) and are kept here with their outcomes rather than
deleted, because an index whose closed items silently vanish is how the same question gets
asked twice.  **What is genuinely still open is items 2 and 7**, plus the third arrangement
that item 8 now points at instead of the one it named.

1. **CLOSED by 12.21 (ext_36), and the answer is on the ACCURACY side only.**  The trade
     was measured at 5.13 and the cheap form says keep the linearity.  **The untested case
     was the TRAINED inner layer, and it is now tested: training `W` and `b` LOSES `0.0264`
     on average and beats the node control on `0 of 3` tasks** -- so the expectation that a
     trained inner layer would do better is wrong on this schedule, and the trade survives
     on accuracy.  **But a SECOND question the item did not ask decides how the answer
     should be worded:** `ext_18` charged every arm `n_train` measurements per strength,
     including a nonlinear layer the design document says can only be SEARCHED.  Under a
     search-style charge the DRAWN arm's `R^2` per measurement is three times the node
     arm's, so **"keep the linearity" is a statement about ACCURACY, not about efficiency**
     -- and which accounting the hardware pays is a modelling question 12.21 surfaces and
     does not settle.

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

4. **ANSWERED by 12.22 (ext_37): the item NAMES THE WRONG TASK, and both halves are now
   measured.**  Half A's readout is `W = C . W_g` with `C` of shape `m x k`, so the
   constraint bites only when `k < min(m, rank(W))`.  **`nd_01`'s 4D classification task is
   TWO-class, so `m = 2` -- and the measured `rank(W)` is 1, not 2**, because the centred
   one-hot target makes its channels complementary.  **So even `k = 1` is already full rank
   there, and running the test on it would have been VACUOUS.**  The repository's genuine
   `m = 5` setting -- its own cached CIFAR-10 five-class data -- is **out of reach** from
   half A's machinery: every `k` sits at `R^2 = 0`.  **So the item's conclusion is right
   and its example is wrong, and `ext_02`'s synthetic `m = 8` was the CORRECT instrument**
   -- it is what makes the constraint bite at `k = 3`.

5. **DONE by 12.23 (ext_38): there is now a THIRD MEDIUM and it is not Gaussian.**  A
   particle in an **anharmonic trap** `V(r) = kappa r^4` has equilibrium density
   `exp(-(kappa/kT) r^4)` -- a quartic exponent, and the trap's minimum and stiffness are
   this project's own movable centre and width.  Implemented as a **layer**, not a formula.
   **The operator transfers to machine precision** (centre `2.0e-14`, width `6.0e-15`), the
   qualification test places it at basis degree 4 exactly as predicted, and **its
   expressiveness is lower than the Gaussian cells'** (`+0.4609` against `+0.5761` even at
   matched effective radius).  **"Any physical system" gains a third medium and is not
   discharged by it** -- a response that is not log-linear in any finite basis stays
   excluded, and 12.16 showed the test that excludes it.

6. **ANSWERED by 12.25 (ext_40), and the item NAMED THE WRONG CHANNEL.**  The base's own
   docstring separates `mismatch` (perturbs `h` = GEOMETRY) from `drift` (multiplies the
   conductances = WEIGHT-only), and a PLACEMENT correction is a width correction -- so its
   channel is `mismatch`, while `ext_15` drifted widths under the name "drift".  **The two
   channels also have different rate semantics:** `drift` is drawn per READ and therefore
   HAS a rate; `mismatch` is drawn ONCE with no time axis and has none.  **The registering
   test `ext_15` asked for now exists** -- sweeping both signs gives `d* = 0.10` for the
   geometry tolerance, and the asymmetry confirms its one-sided explanation.  **The interval
   exists for the weight channel only**: `d*_w = 0.20` and `N* = (d*_w/sigma)^2` gives
   `1 600` / `40 000` / `160 000` reads at the shipped `high`/`medium`/`low` levels.  **A
   re-measurement interval for a PLACEMENT correction cannot be derived from this model at
   all**, and what is missing is a bias-current drift law -- a model, not a measurement.

7. **`ext_03` against the real base trainer.**  It mirrors the base's loop rather than
   calling it, because the base's closed-form hook has no place for a rank constraint.
   Wiring that hook would make the comparison authoritative instead of parallel.

8. **ANSWERED by 12.24 (ext_39): the replacement this item asks for is NOT NEEDED, and
   the premise is too strong.  NOTE FIRST that "combination" here means the READOUT-PATH
   nonlinearity the design memo raises, NOT this project's `W = C . W_graphs`, which is a
   LINEAR per-graph combination and is half A itself** -- see the box at the head of 12.24.
   The item's wording came from the memo's paragraph and pointed at a design nobody
   proposed, and what follows is that paragraph being measured.  The item presumes that a full nonlinear combination
   forfeits the closed-form operator.  **That depends on the ORDER of the operations:**
   `phi(sum_j W_j g_j)` is nonlinear in the strengths and does lose it, while
   **`sum_j W_j phi(g_j)` is nonlinear in the READINGS and LINEAR in the strengths** --
   the closed-form solve is exact to `9.99e-16` and the measurement cost is `n_train * N`,
   **identical to the linear readout's**.  `phi` is in the repository's own device library
   (Gilbert squarer, BJT exponential).  **And the measurement is a clean negative:** all
   four elementwise `phi` are WORSE than the linear readout (best `tanh`, `-0.0475`, `0 of
   3` tasks), so this nonlinearity is free but buys nothing -- which is a **stronger**
   position for the trade than a positive result would have been.
   **What is still open is the THIRD arrangement**, `phi(sum_j W_j g_j)` with its own
   calibration story; 12.21's `-0.0264` is the nearest evidence and it loses.  Item 8
   should be restated as that one arrangement.
---

## 12. An audit of the BASE version's headline, from inside the extension

> **The actionable form of this section is [`BASE_CORRECTIONS.md`](BASE_CORRECTIONS.md).**
> Section 12 is the evidence; that file is the work list -- for each claim below that the
> audit corrects, its exact line in the base, the text as it stands, the replacement, and
> which subsection here backs it.
>
> **THE LIST HAS SINCE BEEN APPLIED, at the repository owner's instruction, in three
> commits** (`2fa691e` the three retractions, `a856dc0` the missing seed in
> `training/himal_nd.py`, `e0cf9b8` the remaining eight entries).  So the sentence that
> stood here -- that no base file had been edited by this audit -- **is now false and is
> kept only as a record of what the position was when the audit was written.**  Every
> entry in `BASE_CORRECTIONS.md` is marked applied or fixed, and **one entry's replacement
> text had itself gone stale before it was used**, which is the defect that file exists to
> fix arriving inside the fix.

**Why this is here and not in `FINDINGS.md`.**  This extension does not modify the base
version -- that rule has held for all 22 experiments -- and the base version's
`FINDINGS.md` still carries its headline as measured at **3 seeds**.  The audit below
supersedes one number in it, so the correction lives here, and this section states
plainly what the base document would need in order to match.

**Why the audit was run at all.**  This extension measured that **3 seeds are not
enough**: `ext_13`'s two published conclusions on `steps` and `noise` were both
**reversed** when the count went from 3 to 12 (5.8, defect 34).  The base version's
headline -- `The calibrated arm beats the best search arm in 9/14 task x architecture
cells` -- rests on **3 seeds** (`outputs/abl_06_task_sweep_s3.txt`).  So the same test
was applied to it.

**What was run, and what was NOT touched.**  The base version's own script, unmodified:

    python experiments/abl_06_task_sweep.py --seeds 9 --tag s9

5506 s, 336 cells trained, 168 reused from the base's own cache.  It writes
`outputs/abl_06_task_sweep_s9.txt` and does not overwrite `_s3`.  **No base file was
edited to produce this.**

**The headline moves: 9/14 -> 8/14.**

| cell | 3 seeds | **9 seeds** | what happened |
|---|---|---|---|
| `2d/G` | +0.1277 | **+0.1276** | held |
| `2d/F2` | +0.2744 | **+0.2890** | held |
| `4d/G` | +0.1247 | **+0.1650** | held |
| `4d/F2` | +0.1608 | **+0.1711** | held |
| `moons/G` | +0.1366 | **+0.0904** | held, narrowed |
| `moons/F2` | +0.2285 | **+0.2494** | held |
| **`steps/F2`** | *+0.0159* (1.3x) | **+0.1515** (40.0x) | **marginal -> STRONG** |
| `shift/G` | *+0.0389* | **+0.0468** | held |
| **`shift/F2`** | *+0.0279* | **-0.0321** | **FLIPPED to a loss** |
| `10d/F2` | -0.0171 | -0.0068 | still a loss |
| `hetero/G` | -0.0318 | -0.0302 | still a loss |
| `steps/G` | -0.0744 | -0.0574 | still a loss |
| `10d/G` | -0.0490 | -0.1010 | loss widened |
| `hetero/F2` | -0.1491 | -0.1099 | still a loss |

Speedup range also moves: `1.2x to 40.0x` at 3 seeds becomes **`2.7x to 40.0x`**.

> **THE AUDIT'S OWN PREDICTION WAS HALF RIGHT, AND THE HALF THAT WAS WRONG MATTERS.**
> It named three cells as at risk -- `steps/F2` (+0.0159), `shift/F2` (+0.0279) and
> `shift/G` (+0.0389).  One flipped (`shift/F2`), one held (`shift/G`), and one
> **strengthened tenfold** (`steps/F2`, `+0.0159` -> `+0.1515`, speedup `1.3x` ->
> `40.0x`).  **So seed noise cuts BOTH ways**: the 3-seed numbers were not
> systematically optimistic, and reporting only the flip would imply a bias the data
> does not show.

> **AND THE ROBUST CORE IS CLEANER THAN THE COUNT SUGGESTS.**  Six cells hold at every
> seed count with margins of **`0.090` to `0.289`** (`2d/G`, `2d/F2`, `4d/G`, `4d/F2`,
> `moons/G`, `moons/F2`), and two more (`steps/F2`, `shift/G`) now have real margins
> too.  The headline number fell by one; the number of cells with a margin above `0.04`
> rose from seven to eight.

### 12.1 WITHDRAWN: an analysis built on a column the base forbids quoting

**What this section said in its first version.**  It quoted the ablation-F report's
`speedup` summary -- `friendly tasks (2d/4d/10d/moons): mean 31.6x`, `adversarial tasks
(steps/hetero/shift): mean 31.0x` -- observed that the two are nearly equal while the
endpoint win count is `8/14`, and concluded that "the rule is task-dependent" applies to
the endpoint rather than to the rate.  It also presented the `hetero/G` row, which shows
`40.0x` while LOSING at the endpoint, as that rule showing through.

**All of that is withdrawn.**  The base version's own section 6.7 says, in bold:

> **The `speedup` column of the ablation-F report must not be quoted.**  It is computed
> as "did the calibrated curve ever reach the best search arm's final value", so a curve
> that reaches it early and then **declines** scores a full 40x: `hetero` shows a mean
> speedup of 40.0x in that report while losing both of its cells.  **Use the
> common-frontier comparison above.**

So the `hetero/G` row is not a rule showing through -- it is **an artifact of how the
column is computed**, named as such by the base document, and this section built an
argument on it.  Recorded as **defect 41**.

**What is left, and it is not nothing.**  The endpoint comparison is allowed and is what
the headline verdict itself uses:

| cell | geometry | best search (readout) | **hybrid** | endpoint verdict |
|---|---|---|---|---|
| `hetero/G` | +0.0859 | +0.1559 | +0.1257 | **loss** -0.0302 |
| `hetero/F2` | +0.0586 | +0.2320 | +0.1221 | **loss** -0.1099 |
| `steps/G` | +0.0559 | +0.1066 | +0.0492 | **loss** -0.0574 |
| `10d/G` | -0.0168 | +0.0079 | -0.0931 | **loss** -0.1010 |
| `10d/F2` | +0.0145 | +0.0213 | +0.0145 | loss, **= geometry** |

**Four of the five losses are real endpoint losses; the fifth is a stable gate refusal.**
And in all four the hybrid arm still never ends below its own geometry arm -- that is the
guarantee, and it is untouched.

### 12.1a Section 11's item 6(b) is STALE, and the counterexample is not missing

The base version's section 11 says:

> a task where the gate *accepts* and the arm still loses has still not been found --
> **that remains the strongest missing counterexample**

**It was found, by the base version itself**, in section 6.7:

> in the fifth (`hetero`/G at 3 seeds) it accepted **one** calibration out of 120, the arm
> improved from +0.0884 to +0.1228, and still lost to `full` at +0.1546.  **That is the
> "gate accepts and the arm still loses" case section 6.11 was looking for**

and again in 6.11:

> the *stronger* claim "the calibrated arm beats the best search arm" **does have a
> counterexample** (`hetero`/G, where one calibration was accepted, improved the arm, and
> still lost to `full`)

**So this audit did not discover anything here.**  It read section 11 in isolation, took
"still not found" at face value, and went looking -- which is exactly the failure mode
**defect 38** records for THIS document's own section 11: an index of open questions that
stopped matching the work.  **The base version's section 11 has the same defect**, and
this is its second instance (the first is item 7's stale "2 seeds", recorded in 12.4).

**What the 9-seed run adds is a strengthening, not a discovery.**  At 3 seeds the
counterexample was one cell and **one accepted calibration out of 120** -- a single
event.  At 9 seeds, Part 1b's rejection counts give:

| cell | calibrations accepted (of 360) | endpoint |
|---|---|---|
| `hetero/G` | 360 - 356 = **4** | loss |
| `hetero/F2` | 360 - 356 = **4** | loss |
| `steps/G` | 360 - 357 = **3** | loss |
| `10d/G` | 360 - 357 = **3** | loss |
| `10d/F2` | 360 - 360 = **0** | pure refusal, not a counterexample |

**Four cells, not one.**  The counterexample is not a lone 1-in-120 event; it recurs in
four of the fourteen cells once the seed count is raised.  Section 6.11's conclusion
therefore holds more strongly than when it was written, and section 11's item 6(b) should
be struck rather than pursued.

### 12.2 What the "gate refusal" reading is and is not

At 3 seeds, three loss cells had `geometry` and `hybrid` **numerically identical**
(`steps/G`, `10d/G`, `hetero/F2`), which reads as "the gate refused every calibration,
so the cell cannot discriminate".  At 9 seeds that signature survives in **one** cell:
`10d/F2`, whose Part 1b row is **360 calibrations, 360 rejected** -- a complete and
**stable** refusal at both seed counts.

**So the phenomenon is real but which cells it hits depends on the seed set**, and an
earlier version of this audit reported the three-cell version as a finding.  It is
recorded as defect 40: an unsteady observation promoted to a finding.  The base
document's own phrasing -- "the five losing cells are now uninformative about the
mechanism, because the gate disables the calibration there" -- needs the same
qualification if it is ever revised: **one** cell is stably uninformative, not five.

### 12.3 The `@3000` low-budget column is noise-dominated

| cell | 3 seeds | 9 seeds |
|---|---|---|
| `steps/F2` | `-0.135 vs +0.018` | `+0.039 vs +0.005` (sign flips) |
| `10d/F2` | `-0.028 vs -0.020` | `-0.274 vs -0.022` (an order of magnitude) |

Any statement resting on that column needs its seed count stated with it.

### 12.4 What the base document would need, and why it is not changed here

1. the headline **`9/14` -> `8/14`**, and the speedup range **`1.2x-40.0x` ->
   `2.7x-40.0x`**;
2. a robustness statement: six decisive cells held from 3 to 9 seeds, one flipped, one
   strengthened;
3. the rate/endpoint distinction of 12.1;
4. **section 11 item 7's "2 seeds" is stale** -- the cited result comes from
   `abl_06_task_sweep_s3.txt`, which is `seeds=3` -- while item 3's "3 seeds" is
   correct;
5. the gate-refusal claim narrowed per 12.2, if it is revised at all.

**The base report demands item 3 itself**, in its own last lines:

> The rule is TASK-DEPENDENT.  Section 6 must state the conditions under which
> calibrating wins instead of recommending it outright.

**That is a request from an artifact to the base document**, and it is recorded here
rather than acted on, because acting on it would be the first modification to a frozen
base.  The correction is findable from this file; the base's `FINDINGS.md` does not yet
point at it.

### 12.5 The epoch axis: the prediction held, and the headline is EPOCH-CONDITIONAL

**The prediction, stated before the run.**  The base version's section 6.10 gives the
mechanism and it points one way: *"the hybrid peaking early and then flattening: the
geometry is still improving at large measurement counts, so a calibrated readout fitted
to an early geometry stops being optimal"*, and the gate rejects ~95% of calibrations,
so the hybrid effectively KEEPS an early one.  Both say the same thing:

    at higher epochs the calibrated arm should LOSE MORE OFTEN, not fewer

**The run.**  The base version's own script, unmodified, at the epoch count section
6.10's grid identifies as decisive -- and **1 seed, following 6.10's own precedent**
("the question is the SHAPE, not the precision of a cell"):

    python experiments/abl_06_task_sweep.py --epochs 160 --seeds 1 --tag e160

3844 s, 56 cells, **0 reused** (the cache is keyed by epoch count).

**THE PREDICTION HELD: `8/14` at 40 epochs becomes `6/14` at 160** -- and see 12.7:
**half of that drop was the criterion's defect, not an epoch effect.**  Under the
corrected criterion the axis is `8/14 -> 7/14`, and this subsection's `6/14` is
retained only as the published-criterion figure.

| cell | 40 ep hybrid | 40 ep verdict | **160 ep hybrid** | **160 ep verdict** | change |
|---|---|---|---|---|---|
| `2d/G` | +0.8276 | win | +0.8276 | **loss** | win -> loss |
| `4d/G` | +0.5004 | win | +0.4749 | **loss** | win -> loss |
| `steps/F2` | +0.2439 | win | +0.1992 | **loss by 0.0001** | win -> loss (a TIE) |
| `shift/G` | +0.1077 | win | +0.1885 | **loss** | win -> loss |
| `10d/F2` | +0.0145 | loss | +0.0485 | **win** | loss -> win |
| `shift/F2` | +0.0739 | loss | +0.2143 | **win** | loss -> win |
| `2d/F2` `4d/F2` `moons/G` `moons/F2` | -- | win | -- | **win** | held |
| `10d/G` `steps/G` `hetero/G` `hetero/F2` | -- | loss | -- | **loss** | held |

**And the mechanism is visible in the arms' own numbers**, which is what makes this a
confirmation rather than a coincidence.  Between 40 and 160 epochs the GEOMETRY arm
gains several times what the hybrid does, and on three cells the hybrid FALLS:

| cell | geometry arm, 40 -> 160 ep | hybrid, 40 -> 160 ep |
|---|---|---|
| `2d/G` | +0.4922 -> +0.7365 (**+0.244**) | +0.7968 -> +0.8276 (+0.031) |
| `4d/G` | +0.1968 -> +0.4102 (**+0.213**) | +0.5004 -> +0.4749 (**-0.026**) |
| `moons/G` | +0.8030 -> +0.9774 (**+0.174**) | +0.9847 -> +0.9831 (-0.002) |
| `hetero/G` | +0.0859 -> +0.2043 (**+0.118**) | +0.1257 -> +0.2042 (+0.079) |

The gate closes further as epochs rise, consistently: cells with **100% rejection** go
from **one** at 40 epochs (`10d/F2`) to **six** at 160 (`10d/G`, `10d/F2`, `steps/G`,
`steps/F2`, `hetero/G`, `hetero/F2`).  More epochs give the geometry more time to leave
the calibration behind, and the gate responds by refusing it.

> **SO THE HEADLINE IS EPOCH-CONDITIONAL, not merely seed-noisy.**  Raising the seed
> count at 40 epochs moved it by one cell (`9/14 -> 8/14`, 12.1a).  Raising the EPOCH
> count moved it by two, and moved it in the direction section 6.10 predicts.  The
> headline sentence `"the calibrated arm beats the best search arm in N/14 cells"` is
> incomplete without **both** numbers, and neither the base document nor this audit had
> it before.

**Two caveats that qualify the `6/14`, stated rather than averaged in.**

1. **`steps/F2`'s loss is a tie**, `+0.1992` against `+0.1993` -- it should not be
   counted as evidence either way.
2. **`10d/F2`'s hybrid differs from its geometry while fully rejected** -- VERIFIED,
   and the verification moved the finding somewhere better.  See 12.6.

Dropping both UNDER THE ORIGINAL CRITERION, the clear wins at 160 epochs were
**five**: `2d/F2`, `4d/F2`, `moons/G`, `moons/F2`, `shift/F2`.  **Under the
corrected criterion of 12.7 they are seven**, because both dropped cells return:
`steps/F2` and `10d/F2` were being measured against their own geometry arm.

**What this does to the earlier reading of this section.**  12.1a strengthened the base's
counterexample by showing it recurs in four cells at 9 seeds.  This subsection shows the
same 9-seed baseline was itself taken at the epoch count MOST favourable to the rule --
not by choice, but because 40 epochs is the base's default.  **The rule's advantage is
real and it is finite-time**, which is exactly what section 6.10 says about the capacity
split and what nobody had checked for the headline.

### 12.6 Verifying the `10d/F2` flag: the counter is fine, and the WIN CRITERION is not

**The flag, as written in 12.5 before this check.**  `10d/F2` at 160 epochs reported
`160 of 160` calibrations rejected, yet its hybrid (`+0.0485`) differed from its geometry
(`+0.0462`), while every other fully-refusing cell had the two equal to four decimals.
The wording was "either the rejection counter and the deployed model disagree, or
something else is going on".

**What the code does on rejection**, read rather than assumed --
`abl_common.solve_readout_closed_form`, the guard path:

    before  = [copy of l.W_gauss ...]
    n_written = _stagewise_solve(model, X[tr], y[tr], ...)   # fits into the model
    fitted  = [copy of l.W_gauss ...]
    r_new   = r2(model.forward(X[val]))
    for l, w in zip(model.layers, before):
        l.W_gauss = w                                        # RESTORE
    r_base  = r2(model.forward(X[val]))
    if not keep:  return 0                                   # nothing deployed

So a rejection restores the strengths before returning -- the rejection path IS
write-clean for `W_gauss`, and the caller's comment ("nothing was written to `tgt`")
matches the code.  **There is no counter/state mismatch.**

**What the re-run showed, at two seeds** (`--tasks 10d --archs F2 --epochs 160 --seeds 2`):

    calibrations 320   rejected 320   (100%)
    geometry -0.0830   readout -0.1518   full -0.2106   hybrid -0.0821

The `320/320` is consistent, and the arm difference is small (**`0.0009`**), so the flag
was **overstated**: it is a small numerical difference, not a bookkeeping failure.  Where
that difference comes from is a HYPOTHESIS and is labelled one: `closed_form=True` adds
one held-out forward pass per epoch for the gate, which may perturb the geometry search's
probe stream, so the two arms' geometry can differ slightly even with zero acceptances.
**Not isolated, not asserted.**

**AND THE RE-RUN EXPOSED SOMETHING THAT MATTERS MORE.**  In that cell's Part 3:

    best search -0.0830    hybrid -0.0821    YES

`best search` is `-0.0830`, which is exactly the **`geometry`** value in Part 2.  So for
this cell the strongest of the three search arms **is the geometry arm** -- and `hybrid`
IS `geometry` plus a calibration that carries a **do-no-harm guarantee**.  **When the
geometry arm is the best search arm, "hybrid beats the best search arm" is close to
guaranteed by construction rather than earned by the calibration.**

Measured across all fourteen cells:

| | 40 epochs (9 seeds) | 160 epochs (1 seed) |
|---|---|---|
| cells where `geometry` is the best search arm | **0 / 14** | **2 / 14** |
| which | -- | `10d/F2` (a "win"), `steps/F2` (hybrid `+0.1992` against its own geometry `+0.1993`) |

**So the structural weakness appears only at 160 epochs, and it makes two of that run's
verdicts uninformative** -- the two cells 12.5 had already flagged as a tie and as
inconsistent.  They now have one common and more fundamental explanation.

**What this says about the win criterion, which is the general lesson.**  `abl_06`
compares the hybrid against `max(geometry, readout, full)`.  When `readout` or `full` is
the strongest, the comparison is meaningful.  **When `geometry` is the strongest, the
comparison is against a component of the hybrid itself**, and the do-no-harm guarantee
supplies the verdict rather than the measurement.  A criterion that is meant to ask "is
calibrating worth it" should exclude the geometry arm from the comparison set, or report
those cells separately.  At 40 epochs this affects nothing (`0/14`); at 160 it affects
`2/14`.  **Recorded as a defect of the criterion, to be checked before the epoch result
is quoted.**

### 12.7 Quantifying the criterion's defect: half the epoch effect was an artifact

Section 12.6 found the defect and this measures what it costs, on the base version's own
published tables -- no new run, because both reports already carry the four arms per cell
(`ext_23`, which re-reads their Part 2 tables and asserts it parsed all 14 cells of each,
so a short parse produces an error rather than a wrong count).

| run | original criterion | **geometry EXCLUDED** | cells where geometry was the best search arm |
|---|---|---|---|
| 40 epochs, 9 seeds | **8/14** | **8/14** | **0/14** |
| 160 epochs, 1 seed | **6/14** | **7/14** | **2/14** |

| the epoch axis | 40 ep | 160 ep | change |
|---|---|---|---|
| original criterion | 8/14 | 6/14 | **-2** |
| **corrected** | 8/14 | **7/14** | **-1** |

> **SO HALF OF SECTION 12.5'S EPOCH EFFECT WAS THE CRITERION, NOT THE EPOCHS.**  The
> drop is real but it is **one** cell, not two, and the honest statement of the epoch
> axis is **`8/14 -> 7/14`**.  This does not touch the mechanism -- the geometry arm still
> gains several times what the hybrid does between 40 and 160 epochs -- it corrects the
> SIZE of the effect.

**Exactly one cell changes verdict**, and the change is the whole point:

    steps/F2   original : best search was GEOMETRY at +0.1993, hybrid +0.1992 -> "loss"
               corrected: best search is  FULL   at +0.1513, hybrid +0.1992 -> WIN +0.0479

**And one cell keeps its verdict while changing its basis entirely:**

   10d/F2   original : best search was GEOMETRY at +0.0462 -> "win" by +0.0023
            corrected: best search is  FULL   at -0.0210 -> win by +0.0695

So `10d/F2`'s win was **real but 30x understated**, and `steps/F2`'s "loss" was an artifact
of being measured against a component of itself.  **Both, plus the two cells 12.6
identified, are now accounted for by one cause.**

**WHY THE DEFECT IS EPOCH-DEPENDENT.**  At 40 epochs no cell has `geometry` as its best
search arm, so the correction is free there (`0/14`).  At 160 epochs the geometry arm has
improved enough to dominate in two cells -- which is the same phenomenon 12.5 documents
as the mechanism, arriving from the other side.  **So the criterion defect and the epoch
effect share a cause: as the geometry improves, it becomes both the strongest search arm
and the thing the hybrid is built on.**

**THE GUARANTEE IS UNTOUCHED.**  Worst hybrid `R^2` is still never below its own geometry
arm, which is exactly what the do-no-harm guarantee claims.  What 12.6 and this
subsection correct is only the STRONGER, empirical claim about beating the competing
methods.

**THE STATED LIMIT, because the correction is a choice and not a measurement.**
Excluding `geometry` is the criterion THIS audit argues for; the base version did not
endorse it.  An equally defensible alternative keeps geometry in the set and reports the
cells where it dominates separately.  Both fix the same problem.  The choice is about
what the sentence "the calibrated arm beats the best search arm" is meant to claim, and
that is a decision rather than a result -- so it is flagged rather than settled.

### 12.8 Regenerating `train_01` and `train_02`: one vacuous, one meaningful

**What was demanded.**  The base version's section 9 records that
`experiments/train_01_grouped_fd.py` and `train_02_grouped_spsa.py` grouped parameters
with the `p*D` convention while the vector layout is `d*N + p`, that both were **fixed**,
and that "**their previously reported numbers must be regenerated**".  The four
front-page documents still describe `train_01` as proving inter-group parameter
independence, with the README status box flagging it as un-regenerated.

**What was done.**  The old reports were backed up first (`outputs/_pre_regen/`, dated
`2026-07-30`, i.e. before the fix), then both scripts were run **unmodified** -- no base
file was edited to produce any of this.  `train_01`: exit 0, 341 s.  `train_02`: exit 0,
335 s, stderr empty in both.

**`train_01` -- THE REGENERATION IS VACUOUS, and the fix left a second reason.**

Every K gives identical numbers at every epoch:

    epoch   5: R2 = ['0.2519','0.2519','0.2519','0.2519']   max_diff = 0.0000
    epoch 200: R2 = ['0.9888','0.9888','0.9888','0.9888']   max_diff = 0.0000
    Final R2_dense:  K=1 0.9888 | K=2 0.9888 | K=4 0.9888 | K=8 0.9888
    OVERALL VERDICT: PASS   Particle independence confirmed

> **THE IDENTITY IS GUARANTEED BY THE LOOP NESTING, NOT MEASURED.**  The grouping is by
> CONSECUTIVE particle blocks (`particles_per_group = N // K`, then
> `range(g*ppg, (g+1)*ppg)`), so for every K the flattened parameter visit order is the
> same -- particles `0..N-1`, each updated exactly once per epoch, in place.  `K` changes
> only WHERE `set_params_from_vector` is called, which does not change the arithmetic.
> **`max_diff = 0.0000` therefore cannot come out any other way.**

So the PASS verdict confirms that splitting one unchanged update sequence into contiguous
sub-passes changes nothing -- a statement about the code's loop nesting, not about
particle independence.  **And it misses the hypothesis it states**: the script's own
framing is "K circuit blocks **independently** FD-train their own particles", i.e.
PARALLEL, meaning simultaneous gradients from a shared state -- not sequential.

**Section 7 already says this script does not test particles**, for the index reason:
"its group indices use the `p*D` convention ... Its 'groups' mix the same dimension of
different particles.  The claim therefore needs the measurement above, not that script."
**This regeneration adds a SECOND, independent reason that survives the fix**: correcting
the index convention removed one thing that made it vacuous and left another.  And
section 7 already measures what the hypothesis is actually about -- the sequential vs
parallel (Gauss-Seidel vs Jacobi) price, `+0.0009 / +0.0105 / +0.0003`.

> **SO THE CORRECT ACTION IS NOT REGENERATION BUT SUPERSESSION.**  Re-running `train_01`
> closes nothing; its claim is properly measured in section 7 (`abl_02_locality.txt`,
> coupling ratio `0.000e+00` uncoupled against `0.063-0.261` under PU), and the front-page
> line describing it as proving independence should point there instead.

**`train_02` -- THE REGENERATION IS MEANINGFUL, and the claim survives it.**  SPSA uses 2
forward passes per epoch over ALL parameters, so grouping genuinely changes which
parameters are perturbed together.  The results DO vary with K:

| K | R2_dense | drop vs FD | converged at | epochs | speedup |
|---|---|---|---|---|---|
| 1 (exact per-param FD) | 0.9888 | +0.00% | 25 600 | 10 | 1.0x |
| **2** | **0.9990** | **-1.03%** | **1 200** | 267 | **21.3x** |
| 4 | 0.9986 | -0.98% | 2 400 | 134 | 10.7x |
| 8 | 0.9981 | -0.94% | 3 840 | 67 | 6.7x |

    fwd=   500:  K=1 0.6137  K=2 0.8808  K=4 0.8202  K=8 0.6075   max_diff 0.2733
    fwd= 64000:  K=1 0.9888  K=2 0.9990  K=4 0.9986  K=8 0.9981   max_diff 0.0102

**Verdict PASS, with NEGATIVE drops** -- grouped SPSA is *better* than exact per-parameter
FD at the same forward budget, and at `K=2` it converges 21.3x sooner.  The mechanism is
plausible but not measured here: SPSA's stochastic perturbation over a group is a
different (and apparently better-regularised) step than an exact per-coordinate FD step
at this `eps`.  **Stated as a hypothesis, not a finding.**

**THE PAIR IS THE POINT.**  Two scripts written for the same purpose, both fixed for the
same index bug, and regeneration settles them in **opposite** ways: one becomes
meaningful evidence, the other becomes provably vacuous.  **A demand to "regenerate" is
not the same as a demand to "re-test", and only the second would have caught this.**

### 12.9 The particle-budget tail: the comparison window scales away (ext_24)

**The open item, in the base's own words.**  Section 11 closes the scaling law as done
with one tail left:

> "Still to do: check whether the hybrid advantage at 3 000 measurements holds for a
> larger particle budget."

The `@3000` column is a LOW-BUDGET comparison -- the calibrated arm against the best
search arm long before the common frontier.

**The freeze, and how this respected it.**  All five of the base's architectures hold the
total particle count at **52**, so a larger budget needs new entries in
`abl_common.ARCHS` -- inside a base file.  **The file was not edited**: the keys are added
to the `ARCHS` dict AT RUNTIME, no existing key is touched, and the file stays
byte-identical.  Single-layer matrix-free families at **N = 52, 104, 208** (the base's own
total, then 2x and 4x), one task, one seed, the base's own arms and its own
`effective_budget` rule.

**P0 -- FAITHFULNESS, and it passed completely.**  A single-seed run cannot be compared
with the base's published means, so the control does what can be done: it loads the base's
caches for **all nine seeds** of (4d, F2, 40 epochs), averages them at the frontier with
this script's own arithmetic, and compares against the published `_s9` row:

| arm | this script (9-seed mean) | published `_s9` | agree |
|---|---|---|---|
| geometry | +0.1173 | +0.1173 | YES |
| weights | +0.4203 | +0.4203 | YES |
| full | +0.3166 | +0.3166 | YES |
| hybrid | +0.5914 | +0.5914 | YES |

**4/4 reproduced**, so the frontier arithmetic and the low-budget comparison here are the
base's own and P1 can be read as evidence rather than as a reimplementation.

**P1 -- AND THE QUESTION AS POSED CANNOT BE ANSWERED BY THIS DESIGN.**

| arch | N | frontier | best search | hybrid | win | @3000 hybrid | @3000 best | first m |
|---|---|---|---|---|---|---|---|---|
| `S52` | 52 | 48 000 | +0.3839 | **+0.5884** | YES | **+0.4665** | +0.0605 | 1 200 |
| `S104` | 104 | 96 000 | +0.4901 | **-0.3294** | no | **-0.5194** | +0.0399 | 2 400 |
| `S208` | 208 | 200 000 | +0.5219 | +0.3004 | no | **undefined** | undefined | **5 000** |

> **(1) THE COMPARISON WINDOW SCALES AWAY.**  The hybrid curve's first recorded point is
> **twice the architecture's per-epoch budget**, and that budget grows with N -- `1200` at
> N=52, `2400` at N=104, `5000` at N=208.  At N=208 the first point is already past the
> `3000` grid point, so the calibrated arm **has no measurement there** and the column is
> **UNDEFINED: not bad, absent**.  **The window disappears before the advantage does.**

> **(2) WHERE IT IS STILL DEFINED, THE ADVANTAGE IS GONE.**  At N=104 the margin is
> **-0.5593** against **+0.4060** at N=52.  **But the mechanism is NOT established**, and
> the numbers argue against the obvious one: `104/256 = 0.41x` readout-params-per-sample
> sits well inside the base's own `0.20x`-safe to `2.19x`-failing band, and **N=208 does
> BETTER than N=104 while being larger still**.  So this is not a clean monotone capacity
> story, and the collapse is reported unexplained.

> **(3) THE CONFOUND FLAGGED IN ADVANCE IS REAL AND NOW PRICED.**  N changes the
> per-epoch budget (`600 -> 1200 -> 2500`) as well as the geometry, so this design varies
> three things at once.  **A budget-matched variant -- a FIXED per-epoch budget across N
> -- is what would answer the question**, and it is not run here.

**SCOPE, stated because the family is not the base's.**  The scaled architectures are
**single-layer** while the base's own families are three-layer at the same total; the
comparison WITHIN the family is like for like (`S52 -> S104 -> S208` differ only in N),
but `S52` is not the base's `F2`, and P0 reports them separately rather than treating them
as the same thing.

> **SO SECTION 11's TAIL SHOULD BE RESTATED, NOT CLOSED.**  What the measurement shows is
> that the low-budget comparison has a **window whose position depends on the
> architecture**, so "does the advantage at 3 000 measurements hold at a larger particle
> budget" is a question that must name its budget rule before it can be asked.  That is a
> sharper answer than either yes or no, and it is the third time in this audit that a
> question turned out to need restating rather than answering (12.5, 12.7, here).

### 12.10 The budget-matched variant: the answer is NO, and it refutes 12.9's excuse

**Why it exists.**  12.9 could not answer the base's section 11 tail because
`effective_budget` derives the per-epoch budget from the architecture, so raising N raised
the budget too and moved the hybrid curve's FIRST point past the `3000` grid point -- the
comparison became **undefined rather than bad**, and 12.9 priced a confound it did not
remove.  This is that confound removed: **an explicit fixed budget, varying only N.**

**The fix works, and the window is now architecture-independent:**

| budget | first curve point | `@3000` |
|---|---|---|
| `600` | **1200 for every N** | **DEFINED for all N** |
| `2400` | 4800 for every N | uniformly above the grid point -- consistently undefined |

> **CORRECTED BY 12.12: THIS SUBSECTION'S ANSWER IS AN ARTEFACT OF ONE
> HYPERPARAMETER.**  Everything below was measured at the base's default ridge scale
> `closed_form_lam=1e-3`, and **that value is the only one on a five-point grid that
> produces the collapse.**  At `lam="auto"` -- which the base already supports -- the
> low-budget margin is **positive at every N and RISES with N** (`+0.3953 -> +0.5079
> -> +0.6000`).  The subsection is kept because its method is what found this, and its
> conclusion should be read as conditional on the ridge scale.

**AND THE ANSWER IS NEGATIVE, AT `lam=1e-3`.**  At a fixed budget of `600`, with **all three cells
clearing the learning floor** (`3/3`, so the search arms really are training):

| arch | N | sweeps/epoch | best search | hybrid | @3000 hybrid | @3000 best | **margin** |
|---|---|---|---|---|---|---|---|
| `S52` | 52 | 3.85 | +0.3839 | **+0.5884** | **+0.4665** | +0.0605 | **+0.4060** |
| `S104` | 104 | 1.92 | +0.3776 | **-0.3625** | **-0.5424** | +0.0335 | **-0.5759** |
| `S208` | 208 | 0.96 | +0.2560 | +0.0850 | -0.0233 | +0.0007 | **-0.0240** |

**The margin goes from `+0.4060` at N=52 to `-0.0240` at N=208: the advantage inverts.**
At the frontier the same thing happens at both budgets, and the floor check is what makes
this a real negative rather than a starved-competitor artefact -- the search arms reach
`+0.38 / +0.38 / +0.26`, so the hybrid is losing to opponents that ARE learning.

> **AND IT REFUTES 12.9's OWN HYPOTHESIS.**  12.9 suggested the N=104 collapse might be an
> artefact of the budget moving with N.  **Held fixed, it reproduces** (`-0.5424` here
> against `-0.5194` there).  So the collapse is real, and the design built to remove a
> confound removed its own motive for existing -- which is what it was for.  **The
> hypothesis was wrong and the measurement says so.**

**WHAT REMAINS UNEXPLAINED, and it is the interesting part.**  The failure is **not
monotone**: N=104 is catastrophic (`-0.36`) while N=208, larger still, is merely poor
(`+0.085`).  And `104/256 = 0.41x` readout-params-per-sample sits well **inside** the base's
own `0.20x`-safe to `2.19x`-failing band, so the base's own controlling quantity does not
predict it.  **The mechanism is not established here.**

**SCOPE, stated with the conclusion.**  Single-layer matrix-free family, one task, one
seed, 40 epochs.  The scaled family is **not** the base's own, whose families are
three-layer at the same total particle count.  So this gives a **negative answer for THIS
family and makes the question well-posed for the first time** -- the base version's own
tail still needs the same budget-matched run with its own architectures.

**SO SECTION 11's TAIL IS ANSWERED FOR THIS FAMILY AND RESTATED FOR THE BASE'S.**  The
question is no longer ill-posed -- it has a design -- and the answer it gives here is that
the low-budget advantage does not survive four times the particles.

### 12.11 The same design on the BASE's own family: two families agree, and an anchor verifies the rig

12.10 answered section 11's tail negatively for a SINGLE-LAYER family and said, in its own
scope note, that the base's families are three-layer at the same total.  This runs the same
fixed-budget design on **the base's own shape** -- the `F2` configuration, three-layer,
front-loaded, matrix-free -- scaled by keeping its proportions:
`T52 = (40,8,4)` (identical to the base's `F2`, asserted at install time),
`T104 = (80,16,8)`, `T208 = (160,32,16)`.

| budget | arch | N | params | sweeps/ep | best search | hybrid | @3000 hybrid | @3000 best | **margin** | floor |
|---|---|---|---|---|---|---|---|---|---|---|
| 600 | `T52` | 52 | 276 | 4.35 | +0.3837 | **+0.5891** | **+0.4676** | +0.0033 | **+0.4644** | ok |
| 600 | `T104` | 104 | 552 | 2.17 | +0.4987 | +0.0351 | -0.0916 | +0.0153 | **-0.1069** | ok |
| 600 | `T208` | 208 | 1104 | 1.09 | +0.1489 | +0.1047 | -0.0587 | -0.0080 | **-0.0508** | ok |
| 2400 | `T52` | 52 | 276 | 17.39 | +0.4895 | +0.5867 | *undefined* | *undefined* | -- | ok |
| 2400 | `T104` | 104 | 552 | 8.70 | +0.5800 | +0.0936 | *undefined* | *undefined* | -- | ok |
| 2400 | `T208` | 208 | 1104 | 4.35 | +0.1857 | +0.1864 | *undefined* | *undefined* | -- | ok |

> **CORRECTED BY 12.12, in the same way as 12.10.**  Both families were run at the
> base's default `closed_form_lam=1e-3`; at `lam="auto"` this family's margin is
> `+0.4496 / +0.5401 / +0.4957` -- **positive at every N**.  What 12.11 actually shows
> is that the collapse is reproducible across two families AT THAT RIDGE SCALE, which
> is what made it look like a property of the size rather than of a constant.

> **TWO INDEPENDENT FAMILIES GIVE THE SAME ANSWER, AT `lam=1e-3`: the low-budget
> advantage does NOT survive a larger particle budget.**  The margin goes `+0.4644 -> -0.0508` on the base's own
> shape, against `+0.4060 -> -0.0240` on the single-layer family, and **`3/3` cells clear the
> learning floor at budget 600** -- so the hybrid is losing to search arms that really are
> training.  Section 11's tail is therefore negatively answered for the base's own SHAPE, not
> merely for a convenient one.

**AND THE RIG IS VERIFIED BY AN ANCHOR, which also settles what actually differed.**  `T52`
IS the base's `F2`, and at (4d, seed 0) the base's own `effective_budget` returns **600** --
the same value this design fixes.  So the two cells coincide, and they do, bit for bit:

    ext_24, the base's F2 under effective_budget : frontier 48000, best +0.3837,
                                                   hybrid +0.5891, @3000 +0.4676 vs +0.0033
    ext_26, T52 at a FIXED budget of 600         : frontier 48000, best +0.3837,
                                                   hybrid +0.5891, @3000 +0.4676 vs +0.0033

**Identical.**  Two things follow: the cell runner, the frontier arithmetic and the
low-budget comparison are consistent across all three scripts; and **what ext_24 varied was
the SCALING rule for the budget (600 -> 1200 -> 2500), not the baseline value** -- the base's
own budget for that cell was already the one 12.10 and 12.11 fix.

**THE NON-MONOTONICITY REPEATS, in both families.**  At budget 600 the intermediate size is
the WORSE one:

| family | N=104 | N=208 |
|---|---|---|
| single-layer (12.10) | **-0.5759** | -0.0240 |
| the base's shape (here) | **-0.1069** | -0.0508 |

**Two families agreeing makes this a pattern rather than a one-off curiosity**, and it is
still unexplained -- `104/256 = 0.41x` readout-params-per-sample remains inside the base's
own `0.20x`-safe to `2.19x`-failing band, so the base's controlling quantity does not predict
it.  **The mechanism is the open question this line leaves behind, and it is now a
well-posed one.**

**AND ONE THING NOT TO READ AS A WIN.**  At budget 2400, `T208` "wins" at the frontier by
`+0.0007` -- **a tie**, on one seed, and not a result.

**SCOPE.**  One task, one seed, 40 epochs.  The family is the base's own SHAPE scaled, not
its own SIZES, which the base holds fixed at 52.  A multi-seed version would be needed
before any of this is quoted as the base's result -- and the same is true of 12.10.

### 12.12 The mechanism, found: it is ONE hyperparameter (ext_27)

**The pattern to explain.**  12.10 and 12.11 found the low-budget margin non-monotone in
N -- the intermediate size worst -- **in two independent families**, while
`104/256 = 0.41x` readout-params-per-sample sits inside the base's own safe band.  So the
base's controlling quantity did not predict it.

**P1 -- FOUR CANDIDATE DIAGNOSTICS, AND NONE OF THEM TRACKS THE FAILURE.**

| family | N | final R^2 | cond mean | calibs accepted | mean reject gap |
|---|---|---|---|---|---|
| single-layer | 52 | +0.5884 | 8.68e+03 | 1/40 | -0.0252 |
| single-layer | **104** | **-0.3625** | 9.00e+03 | 1/40 | -0.1715 |
| single-layer | 208 | +0.0850 | 1.41e+04 | 0/40 | -2.3974 |
| base shape | 52 | +0.5891 | 7.50e+03 | 1/40 | -0.0510 |
| base shape | **104** | **+0.0351** | 9.98e+03 | 1/40 | -0.1598 |
| base shape | 208 | +0.1047 | 1.88e+04 | 0/40 | -0.9233 |

The failure's shape is **V** in both families.  The condition number is **monotone**, the
acceptance count is **1/1/0**, the acceptance rate likewise, the reject gap **monotone
more negative**.  **None is V-shaped, so none of the four explains it** -- and the gate is
nearly identical at N=52 and N=104 (one acceptance each) while the outcomes are `+0.59` and
`-0.36`, so what differs is the EFFECT of that one accepted calibration, not the gate's
strictness.

**P2 and P3 -- AND THE RIDGE SCALE IS THE WHOLE OF IT.**  P2 swept the frontier R^2; P3
then swept **the quantity 12.10 and 12.11 actually reported, the `@3000` margin** (the
search arms do not use `closed_form_lam`, so their low-budget values are held fixed):

| family | `lam` | margin N=52 | margin N=104 | margin N=208 |
|---|---|---|---|---|
| single-layer | **`auto`** | +0.3953 | **+0.5079** | **+0.6000** |
| single-layer | **1e-3 (the base's)** | +0.4060 | **-0.5759** | **-0.0240** |
| single-layer | 1e-2 | +0.4409 | +0.1763 | -0.0240 |
| single-layer | 1e-1 | +0.4846 | +0.5079 | +0.3558 |
| single-layer | 1.0 | +0.3526 | +0.4974 | +0.6000 |
| base shape | **`auto`** | +0.4496 | **+0.5401** | **+0.4957** |
| base shape | **1e-3 (the base's)** | +0.4644 | **-0.1069** | **-0.0508** |
| base shape | 1e-2 | +0.4915 | +0.3271 | -0.2420 |
| base shape | 1e-1 | +0.4993 | +0.5401 | +0.4957 |
| base shape | 1.0 | +0.3925 | +0.5236 | +0.5947 |

> **THE ADVANTAGE SURVIVES A LARGER PARTICLE BUDGET.  12.10 AND 12.11 ARE OVERTURNED.**
> At `lam="auto"` the margin is positive at every N in both families and **rises with N**
> in the single-layer one (`+0.3953 -> +0.5079 -> +0.6000`).  **`1e-3` -- the base's own
> default -- is the only value on this five-point grid that produces the collapse.**

**WHY THIS TOOK THREE SUBSECTIONS TO FIND.**  12.9 blamed the budget rule, and was wrong.
12.10 removed that confound, got a negative answer and refuted 12.9's excuse -- and was
itself conditional on a constant nobody had varied.  12.11 reproduced it in a second family,
which is exactly what made it look like a property of the SIZE rather than of a
hyperparameter.  **A result reproduced across two architectures and two families still
turned out to be one number.**

> **AND THE BASE'S ENTIRE HEADLINE WAS MEASURED AT THAT NUMBER.**  `abl_06_task_sweep.py`
> never passes `closed_form_lam`, so every cell of the base's task sweep -- and therefore
> `9/14`, `8/14` and `6/14` alike -- was measured at a FIXED `1e-3`.  **This subsection
> establishes that the ridge scale decides a low-budget comparison in the architectures
> tested here; it does NOT establish that the base's own headline changes under `auto`.**
> That is a separate measurement and it is the obvious next one.

**SCOPE.**  One task, one seed, budget 600, 40 epochs, and a five-point `lam` grid of which
one value is `"auto"`.  A finer grid could find other rescues, and a different task or seed
could behave differently -- so "the ridge scale is the mechanism" is a statement about
these cells, stated as such.

### 12.13 The base's headline is NOT an artefact of the ridge scale (ext_28)

12.12 found that the ridge scale decides a low-budget comparison and that
`abl_06_task_sweep.py` **never passes `closed_form_lam`** -- so the base's whole task sweep,
and `9/14` / `8/14` / `6/14` with it, was measured at a fixed `1e-3`.  This asks whether the
headline moves under `auto`.  **It does not.**

**The run.**  `closed_form_lam` reaches only the HYBRID arm; the other three never call
`solve_readout_closed_form`, so their curves are independent of it and are **reused from the
base's own cache**.  Only the hybrid is re-trained: 14 cells x 9 seeds = 126 runs.
**Nothing is written to the base's cache** -- its key carries no lambda, so an `auto` cell
saved there would overwrite the base's `1e-3` cell for the same
`(task, arch, arm, seed)` and destroy the evidence its published numbers rest on.

**P0 -- AND THIS TIME THE COMPARISON IS EXACT, NOT APPROXIMATE.**  All four arms of all 14
cells, read back through this script's arithmetic as 9-seed means, reproduce the published
`_s9` row: **56/56**.  And the consequence is stronger than a matching count: because every
arm mean equals the published one, the `lam=1e-3` column below **IS** the base's published
verdicts, cell by cell.

**P1 -- THE VERDICT.**

| criterion | `lam=1e-3` | `lam=auto` |
|---|---|---|
| the base's `max(geometry, weights, full)` | **8/14** | **8/14** |
| the 12.7-corrected `max(weights, full)` | **8/14** | **8/14** |

**Cells whose verdict differs between the two ridge scales: 0.**

The values DO move, and by a lot in places -- `2d/F2` `+0.8541 -> +0.9634`, `hetero/F2`
`+0.1221 -> +0.2053`, and against that `10d/F2` `+0.0145 -> -0.0367`, `4d/G`
`+0.5004 -> +0.4833` -- but **the verdicts do not**, because except for the four losing
cells the margins are far wider than the ridge scale can move them.

**AND IT CONFIRMS 12.7 INDEPENDENTLY.**  The corrected criterion's margins are identical to
the base criterion's in **all 14 cells**, which is exactly what 12.7 measured when it found
that at 40 epochs **no cell has `geometry` as its best search arm** (`0/14`).  Two unrelated
paths, one fact.

> **SO THE SCOPE OF 12.12 IS NOW DRAWN.**  The ridge scale decides the **low-budget
> comparison** -- the `@3000` margin that 12.10, 12.11 and 12.12 all reported on -- and it
> does **NOT** decide the **frontier verdict** that the base's headline counts.  **The two
> findings are independent, and each holds in its own regime.**  12.12's closing caveat
> ("does NOT establish that the base's own headline changes under `auto`") is now settled
> in the negative, by measurement.

**WHAT THIS DOES NOT SAY.**  It does not say `auto` is worse or better as a default: it
helps some cells and hurts others at the frontier, and its low-budget effect is large and
positive (12.12).  If anything the pair of results is an argument for choosing the scale
rather than fixing it -- but that is a recommendation about a default, not a measurement,
and it is flagged as one.

**SCOPE.**  The base's seven tasks and two architectures, the diffusion substrate, the
base's own `effective_budget`, 40 epochs, 9 seeds.  The `lam` grid behind `auto` is
`ext_27`'s five points, of which `auto` was the one that worked in both families there.

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
