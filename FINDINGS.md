# FINDINGS

What this project establishes, how it was measured, and what it does **not**
establish.  Every number below comes from a script in this repository and is
tagged with the artifact that produces it.  Where a measurement contradicted a
claim, the claim was replaced by the measured one rather than annotated, and
section 9 collects the caveats a reader needs in order to use these numbers
correctly.

Written after the experiments in `experiments/abl_*` and
`experiments/sub_01_diffusion_verify.py`; all numbers are reproducible with the
commands in section 0.

---

## 0. Reproduce

```bash
# substrate verification (kernel vs PDE solve, superposition, cost model)  ~4 s
python experiments/sub_01_diffusion_verify.py

# ablation A: THE HEADLINE RUN (3 seeds, 80 epochs, scaling curve)      ~30 min
python experiments/abl_01_capacity_split.py --archs G,F,F2 \
    --arms geometry,weights,hybrid,hybrid_once,full \
    --seeds 3 --epochs 80 --no-attribute --budget-mode equal_probes

# the same in ~10 min if you only want the ordering, not the seed spread
python experiments/abl_01_capacity_split.py --archs G,F,F2 \
    --arms geometry,weights,hybrid,hybrid_once,full \
    --seeds 2 --epochs 40 --no-attribute --budget-mode equal_probes

# equal coverage instead of equal cost                                  ~3 min
python experiments/abl_01_capacity_split.py --archs G,F,F2 \
    --arms geometry,weights,hybrid,hybrid_once,full \
    --seeds 2 --epochs 40 --no-attribute --budget-mode equal_sweeps

# on the second physical substrate                                      ~5 min
python experiments/abl_01_capacity_split.py --archs F2 \
    --arms geometry,weights,hybrid,hybrid_once,full \
    --seeds 2 --epochs 40 --no-attribute --budget-mode equal_probes \
    --substrate diffusion

# ablation B: particle independence and the price of parallel updates
python experiments/abl_02_locality.py --seeds 1 --epochs 20

# ablation C: which non-ideality damages which parameters               ~2 h
python experiments/abl_03_mismatch_sensitivity.py --substrate diffusion \
    --seeds 3 --epochs 60 --reps 3

# ablation D: calibration vs device non-idealities                     ~25 min
python experiments/abl_04_calibration_robustness.py --seeds 2 --epochs 40 \
    --calib-every 10

# ablation E: calibration vs a WRONG digital twin                      ~30 min
python experiments/abl_05_twin_error.py --seeds 2 --epochs 40 --calib-every 10

# ablation F: the task sweep, and where the rule stops working          ~100 min
# CHECKPOINTED per (task, arch, arm, seed): an interrupted run resumes from
# outputs/abl_cache instead of restarting.
python experiments/abl_06_task_sweep.py --seeds 3 --epochs 40
# NOTE: quote task lists in PowerShell -- `10d` is a numeric literal there, so
# `--tasks 10d,4d` arrives as the numbers 10 and 4 and fails with
# "unknown task kind: 10".  Use --tasks '10d,4d'.

# ablation G: convergence grid, calibration schedule, gate sample size   ~35 min
python experiments/abl_07_convergence.py --seeds 2

# share axis: all five architectures under one protocol (sections 5 and 6.8) ~55 min
python experiments/abl_01_capacity_split.py --archs G,M,W,F,F2 \
    --arms geometry,weights,hybrid,full --seeds 2 --epochs 40 --no-attribute \
    --budget-mode equal_probes --substrate diffusion --tag shareaxis
```

---

## 1. Why there are two measurement protocols

On a physical substrate every training step costs measurements, so "how well does
parameter set X learn" is only meaningful at a stated cost.  Two protocols are
reported because they answer different questions:

| protocol | rule | what it favours |
|---|---|---|
| `equal_probes` | every arm probes the **same number of coordinates per epoch** | the arm with **fewer** parameters (it completes more sweeps of its own space) |
| `equal_sweeps` | every arm completes **one full sweep of its own parameters per epoch** | equal coverage; cost differs |

In a contracting RBF architecture the geometry arm always has more parameters
than the readout arm, because geometry costs `D+1` per particle while a scalar
readout costs `1`.  `equal_probes` is therefore **biased against geometry by
construction**, and reporting only that protocol would overstate the case against
it.  Both are always reported.  An arm that wins one and loses the other is more
efficient per measurement, not a better representation.

---

## 2. Substrate agnosticism is operational, not rhetorical

Two substrates are implemented behind one interface:

* `simulator/layer_nd.py` — the analog-circuit chain (differential pair, Gilbert
  squarer, Kirchhoff sum, BJT exponential, current mirror, memristor MVM)
* `substrate_diffusion/layer.py` — a diffusion cell, where the kernel is the
  Green's function of `u_t = D lap(u)` rather than an engineered circuit

`PhysicalRBFLayerND` and `DiffusionRBFLayerND` share the parameter vector layout
`[centers (D*N, C-order) | h (N) | W_gauss (V*N) | W_act (V*N)]`, so a trained
parameter vector moves between media unchanged.

| check | result | artifact |
|---|---|---|
| parameter layouts identical, element-wise | YES (81 = 81) | `sub_01` Part 0 |
| `set_params_from_vector` round trip identical | YES | `sub_01` Part 0 |
| circuit vs diffusion, same parameters | g:pu=1.7e-08 a:pu=1.4e-08 g:none=3.0e-08 a:none=2.2e-08 | `abl_common.selftest_cross_substrate` |
| ablation A on circuit vs diffusion (same settings) | R^2 agrees to 4 decimals | `abl_01` reports |

Interpretation: the float32 digital reference sets a comparison floor near 6e-08,
so all of these numbers are **rounding-limited, not physics-limited**.  The
circuit substrate sits ~13x above that floor (8.3e-07) because it performs more
float32 arithmetic; the diffusion substrate sits on it (6.5e-08).

Consequence for the ablations: any result obtained on the circuit simulator is
**not** an artifact of that simulator, because the same result is obtained on an
unrelated medium.  This is what licenses the single-substrate runs below.

---

## 3. The kernel is verified as physics, not as an implementation detail

The diffusion substrate's kernel is the closed-form free-space Green's function.
It is checked against an independent finite-difference solution of the diffusion
equation — nothing is compared against itself.

| comparison | max relative error | artifact |
|---|---|---|
| finite-difference PDE solve vs analytic solution | 1.62e-05 | `sub_01` Part 1 |
| substrate propagator (numerical convolution) vs PDE solve | 1.62e-05 | `sub_01` Part 1 |
| substrate propagator vs analytic solution | 4.11e-14 | `sub_01` Part 1 |
| superposition error, 12 signed sources | 2.86e-16 | `sub_01` Part 2 |
| vectorised sum vs literal one-source-at-a-time injection | 0.00e+00 | `sub_01` Part 2 |

Line 2 is the load-bearing one: the substrate's kernel is as accurate as the PDE
solve itself.  Superposition is exact because the PDE is linear, so "activate
then combine" is a physical law on this medium instead of a circuit that
approximates a sum.

**A thresholded medium would fail exactly test 4.**  That is the check to run
before adopting any new substrate, and it is why an excitable/spiking medium
(such as FitzHugh-Nagumo) is *not* a drop-in replacement: above threshold its
`-u^3/3` term breaks linear superposition.

---

## 4. What the readout matrix actually is

> **Notation, because this document overloads `D`.**  In `D+1` and `D*N` (sections 2,
> 5, 11) `D` is the **input dimension**.  In the two physics formulas on this page and
> in section 6.3 — `g_j = (4 pi D t_j)^(dim/2)` and `t_j = h_j^2/(4kD)` — `D` is the
> **diffusion coefficient** (the code's `sub.D`), `k` is the **kernel sharpness** (the
> code's `k` in `pulse_time(h_phys, k, sub.D)`), and `dim` is the medium's spatial
> dimension (the code's `sub.dim`).  Nothing in the results changes; the two uses of
> `D` are simply different quantities, and a reader who conflates them will mis-derive
> the boundary and timing rules of section 8.  When writing code, prefer the attribute
> names `sub.D` and `sub.dim` over any single-letter alias.

`W_ij` is the **excitation strength of particle *j* as seen by output channel
*i***.  It is not a mixing operator over the input space; it is a list of
per-source amplitudes.  On the diffusion substrate the injected pulse energy is
literally `q_ij = W_ij * g_j`, with `g_j = (4 pi D t_j)^(dim/2)`.

Two consequences follow, and both are measured rather than asserted:

1. **The readout is a calibration quantity, not a search quantity.**  Because it
   enters linearly, it can be set by a one-time offline procedure (drive the
   sources, read the field, solve).  Section 6.
2. **Forgetting the amplitude calibration is catastrophic.**  The uncalibrated
   transfer error is 9.98 against 5.92e-08 calibrated — eight orders of magnitude
   (`sub_01` Part 3).  The reason is that sources of different width have
   different peak amplitudes, so each particle's contribution is scaled
   differently unless `q_j` is set per source.  The calibration must be redone
   whenever a width `h_j` changes.

---

## 5. Where the capacity sits, and why the naive answer is wrong

> **How to read every "geometry share" in this document.**  The shares describe the
> **single-channel variant that the ablations actually measure**, not the production
> network.  A layer has TWO memristor crossbars: the simulator's forward fuses them
> as `f = out_g + out_a`, using one kernel of sharpness `k=0.5` and one of `k=0.3`
> (`simulator/network_nd.py:139`, `simulator/layer_nd.py:205-229`).  The ablation
> harness runs **single channel** instead — `W_act` is pinned to zero and excluded
> from the parameter count — explicitly to match the repository's regression trainer
> (`experiments/abl_common.py:535`, `HiMALND._init_layer(task="regression")`).
>
> The consequence is a factor of two in the readout.  For F2, geometry is
> `224 = (4*40+40) + (1*8+8) + (1*4+4)` and readout is `52 = 40+8+4`, giving the
> 81% below; **in the production dual-channel configuration the readout is 104 and
> the share is 68%**.  So `81%` is a property of the measured variant, not of "the
> network", and the same applies to 69% and 77%.
>
> This matters beyond bookkeeping, because the quantity that controls the
> calibration is the readout-to-sample ratio (section 6.8): F2 sits at `52/256 =
> 0.20x` single-channel and `104/256 = 0.41x` dual-channel, while the one
> architecture where calibration fails (W) sits at `2.19x`.  **Widening the readout
> by enabling the second channel moves a network toward that failure boundary, so
> track the ratio, not the share.**

Ablation A trains disjoint parameter subsets at a stated measurement cost.

**Equal measurement cost, 3 seeds, 80 epochs, 4D regression, circuit substrate:**

| arch | geometry share | geometry only | readout only | full |
|---|---|---|---|---|
| G  | 69% | +0.3206 +- 0.0993 | +0.4134 +- 0.0253 | +0.3869 +- 0.0367 |
| F  | 77% | +0.3259 +- 0.1738 | +0.4078 +- 0.1118 | +0.3484 +- 0.1466 |
| F2 | 81% | +0.2684 +- 0.0455 | +0.4346 +- 0.0477 | +0.4722 +- 0.0321 |

`corr(geometry share, geometry-arm retained R^2) = -0.53`

The readout wins 3/3, but note what changed from the 2-seed / 40-epoch run of the
same protocol, where the geometry arm reached only +0.166 / +0.115 / +0.093: **the
geometry arm is strongly epoch-dependent** (it roughly doubles between 40 and 80
epochs), and the share correlation weakens from -0.96 to -0.53.  The ordering is
robust; the "more geometry share makes it worse" reading is not.  Read section 6
for the statement that survives both protocols and both epoch counts.

**Equal coverage, 2 seeds, 40 epochs, five architectures:**

| arch | geometry share | geometry | readout |
|---|---|---|---|
| W  | 47% | **+0.1859 +- 0.0994** | +0.1447 +- 0.0801 |
| M  | 59% | +0.0386 | +0.1536 |
| G  | 69% | +0.0597 | +0.1026 |
| F  | 77% | -0.0114 | +0.0255 |
| F2 | 81% | +0.0186 | +0.0722 |

`corr = -0.73`

**The winner tracks the parameter COUNT, not the parameter TYPE.**

| arch | geometry vs readout params | predicted winner | equal_probes | equal_sweeps |
|---|---|---|---|---|
| G  | 204 > 92 | readout | readout | readout |
| M  | 260 > 184 | readout | **readout** | readout |
| W  | 500 < 560 | **geometry** | **readout (exception)** | **geometry** |
| F  | 176 > 52 | readout | readout | readout |
| F2 | 224 > 52 | readout | readout | readout |

**9/10, with one exception that is itself informative.**  W is the only
architecture in which geometry has fewer parameters than the readout, and the only
one in which geometry wins — but only under **equal coverage**.  Under **equal
measurement cost** the readout wins there too (`+0.4650` against `+0.3556`,
section 6.8), because the arm with FEWER parameters completes more sweeps of its
own space per probe, and equal-cost probing rewards that.  So the rule is: the
winner tracks the parameter count *and* the protocol, and W is the single cell
where the two criteria disagree.

**This result is FINITE-TIME, not asymptotic** — established by the epochs x
budget grid of section 6.10, which is the single most important qualification in
this section.  At budget 1 200 the gap between the readout arm and the geometry arm
falls from +0.310 at 20 epochs to **-0.017 at 160 epochs**: the geometry arm
catches up and slightly overtakes once it is given enough total measurements.  So
"the readout carries more capacity" is really "**the readout learns faster**", and
every endpoint comparison in this section must state its epoch count.

Two independent checks confirm this is an **optimization-cost** effect and not a
statement about representational power:

* Phase 2 of ablation A (freeze one half, fine-tune the other from a trained
  model) gives `kept = 1.10 ... 1.49`: both halves can be re-fitted, so the
  solution is distributed and neither half is "too small to hold it".
* Under equal measurement cost the **readout-only arm beats the full arm** in the
  2-seed / 40-epoch run in 3/3 (G 0.398 vs 0.289, F 0.455 vs 0.247, F2 0.417 vs
  0.288): at a fixed budget, training fewer parameters well beats training more
  parameters badly.  At 80 epochs the two converge (G 0.413 vs 0.387).

**Retraction:** the strong reading of "nodes replace matrices" — that the
movable particle geometry carries the network's capacity — is **not supported**.
The geometry arm loses to the readout arm on every architecture at equal cost
**at the epoch counts used here (40 and 80)**.  The epochs x budget grid of section
6.10 then shows the gap closing to zero, and inverting, at 160 epochs — so the
defensible statement is narrower than a capacity claim and narrower than a pure
rate claim: **the readout is the faster learner; the geometry converges to the same
place if the measurement budget is large enough.**  Neither half is the carrier;
they are the same fit reached at different speeds.

---

## 6. HEADLINE: calibrate the strengths, spend the measurements on the geometry

> **Scope, established in sections 6.5-6.7.**  This rule is safe exactly when the
> readout is an adequate model of the target and the feature matrix is well
> conditioned.  Swept over seven tasks, it **wins 9 of 14 task x architecture
> cells** — including a classification task (largest win) and a distribution-shift
> task — and loses the other five, notably a 10-dimensional task with 8 pure-noise
> dimensions and a heteroscedastic-noise task (which the calibration cannot fit
> because it *is* ordinary least squares).
>
> **The calibrated arm is never worse than not calibrating.**  Every calibration
> must pass a symmetric out-of-sample gate before deployment (sections 6.6-6.7), so
> the arm can only ever improve on the geometry arm.  Worst-case R^2 across all
> fourteen cells is **-0.038** (3 seeds), against **-0.826** for the same sweep
> without the gate.  Note the limit of that guarantee: it is about *not calibrating*,
> not about beating every arm -- in one losing cell the gate did accept a
> calibration, the arm did improve, and it still lost (section 6.7).  Read 6.5-6.7
> before applying this anywhere.
>
> **RETRACTED as stated, and the distinction matters.**  The guarantee holds **per
> run** and not **per deployment**: measuring the arm's test `R^2` immediately before
> and after each write showed **55 of 116 deployed calibrations left the arm worse**
> (extension 12.19).  The end-of-run claim above survives -- every cell still ends
> ahead, so the dips are recovered -- but "the arm can only ever improve" is false at
> the moment of the write.  Evidence: extension `FINDINGS.md` 12.18 and 12.19.

If the strengths are set by closed-form calibration instead of being searched,
and every per-epoch physical measurement goes into the geometry, the same
measurement budget buys far more accuracy.  The effect is largest exactly where a
physical device is short: **at low measurement counts**.

**Equal measurement cost, 3 seeds, 80 epochs, circuit substrate:**

| arch | geometry share | geometry | readout | **hybrid** | hybrid_once | full |
|---|---|---|---|---|---|---|
| G  | 69% | +0.3206 | +0.4134 | **+0.4141 +- 0.0511** | +0.3821 | +0.3869 |
| F  | 77% | +0.3259 | +0.4078 | **+0.5276 +- 0.0720** | +0.4520 | +0.3484 |
| F2 | 81% | +0.2684 | +0.4346 | **+0.6380 +- 0.0069** | +0.5540 | +0.4722 |

**Same ordering on the diffusion substrate** (F2, equal probes, 2 seeds): geometry
0.094, readout 0.417, **hybrid 0.543**, full 0.288.

### 6.1 The result that survives everything: the scaling curve

One run records the whole R^2 trajectory against **physical measurements
consumed** (`--curve`, on by default), so the comparison is a rate rather than a
point estimate.  Test R^2 at fixed measurement milestones, mean over 3 seeds:

| arch | arm | 3 000 | 10 000 | 25 000 | 50 000 | 100 000 |
|---|---|---|---|---|---|---|
| G  | geometry | -0.0932 | +0.0575 | +0.1281 | +0.2061 | +0.3206 |
| G  | readout  | +0.0543 | +0.1776 | +0.2650 | +0.3552 | +0.4134 |
| G  | **hybrid** | **+0.3879** | **+0.4745** | **+0.4945** | +0.4740 | +0.4141 |
| G  | full     | -0.0181 | +0.0833 | +0.2030 | +0.2874 | +0.3869 |
| F  | geometry | -0.1793 | +0.0019 | +0.1031 | +0.2237 | +0.3259 |
| F  | readout  | +0.0442 | +0.1936 | +0.2925 | +0.3790 | +0.4078 |
| F  | **hybrid** | **+0.5257** | +0.4998 | **+0.5223** | **+0.5251** | **+0.5276** |
| F  | full     | -0.0335 | +0.0862 | +0.2313 | +0.2895 | +0.3484 |
| F2 | geometry | -0.1069 | -0.0059 | +0.0529 | +0.1147 | +0.2684 |
| F2 | readout  | +0.0581 | +0.1983 | +0.3184 | +0.3989 | +0.4346 |
| F2 | **hybrid** | **+0.5423** | **+0.6094** | **+0.6190** | **+0.6227** | **+0.6380** |
| F2 | full     | -0.0643 | +0.0376 | +0.1345 | +0.3021 | +0.4722 |

**At 3 000 measurements the calibrated arm is at +0.39 / +0.53 / +0.54 while every
other strategy — searching the strengths, searching the geometry, or searching
both — is at or below +0.06.**  To reach R^2 = 0.30:

| arch | hybrid | readout search | geometry search |
|---|---|---|---|
| G  | < 3 000 | ~40 000 | ~100 000 |
| F  | < 3 000 | ~30 000 | ~90 000 |
| F2 | < 3 000 | ~25 000 | not reached within 100 000 |

i.e. **15-30x fewer physical measurements for the same accuracy**, and in the
matrix-free architectures the geometry-only arm does not get there at all inside
the tested budget.

Two honest qualifications visible in the same table:

* The hybrid curves are **flat or declining after 10 000-25 000 measurements**
  (G: 0.495 at 25k falling to 0.414 at 100k).  The advantage is a *low-budget*
  advantage, and at very large budgets the arms converge (G at 100k: hybrid
  0.414 vs readout 0.413 — a tie).
* `hybrid_once` (calibrate only at epoch 0) is **not** good enough: it is fitted
  to a random particle layout and never refreshed.  It sits below `hybrid`
  everywhere and below both `readout` and `full` on G.  The 2-seed / 40-epoch
  result that made it look competitive is **retracted**.

### 6.2 Does the calibration survive the device?  (ablation D)

The recommendation above rested on an untested assumption: that a calibration
performed **on hardware** still works.  A calibration is a least-squares solve
whose feature matrix is *measured*, and such a solve amplifies noise in its
features.

Design that makes the test falsifiable: device non-idealities degrade **all**
arms, because they sit in the forward path, so every device level is run twice —
once with twin-supplied features and once with features read through the device
sensor chain.  The gap between those two is the calibration's noise sensitivity.
The calibration is performed physically every 10 epochs (per-epoch physical
recalibration costs 13 312 measurements per epoch and is not a defensible
scheme) and its cost is charged to the x-axis.

Architecture F2, 2 seeds, 40 epochs, diffusion cell, compared at the **common
measurement frontier** of each device level (48 000 measurements):

| device axis | level | geometry | readout | full | hybrid (twin) | **hybrid (physical)** | cost |
|---|---|---|---|---|---|---|---|
| sensor noise | 0 | +0.094 | +0.417 | +0.288 | +0.564 | **+0.573** | 2.11x |
| sensor noise | 1e-05 | +0.018 | +0.380 | +0.187 | +0.562 | **+0.570** | 2.11x |
| sensor noise | 1e-04 | -0.029 | +0.101 | +0.021 | +0.556 | **+0.531** | 2.11x |
| sensor noise | 1e-03 | -0.034 | +0.110 | +0.034 | +0.542 | **+0.557** | 2.11x |
| positioner | 0 | +0.094 | +0.417 | +0.288 | +0.564 | **+0.573** | 2.11x |
| positioner | 2e-05 m | -0.055 | +0.179 | +0.057 | +0.519 | **+0.522** | 2.11x |
| positioner | 5e-05 m | -0.042 | +0.200 | +0.066 | +0.515 | **+0.520** | 2.11x |
| positioner | 1e-04 m | -0.103 | +0.192 | +0.035 | +0.497 | **+0.509** | 2.11x |

**The headline survives at 8/8 device levels — while being charged 2.11x the
measurement budget.**  The calibrated arm reaches the frontier having prepaid
four physical calibrations, i.e. at an *earlier training epoch* than every other
arm, and still wins.  (Verified that the control is clean: at zero device noise
the twin-supplied and physically-measured calibrations produce bit-identical
parameters, max difference 0.0, so the frontier gap is purely the epoch at which
each arm arrives.)

Three results worth stating separately:

1. **The device makes the calibration MORE valuable, not less.**  At zero noise
   the calibrated arm leads the best search arm by +0.156; at a realistic readout
   noise of 1e-4 the lead is **+0.430** (0.531 vs 0.101).  Mechanism: the
   calibration solves for the strengths *given the features the device actually
   has*, so it compensates for the imperfection automatically, whereas the search
   arms need a gradient signal that the same noise destroys.
2. **Readout noise at or above 1e-4 (relative to full scale) kills
   finite-difference search** — and not via an untuned probe step.  The probe step
   was adapted to the device noise floor for every arm, a rule that exists because
   of a measurement: at 1e-5 noise the readout arm falls from +0.280 to +0.210 at
   `eps_rel=1e-3` and recovers to +0.279 at `eps_rel=1e-2`.  At 1e-4 and 1e-3 the
   search arms collapse with the step already adapted.
3. **The positioner axis tests the device, not the calibration.**  Positioner
   error is a deterministic perturbation inside every forward, so the twin and
   measured feature matrices are identical on that axis by construction; it
   degrades every arm, the calibrated one least.

Caveats: 2 seeds; only the sensor-noise axis actually probes the calibration's
readout sensitivity; the physical calibration is charged at the naive per-source
rate (`n_train * sum_k N_k`), which is an upper bound — calibrating on fewer
samples or against a tracking twin would cost less.

### 6.3 Does the calibration survive a WRONG digital twin?  (ablation E)

Sections 6.1 and 6.2 leave one assumption untested.  In every experiment above the
calibration read its feature matrix from the **same object** that then performed
the forward pass, i.e. twin and device were identical by construction and twin
error was exactly zero.  That is not the real situation:

* the search arms probe the **real** device loss, so they adapt to whatever the
  device actually does;
* the calibrated arm trusts the twin's description of the cell.

Twin error reaches the strengths through two channels, both modelled here by
calibrating on a **perturbed copy** of the device and then deploying the result
onto the true device: (1) a wrong twin produces a wrong feature matrix `Z` for the
solve; (2) converting the solved `W` into an injected pulse energy uses the twin's
width, `q_j = W_j (4 pi D t_j)^(dim/2)` with `t_j = h_j^2/(4kD)` (here `D` is the
diffusion coefficient, `k` the kernel sharpness and `dim` the medium dimension — see
the notation note in section 4).  The concern is
not hypothetical: the twin of an analog cell carries exactly the process spread
that ablation C measured on the device.

Architecture F2, diffusion cell, 2 seeds, 40 epochs, calibration every 10 epochs,
all device non-idealities OFF so the only mismatch is twin-vs-device.

**Controls, both PASS:**

* twin error 0 is bit-reproducible across the two axes (+0.5892 both) — the
  deployment path is exact;
* the search arms read the twin nowhere, and their spread across the twin-error
  axis is **exactly 0.0000** for all three.  The perturbation is therefore
  confined to the twin, which is what makes the degradation below causally
  attributable to twin error rather than to a leak into the device.

**Result — the calibrated arm wins at every level tested** (50 000 measurements).
**Re-measured at 3 seeds**, which corrected the 2-seed numbers below it:

| twin width error | best search arm | calibrated (3 seeds) | margin |
|---|---|---|---|
| 0 | +0.3962 | **+0.5557** | +0.160 |
| 1% | +0.3962 | **+0.5850** | +0.189 |
| 5% | +0.3962 | **+0.5862** | +0.190 |
| 10% | +0.3962 | **+0.5848** | +0.189 |
| 20% | +0.3962 | **+0.5672** | +0.171 |

**The recommendation of section 6 does not need a twin-accuracy caveat inside the
tested range, and the margin does not shrink: it is flat to 10% and still +0.171 at
20%.**  The 2-seed run of the same sweep showed a monotone decline to +0.069 and
this section originally warned that "40% of the margin is gone by 20% width error".
**That warning was seed noise and is withdrawn** — the 3-seed sweep gives a
margin at 20% error *above* the zero-error margin.  What survives is the
self-deception result below, which is unaffected.  The axis still was not extended
far enough to find a crossover, so "no caveat needed" is a statement about the
tested range only.

**Width error is the dangerous one; a constant position bias is nearly free.**
A twin position offset of 0.005 and 0.020 input units (the data spans ~4 units)
moves the calibrated arm from +0.5892 to +0.5876 and +0.5896 — inside noise.  A
kernel *shape* error cannot be absorbed by a linear solve for the strengths; a
small translation of the centres largely can.

**The finding worth carrying forward: the calibration's self-report is
anti-correlated with reality.**  At a fixed random geometry, one calibration, then
the same strengths read on both:

| twin width error | R^2 the twin believes | R^2 the device delivers | gap | (3 seeds) |
|---|---|---|---|---|
| 0 | +0.4533 | +0.4533 | 0.0000 | |
| 1% | +0.4583 | +0.4541 | +0.0042 | |
| 5% | +0.4762 | +0.4436 | +0.0326 | |
| 10% | +0.4973 | +0.4034 | +0.0939 | |
| 20% | **+0.5172** | **+0.2747** | **+0.2425** | |

(The 2-seed run gave +0.5253 / +0.0957 / +0.4296 at 20%; the *sign and shape* are
identical, only the magnitude of the gap differs, so this is robust to seeds where
the margin trend above was not.)

The twin's own number *improves* as the twin gets more wrong, while the device
truth collapses.  So a calibration that reports a good fit is **not** evidence
that it worked, and twin error cannot be detected from the calibration's own
residuals.  It has to be validated against the device by an independent
measurement — a concrete addition to the device specification in section 8, and a
warning for any twin-trained physical network, not only this one.

### 6.4 Definitions and cost accounting

* `hybrid` — probe the geometry only; re-calibrate the source strengths in
  closed form every epoch.
* `hybrid_once` — the same, but calibrate only at epoch 0 (the classical hybrid
  RBF scheme).

```
one-time calibration, physical route : 13312 measurement-equivalents
                                       = n_train (256) x sum_k N_k (52)
one-time calibration, digital twin   : 0 physical measurements
per-epoch FD probing                 : 1200 forward passes/epoch
```

The x-axis of the scaling table charges the **`hybrid_once`** calibration to the
device once, and counts the **`hybrid`** recalibration as twin-supplied (zero
physical cost).  That asymmetry is deliberate and is the honest reading:
refreshing a full calibration on the device every epoch would cost 13 312
measurements per epoch, far more than the 1 200 forward passes of FD probing it
accompanies, so a per-epoch physical recalibration is not a defensible scheme.
The defensible scheme is: **calibrate once on the device** (or continuously on a
twin that tracks it) **and put every measurement into the geometry.**

**Physical reading.**  This is what the "W is a source excitation strength"
interpretation predicts.  A strength is a physical amplitude (pulse energy,
memristor conductance, laser power) and can therefore be *set* by an offline
procedure; a position in space can only be *found* by search.  Honouring that
distinction is what makes the geometry the effective place to spend measurements —
and it is the geometry, not the strengths, that then carries the fit.

**This supersedes the "four-tier physical training" narrative.**  The four tiers
(SPSA / FD / analog gradient chain / digital twin) differ only in how the
*geometry* is optimised; the strengths should not be searched at all.

### 6.5 Where the rule STOPS working  (ablation F, task sweep)

Sections 6.1-6.3 all measure one 4D synthetic regression.  That leaves the claim
with an untested boundary, so ablation F sweeps six **self-contained** tasks (no
downloads, reproducible offline) that differ deliberately in dimensionality,
irrelevant dimensions, task type, target smoothness and noise structure.  Two of
them are designed as **adversaries of the rule**, not illustrations of it.

Same 256/128 split, same trainer, same seeds, same epoch count; two architectures
spanning the geometry share (G = 69%, F2 = 81%); equal measurement cost.

| task | character | plain linear readout R^2 | verdict (calibrated vs best search) |
|---|---|---|---|
| 2d | D=2 smooth | +0.028 | **wins both archs** (+0.836/+0.921 vs +0.635/+0.584) |
| 4d | D=4 smooth + product | +0.056 | **wins both** (+0.463/+0.534 vs +0.398/+0.417) |
| moons | D=2 two-class boundary | +0.692 | **wins both, largest margin** (+0.982/+0.985 vs +0.833/+0.705) |
| steps | D=4 DISCONTINUOUS | +0.101 | split: F2 wins (+0.239 vs +0.082), G loses (+0.112 vs +0.141) |
| hetero | D=4 HETEROSCEDASTIC | +0.095 | **loses both** (+0.160 vs +0.186; +0.175 vs +0.224) |
| 10d | D=10, 8 pure-noise dims | -0.006 | **loses both, catastrophically** (-0.285 and -0.633 vs ~0) |

**7 of 12 task x architecture cells** — as measured BEFORE the fix of section 6.6.
Section 6.6 repairs the catastrophic failure; the table after that fix, and why
the fix is an improvement even though it wins one cell less, is there.

The boundary is not task *type* —
`moons` is a classification task with a nonlinear decision boundary and it is the
largest win.  The two failure regimes are identifiable:

1. **Ill-conditioned features (`10d`): the closed-form solve OVERFITS, and
   calibrating is worse than not calibrating** — worse than predicting the mean.
   Mechanism: on F2 the readout has 224 features fitted on 256 samples while 8 of
   the 10 input dimensions are pure noise, so the feature matrix is close to
   degenerate and the ridge solve returns a huge `W` that does not generalise.
   The finite-difference arms move 600 coordinates per epoch by a small step and
   stay near the random initialisation, so they do not overfit.  This is a
   **deployment risk with a known fix** (stronger regularisation, and a fallback
   to searching if the calibration cannot beat the mean on held-out data) — not
   evidence that the mechanism is wrong.  It is also the sharpest argument for
   keeping a *validation* slice that the current `_choose_ridge_lam` candidate
   range (1e-4 .. 1e-1) may simply be too small to cover.
2. **Misspecified likelihood (`hetero`): the calibration IS ordinary least
   squares, so non-uniform noise makes it the wrong estimator.**  This failure was
   predicted by the experimental design before it was measured.
3. `steps` is intermediate: with a discontinuous target the ceiling is low for
   every arm (best search +0.14) and the arms are within 0.03, so the comparison
   there is not informative in either direction.

**Consequence for section 6.**  The rule should be stated with its conditions, not
outright: *calibrate the strengths when the readout is an adequate model of the
target and the feature matrix is well conditioned; regularise the calibration and
validate it on held-out data, falling back to searching when it does not beat the
mean.*  The 15-30x figure of section 6.1 is measured on the `4d` task; on the
friendly tasks of this sweep the calibrated arm reaches the best search arm's
final accuracy within the first one or two epochs (the ratio saturates at the
frontier/first-point value of ~40x, so treat that number as a lower bound rather
than a measurement).

### 6.6 Fixing the `10d` failure, and the bug that made the first two attempts useless

Three attempts; the first two failed in an instructive way.

**Attempt 1 -- relative ridge.**  `lam` was an ABSOLUTE ridge (candidates 1e-4 ..
1e-1), which cannot transfer between tasks.  It is now RELATIVE to the Gram scale
(`lam * mean(diag(Z^T Z))` per layer) with candidates spanning 1e-6 .. 1e0.  This
improved `4d` materially (+0.4625 -> +0.5343 and +0.5338 -> +0.5992) but made
`10d` **worse** (-0.2848 -> -0.5365 and -0.6333 -> -0.8262).

**Attempt 2 -- a do-no-harm guard, criterion "beat the strengths in place".**  Made
`10d` worse still, and produced numbers *identical to four decimals* to attempt 1,
which is what exposed the real bug: the guard was scoring a solve that had been
fitted on **all** the training samples against a "held-out" slice drawn from those
same samples.  It was reading **in-sample** fit.  Measured directly on `10d`: the
guarded slice reported **+0.4453** while the true test R^2 was **-0.9258** -- the
guard was blind exactly where it was needed, and it locked in a harmful early
calibration while the geometry trained against it.

**Attempt 3 -- fit-validate-refit (the fix).**  The guard now fits on 3/4 of the
training data, scores on the excluded 1/4, and refits on everything only if that
score passes.  Three solves per calibration, negligible next to FD probing.

| task | arch | calibrations | rejected | hybrid before -> after |
|---|---|---|---|---|
| 10d | G | 80 | **80** | -0.2848 -> **-0.0355** (equals the geometry arm exactly) |
| 10d | F2 | 80 | **80** | -0.6333 -> **+0.0358** (equals the geometry arm exactly) |
| 4d | G | 80 | 78 | +0.4625 -> **+0.5006** (best recorded) |
| 4d | F2 | 80 | 78 | +0.5338 -> **+0.6047** (best recorded) |

Two outcomes worth separating:

1. **`10d` is no longer catastrophic, only useless.**  The guard rejects every
   calibration (80/80), so the arm degrades exactly to the geometry arm instead of
   collapsing below the mean predictor.  It still loses that task by 0.02-0.03, but
   the failure mode is now "no benefit", not "harm".
2. **`4d` improved to its best recorded values although 78 of 80 calibrations were
   rejected.**  Re-fitting the strengths *every* epoch makes them chase the moving
   geometry and overfit; accepting only calibrations that survive validation keeps
   them stable while the geometry trains.  The practical rule is therefore not
   "calibrate every epoch" but **"calibrate and validate; deploy only on
   improvement"** -- and the acceptance rate is whatever validation says, not a
   schedule to be tuned.

This was a bug in the *harness*, exposed by two code versions disagreeing to four
decimals where they should have differed.  It is recorded because the same mistake
-- scoring a fit on the samples it was fitted on -- is the standard way a
validation gate silently stops working.

**The complete table after the fix** (all six tasks re-run, 2 seeds, 40 epochs).
**This is the INTERMEDIATE state**: the gate it uses still had the asymmetric
comparison that section 6.7 removes, so read the table there for the final numbers.

| task | arch | geometry | readout | full | hybrid before | **hybrid after** | margin |
|---|---|---|---|---|---|---|---|
| 2d | G | +0.4884 | +0.5833 | +0.6346 | +0.8364 | **+0.7869** | +0.152 |
| 2d | F2 | +0.2777 | +0.5844 | +0.4712 | +0.9209 | **+0.8606** | +0.276 |
| 4d | G | +0.1657 | +0.3982 | +0.2886 | +0.4625 | **+0.5006** | +0.102 |
| 4d | F2 | +0.0938 | +0.4173 | +0.2875 | +0.5338 | **+0.6047** | +0.187 |
| moons | G | +0.7080 | +0.8328 | +0.8116 | +0.9821 | **+0.9837** | +0.151 |
| moons | F2 | +0.4785 | +0.7049 | +0.6622 | +0.9852 | **+0.9849** | +0.280 |
| steps | G | +0.0492 | +0.1408 | +0.0973 | +0.1121 | +0.0492 | -0.092 |
| steps | F2 | +0.0349 | +0.0815 | +0.0759 | +0.2393 | **+0.0984** | +0.017 |
| hetero | G | +0.0820 | +0.1858 | +0.1243 | +0.1601 | +0.0820 | -0.104 |
| hetero | F2 | +0.0774 | +0.2239 | +0.1718 | +0.1750 | +0.0774 | -0.147 |
| 10d | G | -0.0355 | -0.0066 | -0.0394 | -0.2848 | -0.0355 | -0.029 |
| 10d | F2 | +0.0358 | -0.0099 | +0.0526 | -0.6333 | +0.0358 | -0.017 |

**The fix is worth having even though it wins one cell less:**

| | before the fix | after the fix |
|---|---|---|
| cells won | 7 / 12 | 6 / 12 |
| **worst hybrid R^2 anywhere** | **-0.8262** | **-0.0355** |
| failure mode | collapses BELOW the mean predictor | degrades to the geometry arm (no harm) |
| 4d, the task everything else used | +0.4625 / +0.5338 | **+0.5006 / +0.6047** |

The downside risk falls by a factor of **23**, every decisive win survives, and the
task the project is built on improves.  The cost is margin, not correctness: `2d`
gives up 0.05-0.06 and `steps`/F2 gives up most of its (small) win.  In the six
cells where the guard rejects everything, the calibrated arm is *exactly* the
geometry arm — the guard never leaves the arm worse than not calibrating.
**(Per deployment this is now measured to be false: see the retraction at the end of
6.11. It holds per run, not per write.)**

**Residual limitations of the fit-validate-refit gate (stated, not hidden).**  The
exit criterion "score > score of the strengths in place" compared an honest
out-of-sample fit against a deployed `W` that had itself been fitted on the
training data, so it was biased toward rejecting, and on `steps`/F2 it nearly
flipped a win into a loss.  **Fixed in section 6.7.**  What remains: the acceptance
rate is low (0-4 of 80 depending on the task), so "recalibrate every epoch" is
effectively "try every epoch, deploy rarely" -- the cost of the attempt is
negligible, but the schedule claims in section 6.3 should be read in that light.

### 6.7 A symmetric gate, a distribution-shift test, and the counterexample that IS there

**A -- remove the gate's bias.**  The guard now reserves a slice that **no fit ever
touches**: the candidate strengths are fitted on the complement and the deployed
strengths are scored on the same slice, so both sides of the comparison are
out-of-sample.  The geometry is deliberately NOT handicapped -- the gate validates
the *strengths*, not the geometry, so finite differences still use the full set.
This also makes the calibration cheaper (one fit per attempt instead of three).
The cost is that the calibration fits on 75% of the training data.

| task/arch | biased gate | **symmetric gate** | change |
|---|---|---|---|
| 2d / G | +0.7869 | **+0.8033** | +0.016 |
| 2d / F2 | +0.8606 | +0.8597 | -0.001 |
| 4d / G | +0.5006 | +0.4967 | -0.004 |
| 4d / F2 | +0.6047 | +0.5917 | -0.013 |
| moons / G | +0.9837 | **+0.9842** | +0.001 |
| moons / F2 | +0.9849 | +0.9849 | 0.000 |
| **steps / F2** | +0.0984 (margin +0.017) | **+0.1070 (margin +0.025)** | **+0.009** |
| 10d, hetero, steps/G | unchanged | unchanged | 0 |

Rejections fall on the friendly tasks (2d/G 74 -> 69, 4d/G 78 -> 76), which is the
bias being removed, and **the cell that was nearly flipped is now clear of the
line**.  The 4d loss of 0.004-0.013 is the price of fitting on 192 samples instead
of 256, and it is the only cost.

**B -- the distribution-shift test.**  The gate validates **in-distribution**, so it
structurally cannot see a shift; the missing counterexample was a cell where the
gate ACCEPTS and the arm still loses.  A `shift` task was added: train support
`[-2,2]`, test support `[-3,3]`, identical functional form.

| task | arch | geometry | readout | full | **hybrid** | margin |
|---|---|---|---|---|---|---|
| shift | G | +0.0367 | +0.0463 | +0.0816 | **+0.1853** | **+0.104** |
| shift | F2 | -0.0025 | +0.0769 | +0.0461 | **+0.1595** | **+0.083** |
| shift (3 seeds) | G | +0.0481 | +0.0543 | +0.0871 | **+0.1261** | **+0.039** |
| shift (3 seeds) | F2 | +0.0068 | +0.0916 | +0.0683 | **+0.1195** | **+0.028** |

**The prediction was wrong, in the direction that favours the rule -- but the
margin is small.**  The expectation was that a large calibrated `W` multiplying
activations that decay outside the training support would amplify error and lose;
instead the calibrated arm wins both cells, in the one regime the gate structurally
cannot check.  **At 3 seeds the margins are +0.039 and +0.028, not the
+0.104 / +0.083 the 2-seed run gave**: an earlier version of this section claimed
the shift margin was "larger than it achieves on `4d`", and that was seed noise --
on `4d` the margins are +0.125 / +0.161.  What survives is the qualitative result
(the advantage does transfer out of distribution) plus the honest absolute reading
that the shift makes the task hard for everyone: the best search arm manages +0.09.

**Complete table, seven tasks, symmetric gate** (2 seeds, 40 epochs, all six
earlier tasks re-run so the numbers are consistent):

| task | arch | geometry | readout | full | **hybrid** | verdict | (3 seeds) |
|---|---|---|---|---|---|---|---|
| 2d | G | +0.4884 | +0.5833 | +0.6346 | **+0.8033** | win | +0.4849 / +0.5779 / +0.6479 / **+0.7756** |
| 2d | F2 | +0.2777 | +0.5844 | +0.4712 | **+0.8597** | win | +0.3601 / +0.5817 / +0.5235 / **+0.8561** |
| 4d | G | +0.1657 | +0.3982 | +0.2886 | **+0.4967** | win | +0.2038 / +0.3762 / +0.2768 / **+0.5009** |
| 4d | F2 | +0.0938 | +0.4173 | +0.2875 | **+0.5917** | win | +0.1127 / +0.3962 / +0.2952 / **+0.5570** |
| moons | G | +0.7080 | +0.8328 | +0.8116 | **+0.9842** | win | +0.7383 / +0.8477 / +0.8368 / **+0.9843** |
| moons | F2 | +0.4785 | +0.7049 | +0.6622 | **+0.9849** | win | +0.6214 / +0.7567 / +0.7494 / **+0.9852** |
| shift | G | +0.0367 | +0.0463 | +0.0816 | **+0.1853** | win | +0.0481 / +0.0543 / +0.0871 / **+0.1261** |
| shift | F2 | -0.0025 | +0.0769 | +0.0461 | **+0.1595** | win | +0.0068 / +0.0916 / +0.0683 / **+0.1195** |
| steps | F2 | +0.0349 | +0.0815 | +0.0759 | **+0.1070** | win | +0.0374 / +0.0631 / +0.0695 / **+0.0854** |
| steps | G | +0.0492 | +0.1408 | +0.0973 | +0.0492 | loss (= geometry) | +0.0470 / +0.1214 / +0.0818 / +0.0470 |
| hetero | G | +0.0820 | +0.1858 | +0.1243 | +0.0820 | loss (**1 accepted**) | +0.0884 / +0.1457 / +0.1546 / +0.1228 |
| hetero | F2 | +0.0774 | +0.2239 | +0.1718 | +0.0774 | loss (= geometry) | +0.0750 / +0.2241 / +0.1656 / +0.0750 |
| 10d | G | -0.0355 | -0.0066 | -0.0394 | -0.0355 | loss (= geometry) | -0.0379 / +0.0111 / -0.0425 / -0.0379 |
| 10d | F2 | +0.0358 | -0.0099 | +0.0526 | +0.0358 | loss (= geometry) | +0.0239 / -0.0144 / +0.0410 / +0.0239 |

**9 of 14 cells, reproduced at 3 seeds** (the last column re-runs the whole table;
every verdict is unchanged and the decisive wins stay large, +0.128 to +0.274).

**The guarantee is about NOT CALIBRATING, not about beating every arm.**  Worst
hybrid R^2 anywhere is -0.038, i.e. the calibrated arm never ends below the geometry
arm.  In four of the five losses the gate refused every calibration so the arm *is*
the geometry arm; in the fifth (`hetero`/G at 3 seeds) it accepted **one**
calibration out of 120, the arm improved from +0.0884 to +0.1228, and still lost to
`full` at +0.1546.  That is the "gate accepts and the arm still loses" case section
6.11 was looking for -- and it falsifies "**the calibrated arm beats the best search
arm**", which was only ever an empirical 9/14, not the guarantee.  The guarantee
itself (never worse than not calibrating) is untouched, because +0.1228 > +0.0884.
**THAT LAST SENTENCE IS NOW RETRACTED**: the guarantee is false *per deployment* — 55 of
116 deployed calibrations left the arm worse than immediately before the write (extension
12.19).  This cell's own reading is unaffected: its calibration **helped**, which is why it
is not the counterexample.  See the retraction at the end of 6.11.

**The `speedup` column of the ablation-F report must not be quoted.**  It is
computed as "did the calibrated curve ever reach the best search arm's final value",
so a curve that reaches it early and then **declines** scores a full 40x: `hetero`
shows a mean speedup of 40.0x in that report while losing both of its cells.  Use
the common-frontier comparison above.

### 6.8 What actually controls the calibration?  (share axis, five architectures)

The share axis had only been sampled at three points (69 / 77 / 81 %).  Running all
five architectures under one protocol (equal measurement cost, diffusion cell, 2
seeds, 40 epochs) fills it in and answers the question the earlier sections could
not: *is the geometry share the controlling variable?*

**Read the "readout params" column as the primary quantity and the share column as
a derived one.**  All shares here are single-channel (section 5's opening note); the
readout-to-sample ratio is what actually controls the calibration, and it is what
moves when the second channel is enabled.

| arch | geometry share | readout params / 256 samples | geometry | readout | **hybrid** | full | hybrid verdict |
|---|---|---|---|---|---|---|---|
| F2 | 81% | 0.20x | +0.0938 | +0.4173 | **+0.5917** | +0.2875 | win +0.174 |
| F  | 77% | 0.20x | +0.1145 | +0.4552 | **+0.5065** | +0.2474 | win +0.051 |
| G  | 69% | 0.36x | +0.1657 | +0.3982 | **+0.4964** | +0.2886 | win +0.098 |
| M  | 59% | 0.72x | +0.2015 | +0.4232 | **+0.5340** | +0.4391 | win +0.095 |
| **W** | **47%** | **2.19x** | +0.3556 | +0.4650 | **+0.3201** | **+0.5024** | **loss -0.182** |

**The geometry share does NOT control the calibration.**  The margin at 3 000
measurements is +0.446 / +0.454 / +0.452 / +0.510 across 81 / 77 / 69 / 59 % — a
22-point change in share moves it by 0.065, i.e. essentially flat, even though the
correlation over four points is -0.891.  Report the effect size, not the
correlation.

**What does control it is the readout's parameter-to-sample ratio.**  W is the only
architecture whose readout (560 strengths) exceeds the training set (256 samples),
and it is the only one where the calibration loses.  Combined with section 6.5
(where `10d` failed at a ratio of only 0.20x because 8 of its 10 input dimensions
were pure noise, collapsing the feature matrix's effective rank), the calibration
has **two necessary conditions**:

1. **readout parameters < training samples** — otherwise the closed-form solve is
   underdetermined and the gate refuses it (W);
2. **the feature matrix must not be rank-deficient relative to the signal** — a
   low parameter count is not sufficient (10d).

**The share axis does control the GEOMETRY arm**, strongly and monotonically:
`corr(geometry share, geometry-arm retained R^2) = -0.978` over five architectures
(0.22 at 81% falling to 0.76 at 47%).  So the share axis is the right x-axis for
section 5's parameter-count rule and the wrong x-axis for section 6's calibration
rule -- the two rules are controlled by different quantities, which is worth
stating because it is easy to conflate them.

One measurement artifact to note: W's margin cell reads `nan` because W's budget is
2 100 probes/epoch, so its first recorded point is at 4 200 measurements and the
3 000-measurement milestone does not exist for it.  That is milestone coverage, not
a result.

### 6.9 The calibration schedule is free

The gate accepts only 0-4 of 80 calibration attempts, so "recalibrate every epoch"
looked like wasted effort.  Sweeping the attempt period on `4d` (F2, budget 600,
40 epochs) settles it:

| attempt every | test R^2 | attempts | accepted | forward passes |
|---|---|---|---|---|
| 1 epoch | +0.5891 | 40 | 1 | 48 000 |
| 5 epochs | +0.5891 | 8 | 1 | 48 000 |
| 10 epochs | +0.5891 | 4 | 1 | 48 000 |
| once, at epoch 0 | +0.5891 | 1 | 1 | 48 000 |

**Identical to four decimals, with exactly one acceptance in every case.**  The
attempt period costs nothing and changes nothing: the gate, not the schedule,
decides which calibration is deployed, and in this configuration it is always the
epoch-0 one.  This removes the schedule as a design variable, and it retroactively
answers the cost worry raised in section 6.3 -- a physical device can attempt a
calibration cheaply and let the gate refuse it, or simply calibrate once.

### 6.10 Is the capacity split a convergence effect?  (epochs x budget)

Section 11 flagged that the geometry arm roughly doubles between 40 and 80 epochs,
which threatens the reading of section 5.  A grid over epochs x budget (F2, arms
geometry vs readout, 1 seed per cell -- the question is the SHAPE, not the
precision of a cell) answers it.  `gap` = readout R^2 minus geometry R^2:

| budget | gap @20 ep | gap @40 ep | gap @80 ep | gap @160 ep | trend |
|---|---|---|---|---|---|
| 300 | +0.1845 | +0.2346 | +0.2859 | +0.2101 | asymptotic |
| 600 | +0.2382 | +0.2935 | +0.2776 | **+0.0374** | closing |
| 1200 | +0.3101 | +0.2986 | +0.1165 | **-0.0169** | **closing, and inverting** |

**The gap is finite-time.**  At budget 1 200 it falls from +0.310 to -0.017: given
160 epochs the geometry arm catches up and slightly overtakes.  At budget 600 it
collapses to +0.037.  Only at budget 300 does it stay positive, i.e. with too few
probes per epoch the geometry never gets there.

Consequences, which are corrections rather than additions:

1. Section 5's ordering (**readout > geometry at equal measurement cost**) holds at
   40 and 80 epochs and must be labelled finite-time.  The right sentence is
   "**the readout learns faster**", not "the readout carries more capacity".
2. The parameter-count rule of section 5 is a rule about *which arm wins at a
   fixed effort*, and effort has two axes (budget and epochs) that trade off.  The
   grid shows the trade: 1 200 x 160 and 600 x 160 both close the gap, while
   300 x anything does not.
3. It also explains why section 6.1's scaling curves show the hybrid peaking early
   and then flattening: the *geometry* is still improving at large measurement
   counts, so a calibrated readout fitted to an early geometry stops being optimal.

### 6.11 Three attempts to break the "never worse" guarantee, all of which failed -- and a fourth that SUCCEEDED (see the retraction at the end of this section)

The gate guarantees that deploying a calibration never leaves the arm worse than
not calibrating.  The strongest missing evidence against the whole rule would be a
cell where the gate **accepts** and the arm still loses.  Three designs were tried:

| attempt | prediction | result |
|---|---|---|
| `shift` (train support [-2,2], test [-3,3]) | a large calibrated `W` times decaying activations amplifies error out of distribution, while the gate cannot see the shift | **wrong**: the calibrated arm is the MOST shift-robust, +0.185/+0.160 against +0.046/+0.077 for the best search arm |
| `noise` (y independent of x) | the gate validates on the same noise, so fitting noise raises its score and it accepts a calibration that cannot generalise | **wrong**: the gate REJECTED the pure-noise calibration outright (`rows=0`), because an honest out-of-sample score from fitting noise is negative -- the "beat the mean on data you did not fit" criterion is doing real work |
| validation slice shrunk to 32 and 16 samples | a noisier gate score lets a worse calibration win by luck, so harm leaks through on `10d` | **no change**: the gate still rejected 40/40 at every slice size, because `10d`'s failure is first-order rather than marginal |

So the guarantee has now survived three designed counterexamples — with one
qualification added by the 3-seed task sweep: the *stronger* claim "the calibrated
arm beats the best search arm" does have a counterexample (`hetero`/G, where one
calibration was accepted, improved the arm, and still lost to `full`).  That was
never the guarantee; it was the empirical 9/14.  Two further caveats keep this
honest: the third attempt **did not answer its own question** (a first-order failure
is detectable at any slice size, so it does not probe the gate's statistical power
-- that needs a task sitting near the decision boundary, which was not constructed),
and the script's "harm?" column in that part conflates *losing to another arm* with
*being worse than not calibrating*; since zero calibrations were accepted there, the
arm was exactly the geometry arm and the guarantee held.

> **RETRACTED.  The guarantee as stated is false per deployment, and section 6.11's
> conclusion above is superseded.**  The missing task was constructed (a task near the
> decision boundary: the knob is the **validation-slice size**, not the task's noise, and
> the accept rate moves `0.025 -> 0.508` across it), and the arm's **test** `R^2` was then
> measured **immediately before and after each deployment inside a single run** — so the
> harm is the calibration's and not a second trajectory's.  **55 of 116 deployed
> calibrations left the arm worse than it was immediately before the write.**  The
> mechanism needs no broken gate: the gate judges on a **stride of the training set**,
> while harm is measured on the **evaluation set**, so it can be right about the data it
> checks and wrong about generalisation.
>
> **Two qualifications travel with the retraction, and neither is optional.**  Every cell
> in that grid still ends with a **positive** margin over its geometry arm, so the dips are
> **recovered** and the recommendation this section supports — calibrate, and spend the
> measurements on the geometry — is **unaffected**.  And the harmful range reaches down to
> `-0.0003`, so part of the 55 is numerically indistinguishable from no change: **the
> guarantee is about the sign, and lumping `-0.0003` together with `-0.0502` would
> exaggerate the retraction.**  Evidence: extension `FINDINGS.md` 12.18 and 12.19,
> `outputs/ext_34_deployment_harm.txt`.

---

## 7. Locality, and when the independence claim holds

Measured coupling
`ratio[i,j] = |d z_i / d theta_j| / |d z_i / d theta_i|` for a layer of particles:

| normalisation | geometry block, mean off-diagonal | verdict |
|---|---|---|
| `none` (raw activations) | 0.000e+00 | LOCAL — the README claim holds literally |
| `fixed` (frozen constant scale) | 0.000e+00 | LOCAL |
| `pu` (`z / sum_j z_j`, used by the classification path) | 0.063 / 0.140 / 0.261 (L1/L2/L3) | COUPLED |

Identical on both substrates.  So the claim "RBF particle output contributions
are mutually independent" is **true without normalisation and false under the
partition-of-unity normalisation that the classification path actually uses**.

**Mechanism difference between substrates (a real finding).**  On the circuit
substrate PU is applied per particle before the readout; on the diffusion
substrate it is a single shared division of the summed field.  The function is
identical (section 2), but the coupling lives in different places: on the
diffusion cell the per-particle signals stay strictly local and the only shared
quantity is **one scalar per probe block**.  For parallel hardware training that
is the better arrangement.

`experiments/train_01_grouped_fd.py`, which the README cites for this claim,
does **not** test particles: its group indices use the `p*D` convention while the
parameter vector layout is `d*N+j` (verified: particle j=1 occupies indices
[1, 5, 9], not [3, 4, 5]).  Its "groups" mix the same dimension of different
particles.  The claim therefore needs the measurement above, not that script.

**Parallel-update price** (Gauss-Seidel vs Jacobi, equal forward passes), arch M,
`--seeds 1 --epochs 40 --budget 600`, i.e. 1.35 sweeps/epoch:

| normalisation | sequential R^2 | parallel R^2 | Delta R^2 |
|---|---|---|---|
| `pu` | +0.0191 | +0.0182 | **+0.0009** |
| `none` | +0.3693 | +0.3588 | **+0.0105** |
| `fixed` | +0.0236 | +0.0232 | **+0.0003** |

Counter to the expectation that removing the coupling would make parallel
updates cheaper, the uncoupled mode has the **largest** gap — by a factor of 12
over `pu`, so the effect is not marginal.

**This part needs at least one sweep per epoch, and the script says so.**  At the
repository's default budget for arch M (150 probes/epoch against a 444-parameter
full arm = 0.34 sweeps/epoch) `abl_02` reports M2/M3 as **INCONCLUSIVE**: no
configuration reaches R^2 >= 0.05, so the comparison carries no information.  M1
above is unaffected either way, because it is a property of the forward chain and
not of training.  The settings quoted above are the ones the runner uses and the
ones these numbers come from; an earlier draft of this table quoted a run at
default budget whose M2/M3 the script itself flagged as uninformative, and those
numbers (+0.0005 / +0.0172 / +0.0004) have been replaced.

---

## 8. Substrate cost model and device specifications

Field measurements per layer per block of probes (probes in a block are read in
parallel by a sensor array, so the cost is per block, not per sample):

| substrate / scheme | measurements | vs naive |
|---|---|---|
| analog circuit (memristor crossbar, one shot) | **1** | 0.02x |
| diffusion, strength-encoded, signed, dual channel | **4** | 0.06x |
| diffusion, strength-encoded, differential (thermal-like), dual | 8 | 0.13x |
| diffusion, per-particle naive readout, dual | 62 | 1.00x |

The circuit substrate wins this comparison outright: a memristor crossbar
computes the whole readout in one physical step, while the diffusion cell needs
one injection per output channel plus one for the PU denominator.  Taken with
section 2, the honest cross-substrate statement is a trade-off, not a winner:
**diffusion buys an exact, PDE-verified kernel and exact superposition at the
cost of readout measurements; the circuit buys a one-shot readout at the cost of
a behavioural device model.**

**Design specifications derived from the model** (`sub_01` Part 5):

| requirement | value |
|---|---|
| cell width (centred source) for 1e-3 relative kernel error | >= 5-12 kernel widths |
| source-to-nearest-wall distance for 1e-3 (the operative rule) | >= 2 h |
| positioner resolution for < 0.5% output error | <= 10 um |
| source-amplitude calibration | mandatory, per source, redone when `h` changes |
| **twin width accuracy** | **tested to 20% (section 6.3); a constant positional bias is absorbed, a kernel-shape error is not — validate the twin against the device, since its own residuals cannot reveal the error** |
| signed weights on a positive-only medium | differential two-cell scheme, 2x measurements |
| PU normalisation | one extra field measurement per probe block |

**Embedding requirements.**  The kernel is isotropic in *physical* space, so an
input space whose axes have unequal physical scale cannot be embedded without
changing the effective per-axis widths, unless the medium itself is anisotropic.
The reaction-diffusion analogy is misleading: the Gaussian kernel is the **heat**
kernel, so the natural realisations are diffusive (carrier, photon, thermal),
not excitable.

### 8.1 Which non-ideality actually costs accuracy (ablation C)

The architecture stores two very different analog quantities: **geometry**
(centre coordinates and widths — a voltage/bias reference, damaged additively by
gain and width mismatch) and **readout** (memristor conductances `W`, damaged
multiplicatively by drift).  `NoiseConfig` moves both together, so ablation C
separates them into a 5x5 grid and asks whether damage sensitivity follows *where
the learned capacity sits*:

* H1: drift sensitivity grows with the fraction of learned capacity in the readout
* H2: mismatch sensitivity grows with the fraction of learned capacity in the geometry

Protocol: 3 architectures x 3 arms x 3 seeds = 27 models, 675 grid cells (3 noise
realisations each), 60 epochs, equal measurement cost, **no model selection
anywhere**.  Models are loaded from ablation A's cache.

Sensitivity per arm, averaged over architectures and seeds, at the extreme corner
of each grid:

| substrate | arm | R2 clean | R2 @ geometry damage | drop | relative | R2 @ readout damage | drop | relative |
|---|---|---|---|---|---|---|---|---|
| circuit (mismatch 0.1) | geometry | +0.3244 | +0.2845 | 0.0399 | 12.3% | +0.3228 | 0.0016 | 0.5% |
| circuit | readout | +0.4552 | +0.4023 | 0.0529 | 11.6% | +0.4536 | 0.0016 | 0.3% |
| circuit | full | +0.4575 | +0.3922 | 0.0653 | 14.3% | +0.4551 | 0.0024 | 0.5% |
| diffusion (positioner 3e-4 m) | geometry | +0.3244 | +0.0382 | 0.2863 | 88.2% | +0.3235 | 0.0009 | 0.3% |
| diffusion | readout | +0.4550 | -0.6834 | 1.1384 | 250.2% | +0.4547 | 0.0003 | 0.1% |
| diffusion | full | +0.4577 | +0.0395 | 0.4182 | 91.4% | +0.4580 | -0.0003 | -0.1% |

(the readout-damage axis is memristor drift 0.02 on the circuit, injected-strength
noise 0.02 on the diffusion cell)

**Both hypotheses are falsified on BOTH substrates:**

| | circuit | diffusion |
|---|---|---|
| H1 `corr(learned weight frac, readout damage)` | -0.117 | -0.107 |
| H2 `corr(learned weight frac, geometry damage)` | +0.043 | **+0.322** |
| `corr(arch readout share, geometry damage)` | +0.129 | **+0.513** |
| arm-level agreement with H1 | 1/3 archs | 2/3 archs |

**Three findings come out of the negative result.**

1. **Replicated on two unrelated substrates: multiplicative damage to the source
   strengths / readout is negligible at 2%** — 0.1-0.5% relative on both, versus
   12-250% for geometry damage.  Drift-hardened memristors are not where the
   precision budget belongs; kernel and geometry matching are.
2. **The prediction is falsified in the *opposite* direction on the diffusion
   cell** (H2: +0.322, and +0.513 against the architectural readout share).  That
   is not noise, it is a coherent mechanism: a model whose readout has been tuned
   to precise feature locations is *more* brittle when those locations move, while
   a model with a barely-trained readout loses the same features without a tuned
   readout to invalidate.  So "spend precision where the layer learns" is not just
   unsupported — for positioner error it is backwards.
3. **The two grids are not matched in severity**, so the cross-substrate magnitude
   comparison above is not apples to apples: 10% gain mismatch costs 12-14%,
   whereas 300 um of positioner error costs 88-250%.  What replicates is the
   *ordering* within each substrate (geometry damage >> readout damage), not the
   absolute numbers.

A 1-seed / 20-epoch smoke run of the diffusion cell once suggested H2 held
(r = -0.381).  At full scale the same correlation is **+0.322**, i.e. the opposite
sign: that earlier hint was noise, as suspected.

**Consequence for the training rule of section 6.**  Strength noise damages the
calibrated strengths directly, yet it is the harmless axis on both substrates
(0.1-0.3%).  Section 6.2 reached the same conclusion from the other direction:
the calibrated arm loses at most 0.07 R^2 across two decades of sensor noise while
the search arms lose 0.32.  Calibration is therefore not fragile to the
non-idealities that actually exist on these two substrates.

---

## 9. Caveats that qualify these numbers

Things a reader needs in order to use the numbers above correctly. Each is measured, and each
qualifies a figure a reader would otherwise take at face value.

**The CIFAR line's numbers are single draws from an unseeded process.** All seven `cifar_*`
scripts seeded the data split and never the torch initialisation. Re-run with `SEED = 0`, on the
same splits:

| script | previously published | **seeded** | move |
|---|---|---|---|
| `cifar_01` baseline | 37.3% | **30.7%** | **-6.6** |
| `cifar_02` regularized | 39.3% | **36.0%** | -3.3 |
| `cifar_03` 3-layer | 38.0% | **32.0%** | **-6.0** |
| `cifar_04` large data | 36.7% | **36.0%** | -0.7 |
| `cifar_05` MLP | 34.0% | **28.0%** | **-6.0** |
| `cifar_06` MLP arm | 38.7% | **42.7%** | **+4.0** |
| `cifar_06` CNN arm | 61.3% | **54.7%** | **-6.6** |

**The moves go in both directions**, so this is variance rather than a bias, and the peak epochs
move too (`cifar_03` peaked at epoch 620 in one run and 220 in the other). **Every architecture
comparison that line produced was therefore between arms whose own seed sensitivity exceeds the
differences it reports.** All seven scripts now seed the model initialisation and print `SEED=` in
their reports, so a number and the seed that produced it are recorded together.

**Both sides of that table are tracked, so either column can be checked.** The published values
are `outputs/{cifar_01_baseline,cifar_02_regularized,cifar_03_3layer_reg,cifar_04_large_data,cifar_05_mlp_baseline,cifar_06_three_way_compare}_report_PRE_SEED.txt`,
and the seeded ones are the same names without the suffix. Each script writes the name that
matches it, so a report can be found from the script that produced it.

**And three scripts freeze the reference arm as a literal.** `cifar_04`, `cifar_05` and `cifar_06`
each hard-code the RBF result as `peak test 39.3%, peak train 53.6% (from E21)` rather than
re-running it — so those comparisons set a reference that cannot move against arms that move by up
to 6.6 points, and the frozen value is a favourable draw, since the seeded run gives 36.0%. The
literals are deliberately left in place, because editing them would silently change a comparison's
input; `cifar_06`'s own report prints the discrepancy.

**"Ideal" is not noise-free on the circuit path.** `bjt_gaussian_kernel` adds a `sqrt(|z|)*1e-6`
perturbation whenever `noise_std is None`, which is how `simulator/layer_nd.py` calls it — so
`NoiseConfig.set_level("none")` still perturbs the field, and a digital-physical agreement figure
of `1e-7` is float32 rounding plus that guard rather than a statement of physical fidelity.

**The geometry arm loses under equal cost, and the share correlation is weak.** With the capacity
split measured at equal measurement cost, the readout wins; the ordering geometry < readout is
robust across seeds and epochs, but the correlation between geometry share and geometry-arm
performance is `-0.53` at 3 seeds / 80 epochs, not the `-0.96` a 2-seed / 40-epoch run gave. The
ordering is the result; the share correlation is not.

**The independence claim holds only without normalisation.** With `norm_mode="none"` the sources'
contributions are exactly independent; under the partition-of-unity normalisation the
classification path uses, the measured coupling is `0.063-0.261`. Section 7 states this, and the
parallel-hardware mapping in `THEORY.md` section 0.4 is conditioned on it.

**Two literature notes, since they bound what may be claimed as new.** Multi-layer RBF networks as
the network itself are published (Wurzberger & Schwenker, *Entropy* 26(5):368, 2024), and Park &
Sandberg's result that a single hidden layer of RBF units is already a universal approximator
predates this work by decades. **Architecture is therefore not where this project's contribution
lies** — the operators are, and `THEORY.md` section 4 states them.


---

## 10. What this project is, and is not

**Is:** a substrate-portability and cost study of a particle-source network, with
a verified second physical instantiation, a measured design rule (section 6), a
quantified substrate cost model (section 8), and two clean negative results
(sections 5 and 7).

**Is not:** a new architecture.  Radial basis function networks date to 1988,
analog VLSI RBF chips to 1992 (Anderson, Platt & Kirk, NIPS), and deep RBF
networks to at least 2024.  The four-tier training taxonomy mirrors the one now
standard in the physical-neural-network literature (arXiv:2406.03372).

**Scale limit.**  Everything above is measured on 4D regression with 52
particles, 256 training samples and 40 epochs.  Every conclusion should be read
as holding at that scale until scaled up.

---

## 11. Open items

> **This index was still incomplete, which is the defect its own rebuild was meant to
> fix.**  Three genuinely-open items were being stated elsewhere in this document and
> were absent here, so a reader choosing the next experiment from this list would not
> have seen them.  They are now items 8-10 below.
>
> **And the list is no longer the only place the work is tracked.**  An extension audit
> (see `extensions/multigraph/BASE_CORRECTIONS.md`) holds a line-by-line work list for
> this document, including three claims it retracts outright — item 2's "not
> constructible" below, THEORY 7.3's "5x", and 6.11's "survived three counterexamples".

1. ~~Scaling law of the measurement budget.~~  **Done** — sections 6.1 and 6.8.  The
   rate result is the primary evidence, the share axis is now sampled at all five
   architectures, and section 6.8 identifies the quantity that actually controls
   the calibration (readout parameters vs training samples) as distinct from the
   quantity that controls the geometry arm (geometry share).  **The tail below is
   Settled (extension 12.9-12.13), and the answer is conditional on one constant.**
   At the default `closed_form_lam=1e-3` the low-budget margin goes from `+0.41` at
   52 particles to `-0.02` at 208, in two independent families, with the budget held
   fixed.  At `lam="auto"` it is **positive at every size and rises to `+0.60`**.
   **`1e-3` is the only value on a five-point grid that collapses it.**  The question
   as originally phrased could not be asked, because the comparison window itself
   scales with the architecture — `effective_budget` derives the per-epoch budget
   from the network size, so raising the particle count moves the window past the
   3 000-measurement grid point.  Evidence: extension `FINDINGS.md` 12.9-12.13,
   `outputs/ext_25_budget_matched.txt`, `outputs/ext_26_base_family.txt`,
   `outputs/ext_27_nonmonotone.txt`.
2. ~~Equal-parameter-count comparison.~~  **DONE -- and the checkbox was stale, which is why
   this line no longer reads as open.**  **CONSTRUCTIBLE, contrary to what this item
   said.**  A layer with input dim `D`, output width `V` and `N` particles stores `(D+1)*N`
   geometry numbers and `V*N` readout numbers, so its geometry share is `(D+1)/(D+1+V)` —
   **`V` is a knob on the share**, which is the same knob the box in section 5 already uses
   when it notes that enabling the second channel moves F2 from an 81% share to 68%.
   Holding the total fixed is then an integer problem, solved exactly with two layers
   (`T = (D+1+V)*N1 + 3*N2`), and five points were built at **the same total of 312**
   across a share sweep of **83% to 26%**.  **What that bought:** at a constant total the
   search arms move by up to `0.319` across the axis while the calibrated arm moves by
   `0.091`, and the calibration's margin **grows as the allocation worsens** — so **the
   calibration is a substitute for a good allocation**, and the parameter-count rule of
   section 5 holds for the calibrated arm rather than for the search arms.  Evidence:
   extension `FINDINGS.md` 12.15, `outputs/ext_30_equal_params.txt`.
3. ~~Statistical strength.~~  **DONE -- and this checkbox was stale for the same reason as
   2's and 5's: the item's own text said so and the number did not.**  abl_01 is 3 seeds, abl_03 is 3 seeds on
   both substrates, and abl_04, abl_05 and abl_06 (7 tasks x 2 architectures = 14
   cells) are all 3 seeds.  The seed upgrade was not cosmetic: it **retracted two
   claims** made at 2 seeds — the "40% of the margin is gone at 20% twin error"
   warning in section 6.3, and the "shift margin is larger than on 4d" claim in
   section 6.7 — and it **produced the counterexample** to the stronger claim
   (section 6.7).
4. ~~Epoch sensitivity.~~  **Done — section 6.10.**  The grid shows the capacity gap
   is finite-time: it closes to zero and inverts by 160 epochs at budget 1 200.
   Endpoint comparisons must state their epoch count, and section 5 is now labelled
   accordingly.
5. ~~Reservoir variant.~~  **DONE -- tested, and the checkbox was stale.**  **Tested** (extension 12.14).  A frozen geometry with a single
   closed-form solve beats the geometry arm, at zero probe cost, in most cells (9 of 14)
   — so "only the readout is trainable" is not by itself disqualifying, and the movable
   centres are not earning their keep because a frozen geometry is too weak.  What it
   loses to is search PLUS solve (13 of 14), and the reason is **model selection**: a
   reservoir's one knob is redrawing the geometry, and the redraw has to be picked, which
   on a quarter-of-training slice is **worse than not picking in about half the cells**
   (doubling the slice helps, `+0.0204` mean).  **The gap is a selector, not movable
   centres** — and the same selector failure appears in the operator suite
   (`ext_21`/`ext_22`).  Evidence: extension `FINDINGS.md` 12.14,
   `outputs/ext_29_reservoir.txt`.
6. ~~Regularising the calibration.~~  **Done — sections 6.6-6.7.**  A relative
   ridge plus a *symmetric* out-of-sample gate cut the worst-case hybrid R^2 from
   -0.8262 to -0.0355 while preserving every decisive win; the calibration now wins
   9 of 14 cells over seven tasks.  What remains open: (a) the acceptance rate is
   0-4 of 80, so a cheaper schedule (calibrate every k epochs, or only when the
   geometry has moved enough) would cut cost with no loss; (b) the gate validates
   in-distribution, and although the `shift` task showed the advantage transfers,
   **the counterexample to the *stronger* claim exists and is recorded in section 6.7**:
   `hetero`/G, where one calibration was accepted, improved the arm, and the arm still
   lost to `full` (1 accepted of 120 cells).  **It is not a counterexample to the
   guarantee**, which is about not calibrating rather than about other arms.  On the
   guarantee itself, an extension audit located the gate's **decision boundary** — where
   the accept rate is slice-size-dependent by `0.48` — and then isolated the calibration
   from its own trajectory: **55 of 116 deployed calibrations left the arm worse than
   immediately before the write** (extensions 12.18, 12.19; see the retraction at the end
   of 6.11).  (c) the five losing
   cells are now uninformative about the mechanism, because the gate disables the
   calibration there.  Sections 6.9 and 6.11 close (a) and report three failed
   attempts at (b) — and a fourth that succeeded.
7. ~~Task sweep coverage.~~  **CLOSED BY RESTATEMENT -- the checkbox was stale.**  **Restated, because the counts here were stale and this item
   contradicted item 3 above.**  It is **3 seeds**, not 2, and at 9 seeds the headline is
   **8 of 14**, not nine.  What is genuinely still narrow is not the coverage but the
   reading: **the five losses all reduce to the geometry arm, so they bound the risk
   without discriminating between the mechanisms.**  That is a limitation of the evidence,
   not a gap in it, and item 3 already covers the seeds.
8. ~~Method C — the analog gradient chain.~~  **CLOSED BY REMOVAL, which is one of the two
   branches this item itself offered.  The checkbox said open only because the METHOD is
   unimplemented -- and that was never the test.**  The either/or was: **"Either implement and
   measure this method, or remove it from the feature list."**  The removal branch was taken:

   * **no feature list offers it any more.**  `THEORY.md`'s note states the decision, and the
     README's feature bullets in both languages carry an inline correction saying the
     gradient-estimation half is dead code.
   * **`THEORY.md` still DESCRIBES it**, as unimplemented theory -- which is what a theory
     document is for, and the description is accurate: it says the method is "not implemented"
     and that the functions "are defined and re-exported".
   * **AND THE CODE IS STILL IN THE TREE**, which is the fact this item never recorded and which
     a reader would otherwise assume had been deleted.  All eight functions remain in
     `physical_simulator/components.py`, kept deliberately as the reference implementation of a
     method the theory document still describes.  **Deleting the region wholesale would break
     the live chain**, because `analog_gilbert_multiplier` is ALSO used by `gilbert_square`,
     which is on the working path.  The declaration, the measured call-site census and that
     sharing are recorded AT THE DEFINITION SITE, in the code, where a reader of the code looks.
     The census, measured rather than described: **five functions have zero call sites**
     (`analog_loss_readout`, `analog_capacitor_integrator`, `analog_center_gradient`,
     `analog_width_gradient`, `analog_weight_gradient`) and **three are called only from inside
     the island** (`analog_gilbert_multiplier` five times, `analog_gradient_mac` once,
     `analog_adc` twice).

   **AND THE IMPLEMENT BRANCH IS NO LONGER MERELY UNTAKEN -- IT IS LARGELY UNNECESSARY**, for a
   reason this project found itself: **the source strengths should be CALIBRATED in closed form,
   not differentiated** (§6).  A gradient chain would therefore be needed only for the
   **geometry** -- the smaller half of the parameter space, and the half the operator already
   places from field readings.

   **So this is a decision with a reason rather than a gap with an excuse**, and if the geometry
   half ever does need exact gradients, the eight functions are still there to build on.
   Evidence: the declaration in `physical_simulator/components.py`, `THEORY.md` Method C.
9. ~~The "5x better" direct residual.~~  **Refuted under control, with its sign reversed —
   `THEORY.md` section 3.2 and extension 12.17.**  It had been stated here only in the
   retraction list and never in this index, which is the gap this preamble now names.
   Re-measured under control the sign reverses (direct `-0.0284` against proto `+0.0069`),
   so `residual_mode="proto"` is the default because the alternative is **measured harmful**.
10. ~~A task near the do-no-harm gate's decision boundary.~~  **Done — extension 12.18 and
   12.19; see the retraction at the end of 6.11.**  Section 6.11 said the third
   counterexample attempt "did not answer its own question" because that task "was not
   constructed".  It is now constructed — the knob is the **validation-slice size**, not
   the task's noise — and the isolation it needed found **55 of 116 deployed calibrations
   leaving the arm worse** than immediately before the write.

**And one item that is a defect rather than a question, now fixed:** `training/himal_nd.py`
took no `seed` and never seeded the global RNG, so every number published from it was one
draw from an unseeded process.  It now takes a `seed` and seeds once at the top of `train`.

11. **`outputs/sub_01_diffusion_verify.txt` does not reproduce.**  The suite is the base
   version's verification, it is TRACKED, and `FINDINGS.md` sections 2, 3 and 8 rest on it — but
   **three of its error-floor lines change on every run** (the analog circuit's and the two
   diffusion cells'), by roughly a factor of two, while the script seeds the global RNG itself.

   **THE CAUSE IS NOT ISOLATED AND TWO HYPOTHESES HAVE BEEN REFUTED.**  `PYTHONHASHSEED` was
   tested because a set of STRINGS could order the computation, and two runs with it fixed still
   differed.  An unseeded `np.random.default_rng()` in the diffusion sensor chain was found and
   fixed — **and it is not on the path the suite takes**, so it was not the cause either.

   **THE MAGNITUDE BOUNDS THE DAMAGE.**  The movement is in the 5th to 7th digit while the report
   quotes four decimals, so no published figure changes.  **But a re-run does not reproduce the
   artifact**, which for a tracked file that three sections cite is a defect rather than a
   nicety — and it is recorded as the likely shared cause of the extension's defect 58, whose
   healthy threshold for `log_rms` (~1e-6) sits AT that magnitude.

   Evidence: extension `FINDINGS.md` 12.32 and defect 58.

---
12. **The CIFAR comparisons have not been recomputed under the fixed seeding.**
   Section 9's new entry records that every number in that line was a single draw from an
   unseeded process, that the seed moves those numbers by up to 6.6 points in both
   directions, and that three scripts freeze the RBF reference as a literal.  **The scripts
   are fixed -- they seed and they record their seed -- but NO comparison has been re-run
   under the fix**, so the line's published rankings remain the ones the unseeded draws
   produced.  Re-running the comparisons is a project, not a cleanup: it needs several
   seeds per arm, and the literals would have to become real runs first.

## 12. Artifacts

The evidence set is exactly what `.gitignore` un-ignores: `outputs/abl_*.txt`,
`outputs/abl_*.png` and `outputs/sub_01_*`.  Everything else under `outputs/` is
local (model caches, session scratch, and the legacy `eNN_*` reports, which are the
live outputs of the older experiment scripts and must not be moved or deleted).

**Code**

| file | what it holds |
|---|---|
| `experiments/sub_01_diffusion_verify.py` | substrate verification suite (sections 2, 3, 8) |
| `substrate_diffusion/` | the diffusion-cell substrate and its primitives |
| `experiments/abl_common.py` | substrate dispatch, masked FD trainer, closed-form source-strength calibration, the symmetric gate, trajectory recording |
| `experiments/abl_01_capacity_split.py` | ablation A: both measurement protocols, scaling curves, share axis |
| `experiments/abl_02_locality.py` | ablation B (section 7) |
| `experiments/abl_03_mismatch_sensitivity.py` | ablation C: damage axes (section 8.1) |
| `experiments/abl_04_calibration_robustness.py` | ablation D: device non-idealities (section 6.2) |
| `experiments/abl_05_twin_error.py` | ablation E: a wrong digital twin (section 6.3) |
| `experiments/abl_06_task_sweep.py` | ablation F: the task sweep, cache-resumable (sections 6.5-6.7) |
| `experiments/abl_07_convergence.py` | ablation G: convergence grid, schedule, gate sample size (sections 6.9-6.11) |

**Reports and figures present on disk**

| file | section |
|---|---|
| `outputs/abl_01_capacity_split.txt` | 5, 6.1 (headline + scaling, 3 seeds / 80 epochs) |
| `outputs/abl_01_capacity_split_equal_sweeps.txt` | 5 (equal-coverage protocol) |
| `outputs/abl_01_capacity_split_diffusion.txt` | 2 (second substrate) |
| `outputs/abl_01_capacity_split_diffusion_shareaxis.txt` | 5, 6.8 (five architectures, share axis) |
| `outputs/abl_01_scaling_diffusion_shareaxis_plot.png` | 6.1, 6.8 (the scaling figure) |
| `outputs/abl_01_capacity_split_diffusion_shareaxis_plot.png` | 5 (the capacity-split figure) |
| `outputs/abl_02_locality.txt` | 7 (M1 coupling, M2/M3 parallel price) |
| `outputs/abl_02_coupling.png`, `outputs/abl_02_parallel.png` | 7 (the two figures) |
| `outputs/abl_03_mismatch.txt`, `outputs/abl_03_mismatch_diffusion.txt` | 8.1 (both substrates) |
| `outputs/abl_04_calibration_robustness.txt` | 6.2 (3 seeds) |
| `outputs/abl_05_twin_error.txt` | 6.3 (3 seeds) |
| `outputs/abl_06_task_sweep.txt` | 6.5 (before the gate) |
| `outputs/abl_06_task_sweep_ridgefix.txt`, `_guardfloor.txt`, `_fitval.txt` | 6.6 (the three repair attempts; the first two are the evidence that they produced identical numbers) |
| `outputs/abl_06_task_sweep_symgate.txt` | 6.7 (symmetric gate, 2 seeds) |
| `outputs/abl_06_task_sweep_s3.txt` | 6.7, 11.3 (**final**: seven tasks, 3 seeds) |
| `outputs/abl_06_task_sweep_plot.png` | 6.5 (per-task curves) |
| `outputs/abl_07_convergence.txt`, `_plot.png` | 6.9-6.11 |
| `outputs/sub_01_diffusion_verify.txt`, `_plot.png` | 2, 3, 8 |

**Missing, and how to regenerate**

| file | why it is missing | command |
|---|---|---|
| the `abl_01` plots without a `_shareaxis` suffix | destroyed by an early `--tag` bug (the tag was added to the report name but not the plot names), then superseded by the share-axis figures above | regenerate with any untagged `abl_01` run, or use the share-axis figures |

Nothing else is missing: `outputs/abl_02_locality.txt` was regenerated with
`--seeds 1 --epochs 40 --budget 600` (see section 7 for why that budget matters),
and `experiments/abl_run_all.py` reproduces the whole set.

**Guarding this evidence.**  `python experiments/abl_run_all.py --quick` runs all
eight stages end to end in ~15 min as a pipeline smoke test (its numbers are *not*
the ones above), and `python experiments/sub_01_diffusion_verify.py --assert`
enforces the physical thresholds that sections 2, 3 and 8 rest on in 3 seconds,
exiting non-zero if one breaks.  Both run in CI.  Reproducing the canonical numbers
above takes ~6.5 h (`abl_run_all.py --list` prints the per-stage settings).
