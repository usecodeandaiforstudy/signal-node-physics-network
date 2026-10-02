# Multi-graph readout: design memo

**Status:** design only. Nothing here is implemented, and nothing here changes any
conclusion of the base version. `FINDINGS.md`, `experiments/abl_*.py` and
`outputs/abl_*` are read-only inputs to this document, not outputs of it.

**Scope rule for everything that follows:** any implementation lives in
`extensions/` and depends on the base version **read-only** (`substrate_diffusion`,
`experiments/abl_common.py`). The base version's files are not edited. If a change
to the base version ever looks necessary, that is a finding to report, not an edit
to make.

## 0. Notation — read this before any formula

The combination symbols and the physics symbols have collided historically, and two of
the collisions are dangerous in code (`k` and `D` each mean two different things in the
base version's own material).  This memo uses:

| symbol | meaning | where it comes from |
|---|---|---|
| `n` | **input** dimension (the phase space of a graph) | the proposal's "n-dimensional input" |
| `m` | **output** dimension (number of channels) | the proposal |
| `k` | number of **graphs** | the proposal |
| `s` | number of graphs whose readings are **combined into one channel** (group size) | the proposal's "t graphs per output" — renamed, see below |
| `v` | index of an **output channel**, `v = 1..m` | the code: `out_v = sum_j W_vj z_j` |
| `i` | index of a **graph**, `i = 1..k` | this memo |
| `N` | particles per graph | the code: `n_particles` |
| `C` | the **(m x k) combination coefficients**, `C[v, i]` | the proposal's `c_tk` |
| `h_j` | **kernel width** of particle `j` — a length | the code: `layer.h`, `h_scale` |
| `t_j` | **diffusion time** of particle `j`, `t_j = h_j^2 / (4 k_kern D_diff)` | the code: `pulse_time()`, variable `t` |
| `g_j` | **amplitude prefactor** of particle `j`, `g_j = 1 / (4 pi D_diff t_j)^(dim/2)` | the code: `green_prefactor()` |
| `k_kern` | **kernel sharpness constant** — the code's `k` | the code: `pulse_time(h, k, D)` |
| `D_diff` | **diffusion coefficient** — the code's `sub.D` | the code |
| `dim` | spatial dimension of the **medium** — the code's `sub.dim` | the code |

**Two warnings, because these are the ones that will cause bugs:**

1. **`t` is a time, never a count.**  The letter `t` in this memo and in the code is
   `t_j`, a diffusion time in seconds.  The proposal originally wrote "t graphs per
   output" for the group size; that quantity is **`s`** here.
2. **`k` and `D` are each overloaded somewhere in the base material.**  Two separate
   facts, and it is worth keeping them apart:
   * **`D` is overloaded inside the base `FINDINGS.md` itself**: it is the *input
     dimension* in `D+1` and `D*N` (sections 2, 5, 11) and the *diffusion coefficient*
     in `g_j = (4 pi D t_j)^(dim/2)` and `t_j = h_j^2/(4kD)` (sections 4, 6.3).  A
     notation note has been added to section 4 of that document; the fix here is that
     this memo writes `n` for the input dimension and `D_diff` for the diffusion
     coefficient.
   * **`k` is *not* overloaded inside `FINDINGS.md`** — there it only ever means the
     kernel sharpness (`k=0.5`, `k=0.3`, `t_j = h_j^2/(4kD)`).  The collision for `k`
     appears only when the *proposal's* "k graphs" meets the *code's*
     `pulse_time(h_phys, k, sub.D)`.  In this memo `k` is the number of graphs and the
     physics symbol is written `k_kern`.
   When writing code, prefer the attribute names (`sub.D`, `sub.dim`) over any
   single-letter alias.

---

## 1. What the base version actually does (verified in the code)

One layer is one **graph**: a set of node positions `centers` (D,N) and widths `h`
(N) living in the input space, plus, for each output channel `v`, a vector of
scaling coefficients `W_gauss[v]` (N):

```
out_v(x) = sum_j W_gauss[v, j] * z_j(x),      z_j = exp(-k * ||x - c_j||^2 / h_j^2)
```

* `simulator/layer_nd.py:180-231` — distance → square → Kirchhoff sum → BJT
  exponential → (optional PU) → memristor MVM.
* `simulator/network_nd.py:137-153` — two crossbars per layer fused as
  `f = out_g + out_a`, then **wire-direct + zero-padded residual** between layers.
* Reading the field anywhere is an exact linear superposition of the sources:
  measured error 2.86e-16 (FINDINGS section 3).

So, precisely:

| statement | verdict |
|---|---|
| a layer is a graph of source nodes in a phase space | yes |
| the crossbar holds **m sets of scaling coefficients over one shared graph** | yes |
| a scaling coefficient IS a source's excitation strength | yes |
| the reading at any position is the linear superposition of the sources | yes, exactly |
| the geometry half (centers + h) is already matrix-free | yes |
| the readout half is matrix-free only when V = 1 per layer | yes — F/F2, where the crossbar is a row vector |
| a node stores coordinates and scaling coefficients | **incomplete: it also stores `h`** |

`h` is not bookkeeping. FINDINGS section 6.3 shows the twin's **width** error is the
dangerous mismatch (it changes kernel *shape*, which the linear solve cannot absorb)
while a constant positional offset is nearly free. Any multi-graph design must say
which graph owns each `h`, and whether `h` is shared within a group.

## 2. The pivot: "matrix" means two different things

1. **Notation.** An (m,N) array is just *m* parameter vectors written together. In
   this sense "matrix-free" already holds: m single-output graphs sharing one graph
   of nodes carry exactly the same data, bit for bit.
2. **Device.** The crossbar is the physical network that performs m·N multiplies and
   N-term summations. Changing notation does not change it.

Consequence for this whole memo: a redesign changes **the shape of the summing
network and the parameter sharing**, and can change **the parameter count**. It does
not make the m·N multiply-accumulates disappear. That is not a reason to reject it —
the parameter count and the sharing are exactly the levers that matter (section 5) —
but it is the reason the goal should be stated as *"re-parameterise the readout"*,
not *"remove the matrix"*.

## 3. The proposal, stated precisely (after the author's clarification)

* **Input is not reduced.** Every graph receives the full n-dimensional input. No
  input-space projection, no dimensionality reduction anywhere on the input side.
* Each graph outputs **one scalar** — reading its own field at its own coordinate(s).
* There are **k graphs**. Output channel `v` is formed by **combining** the readings
  of a group of **s** of them.
* The output is **m-dimensional, assembled from low-dimensional pieces**.
* **The combination rule is itself a design variable**, not necessarily a sum.

The last point is what makes this a research programme rather than a re-indexing, so
the design space is enumerated next.

## 4. The combination operator: options and their mathematical consequences

Two structural choices are independent: do the graphs **share node positions**, and
is the combination **linear in the readings**.

| combination | shared geometry | parameters | expressiveness | readout linear in strengths? |
|---|---|---|---|---|
| **selection / concatenation** (channel `v` = graph `v`'s reading) | yes | m·N (= base) | identical to the base matrix readout | **yes** |
| selection / concatenation | no | geom × k, readout m·N | k independent 1-D RBF nets sharing input | **yes** |
| **linear weights** `C[v, i]` | yes | k·N + m·k | `W = C · W_graphs`, rank ≤ min(m,k) | **yes** |
| linear weights | no | geom × k, readout k·N + m·k | sum of k independent RBF expansions | **yes** |
| **fixed code** (Hadamard, random, ±1) | yes | k·N + 0 | rank-k readout with a *frozen* decoder; no learned combination | **yes** |
| **products / interactions** (e.g. g_i·g_j) | either | k·N + 0 | k → m by monomials; captures interactions the linear readout cannot | **no** |
| **gating / attention** (combination depends on x) | either | + gate params | readout becomes input-dependent; the effective W varies per sample | **no** |

The critical row distinction is the last column. It is not a style preference:

> **The base version's headline result is that the source strengths should be
> CALIBRATED IN CLOSED FORM, not searched** (FINDINGS section 6): a linear solve, zero
> optimisation, worth 15-30x fewer measurements, robust at 8/8 device levels and
> against a 20% wrong twin.
>
> That operator exists **only because the readout is linear in the strengths.** Any
> combination that is nonlinear in the readings makes fitting the strengths a
> nonlinear problem again, and trades away that advantage. It may buy expressiveness
> worth more than the advantage — but the trade must be made deliberately, and
> measured, not discovered later.

A second, equally sharp consequence of `k`:

> The k scalar readings are a **bottleneck**. If the target needs more than k
> independent linear functionals of the particle activations, no choice of
> combination rule can recover the lost information — a nonlinear combination uses
> the k numbers better, it does not create more of them. So `k ≥ rank needed` is a
> hard floor, and `k < m` is a deliberate low-rank constraint, not a free lunch.

### 4.1 The kernel width `h` — the parameter that constrains everything above

**"Shared geometry" in the table above means shared CENTERS *and* shared `h`.**  That
is not a phrasing detail; the algebra in the `W = C · W_graphs` rows holds only if the
graphs share the same **feature matrix** `Z`, and

```
z_j(x) = exp( -||x - c_j||^2 / h_j^2 )        depends on h_j, not only on c_j
```

so two graphs that share centers but differ in `h` have different `Z`, are *different*
graphs to the algebra, and fall into the "not shared" rows. Half-sharing is the
configuration that is easiest to build by accident and hardest to reason about.

**`h` sets two independent things, not one.**  In the diffusion layer
(`substrate_diffusion/layer.py:249-251`), writing the code's own names in the middle
column to avoid the `k`/`D` overload:

```
t_j = h_j^2 / (4 * k_kern * D_diff)          code: pulse_time(h_phys, k, sub.D)
g_j = 1 / (4 * pi * D_diff * t_j)^(dim/2)    code: green_prefactor(t, sub.D, sub.dim)
z_j = exp( -r^2 / (4 * D_diff * t_j) )       the exponent uses that source's own t_j
```

1. **shape** — `t_j` in the exponent;
2. **amplitude** — `g_j`, and the injected strength for weight `W_ij` is
   `q_ij = W_ij * g_j`.

Consequences, each of which has been measured on the base version:

* **The amplitude calibration is per source and is invalidated by any width change.**
  The base version calibrates `q_ij = W_ij g_j` and lands at `5.92e-08` transfer error;
  skipping it gives `9.98` — **eight orders of magnitude** (FINDINGS section 4). A
  multi-graph design therefore has **one calibration vector per graph** unless the
  graphs share `h`. Geometry training that touches `h` invalidates it on every step
  that moves a width, which is the real cost the diffusion substrate adds.
* **Width is the dangerous mismatch.**  Of the twin errors tested, a width error
  changes kernel *shape*, which the linear solve cannot absorb, while a constant
  positional offset is nearly free (section 6.3). At a fixed geometry the effect is
  large — the twin's own score *improves* (+0.4533 → +0.5172 at 20% error) while the
  device deliverable collapses (+0.4533 → +0.2747). In a full training loop with the
  gate the final numbers stay flat, because the gate refuses the bad calibrations — so
  **the calibration acceptance rate, not the final R², is the early-warning signal**
  for width problems. Multiplying the number of widths (one `h` vector per graph)
  multiplies the exposure to exactly the non-ideality that is hardest to detect.
* **Device specifications scale with `h`** (section 8): source-to-nearest-wall ≥ 2h,
  cell width ≥ 5-12 h, positioner resolution ≤ 10 µm. Because `t ∝ h²`, a spread of
  `h` across graphs is a **squared** spread in pulse time / sampling time: a 10x width
  spread demands a 100x timing (or pulse-duration) dynamic range from the same readout
  chain.
* **`h` is cheap in parameters, expensive in consequence**: geometry is `n·N + N`, so a
  width costs **1 parameter per particle** against `D` for a centre — yet it is the
  parameter that sets the amplitude calibration, the kernel shape, the boundary rule
  and the timing budget.
* **In-field (pre-sensor) combination is available only to the LINEAR rows.**  The
  medium superposes fields exactly (2.86e-16, section 3), so combining the *s* graphs
  before the sensor — the cost win noted in section 7.5 — is legal for sums of
  readings. A products/gating combination cannot be done in the field at all: a linear
  medium cannot multiply two fields, so those rows must sense each graph separately and
  pay *k* measurements. Section 7.5 is hereby qualified: the one-measurement trick
  buys combination only for the rows whose last column is "yes".

**A dichotomy worth deciding explicitly, because it determines whether "no matrix" is
literally true:**

| realization | what is stored | what is measured | matrix present? |
|---|---|---|---|
| **(a) fold the combination into the injections** — one field, one injection set per channel, `q_tj = (Σ_k A_tk w_kj) · g_j` | `k·N + m·k` | **1 reading per channel** (the base crossbar, unchanged) | hardware: yes (an m-row crossbar). Notation: no — the m rows are *generated* from k graphs |
| **(b) k separate single-output cells** — read each graph, combine after sensing | `k·N` | **k readings per block** | **nowhere.** No `m×k` coefficients exist at all |

(a) is the parameter-economy win with the base hardware; (b) is the literal no-matrix
win at k times the readout cost. They are different claims and must not be conflated in
a write-up: **(a) is "matrix-free notation", (b) is "matrix-free device"**.

**Recommendation for the first experiment: share centers AND `h`.**  That keeps the
`W = C · W_graphs` algebra valid, keeps in-field combination available, keeps one
amplitude-calibration vector instead of k, and minimizes exposure to the width
mismatch. Per-graph widths are a *later*, separate variant — and the variant worth
measuring second, because multi-scale receptive fields are the most plausible reason to
want them at all.

## 5. Parameter economics, with the base version's real numbers

Two quantities govern everything the base version measured:

* **readout parameters / training samples** controls whether closed-form calibration
  works at all (FINDINGS section 6.8). F2 sits at `52/256 = 0.20x` and calibrates
  safely; W sits at `560/256 = 2.19x` and is the one architecture where calibration
  fails.
* **geometry parameters** are the *expensive* half in measurement terms: reaching
  R^2 = 0.30 costs the geometry arm ~100k measurements while the calibrated arm gets
  there in under 3k, and the geometry only closes the gap at 160 epochs (sections 6.1,
  6.10).

Illustration — one layer, n=4, N=40, n_train=256, eight output channels (m=8):

| design | geometry | readout | readout/samples | note |
|---|---|---|---|---|
| base (one graph, m=8 rows) | 200 | 320 | **1.25x** | already in the warning zone |
| k=8 graphs, shared geometry, selection | 200 | 320 | 1.25x | identical to base, as it must be |
| k=4, shared, linear combination | 200 | 160 + 32 = **192** | **0.75x** | 40% fewer readout params, rank ≤ 4 |
| k=4, shared, fixed code | 200 | 160 | 0.63x | cheapest, decoder frozen |
| k=8, **distinct** geometry, selection | **1600** | 320 | 1.25x | geometry × 8 — the measurement budget explodes |

The asymmetry is the practical bottom line:

* **not sharing geometry multiplies the half that measurement budget cannot afford**;
* **sharing geometry with k < m is the configuration that actually helps**, because it
  is the only one that reduces the readout-to-sample ratio, i.e. moves *away* from the
  measured failure boundary.

## 6. Hard constraints any implementation must satisfy

**C1 — combination happens at or after the readout, never inside the medium.**
Field superposition is a physical law on both substrates (2.86e-16, verified against a
PDE solve). A nonlinearity applied in the field domain destroys it. The existing
`sub_01_diffusion_verify.py --assert` enforces superposition, so a violation will be
caught — which is the desired behaviour.

**C2 — any combination nonlinear in the strengths forfeits closed-form calibration.**
State this in the design, measure the trade, keep the linear option available.

**C3 — track `readout params / n_train`, not the geometry share.** This is the
measured control variable (section 6.8), and it is the quantity the design moves.

**C4 — the extension must degenerate exactly to the base version** under
`k = m`, shared geometry, combination = selection, `A = I`. This is the single most
valuable test in the whole programme, because it is a bit-level correctness anchor:
if the special case does not reproduce the base numbers, the new code is wrong and
nothing else it reports can be trusted.

## 7. Open questions, to settle before coding

1. **Where do the combination coefficients come from** if the combination is linear —
   learned by finite differences, calibrated in closed form (they are linear in the
   outputs, so a second-stage solve is available), or fixed by construction? Note that
   a learned `A` is itself an m×k matrix, so "matrix-free" must not be claimed while
   `A` is free.
2. **Shared or distinct geometry per graph**, and for which of the two halves
   (positions, widths) separately?  C4 requires shared positions in the degenerate case,
   and section 4.1 shows the two halves are **not** interchangeable: sharing positions
   but not `h` silently invalidates the `W = C · W_graphs` algebra, because the feature
   matrix depends on `h`.  Recommendation: share both, and treat per-graph widths as a
   deliberate second variant.
3. **Who owns `h`** — per graph, or shared within a group?  Section 6.3 says `h` error
   is the dangerous twin error and section 4.1 lists what `h` controls (kernel shape,
   amplitude calibration, boundary rule, timing budget).  Decision for the first
   experiment: **shared**, so there is one amplitude-calibration vector instead of k and
   the width-mismatch exposure is not multiplied.
4. **What is the readout coordinate of each graph?** A graph may be read at one point
   or at several; the base version reads each layer's field at the next layer's
   layer of particles. This choice fixes what "1-D output" means physically.
5. **Does the combination exploit pre-sensor superposition?** On the diffusion cell,
   combining the *s* graphs *in the field* before sensing would cost **one** measurement per
   probe instead of t. The base version's cost model is charged per output channel
   (section 8: 4 field measurements per block for strength-encoded dual, against 1 for
   the circuit crossbar) and does not exploit pre-sensor combination. This is the most
   promising hardware angle of the whole proposal and should be designed in from the
   start rather than retrofitted. **Qualified by section 4.1: it is available only when
   the combination is linear**, because a linear medium cannot multiply two fields.
6. **Which of the two realizations is the target** — (a) fold the combination into the
   injection strengths and keep the base crossbar, or (b) k separate single-output
   cells with no combination matrix anywhere (section 4.1)? They differ in cost
   (1 vs k readings per channel) and in which claim they support.

## 8. Scaffolding, when implementation starts

```
extensions/multigraph/
    DESIGN.md          this file
    __init__.py
    graphs.py          the multi-graph stack: k single-output graphs + combination
    ops.py             network-specific operators (calibration, geometry search)
    ext_01_baseline.py degenerate-case test: must reproduce the base version exactly
    ext_02_*.py        the first real experiment (see below)
```

* Depends read-only on `substrate_diffusion` and `experiments/abl_common.py`.
  `abl_common` is imported as a library, never edited; if its API is insufficient,
  copy the needed helper into `extensions/` rather than changing the base.
* **Reuse the guard**: call `abl_common.selftest()` and `selftest_cross_substrate()`
  in every extension script. They assert that the forward path is bit-identical to
  the production layer and that the two substrates agree, and they **raise**, so the
  extension inherits the base version's regression protection for free.
* `.gitignore` will need one added whitelist line (`!outputs/ext_*.txt|png`) — a
  repo-plumbing change, to be made deliberately and mentioned in the commit message,
  not a change to the base network.
* Reports go to `outputs/ext_*.txt`, conclusions to `extensions/multigraph/FINDINGS.md`
  (a separate file; the base `FINDINGS.md` is not extended).

**Suggested first experiment** (`ext_02`), in the spirit of the measurement-first
approach this project has used throughout: take the base F2 architecture, add m=8
output channels (readout/samples 1.25x, inside the warning zone), and share **centers
AND `h`** across the graphs of a group (section 4.1 — sharing only the centers would
quietly invalidate the algebra being tested).  Then compare

1. base: one graph, m rows (the matrix readout),
2. k=m graphs, shared geometry, selection (= base, must match bit for bit),
3. k=4 graphs, shared geometry, linear combination (rank ≤ 4, readout 0.75x),
4. k=4 graphs, shared geometry, fixed ±1 code (readout 0.63x),

on the same task, at equal measurement cost, reporting **readout R^2, the
calibration acceptance rate, and the transfer error of the amplitude calibration**
(the base version's gate already reports the first two; section 4.1 explains why the
acceptance rate is the early-warning signal for width problems).  The question that
decides whether the idea is worth pursuing is narrow and falsifiable:

> does sharing one graph of nodes across several output channels buy more, per
> parameter and per measurement, than giving each channel its own row of the base
> matrix?

If the answer is no, the memo's section 4 remains a useful map of the design space and
the base version is unchanged. If the answer is yes, it is a genuine extension of the
base version's headline rule — fewer readout parameters, moving away from the measured
failure boundary — and it can be reported as such.

---

## 9. Wiring checklist — DOCUMENTED ONLY, NOTHING APPLIED

Everything in this section is a note for the implementation step. No file named here
has been touched; in particular `.gitignore` and `.github/workflows/ci.yml` are, as of
writing, exactly as the base version left them apart from the base version's own
earlier edits.

### 9.1 Cache keys — do this FIRST, it can corrupt the base version

`abl_common.cache_path()` builds `{task}_{substrate}_{arch}_{arm}_{norm}_s{seed}_e{epochs}_b{budget}.npz`
in one shared `outputs/abl_cache/`, and `abl_03` **loads** from it to reuse ablation
A's trained models. If an extension trains with the same `task`/`arch`/`arm`/`seed`/
`epochs`/`budget` tuple it will **silently overwrite a base-version cache entry**, and
the next `abl_03` run will then evaluate an extension model as if it were a base
model. That is a corruption of the base version's evidence, produced without editing a
single base file.

Two acceptable defences, either is sufficient, doing both is better:

1. use a **distinct task string** for every extension run (e.g. `"ext4d"` rather than
   `"4d"`), and
2. use **distinct arch keys** (e.g. `"K4S"` for k=4 shared) — never reuse `G/M/W/F/F2`.

Also: never call `abl_common.save_model` with a base arch key, and do not point the
extension at `outputs/abl_cache` for its own caching — give it `outputs/ext_cache/`.

### 9.2 `.gitignore`

The base version now ignores **contents** and whitelists evidence:

```
outputs/*
!outputs/abl_*.txt
!outputs/abl_*.png
!outputs/sub_01_*.txt
!outputs/sub_01_*.png
```

The extension needs its report set whitelisted the same way, or its conclusions page
will cite files that a fresh clone cannot see — exactly the defect found and fixed in
the base version:

```
!outputs/ext_*.txt
!outputs/ext_*.png
```

Plus one new ignored directory for its own caches: `outputs/ext_cache/` (already
covered by `outputs/*`, but worth an explicit comment so nobody "tidies" it).

**Do not change `outputs/*` back to `outputs/`.** Ignoring the directory makes git skip
it entirely and every `!` negation stops working; that is the specific trap the base
version's comment block warns about.

### 9.3 CI

The base CI job now runs three steps: `sub_01_diffusion_verify.py --assert` (seconds,
the sharp guard), `abl_run_all.py --quick` (~15 min, pipeline smoke), and the legacy
`run_all.py`. The extension should **not** be added to that job: it is multiplied by
the three-Python matrix and would slow the base version's own protection.

Add a **separate job**, single Python version, with two steps in this order:

1. `python extensions/multigraph/ext_01_baseline.py --assert` — the degenerate-case
   test. This is the correctness anchor (constraint C4) and is fast, so it belongs in
   CI.
2. `python extensions/multigraph/ext_run_all.py --quick` — the extension's own
   pipeline smoke, only if step 1 is cheap enough to leave budget.

The extension must **not** be added to `experiments/abl_run_all.py`: that file belongs
to the base version, and editing it to know about `extensions/` is precisely the kind
of coupling this memo's scope rule forbids. The extension gets its own runner.

### 9.4 Commit hygiene

Keep the extension separable in history, so that `git log -- extensions/` and
`git log -- experiments/` never interleave:

| commit | contents |
|---|---|
| 1 | plumbing only: `.gitignore` lines from 9.2 (and `ci.yml` from 9.3). Say so in the message; it is the only commit that touches a base-version file. |
| 2 | `extensions/multigraph/*.py` (code) |
| 3 | `extensions/multigraph/DESIGN.md` (this file) |
| 4 | `extensions/multigraph/FINDINGS.md` + `outputs/ext_*.txt` (results) |

### 9.5 Definition of done for the wiring

* `git status` shows the extension's reports as **tracked**, not ignored.
* A base-version run (`abl_run_all.py --only 3`) still loads base caches and produces
  the base numbers — i.e. 9.1 was respected.
* `ext_01 --assert` fails loudly if the degenerate case drifts from the base version.
* No base file appears in `git diff` for commits 2–4.
