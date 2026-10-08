# RBF Particle Neural Network: Architecture and Principles

> **What this document is.** The theory behind the network: the physical model, the forward pass,
> the two training operators and their mathematics, what a substrate must satisfy, and how the
> cost is accounted. **What is measured — including what failed and what was retracted — is in
> [`FINDINGS.md`](FINDINGS.md)**, which is the current statement of what the project establishes.
> Numbers appear here only where the theory needs one to be intelligible, and each is traceable
> to a section of that document or of the extension's.

---

## 0. Core Design Philosophy

### 0.1 "Activate Then Combine" — reversing the standard paradigm

A standard network computes **`activate(combine(x))`**: a weighted sum, then a fixed nonlinearity.
That order forces the physical system to implement a precise non-monotonic function — the hardest
part to build in analog circuits, optics, or most other substrates.

This network reverses the order: **`combine(activate(x))`**. Each particle responds to the probe
point independently, and the responses are then combined linearly.

| | Standard network | This work |
|---|---|---|
| Order | `Wx` then `activate` | `activate(x, particle_i)` then `Σ W_i · z_i` |
| Source of nonlinearity | a fixed function, applied to a sum | the **spatial relation** between the probe and each source |
| Physical meaning | abstract coefficients | each particle is a source at a position, with an influence radius |
| Hardware mapping | needs a precise nonlinear element | source → propagation → receiver, which is a generic physical pattern |

### 0.2 The particle-source superposition model

A layer is a set of **sources** in D-dimensional space:

* **input `X`** — where the probe is;
* **centres `c_i`** — where the sources are;
* **widths `h_i`** — how far each source's influence reaches;
* **activation `z_i`** — how strongly source `i` responds at `X`, which is larger when the probe is
  closer;
* **strengths `W`** — how hard each source's response is driven into the output;
* **layer output** — the superposition `Σ_i W_i · z_i(X)`.

### 0.3 The weights are the activation strengths

`W` is not an abstract coefficient standing in front of a feature. **It is how hard each source is
driven** — a current, an amplitude, a pump power. This matters twice over:

1. **the read-out is physically meaningful**: the output really is a weighted superposition of the
   sources' field contributions;
2. **training may be a calibration**: if the strengths enter linearly, they can be *measured and
   solved for* rather than searched.

### 0.4 Training as a physical procedure

| network concept | physical counterpart |
|---|---|
| moving a centre | migrating a source |
| changing a width | expanding or contracting a source's influence |
| changing a strength | driving a source harder (a conductance, an amplitude) |
| residual training | later sources compensating what earlier ones missed |

Training therefore need not be simulated. It can be a **procedure performed on the device**, and
it can be accounted for in the currency the device actually spends: **measurements**.

### 0.5 Substrate universality, as a condition rather than a claim

The signal chain uses only operations that most physical systems already provide:

| operation | analog circuit | optical | wave / diffusion field |
|---|---|---|---|
| `Δx = (x − c)/h` | differential amplifier | optical path difference | phase or spatial offset |
| `Δx²` | Gilbert squarer | intensity detection | squared amplitude |
| `Σ Δx²` | Kirchhoff summation | beam superposition | interference / scalar field sum |
| `exp(−k d²)` | BJT subthreshold exponential | Gaussian beam | heat kernel / Green's function |
| `z / Σz` | current-mode normalisation | beam-splitter balancing | probability normalisation |
| `W · z` | memristor crossbar | tunable attenuator | coupling coefficients |

**But "maps to" is not the same as "works on", and §6 states the condition that separates them.**

---

## 1. The particle model

Each particle is a point in D-dimensional space with a centre and a width. For an input
`X = (x_1, …, x_D)`, particle `i`'s activation is a kernel of the scaled distance:

$$d_i^2 = \sum_{k=1}^{D} \frac{(x_k - c_{k,i})^2}{h_i^2}$$

$$z_i = \exp(-0.5\, d_i^2) \quad\text{(narrow channel)}, \qquad
  \tilde z_i = \exp(-0.3\, d_i^2) \quad\text{(wide channel)}$$

Two channels are carried, with different kernel coefficients, so that one layer can express both
local detail and broader structure. They are combined by addition at the output, so only their
sum matters to the prediction; the code keeps both because the two coefficients are physically
distinct elements.

**What the kernel is not.** The Gaussian is the *easiest* kernel to build, not the only admissible
one. §6 states the actual condition, and a quartic-exponent source satisfies it with a larger
basis while a Laplace source does not.

---

## 2. One layer

### 2.1 Parameters

| parameter | shape | meaning |
|---|---|---|
| `centers` | `(D, N)` | where the N sources are |
| `h` | `(N,)` | how far each source reaches |
| `W_gauss` | `(d_out, N)` | strengths of the narrow channel |
| `W_act` | `(d_out, N)` | strengths of the wide channel |

### 2.2 The forward pass, as a physical signal chain

1. **differential amplifier** (D times): `dx_k = (x_k − c_{k,i}) / h_i`
2. **Gilbert squarer** (D times): `dx_k²`
3. **Kirchhoff summation**: `d² = Σ_k dx_k²`
4. **BJT exponential** (both channels): `exp(−0.5 d²)` and `exp(−0.3 d²)`
5. **current-mode normalisation** (optional): `ẑ = z / Σ_j z_j`
6. **memristor crossbar**: `W_gauss ẑ_gauss + W_act ẑ_act`

The output width is the number of crossbar rows.

### 2.3 The same thing in closed form

For a batch `X ∈ ℝ^{B×D}`, centres `C ∈ ℝ^{D×N}`, widths `h ∈ ℝ^N` and strengths
`W^g, W^a ∈ ℝ^{d_out×N}`:

$$\mathbf{D}^2_{ij} = \sum_{k=1}^{D}\left(\frac{X_{ik} - C_{kj}}{h_j}\right)^2,
\qquad
\mathbf{Z}^g_{ij} = e^{-0.5 \mathbf{D}^2_{ij}},
\qquad
\mathbf{Z}^a_{ij} = e^{-0.3 \mathbf{D}^2_{ij}}$$

$$\hat{\mathbf{Z}} = \mathbf{Z} \Big/ \textstyle\sum_j \mathbf{Z}_{\cdot j}
\quad\text{(when normalisation is on)}$$

$$\mathbf{F} = \hat{\mathbf{Z}}^g (\mathbf{W}^g)^\top + \hat{\mathbf{Z}}^a (\mathbf{W}^a)^\top$$

**Normalisation is a choice with a consequence.** With it off, the sources' contributions are
**exactly independent** — each output is a sum of terms that do not interact, which is what makes
the parallel hardware mapping in §0.4 honest. With it on, they are not: the normaliser couples
every source to every other. The classification path uses it; the regression path does not. This
is a statement about the architecture, not a defect, and the coupling is measured in
`FINDINGS.md` §7.

---

## 3. Stacking, and the residual

### 3.1 Wire-direct stacking

```
X ─▶ Layer1 (N₁ sources, d₁ out) ─▶ h₁
      Layer2 (N₂ sources, d₂ out) ─▶ δ₂ ─▶ accumulate: f₂ = h₁ ⊕ δ₂
      Layer3 (N₃ sources, d₃ out) ─▶ δ₃ ─▶ accumulate: f₃ = f₂ ⊕ δ₃
```

Layers are connected **wire-direct**: layer `k`'s multi-dimensional output is layer `k+1`'s input.
Where dimensions differ, zero padding aligns them before accumulation.

### 3.2 The residual, and the mode that is shipped

Because layers accumulate, **layers 2 and 3 are trained against a residual** — what the earlier
layers left unexplained. There are two ways to form that residual, and the choice matters:

| mode | how the per-layer target is formed | status |
|---|---|---|
| **`proto`** (shipped default) | the layer's prototype-weighted prediction, `ŷ = Σ_k s_k y_k` with `s` a softmax over prototype distances | **the default, because the alternative is measured harmful** |
| `direct` | the first component of the layer's output, `y − h_{[:,0]}` | measured **worse**: its depth gain is `−0.0284` against `proto`'s `+0.0069` |

**The mechanism is a target mismatch, and it is worth stating because it is not obvious.** With
`direct`, layers 2 and 3 are fitted to the residual of the **raw** output, while the inference path
sums the **prototype** predictions — so the deeper layers are trained against a residual that
inference never reconstructs. `proto` is therefore not merely the safer default; the alternative is
measurably worse for a reason that can be named.

**A note on how this was learned.** The comparison was first made through two different programs
rather than one trainer with one switch, and it reported the opposite sign. Re-measured under
control — the shipped trainer, matched initialisation, and the mode as the only difference — the
sign reverses. The correction is recorded in `FINDINGS.md` §9; the table above states the
conclusion that survived.

---

## 4. The geometry operator

This is the part of the theory the project is built on, and it is what makes training a
*procedure* rather than a *search*.

### 4.1 The identity

Take the logarithm of a Gaussian particle's response. With `k` the kernel coefficient,

$$\log z_j(x) = -\frac{k}{h_j^2}\|x\|^2 + \frac{2k}{h_j^2}c_j \cdot x - \frac{k}{h_j^2}\|c_j\|^2$$

Define the node's **natural parameters**

$$\alpha_j = \frac{k}{h_j^2}, \qquad \beta_j = 2\alpha_j c_j, \qquad \delta_j = \alpha_j \|c_j\|^2$$

Then `log z_j` is **linear in `(α, β, δ)`**, and those three are not free — they satisfy **one exact
relation**:

$$\boxed{\ \delta_j = \frac{\|\beta_j\|^2}{4\alpha_j}\ }$$

So the geometry is recoverable from `D + 2` numbers, while the geometry has only `D + 1`
parameters (a centre and a width): **the extra number is a check, not a fit.**

### 4.2 The operator

Stack the coefficients into a design matrix

$$\phi(x) = \left[\ -\|x\|^2,\ \ x,\ \ -1\ \right] \in \mathbb{R}^{D+2}$$

and solve `log z = φ(x) · θ` by least squares, using **only the readings where the response is
informative** (the code refuses rather than returning a garbage geometry when too few are). Then

$$h_j = \sqrt{k / \alpha_j}, \qquad c_j = \beta_j / (2\alpha_j), \qquad
  \text{residual} = \delta_j - \|\beta_j\|^2 / (4\alpha_j)$$

**The cost is zero forward evaluations.** The operator reads the field the device is already
producing; it searches nothing.

### 4.3 The residual is a diagnostic

The residual of the identity in §4.1 is returned as **`log_rms`**, and it answers a question no
search-based trainer can ask: ***is this response really a single node?***

* a genuine Gaussian node reads `~3e-06`;
* a **Laplace or Yukawa** response reads **`4.16e-01`** — the response is not a node of this family,
  and the operator says so instead of fitting one anyway;
* a **lookup table** in the forward reads `2.5e-05` at 512 entries, `1.6e-03` at 64 — detected
  **with no change to the operator**.

That makes the residual a **hardware diagnostic** as much as a training device: it is how a device
whose response has drifted away from the model is caught.

---

## 5. The strength operator, and calibration

### 5.1 Strengths are linear, so solve for them

The output is `Σ_j W_j z_j`. The strengths enter **linearly**, so given the activations and the
targets they follow from a ridge solve:

$$\mathbf{W} = \operatorname*{arg\,min}_{\mathbf{W}} \|\mathbf{Z}\mathbf{W}^\top - \mathbf{Y}\|^2
  + \lambda \|\mathbf{W}\|^2$$

with `λ` selected on a held-out slice of the layer's own data.

### 5.2 The training rule

> **Calibrate the strengths in closed form, and spend every physical measurement on the geometry.**

The reasoning is a division of labour: the strength problem is linear and therefore *free* to solve
exactly once the activations are known, while the geometry problem is not. Searching the strengths
spends measurements on a problem that has a closed-form answer, and it is worse than that — the
search's answer is discarded when the calibration overwrites it. Measured, the two agree to within
`8e-04` on the mean, which says the search was contributing nothing there.

### 5.3 What the calibration does not do

It is a **read-out** correction, not a **geometry** correction, and it does not help a deep network
learn: in the deep regression path the calibration arm does not beat the search arm
(`−0.0710` on the mean, 1 of 3 seeds). The read-out is not that path's bottleneck; the geometry is.
**That is a measured negative and it is why the operator in §4 is where the work goes.**

---

## 6. What a substrate must satisfy

### 6.1 The precondition

"Any physical system" is not a claim that can be left as a slogan, and it does not have to be. The
operators of §4 read a node's geometry out of the coefficients of `log z`, so what a substrate has
to provide is:

> **`log z(x)` must be LINEAR in some fixed, FINITE basis `φ(x)` of the input.**

A Gaussian satisfies it because `‖x − c‖²` expands to a degree-2 polynomial whose coefficients
encode `c`. **Any polynomial exponent satisfies it**, at the degree of that polynomial — a quartic
well needs a degree-4 basis, and read through the degree-4 basis it transfers to machine precision
(`1.5e-15` centre, `7.3e-15` width), while read through the degree-2 basis it gives a plausible
but wrong answer (`7.82e-02`).

**And the same test excludes.** A Laplace or Yukawa source is not log-linear in any finite basis,
which is not an opinion: §4.3's residual measures it at `4.16e-01` against a Gaussian's `3e-06`.

### 6.2 The three substrates implemented

Each is a **layer**, not a formula in the trainer — the trainer never substitutes its own kernel
for the device's.

| substrate | the medium | its kernel | how it is verified |
|---|---|---|---|
| **Analog circuit** | differential pair → Gilbert squarer → BJT exponentiator → memristor crossbar | `exp(−k d²)` | the repository's original vehicle |
| **Diffusion cell** | a diffusing species in a linear medium | the Green's function of the diffusion equation | **the propagator matches an independent finite-difference PDE solve to `1.62e-05`, the solver's own error**; superposition is exact to `2.86e-16` |
| **Anharmonic trap** | a particle in `V(r) = κr⁴` | `exp(−(κ/kT) r⁴)` | **a different exponent; transfers at machine precision once read through the right basis** |

### 6.3 Widening the class: the lookup table

If a source is a **fixed function of one scalar**, `z = T(u)` with `u = ‖(x−c)/h‖²`, then a single
1-D table serves every particle and every width. That is **wider** than "polynomial exponent" — it
admits any `T` — and it is narrower than "any kernel": a source whose *shape* depends on `h`
(Yukawa) cannot be carried by one table of `u`, and needs a family indexed by `h`.

Two facts make the table practical rather than merely possible:

* **the error is known in closed form** — linear interpolation on a uniform grid gives at most
  `(Δu)²/8`, so a table size can be *chosen* rather than tuned;
* **the parameterisation must match the kernel's smoothness.** Tabulating `exp(−βs)/(s + r₀)` on a
  uniform grid in `u` left the error at `2.2` and unmoved from 4 entries to 64, because
  `s = √(u/k)` has an infinite derivative at `u = 0`. Gridding in `s` instead gives `4.736e-02` at 4×8
  entries and `4.0e-03` at 16×32 — a factor of about 550.

---

## 7. The cost model

### 7.1 The unit is a measurement

Two costs are distinguished and **never added**:

* **forward evaluations** — how many times the device is driven to produce a prediction (what a
  gradient search spends);
* **read-out calibrations** — how many times a sensor is read to obtain the strengths
  (`n_train × Σ N` for a full calibration).

A comparison is run in the second unit. That is what makes the claim "the geometry operator spends
**zero** forward evaluations" a measurement rather than a slogan — and it is why a `2×` slowdown in
a table lookup does not matter here: seconds are not the currency.

### 7.2 What the rule buys

Calibrating the strengths and spending the measurements on the geometry reaches a given accuracy
for **15–30× fewer measurements** than searching either family, and holds at **8 of 8** device
levels while being charged 2.11× the measurement budget. This is the project's headline
quantitative result.

---

## 8. Gradient-based training, as the comparison

These are the arms the operators are measured against, not the pitch.

| method | what it needs from the device | cost per step |
|---|---|---|
| **SPSA** | the loss, at the output only — no per-parameter access | `2 · spsa_avg` forward passes |
| **per-parameter finite differences** | one read per parameter | `O(parameters)` reads |
| **digital twin** | a model of the device | none on the device (but the model must be right) |

**Simultaneous perturbation (SPSA)** estimates the gradient from two loss evaluations at a random
± perturbation, which is what makes it usable when parameters cannot be addressed individually.
**Finite differences** give exact per-parameter gradients at a per-parameter cost. **The digital
twin** trains in PyTorch and exports the weights, and its exposure is that the device must match the
model — a mismatch is measured in `FINDINGS.md` §6.3.

**A fourth scheme — building the gradient chain in analog hardware — is not implemented and is not
offered.** The functions exist in `physical_simulator/components.py` as a declared island with no
entry point, and the reason it is not needed is §5.2: once the strengths are calibrated in closed
form, a gradient chain would only ever be needed for the geometry, which the operator solves
directly.

---

## 9. Scope and limits

* **Everything here is simulation.** The three substrates are models of devices, and the diffusion
  cell's PDE verification checks the *model's* kernel against physics — not a manufactured cell
  against the model. **No part of this has been fabricated.**
* **The scale is small.** 20–40 particles per layer, 3 layers, 2–10 input dimensions.
* **The precondition is necessary, and it is not free to satisfy.** A substrate whose response is
  not log-linear in a finite basis needs a table, and a table needs a fixed single-scalar form.
* **The operator solves the geometry only within its own layer's input space.** In a cascade each
  layer's nodes respond to the previous layer's outputs, so a deep network's geometry *in terms of
  the original input* is not one log-linear model.
* **The deep architecture is not the contribution.** The operators are. What the deep stacking
  does and does not achieve is measured in the extension's `FINDINGS.md` §§12.30–12.31.

---

## Appendix: provenance

Three generations of code exist in the repository, and the older two are kept rather than deleted
so that the evolution can be checked.

| generation | path | what it is |
|---|---|---|
| **v1** | `rbf_network/`, `forward_training/` | the original 2-D RBF network and its trainers |
| **ND** | `simulator/`, `network/`, `physical_simulator/`, `substrate_diffusion/`, `training/` | the D-dimensional physical model, the residual trainers, and the two substrates |
| **`multigraph`** | `extensions/multigraph/` | the operators, the substrate precondition, the cost analysis, and the multi-graph layer |

`training/himal.py`, `training/spsa.py` and `training/twin.py` are thin re-exports of
`forward_training/`'s modules — one implementation, two import paths, kept so that the v1 scripts
continue to import. `FINDINGS.md` §12 lists the artifacts.
