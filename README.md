# Physical-Domain RBF Particle Neural Network

[![CI](https://github.com/usecodeandaiforstudy/signal-node-physics-network/actions/workflows/ci.yml/badge.svg)](https://github.com/usecodeandaiforstudy/signal-node-physics-network/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

A neural network whose **nodes are physical sources**, trained by **operators that read the
device** instead of by gradients. The strengths are solved in closed form; the geometry is
recovered from field readings. The whole thing is meant to run on a physical substrate — this
repository implements and measures three of them.

---

## 1. What this is

Ordinary neural networks store their knowledge in **matrices**, and training means adjusting those
matrices by backpropagation. This project replaces the matrix with **a field**:

* a **node** is a movable source in a physical medium — a particle with a position and an
  influence radius;
* the input perturbs that medium, and each node responds with a nonlinear activation whose
  *shape* is set by the node's geometry;
* the network's output is the **superposition** of those activated sources, weighted;

so `out(x) = Σ_j W_gj · z_j(x)`, where `z_j(x)` is the physical response of node `j` and `W_gj` is
how strongly that node's source is driven. **"Activate first, then combine"** — the nonlinearity
comes from the geometry of the interaction, not from a fixed activation function applied to a
weighted sum.

**Why it is not just an analogy.** The two quantities a training algorithm has to set are both
*physical knobs*: driving a source harder is a real current or amplitude, and moving a source or
changing its radius is a real position or trap stiffness. Training therefore does not have to be
simulated — it can be a procedure performed **on the device**, paid for in **measurements** rather
than in gradients.

---

## 2. The core ideas

**1 — Nodes replace matrices.** A layer is not a weight matrix followed by an activation; it is a
set of sources and the field they jointly produce. The parameters live in the geometry (where each
source is, how wide it is) and in the drive strength of each source.

**2 — The weights ARE the nodes' activation strengths.** `W_g` is not an abstract coefficient: it
is how hard each source is driven. This is what makes the read-out physically meaningful, and what
makes a **calibration** — measuring the response and solving for the drives — a legitimate way to
train.

**3 — Train with operators that read the device, not with gradients.** The training rule is:
**calibrate the source strengths in closed form, and spend every physical measurement on the
geometry.** Because the response of a Gaussian-type node is

```
log z_j(x) = -alpha_j ||x||^2 + gamma_j . x - delta_j
```

the response is **linear in the node's natural parameters**, so a linear solve recovers the centre
and the width **in closed form** from field readings alone — and the strengths follow from a
second closed-form solve. **Neither family needs a gradient search.**

**4 — This works on any physical system meeting one checkable condition.** Not "any system" as a
slogan: the node's response must satisfy

> **`log z(x)` is LINEAR in some fixed, FINITE basis `phi(x)` of the input.**

A Gaussian passes (its exponent is a degree-2 polynomial whose coefficients encode the centre).
**Any polynomial exponent passes at its own degree** — a quartic well needs a degree-4 basis, and
then transfers to machine precision. A source whose response is *not* log-linear in any finite
basis — Laplace, Yukawa — is **excluded by the same test**, and the operator's own residual says
so. A lookup table relaxes the condition further, to **any kernel that is a fixed function of one
scalar**, and an `h`-indexed family of tables reaches Yukawa.

---

## 3. Architecture

```
input x
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER  (one physical cell, N nodes)                             │
│                                                                  │
│   node j:   centre c_j (D-dim)     width h_j                     │
│             activation  z_j(x) = T( ||(x - c_j)/h_j||^2 )        │
│                                                                  │
│   superposition over nodes:   field_i(x) = Σ_j W_ij · z_j(x)      │
│   (the medium is linear, so this sum is EXACT — and verified)    │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
   next layer (wire-direct: the previous output is the next layer's input)
   │
   ▼
   scalar residual read-out:  pred = Σ_layers out_layer[:, 0]
```

**The forward pass** is therefore: compute each node's activation from the distance between the
input and the node's centre, scale by the node's width, and superpose.

**The two training operators:**

| operator | what it reads | what it produces | cost |
|---|---|---|---|
| **geometry operator** | the field at probe points | each node's **centre and width**, in closed form, plus a **consistency residual** | **0 forward evaluations** |
| **strength operator** | the activations and the targets | the drive strengths, by closed-form (ridge) solve | `n_train × N` sensor readings |

**And the residual is a diagnostic, not a by-product.** The geometry operator returns `log_rms`,
the residual of the identity a genuine single node must satisfy. It answers *"is this response
really one node?"* — **a question a search-based trainer cannot ask**, because a search never
checks whether its model matches the object it is modelling.

**Three substrates are implemented**, each as a real layer rather than a formula in the trainer:

| substrate | what the medium is | how well it is verified |
|---|---|---|
| **Analog circuit** — `physical_simulator/` | differential pair → Gilbert squarer → BJT exponentiator → memristor crossbar | the repository's original vehicle, chosen because it simulates easily |
| **Diffusion cell** — `substrate_diffusion/` | a diffusing species in a linear medium; the kernel is the medium's Green's function | **its propagator matches an independent finite-difference PDE solve to `1.62e-05` — the solver's own error** |
| **Anharmonic trap** — `extensions/multigraph/ext_38_third_substrate.py` | a particle in `V(r) = κr⁴`, whose equilibrium density is `exp(−(κ/kT)r⁴)` | **a different exponent, and the operator transfers at machine precision (centre `1.5e-15`, width `7.3e-15`) once read through the right basis** |

---

## 4. What it can do now

**Measured on this repository's tasks, all reproducible from the commands in
[`FINDINGS.md`](FINDINGS.md) and [`extensions/multigraph/FINDINGS.md`](extensions/multigraph/FINDINGS.md).**

| capability | measured result |
|---|---|
| **Geometry from field readings, in closed form** | exact recovery **passes the operator's own `1e-3` relative bar**; under **1% reading noise** the centre error is **`1.2e-03`**, at 3% it is `4.1e-03`. On the quartic substrate: centre **`1.5e-15`**, width **`7.3e-15`** |
| **Operators against gradient training** | **`+0.98` vs `+0.23`** (2-D regression), **`+0.58` vs `+0.05`** (4-D), **`1.00` vs `0.73`** (moons classification) |
| **The operator's cost** | **zero forward evaluations**, against 600 epochs of autograd for the gradient-trained arm |
| **Measurement efficiency of the training rule** | **15–30× fewer measurements** to reach the same accuracy |
| **A wrong substrate is DETECTED, not silently mis-fit** | a Laplace/Yukawa response gives `log_rms` **`4.16e-01`** against a Gaussian node's `3e-06` |
| **A lookup table is detected with no code change** | 3 of 3 tabulated sources flagged by `log_rms` alone |
| **Multi-graph cost** | **pre-sensor superposition cuts sensor reads by `k/m`** — `4` instead of `12` at `k=6, m=2`, reproducing the output to `7.4e-08` |
| **Working size** | 20–40 particles per layer, 3 layers, 2–10 input dimensions |

**What it cannot do, stated plainly:**

* **Nothing here has been fabricated in hardware.** All three substrates are simulated cells. The
  operators have never been run against a real device, and the diffusion cell's PDE verification
  checks the *model's* kernel against physics — not a manufactured cell against the model.
* **The scale is a toy scale.** Every task is small, every hyper-parameter is the shipped default,
  and the high-dimensional axis was measured and found **out of reach** rather than assumed to be.
* **The deep architecture this project started with does not work.** It is beaten by a zero-forward
  baseline on every task the repository can build. The operators are the part that delivers; the
  deep stacking is not.
* **One substrate family is covered so far** — polynomial exponents, plus the table-based
  extension. Non-log-linear kernels need the table, and the table needs a fixed single-scalar form.

The full record — what was measured, how, and what qualifies it — is
[`FINDINGS.md`](FINDINGS.md): [§9](FINDINGS.md) states the caveats that bound the numbers, and
[§11](FINDINGS.md) what remains open. **Read those before citing a number.**

---

## 5. Quick start

```bash
git clone https://github.com/usecodeandaiforstudy/signal-node-physics-network.git
cd signal-node-physics-network
pip install -r requirements.txt      # or: pip install -e .
```

**Verify the substrate physics (~4 s):**

```bash
python experiments/sub_01_diffusion_verify.py --assert
```

**Run the operator suite** — the part the project stands behind; most of its scripts finish in
seconds:

```bash
python extensions/multigraph/ext_01_baseline.py --assert    # correctness anchor
python extensions/multigraph/ext_07_operator_trainer.py     # operators vs gradient training
python extensions/multigraph/ext_06_node_operator.py        # geometry from field readings
```

**Train a small network in the digital model:**

```python
from network.combined import CombinedHiMALRBF
from experiments.benchmarks.datasets import load_dataset

X_tr, X_te, y_tr, y_te, _ = load_dataset("moons")
model = CombinedHiMALRBF(input_dim=2, layer_sizes=(8, 5, 3), output_dims=(3, 2, 2),
                         task="classification", n_classes=2, use_spsa=True, spsa_avg=3)
model.train(X_tr, y_tr, epochs_per_layer=30, lr=0.02)
print((model.predict(X_te).ravel() == y_te.ravel()).mean())
```

---

## 6. Every file, and what it is for

### Documents

| file | what it is |
|---|---|
| `README.md` / `README_CN.md` | this file, in English and Chinese |
| `THEORY.md` / `THEORY_CN.md` | the full derivation: the physical mapping, the training schemes A–D, and the mathematics of the operators |
| `FINDINGS.md` | **the current statement of what the project establishes.** Sections 0–11: reproduce commands, the substrate verification, the capacity analysis, the headline training rule, locality, the cost model, the caveats that bound the numbers, scope, what remains open, artifacts |
| `extensions/multigraph/FINDINGS.md` | the extension's own record: sections 0–11 plus 23 subsections numbered 12.x — every operator result, every negative result, and the substrates' evidence |
| `extensions/multigraph/DESIGN.md` | the design of the multi-graph extension: notation, the two halves, the combination operator's mathematics, parameter economics, and the constraints any implementation must satisfy |
| `extensions/multigraph/BASE_CORRECTIONS.md` | the extension's audit of the base version's headline claims — one item per claim, with the measurement that settles it |
| `CONTRIBUTING.md` | the conventions this repository runs on: how a number gets written down, and how it gets verified before it is published |
| `CITATION.cff` | how to cite it |

### The library

| path | what it is |
|---|---|
| `simulator/layer_nd.py` | **`PhysicalRBFLayerND`, the D-dimensional physical layer**: `centers (D,N)`, `h (N,)`, `W_gauss`, `W_act`; `forward` returns the two channels |
| `simulator/network_nd.py` | stacks layers into a network with the scalar residual read-out |
| `network/core_nd.py` | `RBFLayerND`, the **digital** (PyTorch) layer the physical one is exported from |
| `network/combined.py` | `CombinedHiMALRBF` and `SourceStack`, the digital models the physical ones mirror |
| `network/utils_nd.py` | initialisation, scaling and parameter-count helpers |
| `physical_simulator/components.py` | the **analog circuit's** elements: differential pair, Gilbert multiplier and squarer, BJT exponentiator, memristor crossbar, integrator, ADC, plus the noise/config model |
| `physical_simulator/layer.py` | `PhysicalRBFLayer`, the circuit-backed layer |
| `physical_simulator/network.py` | circuit-backed network assembly and the digital↔physical export |
| `substrate_diffusion/components.py` | the **diffusion cell's** physics: the Green's function, the sensor chain (noise, drift, clipping, quantisation), and the `readout` stage |
| `substrate_diffusion/layer.py` | `DiffusionRBFLayerND`, the diffusion-backed layer, calibratable and seeded |
| `training/himal_nd.py` | `HiMALND`, the D-dimensional hierarchical residual trainer (SPSA and finite differences) — the **gradient-trained comparison arm** |
| `training/proto_nd.py`, `training/proto_layerwise.py` | the prototype-guided trainers, v2 and v1 |
| `training/himal.py`, `training/spsa.py`, `training/twin.py` | thin **re-exports** of `forward_training/`'s v1 modules: one implementation, two import paths |
| `forward_training/` | the **v1 tree**, kept as provenance; `experiments/ref2d/` imports from here |
| `rbf_network/` | the original 2-D RBF network, ancestor of the ND version |

### Experiments — the base version

| path | what it is |
|---|---|
| `experiments/abl_common.py` | **the harness everything shares**: the digital↔physical bridge, the layer builders, the closed-form read-out (`solve_readout_closed_form`), the cost accounting (`readout_calibration_cost`), the measurement path, and `selftest()` |
| `experiments/sub_01_diffusion_verify.py` | **the substrate verification suite** — kernel against a PDE solve, exact superposition, the cost model. `--assert` enforces the thresholds `FINDINGS.md` §§2, 3, 8 rest on |
| `experiments/abl_01…abl_07` | the seven ablations: capacity split, locality, mismatch sensitivity, calibration robustness, twin error, task sweep, convergence grid |
| `experiments/abl_run_all.py` | runs all seven (`--quick` for the CI smoke test) |
| `experiments/nd_01…nd_09` | the ND digital baseline, the physical forward check, physical training, the HiMAL pipeline, comparisons, ablations, verification, benchmarks |
| `experiments/cifar_01…cifar_07` | the CIFAR-10 line — the project's **earliest** work: 3-D point-cloud encoding, autograd training, physical export, three-way comparison. Seeded and reproducible; the numbers are small by design |
| `experiments/reg_01…reg_06` | the regression tasks, including California Housing |
| `experiments/train_01…train_03` | grouped finite differences, grouped SPSA, and zero-ADC analog SPSA |
| `experiments/ref2d/` | the 2-D reference series, including the v1 trainers |
| `experiments/benchmarks/` | dataset loaders, baseline models and the report writer used by the benchmark runner |
| `experiments/legacy_suite.py` | the original nine-part verification suite — digital-physical consistency, the FD depth proofs, noise robustness, the v1 baselines, the depth ablation. Kept because the early sections still cite its numbers; `abl_run_all.py` is the current batch runner |
| `experiments/download_cifar.py` | fetches CIFAR-10 into `cifar/` (gitignored) |

### Experiments — the extension

| path | what it is |
|---|---|
| `extensions/multigraph/trainer.py` | **the operator trainer**: `features` (device-faithful activations), `fit_operator`, `solve_strengths`, and the candidate proposal/refinement loop |
| `extensions/multigraph/operators.py` | **the operators themselves**: `design_matrix`, `node_from_response` (the closed-form geometry solve and its consistency residual), `node_response`, `propose_node`, `star_probes`, `leverage_select`, `identify_placement`, `best_weight` |
| `extensions/multigraph/graphs.py` | the multi-graph layer: `Graph`, the `Linear`/`Nonlinear` combination, `MultiGraphStack`, and the construction that must degenerate exactly to the base version |
| `extensions/multigraph/ext_01…ext_51` | **the experiments**, in order. Each has a reproduce command and a tracked artifact in the extension's `FINDINGS.md`; the numbering matches its section |

### Repository mechanics

| path | what it is |
|---|---|
| `outputs/` | every experiment's artifact. Only the ones a document cites are tracked (see `.gitignore`), so a clone carries the evidence and not the caches |
| `cifar/` | the CIFAR-10 data, downloaded, gitignored |
| `reference/` | local scratch, gitignored |
| `.github/workflows/ci.yml` | CI: the substrate assertions and all seven ablations on three Python versions, plus the extension's correctness anchor, **30 of its 51 scripts with `--assert`**, and a check that every script asserts and every cited artifact exists |
| `setup.py`, `requirements.txt` | packaging |

---

## 7. Where to read more

| if you want | read |
|---|---|
| the mathematics | [`THEORY.md`](THEORY.md) |
| what is actually established, with numbers | [`FINDINGS.md`](FINDINGS.md) |
| the operator work, in full | [`extensions/multigraph/FINDINGS.md`](extensions/multigraph/FINDINGS.md) |
| why the extension is built the way it is | [`extensions/multigraph/DESIGN.md`](extensions/multigraph/DESIGN.md) |
| what qualifies the numbers, and why | [`FINDINGS.md`](FINDINGS.md) §9 |
| what is still open | [`FINDINGS.md`](FINDINGS.md) §11 |

**On the record-keeping.** This project states claims and then measures them, and where a
measurement contradicted a claim, the claim was **replaced by the measured one** rather than
annotated — so what these documents contain is the version that survived, with its number.
`FINDINGS.md` §9 collects the caveats a reader needs in order to use those numbers correctly: the
CIFAR line's published figures being single draws from an unseeded process, "ideal" not being
noise-free on the circuit path, the geometry arm losing under equal cost. **If a number on this
page ever disagrees with those documents, the documents are right.**

---

## 8. Status, license, citation

**Status.** Research code: toy scale, simulated substrates, no hardware. What works and what does
not is in §4 above; the full record is in [`FINDINGS.md`](FINDINGS.md).

**License.** MIT — see [`LICENSE`](LICENSE).

**Citation.** See [`CITATION.cff`](CITATION.cff). Please cite the *methods* — the geometry
operator, the substrate precondition, the measurement-cost accounting — rather than the early
headline numbers, which `FINDINGS.md` §9 qualifies.

**Contributing.** See [`CONTRIBUTING.md`](CONTRIBUTING.md). Contributions that extend the
substrate coverage, replace a simulation with a measurement, or falsify a claim are the ones this
project is built to accept.

