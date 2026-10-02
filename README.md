# Physical-Domain RBF Particle Neural Network

[![CI](https://github.com/usecodeandaiforstudy/singal-node-physics-network/actions/workflows/ci.yml/badge.svg)](https://github.com/usecodeandaiforstudy/singal-node-physics-network/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)

---

> ## ⚠️ Status: six claims on this page have been measured and retracted
>
> This README predates the experiments in [`FINDINGS.md`](FINDINGS.md), which is the
> current statement of what the project establishes. **Read
> [FINDINGS.md section 9](FINDINGS.md) before citing anything below.** In short:
>
> | claim on this page | measured status |
> |---|---|
> | "39.3% on CIFAR-10 (SOTA-class result)" | Not SOTA: 5 of 10 classes, grayscale, 100 images per class, and the model is selected on the **test** set ("peak test accuracy") — as are the MLP and CNN baselines |
> | "digital-physical discrepancy 7.80×10⁻⁷" | That number is float32 rounding plus a `+1e-8` guard in the partition-of-unity divide. It is **not** a physical-fidelity statement. The kernel is now verified against an independent PDE solve instead |
> | "proves parameter-group independence" | The cited experiment (`experiments/train_01_grouped_fd.py`) indexed particles with the **wrong convention** (fixed now; results must be regenerated). Measured properly: independence holds **exactly** without normalisation and is **false** under the PU normalisation the classification path uses |
> | "K parameter groups = K hardware channels, no mutual dependency" | Same as above — coupling under PU measures 0.063–0.261 |
> | "analog gradient chain (O(1) latency)" | Those functions are **defined but never called** anywhere in the repository |
> | "four hardware complexity levels (0 ADC / 1 ADC / …)" | Superseded: the tiers differ only in how the *geometry* is optimised. The source strengths should be **calibrated in closed form, not searched** |
>
> What replaced them is stronger and is measured: a second, PDE-verified physical
> substrate; a substrate cost model; and a training rule — **calibrate the source
> strengths, spend every physical measurement on the geometry** — that buys the
> same accuracy for **15–30× fewer measurements**. See
> [`FINDINGS.md`](FINDINGS.md).

---

## Introduction

> A novel neural network architecture based on physical field superposition — each "neuron" is a movable particle in D-dimensional space, with nonlinearity arising from the spatial distance between particle and input. The architecture is mappable to multiple physical platforms (analog circuits, optical systems, wave fields, etc.); this project uses analog circuits as the implementation vehicle, chosen for ease of computer simulation and validation.

This project explores the feasibility of executing the complete pipeline from training to inference of neural networks within physical systems. The core design is "activate first, then combine": particles first produce a Gaussian response to the input, then linearly superimpose — nonlinearity arises from the spatial distance between particles and inputs, rather than from a fixed activation function. The network supports classification, regression, and high-dimensional image processing, offering multiple training methods including SPSA, finite differences, and digital twins. All forward operations have corresponding analog circuit implementations, and digital-to-physical export achieves extremely high precision.

Key achievements:
- Achieves 39.3% classification accuracy on CIFAR-10 (5 classes) with only 2,884 parameters, surpassing an equivalently sized MLP (34.0%), with a generalization ratio of 73% — the highest among the three compared models *(retracted: see the status box above and [FINDINGS.md](FINDINGS.md) §9.4 — not SOTA, and the model was selected on the test set)*
- Digital-physical simulator discrepancy of only 7.80×10⁻⁷; weights can be directly exported from PyTorch to the physical simulator *(retracted: that is float32 rounding plus a PU epsilon, not physical fidelity — [FINDINGS.md](FINDINGS.md) §9.2. Weight export itself does hold, and is now demonstrated across two unrelated substrates)*
- Proves parameter-group independence — RBF particle output contributions are mutually independent, naturally supporting multi-channel hardware parallel training *(retracted and replaced by a measurement: independence is exact without normalisation, false under PU — [FINDINGS.md](FINDINGS.md) §7)*
- Effective gradient estimation achieved across four hardware complexity levels (0 ADC / 1 ADC / fully analog gradient chain / digital twin) *(the analog gradient chain is dead code and the ADC interface is never called; the four tiers differ only in how the geometry is optimised — [FINDINGS.md](FINDINGS.md) §9)*

## Features

- 🧠 **Physically Deployable**: Every operation (distance, Gaussian kernel, MVM, gradient estimation) has a corresponding analog circuit; fully analog inference with zero digital dependency
- 🔬 **Digital-Physical Equivalence**: The physical simulator is mathematically exactly equivalent to the PyTorch model, supporting direct weight export
- 🖼️ **Breaking the High-Dimensional Barrier**: 3D structured encoding enables RBF to process 1024-dimensional images, with 2,884 parameters outperforming a larger MLP on the same task
- ⚡ **Backpropagation-Free Training**: SPSA (0 ADC) / per-parameter FD (1 ADC) / analog gradient chain (O(1) latency) *(the gradient chain is defined but never called — see the status box)*
- 🔀 **Inherently Highly Parallel**: Parameter groups are independent; K parameter groups = K hardware channels trainable in parallel, with no mutual dependency *(holds without normalisation; false under PU — [FINDINGS.md](FINDINGS.md) §7)*
- 🛡️ **Physically Noise-Robust**: Accuracy drops by only 0.01 across four levels of circuit noise, adapting to real hardware non-idealities *(that measurement was taken on the training set; see [FINDINGS.md](FINDINGS.md) §6.2 for the device-robustness result that replaced it)*

## Architecture Overview

The network consists of movable particles in D-dimensional space. Each layer has N particles, each with a center coordinate and an influence radius h. Input passes through three successive RBF layers; each layer independently computes a Gaussian kernel response and performs linear projection via a memristor crossbar array. Layers are connected wire-direct.

```
Input X (batch, D)
    → Layer 1: differential pair (dx) → Gilbert squarer (d²) → BJT exponentiator (exp) → memristor MVM → output
    → Layer 2: differential pair (dx) → Gilbert squarer (d²) → BJT exponentiator (exp) → memristor MVM → residual addition
    → Layer 3: differential pair (dx) → Gilbert squarer (d²) → BJT exponentiator (exp) → memristor MVM → residual addition
    → final output
```

The physical signal chain includes 6 circuit elements: differential amplifier → Gilbert squarer → Kirchhoff current summation → BJT subthreshold exponentiator → current mirror normalization → memristor crossbar array.

For the complete derivation (physical system mapping, training algorithms, theoretical analysis), see [THEORY.md](THEORY.md).

## Quick Start

### Requirements

- Python >= 3.8
- pip

### Installation

```bash
git clone https://github.com/usecodeandaiforstudy/signal-node-physics-network.git
cd signal-node-physics-network
pip install -r requirements.txt
```

Optional: install in editable mode so project modules can be imported from any path:

```bash
pip install -e .
```

### Minimal Examples

**Method 1: SPSA Training (physical domain, 0 ADC, simplest hardware)**

```python
from network.combined import CombinedHiMALRBF
from experiments.benchmarks.datasets import load_dataset

X_tr, X_te, y_tr, y_te, _ = load_dataset("moons")

model = CombinedHiMALRBF(
    input_dim=2, layer_sizes=(8, 5, 3), output_dims=(3, 2, 2),
    task="classification", n_classes=2, use_spsa=True, spsa_avg=3,
)
model.train(X_tr, y_tr, epochs_per_layer=30, lr=0.02)
print(f"Acc: {(model.predict(X_te).ravel() == y_te.ravel()).mean():.3f}")
```

**Method 2: Per-Parameter FD Training (physical domain, 1 ADC, exact gradients)**

```python
model = CombinedHiMALRBF(
    input_dim=2, layer_sizes=(8, 5, 3), output_dims=(3, 2, 2),
    task="classification", n_classes=2, use_spsa=False,
)
model.train(X_tr, y_tr, epochs_per_layer=40, lr=0.015, eps=1e-3)
print(f"Acc: {(model.predict(X_te).ravel() == y_te.ravel()).mean():.3f}")
```

**Method 3: Digital Twin Training (PyTorch autograd, rapid prototyping)**

```python
# See experiments/cifar_01_baseline.py — build the network, train with autograd,
# then export weights to the physical simulator via from_digital()
```

## Project Structure

```
singal-node-physics-network/
├── README.md / README_CN.md        # project description
├── THEORY.md / THEORY_CN.md        # complete theoretical derivation
├── LICENSE                         # MIT
├── requirements.txt                # dependency list
├── setup.py                        # pip install -e . support
│
├── physical_simulator/             # analog circuit components
│   └── components.py               #   differential pair, Gilbert squarer, BJT exponentiator, memristor MVM, etc.
├── simulator/                      # ND physical simulator
│   ├── layer_nd.py                 #   PhysicalRBFLayerND
│   └── network_nd.py               #   PhysicalResidualRBFNetworkND
├── network/                        # digital network definitions (PyTorch, reference / digital twin)
│   ├── combined.py                 #   CombinedHiMALRBF unified entry point
│   └── core_nd.py                  #   RBFLayerND + ResidualRBFNetworkND
├── training/                       # training algorithms
│   └── himal_nd.py                 #   HiMAL-ND layer-wise dual-network trainer
├── rbf_network/                    # 2D RBF network (early version, kept for reference)
│
├── experiments/
│   ├── run_all.py                  #   quick smoke test (10 basic validations, ~5 min)
│   ├── download_cifar.py           #   download CIFAR-10 dataset to cifar/ directory
│   ├── ref2d/                      #   simple 2D reference implementations (13 files, for introductory reference)
│   ├── reg_01_proto_baseline.py    #   three-channel architecture + Proto residual baseline
│   ├── reg_02_direct_residual.py   #   direct residual enables depth effectiveness
│   ├── reg_03_noise_robustness.py  #   noise robustness verification
│   ├── reg_04_fd_reg_1d.py         #   1D sin(x) minimal verification
│   ├── reg_05_fd_cls_moons.py      #   FD classification depth proof
│   ├── reg_06_california_v6.py     #   California Housing real-data benchmark
│   ├── cifar_01_baseline.py        #   CIFAR-10 5-class: first success
│   ├── cifar_02_regularized.py     #   CIFAR-10 5-class: regularized (best 39.3%)
│   ├── cifar_03_3layer_reg.py      #   CIFAR-10 5-class: three-layer ablation
│   ├── cifar_04_large_data.py      #   CIFAR-10 5-class: capacity bottleneck analysis
│   ├── cifar_05_mlp_baseline.py    #   CIFAR-10 5-class: MLP comparison
│   ├── cifar_06_three_way_compare.py  # CIFAR-10 5-class: RBF vs MLP vs CNN
│   ├── train_01_grouped_fd.py      #   grouped FD: proves inter-group parameter independence
│   ├── train_02_grouped_spsa.py    #   grouped SPSA
│   ├── train_03_analog_spsa.py     #   zero-ADC SPSA: simplest hardware scheme
│   ├── nd_01_baseline.py           #   ND digital baseline
│   ├── nd_02_physical.py           #   physical forward pass verification
│   └── ...（up to nd_09_spsa_benchmark.py, 9 files total）
│
├── outputs/                        # experiment outputs
├── cifar/                          # CIFAR-10 data (downloaded by download_cifar.py, gitignored)
└── reference/                      # references and experiment summaries (gitignored)
```

## Results

### Smoke Test

```bash
python experiments/run_all.py   # quickly validate 10 basic metrics, ~5 min, all PASS
```

`run_all.py` is a quick smoke test that uses small-scale data and reduced training epochs to verify fundamental propositions such as digital-physical consistency, depth effectiveness, noise robustness, and parameter efficiency. **It does not run full CIFAR-10 training or other time-consuming experiments.**

> **Note on reported results**: Due to limited computational resources, some experiments did not reach their optimal configurations in terms of training epochs or parameter counts. For example, the CIFAR-10 experiments at 600 epochs with 2,884 parameters still have room for improvement with more training rounds or more particles. Reported numbers should be viewed as lower-bound certification of method effectiveness, not the architecture's performance ceiling. Users are encouraged to scale up training resources to explore higher accuracy.

### Reproducing Full Experimental Results

Before running CIFAR-10 experiments, download the dataset first (~180MB):

```bash
python experiments/download_cifar.py
```

If you have already downloaded CIFAR-10, you can simply place the extracted data into the `cifar/` directory at the project root.

| Claim to Verify | Command | Estimated Time |
|---|---|---|
| CIFAR-10 5-class RBF best accuracy | `python experiments/cifar_02_regularized.py` | ~30 min |
| CIFAR-10 5-class three-model comparison | `python experiments/cifar_06_three_way_compare.py` | ~60 min |
| MLP same-scale comparison | `python experiments/cifar_05_mlp_baseline.py` | ~1 min |
| Regression depth effectiveness (direct residual) | `python experiments/reg_02_direct_residual.py` | ~5 min |
| Physical noise robustness | `python experiments/reg_03_noise_robustness.py` | ~2 min |
| California Housing real data | `python experiments/reg_06_california_v6.py` | ~10 min |
| Grouped parallel feasibility proof | `python experiments/train_01_grouped_fd.py` | ~5 min |
| Zero-ADC SPSA verification | `python experiments/train_03_analog_spsa.py` | ~2 min |

### Key Results

> Every row below except the last two is **superseded**; see the status box at the
> top and [`FINDINGS.md`](FINDINGS.md). Kept for provenance, not for citation.

| Experiment | Task | Metric | RBF | MLP | CNN |
|------------|------|--------|-----|-----|-----|
| CIFAR-10 5-class | 32×32 grayscale | Accuracy | **39.3%** (2884p) *(test-set model selection — [FINDINGS.md](FINDINGS.md) §9.4)* | 34.0% (4295p) / 38.7% (33644p) | 61.3% (2632p) |
| Digital-Physical Consistency | Various dims & particle counts | Max Error | **7.80×10⁻⁷** *(float32 + PU epsilon, not physical fidelity — §9.2)* | — | — |
| Regression Depth | 2D nonlinear function | R² Gain | +0.064 (direct residual) vs +0.013 (Proto) *(not reproduced; `residual_mode="direct"` now exists in `training/himal_nd.py` so it can be re-measured)* | — | — |
| Regression Real Data | California Housing | R² | 0.530 | 0.551 | — |
| Noise Robustness | 4 circuit noise levels | Accuracy Drop | 0.010 *(measured on the training set — §6.2 supersedes)* | — | — |
| Parameter Efficiency | 4D synthetic classification | Accuracy/Params | 0.81 / 595p | 0.88 / 17410p (29×) | — |

**Current results** (measured, reproducible — see [`FINDINGS.md`](FINDINGS.md)):

| Result | Metric | Value |
|---|---|---|
| Calibrate strengths + measure geometry, vs searching either | measurements to reach R² = 0.30 | **15–30× fewer** |
| Same, at 3 000 measurements | test R² | **+0.39 / +0.53 / +0.54** vs ≤ +0.06 for every search arm |
| Capacity split, equal measurement cost | winner | readout, 8/8 architecture×protocol cells (parameter-count rule) |
| Particle independence | activation coupling | exact 0 without normalisation; 0.063–0.261 under PU |
| Second substrate | kernel vs independent PDE solve | 1.62e-05 (the solver's own error) |
| Calibration robustness | device levels survived | 8/8, while charged 2.11× the measurement budget |

**CIFAR-10 5-Class Core Comparison** *(superseded — test-set model selection; kept for provenance)*:

| Model | Params | Test Acc | Train Acc | Generalization Ratio | Fully Analog Deployable |
|-------|--------|----------|-----------|----------------------|:---:|
| RBF | 2,884 | 39.3% | 53.6% | 73% | ✓ |
| MLP (same-scale) | 4,295 | 34.0% | 99.6% | 34% | ✗ |
| MLP (ample) | 33,644 | 38.7% | 97.8% | 40% | ✗ |
| CNN | 2,632 | 61.3% | 100.0% | 61% | ✗ |

RBF is the only solution deployable entirely via analog circuits. The same-scale MLP suffers from severe overfitting. MLP only approaches RBF's test accuracy after scaling up to 11.7× the parameters.

## License

MIT License

## Citation

```bibtex
@article{singal-node-physics-network,
  title={Physical-Domain RBF Particle Neural Network},
  year={2026}
}
```
