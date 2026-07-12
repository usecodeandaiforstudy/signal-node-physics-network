# RBF Particle Neural Network

A neural network architecture where each neuron is a movable particle in 2D space. Every mathematical operation has a direct analog circuit equivalent — designed from the ground up so that the entire network can run natively on a CMOS + memristor chip.

## Why This Matters

Standard neural networks cannot run efficiently on analog hardware because their core operations (ReLU, softmax, matrix multiply chains) are expensive or impossible to implement directly in circuits. More fundamentally, **backpropagation** — the algorithm that trains all modern neural networks — requires storing a full computation graph and computing exact gradients, which is physically infeasible in analog systems.

This project takes the opposite approach: **design the network architecture around what analog circuits naturally do well**, then develop training methods that don't require backpropagation.

## Key Results

| Metric | Value |
|---|---|
| **Parameter efficiency** | ~150 params achieve ~92% on 4D classification; MLP needs ~1,300 params to match (~9x advantage) |
| **Naturally 2D data** | 100% on circles, 99.3% on moons — with just 4 particles (69 params) |
| **Device estimate** | ~100 transistors + 12 memristors for optimal 3-layer config (6/4/3 particles) |
| **Power estimate** | μW range (subthreshold operation) |
| **Inference latency** | Nanosecond range (purely analog signal path) |
| **Noise robustness** | < 2% accuracy drop from "none" to "high" noise levels |
| **Training methods** | 3 variants: standard backprop, SPSA (2 measurements/iter), HiMAL (no cross-layer gradients) |

Minimal configurations discovered via ablation: increasing particles from 4 to 25 yields zero accuracy gain on 2D benchmarks, only parameter bloat.

## Design

Each neuron is a triplet `(xc, yc, h)` — a 2D coordinate and an influence radius. All four parameters per particle are trainable. The forward pass uses only six primitive operations:

```
Operation              Circuit Equivalent              Device Cost
───────────────────────────────────────────────────────────────────
(x - xc) / h         → Differential amplifier           ~4 transistors
dx² + dy²            → Gilbert multiplier               ~6 transistors
exp(-k · d²)         → BJT subthreshold I-V             1-2 transistors
z / Σz               → Current mirror divider           4 transistors + wire
W · z                → Memristor crossbar               passive (memristors only)
f + δ                → Wire junction (Kirchhoff)        0 devices
```

Three such layers are stacked with residual connections, making a **ResidualRBFNetwork**. A high-dimensional extension (**ProjectedRBFEnsemble**) uses fixed random projections to map D-dimensional data onto multiple 2D subspaces, each processed by an independent RBF network and averaged.

## Project Structure

```
├── rbf_network/              # Core network
│   ├── core.py               # RBFNetwork / ResidualRBFNetwork
│   └── utils.py              # Training tools / visualization / data generation / regularization
│
├── physical_simulator/       # Physical component simulation
│   ├── components.py         # 6 components with non-ideality modeling
│   ├── layer.py              # PhysicalRBFLayer
│   └── network.py            # PhysicalResidualRBFNetwork (3-layer residual)
│
├── forward_training/         # Forward-only training (no backpropagation required)
│   ├── spsa.py               # SPSA zero-order optimizer
│   ├── himal.py              # HiMAL layer-wise dual-network training
│   └── twin.py               # Digital twin proxy-gradient training
│
├── experiments/              # Physical feasibility verification experiments
│   ├── 01_digital_baseline.py
│   ├── 02_physical_forward.py
│   ├── 03_noise_tolerance.py
│   ├── 04_spsa_training.py
│   ├── 05_himal_training.py
│   ├── 06_2d_classification.py
│   ├── 07_projected_ensemble.py
│   └── run_all.py
│
├── outputs/                  # Experiment results (figures, tables)
├── THEORY.md / THEORY_CN.md  # Full mathematical description
├── README_CN.md              # Chinese version
└── .gitignore
```

## Quick Start

```bash
pip install torch numpy matplotlib scipy pandas scikit-learn
python experiments/run_all.py
```

## Training Methods

The project provides **three** ways to train the network, each progressively less dependent on digital infrastructure:

| Method | Requires Backprop? | Forward Passes / Iter | Physical Feasibility |
|---|---|---|---|
| **Standard (autograd)** | Yes | 1 | Digital only |
| **SPSA** | No | 2 | High — only needs loss measurement |
| **HiMAL** | No | 2 × N_params (per layer, ~150) | High — layer-wise, independent |

## Experiment Proof Chain

```
1. Digital baseline    → 96%+ on Moons, 98%+ R² on rotation (20 particles)
2. Physical forward    → Simulator output correlates >0.95 with digital model
3. Noise tolerance     → Accuracy stable across none/low/medium/high noise levels
4. SPSA training       → ~95% of backprop accuracy using only 2 forward measurements/iter
5. HiMAL training      → Layer-wise training reaches ~90% of end-to-end accuracy
6. 2D classification   → 100% circles, 99.3% moons (4 particles, 69 params) with decision boundary plots
7. Projected ensemble  → ~92% on 4D data (~150 params); MLP needs ~1,300 params to match
```

**All 7 pass → RBF particle network can realize the complete training + inference closed loop in a pure analog physical system.**

## Theoretical Background

A companion document ([THEORY.md](THEORY.md) / [THEORY_CN.md](THEORY_CN.md)) provides the full mathematical derivation: particle activation functions, forward propagation, PU normalization, residual connections, gradient flow, and the high-dimensional projection extension.

## License

MIT
