# Physical-Domain RBF Particle Neural Network

A neural network where each neuron is a movable particle in D-dimensional space. Forward propagation, loss computation, and parameter training are all mapped to analog circuit components, exploring the feasibility of implementing the full neural network pipeline — from training to inference — entirely within a physical system simulation.

## Core Idea

Two challenges prevent standard neural networks from running efficiently on analog hardware: nonlinear activations (ReLU, softmax) are costly in circuits, and backpropagation requires a full computation graph with exact gradients, which is physically infeasible in analog systems.

This project's approach: **design the network around operations that analog circuits naturally perform well, and replace backpropagation with finite differences or zero-order optimization.**

### "Activate Then Combine" vs Standard Networks

A standard network computes `activate(combine(x))` — first a weighted sum, then a nonlinearity. This network reverses the paradigm: `combine(activate(x))` — each particle independently produces a Gaussian response to the input (activation), then all particle responses are linearly superimposed (combination). Nonlinearity no longer comes from a fixed activation function, but from the spatial relationship between particle and input: the farther the input point from a particle, the weaker that particle's contribution.

### Particle Source Superposition Model

The network is understood as a set of **particle sources** in D-dimensional space: input X is a probe point position, particle centers are source positions, Gaussian activations are field strengths excited at the probe point, and weights are receiver sensitivities. Network output = weighted superposition of all source field strengths. **Training is equivalent to the physical evolution of sources migrating in space and adaptively adjusting their influence ranges.**

This design makes the network an interpretable physical field superposition system rather than a "black-box function approximator." The signal chain maps not only to analog circuits but also to optical systems (Gaussian beam superposition), wave fields (point-source Green's function superposition), diffusion fields (heat kernel superposition), and more. See [THEORY.md](THEORY.md) Section 0 for details.

## Current Status

- ND architecture supports arbitrary input dimensions; 2D as a special case
- Classification and regression share the same physical component set
- Training, forward, and inference all in physical domain — zero PyTorch autograd
- **Classification (moons, FD)**: L1=0.92 → Full=0.94 (+0.02 depth contribution)
- **Regression (FD residual)**: R²=0.121→0.125, proto_y magnitude 0.92→0.77→0.76 (effective residual)
- **Digital-physical consistency**: max|error| = 7.80×10⁻⁷ (well below 10⁻⁴ threshold)
- **Noise robustness**: none→high accuracy drops only 0.010 (4 noise levels)
- **Parameter efficiency**: RBF 595 params/0.81 vs MLP 17410 params/0.88 (29× fewer params, within 10% accuracy)
- **Digital baselines**: V1 RBF acc=0.97 (140p), V3 ND acc=0.82 (133p)
- **Depth ablation**: equal 9 particles: 1L and 3L both train successfully (depth benefit proven in Part 2: +0.020)
- `experiments/run_all.py` verifies all 9 experiments in one run (ALL PASSED)

## Project Structure

```
singal-node-physics-network/
├── physical_simulator/              # Low-level analog circuit components
├── simulator/                       # ND physical simulator (active)
│   ├── layer_nd.py                  #   PhysicalRBFLayerND
│   └── network_nd.py                #   PhysicalResidualRBFNetworkND
├── network/                         # Network definitions
│   └── combined.py                  #   CombinedHiMALRBF — unified entry point
├── training/                        # Training algorithms
│   ├── himal_nd.py                  #   HiMAL-ND — primary trainer
│   └── ...                          #   (V1 trainers for reference)
├── experiments/                     # Experiments
│   ├── run_all.py                   #   Run full 9-part verification suite
│   ├── e11_fd_cls.py                #   FD classification depth proof
│   ├── e11_fd_reg.py                #   FD regression depth proof
│   ├── e01-e10_*.py                 #   V3 ND experiments (standalone)
│   ├── 01-13_*.py                   #   V1 2D reference experiments (standalone)
│   └── benchmarks/                  #   sklearn baseline comparison
├── outputs/                         # Experiment outputs
└── THEORY.md                        # Full theoretical derivation
```

## Architecture

### Particle Model

Each particle has a center `centers (D, N)` and width `h` in D-dimensional space. Given input `X (batch, D)`, particle activations are Gaussian kernel responses. All parameters (centers, widths, weights) are trainable via finite differences.

### Three-Layer Structure

```
Input X (batch, D)
    -> Layer1: PhysicalRBFLayerND -> output (batch, d1)
    -> Layer2: PhysicalRBFLayerND -> output (batch, d2)  [residual accumulation]
    -> Layer3: PhysicalRBFLayerND -> output (batch, d3)  [residual accumulation]
    -> Final output
```

Inter-layer connection uses **wire-direct**: each layer's multi-dimensional output passes directly as the next layer's multi-dimensional input.

### Physical Signal Chain (6 components)

```
Input voltage -> differential_pair(dx) -> gilbert_square(d^2) -> kirchhoff_sum
              -> bjt_gaussian(exp) -> current_mode_normalize -> memristor_mvm(W.z) -> output
```

### HiMAL-ND Training

**HiMAL (Hierarchical Multi-Alignment Learning)** is adapted here as a layer-wise dual-network training method for RBF networks. Its core philosophy is replacing global optimization with structured decomposition:

- Data is partitioned by median value; A/B subnetworks each learn different regions
- Each layer is trained independently then frozen; transition-zone alignment loss (replacing cross-layer backpropagation) ensures output consistency at the interface
- Classification uses prototype-guided loss (no softmax/log/exp)
- Regression uses residual training: L2/L3 learn what earlier layers could not fit
- Gradients via per-parameter finite differences (FD) or SPSA — zero autograd

## Experiments

### V3 Core Experiments (ND architecture, pure physical domain)

| Experiment | Purpose |
|------------|---------|
| **e11_fd_cls** | FD classification depth proof: multi-seed moons, L1→Full accuracy progression |
| **e11_fd_reg** | FD regression depth proof: residual training + decreasing proto_y magnitude |
| e01_nd_baseline | ND digital baseline: 2D+4D correctness verification |
| e02_nd_physical | Physical forward validation: digital-physical output consistency |
| e03_physical_training | Pure physical FD proto training from random initialization |
| e04-e05 | HiMAL pipeline debug and layer contribution analysis |
| e06_ablation | Single vs dual network, layer count ablation |
| e07-e08 | Physical network verification and comprehensive benchmarking |
| e09_spsa_benchmark | SPSA vs FD speed/accuracy comparison |
| e10_noise_robustness | Noise robustness under 4 noise levels |
| benchmarks/ | sklearn baseline comparison |

### V1 Reference Experiments (2D architecture)

Experiments 01-13 validate the original 2D design: digital baseline, physical forward, noise tolerance, SPSA/HiMAL training, 2D classification with decision boundaries, projected ensembles, physical FD fine-tuning, and depth ablation. These serve as historical reference for the ND upgrade.

## Training Methods

| Method | Backprop | Physical Feasibility |
|--------|----------|---------------------|
| SPSA zero-order | Not needed | High — only loss measurement required |
| HiMAL-ND layer-wise | Not needed | High — per-layer, transition alignment |
| Per-parameter FD | Not needed | Highest — exact gradient, physically measurable |
| Proto-guided physical | Not needed | Highest — zero digital pre-training |

## Quick Start

```bash
pip install torch numpy matplotlib scipy pandas scikit-learn
python experiments/run_all.py
```

## Theory

[THEORY.md](THEORY.md) provides full derivation: ND particle model, physical signal chain, HiMAL-ND training, residual learning.

## Disclaimer

This project was conducted under limited experimental conditions. Because the physical system (analog circuits, memristor crossbars, etc.) is simulated numerically, each forward pass and parameter update must compute through the full physical signal chain, incurring substantially higher computational cost than standard matrix-multiplication-based neural networks. Constrained by available computing power, the current experiments use modest parameter counts and training epochs, and are validated only on simple datasets such as moons, 4D synthetic classification, and sine regression.

The core objective of this project is to **propose a new network architecture paradigm** — the "activate-then-combine" particle source superposition model — that endows neural networks with better interpretability and enables complete training and inference on physical platforms such as analog circuits, optical systems, and wave fields. The nine verification experiments (`run_all.py`) cover key propositions including digital-physical consistency, depth effectiveness, noise robustness, and parameter efficiency, and can serve as a baseline for further research.

Due to limitations in the author's coding ability and computational resources, shortcomings may exist. Users with better resources are encouraged to adapt the network architecture, scale up parameter counts and training epochs, conduct further experiments on more complex datasets, or validate this architecture on real physical hardware. We hope the ideas and code in this project are helpful to you.

## License

MIT
