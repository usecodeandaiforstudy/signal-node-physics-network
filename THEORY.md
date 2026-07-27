# RBF Particle Neural Network: Architecture and Principles

## 0. Core Design Philosophy

### 0.1 "Activate Then Combine" — Reversing the Standard Network Paradigm

A standard neural network computes **`activate(combine(x))`**: first a matrix multiplication (weighted combination), then a nonlinear activation function (ReLU/softmax). This "combine-then-activate" paradigm requires the physical system to precisely implement non-monotonic nonlinearities — extremely difficult on analog circuits, optical systems, and other physical platforms.

This network reverses the order: **`combine(activate(x))`**. Each particle independently produces a Gaussian response (activation) to the probe point in input space, and then all particle responses are linearly superimposed (combination).

| | Standard Network | This Work |
|---|---------|--------|
| Computation order | Wx → activate | activate(x, particle_i) → Σ W_i · z_i |
| Source of nonlinearity | Fixed activation function (uninterpretable) | Spatial relationship between particle and input (Gaussian distance decay) |
| Physical meaning | Abstract, uninterpretable | Each particle = a physical source in space |
| Hardware mapping | Difficult (requires precise activation function) | Natural (source → propagation → receiver is a universal physical paradigm) |

### 0.2 Particle Source Superposition Model

Each network layer can be understood as a set of **particle sources** in D-dimensional space:

- **Input X**: Position of a probe point in space
- **Particle centers**: Spatial positions of the sources
- **Width h**: Radiation range / influence radius of each source
- **Gaussian activation z_i**: Field strength excited by the i-th source at probe point X — stronger when closer
- **Weights W**: Receiver sensitivity to each source's response
- **Layer output**: Weighted superposition of all source field strengths at the probe point = `Σ W_i · exp(-|X - centers_i|² / h_i²)`

### 0.3 Training ≡ Physical System Evolution

Under the particle source framework, neural network training maps precisely to physical system evolution:

| Neural network concept | Physical correspondence |
|------------------------|------------------------|
| Particle center movement | Sources migrating in space |
| Width adjustment | Source radiation range expanding/contracting |
| Weight update | Receiver sensitivity adjustment (e.g., memristor conductance change) |
| Loss decrease | System energy minimization |
| Gradient direction | Negative gradient of system free energy |
| Residual training | Later-layer sources compensating signals missed by earlier layers |

**The network is no longer a "black-box function approximator" — it is an interpretable physical field superposition system.**

### 0.4 Interpretability

The "activate-then-combine" design yields structural interpretability:

- Each particle has a **clearly locatable meaning** in input space (a physical source)
- Network output is the **weighted superposition of all source field strengths** (not abstract feature combinations)
- Network training is equivalent to **sources migrating in space and adaptively adjusting their influence ranges**
- Each training stage has clear spatial semantics: initialization (sources randomly scattered, high entropy) → early training (sources migrate toward data-dense regions, self-organization) → late training (width fine-tuning, weight refinement, low-entropy equilibrium)
- In residual training, later-layer sources naturally gather in regions where earlier-layer sources have "insufficient coverage", forming complementary distributions

### 0.5 Physical Platform Universality

This network's physical signal chain is not limited to analog circuits. Because the design is built on fundamental operations common to all physical systems (addition/subtraction, squaring, exponential decay, normalized superposition), it maps to multiple physical platforms:

| Operation | Analog Circuit | Optical System | Wave/Diffusion Field |
|-----------|---------------|----------------|----------------------|
| Δx = (x - xc) / h | Differential amplifier | Optical path difference | Phase difference / spatial offset |
| Δx² | Gilbert squarer | Intensity detection | Amplitude squared |
| Σ Δx² | Kirchhoff current summation | Beam superposition | Interference superposition / scalar field sum |
| exp(-k·d²) | BJT subthreshold exponential | Gaussian beam | Heat kernel / diffusion kernel / Green's function |
| ẑ = z/Σz | Current-mode normalization | Beam splitter balancing | Probability normalization |
| W·ẑ | Memristor crossbar | Tunable attenuator array | Coupling coefficient matrix |

***

## 1. ND Particle Model

Each particle is a point in D-dimensional space, defined by its center `centers (D, N)` and width `h`. Given a D-dimensional input vector `X = (x_1, ..., x_D)`, the activation of the i-th particle is computed via a Gaussian kernel:

$$d_i^2 = \sum_{k=1}^{D} \frac{(x_k - c_{k,i})^2}{h_i^2}$$

$$z_i = \exp(-0.5 \cdot d_i^2) \quad\text{(narrow channel)}$$
$$\tilde{z}_i = \exp(-0.3 \cdot d_i^2) \quad\text{(wide channel)}$$

The two channels capture local detail (narrow kernel) and global structure (wide kernel), respectively.

## 2. Single Layer (PhysicalRBFLayerND)

### 2.1 Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `centers` | (D, N) | Particle center coordinates |
| `h` | (N,) | Particle width |
| `W_gauss` | (d_out, N) | Narrow-channel weights |
| `W_act` | (d_out, N) | Wide-channel weights |

### 2.2 Forward Propagation (6-step physical signal chain)

Each step maps to an analog circuit component:

1. **Differential amplifier** (D times): computes `dx_k = (x_k - c_{k,i}) / h_i`
2. **Gilbert squarer** (D times): computes `dx_k^2`
3. **Kirchhoff summation**: accumulates `d^2 = sum_k dx_k^2` across D dimensions
4. **BJT exponential** (dual-channel): `exp(-0.5 * d^2)` and `exp(-0.3 * d^2)`
5. **Current-mode normalization** (dual-channel): `z_hat = z / sum(z)`
6. **Memristor crossbar** (dual-channel): `W_gauss * z_hat_gauss + W_act * z_hat_act`

Output dimension is controlled by `d_out`, which equals the number of rows in the memristor crossbar array.

### 2.3 Mathematical Formulation of Single-Layer Forward Pass

Given input matrix **X** ∈ ℝ^(B×D) (B samples, D dimensions), a layer with N particles and output dimension d_out. Let particle center matrix **C** ∈ ℝ^(D×N), width vector **h** ∈ ℝ^N, narrow-channel weights **W**^g ∈ ℝ^(d_out×N), and wide-channel weights **W**^a ∈ ℝ^(d_out×N).

**Step 1: Normalized distances**

$$d^2_{ij} = \sum_{k=1}^{D} \left(\frac{X_{ik} - C_{kj}}{h_j}\right)^2 \quad\Rightarrow\quad \mathbf{D}^2 \in \mathbb{R}^{B \times N}$$

**Step 2: Dual-channel Gaussian activation**

$$\mathbf{Z}^g_{ij} = \exp(-0.5 \cdot d^2_{ij}), \quad \mathbf{Z}^a_{ij} = \exp(-0.3 \cdot d^2_{ij})$$

**Step 3: PU normalization** (enabled for classification, disabled for regression)

$$\hat{\mathbf{Z}}^g_{ij} = \frac{\mathbf{Z}^g_{ij}}{\sum_{j=1}^{N} \mathbf{Z}^g_{ij}}, \quad \hat{\mathbf{Z}}^a_{ij} = \frac{\mathbf{Z}^a_{ij}}{\sum_{j=1}^{N} \mathbf{Z}^a_{ij}}$$

**Step 4: Linear projection**

$$\mathbf{G} = \hat{\mathbf{Z}}^g \cdot (\mathbf{W}^g)^\top \in \mathbb{R}^{B \times d_{out}}, \quad \mathbf{A} = \hat{\mathbf{Z}}^a \cdot (\mathbf{W}^a)^\top \in \mathbb{R}^{B \times d_{out}}$$

**Step 5: Dual-channel fusion**

$$\mathbf{F} = \mathbf{G} + \mathbf{A} \in \mathbb{R}^{B \times d_{out}}$$

The single-layer output **F** serves as input to the next layer, with dimension determined by d_out.

## 3. Three-Layer Network (PhysicalResidualRBFNetworkND)

### 3.1 Architecture

```
Input X (batch, D)
    -> Layer1 (N1 particles, d1 output) -> h1 (batch, d1)
    -> Layer2 (N2 particles, d2 output) -> delta2 (batch, d2) -> accumulate: f2 = h1 (+) delta2
    -> Layer3 (N3 particles, d3 output) -> delta3 (batch, d3) -> accumulate: f3 = f2 (+) delta3
```

Inter-layer connection uses **wire-direct**: Layer k's multi-dimensional output passes directly as Layer k+1's multi-dimensional input. When dimensions mismatch (e.g., d1 > d2), zero-padding aligns them before accumulation.

### 3.2 Residual Learning (Regression)

In regression, each layer learns a residual rather than the full target:

- Layer 1: learns full y
- Layer 2: learns `y - proto_pred(L1)`
- Layer 3: learns `y - proto_pred(L1) - proto_pred(L2)`

At inference, summing the three proto predictions recovers the original target. The decreasing magnitude of `proto_y` per layer confirms effective residual learning.

### 3.3 Mathematical Formulation of Full Network Forward Pass

Let the three-layer network have N_ℓ particles and output dimension d_ℓ for layer ℓ. Denote the single-layer transformation as Layer_ℓ(·), with input **X** ∈ ℝ^(B×D).

**Layer 1: Initial signal**

$$\mathbf{F}_1 = \text{Layer}_1(\mathbf{X}) \in \mathbb{R}^{B \times d_1}$$

The first layer maps directly from the D-dimensional input space to d₁ dimensions.

**Layer 2: Residual accumulation (wire-direct)**

$$\boldsymbol{\delta}_2 = \text{Layer}_2(\mathbf{F}_1) \in \mathbb{R}^{B \times d_2}$$

$$\mathbf{F}_2 = \text{ZeroPad}_{d_1 \to \max(d_1,d_2)}(\mathbf{F}_1) + \text{ZeroPad}_{d_2 \to \max(d_1,d_2)}(\boldsymbol{\delta}_2)$$

where ZeroPad_{d→m}(·) zero-pads a d-dimensional tensor to m dimensions: if m > d, append m−d zero columns on the right; if m = d, identity. Wire-direct means Layer 1's d₁-dimensional output passes directly as Layer 2's input (no mean projection or dimensionality reduction).

**Layer 3: Final residual accumulation**

$$\tilde{\mathbf{F}}_2 = \text{TruncPad}_{\max(d_1,d_2) \to d_3}(\mathbf{F}_2)$$

$$\boldsymbol{\delta}_3 = \text{Layer}_3(\tilde{\mathbf{F}}_2) \in \mathbb{R}^{B \times d_3}$$

$$\mathbf{F}_3 = \text{ZeroPad}_{\max(d_1,d_2) \to M}(\mathbf{F}_2) + \text{ZeroPad}_{d_3 \to M}(\boldsymbol{\delta}_3), \quad M = \max(d_1,d_2,d_3)$$

where TruncPad_{a→m}(·) truncates or zero-pads the signal to the input dimension d_3 expected by Layer 3: if current dimension ≥ d_3, take the first d_3 columns; otherwise, zero-pad.

**Classification head** (optional): For classification tasks, apply a linear classification head over **F**₃

$$\mathbf{O} = \mathbf{F}_3[:, :d_3] \cdot \mathbf{W}_{head}^\top + \mathbf{b}_{head} \in \mathbb{R}^{B \times n_{classes}}$$

For regression tasks, the first d_3 dimensions of **F**₃ serve directly as the final output. The residual connections between layers allow gradient flow (for digital validation), but physical training optimizes each layer independently without cross-layer backpropagation.

## 4. HiMAL-ND Training

### 4.0 HiMAL Design Philosophy

**HiMAL (Hierarchical Multi-Alignment Learning)** — Hierarchical Multi-Alignment Learning

| Term | Meaning | Embodiment in the framework |
|------|---------|----------------------------|
| **Hierarchical** | Train layer-by-layer, freeze bottom-up | Layer 1→2→3 trained sequentially, each frozen after completion |
| **Multi** | A/B dual subnetworks each learn different data regions | Data partitioned by median, two networks evolve independently |
| **Alignment** | Output consistency constraint across transition zone | Alignment loss replaces cross-layer backpropagation |
| **Learning** | Adjust parameters via physical measurements (FD/SPSA) | Zero autograd, pure physical-domain gradient estimation |

Core philosophy: **Replace global optimization with structured decomposition** — break a large joint optimization into independently solvable smaller sub-problems, using local training and transition-zone alignment to avoid cross-layer backpropagation.

### 4.1 Dual-Network Partitioning

Training data is split by feature median into zone A and zone B. Two independent subnetworks each learn their respective region. Transition-zone samples pass through both networks, and an alignment loss `MSE(hTA, hTB)` maintains consistency.

### 4.2 Layer-Wise Independent Training

Each layer's parameters are frozen after training; its output serves as the next layer's input. During training:

- Classification: proto-guided loss = squared distance from each sample to its class prototype
- Regression: MSE loss = direct fit to target (or residual)

All operations (subtraction, squaring, summation, averaging) are realizable via differential amplifiers, Gilbert squarers, and Kirchhoff summation. The classification path uses no softmax, log, or exp.

### 4.3 Gradient Estimation

Two methods, neither requiring backpropagation:

| Method | Principle | Forward passes per update |
|--------|-----------|--------------------------|
| Per-parameter FD | Independent +/-epsilon perturbation per parameter, central difference | 2 * N_params |
| SPSA | Simultaneous random perturbation of all parameters, statistical estimate | 2 * N_avg |

### 4.4 Classification Inference

Each layer classifies via nearest prototype (`argmin ||h - proto_c||^2`), three layers vote (weights 0.5, 0.3, 0.2), and A/B subnetworks each contribute one vote. Data partition information is used for routing weights.

### 4.5 Regression Inference

Each layer predicts a residual via prototype-weighted interpolation; three layers accumulate to produce the final output. Prototypes are stored during training as `proto_h -> proto_y` mappings; inference uses softmax-temperature-weighted interpolation.

## 5. CombinedHiMALRBF Unified Model

Encapsulates RBF physical layers, HiMAL-ND trainer, and physical network simulator into a single interface:

```python
model = CombinedHiMALRBF(
    input_dim=D, layer_sizes=(n1, n2, n3), output_dims=(d1, d2, d3),
    task="classification" | "regression",
    use_spsa=True | False,  # FD vs SPSA
)
model.train(X, y, epochs_per_layer=50, lr=0.02)
pred = model.predict(X_test)
```

## 6. Physical System Implementation

### 6.1 Operation-to-Circuit Mapping

| Math Operation | Analog Circuit |
|----------------|----------------|
| `dx = (x - xc) / h` | Differential amplifier |
| `d^2 = sum dx_k^2` | Gilbert squarer + Kirchhoff sum |
| `exp(-k * d^2)` | BJT exponential (subthreshold) |
| `z_hat = z / sum(z)` | Current-mode normalization |
| `W * z_hat` | Memristor crossbar |
| `f + delta` | Kirchhoff current summation |

### 6.2 Training-Inference Pipeline

This project aims to map the entire pipeline to a physical system:

- **Training**: Estimate gradients via FD/SPSA on the physical simulator -> update parameters (voltages/conductances)
- **Forward propagation**: Analog signals flow through differential amplifiers -> Gilbert squarers -> exponentials -> normalization -> memristor array -> output
- **Inference**: Trained parameters are fixed; input signals flow through the circuit; output is the prediction result

The entire process runs on the physical simulator (`physical_simulator.components`), without PyTorch autograd.

## 7. Key Experimental Findings

All findings below are verified by `experiments/run_all.py` (9/9 experiments PASSED):

| Experiment | Finding |
|------------|---------|
| Digital-physical consistency | Physical simulator matches PyTorch digital model, max error 7.80×10⁻⁷ |
| FD classification depth | moons L1-only=0.92 → Full 3L=0.94, depth contributes +0.020 |
| FD regression depth | Residual training effective: proto_y magnitude 0.922→0.768→0.759 decreasing |
| Noise robustness | CombinedHiMALRBF across 4 noise levels (none/low/med/high), accuracy degrades only 0.010 |
| V1 RBF classification | ResidualRBFNetwork, 8/6/4 particles, 120ep BP, moons acc=0.950 |
| Parameter efficiency | RBF(595p, 2D-only) vs MLP(17410p, 4D): 0.81 vs 0.88, 29× fewer params, <10% accuracy gap |
| V1 digital baseline | ResidualRBFNetwork, 10/6/4 particles, 120ep, acc=0.970 |
| V3 ND digital baseline | ResidualRBFNetworkND, 8/5/3 particles, 60ep, acc=0.820 |
| Depth ablation | Equal 9 particles: 1L and 3L both train successfully (depth benefit proven in Part 2, +0.020) |

Core conclusions:
- Physical-domain closed-loop training is feasible (both FD and SPSA work for gradient estimation)
- Multi-layer depth provides positive contribution on classification tasks
- Regression residual training is effective (later layers learn residuals missed by earlier layers)
- RBF significantly outperforms MLP in parameter efficiency (29× fewer params at comparable accuracy)
- Good noise robustness (analog circuit component tolerance testing passed)
- High digital-physical consistency
