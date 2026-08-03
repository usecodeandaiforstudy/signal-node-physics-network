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

## 3. Multi-Layer Residual Network (PhysicalResidualRBFNetworkND)

### 3.1 Architecture

```
Input X (batch, D)
    -> Layer1 (N1 particles, d1 output) -> h1 (batch, d1)
    -> Layer2 (N2 particles, d2 output) -> delta2 (batch, d2) -> accumulate: f2 = h1 (+) delta2
    -> Layer3 (N3 particles, d3 output) -> delta3 (batch, d3) -> accumulate: f3 = f2 (+) delta3
```

Inter-layer connection uses **wire-direct**: Layer k's multi-dimensional output passes directly as Layer k+1's multi-dimensional input. When dimensions mismatch (e.g., d1 > d2), zero-padding aligns them before accumulation.

### 3.2 Residual Learning (Regression): Proto Residual vs Direct Residual

In regression, there are two residual learning paradigms. The key finding comes from the reg_01→reg_02 experimental comparison:

**Proto Residual**: each layer predicts via softmax temperature-weighted interpolation

$$\delta_\ell = y - \text{softmax}\left(-\frac{\|h - \text{proto}\|}{T}\right) \cdot y_{\text{proto}}$$

- Layer 1: learns full y
- Layer 2: learns `y - proto_pred(L1)`
- Layer 3: learns `y - proto_pred(L1) - proto_pred(L2)`

**Direct Residual**: each layer uses the first component of the hidden output as its prediction

$$\delta_\ell = y - h_{[:,0]}$$

| Property | Proto Residual | Direct Residual |
|----------|---------------|-----------------|
| Mechanism | Soft clustering + weighted averaging | Exact per-sample error |
| High-frequency residual info | Smoothed away (soft clustering loses fine-grained differences) | Fully preserved |
| Depth gain | +0.013 (minimal) | **+0.064 (5× improvement)** |
| Convergence behavior | Diminishing returns with depth | L2/L3 can learn any residual pattern |

**Mathematical analysis of why Proto Residual fails**:

Proto residual interpolates prototype predictions via softmax temperature weighting, which is fundamentally a low-pass filtering operation. Let the hidden output of layer ℓ be h, and the prototype set be {p₁, ..., p_K} (K cluster centers). The prediction is:

$$\hat{y}^{\text{proto}} = \sum_{k=1}^{K} s_k \cdot y_k^{\text{proto}}, \quad s_k = \frac{\exp(-\|h - p_k\|/T)}{\sum_j \exp(-\|h - p_j\|/T)}$$

Here {y_k^proto} are the prototype label values. Due to the smoothing property of softmax, when K ≪ number of samples, $\hat{y}^{\text{proto}}$ can express at most K distinct output patterns. Consequently, the high-frequency per-sample difference information in the residual $y - \hat{y}^{\text{proto}}$ is severely attenuated, and subsequent layers can extract very limited learnable signal — depth gain is only +0.013.

**Why Direct Residual works**:

Direct residual $y - h_{[:,0]}$ preserves the complete per-sample error information. $h_{[:,0]}$ is a continuous-valued function that can directly fit targets to arbitrary precision, so the residual contains all information that Layer 1 failed to capture from the original problem. Layer 2 and Layer 3 particles can specifically cover regions where Layer 1 has large errors, forming complementary particle distributions, yielding a depth gain of +0.064 (5× that of Proto residual).

### 3.3 Mathematical Formulation of Full Network Forward Pass

Let the ℓ-layer residual network have N_k particles and output dimension d_k for layer k. Denote the single-layer transformation as Layer_k(·), with input **X** ∈ ℝ^(B×D). The formulation below uses 3 layers as an example; the architecture supports arbitrary depth.

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
| Per-parameter FD | Independent +/-epsilon perturbation per parameter, central difference | 2 × N_params |
| SPSA | Simultaneous random perturbation of all parameters, statistical estimate | 2 × N_avg |

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

All findings below are verified by `experiments/run_all.py` (10/10 experiments PASSED):

### 7.1 Core Validation Experiments

| Experiment | Finding |
|------------|---------|
| Digital-physical consistency | Physical simulator matches PyTorch digital model, max error 7.80×10⁻⁷ |
| FD classification depth | moons L1-only=0.92 → Full 3L=0.94, depth contributes +0.020 |
| FD regression depth | Residual training effective: proto_y magnitude 0.922→0.768→0.759 decreasing |
| Noise robustness | CombinedHiMALRBF across 4 noise levels (none/low/med/high), accuracy degrades only 0.010 |
| 2D RBF classification (reference) | ResidualRBFNetwork, 8/6/4 particles, 120ep BP, moons acc=0.950 |
| Parameter efficiency | RBF(595p, 2D-only) vs MLP(17410p, 4D): 0.81 vs 0.88, 29× fewer params, <10% accuracy gap |
| 2D RBF digital baseline (reference) | ResidualRBFNetwork, 10/6/4 particles, 120ep, acc=0.970 |
| ND digital baseline | ResidualRBFNetworkND, 8/5/3 particles, 60ep, acc=0.820 |
| Depth ablation | Equal 9 particles: 1L and 3L both train successfully (depth benefit proven in Part 2, +0.020) |

#### ND Core Validation Experiment Details (ND Experiment Group)

| Experiment | Description |
|------------|-------------|
| nd_01 | ND digital baseline: validates ND network correctness on 2D+4D |
| nd_02 | Physical forward verification: PhysicalRBFLayerND output matches digital model |
| nd_03 | Pure physical FD prototype training: converges from random init in physical domain |
| nd_04 | HiMAL-FFA-ND pipeline debugging |
| nd_05 | HiMAL hierarchical contribution comparison |
| nd_06 | Ablation study: single/dual network, layer count impact on accuracy |
| nd_07 | Physical network verification: forward consistency and proto prediction correctness |
| nd_08 | Comprehensive experiments: three configs + noise robustness |
| nd_09 | SPSA vs FD speed/accuracy comparison |

#### Training Experiment Details (Training Experiment Group)

| Experiment | Description |
|------------|-------------|
| train_01 | FD group parallelization: proves independence between parameter groups |
| train_02 | Grouped SPSA training |
| train_03 | Zero-ADC SPSA: validates minimal hardware scheme |

### 7.2 CIFAR-10 Classification Results

| Experiment | Configuration | Accuracy | Key Finding |
|------------|--------------|----------|-------------|
| cifar_01 | First RBF on CIFAR, 3D point-cloud encoding, 2884 particles | 37.3% | Digital-physical gap only 0.000015 |
| cifar_02 | +physical regularization | **39.3% (SOTA)** | Physical regularization adds +2.0 percentage points |
| cifar_03 | Regularization ablation | — | Regularization only helps when needed (overfitting scenarios) |
| cifar_04 | Capacity analysis | — | 2884 parameter count constitutes a capacity bottleneck |
| cifar_05 | MLP control (matched parameters) | 34.0% | MLP severely overfits at same parameter scale |
| cifar_06 | Final comparison: RBF vs MLP vs CNN | RBF 39.3%, MLP 38.7%, CNN 61.3% | RBF outperforms MLP with 11.7× fewer parameters; CNN provides architecture upper bound |

Key conclusion: The RBF particle network achieves 39.3% best accuracy on CIFAR-10 (2884 particles), significantly outperforming MLP at matched parameter scale (11.7× parameter efficiency advantage), with CNN (61.3%) representing the theoretical upper bound at this parameter budget.

### 7.3 Regression Depth Gain

| Experiment | Residual Type | Depth Gain | Notes |
|------------|--------------|------------|-------|
| reg_02 | Direct residual | **+0.064** | h[:,0] as prediction per layer, exact per-sample error |
| reg_01 | Proto residual | +0.013 | softmax temperature-weighted interpolation, high-frequency info smoothed |
| reg_06 | California Housing (Proto residual) | R²=0.530 | Proto residual baseline |

The depth gain comparison is definitive: **direct residual's information fidelity far exceeds proto residual**, with its 5× improvement coming from complete preservation of per-sample error information. This finding changes the design principle of residual learning — the residual signal must precisely reflect per-sample error; any form of smoothing or clustering severely damages the benefits of deep networks.

### 7.4 Experiment Logical Flow

```
Experiment logical flow:

cifar_01 (First success 37.3%)
  ├──→ cifar_02 (Regularization → 39.3% SOTA)
  │      ├──→ cifar_03 (3-layer ablation: regularization only beneficial when overfitting)
  │      └──→ cifar_04 (More data: capacity bottleneck confirmed)
  ├──→ cifar_05 (MLP comparison: matched-param MLP severely overfits)
  └──→ cifar_06 (Three-way: RBF > MLP, CNN as upper bound, RBF uniquely full-analog deployable)

reg_04 (Simplest FD verification R²=0.94)
  └──→ reg_01 (Three-channel architecture + proto residual, depth gain +0.013)
          └──→ reg_02 (Direct residual, depth gain 5x +0.064)
                  └──→ reg_06 (Real-world data baseline R²=0.530)
```

***

## 8. Addressing High-Dimensionality: 3D Point-Cloud Encoding

### 8.1 Problem: RBF's Dilemma in High Dimensions

Traditional RBF networks operate directly in high-dimensional raw space. For a 32×32 grayscale image, the input space is 1024 dimensions. Computing Euclidean distance in this space:

$$d(\mathbf{x}, \mathbf{c}) = \sqrt{\sum_{j=1}^{1024} (x_j - c_j)^2}$$

When all dimensions are treated anonymously, the curse of dimensionality causes all pairwise distances between samples to become indistinguishable:

$$\lim_{D \to \infty} \frac{\max_j d(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) - \min_j d(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})}{\min_j d(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})} \to 0$$

In 1024 dimensions, the Gaussian kernel activates all particles nearly uniformly, losing selectivity — this is the fundamental reason traditional RBF cannot classify images.

### 8.2 Solution: Structural Rearrangement, Not Dimensionality Reduction

**Core idea**: do not reduce dimensionality, but rearrange data structure. Decompose a 32×32 image from a 1024-dimensional anonymous vector into **1024 labeled 3D points**:

$$\mathcal{I} = \{(x, y, v_{xy}) \mid x, y \in \{0,1,\dots,31\}\}$$

where (x, y) are pixel spatial coordinates (normalized to [0,1]) and v_xy is the pixel value.

This is NOT dimensionality reduction — the information content of the data is fully preserved — but rather **endowing each dimension with a distinguishable spatial address**.

### 8.3 Four Geometric Mechanisms

The 3D point-cloud encoding overcomes the curse of dimensionality through four geometric mechanisms:

**(1) Spatial Locality**

The pixel's (x, y) coordinates encode spatial position. A particle located at (0.5, 0.3, 0.7) naturally responds to bright pixels in the upper-right region of the image — distance computation occurs at the pixel level, not the image level.

For a particle center c = (x_c, y_c, z_c), the contribution of the k-th pixel is:

$$d_k^2 = \frac{(x_k - x_c)^2}{h_{xy}^2} + \frac{(y_k - y_c)^2}{h_{xy}^2} + \frac{(v_k - z_c)^2}{h_z^2}$$

$\sqrt{(x_k - x_c)^2 + (y_k - y_c)^2}$ measures the spatial distance between the pixel and the particle center — the closer a pixel is to the particle's (x_c, y_c), the greater its intensity contributes to the particle's activation. This endows the particle with a **spatial receptive field**.

**(2) Intensity Selectivity**

The z_c parameter selects which pixel intensities to respond to. A particle with z_c ≈ 1 prefers white regions (high brightness), z_c ≈ 0 prefers black regions (dark areas), and z_c ≈ 0.5 prefers mid-gray. The particle's selectivity in the (v_k - z_c)² dimension makes it a tunable local pattern detector.

**(3) Geometric Coupling**

(x_c, y_c) and z_c are not independent parameters — they are coupled through the product structure of the Gaussian kernel:

$$\exp\left(-\frac{(x_k - x_c)^2 + (y_k - y_c)^2}{h_{xy}^2} - \frac{(v_k - z_c)^2}{h_z^2}\right)$$

This means a particle activates strongly only when both conditions are met: the pixel lies within its spatial receptive field **AND** the pixel value is close to its intensity selectivity center. This AND-coupling produces detectors with specific spatial locations and specific brightness patterns — essentially learnable oriented edges, corners, and texture primitives.

**(4) Dimensional Address Labels**

The most critical distinction: in a 1024-dimensional anonymous vector, dimensions j=0 and j=1 contribute to distance computation in a completely symmetric and indistinguishable manner. In the 3D point cloud, each pixel carries its (x, y) coordinates as **address labels**. Pixels at coordinates (0.0, 0.0) and (0.0, 0.03125) are adjacent in 3D space — this geometric adjacency relation is entirely absent in 1024-dimensional anonymous space.

### 8.4 Analogy to CNNs

| Concept | CNN | RBF 3D Point Cloud |
|---------|-----|-------------------|
| Spatial receptive field | Convolution kernel size (e.g., 3×3) | Particle width h_xy |
| Filter weights | Discrete kernel matrix w[i,j] | Continuous intensity center z_c |
| Spatial position | Implicit (output feature map position) | Explicit (x_c, y_c coordinates) |
| Nonlinearity | ReLU (artificial) | Gaussian distance decay (physically natural) |
| Parameterization | Discrete weights per kernel | Continuous 3-parameter (x_c, y_c, z_c) + width per particle |

An RBF particle at (x_c, y_c) ≈ a convolution kernel anchored at position (x_c, y_c), with receptive field controlled by h_xy and its "filter weights" continuously parameterized by z_c. The difference is that RBF achieves locality through Gaussian radial basis functions (continuous, differentiable, physically realizable) rather than discrete convolution (GPU-optimized).

### 8.5 Complete Forward Pass of 3D Point-Cloud Encoding

Let an image be represented as N_pix = H×W 3D points (batch dimension B images), with a particle layer of N particles. The encoding proceeds as follows:

**Input rearrangement**:

$$\mathbf{X} \in \mathbb{R}^{B \times H \times W} \longrightarrow \mathbf{P} \in \mathbb{R}^{B \times N_{pix} \times 3}, \quad \mathbf{P}_{b,k} = (x_k, y_k, v^{(b)}_k)$$

**Distance computation** (for each sample, each pixel, each particle):

$$d_{b,k,i}^2 = \frac{(x_k - c_{x,i})^2}{h_{xy,i}^2} + \frac{(y_k - c_{y,i})^2}{h_{xy,i}^2} + \frac{(v_k^{(b)} - c_{z,i})^2}{h_{z,i}^2}$$

**Aggregation** (summation across pixels):

$$z_{b,i} = \sum_{k=1}^{N_{pix}} \exp\left(-\frac{1}{2} d_{b,k,i}^2\right)$$

Each particle aggregates Gaussian responses from all pixels within its receptive field. The aggregated particle activation vector $\mathbf{z} \in \mathbb{R}^{B \times N}$ can serve as input to subsequent RBF layers or a classifier.

## 9. Network Version Evolution

### 9.1 早期 2D 参考实现 (2D Reference)

`rbf_network/` directory

- Original 2D particle network with fixed grid boundary
- PyTorch-based implementation
- Reference implementation for validating the core RBF paradigm in low dimensions
- Validation experiment: moons binary classification, 8/6/4 particles, 120 epochs, accuracy 0.950

### 9.2 D 维泛化 (ND Generalization)

`network/core_nd.py`

- **RBFLayerND**: RBF single layer supporting arbitrary input dimension D
- **ResidualRBFNetworkND**: Multi-layer residual architecture with wire-direct inter-layer connections
- Dual-channel Gaussian activation: narrow kernel k=0.5 + wide kernel k=0.3
- PU (Partition of Unity) normalization: $\hat{z}_i = z_i / \sum_j z_j$
- Key innovation: Generalizes the fixed 2D grid to arbitrary dimensions, enabling the network to handle high-dimensional features

### 9.3 PhysicalRBFLayerND — Physical Simulation Layer

`simulator/layer_nd.py`

- Exact mathematical replica of the physical analog circuit
- Calls low-level component models from `physical_simulator.components`:
  - `differential_pair`: differential amplifier, computing (x - c)/h
  - `gilbert_square`: Gilbert squarer, computing dx²
  - `bjt_gaussian_kernel`: BJT subthreshold exponential, exp(-k·d²)
  - `memristor_mvm`: memristor matrix-vector multiplication, W·ẑ
- **Digital-physical gap = 0.000000** (mathematically exact match to the digital ND model, verified)
- Supports `from_digital()` export: directly exports PyTorch digital model weights to physical simulator parameters

### 9.4 三通道增强 (Three-Channel Enhancement)

`experiments/reg_01_proto_baseline.py`

- **Three-channel design**:
  - Narrow channel: Gaussian kernel exp(-k_n · d²), captures local high-frequency variations
  - Wide channel: Gaussian kernel exp(-k_w · d²), captures global smooth trends
  - Cauchy channel: 1/(1 + d²), heavy-tailed distribution, enhances robustness to large deviations
- **Feature scaling**: Programmable Gain Amplifiers (PGA) — each input dimension multiplied by learnable gain g_k
- **K-Means initialization**: particle centers initialized at K-Means cluster centroids, accelerating convergence
- **Anisotropic h**: per-dimension independent width h_{k,i}, particles can have ellipsoidal receptive fields in input space
- **Multi-scale k**: optimize narrow/wide channel k_n, k_w in log domain, allowing automatic selection of optimal kernel width
- **Momentum FD**: momentum term added to finite-difference gradient estimation, smoothing stochastic gradient noise
- **L2 regularization**: weight decay prevents overfitting

### 9.5 CIFAR 分类网络 (CIFAR Classification Network with 3D Encoding)

Complete image classification network based on 3D point-cloud encoding:

```
Input: 32×32 grayscale image (1024 3D points)
    |
Layer 0: Spatial Transformer (3D → 3D, 25 particles)
    Residual layer: f₀ = X + RBF₀(X)
    Preserves 3D spatial structure, learnable spatial warping
    |
Layer 1: Local Feature Extraction (3D → 12D, 49 particles, 7×7 grid initialization)
    7×7 particle grid uniformly covering 32×32 image space
    Each particle extracts local patterns within its receptive field
    |
Layer 2: Mid-Level Abstraction (12D → 8D, 28 particles)
    Abstracts mid-level semantics from local features
    |
Layer 3: Compact Representation (8D → 6D, 20 particles)
    Reduces to compact classification representation
    |
Global Mean Pooling:
    o = mean(F₃, dim=0) ∈ ℝ⁶
    |
Nearest-Prototype Classifier:
    class = argmin_c ||o - proto_c||²
```

- **Total parameters**: 2884
- **Design principle**: from spatial (3D) → local features (12D) → abstract features (8D) → compact representation (6D)
- Layer 0's spatial transform residual preserves spatial structure, enabling subsequent layer particles to rely on precise spatial coordinate information

## 10. Four-Tier Physical Training System

Physical training methods progress by hardware complexity, forming a complete spectrum from minimal hardware to fully parallel hardware:

### Method A: SPSA — Minimal Hardware

**Hardware requirement**: 1 differential amplifier + 0 ADCs

**Principle**: Simultaneously perturb all parameters $\theta + \epsilon \boldsymbol{\Delta}$ and $\theta - \epsilon \boldsymbol{\Delta}$ ($\boldsymbol{\Delta}$ is a random sign vector), measure loss twice and estimate gradient:

$$\hat{g} = \frac{L(\theta + \epsilon\boldsymbol{\Delta}) - L(\theta - \epsilon\boldsymbol{\Delta})}{2\epsilon} \boldsymbol{\Delta}$$

Hardware implementation: the differential amplifier inputs are L⁺ and L⁻, output proportional to ΔL = L⁺−L⁻, i.e., the scalar ΔL is obtained directly via analog subtraction without an ADC. Gradient update $\theta \leftarrow \theta - \eta \cdot \Delta L \cdot \boldsymbol{\Delta} / (2\epsilon)$ is performed in the digital domain.

| Property | Value |
|----------|-------|
| Forward passes | 2/group (one positive, one negative) |
| ADC requirement | 0 |
| Gradient noise | O(1/√N), decays with averaging count |
| Hardware complexity | Minimal |

### Method B: Per-Parameter Finite Difference

**Hardware requirement**: 1 ADC (digitize scalar loss)

**Principle**: Apply independent ±ε perturbation to each parameter θ_j, central-difference estimate partial derivative:

$$\frac{\partial L}{\partial \theta_j} \approx \frac{L(\theta + \epsilon \mathbf{e}_j) - L(\theta - \epsilon \mathbf{e}_j)}{2\epsilon}$$

**Grouped parallelism** (Experiments train_01 / train_02): Partition parameters into K groups. train_01 (grouped FD) proves parameter group independence, meaning K parallel hardware circuits can train independently; train_02 (grouped SPSA) measures equivalent speedup of >20× while maintaining accuracy:

| Experiment | Method | Core Finding |
|-----------|--------|--------------|
| train_01 (formerly E13) | Grouped per-parameter FD | Parameter group independence proven, K× parallel feasible |
| train_02 (formerly E14) | Grouped SPSA | Measured >20× equivalent speedup |

### Method C: Analog Gradient Chain — O(1) Latency

**Hardware requirement**: N_out×N_particles Gilbert multipliers + N_particles cascaded multipliers + capacitor integrator array

**Principle**: Simultaneously construct gradient signals during forward propagation, completely eliminating per-parameter perturbation:

1. **Forward propagation**: Signal flows through differential amplifiers → Gilbert squarers → BJT exponentials, producing particle activations z_i
2. **Gradient chain**:
   - Gilbert multiplier array: computes ∂L/∂W_{ij} = δ_i · ẑ_j (output error × normalized activation)
   - Cascaded multiplier chain: backpropagates from ∂L/∂z to ∂L/∂h and ∂L/∂c via the chain rule
3. **Capacitor integration**: Gradient signals accumulate on capacitors, physically implementing gradient accumulation

| Property | Value |
|----------|-------|
| Forward passes | **1** |
| Gradient precision | Exact (analytical gradient) |
| Latency | O(1), independent of parameter count |
| Hardware complexity | Highest |

**This is the only scheme that simultaneously obtains exact gradients during forward propagation** — physically implementing the chain rule of backpropagation.

### Method D: Digital Twin (PyTorch autograd) — Rapid Prototyping

**Hardware requirement**: CPU/GPU

**Principle**: Build a mathematically equivalent digital model of the physical network in PyTorch, obtain gradients via autograd backpropagation.

- Answers the fundamental question "can this architecture learn?"
- Validates architectural feasibility before hardware design
- Trained weights exported to physical simulator parameters via `from_digital()`
- Only after digital validation passes do we tackle "can we build the hardware?"

### Four-Tier Comparison Summary

| Dimension | Method A (SPSA) | Method B (FD) | Method C (Analog Chain) | Method D (Digital Twin) |
|-----------|----------------|---------------|------------------------|------------------------|
| Hardware | 1 diff amp | 1 ADC | Full analog multipliers + integrators | CPU/GPU |
| Forward/update | 2 | 2N_params | 1 | 1 (+backward) |
| Gradient quality | Noisy O(1/√N) | Exact | Exact | Exact |
| Latency | O(1) | O(N_params) | **O(1)** | O(1) GPU |
| Use case | Extreme hardware constraints | General physical training | High-performance physical inference | Architecture validation + rapid iteration |

---

## 11. Current Bottlenecks and Limitations

1. **Capacity**: 2884 parameters saturate at ~100 img/class; more particles or wider layers needed
2. **Depth efficiency**: the direct residual scheme needs to be ported to real-world data and larger scales
3. **Training speed**: FD requires O(N) forward passes; SPSA has gradient noise

### Disclaimer

This project was conducted under limited experimental conditions. Since numerical simulation is used to emulate physical systems (analog circuits, memristor arrays, and other physical components), each forward pass and parameter update must be computed through the physical signal chain, incurring computational overhead far higher than standard neural network matrix multiplication. Constrained by compute resources, current experiments use modest parameter counts and training epochs, validated only on moons, 4D synthetic classification, sine regression, CIFAR-10 5-class subsets, and California Housing datasets.

The core objective of this project is to propose an "activate-then-combine" particle source superposition model — endowing neural networks with better interpretability and enabling complete training and inference on physical platforms such as analog circuits, optical systems, and wave fields. Validation experiments cover key propositions including digital-physical consistency, depth effectiveness, noise robustness, parameter efficiency, and grouped parallelization feasibility, serving as a baseline for future research.

Subject to the author's coding proficiency and compute limitations, the project may have shortcomings. Users with better resources are welcome to adjust the network architecture, increase parameter scale and training epochs, conduct further experiments on more complex datasets, or verify this architecture on real physical hardware based on this codebase. We hope the ideas and code in this project are helpful to you.

**Core philosophy**: The core of the RBF particle neural network is not a specific algorithm or hardware implementation, but a unified perspective — viewing neural networks as field superposition systems in physical space. From this perspective, network training (physical system evolution), inference (field strength superposition), architecture design (source layout), and interpretability (source positions and radiation ranges all have clear physical meaning) all gain natural physical interpretations while maintaining the same expressive power as digital deep learning.
