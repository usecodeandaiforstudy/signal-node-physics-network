# RBF Particle Neural Network: Structure and Principles

## 1. Particle Model

Each particle is a triplet `(xc, yc, h)`: a 2D center coordinate `(xc, yc)` and influence radius `h`. Given input `(x, y)`, the particle's activation is:

$$z_i(x,y) = \exp\!\left(-\frac{(x-xc_i)^2 + (y-yc_i)^2}{2h_i^2}\right)$$

- Larger `h` — wider influence range, smoother response
- Smaller `h` — stronger selectivity, only responds to nearby inputs

All parameters `(xc, yc, h)` are trainable via gradient descent. Particles move through space to optimally cover the data distribution.

## 2. Single Layer (RBFLayer)

### 2.1 Forward Pass

A single layer contains `N` particles with output dimension `d_out`.

**Step 1 — Particle activation:**

$$z_i = \exp(-0.5 \cdot d_i^2) \quad\text{(narrow kernel)}$$
$$\tilde{z}_i = \exp(-0.3 \cdot d_i^2) \quad\text{(wide kernel)}$$

where $d_i^2 = \frac{(x-xc_i)^2 + (y-yc_i)^2}{h_i^2}$. The two channels capture local detail and global structure respectively.

**Step 2 — PU normalization:**

$$\hat{z}_i = \frac{z_i}{\sum_{j=1}^N z_j + \epsilon}, \quad \hat{\tilde{z}}_i = \frac{\tilde{z}_i}{\sum_{j=1}^N \tilde{z}_j + \epsilon}$$

Normalization constrains activations to a probability distribution: the sum of all particle activations is always 1.

**Step 3 — Linear projection:**

$$y_{\text{gauss}} = W_g \cdot \hat{z}, \quad y_{\text{act}} = W_a \cdot \hat{\tilde{z}}$$

$$y_{\text{out}} = y_{\text{gauss}} + y_{\text{act}}$$

where $W_g, W_a \in \mathbb{R}^{d_{\text{out}} \times N}$ are trainable linear weight matrices.

### 2.2 Trainable Parameters

| Parameter | Shape | Description |
|---|---|---|
| `xc`, `yc` | (N,) | Particle center coordinates |
| `h` | (N,) or scalar | Particle width |
| `W_g` | (d_out, N) | Narrow kernel channel weights |
| `W_a` | (d_out, N) | Wide kernel channel weights |

Total per layer: approximately `3N + 2·d_out·N` parameters.

## 3. Three-Layer Residual Network (ResidualRBFNetwork)

### 3.1 Architecture

```
Input (x, y)
    │
    ▼
Layer 1 (N₁ particles, d₁ out) → f₁
    │
    ├─ residual: f₁ skip ───────────────────┐
    ▼                                        │
Layer 2 (N₂ particles, d₂ out) → δ₂ ────────┼─→ f₂ = f₁ + δ₂
    │                                        │
    ├─ residual: f₂ skip ───────────────────┐│
    ▼                                        ││
Layer 3 (N₃ particles, d₃ out) → δ₃ ───────┼┼─→ f₃ = f₂ + δ₃
                                              ││
                                              ▼
                                           Output
```

Output dimensions per layer:

- Layer 1: `n_vars = hidden_dim`
- Layer 2: `n_vars = output_dim`
- Layer 3: `n_vars = output_dim`

### 3.2 Inter-layer Dimension Transfer

Layers are connected via mean projection: multi-dimensional outputs `(batch, d)` are projected to `(batch,)` scalar spatial coordinates via `.mean(-1)`, then passed as `(x, y)` inputs to the next layer.

### 3.3 Residual Connection

$$f_{k+1} = f_k + \delta_{k+1}$$

$\delta_{k+1}$ is the correction learned by layer $k+1$. Residual connections ensure that even deep intermediate representations retain fine-grained information from shallow layers.

## 4. Training

### 4.1 Classification

Output layer $d_3 = C$ (number of classes), followed by softmax and cross-entropy loss:

$$L = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(f_{3,i}[y_i])}{\sum_{c=1}^C \exp(f_{3,i}[c])}$$

### 4.2 Regression

$$L = \frac{1}{B}\sum_{i=1}^B \|f_{3,i} - y_i\|^2$$

### 4.3 Gradient Flow

The gradient of particle coordinate `xc_i` computed via the chain rule:

$$\frac{\partial L}{\partial xc_i} = \frac{\partial L}{\partial \hat{z}_i} \cdot \frac{\partial \hat{z}_i}{\partial z_i} \cdot z_i \cdot \frac{x - xc_i}{h_i^2}$$

where $\partial \hat{z}_i / \partial z_j$ includes the coupling term from PU normalization:

$$\frac{\partial \hat{z}_i}{\partial z_j} = \frac{\delta_{ij}\sum_k z_k - z_i}{(\sum_k z_k)^2}$$

Particles move opposite to the gradient direction (i.e., toward lower loss). The width `h` is similarly differentiable.

## 5. High-Dimensional Extension (ProjectedRBFEnsemble)

D-dimensional input is mapped to 2D via `K` fixed random projection matrices $P_k \in \mathbb{R}^{D \times 2}$:

$$x_k^{\text{2D}} = P_k \cdot x$$

Each projection feeds an independent 2D RBF subnetwork, outputs are averaged:

$$f(x) = \frac{1}{K}\sum_{k=1}^K f_k(x_k^{\text{2D}})$$

Projection matrices $P_k$ are randomly initialized and fixed (not trained). Only subnetwork parameters are trainable. In physical implementation, $P_k$ corresponds to a fixed-conductance memristor array.

## 6. Physical System Implementation

Each mathematical operation in the network has a corresponding analog circuit element. The forward-pass signal flows through the circuit as continuous voltage/current, requiring no analog-to-digital conversion.

### 6.1 Operation-to-Circuit Mapping

| Math Operation | Analog Circuit | Physical Principle | Device Cost |
|---|---|---|---|
| `dx = (x-xc)/h` | Differential amplifier | Diff pair + transconductance gain | ~4 transistors/channel |
| `d² = dx² + dy²` | Gilbert squarer | Translinear multiplier | ~6 transistors/channel |
| `exp(-k · d²)` | BJT exponentiator | Subthreshold I_c = I_s · exp(V_be/V_T) | 1-2 transistors/channel |
| `ẑ = z / Σz` | Current mirror normalizer | Kirchhoff sum + translinear divider | 4 transistors + wire |
| `W · ẑ` | Memristor crossbar | Ohm's law I = G·V, column-wise Kirchhoff sum | Passive (memristors only) |
| `f + δ` | Wire junction | Kirchhoff current law | 0 devices |

### 6.2 Physical Layer Signal Chain

```
V_in(x,y) [voltage input]
    │
    ├─→ Differential amplifier (dx = (x-xc)/h)
    │       │
    ├─→ Gilbert squarer (d²)
    │       │
    ├─→ BJT exponentiator (exp(-0.5d²), exp(-0.3d²))  [dual-channel parallel]
    │       │
    ├─→ Current mirror normalize (ẑ = z / Σz)          [Kirchhoff sum]
    │       │
    ├─→ Memristor crossbar (W · ẑ)                     [O(1) MVM]
    │       │
    └─→ Wire junction (f + δ)                          [zero-device residual]
            │
        V_out [voltage/current output]
```

### 6.3 Noise Sources and Non-Idealities

| Non-Ideal Effect | Source | Model Parameter |
|---|---|---|
| Thermal noise | Resistor/transistor Johnson-Nyquist noise | `noise_std` |
| Process mismatch | Random variation in transistor dimensions, threshold voltage | `mismatch_std` |
| Conductance drift | Memristor resistance change from read/write cycles | `drift_std` |
| Stuck memristors | Fabrication defects causing fixed conductance | `stuck_fraction` |
| Saturation clipping | Supply rail voltage limits | `V_SUPPLY` |

### 6.4 Device Count

For a lightweight layer (N=20, d_out=1):

| Component | Count | Notes |
|---|---|---|
| Differential amplifiers | 40 channels | x and y, N channels each |
| Gilbert squarers | 20 channels | One per particle |
| BJT exponentiators | 40 | Dual-channel (narrow+wide), 2 per particle |
| Current mirror dividers | 2 | One per channel |
| Memristors | 20 | N × d_out = 20 × 1 |
| **Total** | **~120 transistors + 20 memristors** | |

A three-layer network totals approximately 300–400 transistors and 50–80 memristors. Estimated power consumption is in the μW range (subthreshold operation); single-inference latency is in the nanosecond range.

### 6.5 Training-Inference Separation

Training is performed in the digital domain (using `physical_simulator` to model non-idealities, or directly in the physical domain via SPSA/HiMAL). After training, parameters are programmed once onto the chip:

- **Particle coordinates `(xc, yc)`**: DAC reference voltages
- **Width `h`**: Bias-current-controlled transconductance gain
- **Weights `W`**: Memristor conductance values

Inference runs entirely on analog signals with zero analog-to-digital conversion.

## 7. Parameter Overview

| Configuration | N₁/N₂/N₃ | Total Parameters | Estimated Transistors |
|---|---|---|---|
| Lightweight (2D) | 20/10/8 | ~230 | ~400 |
| Medium (2D) | 25/12/8 | ~265 | ~480 |
| Ensemble (4D, K=5) | 5 × (8/4/3) | ~485 | ~900 |
| Ensemble (4D, K=5) | 5 × (12/6/4) | ~690 | ~1300 |
