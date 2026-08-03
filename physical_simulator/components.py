"""
Physical Component Simulation for RBF Particle Networks
========================================================
Project: singal-node-physics-network
Goal: Build a fully analog-trainable neural computing framework using RBF
      particle networks mapped to physical circuit primitives (BJT differential
      pairs, Gilbert multipliers, memristor crossbars). Every mathematical
      operation in the network (distance, kernel, normalization, MVM, loss,
      gradient) has a corresponding analog circuit implementation with
      realistic non-idealities (noise, mismatch, drift, quantization).

This module provides:
  Forward-pass components (1-6):   differential_pair, gilbert_square,
      bjt_gaussian_kernel, current_mode_normalize, memristor_mvm, kirchhoff_sum
  ADC interface (7):               analog_adc — physical-to-digital bridge
  Analog loss chain (8):           subtractor → squarer → summer → ADC
  Analog gradient chain (9):       Gilbert multipliers + capacitor integrators
      for computing ∂L/∂W, ∂L/∂xc, ∂L/∂h directly in the analog domain

Non-ideality sources:
  - Thermal noise (Johnson-Nyquist): additive noise proportional to temperature
  - Process mismatch: random deviation of device parameters from nominal values
  - Saturation clipping: output range constraint due to supply rail limits
  - Conductance drift (Memristor Drift): memristor conductance drift over time/read operations
  - Quantization error (ADC): LSB-limited resolution in analog-to-digital conversion
"""

import numpy as np
from typing import Optional


# ──────────────────────────────
#  Global physical constants and noise parameters
# ──────────────────────────────

V_T = 0.026          # Thermal voltage (V), room temperature ~300K
V_SUPPLY = 1.8       # Supply voltage (V)
I_BIAS_MAX = 1e-3    # Maximum bias current (A)


# ──────────────────────────────
#  1. Differential amplifier: (x - xc) / h
# ──────────────────────────────

def differential_pair(
    x: np.ndarray,
    xc: np.ndarray,
    h: np.ndarray,
    mismatch_std: float = 0.02,
    noise_std: float = 1e-6,
) -> np.ndarray:
    """
    Simulate differential pair + transconductance-tunable gain stage.

    V_out = (V_in - V_ref) / gain
    where gain is controlled by bias current = physical equivalent of h.

    Parameters
    ----------
    x : (batch,) or (batch, 1)
        Input voltage
    xc : (n_particles,)
        Reference voltage (particle coordinates)
    h : (n_particles,) or scalar
        Gain parameter (width)
    mismatch_std : float
        Gain mismatch standard deviation (relative)
    noise_std : float
        Output noise standard deviation (V)

    Returns
    -------
    dx : (batch, n_particles)
        Normalized differential signal
    """
    x = np.asarray(x).reshape(-1, 1)  # (batch, 1)
    xc = np.asarray(xc).reshape(1, -1)  # (1, n_particles)
    h = np.asarray(h).reshape(1, -1)   # (1, n_particles)

    # Gain mismatch: each channel's h has independent deviation
    h_noisy = h * (1 + np.random.normal(0, mismatch_std, h.shape))

    # Differential + gain
    dx = (x - xc) / np.clip(h_noisy, 1e-6, None)

    # Output noise
    if noise_std > 0:
        dx += np.random.normal(0, noise_std, dx.shape)

    # Note: no intermediate clipping — differential pairs saturate in real circuits
    # but for mathematical equivalence with the digital model we compute the raw value.
    # Final output clipping is handled at the MVM stage.

    return dx


# ──────────────────────────────
#  2. Gilbert squarer: d² = dx² + dy²
# ──────────────────────────────

def gilbert_square(
    dx: np.ndarray,
    dy: np.ndarray,
    gain: float = 1.0,
    mismatch_std: float = 0.02,
    offset_std: float = 0.0,
    noise_std: float = 1e-6,
) -> np.ndarray:
    """
    Simulate Gilbert multiplier cell computing sum of squares.

    I_out = I_bias * (dx² + dy²) / (2 V_T²)
    This is the nominal output current of the Gilbert cell.

    Parameters
    ----------
    dx, dy : (batch, n_particles)
        Normalized differential signals
    gain : float
        Nominal gain
    mismatch_std : float
        Inter-channel gain mismatch standard deviation
    offset_std : float
        Offset voltage standard deviation
    noise_std : float
        Output noise standard deviation

    Returns
    -------
    d2 : (batch, n_particles)
        Squared distance values
    """
    # Offset: fixed DC offset per channel (sampled once)
    offset_x = np.random.normal(0, offset_std, (1, dx.shape[1]))
    offset_y = np.random.normal(0, offset_std, (1, dy.shape[1]))

    dx_off = dx + offset_x
    dy_off = dy + offset_y

    # Gain mismatch
    gain_error = gain * (1 + np.random.normal(0, mismatch_std, (1, dx.shape[1])))

    # Sum of squares
    d2 = gain_error * (dx_off ** 2 + dy_off ** 2)

    # Output noise
    if noise_std > 0:
        d2 += np.random.normal(0, noise_std, d2.shape)

    # Lower bound protection (avoid numerical issues in exponential)
    d2 = np.clip(d2, 0, None)

    return d2


# ──────────────────────────────
#  3. BJT subthreshold exponential (Gaussian kernel): exp(-k * d²)
# ──────────────────────────────

def bjt_gaussian_kernel(
    d2: np.ndarray,
    k: float = 0.5,
    temperature_voltage: Optional[float] = None,
    saturation_current: float = 1.0,
    noise_std: Optional[float] = None,
) -> np.ndarray:
    """
    Simulate BJT subthreshold exponential relationship implementing Gaussian kernel.

    I_out = I_s * exp(-k * d² / V_T)

    Parameters
    ----------
    d2 : (batch, n_particles)
        Squared distance values
    k : float
        Kernel width coefficient (0.5=narrow, 0.3=wide)
    temperature_voltage : float, optional
        Thermal voltage V_T. If None, use global V_T.
        When set to 1.0, degenerates to pure mathematical Gaussian exp(-k * d²),
        suitable for comparison with digital model.
    saturation_current : float
        Saturation current I_s (set to 1.0 in normalized mode)
    noise_std : float, optional
        Additional noise standard deviation

    Returns
    -------
    z : (batch, n_particles)
        Gaussian kernel activation values
    """
    if temperature_voltage is None:
        temperature_voltage = V_T

    # Exponential mapping: BJT native behavior
    exponent = -k * d2 / max(temperature_voltage, 1e-6)
    exponent = np.clip(exponent, -50, 20)  # Overflow prevention

    z = saturation_current * np.exp(exponent)

    # Noise
    if noise_std is None:
        noise_std = np.sqrt(np.abs(z)) * 1e-6
    z += np.random.normal(0, noise_std, z.shape)

    # Clipping
    z = np.clip(z, 0, 1e3)

    return z


# ──────────────────────────────
#  3b. BJT logarithmic amplifier: V_out = V_T * ln(I_in / I_s)
# ──────────────────────────────

def bjt_log_amplifier(
    currents: np.ndarray,
    saturation_current: float = 1.0,
    temperature_voltage: Optional[float] = None,
    noise_std: Optional[float] = None,
) -> np.ndarray:
    """
    Simulate BJT logarithmic amplifier.

    Physical principle: BJT V_BE = V_T * ln(I_C / I_s)
    By placing BJT in op-amp feedback, output voltage is proportional to ln(input current).

    V_out = V_T * ln(I_in / I_s)

    Parameters
    ----------
    currents : (batch, n_channels)
        Input currents
    saturation_current : float
        Reverse saturation current I_s
    temperature_voltage : float, optional
        Thermal voltage V_T
    noise_std : float, optional
        Output noise

    Returns
    -------
    log_currents : (batch, n_channels)
        Logarithmic voltages
    """
    if temperature_voltage is None:
        temperature_voltage = V_T

    # Clip to positive range (currents must be > 0 for log)
    clipped = np.clip(currents, 1e-12, 1e6)
    log_val = temperature_voltage * np.log(clipped / max(saturation_current, 1e-12))

    if noise_std is not None and noise_std > 0:
        log_val += np.random.normal(0, noise_std, log_val.shape)

    return log_val


# ──────────────────────────────
#  3c. Gilbert translinear divider: out = numerator / denominator
# ──────────────────────────────

def gilbert_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    temperature_voltage: Optional[float] = None,
    noise_std: Optional[float] = None,
) -> np.ndarray:
    """
    Simulate Gilbert translinear divider.

    Physical principle: translinear loop → I_out = I_ref * I_num / I_den
    Configured as divider: I_num = 1.0 (reference), I_den = 1 + d²
    → I_out = I_ref / I_den = 1 / (1 + d²)  → Cauchy kernel

    Implementation via log-domain:
        V_out = exp(ln(numerator) - ln(denominator))
    Using BJT log amp → subtraction → BJT exponential.

    Parameters
    ----------
    numerator : (batch, n_channels)
        Numerator currents (typically 1.0 for Cauchy)
    denominator : (batch, n_channels)
        Denominator currents (e.g., 1 + d²)
    temperature_voltage : float, optional
    noise_std : float, optional

    Returns
    -------
    divided : (batch, n_channels)
        numerator / denominator
    """
    if temperature_voltage is None:
        temperature_voltage = V_T

    # Log-domain: ln(num) - ln(den) → exp
    log_num = bjt_log_amplifier(numerator, temperature_voltage=temperature_voltage)
    log_den = bjt_log_amplifier(denominator, temperature_voltage=temperature_voltage)
    log_ratio = log_num - log_den

    # Exponential: exp(log_ratio / V_T) = numerator / denominator
    # bjt_gaussian_kernel does exp(-k * d² / V_T), so use k = -1, d² = -log_ratio * V_T trick
    # Or directly: z = saturation_current * exp(log_ratio / V_T)
    temp_v = max(temperature_voltage, 1e-6)
    result = np.exp(np.clip(log_ratio / temp_v, -50, 20))

    if noise_std is not None and noise_std > 0:
        result += np.random.normal(0, noise_std, result.shape)

    return np.clip(result, 0, 1e3)


# ──────────────────────────────
#  4. Current mirror normalization: ẑ = z / Σz
# ──────────────────────────────

def current_mode_normalize(
    currents: np.ndarray,
    mirror_mismatch_std: float = 0.005,
) -> np.ndarray:
    """
    Simulate current mirror summation + division normalization.

    Physical implementation:
    - Summation: Kirchhoff's current law, N current branches merge into one wire
    - Division: translinear divider (4 BJTs)

    Parameters
    ----------
    currents : (batch, n_channels)
        Currents of each channel
    mirror_mismatch_std : float
        Current mirror mismatch standard deviation (copy ratio error)

    Returns
    -------
    normalized : (batch, n_channels)
        Normalized currents (sum ≈ 1)
    """
    # Current mirror copy error: each channel's copy ratio has deviation
    mirror_error = np.random.normal(0, mirror_mismatch_std, (1, currents.shape[1]))
    currents_mirrored = currents * (1 + mirror_error)

    # Summation line (Kirchhoff)
    total = np.sum(currents_mirrored, axis=1, keepdims=True)

    # Prevent division by zero
    total = np.clip(total, 1e-12, None)

    # Division
    normalized = currents_mirrored / total

    return normalized


# ──────────────────────────────
#  5. Memristor crossbar array MVM: V_out = W · V_in
# ──────────────────────────────

def memristor_mvm(
    inputs: np.ndarray,
    weights: np.ndarray,
    conductance_drift_std: float = 0.001,
    read_noise_std: float = 1e-7,
    stuck_fraction: float = 0.0,
) -> np.ndarray:
    """
    Simulate matrix-vector multiplication on memristor crossbar array.

    Physical principle:
    - Input voltage V_in applied at input terminals
    - Memristor conductance G_ij encodes weight W_ij at crosspoints
    - Column output current I_out = Sum_j G_ij * V_j (Kirchhoff + Ohm's law)
    - Entire MVM completes in O(1) physical time

    Parameters
    ----------
    inputs : (batch, n_in)
        Input voltages
    weights : (n_out, n_in)
        Memristor conductance matrix (= Linear weights)
    conductance_drift_std : float
        Conductance drift standard deviation (per read operation)
    read_noise_std : float
        Readout circuit noise
    stuck_fraction : float
        Fraction of stuck memristors (0~1)

    Returns
    -------
    outputs : (batch, n_out)
        Output current/voltage
    """
    # Uniform shape: (batch, n_in)
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    if inputs.ndim == 2:
        pass  # (batch, n_in) — expected format
    else:
        raise ValueError(f"memristor_mvm expects 2D inputs, got shape {inputs.shape}")

    n_out, n_in = weights.shape

    # Stuck memristors: some conductances fixed at extreme values
    if stuck_fraction > 0:
        n_stuck = int(n_out * n_in * stuck_fraction)
        stuck_mask = np.zeros((n_out, n_in), dtype=bool)
        stuck_indices = np.random.choice(n_out * n_in, n_stuck, replace=False)
        stuck_mask.flat[stuck_indices] = True
        weights_masked = weights.copy()
        weights_masked[stuck_mask] = np.random.choice(
            [weights.min() * 0.1, weights.max() * 10], n_stuck
        )
    else:
        weights_masked = weights

    # Conductance drift (per read operation)
    drift = np.random.normal(0, conductance_drift_std, (n_out, n_in))
    g_matrix = weights_masked * (1 + drift)

    # MVM: O(1) physical latency — inputs @ G^T
    outputs = inputs @ g_matrix.T  # (batch, n_in) @ (n_in, n_out) = (batch, n_out)

    # Readout noise
    if read_noise_std > 0:
        outputs += np.random.normal(0, read_noise_std, outputs.shape)

    # Supply rail clipping
    outputs = np.clip(outputs, -V_SUPPLY, V_SUPPLY)

    return outputs


# ──────────────────────────────
#  6. Kirchhoff current summation: out = I1 + I2
# ──────────────────────────────

def kirchhoff_sum(
    i1: np.ndarray,
    i2: np.ndarray,
    wire_noise_std: float = 1e-9,
) -> np.ndarray:
    """
    Simulate Kirchhoff's current law: two current lines merge.

    Physically requires zero active devices.
    The purest "physical computation" — the physical law itself is the operation.

    Parameters
    ----------
    i1, i2 : ndarray
        Two current lines
    wire_noise_std : float
        Wire thermal noise (extremely small)

    Returns
    -------
    i_sum : ndarray
        Total current = i1 + i2
    """
    result = i1 + i2

    # Wire thermal noise (Johnson noise, extremely small)
    if wire_noise_std > 0:
        result += np.random.normal(0, wire_noise_std, result.shape)

    return result


# ──────────────────────────────
#  Batch set global noise level
# ──────────────────────────────

class NoiseConfig:
    """
    Global noise configuration, supports unified adjustment of noise levels for all components.

    Usage:
        from physical_simulator.components import NoiseConfig
        NoiseConfig.set_level("high")  # High noise mode

    Noise levels:
        "none"  → No noise, pure ideal transfer function
        "low"   → Low noise (good real circuit conditions)
        "medium"→ Medium noise (typical)
        "high"  → High noise (worst-case boundary)
    """

    _levels = {
        "none":   {"mismatch": 0.0, "noise": 0.0, "drift": 0.0, "stuck": 0.0, "v_t": 1.0},
        "low":    {"mismatch": 0.01, "noise": 1e-7, "drift": 0.0005, "stuck": 0.0, "v_t": 1.0},
        "medium": {"mismatch": 0.02, "noise": 1e-6, "drift": 0.001, "stuck": 0.01, "v_t": 1.0},
        "high":   {"mismatch": 0.05, "noise": 1e-5, "drift": 0.005, "stuck": 0.05, "v_t": 1.0},
    }

    _current = "medium"

    @classmethod
    def set_level(cls, level: str):
        if level not in cls._levels:
            raise ValueError(f"Unknown noise level: {level}. Choose from {list(cls._levels.keys())}")
        cls._current = level

    @classmethod
    def get_config(cls) -> dict:
        return cls._levels[cls._current]

    @classmethod
    def mismatch_std(cls) -> float:
        return cls.get_config()["mismatch"]

    @classmethod
    def noise_std(cls) -> float:
        return cls.get_config()["noise"]

    @classmethod
    def drift_std(cls) -> float:
        return cls.get_config()["drift"]

    @classmethod
    def stuck_fraction(cls) -> float:
        return cls.get_config()["stuck"]


# ══════════════════════════════════════════════════════════
#  7. Analog-to-Digital Converter (ADC)
# ══════════════════════════════════════════════════════════

def analog_adc(
    signal: np.ndarray,
    n_bits: int = 10,
    v_ref: float = 1.8,
    offset_error_std: float = 0.002,
    gain_error_std: float = 0.001,
    sampling_jitter_std: float = 0.0,
) -> np.ndarray:
    """
    Simulate an analog-to-digital converter.

    Converts analog voltage/current to digital code with quantization and
    non-idealities. This is the physical-digital interface: all physical
    signals must pass through ADC before entering the digital controller.

    Key non-idealities:
      - Quantization noise: inherent to n_bits resolution, ~LSB/sqrt(12) RMS
      - Offset error: DC shift in the conversion characteristic
      - Gain error: slope deviation from ideal transfer function
      - Sampling jitter: timing uncertainty in sample-and-hold

    Physical implementation: SAR (Successive Approximation Register) or
    Flash ADC depending on speed/resolution trade-off.

    Parameters
    ----------
    signal : ndarray
        Analog input signal (voltage or current, any shape)
    n_bits : int
        ADC resolution in bits (typ. 8-16 for embedded, 24 for precision)
    v_ref : float
        Reference voltage defining full-scale range [-v_ref, +v_ref]
    offset_error_std : float
        Offset error standard deviation (fraction of v_ref)
    gain_error_std : float
        Gain error standard deviation (fraction of nominal gain)
    sampling_jitter_std : float
        Sampling clock jitter standard deviation (seconds)

    Returns
    -------
    digital : ndarray (same shape)
        Digitized signal, quantized to 2^n_bits levels within [-v_ref, +v_ref]
    """
    # Offset error (fixed per conversion, sample-once per array)
    offset = np.random.normal(0, offset_error_std) * v_ref

    # Gain error
    gain = 1.0 + np.random.normal(0, gain_error_std)

    # Apply offset and gain
    v_signal = signal * gain + offset

    # Clamp to full-scale range (ADC saturation)
    v_signal = np.clip(v_signal, -v_ref, +v_ref)

    # Quantization
    n_levels = 2 ** n_bits
    lsb = 2 * v_ref / n_levels  # Least Significant Bit size
    digital = np.round(v_signal / lsb) * lsb

    # Sampling jitter (modeled as additive noise proportional to signal derivative)
    if sampling_jitter_std > 0:
        jitter_noise = np.random.normal(0, sampling_jitter_std, signal.shape)
        digital += jitter_noise * v_ref

    # Re-clamp after jitter
    digital = np.clip(digital, -v_ref, +v_ref)

    return digital


# ══════════════════════════════════════════════════════════
#  8. Analog Loss Computation Chain
# ══════════════════════════════════════════════════════════

def analog_subtractor(
    a: np.ndarray,
    b: np.ndarray,
    mismatch_std: float = 0.02,
    noise_std: float = 1e-6,
) -> np.ndarray:
    """
    Simulate analog subtractor: e = a - b.

    Physical implementation: differential amplifier with unity gain.
    Reuses the same differential_pair BJT topology, configured for
    unity-gain subtraction (h=1.0) instead of the (x-xc)/h variant.

    This is the first stage of the analog loss chain:
        prediction (analog) ─┐
                              ├─ subtractor → error
        target (analog)    ──┘

    Parameters
    ----------
    a, b : ndarray
        Two analog signals (must be broadcastable)
    mismatch_std : float
        Gain mismatch between the two input branches
    noise_std : float
        Output noise standard deviation

    Returns
    -------
    error : ndarray
        a - b (analog error signal)
    """
    # Gain mismatch between positive and negative input paths
    gain_a = 1.0 + np.random.normal(0, mismatch_std)
    gain_b = 1.0 + np.random.normal(0, mismatch_std)

    error = gain_a * np.asarray(a) - gain_b * np.asarray(b)

    if noise_std > 0:
        error += np.random.normal(0, noise_std, error.shape)

    return error


def analog_squarer(
    x: np.ndarray,
    mismatch_std: float = 0.02,
    offset_std: float = 0.0,
    noise_std: float = 1e-6,
) -> np.ndarray:
    """
    Simulate analog squarer: y = x².

    Physical implementation: Gilbert multiplier cell with both inputs
    tied to the same signal → self-multiplication = square.

    This is the second stage of the analog loss chain:
        error → squarer → error²

    Parameters
    ----------
    x : ndarray
        Input signal
    mismatch_std : float
        Multiplier gain mismatch
    offset_std : float
        Input offset voltage
    noise_std : float
        Output noise standard deviation

    Returns
    -------
    x2 : ndarray
        x² (analog squared signal)
    """
    x = np.asarray(x)
    offset = np.random.normal(0, offset_std)
    gain = 1.0 + np.random.normal(0, mismatch_std)

    x2 = gain * (x + offset) ** 2

    if noise_std > 0:
        x2 += np.random.normal(0, noise_std, x2.shape)

    return np.clip(x2, 0, None)


def analog_summer(
    *signals: np.ndarray,
    wire_noise_std: float = 1e-9,
) -> np.ndarray:
    """
    Simulate analog summer: sum = Σ signals.

    Physical implementation: Kirchhoff current summation — all signal
    lines merge into one wire. Zero active devices required.

    This is the third stage of the analog loss chain:
        error² channels → summer → Σ error² (= MSE numerator)

    Parameters
    ----------
    *signals : ndarray
        Analog signals to sum (must be broadcastable)
    wire_noise_std : float
        Wire thermal noise

    Returns
    -------
    total : ndarray
        Σ signals
    """
    total = sum(np.asarray(s) for s in signals)

    if wire_noise_std > 0:
        total += np.random.normal(0, wire_noise_std, total.shape)

    return total


def analog_loss_readout(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_adc_bits: int = 10,
    v_ref_adc: float = 1.8,
) -> np.ndarray:
    """
    Complete analog MSE loss computation chain → single ADC readout.

    Signal chain:
        predictions ─┐
                      ├─ subtractor → error
        targets     ─┘        │
                              ├─ squarer → error² (per element)
                              │
                              ├─ summer  → Σ error²  (scalar)
                              │
                              └─ ADC     → digital loss value

    This is the minimal hardware needed for SPSA/FD training:
    1 analog subtractor + 1 analog squarer + 1 analog summer + 1 ADC.
    The output is ONE scalar — the loss value — digitized for the
    digital controller to compute parameter updates.

    No gradient circuits needed for SPSA/FD — only the scalar loss.

    Parameters
    ----------
    predictions : ndarray, shape (batch, n_out) or (n_out,)
        Network output (analog voltage/current)
    targets : ndarray, same shape
        Ground truth target values
    n_adc_bits : int
        ADC resolution for digitizing the scalar loss
    v_ref_adc : float
        ADC reference voltage

    Returns
    -------
    loss : ndarray (scalar)
        Digitized MSE loss value
    """
    # Stage 1: error = pred - target
    error = analog_subtractor(predictions, targets)

    # Stage 2: error²
    squared = analog_squarer(error)

    # Stage 3: Σ error² (MSE numerator)
    sum_sq = analog_summer(*[squared])  # already summed over all dimensions
    # Actually, we need scalar sum over all elements
    total = np.sum(squared)  # Kirchhoff sum over all current lines

    # Normalize by N for MSE (digital computation — just a scaling factor)
    mse = total / max(np.prod(predictions.shape), 1)

    # Stage 4: ADC digitization
    loss_digital = analog_adc(np.array(mse), n_bits=n_adc_bits, v_ref=v_ref_adc)

    return loss_digital


# ══════════════════════════════════════════════════════════
#  9. Analog Gradient Computation — Multiply-Accumulate Chain
# ══════════════════════════════════════════════════════════

def analog_gilbert_multiplier(
    a: np.ndarray,
    b: np.ndarray,
    gain: float = 1.0,
    mismatch_std: float = 0.02,
    offset_std: float = 0.001,
    noise_std: float = 1e-6,
) -> np.ndarray:
    """
    Simulate Gilbert multiplier cell: out = a × b.

    Physical implementation: Gilbert translinear multiplier (4 BJTs).
    This is the core element of analog gradient computation.

    I_out = gain × I_a × I_b

    In gradient computation, this performs:
        ∂L/∂W_ij = error_i × activation_j

    The same multiplier cell that computes squares in the forward pass
    can be reconfigured for multiplication of two different signals.

    Parameters
    ----------
    a, b : ndarray
        Two analog signals (must be broadcastable)
    gain : float
        Multiplication gain (set by bias current)
    mismatch_std : float
        Transistor pair mismatch
    offset_std : float
        Input offset voltage
    noise_std : float
        Output noise standard deviation

    Returns
    -------
    product : ndarray
        a × b (analog product)
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Offset per input
    offset_a = np.random.normal(0, offset_std, a.shape)
    offset_b = np.random.normal(0, offset_std, b.shape)

    # Gain mismatch (each multiplier cell has independent gain error)
    gain_error = gain * (1 + np.random.normal(0, mismatch_std, a.shape))

    product = gain_error * (a + offset_a) * (b + offset_b)

    if noise_std > 0:
        product += np.random.normal(0, noise_std, product.shape)

    return product


def analog_capacitor_integrator(
    signal: np.ndarray,
    capacitance: float = 1e-12,
    leakage_conductance: float = 1e-9,
    integration_time: float = 1e-6,
    noise_std: float = 1e-9,
    reset: bool = False,
    _stored_charge: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate capacitor-based analog integrator: V_out = (1/C) ∫ I_in dt.

    Physical implementation: op-amp integrator with capacitor in feedback.
    Accumulates charge over time — the analog equivalent of Σ (summation).

    In gradient computation, this accumulates over a batch:
        ∂L/∂θ = Σ_i error_i × ∂f(x_i)/∂θ

    where each sample (x_i, y_i) contributes one charge increment.

    Parameters
    ----------
    signal : ndarray
        Input current to integrate
    capacitance : float
        Integration capacitor value (F). Smaller = higher gain.
    leakage_conductance : float
        Capacitor leakage conductance (S). Causes charge decay over time.
    integration_time : float
        Integration duration per sample (s)
    noise_std : float
        Integration noise (kT/C noise + op-amp noise)
    reset : bool
        If True, reset stored charge to zero (new batch)
    _stored_charge : ndarray, optional
        Previously stored charge (for multi-sample accumulation)

    Returns
    -------
    voltage : ndarray
        Output voltage proportional to accumulated charge
    stored_charge : ndarray
        Stored charge for next accumulation step (pass as _stored_charge)
    """
    dt = integration_time

    # Initialize or reset
    if reset or _stored_charge is None:
        charge = np.zeros_like(np.asarray(signal))
    else:
        charge = np.asarray(_stored_charge)

    # Leakage: charge decays through leakage conductance
    # Q(t+dt) = Q(t) * exp(-G_leak * dt / C)
    decay = np.exp(-leakage_conductance * dt / max(capacitance, 1e-15))
    charge = charge * decay

    # Integration: ΔQ = I * dt
    charge += np.asarray(signal) * dt

    # kT/C noise (thermal noise on capacitor)
    k = 1.38e-23
    T = 300.0
    ktc_noise = np.sqrt(k * T / max(capacitance, 1e-15))
    if noise_std > 0:
        charge += np.random.normal(0, noise_std + ktc_noise, charge.shape)

    # V = Q/C
    voltage = charge / max(capacitance, 1e-15)

    return voltage, charge


def analog_gradient_mac(
    error: np.ndarray,
    activation: np.ndarray,
    mismatch_std: float = 0.02,
    offset_std: float = 0.001,
    noise_std: float = 1e-6,
) -> np.ndarray:
    """
    Analog Multiply-Accumulate for gradient computation.

    Computes one element of the weight gradient in analog:
        ∂L/∂W_ij = error_i × activation_j

    This is the fundamental building block for analog BP. Each (i,j) pair
    needs one Gilbert multiplier cell. For a full layer, this scales as
    N_outputs × N_particles multiplier cells.

    Physical implementation:
        error_i (voltage) ─┐
                            ├─ Gilbert multiplier → I_out = error_i × act_j
        activation_j     ──┘                          │
                                                     ├─ capacitor integrator
                                                     └─ V_grad = ∂L/∂W_ij

    Parameters
    ----------
    error : ndarray, shape (batch, n_out)
        Error signal from output layer (prediction - target)
    activation : ndarray, shape (batch, n_particles)
        Activation values (post-kernel, pre-MVM) from the same layer
    mismatch_std : float
        Multiplier transistor mismatch
    offset_std : float
        Input offset
    noise_std : float
        Output noise

    Returns
    -------
    grad_w : ndarray, shape (n_out, n_particles)
        ∂L/∂W, averaged over batch
    """
    error = np.asarray(error)
    activation = np.asarray(activation)
    batch_size = error.shape[0]

    n_out = error.shape[1] if error.ndim > 1 else 1
    n_p = activation.shape[1] if activation.ndim > 1 else 1

    if error.ndim == 1:
        error = error.reshape(-1, 1)
    if activation.ndim == 1:
        activation = activation.reshape(-1, 1)

    # Initialize gradient accumulator
    grad_w = np.zeros((n_out, n_p), dtype=np.float32)

    # Accumulate over batch: ∂L/∂W = Σ_i error_i ⊗ activation_i
    for i in range(n_out):
        for j in range(n_p):
            # Gilbert multiplier: error[:, i] × activation[:, j]
            product = analog_gilbert_multiplier(
                error[:, i], activation[:, j],
                mismatch_std=mismatch_std,
                offset_std=offset_std,
                noise_std=noise_std,
            )
            # Accumulate over batch (average)
            grad_w[i, j] = float(np.mean(product))

    return grad_w


def analog_center_gradient(
    dx: np.ndarray,
    g: np.ndarray,
    error_weighted: np.ndarray,
    h: np.ndarray,
    k: float = 0.5,
    mismatch_std: float = 0.02,
    noise_std: float = 1e-6,
) -> np.ndarray:
    """
    Compute gradient w.r.t. particle centers in analog domain.

    Analytic form:
        ∂L/∂xc_j = Σ_i error_weighted_i × g_i × 2k × (x_i - xc_j) / h_j²

    Physical meaning:
        (x - xc)/h  ← already available from forward pass (differential_pair output)
        × g          ← activation output
        × error_w    ← backpropagated error through weights
        × 2k/h       ← scaling factor (can be implemented as current mirror ratio)

    Physical implementation:
        (x-xc)/h signal (from forward pass) ─┐
        g signal (from forward pass)       ──├─ 3-input Gilbert multiplier
        error_weighted signal              ──┘   (cascaded 2-stage)

    Parameters
    ----------
    dx : ndarray, shape (batch, n_particles)
        (x - xc)/h from forward pass (single dimension)
    g : ndarray, shape (batch, n_particles)
        Gaussian activation values from forward pass
    error_weighted : ndarray, shape (batch, n_particles)
        Error signal backpropagated through weights: Σ_i error_i × W_ij
    h : ndarray, shape (n_particles,)
        Width parameter per particle
    k : float
        Kernel width coefficient (0.5 for narrow, 0.3 for wide)
    mismatch_std : float
        Multiplier mismatch
    noise_std : float
        Output noise

    Returns
    -------
    grad_xc : ndarray, shape (n_particles,) or (batch, n_particles)
        ∂L/∂xc_j
    """
    dx = np.asarray(dx)
    g = np.asarray(g)
    error_weighted = np.asarray(error_weighted)
    h = np.asarray(h).reshape(1, -1)

    # Scale factor: 2k/h
    scale = 2.0 * k / np.clip(h, 1e-6, None)

    # Three-way product: dx × g × error_weighted
    # Implemented as cascaded 2-input Gilbert multipliers
    # Stage 1: dx × g
    dg = analog_gilbert_multiplier(dx, g, mismatch_std=mismatch_std, noise_std=noise_std)
    # Stage 2: (dx × g) × error_weighted × scale
    dge = analog_gilbert_multiplier(dg, error_weighted, mismatch_std=mismatch_std, noise_std=noise_std)

    grad_xc = -scale * dge  # negative sign from derivative of -d²
    grad_xc = grad_xc.mean(axis=0)  # average over batch

    return grad_xc


def analog_width_gradient(
    d2: np.ndarray,
    g: np.ndarray,
    error_weighted: np.ndarray,
    h: np.ndarray,
    k: float = 0.5,
    mismatch_std: float = 0.02,
    noise_std: float = 1e-6,
) -> np.ndarray:
    """
    Compute gradient w.r.t. particle widths (h) in analog domain.

    Analytic form:
        ∂L/∂h_j = Σ_i error_weighted_i × g_i × 2k × d²_ij / h_j³

    Physical meaning:
        d²      ← already available from forward pass (gilbert_square output)
        × g     ← activation output
        × error_weighted ← backpropagated error
        × 2k/h³  ← scaling factor

    Physical implementation: cascaded Gilbert multipliers, same topology
    as center gradient but with d² instead of dx and h³ scaling instead of h².

    Parameters
    ----------
    d2 : ndarray, shape (batch, n_particles)
        Squared distance values from forward pass
    g : ndarray, shape (batch, n_particles)
        Gaussian activation values
    error_weighted : ndarray, shape (batch, n_particles)
        Error signal backpropagated through weights
    h : ndarray, shape (n_particles,)
        Width parameter per particle
    k : float
        Kernel width coefficient
    mismatch_std : float
    noise_std : float

    Returns
    -------
    grad_h : ndarray, shape (n_particles,)
        ∂L/∂h_j
    """
    d2 = np.asarray(d2)
    g = np.asarray(g)
    error_weighted = np.asarray(error_weighted)
    h = np.asarray(h).reshape(1, -1)

    # Scale factor: 2k/h³
    scale = 2.0 * k / np.clip(h ** 3, 1e-6, None)

    # Three-way product: d² × g × error_weighted
    d2g = analog_gilbert_multiplier(d2, g, mismatch_std=mismatch_std, noise_std=noise_std)
    d2ge = analog_gilbert_multiplier(d2g, error_weighted, mismatch_std=mismatch_std, noise_std=noise_std)

    grad_h = -scale * d2ge  # negative sign from derivative of -d²/h
    grad_h = grad_h.mean(axis=0)

    return grad_h


def analog_weight_gradient(
    error: np.ndarray,
    activation: np.ndarray,
    mismatch_std: float = 0.02,
    noise_std: float = 1e-6,
) -> np.ndarray:
    """
    Compute gradient w.r.t. output weights (W_gauss, W_act) in analog.

    This is the simplest gradient: ∂L/∂W = error ⊗ activation (outer product).

    For the RBF physical implementation, this gradient directly updates
    the memristor conductances in the crossbar array.

    Physical implementation:
        For each (i,j): error_i → Gilbert multiplier ← activation_j
        Output current directly programs memristor G_ij via write pulse.

    Parameters
    ----------
    error : ndarray, shape (batch, n_out)
        Prediction error signal
    activation : ndarray, shape (batch, n_particles)
        Post-kernel activation values
    mismatch_std : float
    noise_std : float

    Returns
    -------
    grad_W : ndarray, shape (n_out, n_particles)
        ∂L/∂W, averaged over batch
    """
    return analog_gradient_mac(
        error, activation,
        mismatch_std=mismatch_std,
        noise_std=noise_std,
    )
