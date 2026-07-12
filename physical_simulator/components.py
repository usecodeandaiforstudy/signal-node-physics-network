"""
Physical component simulation

Each function corresponds to an analog circuit element, computing ideal transfer function + non-ideal effects.

Non-ideality sources:
  - Thermal noise (Johnson-Nyquist): additive noise proportional to temperature
  - Process mismatch: random deviation of device parameters from nominal values
  - Saturation clipping: output range constraint due to supply rail limits
  - Conductance drift (Memristor Drift): memristor conductance drift over time/read operations
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
