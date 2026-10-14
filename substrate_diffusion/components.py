"""
Diffusion-field substrate for the RBF particle network
======================================================
Project: singal-node-physics-network
Role:    SECOND physical instantiation of the same five-primitive mapping

The architecture needs a substrate that provides exactly five primitives:

    1. a movable source position            -> particle centre c_j
    2. a distance metric in the input space -> d^2 = sum_k ((x_k - c_k)/h)^2
    3. a localised decaying kernel          -> exp(-k d^2)
    4. a superposable readout               -> sum_j W_ij z_j
    5. a tunable source strength            -> W_ij

A diffusion medium provides all five, and unlike the analog-circuit substrate it
provides three of them EXACTLY rather than by engineering approximation:

  * the kernel is the Green's function of the diffusion equation.  The free-space
    response at time t to a unit point source released at the origin is

        G(r, t) = (4 pi D t)^(-dim/2) * exp(-r^2 / (4 D t))

    which is a Gaussian in r with variance 2 D t.  Setting t = h^2 / (4 k D)
    reproduces the digital kernel exp(-k r^2 / h^2) exactly.  No BJT, no
    calibration fudge, no v_t = 1.0.

  * superposition is EXACT because the PDE is linear:

        u_t = D lap(u),   u(x,0) = sum_j q_j delta(x - c_j)
        =>  u(x,t) = sum_j q_j G(|x - c_j|, t)

    "activate then combine" is therefore a statement about a physical law here,
    not about a circuit that approximates a sum.

  * the kernel width h is set by PULSE TIMING, t_j = h_j^2 / (4 k D).  Each
    source keeps its OWN width, and the knob is a time delay (digitally
    controllable) rather than a device parameter (process-dependent).  Both the
    narrow (k=0.5) and wide (k=0.3) channels are two different readout delays of
    the SAME physical process.

Physical realisations of this one equation (the substrate is the equation, not
the material):

    carrier diffusion in a semiconductor slab : D ~ 1e-3 m^2/s, h ~ 10 um -> t ~ 50 ns
    photon diffusion in a turbid medium        : D ~ 1e+5 m^2/s, h ~  1 mm -> t ~  5 ps
    heat diffusion in a solid                  : D ~ 1e-5 m^2/s, h ~  1 mm -> t ~ 50 s

The first two make this a fast substrate.  The third is the slow textbook case
and the reason "thermal" should not be the headline name for this substrate.

HONEST PHYSICAL CONSTRAINTS (measured by experiments/sub_01_diffusion_verify.py,
not hidden):

  * AMPLITUDE: G carries a prefactor (4 pi D t)^(-dim/2), so sources with
    different widths have different peak amplitudes.  Weights transfer verbatim
    only after a per-source pulse-energy calibration

        q_j = W_ij * (4 pi D t_j)^(dim/2)

    which must be redone whenever a width changes.
  * SIGN: heat sources cannot be negative.  A signed weight therefore needs
    either a differential medium (two identical cells, one driven negative) or a
    medium with two carrier signs (electron/hole diffusion gives both natively).
    The differential scheme costs 2x the measurements.
  * FINITE MEDIUM: a reflecting boundary adds image sources and corrupts the
    kernel; an absorbing boundary or an explicit image-source correction is
    required.
  * ISOTROPY: the kernel is isotropic in PHYSICAL space.  An input space whose
    axes have unequal physical scale cannot be embedded without changing the
    effective per-axis widths, unless the medium itself is anisotropic.
  * NORMALISATION: the partition-of-unity denominator sum_j z_j needs one extra
    unit-strength field measurement per input.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


# ======================================================================
#  1. Kernel: exact Green's function of the diffusion equation
# ======================================================================

def pulse_time(h, k, D):
    """
    Effective measurement time that turns the diffusion kernel into the digital
    kernel:  exp(-r^2/(4 D t)) == exp(-k r^2 / h^2)  =>  t = h^2 / (4 k D).

    h is in metres (physical), D in m^2/s, k is the channel coefficient
    (0.5 narrow, 0.3 wide in the digital model).
    """
    h = np.asarray(h, dtype=np.float64)
    return h ** 2 / (4.0 * float(k) * float(D))


def green_prefactor(t, D, dim):
    """Peak amplitude of the free-space Green's function: (4 pi D t)^(-dim/2)."""
    t = np.asarray(t, dtype=np.float64)
    return (4.0 * np.pi * float(D) * t) ** (-0.5 * float(dim))


def source_response(r2, t, D, dim):
    """Field at squared distance r2 (m^2) from a unit source after time t."""
    t = np.asarray(t, dtype=np.float64)
    return green_prefactor(t, D, dim) * np.exp(-np.asarray(r2, dtype=np.float64)
                                               / (4.0 * float(D) * t))


def unit_peak_strength(h, k, D, dim):
    """
    Pulse energy (source strength) that makes the measured activation peak at
    1.0, i.e. the inverse of the Green's-function prefactor:

        q_unit_j = (4 pi D t_j)^(dim/2) = (2 pi h_j^2 / k)^(dim/2)

    This is the per-source calibration gain.  Injecting q_j = W_ij * q_unit_j
    makes the measured field equal sum_j W_ij z_j with z_j the digital
    activation, so trained weights transfer verbatim.
    """
    return 1.0 / green_prefactor(pulse_time(h, k, D), D, dim)


# ======================================================================
#  2. Finite-difference validation of the closed form (physical fidelity)
# ======================================================================

def solve_heat_1d(x, T, D, sigma0, cfl=0.4, bc="dirichlet"):
    """
    Explicit finite-difference solution of u_t = D u_xx on the grid x, from the
    Gaussian initial condition exp(-x^2/(2 sigma0^2)) / sqrt(2 pi sigma0^2),
    integrated to time T.

    This exists so the substrate's closed-form kernel can be checked against an
    independent numerical solution of the PDE rather than against itself.
    Stability requires D*dt/dx^2 <= 0.5; cfl sets that ratio directly.
    """
    x = np.asarray(x, dtype=np.float64)
    dx = float(x[1] - x[0])
    dt = float(cfl) * dx * dx / float(D)
    n_steps = int(np.ceil(float(T) / dt))
    dt = float(T) / max(n_steps, 1)

    u = np.exp(-x ** 2 / (2.0 * sigma0 ** 2)) / np.sqrt(2.0 * np.pi * sigma0 ** 2)
    r = float(D) * dt / (dx * dx)

    for _ in range(n_steps):
        lap = np.zeros_like(u)
        lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
        u = u + r * lap
        if bc == "dirichlet":
            u[0] = u[-1] = 0.0
        else:                       # neumann: no flux out of the domain
            u[0] = u[1]
            u[-1] = u[-2]
    return u, dict(dx=dx, dt=dt, n_steps=n_steps, cfl=r)


def analytic_gaussian_1d(x, T, D, sigma0):
    """Exact solution for the same initial condition: variance sigma0^2 + 2 D T."""
    x = np.asarray(x, dtype=np.float64)
    s2 = sigma0 ** 2 + 2.0 * float(D) * float(T)
    return np.exp(-x ** 2 / (2.0 * s2)) / np.sqrt(2.0 * np.pi * s2)


# ======================================================================
#  3. Placement and readout chain
# ======================================================================

def quantize_position(x, resolution):
    """
    Positioner / fabrication resolution: source and probe coordinates snap to a
    grid of `resolution` metres.  resolution=0 leaves positions exact.
    """
    x = np.asarray(x, dtype=np.float64)
    if resolution is None or resolution <= 0:
        return x
    return np.round(x / float(resolution)) * float(resolution)


def readout(field, bits=32, full_scale=10.0, noise_rel=0.0, drift=0.0, rng=None):
    """
    Sensor chain: additive noise, ambient drift, full-scale clipping, then
    quantisation to `bits` levels.

    noise_rel is relative to full_scale (Johnson noise of the sensing element);
    drift is a DC offset in units of full_scale.  bits >= 32 means "no
    quantisation", which is what the ideal level uses so that substrate-to-model
    comparisons are not floor-limited by the readout.

    full_scale has no universal value: it is set by the readout electronics and
    the source energy is then chosen to fill it.  The default of 10.0 is large
    enough that the field of a full layer (sum_j W_ij z_j) does not clip.
    """
    # A CALLER THAT OMITS `rng` USED TO GET OS ENTROPY, silently: `default_rng()` with
    # no argument seeds itself from the operating system, so no amount of
    # `np.random.seed(...)` upstream could make such a call reproducible.  No caller in
    # this repository takes that path today -- `DiffusionRBFLayerND._readout` threads
    # `self._rng` and `abl_common._measure_features` derives a seed from the layer's --
    # but a SENSOR CHAIN whose default is nondeterministic is a trap for the next caller.
    # The default is now a FIXED SEED, so an omission is reproducible and a caller who
    # wants entropy has to ask for it by passing an unseeded generator explicitly.
    rng = np.random.default_rng(0) if rng is None else rng
    v = np.asarray(field, dtype=np.float64) + float(drift) * float(full_scale)
    if noise_rel > 0:
        v = v + rng.normal(0.0, float(noise_rel) * float(full_scale), v.shape)
    v = np.clip(v, -float(full_scale), float(full_scale))
    if bits is None or int(bits) >= 32:
        return v
    n_levels = 2 ** int(bits)
    lsb = 2.0 * float(full_scale) / n_levels
    return np.round(v / lsb) * lsb


def mirror_positions(c, lo, hi, n_images):
    """
    Image-source POSITIONS for a reflecting (Neumann) box [lo, hi] on one axis.

    A Neumann reflection produces two image families

        family 1 : c + 2 m W
        family 2 : 2 lo - c + 2 m W          with W = hi - lo

    and both are needed.  Returning positions rather than offsets makes the
    dependence on the source coordinate explicit: every particle has its own
    image lattice, so images can NOT be applied as one global translation of all
    sources.  An implementation that shifts all sources together silently drops
    family 2 and understates the boundary error.
    """
    W = float(hi) - float(lo)
    out = []
    for m in range(-int(n_images), int(n_images) + 1):
        out.append(float(c) + 2.0 * m * W)
        out.append(2.0 * float(lo) - float(c) + 2.0 * m * W)
    vals = np.unique(np.round(np.asarray(out, dtype=np.float64), 12))
    return vals


class DiffusionNoiseConfig:
    """
    Non-ideality levels for the diffusion substrate, mirroring the analog
    circuit's NoiseConfig so the two substrates can be compared level by level.

        "none"   : exact closed-form kernel, infinite domain, no readout error
        "low"    : good lab conditions
        "medium" : typical
        "high"   : worst case
    """

    # Keys are the DiffusionSubstrate field names, so a level dictionary can be
    # splatted straight into the constructor.
    _levels = {
        "none":   dict(sensor_noise=0.0,  sensor_bits=32, drift=0.0,
                       position_resolution=0.0,  n_images=0),
        "low":    dict(sensor_noise=1e-5, sensor_bits=14, drift=1e-4,
                       position_resolution=1e-6, n_images=0),
        "medium": dict(sensor_noise=1e-4, sensor_bits=12, drift=1e-3,
                       position_resolution=1e-5, n_images=1),
        "high":   dict(sensor_noise=1e-3, sensor_bits=10, drift=1e-2,
                       position_resolution=1e-4, n_images=3),
    }
    _current = "none"

    @classmethod
    def set_level(cls, level):
        if level not in cls._levels:
            raise ValueError(f"unknown level {level!r}, choose from {list(cls._levels)}")
        cls._current = level

    @classmethod
    def get(cls):
        return dict(cls._levels[cls._current])

    @classmethod
    def level_dict(cls, level):
        return dict(cls._levels[level])


# ======================================================================
#  4. Substrate description
# ======================================================================

@dataclass
class DiffusionSubstrate:
    """
    One diffusion cell.  Everything is in SI units; the ABSTRACT input space of
    the network must be mapped into metres by `axis_scale`, and this mapping is
    the user's responsibility (it is the price of porting to a physical cell).

    D                : diffusion coefficient, m^2/s
    dim              : spatial dimension of the field, sets the kernel prefactor
    axis_scale       : metres per unit of input space, one entry per input axis
    h_scale          : metres per unit of the network's width h (default: mean
                       of axis_scale).  Separate, because h lives in input units
                       while the kernel lives in metres.
    domain           : optional ((lo, hi), ...) per axis in metres
    n_images         : mirror images per axis when the domain reflects
    sensor_bits/full_scale/sensor_noise/drift : readout chain (bits>=32 = no
                       quantisation, which is the default ideal case)
    position_resolution : placement resolution in metres
    strength_noise   : relative error on every injected strength q_ij, drawn
                       independently per (output, particle).  This is the exact
                       analogue of memristor conductance drift on the circuit
                       substrate, because on this substrate the injected
                       strengths ARE the readout weights.
    signed_sources   : True  -> source strength may be negative (electron/hole
                       or field amplitude), False -> thermal-like, needs the
                       differential two-cell scheme (2x measurements)
    """
    D: float = 1.0e-3
    dim: int = 2
    axis_scale: Sequence[float] = (1.0e-3,)
    h_scale: Optional[float] = None
    domain: Optional[Sequence[Tuple[float, float]]] = None
    n_images: int = 0
    sensor_bits: int = 32
    sensor_range: float = 10.0
    sensor_noise: float = 0.0
    drift: float = 0.0
    position_resolution: float = 0.0
    strength_noise: float = 0.0
    signed_sources: bool = True
    name: str = "semiconductor carrier diffusion"

    def __post_init__(self):
        self.axis_scale = tuple(float(s) for s in self.axis_scale)
        if self.h_scale is None:
            self.h_scale = float(np.mean(self.axis_scale))

    # -- helpers ------------------------------------------------------
    def scales_for(self, dim):
        """axis_scale broadcast to `dim`, with a clear error if it cannot be."""
        s = np.asarray(self.axis_scale, dtype=np.float64)
        if s.size == dim:
            return s
        if s.size == 1:
            return np.full(dim, float(s[0]))
        raise ValueError(
            f"axis_scale has {s.size} entries but the input has {dim} axes; "
            f"the physical embedding must be given explicitly for anisotropic "
            f"input spaces"
        )

    def apply_noise_level(self, level):
        cfg = DiffusionNoiseConfig._levels[level]
        for key, val in cfg.items():
            setattr(self, key, val)
        return self

    def describe(self):
        dom = "infinite" if self.domain is None else f"{self.domain}"
        return (f"{self.name}: D={self.D:g} m^2/s dim={self.dim} "
                f"axis_scale={self.axis_scale} h_scale={self.h_scale:g} m/unit "
                f"domain={dom} images={self.n_images} "
                f"sensor={self.sensor_bits}bit noise={self.sensor_noise:g} "
                f"pos_res={self.position_resolution:g} m "
                f"strength_noise={self.strength_noise:g} "
                f"sign={'signed' if self.signed_sources else 'differential'}")


# A few ready-made cells, so the same model can be run on very different physics.
PRESETS = {
    "circuit_like": DiffusionSubstrate(
        D=1.0e-3, dim=2, axis_scale=(1.0e-3, 1.0e-3), name="carrier diffusion"),
    "fast_photon": DiffusionSubstrate(
        D=1.0e5, dim=3, axis_scale=(1.0e-3, 1.0e-3, 1.0e-3),
        name="photon diffusion in a turbid medium"),
    "slow_thermal": DiffusionSubstrate(
        D=1.0e-5, dim=3, axis_scale=(1.0e-3, 1.0e-3, 1.0e-3),
        signed_sources=False, name="heat diffusion in a solid"),
}
