"""
Diffusion-substrate RBF layer
=============================
Project: singal-node-physics-network

DiffusionRBFLayerND is a drop-in replacement for
simulator.layer_nd.PhysicalRBFLayerND on a completely different physical
substrate.  Same parameter vector layout, same method signatures, same
forward(X) -> (out_gauss, out_act) contract, same from_digital() export path.
That is the operational meaning of "the architecture is substrate agnostic":
the trained parameters do not know which medium they are running on.

WHAT IS DIFFERENT FROM THE CIRCUIT SUBSTRATE
--------------------------------------------
The circuit layer computes the six-step chain
    differential pair -> Gilbert squarer -> Kirchhoff sum -> BJT exponential
    -> current mirror -> memristor MVM
i.e. it ENGINEERS exp(-k d^2) out of device physics.

This layer does not compute a kernel at all.  It RELEASES A PULSE and lets the
medium's own Green's function be the kernel:

    source j is a point source at c_j, released at time 0
    the probe is a sensor at the input position X
    after a delay t_j = h_j^2 / (4 k D) the field at the sensor is

        G(|X - c_j|, t_j) = (4 pi D t_j)^(-dim/2) exp(-|X - c_j|^2 / (4 D t_j))

    which for t_j = h_j^2/(4 k D) is exactly exp(-k |X - c_j|^2 / h_j^2) times
    the prefactor.  The prefactor is removed by calibrating the pulse energy.

Consequences worth stating:

  * the kernel needs no calibration of shape, only of amplitude, and the
    amplitude calibration is an explicit per-source gain g_j = (4 pi D t_j)^(dim/2)
  * the width h_j is a TIME, so all N particles can have different widths on one
    physical cell by firing them at different delays
  * because the PDE is linear, the sum over particles is exact.  The whole layer
    is ONE field measurement per output channel, not one per particle, provided
    the sources are injected simultaneously with strengths q_ij = W_ij g_j
    (this is the substrate-native meaning of "the matrix is just per-node
    weights turned into source strengths")
  * the narrow and wide channels are two readout delays of the SAME process

MEASUREMENT COST (per layer, per block of probes read by a parallel sensor array)
    signed medium     : n_vars + (1 if use_pu else 0)
    differential      : 2 * (n_vars + (1 if use_pu else 0))   [thermal-like]
    per-particle naive: N + (1 if use_pu else 0)
    every scheme x2 if the dual channel is used
"""

import itertools

import numpy as np

from .components import (
    DiffusionSubstrate,
    green_prefactor,
    mirror_positions,
    pulse_time,
    quantize_position,
    readout,
    unit_peak_strength,
)

# channel coefficients identical to the digital model / the circuit substrate
KERNELS = (0.5, 0.3)          # narrow, wide
_TINY = 1e-30


class DiffusionRBFLayerND:
    """
    D-dimensional RBF particle layer implemented on a diffusion cell.

    Parameters
    ----------
    centers : (D, N) ndarray
        Particle positions in INPUT space (same convention as the digital model)
    h : (N,) ndarray
        Particle widths in input units
    W_gauss, W_act : (n_vars, N) ndarray
        Readout weights; each column is one particle's strength vector
    use_pu : bool
        Partition-of-unity normalisation.  Costs one extra field measurement.
    substrate : DiffusionSubstrate, optional
        The cell.  Defaults to a 2D carrier-diffusion cell.
    calibrate : bool
        True  : inject q_ij = W_ij * g_j so trained weights transfer verbatim
        False : inject q_ij = W_ij, i.e. forget the Green's-function prefactor.
                Kept so the failure mode can be measured instead of asserted.
    seed : int
        Seed for the readout noise.
    """

    def __init__(self, centers, h, W_gauss, W_act, use_pu=True,
                 substrate=None, calibrate=True, seed=0, return_dtype=np.float32,
                 norm_mode=None, fixed_scale=None):
        self.centers = np.asarray(centers, dtype=np.float64)
        if self.centers.ndim == 1:
            self.centers = self.centers.reshape(1, -1)
        self.h = np.atleast_1d(np.asarray(h, dtype=np.float64)).ravel()
        self.W_gauss = np.asarray(W_gauss, dtype=np.float64)
        self.W_act = np.asarray(W_act, dtype=np.float64)
        self.use_pu = bool(use_pu)
        self.substrate = DiffusionSubstrate() if substrate is None else substrate
        self.calibrate = bool(calibrate)
        self.return_dtype = return_dtype
        # Normalisation modes, so this layer can be dropped into the ablation
        # framework that compares "pu" (production), "none" and "fixed" (frozen
        # scale, strictly non-coupling) on equal terms across substrates.
        #   "pu"    -> divide by the measured sum_j z_j   (1 extra measurement)
        #   "none"  -> raw field
        #   "fixed" -> divide by a constant frozen at init (no coupling)
        self.norm_mode = norm_mode if norm_mode is not None else (
            "pu" if self.use_pu else "none")
        if self.norm_mode not in ("pu", "none", "fixed"):
            raise ValueError(f"unknown norm_mode {self.norm_mode!r}")
        self.fixed_scale = fixed_scale
        self._rng = np.random.default_rng(seed)

        self.dim = int(self.centers.shape[0])
        self.n_particles = int(self.centers.shape[1])
        self.n_vars = int(self.W_gauss.shape[0])

        if self.h.size == 1:
            self.h = np.full(self.n_particles, float(self.h[0]))
        if self.h.size != self.n_particles:
            raise ValueError("h must be scalar or have one entry per particle")
        if self.W_gauss.shape[1] != self.n_particles:
            raise ValueError("W_gauss columns must equal the number of particles")
        self.substrate.scales_for(self.dim)      # fail fast on a bad embedding

    # ------------------------------------------------------------------
    #  Construction from the digital model (same export path as the circuit)
    # ------------------------------------------------------------------
    @classmethod
    def from_digital(cls, digital_layer, **kwargs):
        centers = digital_layer.centers()
        h = digital_layer.widths()
        centers = centers.detach().cpu().numpy() if hasattr(centers, "detach") else centers
        h = h.detach().cpu().numpy() if hasattr(h, "detach") else h
        return cls(
            centers=centers, h=h,
            W_gauss=digital_layer.output_gauss.weight.detach().cpu().numpy(),
            W_act=digital_layer.output_act.weight.detach().cpu().numpy(),
            use_pu=digital_layer.use_pu, **kwargs,
        )

    # ------------------------------------------------------------------
    #  Parameter vector: identical layout to PhysicalRBFLayerND
    #      [ centers (D*N, C-order) | h (N) | W_gauss (V*N) | W_act (V*N) ]
    # ------------------------------------------------------------------
    def get_params_vector(self):
        return np.concatenate([
            self.centers.flatten(), self.h.flatten(),
            self.W_gauss.flatten(), self.W_act.flatten(),
        ])

    def set_params_from_vector(self, vec):
        vec = np.asarray(vec, dtype=np.float64)
        D, N, V = self.dim, self.n_particles, self.n_vars
        n_c, n_h, n_w = D * N, N, V * N
        self.centers = vec[:n_c].reshape(D, N).copy()
        self.h = vec[n_c:n_c + n_h].copy()
        self.W_gauss = vec[n_c + n_h:n_c + n_h + n_w].reshape(V, N).copy()
        self.W_act = vec[n_c + n_h + n_w:n_c + n_h + 2 * n_w].reshape(V, N).copy()

    # ------------------------------------------------------------------
    #  Physics
    # ------------------------------------------------------------------
    def _geometry(self, X):
        """
        Map the abstract input space into metres, apply the positioner
        resolution, and return the squared physical distances (B, N) plus the
        per-particle pulse time and Green's-function prefactor.

        Everything below is in SI units; this is where the abstract-to-physical
        embedding enters.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.dim:
            raise ValueError(f"input has {X.shape[1]} axes, layer expects {self.dim}")

        sub = self.substrate
        sc = sub.scales_for(self.dim)
        res = sub.position_resolution

        probe = quantize_position(X * sc[None, :], res)              # (B, D) metres
        src = quantize_position(self.centers.T * sc[None, :], res)    # (N, D) metres

        return probe, src

    def _image_source_positions(self, src):
        """
        Per-particle image source positions for a reflecting box domain.

        `src` is (N, D) in METRES -- the domain bounds are in metres too, so
        passing the raw input-unit centres here would silently place every image
        kilometres away and zero out the activations.  (An isolated source at the
        origin does not expose that mistake, which is why the unit test for this
        lives on a source that is off-centre.)

        Returns (positions (M, D), owner (M,)) where owner gives the particle each
        image belongs to.  Images cannot be a global translation: every particle
        has its own Neumann image lattice (see components.mirror_positions).
        """
        sub = self.substrate
        n_img = int(sub.n_images)
        if sub.domain is None or n_img <= 0:
            return None, None

        per_axis_cache = []
        for ax in range(self.dim):
            lo, hi = sub.domain[ax]
            cache = {}
            for j in range(self.n_particles):
                c = float(src[j, ax])
                key = round(c, 12)
                if key not in cache:
                    cache[key] = mirror_positions(c, lo, hi, n_img)
            per_axis_cache.append(cache)

        positions, owner = [], []
        for j in range(self.n_particles):
            axes = [per_axis_cache[ax][round(float(src[j, ax]), 12)]
                    for ax in range(self.dim)]
            for combo in itertools.product(*axes):
                positions.append(combo)
                owner.append(j)
                if len(positions) > 4096:
                    raise ValueError("too many image sources; reduce n_images or dim")
        return (np.asarray(positions, dtype=np.float64),
                np.asarray(owner, dtype=np.int64))

    def _activations(self, X, k):
        """
        Unit-peak activations z (B, N) for channel coefficient k, plus the
        per-particle calibration gain g (N,) and the pulse time t (N,).

        z is what the DIGITAL model computes; a unit-strength measurement returns
        z / g_j, which is why the injection strength for weight W_ij has to be
        W_ij * g_j.
        """
        sub = self.substrate
        probe, src = self._geometry(X)

        h_phys = np.maximum(self.h * sub.h_scale, 1e-12)             # metres
        t = pulse_time(h_phys, k, sub.D)                             # seconds
        g = 1.0 / np.maximum(green_prefactor(t, sub.D, sub.dim), _TINY)

        img_pos, owner = self._image_source_positions(src)
        if img_pos is None:
            d = probe[:, None, :] - src[None, :, :]
            r2 = np.einsum("bnd,bnd->bn", d, d)
            z = np.exp(-r2 / (4.0 * sub.D * t[None, :]))
            return z, g, t

        # with images: accumulate the field of every image onto its owner particle
        d = probe[:, None, :] - img_pos[None, :, :]
        r2_ext = np.einsum("bmd,bmd->bm", d, d)
        t_ext = t[owner][None, :]
        z_ext = np.exp(-r2_ext / (4.0 * sub.D * t_ext))
        z = np.zeros((probe.shape[0], self.n_particles))
        np.add.at(z, (slice(None), owner), z_ext)
        return z, g, t

    def _measure_channel(self, z, g, W, add_strength_noise=True, rng=None):
        """
        One field measurement per output channel.

        With calibration the injected strength is q_ij = W_ij * g_j, and the
        measured field is  sum_j q_ij / g_j * z_j = sum_j W_ij z_j  exactly.
        Without calibration q_ij = W_ij, so the field is sum_j W_ij z_j / g_j,
        wrong by the per-particle prefactor.

        The physics is a single simultaneous injection of all sources: the sum
        over particles IS the physical superposition, so the shipped path is the
        matmul below.  `measurement_loop()` performs the same computation one
        source at a time and is used by the verification suite to prove the two
        agree -- i.e. to show that the matmul is a hardware operation and not an
        algebraic convenience.

        strength_noise draws an independent relative error on every injected
        strength q_ij.  This is the exact analogue of memristor conductance
        drift, because on this substrate the strengths ARE the readout weights.
        """
        rng = self._rng if rng is None else rng
        sub = self.substrate
        gain = g[None, :] if self.calibrate else np.ones_like(g)[None, :]
        q = W * gain                                   # (V, N) injected strengths
        if add_strength_noise and sub.strength_noise > 0:
            q = q * (1.0 + rng.normal(0.0, sub.strength_noise, q.shape))
        return (q / g[None, :]) @ z.T                  # (V, B) measured field

    def measurement_loop(self, X, k, W, max_sources=None):
        """
        Literal hardware emulation: inject one source at a time, read the probe
        block, accumulate.  Same result as _measure_channel, by superposition.
        """
        z, g, _t = self._activations(X, k)
        n = z.shape[1] if max_sources is None else min(int(max_sources), z.shape[1])
        total = np.zeros((W.shape[0], X.shape[0] if np.ndim(X) > 1 else 1))
        for j in range(n):
            qj = W[:, j] * (g[j] if self.calibrate else 1.0)
            total = total + qj[:, None] * (z[None, :, j] / g[j])
        return total

    def _readout(self, raw):
        sub = self.substrate
        return readout(raw, bits=sub.sensor_bits, full_scale=sub.sensor_range,
                       noise_rel=sub.sensor_noise, drift=sub.drift,
                       rng=self._rng)

    def forward(self, X):
        """
        Returns (out_gauss, out_act), each (B, n_vars) -- the same contract as
        PhysicalRBFLayerND.forward().
        """
        outs = []
        for ci, (k, W) in enumerate(zip(KERNELS, (self.W_gauss, self.W_act))):
            z, g, _t = self._activations(X, k)
            raw = self._measure_channel(z, g, W)                    # (V, B)
            if self.norm_mode == "pu":
                # one extra unit-strength measurement gives sum_j z_j
                den = self._measure_channel(z, g, np.ones((1, self.n_particles)))
                raw = raw / np.maximum(np.abs(den), _TINY)
            elif self.norm_mode == "fixed":
                if self.fixed_scale is None:
                    raise ValueError("norm_mode='fixed' needs fixed_scale")
                key = "g" if ci == 0 else "a"
                raw = raw / max(float(self.fixed_scale[key]), _TINY)
            outs.append(self._readout(raw).T.astype(self.return_dtype))
        return outs[0], outs[1]

    # ------------------------------------------------------------------
    #  Cost model
    # ------------------------------------------------------------------
    def measurement_plan(self, dual=True):
        """
        Field measurements per layer per block of probes (probes within a block
        are read in parallel by a sensor array, so the cost is per block, not
        per sample).
        """
        pu = 1 if self.norm_mode == "pu" else 0
        chan = 2 if dual else 1
        signed = chan * (self.n_vars + pu)                    # one injection per channel
        diff = 2 * signed                                     # thermal-like: two cells
        naive = chan * (self.n_particles + pu)                # one read per particle
        return {
            "strength_encoded_signed": signed,
            "strength_encoded_differential": diff,
            "per_particle_naive": naive,
            "speedup_vs_naive_signed": naive / max(signed, 1),
            "n_particles": self.n_particles,
            "n_vars": self.n_vars,
            "use_pu": self.norm_mode == "pu",
            "norm_mode": self.norm_mode,
        }

    def describe(self):
        return (f"DiffusionRBFLayerND D={self.dim} N={self.n_particles} "
                f"V={self.n_vars} norm={self.norm_mode} calibrate={self.calibrate}\n"
                f"  {self.substrate.describe()}")
