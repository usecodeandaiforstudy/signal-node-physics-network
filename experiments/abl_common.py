"""
Shared infrastructure for the three architecture ablations
==========================================================
Project: singal-node-physics-network
Scope:   abl_01 (capacity split), abl_02 (locality), abl_03 (mismatch sensitivity)

WHY THIS FILE EXISTS
--------------------
The three ablations ask questions the current codebase cannot answer, because
the production trainer optimises centers, widths AND readout weights jointly
(HiMALND.get_params_vector / set_params_from_vector), and there is no way to
restrict training to a subset of them.

Everything below is therefore built on top of the *unmodified* production
components (physical_simulator.components) and mirrors
simulator.layer_nd.PhysicalRBFLayerND.forward() exactly, adding three things:

  1. maskable parameter groups  -> train only {centers,h}, only {W}, or both
  2. three normalization modes  -> "pu" (production behaviour), "none",
                                   "fixed" (non-coupling scale control)
  3. access to pre-MVM activations z -> required to *measure* particle
                                   independence instead of merely asserting it

`selftest()` asserts that forward_ex() is numerically identical to
PhysicalRBFLayerND.forward() for both modes the production layer supports.
Every ablation script calls it first and refuses to report if it fails.

NO PRODUCTION FILE IS MODIFIED BY ANY abl_* SCRIPT.

MODEL FAMILY USED BY THE ABLATIONS
----------------------------------
A stack of wire-direct RBF layers with a *scalar* readout:

    h_0 = X
    out_k = W_k^g . z_k^g(h_{k-1}) + W_k^a . z_k^a(h_{k-1})     (single channel
    pred  = sum_k out_k[:, 0]                                    in these runs)
    h_k   = out_k                        <- wire-direct: no inter-layer matrix

This is the configuration that actually matches "nodes replace matrices"
(compare experiments/ref2d/ref2d_05_himal_training.py output_dims=(1,1,1) and
experiments/reg_02_direct_residual.py pred = h1[:,0]+h2[:,0]+h3[:,0]), as
opposed to the CIFAR model whose parameters are ~70% readout matrix.

Parameter share is controlled by out_dims:
    geometry share = sum_k (dim_k + 1) * N_k  /  total
    readout  share = sum_k  out_dim_k  * N_k  /  total
Layers that CONTRACT (out_dim < in_dim) put capacity in geometry; layers that
EXPAND put capacity in the readout matrix.
"""

import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physical_simulator.components import (          # noqa: E402
    NoiseConfig,
    V_SUPPLY,
    differential_pair,
    gilbert_square,
    bjt_gaussian_kernel,
    current_mode_normalize,
    memristor_mvm,
    kirchhoff_sum,
)
from simulator.layer_nd import PhysicalRBFLayerND    # noqa: E402
from substrate_diffusion import (                    # noqa: E402
    DiffusionNoiseConfig,
    DiffusionRBFLayerND,
    DiffusionSubstrate,
)

# forward_fast() skips the unconditional sqrt(|z|)*1e-6 noise that
# bjt_gaussian_kernel adds when noise_std is None (which is how the production
# layer calls it).  That term is ~1e-6 relative -- below every effect measured
# here -- and it costs two array-scaled RNG draws per layer per forward.  Note
# this also means NoiseConfig("none") is NOT noise-free in production; that is
# reported as a finding, not hidden.
FAST_SKIP_KERNEL_NOISE = True

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
CACHE_DIR = os.path.join(OUT_DIR, "abl_cache")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

ROOT_DIR = os.path.dirname(OUT_DIR)


def rel(path):
    """
    A path as a reader of a report should see it: relative to the repository root.

    Reports print where their artifacts went, and those lines used to carry the absolute
    path -- `Report: F:\\...\\outputs\\...` -- which puts the maintainer's drive layout into
    every published report.  It is not a secret, since no user name is in it, but it is
    information about the machine rather than about the measurement, and it makes a report
    non-portable.  Wrapping the value here keeps the path usable for I/O (which needs the
    absolute form) while the printed form stays repository-relative.

    Reports ALREADY written keep their absolute lines.  They are evidence, and editing
    evidence to tidy its formatting is the trade this repository refuses elsewhere.
    """
    try:
        return os.path.relpath(str(path), ROOT_DIR).replace(os.sep, "/")
    except ValueError:                      # a different drive: no relative path exists
        return str(path)

IDEAL = dict(mismatch=0.0, noise=0.0, drift=0.0, stuck=0.0, v_t=1.0)

# Architecture family. Particle counts are held FIXED across the three, so any
# difference comes from where the capacity sits, not from how much of it exists.
ARCHS = {
    "G": dict(particles=(24, 16, 12), out_dims=(2, 2, 1),
              label="geometry-dominated"),
    "M": dict(particles=(24, 16, 12), out_dims=(4, 4, 2), label="balanced"),
    "W": dict(particles=(24, 16, 12), out_dims=(16, 8, 4),
              label="matrix-dominated"),
    # The strictly matrix-free forms the thesis actually claims: one scalar per
    # layer, summed as a residual (identical to the repo's own reg_02 design).
    # Geometry then carries 77-81% of the stored quantities instead of 47-69%.
    # F2 keeps the SAME total particle count (52) as G/M/W but front-loads the
    # particles into layer 1, which is where an input-space distance lives.
    "F": dict(particles=(24, 16, 12), out_dims=(1, 1, 1),
              label="matrix-free (scalar readout)"),
    "F2": dict(particles=(40, 8, 4), out_dims=(1, 1, 1),
               label="matrix-free, front-loaded"),
}

NORM_MODES = ("pu", "none", "fixed")

# ----------------------------------------------------------------------
#  Substrate dispatch
# ----------------------------------------------------------------------
# The network is defined by its parameters and its five primitives, not by a
# medium, so everything below is substrate-agnostic:
#
#   "circuit"   -> simulator.layer_nd.PhysicalRBFLayerND (6-element analog chain)
#   "diffusion" -> substrate_diffusion.DiffusionRBFLayerND (Green's function of
#                  the diffusion equation, verified against a PDE solve in
#                  experiments/sub_01_diffusion_verify.py)
#
# Both expose an identical parameter vector layout, so a model trained on one
# runs on the other with no change to the trainers.
SUBSTRATES = ("circuit", "diffusion")

# Mutable default for the diffusion cell used by build_model, mirroring the way
# NoiseConfig holds global state for the circuit substrate.
_DIFF_STATE = {"kwargs": {}}


def set_diffusion_defaults(**kwargs):
    """
    Set the diffusion cell used by subsequent build_model calls, e.g.
        set_diffusion_defaults(sensor_noise=1e-4, position_resolution=1e-5)
    level="none"/"low"/"medium"/"high" applies a whole preset at once.
    """
    lvl = kwargs.pop("level", None)
    if lvl is not None:
        _DIFF_STATE["kwargs"].update(DiffusionNoiseConfig.level_dict(lvl))
    _DIFF_STATE["kwargs"].update(kwargs)


def make_cell(substrate):
    """One cell shared by every layer of a model (they sit in one medium)."""
    if substrate == "circuit":
        return None
    if substrate == "diffusion":
        base = dict(D=1e-3, dim=3, axis_scale=(1e-3,), h_scale=1e-3)
        base.update(_DIFF_STATE["kwargs"])
        return DiffusionSubstrate(**base)
    raise ValueError(f"unknown substrate {substrate!r}")


def make_layer(substrate, centers, h, W_gauss, W_act, use_pu=True,
               norm_mode="none", cell=None, seed=0, fixed_scale=None):
    """Build one layer of the requested substrate with identical parameters."""
    if substrate == "circuit":
        return PhysicalRBFLayerND(centers, h, W_gauss, W_act, use_pu=use_pu)
    if substrate == "diffusion":
        return DiffusionRBFLayerND(
            centers, h, W_gauss, W_act,
            use_pu=(norm_mode == "pu"), substrate=cell, seed=seed,
            norm_mode=norm_mode, fixed_scale=fixed_scale,
        )
    raise ValueError(f"unknown substrate {substrate!r}")


def layer_forward(layer, X, norm_mode="none", scales=None, substrate="circuit",
                  want_z=False):
    """
    Substrate-agnostic forward.  Returns a dict with at least out_g / out_a.
    With want_z the diffusion path additionally exposes the NORMALISED
    per-particle basis value, which ablation B needs for the coupling
    measurement.
    """
    if substrate == "circuit":
        return forward_fast(layer, X, norm_mode=norm_mode, scales=scales)
    if substrate == "diffusion":
        out_g, out_a = layer.forward(X)
        r = {"out_g": np.asarray(out_g, dtype=np.float64),
             "out_a": np.asarray(out_a, dtype=np.float64)}
        if want_z:
            from substrate_diffusion import KERNELS
            Xf = np.asarray(X, dtype=np.float64)
            z = layer._activations(Xf, KERNELS[0])[0]
            # Return the quantity that is comparable across substrates: the
            # per-particle basis value multiplying the readout weight, i.e. the
            # NORMALISED activation.  On the circuit substrate the normalisation
            # is applied per particle before the MVM; on the diffusion substrate
            # it is one shared division of the summed field.  The product W.z is
            # identical either way (checked by selftest_cross_substrate), so the
            # coupling measurement has to normalise identically or it would
            # report a substrate difference that does not exist in the function.
            if norm_mode == "pu":
                z = z / np.maximum(z.sum(axis=1, keepdims=True), 1e-30)
            elif norm_mode == "fixed":
                if scales is None:
                    raise ValueError("norm_mode='fixed' requires scales")
                z = z / max(float(scales["g"]), 1e-30)
            r["z_g"] = z
        return r
    raise ValueError(f"unknown substrate {substrate!r}")


def apply_substrate_noise(model, substrate, **kwargs):
    """
    Damage a trained model's medium in place.

    circuit   : forwarded to set_noise(mismatch=..., drift=...) on NoiseConfig
    diffusion : sets fields on the cell the model's layers share
                (position_resolution ~ additive geometry damage;
                 strength_noise ~ per-weight multiplicative damage, which is
                 what memristor conductance drift is on the circuit substrate)
    """
    if substrate == "circuit":
        set_noise(**kwargs)
        return
    if substrate == "diffusion":
        cell = getattr(model, "cell", None)
        if cell is None:
            raise ValueError("diffusion model has no cell to damage")
        for key, val in kwargs.items():
            if not hasattr(cell, key):
                raise AttributeError(f"diffusion cell has no field {key!r}")
            setattr(cell, key, val)
        return
    raise ValueError(f"unknown substrate {substrate!r}")


# --------------------------------------------------------------
#  Noise control (finer-grained than NoiseConfig's 4 presets)
# --------------------------------------------------------------

def set_noise(mismatch=0.0, noise=0.0, drift=0.0, stuck=0.0, v_t=1.0,
              name="abl_custom"):
    """
    Register and activate a custom non-ideality level.

    NoiseConfig ships four presets (components.py, _levels) that always move
    mismatch and drift together. Ablation C needs them decoupled:
      mismatch -> perturbs h (differential_pair gain) and the MVM gain
                  (gilbert_square) and the current mirror  = GEOMETRY-ish
      drift    -> multiplies memristor conductances W        = WEIGHT-only
    """
    NoiseConfig._levels[name] = {
        "mismatch": float(mismatch),
        "noise": float(noise),
        "drift": float(drift),
        "stuck": float(stuck),
        "v_t": float(v_t),
    }
    NoiseConfig.set_level(name)


def set_ideal():
    set_noise(name="abl_ideal", **IDEAL)


# --------------------------------------------------------------
#  Physical forward, with exposed activations and norm modes
# --------------------------------------------------------------

def _normalize(z, mode, scale, mismatch):
    if mode == "pu":
        # production behaviour, incl. the mirror mismatch factor of 0.5
        return current_mode_normalize(z, mirror_mismatch_std=mismatch * 0.5)
    if mode == "none":
        return z
    if mode == "fixed":
        if scale is None:
            raise ValueError("norm_mode='fixed' requires a calibrated scale")
        return z / float(scale)
    raise ValueError(f"unknown norm_mode: {mode}")


def forward_ex(layer, X, norm_mode="none", scales=None):
    """
    Exact replica of PhysicalRBFLayerND.forward(), plus:
      - returns pre-MVM activations z_g / z_a  (needed by ablation B)
      - norm_mode in {"pu", "none", "fixed"}

    Call order of the component functions is identical to the production layer,
    including RNG consumption, so seeded results match bit-for-bit.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    cfg = NoiseConfig.get_config()
    mish = cfg["mismatch"]
    nz = cfg["noise"]

    d2 = None
    for i in range(layer.dim):
        dx = differential_pair(
            X[:, i].astype(np.float32).ravel(), layer.centers[i, :], layer.h,
            mismatch_std=mish, noise_std=nz,
        )
        d2_i = gilbert_square(
            dx, np.zeros_like(dx), mismatch_std=mish, noise_std=nz,
        )
        d2 = d2_i if d2 is None else kirchhoff_sum(d2, d2_i)

    if d2 is None:
        d2 = np.zeros((X.shape[0], layer.n_particles), dtype=np.float32)

    v_t = cfg.get("v_t", 1.0)
    z_g = bjt_gaussian_kernel(d2, k=0.5, temperature_voltage=v_t)
    z_a = bjt_gaussian_kernel(d2, k=0.3, temperature_voltage=v_t)

    sg = None if scales is None else scales.get("g")
    sa = None if scales is None else scales.get("a")
    zg = _normalize(z_g, norm_mode, sg, mish)
    za = _normalize(z_a, norm_mode, sa, mish)

    out_g = memristor_mvm(
        zg, layer.W_gauss,
        conductance_drift_std=cfg["drift"], read_noise_std=nz,
        stuck_fraction=cfg["stuck"],
    )
    out_a = memristor_mvm(
        za, layer.W_act,
        conductance_drift_std=cfg["drift"], read_noise_std=nz,
        stuck_fraction=cfg["stuck"],
    )
    return {"d2": d2, "z_g": zg, "z_a": za, "out_g": out_g, "out_a": out_a}


def selftest_cross_substrate(verbose=True):
    """
    Assert that the two substrates compute the SAME function from the SAME
    parameters, in the ideal limit, through the ablation framework's own code
    path (layer_forward).  This is what licenses running ablations A/B/C on
    either medium and comparing the answers.
    """
    rng = np.random.default_rng(0)
    D, N, V = 3, 7, 2
    centers = rng.uniform(-2, 2, (D, N)).astype(np.float32)
    h = rng.uniform(0.8, 1.6, N).astype(np.float32)
    Wg = rng.uniform(-0.3, 0.3, (V, N)).astype(np.float32)
    Wa = rng.uniform(-0.3, 0.3, (V, N)).astype(np.float32)
    X = rng.uniform(-2, 2, (13, D)).astype(np.float32)

    set_ideal()
    errs = {}
    for mode in ("pu", "none"):
        cir = make_layer("circuit", centers, h, Wg, Wa,
                         use_pu=(mode == "pu"), norm_mode=mode)
        cell = DiffusionSubstrate(D=1e-3, dim=D, axis_scale=(1e-3,))
        dif = make_layer("diffusion", centers, h, Wg, Wa,
                         use_pu=(mode == "pu"), norm_mode=mode, cell=cell)
        a = layer_forward(cir, X, mode, None, substrate="circuit")
        b = layer_forward(dif, X, mode, None, substrate="diffusion")
        errs["g:" + mode] = maxerr(a["out_g"], b["out_g"])
        errs["a:" + mode] = maxerr(a["out_a"], b["out_a"])
    worst = max(errs.values())
    # the circuit path is float32 and the digital PU adds a 1e-8 guard, so 1e-5
    # is the right tolerance for "same function"
    ok = worst < 1e-5
    if verbose:
        print("  [selftest] circuit vs diffusion, same params: "
              + "  ".join(f"{k}={v:.2e}" for k, v in errs.items())
              + f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"substrates disagree: {errs}")
    return errs


def selftest(verbose=True):
    """
    Assert that both forward paths reproduce PhysicalRBFLayerND.forward()
    under ideal conditions, for both norm modes the production layer
    implements.  Raises on mismatch.

    forward_ex   : verbatim replica, same per-dimension call order -> ~1e-7
    forward_fast : vectorised over dimensions, used for training/evaluation
                   (same math and same per-entry noise distributions, different
                   float32 summation order) -> ~1e-6
    """
    rng = np.random.default_rng(0)
    D, N, V = 3, 6, 4
    centers = rng.uniform(-2, 2, (D, N)).astype(np.float32)
    h = rng.uniform(0.8, 1.6, N).astype(np.float32)
    Wg = rng.uniform(-0.3, 0.3, (V, N)).astype(np.float32)
    Wa = rng.uniform(-0.3, 0.3, (V, N)).astype(np.float32)
    X = rng.uniform(-2, 2, (11, D)).astype(np.float32)

    set_ideal()
    errs = {}
    for use_pu, mode in ((True, "pu"), (False, "none")):
        layer = PhysicalRBFLayerND(centers, h, Wg, Wa, use_pu=use_pu)
        np.random.seed(0)
        pg, pa = layer.forward(X)
        np.random.seed(0)
        r = forward_ex(layer, X, norm_mode=mode)
        np.random.seed(0)
        rf = forward_fast(layer, X, norm_mode=mode)
        errs["ex:" + mode] = max(
            float(np.max(np.abs(pg - r["out_g"]))),
            float(np.max(np.abs(pa - r["out_a"]))),
        )
        errs["fast:" + mode] = max(
            float(np.max(np.abs(pg - rf["out_g"]))),
            float(np.max(np.abs(pa - rf["out_a"]))),
        )
    worst = max(errs.values())
    ok = worst < 1e-4
    if verbose:
        print("  [selftest] vs production layer: "
              + "  ".join(f"{k}={v:.2e}" for k, v in errs.items())
              + f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"forward path diverges from the production layer: {errs}")
    return errs


def forward_fast(layer, X, norm_mode="none", scales=None):
    """
    Vectorised equivalent of forward_ex(): all D dimensions at once.

    Identical mathematics and identical per-entry noise distributions.  It
    differs from the per-dimension replica only in (a) the float32 summation
    order of the Kirchhoff D-sum and (b) the realization order of the RNG
    draws -- both verified by selftest().  Used for training and evaluation
    because the faithful replica costs ~10x more in pure Python.

    Production details replicated exactly:
      * differential_pair draws an independent h mismatch per dimension
      * gilbert_square is called with offset_std=0, so its offsets are exactly
        zero and can be dropped; its per-dimension gain mismatch is kept
      * d2 is clipped at 0 inside gilbert_square, i.e. BEFORE the Kirchhoff sum
      * kirchhoff_sum adds a 1e-9 wire noise per accumulation step
      * bjt_gaussian_kernel is called without noise_std, so production always
        adds sqrt(|z|)*1e-6 noise (this is why even NoiseConfig("none") is not
        exactly noise-free)
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    cfg = NoiseConfig.get_config()
    mish = float(cfg["mismatch"])
    nz = float(cfg["noise"])
    D, N = layer.dim, layer.n_particles

    # 1-2. differential_pair + gilbert_square, all dimensions at once
    # dx is laid out (batch, particle, dim), so the per-dimension quantities are
    # drawn as (D, N) exactly like production then transposed to (1, N, D).
    # Zero-std draws are skipped: normal(0, 0) is exactly 0, so this is an exact
    # shortcut that removes most of the RNG overhead during training.
    if mish > 0:
        h_noisy = layer.h.reshape(1, N) * (1.0 + np.random.normal(0.0, mish, (D, N)))
    else:
        h_noisy = np.broadcast_to(layer.h.reshape(1, N), (D, N))
    h_noisy = np.clip(h_noisy, 1e-6, None)                       # (D,N)
    dx = ((X[:, None, :] - layer.centers.T[None, :, :])
          / h_noisy.T[None, :, :])                               # (B,N,D)
    if nz > 0:
        dx = dx + np.random.normal(0.0, nz, dx.shape)

    if mish > 0:
        gain = (1.0 + np.random.normal(0.0, mish, (D, N))).T[None, :, :]  # (1,N,D)
    else:
        gain = 1.0
    d2_d = gain * (dx * dx)
    if nz > 0:
        d2_d = d2_d + np.random.normal(0.0, nz, d2_d.shape)
    d2_d = np.clip(d2_d, 0, None)

    # 3. Kirchhoff summation across D (+ the default wire noise per step)
    d2 = d2_d.sum(axis=2)
    if D > 1:
        d2 = d2 + np.random.normal(0.0, 1e-9 * np.sqrt(D - 1), d2.shape)
    d2 = np.asarray(d2, dtype=np.float32)

    # 4. BJT subthreshold exponential, dual channel
    v_t = max(float(cfg.get("v_t", 1.0)), 1e-6)

    def _kernel(k):
        e = np.clip(-k * d2.astype(np.float64) / v_t, -50.0, 20.0)
        z = np.exp(e).astype(np.float32)
        if not FAST_SKIP_KERNEL_NOISE:
            z = z + np.random.normal(0.0, np.sqrt(np.abs(z)) * 1e-6, z.shape)
        return np.clip(z, 0, 1e3)

    z_g = _kernel(0.5)
    z_a = _kernel(0.3)

    sg = None if scales is None else scales.get("g")
    sa = None if scales is None else scales.get("a")
    zg = _normalize(z_g, norm_mode, sg, mish)
    za = _normalize(z_a, norm_mode, sa, mish)

    # Exact shortcut of memristor_mvm when the only non-idealities are drift and
    # stuck devices: with both zero, g_matrix == W and the drift draw is exactly
    # zero, so it can be skipped (removes 2 RNG draws per channel per forward).
    use_plain_mvm = float(cfg["drift"]) <= 0 and float(cfg["stuck"]) <= 0

    def _mvm(inp, W):
        if not use_plain_mvm:
            return memristor_mvm(
                inp, W, conductance_drift_std=cfg["drift"],
                read_noise_std=nz, stuck_fraction=cfg["stuck"],
            )
        out = inp @ W.T
        if nz > 0:
            out = out + np.random.normal(0.0, nz, out.shape)
        return np.clip(out, -V_SUPPLY, V_SUPPLY)

    out_g = _mvm(zg, layer.W_gauss)
    # Exact shortcut: with W_act identically zero AND no drift/read-noise/stuck
    # non-ideality, memristor_mvm returns exactly zeros, so skip the channel
    # (this is the single-channel configuration used by the ablations).
    skip_a = (not np.any(layer.W_act)) and nz <= 0 and use_plain_mvm
    if skip_a:
        out_a = np.zeros((X.shape[0], layer.n_vars), dtype=np.float32)
    else:
        out_a = _mvm(za, layer.W_act)
    return {"d2": d2, "z_g": zg, "z_a": za, "out_g": out_g, "out_a": out_a}


# --------------------------------------------------------------
#  Model: wire-direct scalar-readout RBF stack
# --------------------------------------------------------------

class SourceStack:
    """
    Stack of PhysicalRBFLayerND with wire-direct connections and the scalar
    residual readout used by the repo's regression path:

        pred = sum_k (out_k[:, 0])

    W_act is left at zero and is NOT counted as a parameter (single channel),
    matching HiMALND._init_layer(task="regression").
    """

    def __init__(self, layers, norm_mode="none", scales=None, dual_channel=False,
                 substrate="circuit", cell=None):
        self.layers = list(layers)
        self.norm_mode = norm_mode
        self.scales = list(scales) if scales is not None else [None] * len(self.layers)
        self.dual_channel = bool(dual_channel)
        self.substrate = substrate
        self.cell = cell

    # -- forward --
    def forward(self, X):
        h = np.asarray(X, dtype=np.float32)
        pred = None
        for k, layer in enumerate(self.layers):
            r = layer_forward(layer, h, self.norm_mode, self.scales[k],
                              substrate=self.substrate)
            out = r["out_g"] + r["out_a"]
            pred = out[:, :1].copy() if pred is None else pred + out[:, :1]
            h = out
        return pred

    def activations(self, X):
        """Return the per-layer (input, z_g, out) trace; used by ablation B."""
        h = np.asarray(X, dtype=np.float32)
        trace = []
        for k, layer in enumerate(self.layers):
            r = layer_forward(layer, h, self.norm_mode, self.scales[k],
                              substrate=self.substrate, want_z=True)
            out = r["out_g"] + r["out_a"]
            trace.append({"input": h.copy(), "z_g": r["z_g"], "out": out})
            h = out
        return trace

    # -- flat parameter vector --
    def get_params(self):
        return np.concatenate([l.get_params_vector() for l in self.layers])

    def set_params(self, vec):
        vec = np.asarray(vec)
        off = 0
        for l in self.layers:
            n = l.get_params_vector().size
            l.set_params_from_vector(vec[off:off + n].copy())
            off += n

    def n_params(self):
        return int(self.get_params().size)

    # -- parameter grouping --
    def layout(self):
        """Global [start, end) offsets of every parameter block, per layer."""
        out = []
        off = 0
        for layer in self.layers:
            D, N, V = layer.dim, layer.n_particles, layer.n_vars
            n_c, n_h, n_w = D * N, N, V * N
            d = {"centers": (off, off + n_c)}
            off += n_c
            d["h"] = (off, off + n_h)
            off += n_h
            d["W_gauss"] = (off, off + n_w)
            off += n_w
            d["W_act"] = (off, off + n_w)
            off += n_w
            out.append(d)
        return out

    def index_set(self, arm):
        """
        arm in {"geometry", "weights", "hybrid", "full"}.
          geometry = particle centers + widths   (the "node coordinates")
          weights  = readout weights             (W_gauss = per-source strengths)
          hybrid   = geometry, with the strengths CALIBRATED in closed form
                     instead of being probed (see solve_readout_closed_form)
          full     = geometry + weights
        """
        lay = self.layout()
        if arm in ("geometry", "hybrid", "hybrid_once"):
            keys = ("centers", "h")
        elif arm == "weights":
            keys = ("W_gauss",)
        elif arm == "full":
            keys = ("centers", "h", "W_gauss")
        else:
            raise ValueError(f"unknown arm: {arm}")
        if self.dual_channel and "W_gauss" in keys:
            keys = keys + ("W_act",)
        parts = []
        for d in lay:
            for k in keys:
                a, b = d[k]
                parts.append(np.arange(a, b, dtype=np.int64))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def breakdown(self):
        """Count stored analog quantities, split geometry vs readout matrix."""
        geom = 0
        read = 0
        mult = 2 if self.dual_channel else 1
        for layer in self.layers:
            D, N, V = layer.dim, layer.n_particles, layer.n_vars
            geom += D * N + N
            read += mult * V * N
        total = geom + read
        return {
            "geometry": int(geom),
            "readout": int(read),
            "total": int(total),
            "geometry_share": float(geom) / max(total, 1),
            "readout_share": float(read) / max(total, 1),
            "weight_share": float(read) / max(total, 1),
        }

    def describe(self):
        b = self.breakdown()
        dims = [l.dim for l in self.layers]
        outs = [l.n_vars for l in self.layers]
        return (f"layers={len(self.layers)} in_dims={dims} out_dims={outs} "
                f"geom={b['geometry']} ({b['geometry_share']*100:.0f}%) "
                f"readout={b['readout']} ({b['readout_share']*100:.0f}%)")


def build_model(arch_key, X_ref, seed=0, norm_mode="none", dual_channel=False,
                h_scale=2.0, substrate="circuit"):
    """
    Build an ARCHS configuration with a cascading, data-scaled initialization:
    each layer's centers span the observed range of ITS OWN input, and its
    widths start proportional to the per-dimension spread of that input.

    `substrate` selects the medium ("circuit" or "diffusion"); the layer
    parameters, the parameter vector layout and every downstream routine are
    identical either way.  That is the operational meaning of substrate
    agnosticism: nothing above this function needs to know.

    Random-unif + data-scaled h is used instead of k-means so that the ablation
    does not silently depend on clustering quality; this is noted as a confound
    in the reports (k-means init is a documented improvement, see
    reg_01/reg_06 and Wurzberger & Schwenker 2024).
    """
    if substrate not in SUBSTRATES:
        raise ValueError(f"unknown substrate {substrate!r}")
    cfg = ARCHS[arch_key]
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # The cascading init runs the physical forward chain, so it must be ideal
    # regardless of what noise level the caller left active.
    _prev_level = NoiseConfig._current
    set_ideal()

    cell = make_cell(substrate)
    X_ref = np.asarray(X_ref, dtype=np.float32)
    layers = []
    h_in = X_ref
    dim_in = X_ref.shape[1]

    for N, d_out in zip(cfg["particles"], cfg["out_dims"]):
        lo = h_in.min(axis=0)
        hi = h_in.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        centers = (lo[None, :] + span[None, :] * rng.random((N, dim_in))).T
        centers = centers.astype(np.float32)

        spread = float(np.mean(span))
        h = np.full(N, max(spread / max(np.sqrt(N), 1.0) * h_scale, 1e-3),
                    dtype=np.float32)
        Wg = rng.uniform(-0.3, 0.3, (d_out, N)).astype(np.float32)
        Wa = (rng.uniform(-0.3, 0.3, (d_out, N)).astype(np.float32)
              if dual_channel else np.zeros((d_out, N), dtype=np.float32))

        layer = make_layer(substrate, centers, h, Wg, Wa,
                           use_pu=(norm_mode == "pu"), norm_mode=norm_mode,
                           cell=cell, seed=seed)
        layers.append(layer)

        # cascade: next layer sees this layer's actual output (unnormalised)
        zg, za, og, oa = layer_activations(layer, h_in, substrate, "none", None)
        h_in = (og + oa).astype(np.float32)
        dim_in = d_out

    model = SourceStack(layers, norm_mode=norm_mode, dual_channel=dual_channel,
                        substrate=substrate, cell=cell)
    if norm_mode == "fixed":
        model.scales = calibrate_fixed_scales(model.layers, X_ref, substrate)
        for lay, sc in zip(model.layers, model.scales):
            if substrate == "diffusion":
                lay.fixed_scale = sc
    NoiseConfig.set_level(_prev_level)
    return model


def effective_budget(arch_key, X_ref, arms, budget, norm_mode="none",
                     sweeps=2.0, substrate="circuit", mode="equal_probes"):
    """
    The measurement protocol, returned as {arm: probes per epoch}.

    TWO PROTOCOLS, and they answer different questions:

    mode="equal_probes"  (default) every arm probes the SAME number of
        coordinates per epoch, so the analog measurement cost per epoch is
        identical by construction.  The consequence is that an arm with more
        parameters covers a smaller fraction of its own space per epoch -- so
        this protocol systematically favours the SMALLER arm.  In a contracting
        RBF architecture the geometry arm always has more parameters than the
        readout arm (geometry costs D+1 per particle, a scalar readout costs 1),
        so equal_probes is biased against geometry by construction.

    mode="equal_sweeps"  every arm completes exactly ONE full sweep of its own
        parameters per epoch.  Coverage is then identical, and the cost differs
        (the bigger arm costs more measurements).  This isolates the question
        "is geometry a worse parameterisation?" from "does geometry simply have
        more parameters to learn at a fixed measurement rate?".

    Reporting both is the honest way to state the result: an arm that wins under
    equal_probes and loses under equal_sweeps is more efficient per measurement,
    not a better representation.

    budget=None -> AUTO: 2 full sweeps per epoch of the architecture's LARGEST
    arm, rounded to 100 (equal_probes only).  A single global number does not
    work across architectures: a contracting architecture has a tiny readout
    arm, so a budget that suits it starves the matrix-dominated architecture and
    nothing learns at all.

    Returns (budgets, {arm: n_params}).
    """
    probe = build_model(arch_key, X_ref, seed=0, norm_mode=norm_mode,
                        substrate=substrate)
    sizes = {a: int(probe.index_set(a).size) for a in arms}

    if mode == "equal_sweeps":
        return {a: max(sizes[a], 1) for a in arms}, sizes

    if mode != "equal_probes":
        raise ValueError(f"unknown budget mode {mode!r}")

    if budget is None:
        biggest = max(sizes.values()) if sizes else 1
        b = int(round(sweeps * biggest / 100.0) * 100)
        b = max(b, 100)
    else:
        b = int(budget)
    return {a: b for a in arms}, sizes


def layer_activations(layer, X, substrate="circuit", norm_mode="none", scales=None):
    """
    Substrate-agnostic (z_g, z_a, out_g, out_a) with the requested normalisation
    and NO cascade, computed on the ideal (undamaged) path.

    Used by the cascading initialisation and by the frozen-scale calibration,
    both of which need the *un*normalised activations and therefore cannot call
    layer.forward() -- a layer whose norm_mode is "fixed" has no scale yet on
    the first pass.
    """
    if substrate == "circuit":
        r = forward_ex(layer, X, norm_mode, scales)
        return r["z_g"], r["z_a"], r["out_g"], r["out_a"]
    if substrate == "diffusion":
        from substrate_diffusion import KERNELS
        Xf = np.asarray(X, dtype=np.float64)
        zg = layer._activations(Xf, KERNELS[0])[0]
        za = layer._activations(Xf, KERNELS[1])[0]
        outs = []
        for ci, (z, W) in enumerate(zip((zg, za), (layer.W_gauss, layer.W_act))):
            raw = W @ z.T                                   # (V, B) = sum_j W_ij z_j
            if norm_mode == "pu":
                raw = raw / np.maximum(z.sum(axis=1)[None, :], 1e-30)
            elif norm_mode == "fixed":
                if scales is None:
                    raise ValueError("norm_mode='fixed' requires scales")
                key = "g" if ci == 0 else "a"
                raw = raw / max(float(scales[key]), 1e-30)
            outs.append(raw.T)
        return zg, za, outs[0], outs[1]
    raise ValueError(f"unknown substrate {substrate!r}")


def calibrate_fixed_scales(layers, X_ref, substrate="circuit"):
    """
    Freeze a per-layer, per-channel constant normalization for norm_mode="fixed":
        z_hat = z / S,  S = mean_over_batch(sum_j z_j) computed ONCE at init.

    Rationale: PU keeps activations bounded but makes every particle's
    activation depend on every other particle (z_i / sum_j z_j). A frozen S
    keeps the same dynamic range while remaining strictly non-coupling -- on
    either substrate.
    """
    scales = []
    h = np.asarray(X_ref, dtype=np.float32)
    for layer in layers:
        zg, za, og, oa = layer_activations(layer, h, substrate, "none", None)
        sg = max(float(np.mean(np.sum(zg, axis=1))), 1e-6)
        sa = max(float(np.mean(np.sum(za, axis=1))), 1e-6)
        scales.append({"g": sg, "a": sa})
        h = (og + oa).astype(np.float32)
    return scales


# --------------------------------------------------------------
#  Masked per-parameter finite-difference trainer
# --------------------------------------------------------------

def solve_readout_closed_form(model, X, y, lam=1e-3, substrate="circuit",
                              rows=(0,), measured_features=False, cond_log=None,
                              guard=True, reject_log=None, guard_floor=0.0,
                              guard_holdout=4):
    """
    Set every particle's SOURCE STRENGTH so the superposition of sources best
    matches the residual field.

    THIS IS A CALIBRATION, NOT A NUMERICAL SHORTCUT.  On this network W_ij IS
    the excitation strength of particle j as seen by output channel i: the
    readout matrix is not a mixing operator, it is a list of per-source
    amplitudes.  On the diffusion substrate the injected pulse energy is exactly
    q_ij = W_ij * g_j, so solving for W is solving for the pulse energies.
    Physically: drive the sources at unit strength, read the field at every
    training probe, and solve a linear system for the amplitudes that minimise
    the residual.  A device does this once, offline.

    Procedure -- FORWARD STAGEWISE, deliberately not iterative:
        layer 1 strengths <- ridge fit to the target
        layer 2 strengths <- ridge fit to what layer 1 left behind
        ...
    Revisiting an earlier layer after a later one has changed its input makes
    the recursion oscillate and diverge (measured: R^2 fell from -0.8 to -20
    after 10 greedy revisits on a random geometry).  One forward pass is
    monotone, because each ridge fit can only reduce the residual it was handed.
    It is also what a physical calibration does: set layer 1's amplitudes, then
    measure what is left and set layer 2's.

    Only the requested rows are written.  Row 0 is the scalar readout that feeds
    the prediction; rows >= 1 are next-layer features and stay frozen, which
    matters only for architectures with n_vars > 1 (the matrix-free ones have
    n_vars = 1, so there the calibration sets the entire readout).

    COST ACCOUNTING, reported rather than hidden:
      * physical route : N_k field measurements per training sample per layer,
                         i.e. n_train * sum_k N_k measurement-equivalents, ONCE
      * digital twin   : zero physical measurements, since the twin supplies the
                         feature matrix exactly
    Either way it is a ONE-TIME cost, whereas FD/SPSA spends measurements EVERY
    epoch.  That asymmetry is what the "hybrid" arm exists to expose.

    Returns the number of strength rows written.

    measured_features : if True the feature matrix handed to the solve is first
        passed through the device READOUT CHAIN (sensor noise, drift, clipping,
        quantisation).  That is what actually happens when the calibration is
        performed on hardware instead of on a digital twin, and it is the
        assumption the headline result rests on -- so it is switchable rather
        than assumed.
    cond_log : optional list; the 2-norm condition number of each layer's ridge
        matrix is appended, so a noise-induced failure can be attributed to
        ill-conditioning instead of guessed at.
    guard : if True (default), the calibration is DEPLOYED ONLY IF IT HELPS.
        The stagewise solution is scored on a held-out slice of the training data
        against the strengths that were already in place, and rejected if it does
        not beat them.  Without this, an ill-conditioned feature matrix makes the
        solve overfit catastrophically and calibrating becomes WORSE than not
        calibrating (measured: R^2 -0.63 on a 10D task with 8 pure-noise dims,
        against ~0 for every search arm -- section 6.5).  The guard costs one
        forward pass on the held-out slice per calibration.  It must be OFF
        during ridge selection, which needs to RANK candidates, including bad ones.
    reject_log : optional list; (R^2_new - R^2_old) is appended when a
        calibration is rejected, so rejections are visible rather than silent.

    RIDGE SCALE.  `lam` is RELATIVE to the Gram matrix scale: the applied ridge
    is lam * mean(diag(Z^T Z)) per layer.  An absolute ridge cannot transfer
    between tasks -- the same 0.1 is negligible on one feature matrix and
    overwhelming on another -- and the previous absolute candidate range
    (1e-4 .. 1e-1) was too narrow to regularise a near-degenerate feature matrix.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X)

    if isinstance(lam, str) and lam == "auto":
        lam = _choose_ridge_lam(model, X, y, substrate=substrate, rows=rows,
                                measured_features=measured_features)

    if not guard:
        return _stagewise_solve(model, X, y, lam, substrate, rows,
                                measured_features, cond_log)

    # ---- do-no-harm gate: SYMMETRIC, OUT-OF-SAMPLE ON BOTH SIDES ------------
    # The guard slice is never used to fit strengths -- not by the candidate and
    # not by whatever is already deployed -- so the two scores it compares are
    # honest.  An earlier version scored a fit made on ALL the training samples
    # against a slice drawn from those same samples; it reported +0.45 while the
    # true test R^2 was -0.93, i.e. it was blind exactly where it was needed.  A
    # later version fixed that side but still scored the DEPLOYED strengths on
    # data they had been fitted on, which biased the gate toward rejecting and
    # cost real margin (on `steps`/F2 it nearly flipped a win into a loss).
    #
    # Cost: the calibration fits on (1 - 1/holdout) of the training data.  The
    # geometry is NOT affected -- it is fitted by finite differences on the full
    # set, because the gate validates the STRENGTHS, not the geometry.
    n = X.shape[0]
    val = np.zeros(n, dtype=bool)
    val[::max(int(guard_holdout), 2)] = True
    tr = ~val
    if val.sum() < 2 or tr.sum() < 8:
        return _stagewise_solve(model, X, y, lam, substrate, rows,
                                measured_features, cond_log)

    before = [np.array(l.W_gauss, copy=True) for l in model.layers]
    n_written = _stagewise_solve(model, X[tr], y[tr], lam, substrate, rows,
                                 measured_features, cond_log)
    fitted = [np.array(l.W_gauss, copy=True) for l in model.layers]
    r_new = float(r2(y[val], model.forward(X[val])))
    for l, w in zip(model.layers, before):
        l.W_gauss = np.asarray(w, dtype=l.W_gauss.dtype)
    r_base = float(r2(y[val], model.forward(X[val])))     # also never fitted on val
    keep = (np.isfinite(r_new)
            and (not np.isfinite(r_base) or r_new > r_base)
            and r_new > float(guard_floor))
    if not keep:
        if reject_log is not None:
            reject_log.append(float(min(r_new - r_base,
                                        r_new - float(guard_floor))))
        return 0
    for l, w in zip(model.layers, fitted):                # deploy the same fit
        l.W_gauss = np.asarray(w, dtype=l.W_gauss.dtype)
    return n_written


def _stagewise_solve(model, X, y, lam, substrate, rows, measured_features,
                     cond_log=None):
    """
    One forward-stagewise calibration pass.  Separated from the guard so the guard
    can fit on a sub-slice and refit on everything without duplicating the loop.

    FORWARD STAGEWISE, deliberately not iterative:
        layer 1 strengths <- ridge fit to the target
        layer 2 strengths <- ridge fit to what layer 1 left behind
    Revisiting an earlier layer after a later one has changed its input makes the
    recursion oscillate and diverge (measured: R^2 fell from -0.8 to -20 after 10
    greedy revisits on a random geometry).  One forward pass is monotone, and it
    is also what a physical calibration does.

    `lam` is RELATIVE to the Gram scale: the applied ridge is
    lam * mean(diag(Z^T Z)) per layer, so one lambda transfers between tasks.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    residual = y.copy()
    n_written = 0
    for k, layer in enumerate(model.layers):
        trace = model.activations(X)                    # inputs follow layers < k
        Z = np.asarray(trace[k]["z_g"], dtype=np.float64)      # (B, N_k)
        if measured_features:
            Z = _measure_features(layer, Z, substrate)
        G = Z.T @ Z
        scale = float(np.mean(np.diag(G))) if G.size else 1.0
        A = G + float(lam) * max(scale, 1e-12) * np.eye(Z.shape[1])
        if cond_log is not None:
            try:
                cond_log.append(float(np.linalg.cond(A)))
            except np.linalg.LinAlgError:
                cond_log.append(float("inf"))
        for row in rows:
            if row >= layer.n_vars:
                continue
            w = np.linalg.solve(A, Z.T @ residual)
            if substrate == "circuit":
                layer.W_gauss[row, :] = w.astype(layer.W_gauss.dtype)
            else:
                layer.W_gauss[row, :] = w
            n_written += 1
            if row == 0:
                # what this layer now contributes is exactly the fitted field,
                # so that is what leaves the residual for the next layer
                residual = residual - Z @ w
    return n_written


def _heldout_r2(model, X, y, holdout=4):
    """
    R^2 on a held-out slice of the training data (every `holdout`-th sample).
    Kept for diagnostics.  NOTE the gate no longer uses it: scoring the deployed
    strengths this way is optimistic when they were fitted on the same samples,
    and that asymmetry biased the gate toward rejecting.  See
    solve_readout_closed_form.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = X.shape[0]
    val = np.zeros(n, dtype=bool)
    val[::int(holdout)] = True
    if val.sum() < 2:
        return float("nan")
    return float(r2(y[val], model.forward(X[val])))


def _choose_ridge_lam(model, X, y, substrate="circuit", rows=(0,),
                      candidates=None, holdout=4,
                      measured_features=False):
    """
    Pick the RELATIVE ridge strength on a HELD-OUT slice of the training data,
    not on the test set.  Every 4th sample is held out, each candidate is run as a
    full stagewise calibration, and the model's strength rows are restored
    afterwards so the caller's model is untouched.

    The candidate range spans seven decades (1e-6 .. 1e0 relative), because the
    previous absolute range of four candidates (1e-4 .. 1e-1) could not regularise
    a near-degenerate feature matrix at all: on the 10D task of section 6.5 the
    best of those four was still far too weak, the solve overfit, and calibrating
    ended up worse than not calibrating.

    The guard is disabled inside this function on purpose: it exists to RANK
    candidates, and a guard would reject the weak ones before they could be
    compared.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = X.shape[0]
    val = np.zeros(n, dtype=bool)
    val[::int(holdout)] = True
    tr = ~val
    if tr.sum() < 8 or val.sum() < 2:
        return 1e-3

    if candidates is None:
        candidates = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0)

    saved = [l.W_gauss.copy() for l in model.layers]
    best_lam, best_r2 = 1e-3, -np.inf
    for lam in candidates:
        for l, w in zip(model.layers, saved):
            l.W_gauss = w.copy()
        solve_readout_closed_form(model, X[tr], y[tr], lam=lam,
                                  substrate=substrate, rows=rows,
                                  measured_features=measured_features,
                                  guard=False)
        score = r2(y[val], model.forward(X[val]))
        if np.isfinite(score) and score > best_r2:
            best_lam, best_r2 = float(lam), float(score)
    for l, w in zip(model.layers, saved):
        l.W_gauss = w
    return best_lam


def _measure_features(layer, Z, substrate):
    """
    Put the feature matrix through the device's readout chain.

    On the diffusion cell the activations are fields read by sensors, so a
    calibration performed on hardware sees noise, drift, clipping and
    quantisation on Z.  The circuit substrate has no separate sensor stage --
    its non-idealities sit inside the forward (mismatch, drift) and are already
    reflected in Z -- so it is returned unchanged, and the robustness study uses
    the diffusion cell.
    """
    if substrate != "diffusion":
        return Z
    from substrate_diffusion.components import readout as _ro
    sub = layer.substrate
    seed = int(layer._rng.integers(1 << 30))
    return _ro(Z, bits=sub.sensor_bits, full_scale=sub.sensor_range,
               noise_rel=sub.sensor_noise, drift=sub.drift,
               rng=np.random.default_rng(seed))


def readout_calibration_cost(model, n_train):
    """
    One-time physical measurement count for a full readout calibration:
    N_k per training sample per layer (the naive per-source readout).  Reported
    next to the per-epoch FD cost so the two are never confused.
    """
    return int(n_train * sum(int(l.n_particles) for l in model.layers))


def train_model(model, X_tr, y_tr, X_val=None, y_val=None, *, arm="full",
                epochs=120, lr=3e-3, eps_rel=1e-4, budget=None,
                update_mode="sequential", seed=0, verbose=False,
                log_every=25, probe_batch=96, closed_form=None,
                closed_form_every=1, closed_form_lam=1e-3,
                substrate="circuit", eval_every=0, eval_X=None, eval_y=None,
                calib_physical=False, closed_form_measured=False,
                closed_form_model=None, closed_form_guard=True,
                closed_form_guard_floor=0.0, closed_form_guard_holdout=4):
    """
    Per-parameter central-difference training restricted to `arm`.

    arm="hybrid" probes the GEOMETRY only (same physical knobs, same budget) but
    additionally re-calibrates the source strengths in closed form, so the
    comparison "hybrid vs geometry" isolates exactly what the calibration buys
    at an identical physical measurement cost.

    Design decisions that matter for a fair ablation:

    * ADAM on the FD gradients.  Plain SGD on zeroth-order gradients makes the
      three arms incomparable, because centers (scale ~2), widths (~1) and
      weights (~0.3) receive wildly different step sizes from one global lr.
      Adam normalises the step per coordinate and removes that confound.
      (t advances once per epoch for both update modes, so the bias-correction
      schedule is identical between them.)

    * eps is RELATIVE: eps_j = eps_rel * (1 + |theta_j|).

    * `probe_batch` limits the loss measurement to a random mini-batch per
      epoch (fixed for both +/- perturbations of a coordinate, so the finite
      difference is not biased by batch noise).  Set to 0 for full batch.
      This is both 3-4x cheaper and closer to what an analog trainer can do.

    * `budget` is the EXACT number of coordinates probed per epoch (2 forward
      passes each).  This is the analog-measurement budget, and it is the same
      for every arm, so the comparison is cost-matched.  When budget exceeds
      the arm size the arm cycles through independent permutations (see
      effective_budget).  budget=None means one full sweep per epoch.

    * update_mode:
        "sequential" = Gauss-Seidel: probe + update one coordinate at a time,
                       later probes in the same epoch see earlier updates.
        "parallel"   = Jacobi: probe every selected coordinate from the frozen
                       epoch-start state, then apply all updates at once.
                       This is what K parallel hardware channels actually do.
      Both spend exactly 2*|selected| forward passes per epoch.

    Returns a history dict.
    """
    if update_mode not in ("sequential", "parallel"):
        raise ValueError(f"unknown update_mode: {update_mode}")

    np.random.seed(seed)
    rng = np.random.default_rng(seed + 99991)

    idx_all = model.index_set(arm)
    if idx_all.size == 0:
        raise ValueError(f"arm '{arm}' selects no parameters")

    # "hybrid" probes the geometry but calibrates the strengths in closed form.
    # "hybrid_once" does that calibration only at epoch 0 (the classical hybrid
    # RBF scheme); the recalibration schedule turned out to be architecture
    # dependent, so both are offered as separate arms rather than tuned.
    if closed_form is None:
        closed_form = arm in ("hybrid", "hybrid_once")
    if arm == "hybrid_once":
        closed_form_every = 10 ** 9

    p = model.get_params()
    eps_vec = eps_rel * (1.0 + np.abs(p))
    m = np.zeros_like(p)
    v = np.zeros_like(p)

    hist = {"train_mse": [], "val_r2": [], "fwd": 0, "budget": budget,
            "n_selected": int(idx_all.size), "epochs": epochs,
            "calib_cost": (readout_calibration_cost(model, X_tr.shape[0])
                           if closed_form else 0),
            "closed_form": bool(closed_form),
            # A SCALING CURVE, not a point: `measurements` is the physical
            # measurement count consumed so far (probe forward passes plus any
            # calibration that is actually done on the device), and `curve`
            # records (measurements, R^2) along the way.  One run therefore
            # yields the whole R^2(measurements) curve, which is the quantity a
            # hardware designer needs; a point estimate at one budget is not.
            "measurements": 0, "curve": [], "calib_physical_total": 0,
            "cond": [], "n_calibrations": 0, "twin_calibrations": 0,
            "calib_rejected": 0, "calib_reject_gap": [],
            "x_unit": "field measurements (probe forward passes"
                      + (" + physical calibrations)" if calib_physical
                         else "; calibrations assumed twin-supplied)")}

    y_tr_flat = np.asarray(y_tr).ravel()
    n_train = X_tr.shape[0]

    def full_loss(vec):
        model.set_params(vec)
        pred = model.forward(X_tr)
        return float(np.mean((pred.ravel() - y_tr_flat) ** 2))

    for ep in range(epochs):
        # Closed-form source-strength calibration: a ONE-TIME offline operation
        # (see solve_readout_closed_form), refreshed here so that the strengths
        # always match the current particle layout.  It costs no per-epoch
        # physical measurements; its one-time cost is reported separately.
        if closed_form and (ep % max(int(closed_form_every), 1) == 0):
            # Calibrate on `closed_form_model` when given, then DEPLOY the
            # resulting strengths onto the device.  This models the real
            # situation: the strengths are computed from a digital twin's
            # description of the cell and then written into the cell, so any
            # disagreement between twin and device lands on the strengths.
            # closed_form_model=None means the twin IS the device (zero twin error).
            tgt = closed_form_model if closed_form_model is not None else model
            n_ret = solve_readout_closed_form(
                tgt, X_tr, y_tr, lam=closed_form_lam, substrate=substrate,
                measured_features=closed_form_measured, cond_log=hist["cond"],
                guard=closed_form_guard, reject_log=hist["calib_reject_gap"],
                guard_floor=closed_form_guard_floor,
                guard_holdout=closed_form_guard_holdout)
            hist["n_calibrations"] += 1
            if n_ret == 0:
                # the do-no-harm guard rejected it: nothing was written to `tgt`,
                # so nothing may be deployed to the device either
                hist["calib_rejected"] += 1
            elif tgt is not model:
                for ld, lt in zip(model.layers, tgt.layers):
                    ld.W_gauss[...] = np.asarray(lt.W_gauss,
                                                 dtype=ld.W_gauss.dtype)
                hist["twin_calibrations"] += 1
            if calib_physical:
                # the calibration is done ON the device: n_train * sum_k N_k
                # field measurements, every time it is refreshed
                c = readout_calibration_cost(model, X_tr.shape[0])
                hist["measurements"] += c
                hist["calib_physical_total"] += c
            p = model.get_params()
            eps_vec = eps_rel * (1.0 + np.abs(p))

        # Exact probe budget per epoch, split into chunks that never exceed the
        # arm size.  Each chunk is an independent random permutation of the
        # arm's parameters, so every arm spends exactly 2*budget forward passes
        # per epoch while every arm is measured as often as the protocol says.
        chunks = []
        remaining = int(budget) if budget else idx_all.size
        while remaining > 0:
            take = min(remaining, idx_all.size)
            chunks.append(rng.choice(idx_all, size=take, replace=False))
            remaining -= take

        # Mini-batch probe set, fixed within the epoch so that the +eps and
        # -eps measurements of one coordinate see the SAME data (otherwise the
        # finite difference is biased by batch noise).  This is also what a real
        # analog trainer does: it cannot hold the whole training set in the loop.
        if probe_batch and 0 < int(probe_batch) < n_train:
            sel = rng.choice(n_train, size=int(probe_batch), replace=False)
            Xb = X_tr[sel]
            yb = y_tr_flat[sel]
        else:
            Xb, yb = X_tr, y_tr_flat

        def loss_of(vec, _Xb=Xb, _yb=yb):
            model.set_params(vec)
            pred = model.forward(_Xb)
            return float(np.mean((pred.ravel() - _yb) ** 2))

        t = ep + 1

        if update_mode == "sequential":
            for chunk in chunks:
                for j in chunk:
                    p0 = p[j]
                    e = eps_vec[j]
                    p[j] = p0 + e
                    Lp = loss_of(p)
                    p[j] = p0 - e
                    Lm = loss_of(p)
                    p[j] = p0
                    g = (Lp - Lm) / (2.0 * e)
                    hist["fwd"] += 2

                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g
                    mh = m[j] / (1.0 - 0.9 ** t)
                    vh = v[j] / (1.0 - 0.999 ** t)
                    p[j] -= lr * mh / (np.sqrt(vh) + 1e-8)
        else:
            g = np.zeros_like(p)
            for chunk in chunks:
                for j in chunk:
                    p0 = p[j]
                    e = eps_vec[j]
                    p[j] = p0 + e
                    Lp = loss_of(p)
                    p[j] = p0 - e
                    Lm = loss_of(p)
                    p[j] = p0
                    g[j] = (Lp - Lm) / (2.0 * e)
                    hist["fwd"] += 2

                m[chunk] = 0.9 * m[chunk] + 0.1 * g[chunk]
                v[chunk] = 0.999 * v[chunk] + 0.001 * g[chunk] ** 2
                mh = m[chunk] / (1.0 - 0.9 ** t)
                vh = v[chunk] / (1.0 - 0.999 ** t)
                p[chunk] -= lr * mh / (np.sqrt(vh) + 1e-8)

        model.set_params(p)
        hist["measurements"] = hist["fwd"] + hist["calib_physical_total"]
        hist["train_mse"].append(full_loss(p))
        if X_val is not None:
            hist["val_r2"].append(r2(y_val, model.forward(X_val)))
        if eval_every and eval_X is not None and (ep + 1) % int(eval_every) == 0:
            hist["curve"].append((int(hist["measurements"]),
                                  float(r2(eval_y, model.forward(eval_X)))))

        if verbose and (ep + 1) % log_every == 0:
            print(f"    ep {ep+1:4d}/{epochs} | train MSE={hist['train_mse'][-1]:.5f}"
                  + (f" | val R2={hist['val_r2'][-1]:.4f}" if X_val is not None else "")
                  + f" | fwd={hist['fwd']}")

    model.set_params(p)
    return hist


# --------------------------------------------------------------
#  Metrics
# --------------------------------------------------------------

def r2(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = float(np.mean((y_pred - y_true) ** 2))
    var = float(np.var(y_true))
    return 1.0 - mse / max(var, 1e-12)


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_pred).ravel() - np.asarray(y_true).ravel()) ** 2))


def evaluate(model, X, y, reps=1):
    """Mean/std of R^2 over `reps` stochastic noise realizations."""
    vals = [r2(y, model.forward(X)) for _ in range(max(int(reps), 1))]
    return float(np.mean(vals)), float(np.std(vals))


# --------------------------------------------------------------
#  Data
# --------------------------------------------------------------

def make_regression_data(kind="4d", n_train=256, n_test=128, seed=42, noise=0.10):
    """
    Toy regression / two-class tasks, all SELF-CONTAINED (no downloads), all with
    the same 256/128 split and train-set standardisation, so a task sweep compares
    like with like and reproduces offline.

       2d     y = sin(3x1) + cos(2x2) + 0.5 x1 x2          D=2, smooth, 2 relevant
       4d     y = sin(3x1) + cos(2x2) + 0.5 x3 x4          D=4, the task used
                                                             everywhere else
       10d     y = sin(3x1) + cos(2x2)                      D=10, 8 PURE-NOISE dims
        moons  two interleaving half circles, target +-1     D=2, a classification
                                                             boundary, i.e. a
                                                             different TASK TYPE
       steps  y = sign(sin(3x1)) + 0.5 cos(2x2)             D=4, DISCONTINUOUS:
                                                             a Gaussian basis
                                                             cannot fit it well
       hetero y = sin(3x1) + cos(2x2), noise ~ |x1|         D=4, HETEROSCEDASTIC:
                                                             least squares is the
                                                             wrong likelihood

    `steps` and `hetero` are deliberate adversaries of the headline rule.  The rule
    is "solve the readout in closed form", which is an ordinary least-squares fit;
    it should look worst when the target is discontinuous (no linear readout can
    fit it) and when the noise is not uniform (plain least squares is misspecified).
    Including them is what makes a positive sweep informative instead of circular.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test

    if kind == "shift":
        # DISTRIBUTION SHIFT, and the point of the task: the gate validates on
        # held-out data FROM THE TRAINING DISTRIBUTION, so it structurally cannot
        # see a shift.  Train support is [-2,2], test support is [-3,3]; the
        # functional form is identical, so the ONLY thing that changes is where
        # the samples live.  A calibration that validates well in-distribution can
        # still be the wrong thing to deploy out of distribution, and a Gaussian
        # basis decays to zero outside its support -- which a large calibrated W
        # multiplies into something large, while a small searched W does not.
        D = 4
        Xtr = rng.uniform(-2.0, 2.0, (n_train, D)).astype(np.float32)
        Xte = rng.uniform(-3.0, 3.0, (n_test, D)).astype(np.float32)
        f = lambda Z: (np.sin(3.0 * Z[:, 0]) + np.cos(2.0 * Z[:, 1])
                       + 0.5 * Z[:, 2] * Z[:, 3])
        y_tr = (f(Xtr) + rng.normal(0.0, noise, n_train)).astype(np.float32)
        y_te = (f(Xte) + rng.normal(0.0, noise, n_test)).astype(np.float32)
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        return (((Xtr - mu) / sd).astype(np.float32), y_tr,
                ((Xte - mu) / sd).astype(np.float32), y_te)

    if kind == "noise":
        # PURE NOISE: y is independent of x, so no method can beat R^2 = 0 and the
        # only way to score below zero is to FIT the training noise.  This is the
        # designed counterexample to "the gate guarantees the calibrated arm is
        # never worse than not calibrating": the gate validates on held-out data
        # drawn from the SAME noise, and fitting noise raises that score, so the
        # gate cannot tell signal from noise.  The search arms move by small steps
        # and stay near the mean predictor; a closed-form solve has no small steps.
        D = 4
        Xn = rng.uniform(-2.0, 2.0, (n, D)).astype(np.float32)
        yn = rng.normal(0.0, 1.0, n).astype(np.float32)
        Xtr_n, Xte_n = Xn[:n_train], Xn[n_train:]
        mu_n = Xtr_n.mean(axis=0, keepdims=True)
        sd_n = Xtr_n.std(axis=0, keepdims=True) + 1e-6
        return (((Xtr_n - mu_n) / sd_n).astype(np.float32), yn[:n_train],
                ((Xte_n - mu_n) / sd_n).astype(np.float32), yn[n_train:])

    if kind == "moons":
        # two interleaving half circles, encoded as +-1 (least-squares
        # classification, so the same trainer and the same R^2 apply)
        m = n // 2
        t1 = rng.uniform(0.0, np.pi, m)
        t2 = rng.uniform(0.0, np.pi, n - m)
        X = np.vstack([
            np.stack([np.cos(t1), np.sin(t1)], axis=1),
            np.stack([1.0 - np.cos(t2), 1.0 - np.sin(t2) - 0.5], axis=1),
        ]).astype(np.float32)
        y = np.concatenate([np.ones(m), -np.ones(n - m)]).astype(np.float32)
        X = (X + rng.normal(0.0, 0.05, X.shape)).astype(np.float32)
        y = (y + rng.normal(0.0, noise, n)).astype(np.float32)
    else:
        D = {"2d": 2, "4d": 4, "10d": 10, "steps": 4, "hetero": 4}.get(kind)
        if D is None:
            raise ValueError(f"unknown task kind: {kind}")
        X = rng.uniform(-2.0, 2.0, (n, D)).astype(np.float32)
        base = np.sin(3.0 * X[:, 0]) + np.cos(2.0 * X[:, 1])
        if kind == "steps":
            # discontinuous: no smooth basis fits this, and the linear readout
            # must approximate a square wave
            y = np.sign(np.sin(3.0 * X[:, 0])) + 0.5 * np.cos(2.0 * X[:, 1])
        elif kind == "hetero":
            # noise scale grows with |x1|, so uniform-weight least squares is
            # misspecified and the closed-form solve is not the right estimator
            y = base + rng.normal(0.0, 1.0, n) * (noise + 0.5 * np.abs(X[:, 0]))
        elif kind == "10d":
            y = base
        else:
            y = base + 0.5 * X[:, 2 % D] * X[:, 3 % D]
        y = (y + (rng.normal(0.0, noise, n) if kind != "hetero" else 0.0)
             ).astype(np.float32)

    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr = ((X_tr - mu) / sd).astype(np.float32)
    X_te = ((X_te - mu) / sd).astype(np.float32)
    return X_tr, y_tr, X_te, y_te


def task_description(kind):
    """One-line description used in the task-sweep report."""
    return {
        "2d": "D=2 smooth, 2 relevant dims",
        "4d": "D=4 smooth + product term (the task used elsewhere)",
        "10d": "D=10, 8 pure-noise dims",
        "moons": "D=2 two-class boundary, targets +-1",
        "steps": "D=4 DISCONTINUOUS target (adversarial to a linear readout)",
        "hetero": "D=4 HETEROSCEDASTIC noise (adversarial to least squares)",
        "shift": "D=4 DISTRIBUTION SHIFT: train on [-2,2], test on [-3,3]",
        "noise": "D=4 PURE NOISE: y independent of x (no signal to learn)",
    }.get(kind, kind)


# --------------------------------------------------------------
#  Model cache (ablation C reuses A's trained models)
# --------------------------------------------------------------

def cache_path(arch, arm, norm_mode, seed, task, epochs, budget, substrate="circuit"):
    """
    Cache key includes the substrate, the epoch count and the EFFECTIVE probe
    budget, so a --quick smoke run, or a run on a different medium, can never be
    silently reused as a real result by abl_03.
    """
    return os.path.join(
        CACHE_DIR,
        f"{task}_{substrate}_{arch}_{arm}_{norm_mode}_s{seed}"
        f"_e{int(epochs)}_b{int(budget)}.npz",
    )


def save_model(model, arch, arm, norm_mode, seed, task, epochs, budget,
               substrate="circuit", extra=None):
    payload = {"params": model.get_params()}
    if extra:
        payload.update({k: np.asarray(v) for k, v in extra.items()})
    np.savez(cache_path(arch, arm, norm_mode, seed, task, epochs, budget,
                        substrate), **payload)


def load_model(arch, arm, norm_mode, seed, task, epochs, budget, X_ref,
               substrate="circuit"):
    path = cache_path(arch, arm, norm_mode, seed, task, epochs, budget, substrate)
    if not os.path.exists(path):
        return None
    model = build_model(arch, X_ref, seed=seed, norm_mode=norm_mode,
                        substrate=substrate)
    with np.load(path) as z:
        model.set_params(z["params"])
    return model


# --------------------------------------------------------------
#  Reporting helpers
# --------------------------------------------------------------

class Report:
    """Tiny tee-to-file reporter, same pattern as experiments/legacy_suite.py."""

    def __init__(self, filename):
        self.path = os.path.join(OUT_DIR, filename)
        self.lines = []

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(str(s))

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        # Return a path RELATIVE to the repository root, not the absolute one that
        # `self.path` holds.  Scripts print this value into their reports (and into the
        # `Report:` line of the artifact itself), and an absolute path puts the
        # maintainer's drive layout into every published report.  It is not a secret --
        # no user name is in it -- but it is information about the machine rather than
        # about the measurement, and it makes the reports non-portable.  Reports written
        # before this change still carry the absolute form; they were left alone,
        # because they are evidence and editing evidence to tidy it is the wrong trade.
        try:
            return os.path.relpath(self.path, os.path.dirname(OUT_DIR)).replace(os.sep, "/")
        except ValueError:                      # different drives: no relative path exists
            return self.path


def header(title, subtitle=""):
    lines = ["=" * 66, f"  {title}"]
    if subtitle:
        lines.append(f"  {subtitle}")
    lines += [f"  {time.strftime('%Y-%m-%d %H:%M:%S')}", "=" * 66]
    return lines


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                               - np.asarray(b, dtype=np.float64))))


def curve_at(curve, x):
    """
    R^2 recorded at the largest measurement count <= x (nan if the run had not
    reached x).  Used to turn trajectories into a milestone table.
    """
    val = float("nan")
    for m, v in curve:
        if m <= x:
            val = v
        else:
            break
    return val


def merge_curves(curves, grid):
    """
    Sample several runs' trajectories on a common measurement grid.

    Returns (mean, std, n) arrays.  Runs that had not reached a grid point
    contribute nan and are excluded from the mean, so the number of contributing
    seeds is reported alongside.
    """
    xs = np.asarray(grid, dtype=np.float64)
    vals = np.full((len(curves), xs.size), np.nan)
    for i, c in enumerate(curves):
        for j, x in enumerate(xs):
            vals[i, j] = curve_at(c, x)
    mu = np.full(xs.size, np.nan)
    sd = np.full(xs.size, np.nan)
    n = np.zeros(xs.size, dtype=int)
    for j in range(xs.size):
        col = vals[:, j]
        col = col[np.isfinite(col)]
        n[j] = col.size
        if col.size:
            mu[j] = col.mean()
            sd[j] = col.std()
    return mu, sd, n


def mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std())


def nanmean_safe(vals):
    """np.nanmean that does not warn when every entry is NaN."""
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0 or np.all(np.isnan(a)):
        return float("nan")
    return float(np.nanmean(a))


def learning_ok(r2_values, floor=0.05):
    """
    True when at least one model actually fit the task, i.e. when a verdict
    about architecture is meaningful.  Reported alongside every verdict so a
    later reader cannot mistake "the models did not learn" for a finding.
    """
    vals = np.asarray(r2_values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    return bool(vals.size > 0 and float(vals.max()) >= floor)


INCONCLUSIVE = ("INCONCLUSIVE: no configuration reached R2 >= {floor:.2f} "
                "(best {best:+.4f}). The models did not learn the task, so the "
                "comparison below carries no information -- raise --budget / "
                "--epochs and rerun.")


def inconclusive_text(r2_values, floor=0.05):
    vals = np.asarray(r2_values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    best = float(vals.max()) if vals.size else float("nan")
    return INCONCLUSIVE.format(floor=floor, best=best)


def try_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        return plt
    except Exception as exc:            # pragma: no cover
        print(f"  [warn] plotting disabled: {exc}")
        return None
