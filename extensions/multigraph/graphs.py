"""
Multi-graph readout: k single-output graphs per stage, combined into m channels.

See DESIGN.md for the specification and, more importantly, for its measured
constraints.  The short version:

  * a base layer is ONE graph of nodes carrying V sets of scaling coefficients
    (the crossbar rows);
  * here a stage is k GRAPHS, each with a single scalar reading, and the m
    outputs are formed by COMBINING those readings;
  * the combination is a design variable, but only the LINEAR ones keep the
    readout linear in the strengths -- and that linearity is what makes the base
    version's closed-form calibration possible at all (DESIGN section 4).

This module deliberately reuses the base version's `layer_forward` for every
graph reading.  That is what makes `ext_01`'s degenerate test meaningful: with
k = m, shared geometry and an identity combination, the extension must reproduce
the base forward BIT FOR BIT, and it can only do that if it runs the same code.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
#  Cache / arch-key isolation.  DESIGN section 9.1: abl_common's model cache is
#  SHARED, and abl_03 loads from it to reuse ablation A's trained models.  An
#  extension that trains with a base arch key or task string would silently
#  overwrite a base cache entry and corrupt that reuse.  Every extension run must
#  therefore use an arch key starting with this prefix and a task string that no
#  base run uses.
# ---------------------------------------------------------------------------
EXT_ARCH_PREFIX = "K"
BASE_ARCH_KEYS = ("G", "M", "W", "F", "F2")
BASE_TASK_STRINGS = ("2d", "4d", "10d", "moons", "steps", "hetero", "shift", "noise")


def ext_arch_key(k, shared=True, code="L"):
    """
    Arch key for an extension model, e.g. ``K4S-L`` = 4 graphs, shared geometry,
    linear combination.  Never collides with a base key, and says what it is.
    """
    return f"{EXT_ARCH_PREFIX}{int(k)}{'S' if shared else 'D'}-{code}"


def check_isolation(arch_key, task):
    """Raise if an extension would collide with a base cache key."""
    if arch_key in BASE_ARCH_KEYS:
        raise ValueError(f"arch key {arch_key!r} is a BASE arch; use ext_arch_key()")
    if task in BASE_TASK_STRINGS:
        raise ValueError(f"task {task!r} is a BASE task string; prefix it, e.g. 'ext4d'")


# ---------------------------------------------------------------------------
#  Combination operators
# ---------------------------------------------------------------------------
class Linear:
    """
    ``channel = C @ readings`` with C an (m, k) matrix.

    One implementation covers three rows of DESIGN section 4's table:

      * C = I (k == m)  -> "selection / concatenation", the degenerate case that
        must reproduce the base version exactly;
      * a learned or calibrated C -> "linear weights", i.e. W = C . W_graphs,
        rank <= min(m, k);
      * a frozen C (Hadamard, random +-1) -> "fixed code".

    All three keep the readout LINEAR in the strengths, so the base version's
    closed-form calibration operator remains applicable.
    """

    def __init__(self, C):
        self.C = np.asarray(C, dtype=np.float64)
        if self.C.ndim != 2:
            raise ValueError("C must be (m, k)")
        # An identity combination is PASSED THROUGH, not multiplied.  Two reasons:
        # multiplying by I would be a no-op mathematically but not necessarily
        # bit-wise (dtype round-trips), and the degenerate case is required to be
        # BIT-identical to the base version, not merely close (DESIGN 6, C4).
        self.is_identity = bool(
            self.C.shape[0] == self.C.shape[1]
            and np.array_equal(self.C, np.eye(self.C.shape[0])))

    @property
    def m(self):
        return self.C.shape[0]

    @property
    def k(self):
        return self.C.shape[1]

    def apply(self, readings):
        """
        readings: (B, k) -> (B, m).

        The non-identity path computes in float64 and returns the readings'
        original dtype.  Bit-identity is therefore only claimed for the identity
        case; a genuine combination is compared with a float32 tolerance instead
        (see ext_01's T2).
        """
        readings = np.asarray(readings)
        if self.is_identity:
            return readings
        return (readings.astype(np.float64) @ self.C.T).astype(readings.dtype)

    @classmethod
    def identity(cls, k, m=None):
        m = k if m is None else m
        if m != k:
            raise ValueError("identity combination requires m == k")
        return cls(np.eye(k))

    @classmethod
    def fixed_code(cls, k, m, seed=0, kind="sign"):
        rng = np.random.default_rng(seed)
        if kind == "sign":
            C = rng.choice([-1.0, 1.0], size=(m, k))
        elif kind == "hadamard":
            from itertools import product
            if k & (k - 1):
                raise ValueError("hadamard needs a power-of-two k")
            C = np.array([[(-1.0) ** bin(i & j).count("1") for j in range(k)]
                          for i in range(k)])
            C = C[:m]
        else:
            raise ValueError(f"unknown fixed code {kind!r}")
        return cls(C)


class Nonlinear:
    """
    Placeholder for the rows of DESIGN section 4 whose last column is "no".

    NOT implemented, on purpose.  A combination that is nonlinear in the readings
    (products, gating, attention) makes fitting the strengths a nonlinear problem
    again, which forfeits the base version's closed-form calibration and the
    15-30x measurement advantage that comes with it.  It must also happen AFTER
    sensing, because a linear medium cannot multiply two fields (DESIGN 4.1).
    Raising here is better than silently shipping a combination that quietly
    breaks the calibration operator.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "nonlinear combination is a deliberate non-feature for now: it "
            "forfeits closed-form calibration (DESIGN section 4). Implement it "
            "only together with a replacement for that operator."
        )


# ---------------------------------------------------------------------------
#  Graph and stack
# ---------------------------------------------------------------------------
class Graph:
    """
    One graph: a base layer with a SINGLE output row (n_vars == 1).

    Reusing the base layer class is the point -- it is what lets the degenerate
    configuration be bit-identical rather than merely close.  Geometry sharing is
    expressed by construction: the graphs of a stage are handed the SAME
    ``centers`` and ``h`` arrays (see :func:`graphs_from_base`), and DESIGN 4.1
    explains why sharing must include ``h`` and not just the centres.
    """

    def __init__(self, layer, name=""):
        if getattr(layer, "n_vars", None) != 1:
            raise ValueError(
                f"a Graph needs n_vars == 1 (one scalar reading); got "
                f"{getattr(layer, 'n_vars', None)}")
        self.layer = layer
        self.name = name

    @property
    def n_particles(self):
        return int(self.layer.n_particles)

    def read(self, X, norm_mode="none", scale=None, substrate="circuit"):
        """
        The graph's scalar reading at X, as (B, 1).

        NO dtype coercion: the base forward computes `out_g + out_a` and passes
        the result straight to the next layer without casting, and the diffusion
        path works in float64 while the circuit path works in float32.  Casting
        here would make the degenerate case merely close instead of identical.
        """
        from abl_common import layer_forward
        r = layer_forward(self.layer, X, norm_mode, scale, substrate=substrate)
        return r["out_g"] + r["out_a"]


class MultiGraphStack:
    """
    k single-output graphs per stage, combined into m channels, wire-direct
    between stages, with the base version's scalar residual readout:

        pred = sum_s out_s[:, 0]

    With k = m, shared geometry and C = I per stage this is EXACTLY the base
    ``SourceStack``; ``ext_01`` asserts that bit for bit.
    """

    def __init__(self, stages, combines, norm_mode="none", scales=None,
                 substrate="circuit", metrics=None):
        if len(stages) != len(combines):
            raise ValueError("one combination per stage")
        self.stages = [list(s) for s in stages]
        self.combines = list(combines)
        self.norm_mode = norm_mode
        self.scales = list(scales) if scales is not None else [None] * len(stages)
        self.substrate = substrate
        self.metrics = metrics or {}
        for s, comb in zip(self.stages, self.combines):
            if len(s) != comb.k:
                raise ValueError(
                    f"stage has {len(s)} graphs but C expects {comb.k}")

    # -- shape of the thing ------------------------------------------------
    @property
    def n_stages(self):
        return len(self.stages)

    @property
    def k(self):
        return [len(s) for s in self.stages]

    @property
    def m(self):
        return [c.m for c in self.combines]

    def geometry_parameters(self):
        """centers + h, counted ONCE per stage even though k graphs share them."""
        g = 0
        for s in self.stages:
            lay = s[0].layer
            g += int(lay.dim) * int(lay.n_particles) + int(lay.n_particles)
        return g

    def readout_parameters(self):
        """k*N scale coefficients + the combination entries (DESIGN section 5)."""
        r = 0
        for s, comb in zip(self.stages, self.combines):
            r += sum(g.n_particles for g in s)
            # An IDENTITY combination stores no coefficients: with k == m the
            # channel assignment is structural.  Counting its entries anyway made
            # the degenerate case report MORE parameters than the base model it is
            # required to equal -- and DESIGN section 5 is a parameter-count
            # argument, so the accounting has to agree with the anchor.
            if not getattr(comb, "is_identity", False):
                r += int(np.asarray(comb.C).size)
        return r

    def parameter_report(self):
        return dict(stages=self.n_stages, k=self.k, m=self.m,
                    geometry=self.geometry_parameters(),
                    readout=self.readout_parameters(),
                    total=self.geometry_parameters() + self.readout_parameters())

    # -- forward -----------------------------------------------------------
    def forward(self, X):
        """
        Same structure as the base SourceStack.forward, with the reading step
        replaced by "read k graphs, then combine":

            for each stage:  readings (B,k) -> C -> (B,m)
                             pred += out[:, 0]
                             h = out            (wire-direct, full vector)
        """
        h = X
        pred = None
        for si, (stage, comb) in enumerate(zip(self.stages, self.combines)):
            cols = [g.read(h, self.norm_mode, self.scales[si], self.substrate)
                    for g in stage]
            readings = np.concatenate(cols, axis=1)          # (B, k)
            out = comb.apply(readings)                       # (B, m), same dtype
            pred = out[:, :1].copy() if pred is None else pred + out[:, :1]
            h = out
        return pred

    def stage_outputs(self, X):
        """Per-stage (B, m) outputs, for tests that compare stage by stage."""
        h = X
        outs = []
        for si, (stage, comb) in enumerate(zip(self.stages, self.combines)):
            readings = np.concatenate(
                [g.read(h, self.norm_mode, self.scales[si], self.substrate)
                 for g in stage], axis=1)
            out = comb.apply(readings)
            outs.append(out)
            h = out
        return outs


# ---------------------------------------------------------------------------
#  Construction from a base model -- the degenerate direction
# ---------------------------------------------------------------------------
def graphs_from_base(base_model, substrate="circuit", norm_mode=None):
    """
    Split a base ``SourceStack`` into k = m single-output graphs PER STAGE, all
    sharing that stage's geometry.

    This is the direction that must be EXACT, and it is the only construction
    ``ext_01`` trusts: it starts from the base model's own parameters rather than
    generating its own, so any disagreement is a real disagreement.

    Stage s of the base has a layer with V = n_vars output rows.  This returns V
    graphs for it, graph i holding row i of ``W_gauss`` and ``W_act`` and the
    SAME centres/widths.  With ``Linear.identity(V)`` the stage output is then
    ``out_g + out_a`` -- the base's own expression.
    """
    from abl_common import make_layer

    norm_mode = base_model.norm_mode if norm_mode is None else norm_mode
    cell = getattr(base_model, "cell", None)
    stages, combines = [], []
    for s, layer in enumerate(base_model.layers):
        V, N = int(layer.n_vars), int(layer.n_particles)
        centers = np.array(layer.centers, copy=True)
        h = np.array(layer.h, copy=True)
        graphs = []
        for i in range(V):
            wg = np.array(layer.W_gauss[i:i + 1, :], copy=True)
            wa = np.array(layer.W_act[i:i + 1, :], copy=True)
            sub_layer = make_layer(substrate, centers, h, wg, wa,
                                   use_pu=(norm_mode == "pu"), norm_mode=norm_mode,
                                   cell=cell, seed=0,
                                   fixed_scale=(base_model.scales[s]
                                                if base_model.scales[s] is not None
                                                else None))
            graphs.append(Graph(sub_layer, name=f"s{s}g{i}"))
        stages.append(graphs)
        combines.append(Linear.identity(V))
    scales = list(base_model.scales) if base_model.scales else None
    return MultiGraphStack(stages, combines, norm_mode=norm_mode, scales=scales,
                           substrate=substrate)
