"""
V3 HiMAL-FFA-ND: Layer-Wise Dual-Network Training with Prototype-Guided Loss

Adapts v1 HiMALTrainer (forward_training/himal.py) for D-dimensional particles,
with prototype-guided FFA-style loss and Otsu thresholding.

Supports both regression and classification tasks:
  - Regression: residual targets, proto-weighted predict
  - Classification: physical proto attraction + confidence-gated ensemble

All loss operations map to analog circuit components:
  - subtract → differential_pair
  - square → gilbert_square
  - sum → kirchhoff_sum
  - mean → kirchhoff + current_mirror
No softmax, log, or exp in the classification path.

Key components:
  - Layer uses PhysicalRBFLayerND (centers (D, N) instead of (xc, yc))
  - Inter-layer: wire-direct (no mean projection)
  - Residual training: L2/L3 learn (y - accumulated) instead of raw y
  - Data partitioning: identical to v1 (median_x, gradient)
  - Dual-network (A/B) + transition alignment loss preserved
  - Proto agreement loss ensures A/B prototype semantics align
  - All gradients via per-parameter FD — zero autograd

Classification mode:
  - Physical loss: mean(||h - proto[label]||²) + lambda * proto_reg
  - Proto agreement: MSE(protos_A, protos_B) — A/B semantic alignment
  - Confidence-gated voting: confidence = 1/(1+min_d²)
  - Ensemble across layers: (0.5, 0.3, 0.2) weights

New vs original HiMALND:
  - L_proto: prototype attraction loss (soft assignment, neg-log-likelihood)
  - L_reg: prototype geometry regularization (pairwise dist matches y-dist)
  - L_otsu: Otsu soft separation (sigmoid-weighted goodness)
  - L_ratio: easy/hard sample ratio balance
  - tau warmup: adaptive Otsu threshold
  - Predict: prototype-weighted instead of direct h->y mapping
"""

import numpy as np
from typing import Tuple, List, Optional
from copy import deepcopy

from simulator.layer_nd import PhysicalRBFLayerND
from physical_simulator import NoiseConfig


def _zero_pad_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two arrays with zero-padding for mismatched last dimensions.
    Returns (batch, max(da, db))."""
    da, db = a.shape[-1], b.shape[-1]
    if da > db:
        out = a.copy()
        out[:, :db] += b
        return out
    elif db > da:
        out = b.copy()
        out[:, :da] += a
        return out
    else:
        return a + b


class HiMALND:
    """
    HiMAL-FFA-ND: layer-wise dual-network trainer with prototype-guided loss.

    Training flow:
      1. Data partition: split into A-zone, B-zone, T-zone (transition)
      2. For each layer:
         a. Forward: hA = LayerA_i(inA), hB = LayerB_i(inB)
         b. Loss: L = L_proto_A + L_proto_B + gamma * L_align
            where L_proto = L_proto + lambda_reg * L_reg + beta * (L_otsu + 0.1 * L_ratio)
         c. Update LayerA_i params and LayerB_i params via FD
         d. Freeze; compute residual target for next layer
      3. Inference: prototype-weighted prediction with residual accumulation

    Parameters
    ----------
    layer_sizes : tuple of int
        Particles per layer [n1, n2, n3]
    output_dims : tuple of int
        Output dimension per layer [d1, d2, d3]
    input_dim : int
        Input dimensionality D
    init_range : tuple of float
        Range for particle initialization
    alignment_weight : float
        Alignment loss weight gamma
    n_protos : int
        Number of prototypes per network
    temperature : float
        Softmax temperature for prototype assignment
    lambda_reg : float
        Prototype geometry regularization weight
    beta : float
        Otsu+ratio loss weight
    tau : float
        Otsu threshold initial value
    tau_warmup_epochs : int
        Epochs to warm up tau from start to final value
    """

    def __init__(
        self,
        layer_sizes: Tuple[int, int, int] = (20, 20, 20),
        output_dims: Tuple[int, int, int] = (4, 2, 2),
        input_dim: int = 4,
        init_range: Tuple[float, float] = (-3.0, 3.0),
        alignment_weight: float = 0.3,
        n_protos: int = 8,
        temperature: float = 0.1,
        lambda_reg: float = 0.1,
        beta: float = 0.5,
        tau: float = 0.5,
        tau_warmup_epochs: int = 50,
        use_spsa: bool = True,
        spsa_avg: int = 1,
        spsa_clip: float = 1.0,
        spsa_decay_steps: float = 100.0,
        task: str = "regression",
        n_classes: int = 2,
        residual_mode: str = "proto",
        seed: int = 0,
    ):
        if task not in ("regression", "classification"):
            raise ValueError(f"Unknown task: {task}")
        # How the residual target for layers 2..3 is formed in regression:
        #   "proto"  -- original y minus the accumulated PROTO-WEIGHTED
        #               prediction (what the predict path sums, so train and
        #               predict agree)
        #   "direct" -- original y minus the accumulated RAW scalar output
        #               h_m[:, 0] of the previous layers.  THEORY.md 3.2 and 7.3
        #               reported this as ~5x better with the sign in its favour;
        #               re-measured under control (extension FINDINGS.md 12.17)
        #               the sign REVERSES -- direct's depth gain is -0.0284 against
        #               proto's +0.0069 -- because "direct" fits layers 2-3 to the
        #               residual of the RAW output while the inference path sums
        #               the PROTO predictions, so the deeper layers are trained
        #               against a residual the inference never reconstructs.
        #               "proto" is therefore the default because the alternative is
        #               MEASURED HARMFUL, not merely unverified.
        if residual_mode not in ("proto", "direct"):
            raise ValueError(f"Unknown residual_mode: {residual_mode}")
        self.residual_mode = residual_mode

        # Reproducibility.  EVERY random draw this trainer makes -- the layer
        # initialisation in `_init_layer` and the SPSA perturbation in
        # `_spsa_step` -- comes from the GLOBAL `np.random` state, and `train`
        # seeds it once at entry.  Before this, two calls with identical
        # arguments produced DIFFERENT networks depending on what had run
        # before, so every number published from this class was a single draw
        # from an unseeded process (extension FINDINGS.md 12.17).
        self.seed = int(seed)

        self.layer_sizes = layer_sizes
        self.output_dims = output_dims
        self.input_dim = input_dim
        self.init_range = init_range
        self.alignment_weight = alignment_weight
        self.task = task
        self.n_classes = n_classes

        # FFA parameters
        self.n_protos = n_protos
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.tau_init = tau
        self.tau = tau
        self.tau_warmup_epochs = tau_warmup_epochs

        # SPSA parameters
        self.use_spsa = use_spsa
        self.spsa_avg = spsa_avg
        self._spsa_clip = spsa_clip
        self._spsa_decay_steps = spsa_decay_steps

        # Stored prototypes for inference (per layer) — regression
        self.proto_h_A: List[np.ndarray] = []
        self.proto_y_A: List[np.ndarray] = []
        self.proto_h_B: List[np.ndarray] = []
        self.proto_y_B: List[np.ndarray] = []

        # Stored class prototypes for inference (per layer) — classification
        self.class_protos_A: List[np.ndarray] = []
        self.class_protos_B: List[np.ndarray] = []

        # Internal: current proto refs for loss closure access during training
        self._cur_protos_A: Optional[np.ndarray] = None
        self._cur_protos_B: Optional[np.ndarray] = None

        # Stored split info for routing inference
        self._split_value: Optional[float] = None
        self._split_method: Optional[str] = None

    # ── Data partitioning ──

    def split_data(
        self, X: np.ndarray, y: np.ndarray,
        method: str = "median_x", transition_width: float = 0.2,
    ) -> dict:
        """Split data into A-zone, B-zone, T-zone."""
        if method in ("median_x", "median_y"):
            dim = 0 if method == "median_x" else 1
            split_feature = X[:, min(dim, X.shape[1] - 1)]
        elif method == "gradient":
            y_flat = y.flatten() if y.ndim == 2 else y
            if np.issubdtype(y_flat.dtype, np.integer) or self.task == "classification":
                # Fall back to median_x for integer/classification labels
                split_feature = X[:, min(0, X.shape[1] - 1)]
            else:
                split_feature = np.abs(np.gradient(y_flat))
        else:
            raise ValueError(f"Unknown split method: {method}")

        median = np.median(split_feature)
        data_range = np.ptp(split_feature)
        width = transition_width * max(data_range, 1e-8)

        mask_A = split_feature <= median
        mask_B = split_feature > median
        trans_mask = (split_feature >= median - width) & (split_feature <= median + width)

        return {
            "XA": X[mask_A].copy(), "yA": y[mask_A].copy(),
            "XB": X[mask_B].copy(), "yB": y[mask_B].copy(),
            "XT": X[trans_mask].copy(), "yT": y[trans_mask].copy(),
            "split_value": float(median),
            "masks": {"A": mask_A, "B": mask_B, "T": trans_mask},
        }

    # ── Layer initialization ──

    def _init_layer(self, n_particles: int, input_dim: int, n_vars: int) -> PhysicalRBFLayerND:
        """Create a layer with random particles of given input dimension."""
        lo, hi = self.init_range

        xc = np.random.uniform(lo, hi, (input_dim, n_particles)).astype(np.float32)
        dsize = hi - lo

        # Classification: tighter RBF kernels so deeper layers can resolve
        # subtle differences in wire-direct output (which clusters around ~±0.3).
        # Regression: wider kernels for smooth function approximation.
        if self.task == "classification":
            h_scale = 3.0  # sharper: dsize/(3*sqrt(n))
        else:
            h_scale = 1.0  # standard
        h = (dsize / (h_scale * np.sqrt(max(n_particles, 1)))) * np.ones(
            n_particles, dtype=np.float32)
        W_gauss = np.random.uniform(-0.3, 0.3, (n_vars, n_particles)).astype(np.float32)

        # For regression: disable PU (flat gradient under normalization)
        # and zero W_act (single-channel avoids dual-channel competition).
        if self.task == "regression":
            use_pu = False
            W_act = np.zeros_like(W_gauss)
        else:
            use_pu = True
            W_act = np.random.uniform(-0.3, 0.3, (n_vars, n_particles)).astype(np.float32)

        return PhysicalRBFLayerND(
            centers=xc, h=h, W_gauss=W_gauss, W_act=W_act, use_pu=use_pu,
        )

    def _init_random_layer(self, n_particles: int, input_dim: int, n_vars: int):
        """Initialize a random layer pair (one copy each for A, B)."""
        return self._init_layer(n_particles, input_dim, n_vars), \
               self._init_layer(n_particles, input_dim, n_vars)

    # ── Physical forward ──

    def _forward_layer(self, layer: PhysicalRBFLayerND, X: np.ndarray) -> np.ndarray:
        """Physical layer forward: returns fused (gauss + act) output."""
        out_g, out_a = layer.forward(X.astype(np.float32))
        return out_g + out_a  # (batch, n_vars)

    @staticmethod
    def _match_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Pad/truncate arr to match target last-dimension."""
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        dh, dt = arr.shape[-1], target_shape[-1]
        if dh > dt:
            return arr[:, :dt]
        elif dh < dt:
            pad = np.zeros((arr.shape[0], dt - dh), dtype=arr.dtype)
            return np.concatenate([arr, pad], axis=1)
        return arr

    # ── Prototype helpers ──

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _create_prototypes(
        self, y: np.ndarray, h: np.ndarray, n_protos: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create prototypes by binning samples along sorted y-values.

        Parameters
        ----------
        y : (N,) or (N, d)
            Target values (flattened to 1D for sorting)
        h : (N, h_dim)
            Hidden representations from current layer
        n_protos : int
            Number of prototypes

        Returns
        -------
        y_protos : (n_protos, d) — mean y per prototype
        h_protos : (n_protos, h_dim) — mean h per prototype
        assign : (N,) — prototype index [0, n_protos-1] for each sample
        """
        y_flat = y.ravel() if y.ndim > 1 else y
        N = len(y_flat)
        n_protos = min(n_protos, N)  # clamp to sample count

        # Sort by y, partition into equal bins
        idx = np.argsort(y_flat)
        bin_size = N // n_protos
        assign = np.zeros(N, dtype=np.int32)

        # Keep original multi-dim y if available
        y_2d = y.reshape(N, -1) if y.ndim == 1 else y
        out_dim = y_2d.shape[1]
        y_protos = np.zeros((n_protos, out_dim), dtype=np.float32)
        h_protos = np.zeros((n_protos, h.shape[-1]), dtype=np.float32)

        for k in range(n_protos):
            start = k * bin_size
            end = (k + 1) * bin_size if k < n_protos - 1 else N
            bin_idx = idx[start:end]
            assign[bin_idx] = k
            y_protos[k] = y_2d[bin_idx].mean(axis=0)
            h_protos[k] = h[bin_idx].mean(axis=0)

        return y_protos, h_protos, assign

    def _compute_single_proto_loss(
        self, h: np.ndarray, y: np.ndarray, n_protos: int,
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute FFA prototype-based loss for a single network.

        L = L_proto + lambda_reg * L_reg + beta * (L_otsu + 0.1 * L_ratio)

        Returns (total, L_proto, L_reg, L_otsu, L_ratio)
        """
        y_protos, h_protos, assign = self._create_prototypes(y, h, n_protos)
        N = h.shape[0]
        K = len(y_protos)
        if K < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # ── L_proto: prototype attraction ──
        # dist_all[i,k] = ||h_i - proto_h_k||
        dist_all = np.linalg.norm(h[:, None, :] - h_protos[None, :, :], axis=2)  # (N, K)
        probs = self._softmax(-dist_all / self.temperature, axis=1)  # (N, K)
        loss_proto = float(-np.log(probs[range(N), assign] + 1e-8).mean())

        # ── L_reg: prototype geometry regularization ──
        pairwise_dist = np.linalg.norm(
            h_protos[:, None, :] - h_protos[None, :, :], axis=2,
        )  # (K, K)
        ideal_dist = np.abs(y_protos[:, None, :] - y_protos[None, :, :])
        if ideal_dist.ndim > 2:
            ideal_dist = ideal_dist.sum(axis=-1)  # sum over output dims
        loss_reg = float(np.mean((pairwise_dist - ideal_dist) ** 2))

        # ── L_otsu: Otsu soft separation ──
        target_dist = dist_all[range(N), assign]  # (N,) distance to assigned proto
        goodness = -target_dist  # closer = better
        otsu_temp = self.temperature * 2.0
        w_pos = self._sigmoid((self.tau - target_dist) / otsu_temp)  # easy samples (close)
        w_neg = 1.0 - w_pos  # hard samples (far)
        loss_otsu = float(-(w_pos * goodness).mean() + (w_neg * goodness).mean())

        # ── L_ratio: easy/hard ratio balance ──
        pos_frac = float(np.mean(w_pos))
        loss_ratio = float((pos_frac - 0.5) ** 2)

        total = float(
            loss_proto +
            self.lambda_reg * loss_reg +
            self.beta * (loss_otsu + 0.1 * loss_ratio)
        )
        return total, loss_proto, loss_reg, loss_otsu, loss_ratio

    def _proto_predict(
        self, h: np.ndarray, y_protos: np.ndarray, h_protos: np.ndarray,
    ) -> np.ndarray:
        """
        Predict y from h using prototype-weighted interpolation.

        y_hat_i = sum_k w_ik * y_proto_k
        where w_ik = softmax(-||h_i - h_proto_k|| / T)

        Parameters
        ----------
        h : (N, h_dim)
        y_protos : (K, d)
        h_protos : (K, h_dim)

        Returns
        -------
        pred : (N, d)
        """
        K = len(y_protos)
        dist = np.linalg.norm(h[:, None, :] - h_protos[None, :, :], axis=2)  # (N, K)
        weights = self._softmax(-dist / self.temperature, axis=1)  # (N, K)
        return weights @ y_protos  # (N, d)

    # ── Loss ──

    def _compute_pair_loss(
        self,
        layer_A: PhysicalRBFLayerND, layer_B: PhysicalRBFLayerND,
        data: dict, gamma: float,
    ) -> float:
        """
        Pure physical MSE loss (v1-style, fully analog-trainable).

        L = mean((hA - yA)^2) + mean((hB - yB)^2) + gamma * mean((hTA - hTB)^2)

        All operations (subtraction, squaring, mean) are physically realizable
        via differential pairs and Gilbert squarers.
        """
        XA, yA = data["XA"], data["yA"]
        XB, yB = data["XB"], data["yB"]
        XT = data["XT"]

        hA = self._forward_layer(layer_A, XA)
        hB = self._forward_layer(layer_B, XB)

        if hA.shape[0] != yA.shape[0]:
            hA = hA.T
        if hB.shape[0] != yB.shape[0]:
            hB = hB.T

        # MSE: compare only the first min(h_dim, y_dim) columns.
        # Supports multi-D output (wire-direct) with 1D regression targets.
        match_dim = min(hA.shape[1], yA.shape[1])
        fit_A = float(np.mean((hA[:, :match_dim] - yA[:, :match_dim]) ** 2))
        fit_B = float(np.mean((hB[:, :match_dim] - yB[:, :match_dim]) ** 2))

        if len(XT) > 0:
            hTA = self._forward_layer(layer_A, XT)
            hTB = self._forward_layer(layer_B, XT)
            align = float(np.mean((hTA.ravel() - hTB.ravel()) ** 2))
        else:
            align = 0.0

        self._last_losses = {"fit_A": fit_A, "fit_B": fit_B, "align": align}
        return float(fit_A + fit_B + gamma * align)

    # ── Classification loss ──

    def _init_class_prototypes(self, n_vars: int) -> np.ndarray:
        """
        Initialize K=n_classes prototype vectors for classification.

        Strategy: opposite poles on first axis for binary,
        one-hot-like spread for multi-class.
        """
        K = self.n_classes
        assert K <= n_vars, (
            f"n_classes ({K}) cannot exceed output dimension n_vars ({n_vars}). "
            f"Increase output_dims or reduce n_classes."
        )
        if K == 2:
            protos = np.zeros((K, n_vars), dtype=np.float32)
            protos[0, 0] = 0.5
            protos[1, 0] = -0.5
        else:
            protos = np.zeros((K, n_vars), dtype=np.float32)
            for k in range(K):
                dim = k % n_vars
                sign = 1.0 if k < n_vars else -1.0
                protos[k, dim] = sign * 0.5
        return protos

    def _class_proto_regularization(self, protos: np.ndarray) -> float:
        """
        Push different-class prototypes apart. All ops physically realizable.

        L = mean((d²_pairwise - ideal_sq)²)
        where d² = squared distance (no sqrt), ideal_sq = 1 for i≠j, 0 for i=j.

        Signal chain: subtract → square → sum → square → mean
        Components: diff_pair → gilbert_square → kirchhoff_sum →
                    gilbert_square → kirchhoff_sum / current_mirror
        """
        K = protos.shape[0]
        diff = protos[:, np.newaxis, :] - protos[np.newaxis, :, :]  # (K, K, d)
        d2 = np.sum(diff * diff, axis=2)  # squared pairwise distances
        ideal = 1.0 - np.eye(K, dtype=np.float32)  # 1 for i≠j, 0 for i=j
        return float(np.mean((d2 - ideal) ** 2))

    def _compute_single_class_loss(
        self, h: np.ndarray, y_labels: np.ndarray, protos: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Physical classification loss — all ops map to analog circuits.

        L = mean(||h - proto[label]||²) + lambda_reg * L_reg

        Signal chain per sample:
          1. diff_pair: h - proto[k] for each k
          2. gilbert_square: (h - proto[k])²
          3. kirchhoff_sum: sum over dimensions → d²_k
          4. select d² for label class only (demux / switch)
          5. mean over batch (kirchhoff + current_mirror)

        No softmax, no log, no exp — purely quadratic.

        Returns (total, attract, reg).
        """
        K = protos.shape[0]
        if h.ndim == 1:
            h = h.reshape(-1, 1)

        # Squared L2 distance to each prototype → (N, K)
        diff = h[:, np.newaxis, :] - protos[np.newaxis, :, :]  # subtract
        d2 = np.sum(diff * diff, axis=2) + 1e-12               # square + sum

        y_int = y_labels.astype(np.int64).ravel()
        d2_correct = d2[np.arange(len(y_int)), y_int]  # (N,)
        loss_attract = float(np.mean(d2_correct))

        reg = self._class_proto_regularization(protos)

        return loss_attract + self.lambda_reg * reg, loss_attract, reg

    def _compute_pair_class_loss(
        self,
        layer_A: PhysicalRBFLayerND, layer_B: PhysicalRBFLayerND,
        protos_A: np.ndarray, protos_B: np.ndarray,
        data: dict, gamma: float,
    ) -> float:
        """
        Dual-network classification loss — fully physical.

        L = L_class_A + L_class_B + gamma * L_align + 0.1 * L_agree

        L_class:  mean(||h - proto[label]||²) + lambda_reg * L_reg
        L_align:  MSE(hTA, hTB) on transition zone (diff_pair + gilbert)
        L_agree:  MSE(protos_A, protos_B) — ensures A/B proto semantics match
        """
        XA, yA = data["XA"], data["yA"]
        XB, yB = data["XB"], data["yB"]
        XT = data["XT"]

        hA = self._forward_layer(layer_A, XA)
        hB = self._forward_layer(layer_B, XB)

        if hA.shape[0] != len(yA):
            hA = hA.T
        if hB.shape[0] != len(yB):
            hB = hB.T

        L_A, attract_A, reg_A = self._compute_single_class_loss(hA, yA, protos_A)
        L_B, attract_B, reg_B = self._compute_single_class_loss(hB, yB, protos_B)

        # Proto agreement: A and B prototypes should represent the same class semantics
        proto_agree = float(np.mean((protos_A - protos_B) ** 2))

        # Alignment on transition zone
        if len(XT) > 0:
            hTA = self._forward_layer(layer_A, XT)
            hTB = self._forward_layer(layer_B, XT)
            if hTA.shape[0] != len(XT):
                hTA = hTA.T
            if hTB.shape[0] != len(XT):
                hTB = hTB.T
            align = float(np.mean((hTA.ravel() - hTB.ravel()) ** 2))
        else:
            align = 0.0

        self._last_losses = {"attract_A": attract_A, "reg_A": reg_A,
                             "attract_B": attract_B, "reg_B": reg_B,
                             "align": align, "proto_agree": proto_agree}
        return float(L_A + L_B + gamma * align + 0.1 * proto_agree)

    # ── Single-network class prediction ──

    def _class_predict_single(
        self, h: np.ndarray, protos: np.ndarray,
    ) -> np.ndarray:
        """Predict class from h via nearest-prototype."""
        if h.ndim == 1:
            h = h.reshape(-1, 1)
        dist = np.sqrt(np.sum(
            (h[:, np.newaxis, :] - protos[np.newaxis, :, :]) ** 2, axis=2,
        ) + 1e-12)
        return dist.argmin(axis=1)

    # ── Train one layer ──

    def _spsa_step(
        self, layer_A: PhysicalRBFLayerND, layer_B: PhysicalRBFLayerND,
        net_name: str, data: dict, gamma: float, eps: float,
        epoch: int = 0,
    ) -> None:
        """
        Single SPSA update for one network.

        Uses Rademacher perturbation Δ, evaluates L(θ±εΔ),
        computes gradient estimate g ≈ (L₊ - L₋) / (2ε) · Δ.

        Features:
          - Multi-averaging: spsa_avg independent perturbations averaged
          - Gradient clipping: max |grad| = spsa_clip (default 1.0)
          - LR decay: lr_k = lr / (1 + epoch / decay_steps)

        Computational cost: 2·spsa_avg forward passes.
        """
        layer = layer_A if net_name == "A" else layer_B
        vec = layer.get_params_vector()

        # ── LR decay: lr_k = lr / (1 + k/decay) ──
        decay_steps = self._spsa_decay_steps
        lr = self._spsa_lr / (1.0 + epoch / max(decay_steps, 1))

        # ── Multi-average gradient estimate ──
        grad = np.zeros_like(vec)
        for _ in range(self.spsa_avg):
            delta = np.random.choice([-1.0, 1.0], size=len(vec)).astype(np.float32)

            vec_plus = vec + eps * delta
            layer.set_params_from_vector(vec_plus)
            L_plus = self._loss_fn(layer_A, layer_B, data, gamma)

            vec_minus = vec - eps * delta
            layer.set_params_from_vector(vec_minus)
            L_minus = self._loss_fn(layer_A, layer_B, data, gamma)

            layer.set_params_from_vector(vec)

            grad += (L_plus - L_minus) / (2.0 * eps) * delta

        grad /= max(self.spsa_avg, 1)

        # ── Gradient clipping ──
        clip_val = getattr(self, '_spsa_clip', 1.0)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > clip_val and grad_norm > 1e-12:
            grad = grad * (clip_val / grad_norm)

        # ── Apply update ──
        vec_new = vec - lr * grad
        layer.set_params_from_vector(vec_new)

    def train_one_layer(
        self, data: dict, n_particles: int, input_dim: int, n_vars: int,
        epochs: int = 200, lr: float = 0.01, eps: float = 1e-4,
        verbose: bool = True,
    ) -> Tuple[PhysicalRBFLayerND, PhysicalRBFLayerND]:
        """
        Train one dual-network layer.

        Regression: MSE loss (SPSA or per-param FD).
        Classification: proto-guided CE loss + separate proto FD updates.

        If use_spsa=True: SPSA for RBF params (2 fwd/epoch/network).
        If use_spsa=False: per-parameter FD (2·n_params fwd/epoch/network).
        """
        layer_A, layer_B = self._init_random_layer(n_particles, input_dim, n_vars)
        gamma = self.alignment_weight
        self._last_losses = {}
        self._spsa_lr = lr

        # ── Loss function dispatch ──
        if self.task == "classification":
            protos_A = self._init_class_prototypes(n_vars)
            protos_B = self._init_class_prototypes(n_vars)

            def make_loss_fn():
                return lambda la, lb, d, g: self._compute_pair_class_loss(
                    la, lb, protos_A, protos_B, d, g)
            self._loss_fn = make_loss_fn()
        else:
            self._loss_fn = self._compute_pair_loss

        for epoch in range(epochs):
            if self.use_spsa:
                self._spsa_step(layer_A, layer_B, "A", data, gamma, eps, epoch=epoch)
                self._spsa_step(layer_A, layer_B, "B", data, gamma, eps, epoch=epoch)
            else:
                for net_name in ["A", "B"]:
                    layer = layer_A if net_name == "A" else layer_B
                    vec = layer.get_params_vector()
                    grad = np.zeros_like(vec)

                    for j in range(len(vec)):
                        vec[j] += eps
                        layer.set_params_from_vector(vec)
                        L_plus = self._loss_fn(layer_A, layer_B, data, gamma)

                        vec[j] -= 2 * eps
                        layer.set_params_from_vector(vec)
                        L_minus = self._loss_fn(layer_A, layer_B, data, gamma)

                        vec[j] += eps
                        grad[j] = (L_plus - L_minus) / (2 * eps)

                    vec = vec - lr * grad
                    layer.set_params_from_vector(vec)

            # ── Classification: also update proto vectors via FD ──
            if self.task == "classification":
                for net_name in ["A", "B"]:
                    layer = layer_A if net_name == "A" else layer_B
                    protos = protos_A if net_name == "A" else protos_B
                    K, d = protos.shape
                    proto_vec = protos.flatten().copy()
                    grad_p = np.zeros(len(proto_vec), dtype=np.float64)

                    for j in range(len(proto_vec)):
                        proto_vec[j] += eps
                        tmp = proto_vec.reshape(K, d)
                        if net_name == "A":
                            L_plus = self._compute_pair_class_loss(
                                layer_A, layer_B, tmp, protos_B, data, gamma)
                        else:
                            L_plus = self._compute_pair_class_loss(
                                layer_A, layer_B, protos_A, tmp, data, gamma)

                        proto_vec[j] -= 2 * eps
                        tmp = proto_vec.reshape(K, d)
                        if net_name == "A":
                            L_minus = self._compute_pair_class_loss(
                                layer_A, layer_B, tmp, protos_B, data, gamma)
                        else:
                            L_minus = self._compute_pair_class_loss(
                                layer_A, layer_B, protos_A, tmp, data, gamma)

                        proto_vec[j] += eps
                        grad_p[j] = (L_plus - L_minus) / (2 * eps)

                    proto_vec_new = proto_vec - lr * grad_p
                    if net_name == "A":
                        protos_A = proto_vec_new.reshape(K, d).astype(np.float32)
                    else:
                        protos_B = proto_vec_new.reshape(K, d).astype(np.float32)

                # Rebuild loss_fn with updated protos
                self._loss_fn = lambda la, lb, d, g: self._compute_pair_class_loss(
                    la, lb, protos_A, protos_B, d, g)

            if verbose and (epoch + 1) % 40 == 0:
                loss = self._loss_fn(layer_A, layer_B, data, gamma)
                d = self._last_losses
                mode = "SPSA" if self.use_spsa else "FD"
                if self.task == "classification":
                    acc_A = (self._class_predict_single(
                        self._forward_layer(layer_A, data["XA"]),
                        protos_A if self.task == "classification" else None,
                    ) == data["yA"].ravel()).mean()
                    print(f"  Layer epoch {epoch+1:3d}/{epochs} [{mode}] | loss={loss:.4f} | "
                          f"acc_A={acc_A:.3f} | "
                          f"att_A={d.get('attract_A',0):.3f} att_B={d.get('attract_B',0):.3f} "
                          f"align={d.get('align',0):.3f} agree={d.get('proto_agree',0):.3f}")
                else:
                    print(f"  Layer epoch {epoch+1:3d}/{epochs} [{mode}] | loss={loss:.6f} | "
                          f"fit_A={d.get('fit_A',0):.4f} fit_B={d.get('fit_B',0):.4f} "
                          f"align={d.get('align',0):.4f}")

        # Store class protos for inference
        if self.task == "classification":
            self._cur_protos_A = protos_A
            self._cur_protos_B = protos_B

        return layer_A, layer_B

    # ── Full training pipeline ──

    def train(
        self, X: np.ndarray, y: np.ndarray,
        split_method: str = "median_x",
        epochs_per_layer: int = 200, lr: float = 0.01,
        eps: float = 1e-4,
        verbose: bool = True,
    ) -> Tuple[List[PhysicalRBFLayerND], List[PhysicalRBFLayerND], dict]:
        """
        Full HiMAL-FFA-ND training: 3 layers, wire-direct.

        Regression: residual targets (L2/L3 learn y - accumulated_pred).
        Classification: same class labels for all layers, proto voting.

        Returns layers_A, layers_B, data_split.
        """
        # ONE seed covers every random draw below: the layer initialisation
        # (`_init_layer`) and the SPSA perturbation (`_spsa_step`) are the only
        # two sources, and both are reached from here.  Two calls with identical
        # arguments and the same `seed` now give identical networks.
        np.random.seed(self.seed)

        data = self.split_data(X, y, method=split_method)
        self._split_value = float(data["split_value"])
        self._split_method = split_method
        layers_A, layers_B = [], []

        # Clear stored prototypes
        self.proto_h_A, self.proto_y_A = [], []
        self.proto_h_B, self.proto_y_B = [], []
        self.class_protos_A, self.class_protos_B = [], []

        cur_XA = data["XA"]
        cur_XB = data["XB"]
        cur_XT = data["XT"]
        cur_yA = data["yA"].copy()
        cur_yB = data["yB"].copy()
        if cur_yA.ndim == 1:
            cur_yA = cur_yA.reshape(-1, 1)
        if cur_yB.ndim == 1:
            cur_yB = cur_yB.reshape(-1, 1)

        # Track current input dimension
        cur_dim = self.input_dim

        # Accumulated predictions for residual computation
        # Each layer learns: residual = y_original - acc_prev
        #   residual_mode="proto"  -> acc is the proto-weighted prediction that
        #                             the predict path also sums (default)
        #   residual_mode="direct" -> acc is the raw scalar output h[:, 0] of
        #                             each layer, i.e. the scheme THEORY.md 7.3
        #                             claims is ~5x better
        acc_proto_A = None  # proto-weighted accumulated prediction on A-zone
        acc_proto_B = None
        acc_direct_A = None  # raw h[:, 0] accumulation, used by "direct"
        acc_direct_B = None
        y_orig_A = data["yA"].copy().reshape(-1, 1) if data["yA"].ndim == 1 else data["yA"].copy()
        y_orig_B = data["yB"].copy().reshape(-1, 1) if data["yB"].ndim == 1 else data["yB"].copy()

        for i in range(3):
            if verbose:
                task_tag = "HiMAL-CLS" if self.task == "classification" else "HiMAL-FFA"
                print(f"\n===== {task_tag} Layer {i+1} ({self.layer_sizes[i]} particles, "
                      f"in={cur_dim}D, out={self.output_dims[i]}D) =====")

            n_p = self.layer_sizes[i]
            n_v = self.output_dims[i]

            # ── Residual targets for layers 2+ ──
            # Layer 1: learns full y.  Layers 2/3: learn y - accumulated_proto_pred.
            # This matches the predict path where proto predictions are summed.
            if self.task == "regression":
                if i == 0:
                    cur_yA = y_orig_A.copy()
                    cur_yB = y_orig_B.copy()
                else:
                    # Residual target = original y - accumulated prediction
                    if self.residual_mode == "direct":
                        cur_yA = y_orig_A - acc_direct_A[:, :y_orig_A.shape[1]]
                        cur_yB = y_orig_B - acc_direct_B[:, :y_orig_B.shape[1]]
                    else:
                        cur_yA = y_orig_A - acc_proto_A[:, :y_orig_A.shape[1]]
                        cur_yB = y_orig_B - acc_proto_B[:, :y_orig_B.shape[1]]
            else:
                cur_yA = data["yA"].copy().reshape(-1, 1) if data["yA"].ndim == 1 else data["yA"].copy()
                cur_yB = data["yB"].copy().reshape(-1, 1) if data["yB"].ndim == 1 else data["yB"].copy()

            layer_data = {
                "XA": cur_XA, "yA": cur_yA,
                "XB": cur_XB, "yB": cur_yB,
                "XT": cur_XT,
            }

            layer_A, layer_B = self.train_one_layer(
                layer_data, n_p, cur_dim, n_v,
                epochs=epochs_per_layer, lr=lr, eps=eps,
                verbose=verbose,
            )

            layers_A.append(layer_A)
            layers_B.append(layer_B)

            # Compute final outputs
            hA = self._forward_layer(layer_A, cur_XA)
            hB = self._forward_layer(layer_B, cur_XB)

            if hA.shape[0] != cur_XA.shape[0]:
                hA = hA.T
            if hB.shape[0] != cur_XB.shape[0]:
                hB = hB.T

            # ── Store prototypes for inference ──
            if self.task == "classification":
                self.class_protos_A.append(
                    self._cur_protos_A.copy() if self._cur_protos_A is not None
                    else self._init_class_prototypes(n_v))
                self.class_protos_B.append(
                    self._cur_protos_B.copy() if self._cur_protos_B is not None
                    else self._init_class_prototypes(n_v))
            else:
                # Proto maps h_i → residual target (or full y for layer 1)
                # At predict time these are summed → recovers original y
                yA_2d = cur_yA.reshape(-1, 1) if cur_yA.ndim == 1 else cur_yA
                yB_2d = cur_yB.reshape(-1, 1) if cur_yB.ndim == 1 else cur_yB
                y_proto_A, h_proto_A, _ = self._create_prototypes(yA_2d, hA, self.n_protos)
                y_proto_B, h_proto_B, _ = self._create_prototypes(yB_2d, hB, self.n_protos)
                self.proto_h_A.append(h_proto_A)
                self.proto_y_A.append(y_proto_A)
                self.proto_h_B.append(h_proto_B)
                self.proto_y_B.append(y_proto_B)

                # ── Update accumulated predictions for the next layer's target ──
                # Use the freshly-stored protos to compute this layer's contribution
                pred_iA = self._proto_predict(hA, y_proto_A, h_proto_A)
                pred_iB = self._proto_predict(hB, y_proto_B, h_proto_B)
                if acc_proto_A is None:
                    acc_proto_A = pred_iA
                    acc_proto_B = pred_iB
                else:
                    acc_proto_A = _zero_pad_sum(acc_proto_A, pred_iA)
                    acc_proto_B = _zero_pad_sum(acc_proto_B, pred_iB)

                # Raw scalar output of this layer, for residual_mode="direct".
                # The orientation of h is not guaranteed, so the sample axis is
                # identified against the A-zone sample count.
                def _first_col(h, n_ref):
                    h = np.asarray(h, dtype=np.float64)
                    if h.ndim == 1:
                        return h.reshape(-1, 1)
                    if h.shape[0] != n_ref and h.shape[1] == n_ref:
                        h = h.T
                    return h[:, :1]

                d_iA = _first_col(hA, y_orig_A.shape[0])
                d_iB = _first_col(hB, y_orig_B.shape[0])
                if acc_direct_A is None:
                    acc_direct_A, acc_direct_B = d_iA, d_iB
                else:
                    acc_direct_A = _zero_pad_sum(acc_direct_A, d_iA)
                    acc_direct_B = _zero_pad_sum(acc_direct_B, d_iB)

            # Next layer input = this layer's output (wire-direct)
            cur_XA, cur_XB = hA.copy(), hB.copy()
            cur_dim = n_v

            # Transition zone via both networks, averaged
            if len(cur_XT) > 0:
                hTA = self._forward_layer(layer_A, cur_XT)
                hTB = self._forward_layer(layer_B, cur_XT)
                if hTA.shape[0] != cur_XT.shape[0]:
                    hTA = hTA.T
                if hTB.shape[0] != cur_XT.shape[0]:
                    hTB = hTB.T
                cur_XT = (hTA + hTB) / 2.0

        return layers_A, layers_B, data

    # ── Inference ──

    def _proto_accumulate(
        self, layers: List[PhysicalRBFLayerND], X: np.ndarray,
        proto_h: List[np.ndarray], proto_y: List[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Run one network's cascade and SUM each layer's proto-weighted contribution.

        This is the prediction `train` is built around -- its accumulation loop carries the
        comment "Proto maps h_i -> residual target (or full y for layer 1).  At predict
        time these are summed -> recovers original y" -- and layers 2 and 3 are trained on
        `y - acc`, so each layer's own output is a RESIDUAL rather than the target.

        Returns None when the prototypes are not available for every layer, so the caller
        can fall back rather than guess.
        """
        if not layers or len(proto_h) < len(layers) or len(proto_y) < len(layers):
            return None
        h = np.asarray(X, dtype=np.float32).copy()
        acc = None
        for i, layer in enumerate(layers):
            h = self._forward_layer(layer, h)
            if h.shape[0] != X.shape[0]:
                h = h.T
            h = self._match_shape(h, (X.shape[0], h.shape[-1]))
            ph = np.asarray(proto_h[i], dtype=np.float64)
            py = np.asarray(proto_y[i], dtype=np.float64)
            # the stored prototype matrix can be in the transposed frame
            if ph.shape[1] != h.shape[1] and ph.shape[0] == h.shape[1]:
                ph = ph.T
            if ph.shape[1] != h.shape[1]:
                return None
            pred_i = self._proto_predict(np.asarray(h, dtype=np.float64), py, ph)
            acc = pred_i if acc is None else _zero_pad_sum(acc, pred_i)
        return acc

    def predict(
        self, layers_A: List[PhysicalRBFLayerND], layers_B: List[PhysicalRBFLayerND],
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Dual-network inference.

        Regression: proto-weighted prediction with residual accumulation.
          output = (acc_A + acc_B) / 2

        Classification: weighted ensemble voting across layers.
          Each layer votes via nearest-prototype; weights = (0.5, 0.3, 0.2).
        """
        if self.task == "classification":
            return self._predict_classification(layers_A, layers_B, X)

        # REGRESSION RETURNS THE ACCUMULATED SUM, NOT THE LAST LAYER'S OUTPUT.
        #
        # This branch used to `return (hA + hB) / 2.0`, i.e. the last layer's raw output --
        # which contradicts the docstring above AND the training path, where layers 2/3
        # learn `y - acc` and so emit RESIDUALS.  Measured (extension FINDINGS.md 12.29):
        # the same trained models score -0.0341 through the old return and +0.2291 through
        # the accumulation, with the per-zone scores near zero either way, which rules out
        # the A/B median split as the explanation.  The defect survived because no
        # published number depends on the regression path -- its only two callers were a
        # speed benchmark and a script that was retracted.
        acc_A = self._proto_accumulate(layers_A, X, self.proto_h_A, self.proto_y_A)
        acc_B = self._proto_accumulate(layers_B, X, self.proto_h_B, self.proto_y_B)
        if acc_A is not None and acc_B is not None:
            d = min(acc_A.shape[1], acc_B.shape[1])
            return (acc_A[:, :d] + acc_B[:, :d]) / 2.0

        # FALL BACK, and say so rather than silently returning a residual: a model restored
        # without its prototypes cannot be read the intended way.
        hA = X.copy()
        hB = X.copy()
        for i in range(len(layers_A)):
            hA_out = self._forward_layer(layers_A[i], hA)
            hB_out = self._forward_layer(layers_B[i], hB)
            if hA_out.shape[0] != hA.shape[0]:
                hA_out = hA_out.T
            if hB_out.shape[0] != hB.shape[0]:
                hB_out = hB_out.T
            hA, hB = hA_out, hB_out
        return (hA + hB) / 2.0

    def _predict_classification(
        self, layers_A: List[PhysicalRBFLayerND], layers_B: List[PhysicalRBFLayerND],
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Classification inference with confidence gating and A/B routing.

        Each layer forwards through both A and B networks independently.
        Confidence = 1 / (1 + min_squared_distance_to_any_prototype).

        Routing (L1 only): samples in A-zone give A's vote 2x weight,
        samples in B-zone give B's vote 2x weight. This preserves the
        dual-network specialization learned during HiMAL training.

        Layer weights: (0.5, 0.3, 0.2).
        """
        layer_weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        hA = X.copy()
        hB = X.copy()
        votes = np.zeros((len(X), self.n_classes), dtype=np.float64)

        # ── Routing weights for L1 ──
        if (self._split_method in ("median_x", "median_y")
                and self._split_value is not None):
            dim = 0 if self._split_method == "median_x" else 1
            if X.shape[1] > dim:
                in_A_zone = X[:, dim].astype(np.float64) <= self._split_value
                route_A = np.where(in_A_zone, 2.0, 0.5)  # boost in A-zone
                route_B = np.where(in_A_zone, 0.5, 2.0)  # boost in B-zone
            else:
                route_A = route_B = np.ones(len(X), dtype=np.float64)
        else:
            route_A = route_B = np.ones(len(X), dtype=np.float64)

        for i in range(len(layers_A)):
            hA_out = self._forward_layer(layers_A[i], hA)
            hB_out = self._forward_layer(layers_B[i], hB)

            if hA_out.shape[0] != hA.shape[0]:
                hA_out = hA_out.T
            if hB_out.shape[0] != hB.shape[0]:
                hB_out = hB_out.T

            # ── Network A ──
            if i < len(self.class_protos_A):
                pred_A, conf_A = self._predict_with_confidence(
                    hA_out, self.class_protos_A[i])
                w = conf_A * layer_weights[i]
                if i == 0:  # routing only for L1
                    w = w * route_A
                np.add.at(votes, (np.arange(len(X)), pred_A.astype(int)), w)

            # ── Network B ──
            if i < len(self.class_protos_B):
                pred_B, conf_B = self._predict_with_confidence(
                    hB_out, self.class_protos_B[i])
                w = conf_B * layer_weights[i]
                if i == 0:
                    w = w * route_B
                np.add.at(votes, (np.arange(len(X)), pred_B.astype(int)), w)

            # Wire-direct
            hA, hB = hA_out, hB_out

        return votes.argmax(axis=1)

    @staticmethod
    def _predict_with_confidence(h: np.ndarray, protos: np.ndarray):
        """
        Nearest-prototype prediction with confidence.

        confidence = 1 / (1 + ||h - nearest_proto||²)
        Higher confidence when h is close to a prototype.
        All ops: subtract, square, sum → physically realizable.
        """
        if h.ndim == 1:
            h = h.reshape(-1, 1)
        diff = h[:, np.newaxis, :] - protos[np.newaxis, :, :]  # (N, K, d)
        d2 = np.sum(diff * diff, axis=2)  # (N, K) squared distances
        pred = d2.argmin(axis=1)
        min_d2 = d2.min(axis=1)
        conf = 1.0 / (1.0 + min_d2)  # (N,)
        return pred, conf
