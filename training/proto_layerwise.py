"""
Proto-Guided Layer-wise Trainer (Pure Physical Domain, No Digital Pretrain)

Core idea:
  - Each RBF layer is paired with K learnable prototype vectors (K = n_classes)
  - Forward path: 6-step analog circuit chain → hidden representation h →
                    L2 distance to prototypes → soft assignment probability
  - Loss: CrossEntropy(soft_assignment, labels) + prototype_regularization
  - Prototype-distance loss provides strong gradient signal FROM RANDOM INIT,
    unlike raw logit cross-entropy which collapses when outputs are near-uniform
  - All gradients estimated via per-parameter finite differences on the
    physical simulator — ZERO digital autograd involvement
  - Layers trained sequentially bottom-to-top; each layer frozen after training

Why this works from scratch:
  Random prototypes naturally induce a nearest-prototype assignment for each
  sample. The cross-entropy on this assignment yields a non-zero gradient
  even when the hidden representation is random, because the distance to
  the correct-class prototype differs from the distance to wrong-class ones.
  Prototype regularization (pushing different-class prototypes apart) prevents
  collapse and maintains a meaningful geometry throughout training.

Contrast with other trainers:
  - HiMALTrainer: dual-network, data partition, transition alignment
  - SingleLayerwiseTrainer: single-network MSE/Procrustes, needs pretrain
  - ProtoGuidedTrainer: single-network prototype-guided, works from scratch
"""

import numpy as np
from typing import Tuple, List, Optional
from copy import deepcopy

from physical_simulator.layer import PhysicalRBFLayer
from physical_simulator import NoiseConfig
from physical_simulator.components import (
    differential_pair, gilbert_square,
    bjt_gaussian_kernel, current_mode_normalize,
    memristor_mvm, kirchhoff_sum,
)


class ProtoGuidedTrainer:
    """
    Proto-guided layer-wise trainer for pure physical-domain optimization.

    Training flow (per layer):
      1. Initialize random RBF particles + wide-spread prototypes
      2. For each epoch:
         a. Physical forward → hidden representation h (batch, n_vars)
         b. L2 distance: dist[j] = ||h - proto_j||  → (batch, K)
         c. Soft assignment: p = softmax(-dist / temperature)
         d. Loss = CE(p, labels) + λ * proto_reg(prototypes)
         e. FD gradient on all params (RBF + prototypes)
         f. Update params
      3. Freeze layer; compute next-layer input via mean projection

    Parameters
    ----------
    layer_sizes : tuple of int
        Particles per layer, e.g. (3, 3, 3)
    output_dims : tuple of int
        Output dimension per layer, e.g. (2, 2, 2)
    n_classes : int
        Number of classes for classification
    x_range, y_range : tuple
        Input space range for RBF particle initialization
    temperature : float
        Softmax temperature for prototype assignment (smaller = sharper)
    proto_reg_weight : float
        Weight λ for prototype regularization loss
    """

    def __init__(
        self,
        layer_sizes: Tuple[int, int, int] = (3, 3, 3),
        output_dims: Tuple[int, int, int] = (2, 2, 2),
        n_classes: int = 2,
        x_range: Tuple[float, float] = (-3.0, 3.0),
        y_range: Tuple[float, float] = (-3.0, 3.0),
        temperature: float = 0.1,
        proto_reg_weight: float = 0.1,
    ):
        self.layer_sizes = layer_sizes
        self.output_dims = output_dims
        self.n_classes = n_classes
        self.x_range = x_range
        self.y_range = y_range
        self.temperature = temperature
        self.proto_reg_weight = proto_reg_weight

    # ──────────────────────────────────────────────
    #  RBF particle initialization (numpy-based PhysicalRBFLayer)
    # ──────────────────────────────────────────────

    def _init_layer(
        self,
        n_particles: int,
        n_vars: int,
    ) -> PhysicalRBFLayer:
        """
        Create a PhysicalRBFLayer with random RBF particles.

        Parameters
        ----------
        n_particles : int
            Number of RBF particles
        n_vars : int
            Output dimension

        Returns
        -------
        layer : PhysicalRBFLayer
        """
        xL, xR = self.x_range
        yL, yR = self.y_range

        xc = np.random.uniform(xL, xR, n_particles).astype(np.float32)
        yc = np.random.uniform(yL, yR, n_particles).astype(np.float32)
        dsize = max(xR - xL, yR - yL)
        h = (dsize / np.sqrt(max(n_particles, 1))) * np.ones(n_particles, dtype=np.float32)
        W_gauss = np.random.uniform(-0.5, 0.5, (n_vars, n_particles)).astype(np.float32)
        W_act = np.random.uniform(-0.5, 0.5, (n_vars, n_particles)).astype(np.float32)

        return PhysicalRBFLayer(
            xc=xc, yc=yc, h=h,
            W_gauss=W_gauss, W_act=W_act,
            use_pu=True,
        )

    # ──────────────────────────────────────────────
    #  Prototype initialization
    # ──────────────────────────────────────────────

    def _init_prototypes(self, n_vars: int) -> np.ndarray:
        """
        Initialize K prototype vectors spread far apart in feature space.

        Strategy: place prototypes at vertices of a regular simplex scaled
        to the output dimension. This guarantees that no two prototypes are
        close, giving meaningful distance-based assignments from the start.

        Parameters
        ----------
        n_vars : int
            Dimensionality of each prototype

        Returns
        -------
        protos : (n_classes, n_vars) ndarray
        """
        K = self.n_classes
        if K == 2:
            # Binary: place prototypes at opposite poles on first axis
            protos = np.zeros((K, n_vars), dtype=np.float32)
            protos[0, 0] = 2.0
            protos[1, 0] = -2.0
        else:
            # Multi-class: use one-hot-like initialization spread across dims
            protos = np.zeros((K, n_vars), dtype=np.float32)
            for k in range(K):
                dim = k % n_vars
                sign = 1.0 if k < n_vars else -1.0
                protos[k, dim] = sign * 2.0
        return protos

    # ──────────────────────────────────────────────
    #  Prototype manipulation helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _vectorize_prototypes(protos: np.ndarray) -> np.ndarray:
        """Flatten prototype matrix (K, d) → vector of length K*d."""
        return protos.flatten().copy()

    @staticmethod
    def _devectorize_prototypes(vec: np.ndarray, K: int, d: int) -> np.ndarray:
        """Reshape vector → prototype matrix (K, d)."""
        return vec.reshape(K, d).copy()

    # ──────────────────────────────────────────────
    #  Prototype regularization loss
    # ──────────────────────────────────────────────

    def _proto_regularization(self, protos: np.ndarray) -> float:
        """
        Encourage different-class prototypes to be far apart.

        Loss = MSE( pairwise_L2_distances, ideal_distance_matrix )
        where ideal_dist[k1][k2] = 1 if k1 != k2 else 0.

        This prevents all prototypes from collapsing to a single point
        and maintains meaningful class separation geometry.

        Parameters
        ----------
        protos : (K, d) ndarray

        Returns
        -------
        reg_loss : float
        """
        K = protos.shape[0]
        # Pairwise L2 distances
        diff = protos[:, np.newaxis, :] - protos[np.newaxis, :, :]  # (K, K, d)
        d = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)               # (K, K)
        ideal = 1.0 - np.eye(K, dtype=np.float32)                     # 1 for i≠j, 0 for i=j
        return float(np.mean((d - ideal) ** 2))

    # ──────────────────────────────────────────────
    #  Physical forward (same 6-step circuit chain)
    # ──────────────────────────────────────────────

    def _forward_layer(
        self,
        layer: PhysicalRBFLayer,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-layer physical forward through the analog circuit chain.

        Returns both Gaussian and activation channel outputs.

        Parameters
        ----------
        layer : PhysicalRBFLayer
        x, y : (batch,) ndarray

        Returns
        -------
        out_gauss : (batch, n_vars)
        out_act   : (batch, n_vars)
        """
        return layer.forward(x, y)

    def _forward_full(
        self,
        layers: List[PhysicalRBFLayer],
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Full 3-layer physical forward with residual accumulation.

        Layer 1: f1 = out_g1 + out_a1
        Layer 2: f2 = f1 + delta2       (delta2 = out_g2 + out_a2)
        Layer 3: f3 = f2 + delta3

        Inter-layer projection: each layer's gaussian and activation outputs
        are reduced to scalars via .mean(-1), then paired as (x, y) for the
        next layer's input.

        Parameters
        ----------
        layers : list of PhysicalRBFLayer (length 3)
        X : (N, 2) ndarray
            Original input coordinates

        Returns
        -------
        out : (N, n_vars_last_layer) ndarray
        """
        x = X[:, 0].astype(np.float32).ravel()
        y = X[:, 1].astype(np.float32).ravel() if X.ndim > 1 and X.shape[1] >= 2 else x.copy()

        # Layer 1
        out_g1, out_a1 = layers[0].forward(x, y)
        f1 = out_g1 + out_a1
        g1 = out_g1.mean(-1) if out_g1.ndim > 1 and out_g1.shape[-1] > 1 else out_g1.ravel()
        a1 = out_a1.mean(-1) if out_a1.ndim > 1 and out_a1.shape[-1] > 1 else out_a1.ravel()

        # Layer 2 (residual)
        out_g2, out_a2 = layers[1].forward(g1, a1)
        delta2 = out_g2 + out_a2
        # Pad/truncate shapes to match
        if f1.shape[1] > delta2.shape[1]:
            d2 = np.zeros_like(f1); d2[:, :delta2.shape[1]] = delta2; delta2 = d2
        elif delta2.shape[1] > f1.shape[1]:
            f1p = np.zeros_like(delta2); f1p[:, :f1.shape[1]] = f1; f1 = f1p
        f2 = f1 + delta2
        g2 = out_g2.mean(-1) if out_g2.ndim > 1 and out_g2.shape[-1] > 1 else out_g2.ravel()
        a2 = out_a2.mean(-1) if out_a2.ndim > 1 and out_a2.shape[-1] > 1 else out_a2.ravel()

        # Layer 3 (residual)
        out_g3, out_a3 = layers[2].forward(g2, a2)
        delta3 = out_g3 + out_a3
        if f2.shape[1] > delta3.shape[1]:
            d3 = np.zeros_like(f2); d3[:, :delta3.shape[1]] = delta3; delta3 = d3
        elif delta3.shape[1] > f2.shape[1]:
            f2p = np.zeros_like(delta3); f2p[:, :f2.shape[1]] = f2; f2 = f2p
        f3 = f2 + delta3

        out_dim = min(f3.shape[1], self.output_dims[-1])
        return f3[:, :out_dim]

    # ──────────────────────────────────────────────
    #  Prototype-guided loss (the key insight)
    # ──────────────────────────────────────────────

    def _compute_proto_loss(
        self,
        layer_params: dict,
        protos: np.ndarray,
        layer: PhysicalRBFLayer,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Compute prototype-guided classification loss for a single layer.

        Loss = CrossEntropy(softmax(-dist), labels) + λ * proto_reg

        This is computed in ISOLATION for one layer: only the current layer's
        hidden representation and prototypes are evaluated. No downstream
        layers contribute to the loss.

        Parameters
        ----------
        layer_params : dict
            Current state of the layer for this FD perturbation.
            Devectorized into `layer` before forward.
        protos : (K, d) ndarray
            Current prototype values
        layer : PhysicalRBFLayer
            The layer object (params may be temporarily modified by FD loop)
        X : (N, 2) ndarray
            Input data for this layer
        y : (N,) ndarray (int)
            Class labels

        Returns
        -------
        loss : float
        """
        x_coord = X[:, 0].astype(np.float32).ravel()
        y_coord = X[:, 1].astype(np.float32).ravel() if X.ndim > 1 and X.shape[1] >= 2 else x_coord.copy()

        # Physical forward
        out_g, out_a = layer.forward(x_coord, y_coord)
        h = out_g + out_a  # (batch, n_vars) — fused hidden representation

        # Ensure shape
        if h.ndim == 1:
            h = h.reshape(-1, 1)

        # Pairwise L2 distance: h → prototypes
        # h: (N, d), protos: (K, d) → dist: (N, K)
        h_exp = h[:, np.newaxis, :]                       # (N, 1, d)
        p_exp = protos[np.newaxis, :, :]                   # (1, K, d)
        dist = np.sqrt(np.sum((h_exp - p_exp) ** 2, axis=2) + 1e-12)  # (N, K)

        # Soft assignment: softmax(-dist / temperature)
        logits = -dist / self.temperature
        logits = logits - logits.max(1, keepdims=True)    # numerical stability
        probs = np.exp(logits)
        probs = probs / probs.sum(1, keepdims=True)

        # Cross-entropy
        y_int = y.astype(np.int64).ravel()
        ce_loss = float(-np.log(probs[np.arange(len(y_int)), y_int] + 1e-12).mean())

        # Prototype regularization
        reg_loss = self._proto_regularization(protos)

        return ce_loss + self.proto_reg_weight * reg_loss

    # ──────────────────────────────────────────────
    #  Finite-difference gradient estimation
    # ──────────────────────────────────────────────

    def _fd_update(
        self,
        params: np.ndarray,
        loss_fn,
        lr: float,
        eps: float,
    ) -> Tuple[np.ndarray, float]:
        """
        One round of symmetric finite-difference update on a parameter vector.

        Parameters
        ----------
        params : ndarray
            Current parameter vector
        loss_fn : callable(params) → float
            Loss function that takes the parameter vector and returns scalar loss
        lr : float
            Learning rate
        eps : float
            FD perturbation step

        Returns
        -------
        new_params : ndarray
            Updated parameter vector
        current_loss : float
            Loss at the updated position
        """
        n = len(params)
        grad = np.zeros(n, dtype=np.float64)

        for j in range(n):
            params[j] += eps
            L_plus = loss_fn(params)
            params[j] -= 2 * eps
            L_minus = loss_fn(params)
            params[j] += eps  # restore
            grad[j] = (L_plus - L_minus) / (2 * eps)

        params = params - lr * grad.astype(params.dtype)
        current_loss = loss_fn(params)
        return params, current_loss

    # ──────────────────────────────────────────────
    #  Single-layer training
    # ──────────────────────────────────────────────

    def train_one_layer(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_particles: int,
        n_vars: int,
        epochs: int = 200,
        lr: float = 0.01,
        eps: float = 1e-3,
        verbose: bool = True,
    ) -> Tuple[PhysicalRBFLayer, np.ndarray]:
        """
        Train one layer + its prototypes from scratch using FD.

        Both RBF particle parameters and prototype vectors are optimized
        jointly. The loss is computed entirely through the physical simulator.

        Parameters
        ----------
        X : (N, 2) ndarray
            Input data for this layer
        y : (N,) ndarray (int)
            Class labels
        n_particles : int
            Number of RBF particles in this layer
        n_vars : int
            Output dimension of this layer
        epochs : int
            Training epochs
        lr : float
            Learning rate
        eps : float
            FD perturbation step size
        verbose : bool

        Returns
        -------
        layer : PhysicalRBFLayer
            Trained layer
        protos : (K, n_vars) ndarray
            Trained prototypes
        """
        # Initialize
        layer = self._init_layer(n_particles, n_vars)
        protos = self._init_prototypes(n_vars)

        # Build loss closure over the concatenated parameter vector
        def make_loss_fn():
            """Factory: creates a loss_fn that captures the current layer and protos."""
            # We'll rebuild this each epoch because the closure captures references
            pass

        rbf_vec = layer.get_params_vector()
        proto_vec = self._vectorize_prototypes(protos)

        best_layer = deepcopy(layer)
        best_protos = protos.copy()
        best_loss = float("inf")

        for epoch in range(epochs):
            # Re-read params (they may have been updated)
            rbf_vec = layer.get_params_vector()
            proto_vec = self._vectorize_prototypes(protos)

            # ---- Update RBF parameters ----
            def rbf_loss_fn(p):
                layer.set_params_from_vector(p)
                return self._compute_proto_loss(None, protos, layer, X, y)

            rbf_vec_old = rbf_vec.copy()
            rbf_vec, _ = self._fd_update(rbf_vec, rbf_loss_fn, lr, eps)
            # Reject NaN updates
            if not np.isnan(rbf_vec).any():
                layer.set_params_from_vector(rbf_vec)
            else:
                rbf_vec = rbf_vec_old
                layer.set_params_from_vector(rbf_vec)

            # ---- Update prototype parameters ----
            def proto_loss_fn(p):
                nonlocal protos
                K, d = protos.shape
                protos = self._devectorize_prototypes(p, K, d)
                return self._compute_proto_loss(None, protos, layer, X, y)

            proto_vec_old = proto_vec.copy()
            proto_vec, _ = self._fd_update(proto_vec, proto_loss_fn, lr, eps)
            if not np.isnan(proto_vec).any():
                protos = self._devectorize_prototypes(proto_vec, self.n_classes, n_vars)
            else:
                proto_vec = proto_vec_old
                protos = self._devectorize_prototypes(proto_vec, self.n_classes, n_vars)

            # Track best
            current_loss = self._compute_proto_loss(None, protos, layer, X, y)
            if current_loss < best_loss:
                best_loss = current_loss
                best_layer = PhysicalRBFLayer(
                    xc=layer.xc.copy(), yc=layer.yc.copy(), h=layer.h.copy(),
                    W_gauss=layer.W_gauss.copy(), W_act=layer.W_act.copy(),
                    use_pu=layer.use_pu,
                )
                best_protos = protos.copy()

            if verbose and (epoch + 1) % 40 == 0:
                # Compute accuracy for monitoring
                pred = self._predict_single_layer(layer, protos, X)
                acc = (pred == y.ravel()).mean()
                print(f"  Proto layer epoch {epoch+1:4d}/{epochs} | loss={current_loss:.4f} "
                      f"acc={acc:.4f} | best_loss={best_loss:.4f}")

        return best_layer, best_protos

    def _predict_single_layer(
        self,
        layer: PhysicalRBFLayer,
        protos: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class labels using one layer + its prototypes.

        Parameters
        ----------
        layer : PhysicalRBFLayer
        protos : (K, d) ndarray
        X : (N, 2) ndarray

        Returns
        -------
        pred : (N,) ndarray (int)
        """
        x_coord = X[:, 0].astype(np.float32).ravel()
        y_coord = X[:, 1].astype(np.float32).ravel() if X.ndim > 1 and X.shape[1] >= 2 else x_coord.copy()
        out_g, out_a = layer.forward(x_coord, y_coord)
        h = out_g + out_a
        if h.ndim == 1:
            h = h.reshape(-1, 1)

        h_exp = h[:, np.newaxis, :]
        p_exp = protos[np.newaxis, :, :]
        dist = np.sqrt(np.sum((h_exp - p_exp) ** 2, axis=2) + 1e-12)
        return dist.argmin(axis=1)

    # ──────────────────────────────────────────────
    #  Full training pipeline
    # ──────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs_per_layer: int = 200,
        lr: float = 0.01,
        eps: float = 1e-3,
        verbose: bool = True,
    ) -> Tuple[List[PhysicalRBFLayer], List[np.ndarray]]:
        """
        Full 3-layer proto-guided physical training from scratch.

        For each layer i:
          1. Train Layer i + prototypes_i on (input_i, labels)
          2. Freeze Layer i
          3. Compute input_{i+1} by projecting Layer i output to 2D

        Parameters
        ----------
        X : (N, 2) ndarray
            Original input data
        y : (N,) ndarray (int)
            Class labels
        epochs_per_layer : int
            Training epochs per layer
        lr : float
            Learning rate
        eps : float
            FD perturbation step
        verbose : bool

        Returns
        -------
        layers : list of PhysicalRBFLayer (length 3)
        all_protos : list of (K, d) ndarray (length 3)
        """
        layers = []
        all_protos = []
        cur_in = X.copy()

        for i in range(3):
            if verbose:
                print(f"\n===== Proto Layer {i+1} ({self.layer_sizes[i]} particles, "
                      f"dim={self.output_dims[i]}) =====")

            n_p = self.layer_sizes[i]
            n_v = self.output_dims[i]

            # Train this layer from scratch
            layer, protos = self.train_one_layer(
                X=cur_in,
                y=y,
                n_particles=n_p,
                n_vars=n_v,
                epochs=epochs_per_layer,
                lr=lr,
                eps=eps,
                verbose=verbose,
            )

            layers.append(layer)
            all_protos.append(protos)

            # Freeze and compute next layer input
            x_coord = cur_in[:, 0].astype(np.float32).ravel()
            y_coord = cur_in[:, 1].astype(np.float32).ravel() if cur_in.ndim > 1 and cur_in.shape[1] >= 2 else x_coord.copy()
            out_g, out_a = layer.forward(x_coord, y_coord)
            cur_in = self._project_to_2d(out_g, out_a)

        return layers, all_protos

    # ──────────────────────────────────────────────
    #  Inter-layer projection helper
    # ──────────────────────────────────────────────

    @staticmethod
    def _project_to_2d(
        out_g: np.ndarray,
        out_a: np.ndarray,
    ) -> np.ndarray:
        """
        Project multi-dimensional layer output to 2D (x, y) for the next layer.

        Strategy: take the first 2 output dimensions directly from each channel.
        This preserves more representational structure than mean(-1) when the
        output dimension is small (e.g. 4 → 2). When output_dim < 2, fall back
        to broadcasting.

        Parameters
        ----------
        out_g : (N, d) ndarray
            Gaussian channel output
        out_a : (N, d) ndarray
            Activation channel output

        Returns
        -------
        next_input : (N, 2) ndarray
        """
        if out_g.ndim == 1:
            return np.column_stack([out_g, out_a]).astype(np.float32)
        d = out_g.shape[1]
        if d >= 2:
            cur_x = out_g[:, 0].ravel()
            cur_y = out_a[:, 0].ravel() if out_a.shape[1] >= 2 else out_a[:, 0].ravel()
        else:
            cur_x = out_g.mean(-1).ravel()
            cur_y = out_a.mean(-1).ravel()
        return np.column_stack([cur_x, cur_y]).astype(np.float32)

    # ──────────────────────────────────────────────
    #  Inference
    # ──────────────────────────────────────────────

    def predict(
        self,
        layers: List[PhysicalRBFLayer],
        all_protos: List[np.ndarray],
        X: np.ndarray,
        ensemble_weights: Tuple[float, float, float] = (0.6, 0.25, 0.15),
    ) -> np.ndarray:
        """
        Weighted ensemble inference across independently-trained layers.

        Each layer was trained independently as a standalone classifier.
        Layer 1 (closest to raw input) typically achieves the highest accuracy.
        Layer 2 and 3 serve as refinement — they get lower weight but can
        correct Layer 1 mistakes when their confidence is high.

        Parameters
        ----------
        layers : list of PhysicalRBFLayer (length 3)
        all_protos : list of (K, d) ndarray (length 3)
        X : (N, 2) ndarray
        ensemble_weights : tuple of 3 floats summing to 1.0
            Weight per layer in the final vote

        Returns
        -------
        pred : (N,) ndarray (int)
        """
        cur_in = X.copy()
        weighted_votes = np.zeros((len(X), self.n_classes), dtype=np.float64)

        for i in range(3):
            # Classify using this layer's prototypes on its current input
            p = self._predict_single_layer(layers[i], all_protos[i], cur_in)
            # Accumulate weighted one-hot votes
            one_hot = np.zeros((len(X), self.n_classes), dtype=np.float64)
            one_hot[np.arange(len(X)), p.astype(int)] = ensemble_weights[i]
            weighted_votes += one_hot

            # Project to next layer's input space
            if i < 2:  # No need to project after the last layer
                xc = cur_in[:, 0].astype(np.float32).ravel()
                yc = cur_in[:, 1].astype(np.float32).ravel() if cur_in.ndim > 1 and cur_in.shape[1] >= 2 else xc.copy()
                out_g, out_a = layers[i].forward(xc, yc)
                cur_in = self._project_to_2d(out_g, out_a)

        return weighted_votes.argmax(axis=1)

    def predict_single(
        self,
        layer: PhysicalRBFLayer,
        protos: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict using a single layer + its prototypes (no residual chain).

        Useful for per-layer diagnostic evaluation.

        Parameters
        ----------
        layer : PhysicalRBFLayer
        protos : (K, d) ndarray
        X : (N, 2) ndarray

        Returns
        -------
        pred : (N,) ndarray (int)
        """
        return self._predict_single_layer(layer, protos, X)
