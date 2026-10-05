"""
V3 Proto-Guided Layer-wise Trainer for D-Dimensional Particles

Adapts v1 ProtoGuidedTrainer (forward_training/proto_layerwise.py) for
D-dimensional particles.

Key differences from v1:
  - Layer uses PhysicalRBFLayerND (centers (D, N) instead of (xc, yc))
  - Inter-layer: wire-direct (output IS next input, no projection needed)
  - No _project_to_2d — the D-dimensional generalization eliminates this step
  - Forward path: 6-step analog circuit chain → hidden h → L2 to prototypes → CE loss
  - All gradients via per-parameter FD on physical simulator — ZERO autograd
"""

import numpy as np
from typing import Tuple, List
from copy import deepcopy

from simulator.layer_nd import PhysicalRBFLayerND
from simulator.network_nd import PhysicalResidualRBFNetworkND
from physical_simulator import NoiseConfig


class ProtoGuidedTrainerND:
    """
    Proto-guided layer-wise trainer for pure physical-domain optimization.

    Training flow (per layer):
      1. Initialize random RBF particles + spread-out prototypes
      2. For each epoch:
         a. Physical forward → hidden h (batch, n_vars)
         b. L2 distance: dist[j] = ||h - proto_j|| → (batch, K)
         c. Soft assignment: p = softmax(-dist / temperature)
         d. Loss = CE(p, labels) + lambda * proto_reg(protos)
         e. FD gradient on all params (RBF + prototypes)
         f. Update params
      3. Freeze layer; output becomes next layer input (wire-direct)

    Parameters
    ----------
    layer_sizes : tuple of int
        Particles per layer, e.g. (3, 3, 3)
    output_dims : tuple of int
        Output dimension per layer, e.g. (4, 3, 2)
    n_classes : int
        Number of classes
    input_dim : int
        Input dimensionality D
    init_range : tuple of float
        Range for RBF particle initialization
    temperature : float
        Softmax temperature (smaller = sharper)
    proto_reg_weight : float
        Prototype regularization weight
    """

    def __init__(
        self,
        layer_sizes: Tuple[int, int, int] = (3, 3, 3),
        output_dims: Tuple[int, int, int] = (4, 3, 2),
        n_classes: int = 2,
        input_dim: int = 4,
        init_range: Tuple[float, float] = (-3.0, 3.0),
        temperature: float = 0.1,
        proto_reg_weight: float = 0.1,
        seed: int = 0,
    ):
        self.layer_sizes = layer_sizes
        self.output_dims = output_dims
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.init_range = init_range
        self.temperature = temperature
        self.proto_reg_weight = proto_reg_weight
        # ONE seed covers every random draw in this class.  There are exactly three,
        # all `np.random.uniform` in `_init_layer`, and they go through the GLOBAL
        # numpy RNG -- so without this, two constructions with identical arguments
        # gave DIFFERENT networks and every number published from this trainer was a
        # single draw from an unseeded process.  This is the same defect that was
        # found and fixed in `himal_nd.py` (extension FINDINGS.md 12.17); it was
        # scoped to that module at the time and is fixed here for the same reason.
        self.seed = int(seed)

    # ── Layer initialization ──

    def _init_layer(self, n_particles: int, input_dim: int, n_vars: int) -> PhysicalRBFLayerND:
        """Create a PhysicalRBFLayerND with random particles of given input dimension."""
        lo, hi = self.init_range

        xc = np.random.uniform(lo, hi, (input_dim, n_particles)).astype(np.float32)
        dsize = hi - lo
        h = (dsize / np.sqrt(max(n_particles, 1))) * np.ones(n_particles, dtype=np.float32)
        W_gauss = np.random.uniform(-0.5, 0.5, (n_vars, n_particles)).astype(np.float32)
        W_act = np.random.uniform(-0.5, 0.5, (n_vars, n_particles)).astype(np.float32)

        return PhysicalRBFLayerND(
            centers=xc, h=h, W_gauss=W_gauss, W_act=W_act, use_pu=True,
        )

    # ── Prototype initialization ──

    def _init_prototypes(self, n_vars: int) -> np.ndarray:
        """
        Initialize K prototype vectors spread far apart.
        Strategy: opposite poles on first axis for binary, one-hot for multi-class.
        """
        K = self.n_classes
        if K == 2:
            protos = np.zeros((K, n_vars), dtype=np.float32)
            protos[0, 0] = 2.0
            protos[1, 0] = -2.0
        else:
            protos = np.zeros((K, n_vars), dtype=np.float32)
            for k in range(K):
                dim = k % n_vars
                sign = 1.0 if k < n_vars else -1.0
                protos[k, dim] = sign * 2.0
        return protos

    # ── Prototype helpers ──

    @staticmethod
    def _vectorize_prototypes(protos: np.ndarray) -> np.ndarray:
        return protos.flatten().copy()

    @staticmethod
    def _devectorize_prototypes(vec: np.ndarray, K: int, d: int) -> np.ndarray:
        return vec.reshape(K, d).copy()

    def _proto_regularization(self, protos: np.ndarray) -> float:
        """Push different-class prototypes apart. Ideal distance = 1 for i≠j, 0 for i=j."""
        K = protos.shape[0]
        diff = protos[:, np.newaxis, :] - protos[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        ideal = 1.0 - np.eye(K, dtype=np.float32)
        return float(np.mean((d - ideal) ** 2))

    # ── Proto-guided loss ──

    def _compute_proto_loss(
        self,
        layer: PhysicalRBFLayerND,
        protos: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Prototype-guided classification loss for a single layer.

        Loss = CrossEntropy(softmax(-dist), labels) + lambda * proto_reg(protos)
        """
        out_g, out_a = layer.forward(X)
        h = out_g + out_a  # (batch, n_vars)

        if h.ndim == 1:
            h = h.reshape(-1, 1)

        # Pairwise L2: h (N, d) vs protos (K, d) → dist (N, K)
        h_exp = h[:, np.newaxis, :]
        p_exp = protos[np.newaxis, :, :]
        dist = np.sqrt(np.sum((h_exp - p_exp) ** 2, axis=2) + 1e-12)

        # Soft assignment
        logits = -dist / self.temperature
        logits = logits - logits.max(1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(1, keepdims=True)

        # Cross-entropy
        y_int = y.astype(np.int64).ravel()
        ce = float(-np.log(probs[np.arange(len(y_int)), y_int] + 1e-12).mean())

        # Regularization
        reg = self._proto_regularization(protos)

        return ce + self.proto_reg_weight * reg

    # ── FD gradient estimation ──

    def _fd_update(self, params: np.ndarray, loss_fn, lr: float, eps: float):
        """One round of symmetric FD update on a parameter vector."""
        n = len(params)
        grad = np.zeros(n, dtype=np.float64)

        for j in range(n):
            params[j] += eps
            L_plus = loss_fn(params)
            params[j] -= 2 * eps
            L_minus = loss_fn(params)
            params[j] += eps
            grad[j] = (L_plus - L_minus) / (2 * eps)

        params_new = params - lr * grad.astype(params.dtype)
        current_loss = loss_fn(params_new)
        return params_new, current_loss

    # ── Single-layer training ──

    def train_one_layer(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_particles: int,
        input_dim: int,
        n_vars: int,
        epochs: int = 200,
        lr: float = 0.01,
        eps: float = 1e-3,
        verbose: bool = True,
    ) -> Tuple[PhysicalRBFLayerND, np.ndarray]:
        """
        Train one layer + its prototypes from scratch using FD.

        Both RBF particle params and prototype vectors are optimized jointly.
        All computations through physical simulator.
        """
        layer = self._init_layer(n_particles, input_dim, n_vars)
        protos = self._init_prototypes(n_vars)

        best_layer = deepcopy(layer)
        best_protos = protos.copy()
        best_loss = float("inf")

        for epoch in range(epochs):
            # Update RBF parameters
            rbf_vec = layer.get_params_vector()
            def rbf_loss_fn(p):
                layer.set_params_from_vector(p)
                return self._compute_proto_loss(layer, protos, X, y)

            rbf_vec_old = rbf_vec.copy()
            rbf_vec, _ = self._fd_update(rbf_vec, rbf_loss_fn, lr, eps)
            if not np.isnan(rbf_vec).any():
                layer.set_params_from_vector(rbf_vec)
            else:
                layer.set_params_from_vector(rbf_vec_old)

            # Update prototypes
            proto_vec = self._vectorize_prototypes(protos)
            K, d = protos.shape
            def proto_loss_fn(p):
                nonlocal protos
                protos = self._devectorize_prototypes(p, K, d)
                return self._compute_proto_loss(layer, protos, X, y)

            proto_vec_old = proto_vec.copy()
            proto_vec, _ = self._fd_update(proto_vec, proto_loss_fn, lr, eps)
            if not np.isnan(proto_vec).any():
                protos = self._devectorize_prototypes(proto_vec, K, d)
            else:
                proto_vec = proto_vec_old
                protos = self._devectorize_prototypes(proto_vec, K, d)

            # Track best
            current_loss = self._compute_proto_loss(layer, protos, X, y)
            if current_loss < best_loss:
                best_loss = current_loss
                best_layer = PhysicalRBFLayerND(
                    centers=layer.centers.copy(), h=layer.h.copy(),
                    W_gauss=layer.W_gauss.copy(), W_act=layer.W_act.copy(),
                    use_pu=layer.use_pu,
                )
                best_protos = protos.copy()

            if verbose and (epoch + 1) % 40 == 0:
                pred = self._predict_single_layer(layer, protos, X)
                acc = (pred == y.ravel()).mean()
                print(f"  Proto layer epoch {epoch+1:4d}/{epochs} | loss={current_loss:.4f} "
                      f"acc={acc:.4f} | best_loss={best_loss:.4f}")

        return best_layer, best_protos

    def _predict_single_layer(self, layer: PhysicalRBFLayerND, protos: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Predict class labels using one layer + prototypes (nearest-prototype)."""
        out_g, out_a = layer.forward(X)
        h = out_g + out_a
        if h.ndim == 1:
            h = h.reshape(-1, 1)
        h_exp = h[:, np.newaxis, :]
        p_exp = protos[np.newaxis, :, :]
        dist = np.sqrt(np.sum((h_exp - p_exp) ** 2, axis=2) + 1e-12)
        return dist.argmin(axis=1)

    # ── Full training pipeline ──

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs_per_layer: int = 200,
        lr: float = 0.01,
        eps: float = 1e-3,
        verbose: bool = True,
    ) -> Tuple[List[PhysicalRBFLayerND], List[np.ndarray]]:
        """
        Full 3-layer proto-guided physical training from scratch.

        Wire-direct: each layer's fused output IS the next layer's input.
        No projection needed.
        """
        # ONE seed covers every random draw below: the layer initialisation in
        # `_init_layer`.  Same arguments and same `seed` now give identical networks.
        np.random.seed(self.seed)
        layers = []
        all_protos = []
        cur_in = X.copy()

        # Track current input dimension: starts as original D, then becomes
        # each layer's output dimension (wire-direct).
        cur_dim = self.input_dim

        for i in range(3):
            if verbose:
                print(f"\n===== Proto Layer {i+1} ({self.layer_sizes[i]} particles, "
                      f"in={cur_dim}D, out={self.output_dims[i]}D) =====")

            n_p = self.layer_sizes[i]
            n_v = self.output_dims[i]

            layer, protos = self.train_one_layer(
                X=cur_in, y=y, n_particles=n_p,
                input_dim=cur_dim, n_vars=n_v,
                epochs=epochs_per_layer, lr=lr, eps=eps, verbose=verbose,
            )

            layers.append(layer)
            all_protos.append(protos)

            # Compute next layer input = this layer's output (wire-direct)
            out_g, out_a = layer.forward(cur_in)
            cur_in = (out_g + out_a).astype(np.float32)
            cur_dim = n_v  # Next layer's input dim = this layer's output dim

        return layers, all_protos

    # ── Inference ──

    def predict(
        self,
        layers: List[PhysicalRBFLayerND],
        all_protos: List[np.ndarray],
        X: np.ndarray,
        ensemble_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Majority-vote ensemble inference across independently-trained layers.

        Each layer casts one vote; final prediction = argmax of accumulated votes.
        Default to equal weights (1,1,1) = standard majority voting.
        """
        cur_in = X.copy()
        weighted = np.zeros((len(X), self.n_classes), dtype=np.float64)

        for i in range(3):
            p = self._predict_single_layer(layers[i], all_protos[i], cur_in)
            onehot = np.zeros((len(X), self.n_classes), dtype=np.float64)
            onehot[np.arange(len(X)), p.astype(int)] = ensemble_weights[i]
            weighted += onehot

            # Next layer input (wire-direct)
            if i < 2:
                out_g, out_a = layers[i].forward(cur_in)
                cur_in = (out_g + out_a).astype(np.float32)

        return weighted.argmax(axis=1)

    def predict_single(
        self, layer: PhysicalRBFLayerND, protos: np.ndarray, X: np.ndarray,
    ) -> np.ndarray:
        """Predict using single layer + prototypes (per-layer diagnostic)."""
        return self._predict_single_layer(layer, protos, X)
