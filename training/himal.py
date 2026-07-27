"""
HiMAL (Hierarchical Multi-Agent Learning) layer-wise dual-network training

Core idea:
  - Two networks (A and B) learn different partitions of the data
  - Layer-wise independent training: each layer is frozen after training before the next
  - Transition zone alignment loss replaces cross-layer backpropagation
  - Each layer has few parameters (~150), updated via finite differences

Completely backpropagation-free.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from copy import deepcopy


class HiMALTrainer:
    """
    HiMAL layer-wise dual-network trainer.

    Training flow:
      1. Data partition: split into A-zone, B-zone, T-zone (transition) by feature
         value / gradient magnitude
      2. For each layer i:
         a. Forward: hA = LayerA_i(inA), hB = LayerB_i(inB)
         b. Compute loss: L = L_fit_A + L_fit_B + γ * L_align(hTA, hTB)
         c. Update LayerA_i and LayerB_i params via finite differences
         d. Freeze, output in_next = h

    Parameters
    ----------
    layer_sizes : list of int
        Number of particles per layer [n1, n2, n3]
    output_dims : list of int
        Output dimension per layer [d1, d2, d3]
    x_range, y_range : tuple
        Input space range
    alignment_weight : float
        Alignment loss weight γ
    """

    def __init__(
        self,
        layer_sizes: Tuple[int, int, int] = (20, 20, 20),
        output_dims: Tuple[int, int, int] = (1, 1, 1),
        x_range: Tuple[float, float] = (-3.0, 3.0),
        y_range: Tuple[float, float] = (-3.0, 3.0),
        alignment_weight: float = 0.3,
    ):
        self.layer_sizes = layer_sizes
        self.output_dims = output_dims
        self.x_range = x_range
        self.y_range = y_range
        self.alignment_weight = alignment_weight

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "median_x",
        transition_width: float = 0.2,
    ) -> dict:
        """
        Split data into A-zone, B-zone, T-zone.

        Parameters
        ----------
        X : (N, 2)
            Input data (x_coord, y_coord)
        y : (N,) or (N, d)
            Labels / targets
        method : str
            "median_x" - partition by x median
            "median_y" - partition by y median
            "gradient" - partition by label gradient magnitude (regression)
        transition_width : float
            Transition zone width (proportion of feature range)

        Returns
        -------
        dict with keys: XA, yA, XB, yB, XT, yT, split_value
        """
        if method == "median_x":
            split_feature = X[:, 0]
        elif method == "median_y":
            split_feature = X[:, 1]
        elif method == "gradient":
            assert y.ndim in (1, 2)
            if y.ndim == 2:
                y_flat = y.flatten()
            else:
                y_flat = y
            split_feature = np.abs(np.gradient(y_flat))
        else:
            raise ValueError(f"Unknown split method: {method}")

        median = np.median(split_feature)
        data_range = np.ptp(split_feature)
        width = transition_width * data_range

        mask_A = split_feature <= median
        mask_B = split_feature > median
        trans_mask = (split_feature >= median - width) & (split_feature <= median + width)

        return {
            "XA": X[mask_A], "yA": y[mask_A],
            "XB": X[mask_B], "yB": y[mask_B],
            "XT": X[trans_mask], "yT": y[trans_mask],
            "split_value": float(median),
            "masks": {"A": mask_A, "B": mask_B, "T": trans_mask},
        }

    def _init_random_layer(
        self,
        n_particles: int,
        n_vars: int,
    ) -> tuple:
        """Initialize a random layer (one copy each for dual networks A, B)"""
        xL, xR = self.x_range
        yL, yR = self.y_range

        def make_layer():
            xc = np.random.uniform(xL, xR, n_particles).astype(np.float32)
            yc = np.random.uniform(yL, yR, n_particles).astype(np.float32)
            dsize = max(xR - xL, yR - yL)
            h = (dsize / np.sqrt(n_particles)) * np.ones(n_particles, dtype=np.float32)
            W_gauss = np.random.uniform(-0.1, 0.1, (n_vars, n_particles)).astype(np.float32)
            W_act = np.random.uniform(-0.1, 0.1, (n_vars, n_particles)).astype(np.float32)
            return {"xc": xc, "yc": yc, "h": h, "W_gauss": W_gauss, "W_act": W_act, "use_pu": True}

        return make_layer(), make_layer()

    def _vectorize_params(self, layer_params: dict) -> np.ndarray:
        """Flatten layer parameter dict into a vector"""
        return np.concatenate([
            layer_params["xc"].flatten(),
            layer_params["yc"].flatten(),
            layer_params["h"].flatten(),
            layer_params["W_gauss"].flatten(),
            layer_params["W_act"].flatten(),
        ])

    def _devectorize_params(self, vec: np.ndarray, layer_params: dict):
        """Restore layer params from flattened vector"""
        n = len(layer_params["xc"])
        n_wg = layer_params["W_gauss"].size
        n_wa = layer_params["W_act"].size

        layer_params["xc"] = vec[0:n].copy()
        layer_params["yc"] = vec[n:2*n].copy()
        layer_params["h"] = vec[2*n:3*n].copy()
        offset = 3 * n
        layer_params["W_gauss"] = vec[offset:offset + n_wg].reshape(layer_params["W_gauss"].shape).copy()
        offset += n_wg
        layer_params["W_act"] = vec[offset:offset + n_wa].reshape(layer_params["W_act"].shape).copy()

    def _forward_layer(self, layer_params: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Physical layer forward propagation (uses physical functions from components).
        Returns (batch, n_vars) fused output.
        """
        from physical_simulator.components import (
            differential_pair, gilbert_square,
            bjt_gaussian_kernel, current_mode_normalize,
            memristor_mvm, kirchhoff_sum,
        )

        n = layer_params["xc"].shape[0]

        # Step 1: Differential pair
        dx = differential_pair(x, layer_params["xc"], layer_params["h"])
        dy = differential_pair(y, layer_params["yc"], layer_params["h"])

        # Step 2: Gilbert squarer
        d2 = gilbert_square(dx, dy)

        # Step 3: BJT Gaussian kernel (dual channel)
        z_gauss = bjt_gaussian_kernel(d2, k=0.5)
        z_act = bjt_gaussian_kernel(d2, k=0.3)

        # Step 4: PU normalization
        if layer_params["use_pu"]:
            z_gauss = current_mode_normalize(z_gauss)
            z_act = current_mode_normalize(z_act)

        # Step 5: Memristor MVM
        out_gauss = memristor_mvm(z_gauss.T, layer_params["W_gauss"]).T
        out_act = memristor_mvm(z_act.T, layer_params["W_act"]).T

        # Step 6: Addition
        return kirchhoff_sum(out_gauss, out_act)

    def train_one_layer(
        self,
        data: dict,
        n_particles: int,
        n_vars: int,
        in_dim_A: int = 2,
        in_dim_B: int = 2,
        epochs: int = 200,
        lr: float = 0.01,
        eps: float = 1e-4,
        verbose: bool = True,
    ) -> Tuple[dict, dict]:
        """
        Train one layer (dual networks A, B) using finite differences.

        Parameters
        ----------
        data : dict
            Partitioned data {"XA", "yA", "XB", "yB", "XT", "yT"}
        n_particles : int
            Number of particles in this layer
        n_vars : int
            Output dimension of this layer
        in_dim_A, in_dim_B : int
            Input dimensions (for reshape)
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        eps : float
            Finite difference step size

        Returns
        -------
        layer_A, layer_B : dict
            Trained parameter dicts for both layers
        """
        layer_A, layer_B = self._init_random_layer(n_particles, n_vars)
        gamma = self.alignment_weight

        for epoch in range(epochs):
            # ---- Concatenate params ----
            pA = self._vectorize_params(layer_A)
            pB = self._vectorize_params(layer_B)
            all_params = {"A": layer_A, "B": layer_B}

            # ---- Per-parameter finite difference ----
            for net_name, net_params in [("A", layer_A), ("B", layer_B)]:
                other_name = "B" if net_name == "A" else "A"
                other_params = layer_B if net_name == "B" else layer_A
                vec = self._vectorize_params(net_params)
                grad = np.zeros_like(vec)

                for j in range(len(vec)):
                    # +eps
                    vec[j] += eps
                    self._devectorize_params(vec, net_params)
                    L_plus = self._compute_pair_loss(
                        layer_A, layer_B, data,
                        in_dim_A, in_dim_B, gamma,
                    )
                    # -eps
                    vec[j] -= 2 * eps
                    self._devectorize_params(vec, net_params)
                    L_minus = self._compute_pair_loss(
                        layer_A, layer_B, data,
                        in_dim_A, in_dim_B, gamma,
                    )
                    # Restore
                    vec[j] += eps

                    grad[j] = (L_plus - L_minus) / (2 * eps)

                # Update
                vec = vec - lr * grad
                self._devectorize_params(vec, net_params)

            if verbose and (epoch + 1) % 40 == 0:
                loss = self._compute_pair_loss(layer_A, layer_B, data, in_dim_A, in_dim_B, gamma)
                print(f"  Layer epoch {epoch+1:3d}/{epochs} | loss={loss:.6f}")

        return layer_A, layer_B

    def _compute_pair_loss(
        self,
        layer_A: dict,
        layer_B: dict,
        data: dict,
        in_dim_A: int,
        in_dim_B: int,
        gamma: float,
    ) -> float:
        """Compute total dual-network loss: L_fit_A + L_fit_B + γ·L_align"""
        # Forward
        hA = self._forward_layer(layer_A, data["XA"][:, 0], data["XA"][:, 1])
        hB = self._forward_layer(layer_B, data["XB"][:, 0], data["XB"][:, 1])

        # Fitting loss (MSE)
        # Handle shape: hA may be (n_vars, batch) or (batch, n_vars)
        if hA.shape[0] == data["yA"].shape[0]:
            fit_A = np.mean((hA.reshape(data["yA"].shape) - data["yA"]) ** 2)
        else:
            fit_A = np.mean((hA.T.reshape(data["yA"].shape) - data["yA"]) ** 2)

        if hB.shape[0] == data["yB"].shape[0]:
            fit_B = np.mean((hB.reshape(data["yB"].shape) - data["yB"]) ** 2)
        else:
            fit_B = np.mean((hB.T.reshape(data["yB"].shape) - data["yB"]) ** 2)

        # Alignment loss (MSE on transition zone)
        if len(data["XT"]) > 0:
            hTA = self._forward_layer(layer_A, data["XT"][:, 0], data["XT"][:, 1])
            hTB = self._forward_layer(layer_B, data["XT"][:, 0], data["XT"][:, 1])
            # Ensure consistent shapes
            shape = hTA.shape
            align = np.mean((hTA.reshape(-1) - hTB.reshape(-1)) ** 2)
        else:
            align = 0.0

        return float(fit_A + fit_B + gamma * align)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_method: str = "median_x",
        epochs_per_layer: int = 200,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> Tuple[List[dict], List[dict], dict]:
        """
        Full HiMAL training pipeline.

        Parameters
        ----------
        X : (N, 2)
            Input data
        y : (N,) or (N, d)
            Targets
        split_method : str
            Partitioning method
        epochs_per_layer : int
            Training epochs per layer
        lr : float
            Learning rate
        verbose : bool

        Returns
        -------
        layers_A, layers_B : list of dict
            Trained parameters (3 layers)
        data_split : dict
            Data partition info
        """
        data = self.split_data(X, y, method=split_method)
        layers_A, layers_B = [], []

        # Current input (changes per layer)
        cur_XA, cur_XB, cur_XT = data["XA"].copy(), data["XB"].copy(), data["XT"].copy()
        cur_yA, cur_yB = data["yA"].copy(), data["yB"].copy()

        for i in range(3):
            if verbose:
                print(f"\n===== HiMAL Layer {i+1} ({self.layer_sizes[i]} particles) =====")

            n_p = self.layer_sizes[i]
            n_v = self.output_dims[i]

            # Build training data using current layer's input dimensions
            layer_data = {
                "XA": cur_XA, "yA": cur_yA,
                "XB": cur_XB, "yB": cur_yB,
                "XT": cur_XT, "yT": data["yT"] if i == 0 else None,
            }

            # Train this layer
            layer_A, layer_B = self.train_one_layer(
                layer_data, n_p, n_v,
                in_dim_A=cur_XA.shape[1], in_dim_B=cur_XB.shape[1],
                epochs=epochs_per_layer, lr=lr,
                verbose=verbose,
            )

            layers_A.append(layer_A)
            layers_B.append(layer_B)

            # Compute next layer input
            hA = self._forward_layer(layer_A, cur_XA[:, 0], cur_XA[:, 1])
            hB = self._forward_layer(layer_B, cur_XB[:, 0], cur_XB[:, 1])

            # Transpose shape to (batch, dim)
            if hA.shape[0] != cur_XA.shape[0]:
                hA = hA.T
            if hB.shape[0] != cur_XB.shape[0]:
                hB = hB.T

            # Next layer target: use residual form
            cur_yA = cur_yA.reshape(hA.shape) if cur_yA.ndim == 1 or cur_yA.shape[1] != hA.shape[1] else cur_yA
            cur_yB = cur_yB.reshape(hB.shape) if cur_yB.ndim == 1 or cur_yB.shape[1] != hB.shape[1] else cur_yB

            # For Layer 2 and 3, the target is residual (simplified: keep using original target)
            cur_XA, cur_XB = hA, hB

            # Transition zone forward
            if len(cur_XT) > 0:
                hTA = self._forward_layer(layer_A, cur_XT[:, 0], cur_XT[:, 1])
                hTB = self._forward_layer(layer_B, cur_XT[:, 0], cur_XT[:, 1])
                if hTA.shape[0] != cur_XT.shape[0]:
                    hTA = hTA.T
                if hTB.shape[0] != cur_XT.shape[0]:
                    hTB = hTB.T
                cur_XT = (hTA + hTB) / 2.0

        return layers_A, layers_B, data

    def predict(
        self,
        layers_A: List[dict],
        layers_B: List[dict],
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Inference using the trained dual networks.

        Parameters
        ----------
        layers_A, layers_B : list of dict
            3-layer parameters
        X : (N, 2)
            Input data

        Returns
        -------
        pred : (N, output_dim)
            Averaged prediction from dual networks
        """
        hA = X.copy()
        hB = X.copy()

        for i in range(3):
            hA_out = self._forward_layer(layers_A[i], hA[:, 0], hA[:, 1])
            hB_out = self._forward_layer(layers_B[i], hB[:, 0], hB[:, 1])

            if hA_out.shape[0] != hA.shape[0]:
                hA_out = hA_out.T
            if hB_out.shape[0] != hB.shape[0]:
                hB_out = hB_out.T

            if i == 0:
                hA, hB = hA_out, hB_out
            else:
                hA = hA + hA_out  # Residual
                hB = hB + hB_out

        return (hA + hB) / 2.0
