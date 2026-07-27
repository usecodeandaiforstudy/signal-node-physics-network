"""
V3 Combined HiMAL-RBF Network Model

Integrates:
  - RBF backbone: PhysicalRBFLayerND (analog circuit chain)
  - HiMAL training: dual-network, layer-wise FD, alignment loss
  - Physical inference: PhysicalResidualRBFNetworkND + proto-weighted predict

Supports regression and classification tasks.

This is the unified model class for end-to-end physical-domain workflow:
  1.initialize_random() → random RBF particle layers
  2. train() → pure physical FD training (HiMAL dual-network)
  3. predict() → proto-weighted inference or ensemble voting (digital readout)
  4. forward() → pure analog 3-layer forward (no softmax)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict

from simulator.layer_nd import PhysicalRBFLayerND
from simulator.network_nd import PhysicalResidualRBFNetworkND
from training.himal_nd import HiMALND
from physical_simulator import NoiseConfig


class CombinedHiMALRBF:
    """
    Combined RBF + HiMAL model for physical-domain neural computing.

    This is the primary user-facing class. It encapsulates:
      - RBF particle layer architecture (PhysicalRBFLayerND)
      - HiMAL dual-network training (layer-wise FD, A/B partition)
      - Physical inference with proto-weighted readout

    All forward propagation uses analog circuit simulation.
    Training uses per-parameter finite differences on physical MSE loss.
    Inference uses proto-weighted interpolation (digital readout).

    Parameters
    ----------
    input_dim : int
        Input dimensionality D
    layer_sizes : tuple of int
        Particles per layer [n1, n2, n3]
    output_dims : tuple of int
        Output dimension per layer [d1, d2, d3]
    init_range : tuple of float
        Range for random particle initialization
    alignment_weight : float
        HiMAL A/B transition-zone alignment weight
    n_protos : int
        Number of prototypes per network
    temperature : float
        Softmax temperature for proto-weighted predict
    use_spsa : bool
        Use SPSA gradient estimation (fast, ~2 fwd/epoch) instead of per-param FD
    spsa_avg : int
        Number of SPSA perturbations to average per update (reduces gradient noise)
    """

    def __init__(
        self,
        input_dim: int = 2,
        layer_sizes: Tuple[int, int, int] = (8, 6, 4),
        output_dims: Tuple[int, int, int] = (6, 4, 2),
        init_range: Tuple[float, float] = (-3.0, 3.0),
        alignment_weight: float = 0.3,
        n_protos: int = 8,
        temperature: float = 0.1,
        use_spsa: bool = False,
        spsa_avg: int = 1,
        spsa_clip: float = 1.0,
        spsa_decay_steps: float = 100.0,
        task: str = "regression",
        n_classes: int = 2,
    ):
        self.input_dim = input_dim
        self.layer_sizes = tuple(layer_sizes)
        self.output_dims = tuple(output_dims)
        self.init_range = init_range
        self.alignment_weight = alignment_weight
        self.n_protos = n_protos
        self.temperature = temperature
        self.use_spsa = use_spsa
        self.spsa_avg = spsa_avg
        self._spsa_clip = spsa_clip
        self._spsa_decay_steps = spsa_decay_steps
        self.task = task
        self.n_classes = n_classes

        # Internal trainer (created at train time)
        self._trainer: Optional[HiMALND] = None

        # Trained layers (populated after train())
        self.layers_A: List[PhysicalRBFLayerND] = []
        self.layers_B: List[PhysicalRBFLayerND] = []

        # Physical network wrappers (populated after train())
        self.net_A: Optional[PhysicalResidualRBFNetworkND] = None
        self.net_B: Optional[PhysicalResidualRBFNetworkND] = None

        # Dataset partition (populated after train())
        self.data_split: Dict = {}

    # ── Training ──

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_method: str = "median_x",
        epochs_per_layer: int = 80,
        lr: float = 0.012,
        eps: float = 1e-3,
        verbose: bool = True,
    ) -> "CombinedHiMALRBF":
        """
        Train the model using HiMAL dual-network layer-wise FD.

        Parameters
        ----------
        X : (N, D) ndarray
            Input data
        y : (N,) or (N, d) ndarray
            Target values
        split_method : str
            Data partition method ("median_x", "median_y", "gradient")
        epochs_per_layer : int
            Training epochs per layer
        lr : float
            Learning rate for FD updates
        eps : float
            Finite difference step size
        verbose : bool
            Print training progress

        Returns
        -------
        self
        """
        NoiseConfig.set_level("none")

        self._trainer = HiMALND(
            layer_sizes=self.layer_sizes,
            output_dims=self.output_dims,
            input_dim=self.input_dim,
            init_range=self.init_range,
            alignment_weight=self.alignment_weight,
            n_protos=self.n_protos,
            temperature=self.temperature,
            use_spsa=self.use_spsa,
            spsa_avg=self.spsa_avg,
            spsa_clip=self._spsa_clip,
            spsa_decay_steps=self._spsa_decay_steps,
            task=self.task,
            n_classes=self.n_classes,
        )

        self.layers_A, self.layers_B, self.data_split = self._trainer.train(
            X, y,
            split_method=split_method,
            epochs_per_layer=epochs_per_layer,
            lr=lr, eps=eps,
            verbose=verbose,
        )

        # Build physical network wrappers for forward() / forward_dual()
        if self.task == "regression":
            self.net_A, self.net_B = PhysicalResidualRBFNetworkND.from_himal_dual(
                self.layers_A, self.layers_B,
                self._trainer.proto_h_A, self._trainer.proto_y_A,
                self._trainer.proto_h_B, self._trainer.proto_y_B,
            )
        else:
            # Classification: build nets for pure analog forward only
            self.net_A = PhysicalResidualRBFNetworkND([self.layers_A[0],
                                                       self.layers_A[1],
                                                       self.layers_A[2]])
            self.net_A._proto_h = []
            self.net_A._proto_y = []
            self.net_B = PhysicalResidualRBFNetworkND([self.layers_B[0],
                                                       self.layers_B[1],
                                                       self.layers_B[2]])
            self.net_B._proto_h = []
            self.net_B._proto_y = []

        return self

    # ── Inference ──

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dual-network inference.

        Regression: proto-weighted residual accumulation.
        Classification: weighted ensemble voting across layers.

        Parameters
        ----------
        X : (N, D) ndarray

        Returns
        -------
        pred : (N,) or (N, d_y) ndarray
        """
        if self._trainer is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if self.task == "classification":
            return self._trainer.predict(self.layers_A, self.layers_B, X)

        if self.net_A is None or self.net_B is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return PhysicalResidualRBFNetworkND.predict_dual(
            self.net_A, self.net_B, X, temperature=self.temperature,
        )

    def forward(self, X: np.ndarray, noise_level: str = "none") -> np.ndarray:
        """
        Pure physical analog forward (no softmax, no proto).

        Uses the A-network layers. This is the path that maps to
        physical analog hardware.

        Parameters
        ----------
        X : (N, D) ndarray
        noise_level : str
            "none", "low", "medium", "high"

        Returns
        -------
        h : (N, d3) ndarray — raw output of 3-layer RBF chain
        """
        if self.net_A is None:
            raise RuntimeError("Model not trained. Call train() first.")
        NoiseConfig.set_level(noise_level)
        result = self.net_A.forward(X)
        NoiseConfig.set_level("none")
        return result

    def forward_dual(self, X: np.ndarray, noise_level: str = "none") -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure physical forward from both A and B networks.

        Returns (hA, hB).
        """
        if self.net_A is None or self.net_B is None:
            raise RuntimeError("Model not trained. Call train() first.")
        NoiseConfig.set_level(noise_level)
        hA = self.net_A.forward(X)
        hB = self.net_B.forward(X)
        NoiseConfig.set_level("none")
        return hA, hB

    # ── Properties ──

    @property
    def n_params_per_network(self) -> int:
        """Number of trainable parameters per network (A or B)."""
        if not self.layers_A:
            return 0
        return sum(l.get_params_vector().size for l in self.layers_A)

    @property
    def n_total_params(self) -> int:
        """Total trainable parameters across both networks."""
        return self.n_params_per_network * 2

    @property
    def trained(self) -> bool:
        """Whether the model has been trained."""
        return len(self.layers_A) > 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 50,
            "  Combined HiMAL-RBF Model",
            "=" * 50,
            f"  Task:               {self.task}" +
            (f" ({self.n_classes} classes)" if self.task == "classification" else ""),
            f"  Input dim:          {self.input_dim}D",
            f"  Layer particles:    {self.layer_sizes}",
            f"  Output dims:        {self.output_dims}",
            f"  Proto-weighted:     K={self.n_protos}, T={self.temperature}",
            f"  Dual-network:       A/B + alignment (gamma={self.alignment_weight})",
            f"  Gradient:           {'SPSA' if self.use_spsa else 'per-parameter FD'} "
            f"(avg={self.spsa_avg}, clip={self._spsa_clip})",
            f"  Params per network: {self.n_params_per_network}",
            f"  Total params:       {self.n_total_params}",
            f"  Trained:            {self.trained}",
            "-" * 50,
            "  Physical signal chain per layer:",
            "    1. differential_pair x D",
            "    2. gilbert_square x D",
            "    3. kirchhoff_sum",
            "    4. bjt_gaussian_kernel x 2",
            "    5. current_mode_normalize x 2",
            "    6. memristor_mvm x 2",
            "-" * 50,
            "  Training: HiMAL dual-network, layer-wise FD",
            "  Inference: PhysicalRBF forward + proto readout",
            "=" * 50,
        ]
        return "\n".join(lines)
