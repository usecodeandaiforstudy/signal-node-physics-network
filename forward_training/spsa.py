"""
SPSA (Simultaneous Perturbation Stochastic Approximation) zeroth-order optimizer

Each iteration requires only 2 loss evaluations; gradient estimation is
independent of the number of parameters. Naturally suited for physical
systems — only loss measurements are needed, no gradient computation.

Reference: Spall, J.C. (1998) "An Overview of SPSA"
"""

import numpy as np
from typing import Callable, Optional, Tuple


class SPSAOptimizer:
    """
    SPSA optimizer.

    Each iteration:
      1. Generate random perturbation vector Δ ∈ {+c, -c}^n
      2. Measure L(θ + Δ) and L(θ - Δ)
      3. Gradient estimate: g = (L(θ+Δ) - L(θ-Δ)) / (2cΔ)
      4. Parameter update: θ ← θ - lr * g

    Key properties:
    - Only 2 function evaluations per iteration
    - Gradient estimate independent of parameter dimension (perturbs all params
      simultaneously)
    - Naturally suited for noisy systems (gradient estimate is unbiased for
      zero-mean noise)
    - Perturbation step c and learning rate decay automatically

    Parameters
    ----------
    lr0 : float
        Initial learning rate
    c0 : float
        Initial perturbation step
    alpha : float
        Learning rate decay exponent (typically ~0.602)
    gamma : float
        Perturbation decay exponent (typically ~0.101)
    lr_min : float
        Minimum learning rate
    c_min : float
        Minimum perturbation step
    """

    def __init__(
        self,
        lr0: float = 0.01,
        c0: float = 0.01,
        alpha: float = 0.602,
        gamma: float = 0.101,
        lr_min: float = 1e-6,
        c_min: float = 1e-6,
    ):
        self.lr0 = lr0
        self.c0 = c0
        self.alpha = alpha
        self.gamma = gamma
        self.lr_min = lr_min
        self.c_min = c_min

        self._iteration = 0
        self._history: list = []

    @property
    def learning_rate(self) -> float:
        """Current learning rate (with decay)"""
        k = max(self._iteration, 1)
        return max(self.lr0 / (k ** self.alpha), self.lr_min)

    @property
    def perturbation(self) -> float:
        """Current perturbation magnitude (with decay)"""
        k = max(self._iteration, 1)
        return max(self.c0 / (k ** self.gamma), self.c_min)

    def step(
        self,
        loss_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Perform one SPSA update.

        Parameters
        ----------
        loss_fn : callable
            Loss function, takes parameter vector, returns scalar loss
        params : np.ndarray
            Current parameters (modified in-place)

        Returns
        -------
        params : np.ndarray
            Updated parameters
        loss : float
            Current loss (before perturbation)
        grad_est : np.ndarray
            SPSA gradient estimate
        """
        ck = self.perturbation
        lr = self.learning_rate

        # 1. Generate symmetric Bernoulli perturbation vector
        delta = 2 * np.random.binomial(1, 0.5, size=params.shape) - 1  # ±1

        # 2. Two loss evaluations
        params_plus = params + ck * delta
        params_minus = params - ck * delta

        loss_plus = loss_fn(params_plus)
        loss_minus = loss_fn(params_minus)

        # 3. Gradient estimate
        # g_i = (L(θ + cΔ) - L(θ - cΔ)) / (2c * Δ_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            grad_est = (loss_plus - loss_minus) / (2 * ck * delta)
            grad_est = np.nan_to_num(grad_est, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Parameter update
        params = params - lr * grad_est

        # 5. Current loss (average as record)
        current_loss = (loss_plus + loss_minus) / 2.0

        self._iteration += 1
        self._history.append({
            "iteration": self._iteration,
            "lr": lr,
            "c": ck,
            "loss": current_loss,
            "grad_norm": float(np.linalg.norm(grad_est)),
        })

        return params, current_loss, grad_est

    def optimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        init_params: np.ndarray,
        n_iter: int,
        verbose: bool = True,
        verbose_interval: int = 20,
        callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Full SPSA optimization loop.

        Parameters
        ----------
        loss_fn : callable
            Loss function
        init_params : np.ndarray
            Initial parameters
        n_iter : int
            Number of iterations
        verbose : bool
            Whether to print progress
        verbose_interval : int
            Print interval
        callback : callable, optional
            Callback function callback(params, iteration, loss)

        Returns
        -------
        params : np.ndarray
            Optimized parameters
        """
        params = init_params.copy()

        for i in range(n_iter):
            params, loss, _ = self.step(loss_fn, params)

            if verbose and (i + 1) % verbose_interval == 0:
                print(f"  SPSA iter {i+1:4d}/{n_iter} | loss={loss:.6f} | "
                      f"lr={self.learning_rate:.6f} | c={self.perturbation:.6f}")

            if callback:
                callback(params, i + 1, loss)

        return params

    def get_history(self) -> list:
        return self._history


def spsa_train_layer(
    layer,
    x: np.ndarray,
    y_target: np.ndarray,
    loss_fn: Callable,
    n_iter: int = 200,
    lr0: float = 0.01,
    c0: float = 0.01,
    verbose: bool = True,
) -> np.ndarray:
    """
    Train a physical layer using SPSA.

    Parameters
    ----------
    layer : PhysicalRBFLayer
        Layer to train
    x, y_target : np.ndarray
        Input (2 channels: x_coord, y_coord) and targets
    loss_fn : callable(forward_output, y_target) -> float
        Loss function
    n_iter : int
        Number of SPSA iterations
    lr0, c0 : float
        SPSA hyperparameters
    verbose : bool

    Returns
    -------
    params : np.ndarray
        Trained parameter vector
    """
    init_params = layer.get_params_vector()

    def spsa_loss(params):
        layer.set_params_from_vector(params)
        output = layer.forward(x[:, 0] if x.ndim > 1 else x,
                               x[:, 1] if x.ndim > 1 else x)
        # Get fused output
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output
        # Ensure shape match
        if pred.ndim > 1 and pred.shape[0] != y_target.shape[0]:
            pred = pred.T
        return float(loss_fn(pred, y_target))

    opt = SPSAOptimizer(lr0=lr0, c0=c0)
    params = opt.optimize(spsa_loss, init_params, n_iter, verbose=verbose)

    # Restore best params to layer
    layer.set_params_from_vector(params)

    return params
