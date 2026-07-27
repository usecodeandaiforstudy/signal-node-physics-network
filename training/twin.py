"""
Digital twin surrogate gradient training

Uses PyTorch ResidualRBFNetwork's autograd to provide gradient direction,
but forward loss computation goes through physical_simulator to verify the
impact of physical non-idealities on training.

Core approach:
  - Forward: physical_simulator (with noise/mismatch) → compute loss
  - Backward: PyTorch autograd (nominal gradient direction)
  - Parameter sync: after each update, digital model params → physical simulator params
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


class DigitalTwinTrainer:
    """
    Digital twin trainer.

    Training method:
      θ_digital ← θ_digital - lr * ∇_θ L( f_physical(θ_digital; X), y )

    Where:
    - f_physical() is the physical simulator forward (with non-idealities)
    - L is the loss function
    - ∇_θ is computed via PyTorch autograd on the digital model
    - After each update, digital params are synced to the physical simulator

    Parameters
    ----------
    digital_model : ResidualRBFNetwork
        PyTorch digital model (provides gradients)
    physical_model : PhysicalResidualRBFNetwork
        Physical simulator (provides real forward output)
    device : torch.device
    """

    def __init__(self, digital_model, physical_model, device="cpu"):
        self.digital = digital_model
        self.physical = physical_model
        self.device = device
        self.history = {"train_loss": [], "eval_loss": []}

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        cov_reg_fn=None,
        cov_lambda: float = 0.0,
    ) -> float:
        """
        One digital twin training step.

        Flow:
          1. Sync params: digital → physical
          2. Physical forward: with noise
          3. Digital forward: autograd (nominal)
          4. Compute loss using physical output
          5. Backpropagate on digital model
          6. Sync params: digital → physical

        Parameters
        ----------
        x : (N, 2) numpy
            Input data
        y : (N, d) numpy
            Targets
        optimizer : torch optimizer
        criterion : torch loss
        cov_reg_fn : callable, optional
            Covariance regularization function
        cov_lambda : float

        Returns
        -------
        loss : float
            Total loss value
        """
        self.digital.train()

        # 1. Sync digital → physical
        self.physical.sync_from_digital(self.digital)

        # 2. Physical forward (numpy)
        x_np = np.asarray(x)
        pred_physical = self.physical.forward(x_np[:, 0], x_np[:, 1])

        # Ensure shape match
        if pred_physical.ndim == 1:
            pred_physical = pred_physical.reshape(-1, 1)

        y_np = np.asarray(y).reshape(pred_physical.shape)

        # 3. Compute MSE (as primary loss)
        mse = float(np.mean((pred_physical - y_np) ** 2))

        # 4. Compute gradients on digital model (autograd)
        x_t = torch.tensor(x_np[:, 0].astype(np.float32), requires_grad=False).to(self.device)
        y_t = torch.tensor(x_np[:, 1].astype(np.float32), requires_grad=False).to(self.device)
        target_t = torch.tensor(y_np.astype(np.float32)).to(self.device)

        optimizer.zero_grad()
        pred_digital = self.digital(x_t, y_t)

        # Ensure shape match
        if pred_digital.shape != target_t.shape:
            pred_digital = pred_digital[:, :target_t.shape[1]]

        loss_digital = criterion(pred_digital, target_t)

        # Covariance regularization
        if cov_reg_fn is not None and cov_lambda > 0:
            reg = cov_reg_fn(self.digital, torch.tensor(x_np, dtype=torch.float32))
            loss_digital = loss_digital + cov_lambda * reg

        loss_digital.backward()
        optimizer.step()

        # 5. Sync back to physical
        self.physical.sync_from_digital(self.digital)

        return mse

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 300,
        batch_size: int = 64,
        cov_reg_fn=None,
        cov_lambda: float = 0.0,
        eval_fn=None,
        verbose: bool = True,
        verbose_interval: int = 50,
    ) -> dict:
        """
        Full digital twin training loop.

        Parameters
        ----------
        X, y : ndarray
            Training data
        optimizer, criterion
        epochs : int
            Number of training epochs
        batch_size : int
        cov_reg_fn, cov_lambda
            Covariance regularization
        eval_fn : callable, optional
            eval_fn(model, X_eval, y_eval) -> metrics_dict
        verbose : bool
        verbose_interval : int

        Returns
        -------
        history : dict
            Training history
        """
        N = len(X)
        X_np = np.asarray(X)
        y_np = np.asarray(y)

        for epoch in range(epochs):
            # Mini-batch shuffle
            indices = np.random.permutation(N)
            epoch_losses = []

            for start in range(0, N, batch_size):
                batch_idx = indices[start:start + batch_size]
                x_batch = X_np[batch_idx]
                y_batch = y_np[batch_idx]

                loss = self.train_step(
                    x_batch, y_batch,
                    optimizer, criterion,
                    cov_reg_fn, cov_lambda,
                )
                epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(avg_loss)

            if verbose and (epoch + 1) % verbose_interval == 0:
                msg = f"  Epoch {epoch+1:4d}/{epochs} | loss={avg_loss:.6f}"
                if eval_fn:
                    eval_metrics = eval_fn(self, X_np, y_np)
                    msg += f" | eval={eval_metrics}"
                print(msg)

        return self.history

    def eval_forward(self, X: np.ndarray, use_physical: bool = True) -> np.ndarray:
        """Forward prediction"""
        X_np = np.asarray(X)
        if use_physical:
            return self.physical.forward(X_np[:, 0], X_np[:, 1])
        else:
            with torch.no_grad():
                x_t = torch.tensor(X_np[:, 0], dtype=torch.float32).to(self.device)
                y_t = torch.tensor(X_np[:, 1], dtype=torch.float32).to(self.device)
                return self.digital(x_t, y_t).cpu().numpy()
