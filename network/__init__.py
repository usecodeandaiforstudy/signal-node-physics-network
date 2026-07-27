"""
V3 Digital-Domain Core Network

Contains:
  - ShiftND: D-dimensional particle position manager
  - RBFLayerND: Single RBF layer operating on D-dimensional particles
  - ResidualRBFNetworkND: Three-layer residual network with wire-direct connections
"""

from .core_nd import ShiftND, RBFLayerND, ResidualRBFNetworkND
from .utils_nd import (
    train_epoch, evaluate, fit,
    generate_rotation_data_nd, generate_2d_data_for_nd,
    plot_decision_boundary, plot_particles, plot_training_history,
    compute_cov_regularization, analyze_node_coverage,
    pca_project,
)

__all__ = [
    "ShiftND", "RBFLayerND", "ResidualRBFNetworkND",
    "train_epoch", "evaluate", "fit",
    "generate_rotation_data_nd", "generate_2d_data_for_nd",
    "plot_decision_boundary", "plot_particles", "plot_training_history",
    "compute_cov_regularization", "analyze_node_coverage",
    "pca_project",
]
