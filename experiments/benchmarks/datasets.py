"""
Standard Dataset Loader

Unified interface for regression and classification benchmarks.
All datasets return (X_tr, X_te, y_tr, y_te) with standardized features.

Supported datasets:
  Classification: iris, wine, breast_cancer, digits, moons, circles
  Regression:     diabetes, friedman1, california_housing, boston
"""

import numpy as np
from typing import Tuple, Dict, List
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    load_diabetes, make_friedman1,
)
from sklearn.datasets import fetch_california_housing  # noqa: E402
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Dataset registry ──

_DATASETS: Dict[str, dict] = {
    # Classification
    "iris":           {"task": "classification", "loader": "_load_iris"},
    "wine":           {"task": "classification", "loader": "_load_wine"},
    "breast_cancer":  {"task": "classification", "loader": "_load_breast_cancer"},
    "digits":         {"task": "classification", "loader": "_load_digits"},
    "moons":          {"task": "classification", "loader": "_load_moons"},
    "circles":        {"task": "classification", "loader": "_load_circles"},
    # Regression
    "diabetes":       {"task": "regression", "loader": "_load_diabetes"},
    "friedman1":      {"task": "regression", "loader": "_load_friedman1"},
    "california":     {"task": "regression", "loader": "_load_california"},
}


def list_datasets() -> List[str]:
    """Return list of available dataset names."""
    return sorted(_DATASETS.keys())


def load_dataset(name: str, test_size: float = 0.2, seed: int = 42,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load a standard dataset and return train/test splits.

    Returns
    -------
    X_tr, X_te, y_tr, y_te : ndarray
    info : dict with keys {"task", "n_features", "n_classes" (if cls), "name"}
    """
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list_datasets()}")

    loader_name = _DATASETS[name]["loader"]
    task = _DATASETS[name]["task"]
    loader_fn = globals()[loader_name]
    X, y = loader_fn()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if task == "classification" else None,
    )

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)

    y_tr = np.asarray(y_tr, dtype=np.float32 if task == "regression" else np.int64)
    y_te = np.asarray(y_te, dtype=np.float32 if task == "regression" else np.int64)

    info = {
        "name": name, "task": task,
        "n_features": X_tr.shape[1],
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    if task == "classification":
        info["n_classes"] = len(np.unique(y_tr))

    return X_tr, X_te, y_tr, y_te, info


# ── Classification loaders ──

def _load_iris():
    data = load_iris()
    return data.data, data.target


def _load_wine():
    data = load_wine()
    return data.data, data.target


def _load_breast_cancer():
    data = load_breast_cancer()
    return data.data, data.target


def _load_digits():
    data = load_digits()
    return data.data, data.target


def _load_moons(n_samples: int = 500, noise: float = 0.12):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y


def _load_circles(n_samples: int = 500, noise: float = 0.08):
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=42)
    return X, y


# ── Regression loaders ──

def _load_diabetes():
    data = load_diabetes()
    return data.data, data.target


def _load_friedman1(n_samples: int = 500):
    X, y = make_friedman1(n_samples=n_samples, noise=0.1, random_state=42)
    return X, y


def _load_california():
    data = fetch_california_housing()
    return data.data, data.target
