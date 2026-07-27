"""
Sklearn Baseline Comparison

Runs standard sklearn models on the same train/test split
for direct comparison with v3 Physical RBF models.
"""

import numpy as np
import time
from typing import Dict, Optional
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.dummy import DummyRegressor, DummyClassifier


# ── Model registry ──

_REGRESSION_MODELS = {
    "Mean":        lambda: DummyRegressor(strategy="mean"),
    "Ridge":       lambda: Ridge(alpha=1.0),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, max_depth=10,
                                                   random_state=42, n_jobs=-1),
    "KNN":         lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    "SVR":         lambda: SVR(kernel="rbf", C=1.0),
    "MLP":         lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                         random_state=42, early_stopping=True),
}

_CLASSIFICATION_MODELS = {
    "Majority":      lambda n_classes: DummyClassifier(strategy="most_frequent"),
    "Logistic":      lambda n_classes: LogisticRegression(max_iter=500, random_state=42),
    "RandomForest":  lambda n_classes: RandomForestClassifier(n_estimators=100, max_depth=10,
                                                               random_state=42, n_jobs=-1),
    "KNN":           lambda n_classes: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVC":           lambda n_classes: SVC(kernel="rbf", C=1.0, random_state=42),
    "MLP":           lambda n_classes: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                                      random_state=42, early_stopping=True),
}


def run_sklearn_baselines(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    task: str = "regression",
    n_classes: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Run all sklearn baseline models and return metrics.

    Regression metrics: MSE, R2, time
    Classification metrics: accuracy, time

    Parameters
    ----------
    X_tr, y_tr, X_te, y_te : training and test data
    task : "regression" or "classification"
    n_classes : int (required for classification)

    Returns
    -------
    results : {model_name: {"mse": float, "r2": float, "time": float, ...}}
    """
    if task == "classification":
        models = _CLASSIFICATION_MODELS
    else:
        models = _REGRESSION_MODELS

    results = {}
    for name, factory in models.items():
        t0 = time.time()
        try:
            if task == "classification":
                model = factory(n_classes or 2)
            else:
                model = factory()

            model.fit(X_tr, y_tr.ravel())
            preds = model.predict(X_te)
            train_time = time.time() - t0

            entry = {"time": train_time}
            if task == "classification":
                entry["accuracy"] = float((preds == y_te.ravel()).mean())
            else:
                mse = float(np.mean((preds.ravel() - y_te.ravel()) ** 2))
                ss_tot = float(np.mean((y_te.ravel() - y_te.mean()) ** 2))
                entry["mse"] = mse
                entry["r2"] = float(1.0 - mse / max(ss_tot, 1e-10))

            results[name] = entry
        except Exception as e:
            results[name] = {"error": str(e), "time": time.time() - t0}

    # ── Sort by primary metric ──
    if task == "classification":
        results = dict(sorted(results.items(),
                              key=lambda kv: kv[1].get("accuracy", 0),
                              reverse=True))
    else:
        results = dict(sorted(results.items(),
                              key=lambda kv: kv[1].get("mse", float("inf"))))

    return results
