"""
Main Benchmark Runner

Runs v3 models against sklearn baselines on standard datasets.
Produces citable comparison tables.

Usage:
    python v3/experiments/benchmarks/run_benchmark.py  [--task regression|classification|all]
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from network.combined import CombinedHiMALRBF
from training.himal_nd import HiMALND
from physical_simulator import NoiseConfig
from .datasets import load_dataset, list_datasets
from .baselines import run_sklearn_baselines
from .report import generate_table, generate_summary_ranking

NoiseConfig.set_level("none")


# ── V3 model factory ──

def _create_v3_model(input_dim: int, task: str, n_classes: int = 2,
                     n_particles: int = 8) -> CombinedHiMALRBF:
    """Create a v3 model sized for benchmarks (small particles for speed)."""
    n1, n2, n3 = n_particles, max(n_particles // 2, 4), max(n_particles // 3, 3)
    d1, d2, d3 = max(input_dim, 4), max(input_dim // 2, 3), max(input_dim // 3, 2)
    if task == "classification":
        d3 = n_classes

    return CombinedHiMALRBF(
        input_dim=input_dim,
        layer_sizes=(n1, n2, n3),
        output_dims=(d1, d2, d3),
        init_range=(-3.0, 3.0),
        alignment_weight=0.3,
        n_protos=5,
        temperature=0.15,
        use_spsa=True,
        spsa_avg=4,
        spsa_clip=1.0,
        spsa_decay_steps=50.0,
        task=task,
        n_classes=n_classes,
    )


def evaluate_v3_regression(model, trainer, X, y):
    """Evaluate regression model: MSE, R2."""
    pred = model.predict(X)
    mse = float(np.mean((pred.ravel() - y.ravel()) ** 2))
    ss_tot = float(np.mean((y.ravel() - y.mean()) ** 2))
    r2 = float(1.0 - mse / max(ss_tot, 1e-10))
    return {"mse": mse, "r2": r2, "n_params": model.n_total_params}


def evaluate_v3_classification(model, trainer, X, y):
    """Evaluate classification model: accuracy."""
    pred = model.predict(X)
    acc = float((pred.ravel() == y.ravel()).mean())
    return {"accuracy": acc, "n_params": model.n_total_params}


# ── Main benchmark orchestration ──

def run_benchmark(datasets=None, tasks=None, epochs_per_layer=60,
                  lr=0.012, verbose=True):
    """
    Run full benchmark suite.

    Parameters
    ----------
    datasets : list of str or None
        Dataset names to run. None = all.
    tasks : list of str or None
        "regression", "classification", or None = both.
    epochs_per_layer : int
        Training epochs per layer (keep low for speed).
    lr : float
        Learning rate.
    verbose : bool

    Returns
    -------
    results : dict {dataset_name: {"metrics": {...}, "info": {...}}}
    """
    if datasets is None:
        datasets = list_datasets()
    if tasks is None:
        tasks = ["regression", "classification"]

    all_results = {}

    for ds_name in datasets:
        info = {}
        try:
            X_tr, X_te, y_tr, y_te, info = load_dataset(ds_name)
        except Exception as e:
            print(f"  SKIP {ds_name}: {e}")
            continue

        task = info["task"]
        if task not in tasks:
            continue

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Benchmark: {ds_name} ({task})")
            print(f"  Features: {info['n_features']} | "
                  f"Train: {info['n_train']} | Test: {info['n_test']}")
            if task == "classification":
                print(f"  Classes: {info.get('n_classes', '?')}")
            print(f"{'=' * 60}")

        # ── 1. Sklearn baselines ──
        if verbose:
            print("  Running sklearn baselines...")
        t0 = time.time()
        sk_results = run_sklearn_baselines(
            X_tr, y_tr, X_te, y_te, task=task,
            n_classes=info.get("n_classes"),
        )
        if verbose:
            print(f"  Done ({time.time() - t0:.1f}s)")

        # ── 2. V3 model ──
        n_classes = info.get("n_classes", 2)
        if verbose:
            print("  Training V3 CombinedHiMALRBF...")

        model = _create_v3_model(
            input_dim=info["n_features"],
            task=task, n_classes=n_classes,
            n_particles=10,
        )

        t0 = time.time()
        try:
            model.train(
                X_tr, y_tr,
                split_method="median_x",
                epochs_per_layer=epochs_per_layer,
                lr=lr, eps=1e-3,
                verbose=verbose,
            )
            train_time = time.time() - t0

            # Evaluate
            if task == "regression":
                v3_metrics = evaluate_v3_regression(model, model._trainer, X_te, y_te)
            else:
                v3_metrics = evaluate_v3_classification(model, model._trainer, X_te, y_te)
            v3_metrics["time"] = train_time

        except Exception as e:
            v3_metrics = {"error": str(e), "time": time.time() - t0}
            if verbose:
                print(f"  V3 FAILED: {e}")

        # ── 3. Assemble results ──
        metrics = dict(sk_results)
        metrics["V3_HiMAL"] = v3_metrics

        all_results[ds_name] = {
            "metrics": metrics,
            "info": info,
        }

        if verbose and "mse" in v3_metrics:
            print(f"  V3 MSE={v3_metrics['mse']:.4f} "
                  f"R2={v3_metrics.get('r2',0):.4f} "
                  f"({v3_metrics.get('n_params','?')} params)")
        elif verbose and "accuracy" in v3_metrics:
            print(f"  V3 Acc={v3_metrics['accuracy']:.4f} "
                  f"({v3_metrics.get('n_params','?')} params)")

    # ── 4. Generate report ──
    if verbose:
        print("\n")
        print(generate_table(all_results))
        print(generate_summary_ranking(all_results))

    return all_results


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="V3 Physical RBF Benchmark Runner")
    parser.add_argument("--task", type=str, default="all",
                        choices=["regression", "classification", "all"])
    parser.add_argument("--epochs", type=int, default=60,
                        help="Training epochs per layer")
    parser.add_argument("--lr", type=float, default=0.012,
                        help="Learning rate")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Specific dataset names (default: all)")
    args = parser.parse_args()

    tasks = None if args.task == "all" else [args.task]

    print("=" * 60)
    print("  V3 Physical RBF Benchmark Suite")
    print(f"  Tasks: {args.task}")
    print(f"  Datasets: {args.datasets or 'all'}")
    print(f"  Epochs/layer: {args.epochs}")
    print("=" * 60)

    t0 = time.time()
    results = run_benchmark(
        datasets=args.datasets,
        tasks=tasks,
        epochs_per_layer=args.epochs,
        lr=args.lr,
        verbose=True,
    )

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
