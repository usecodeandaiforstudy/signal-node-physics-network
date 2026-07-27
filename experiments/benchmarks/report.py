"""
Benchmark Report Generation

Produces formatted tables comparing v3 models against sklearn baselines.
"""

from typing import Dict, List
import numpy as np


# ── ASCII table generator ──

def _pad_col(content: str, width: int, align: str = "left") -> str:
    """Pad a column cell to the given width."""
    s = str(content)
    if len(s) > width:
        s = s[:width - 2] + ".."
    if align == "right":
        return s.rjust(width)
    return s.ljust(width)


def generate_table(
    results: Dict[str, Dict[str, dict]],
    datasets_order: List[str] = None,
    width: int = 90,
) -> str:
    """
    Generate an ASCII comparison table from benchmark results.

    Parameters
    ----------
    results : {dataset_name: {"metrics": {"model": {metric: value, ...}},
                              "info": {...}}}
    datasets_order : optional ordering of datasets
    width : total table width

    Returns
    -------
    table_str : formatted multi-line string
    """
    if datasets_order is None:
        datasets_order = list(results.keys())

    lines = []
    sep = "─" * width

    lines.append("=" * width)
    lines.append("  V3 Physical RBF Benchmark Results".center(width))
    lines.append("=" * width)

    for ds_name in datasets_order:
        if ds_name not in results:
            continue
        entry = results[ds_name]
        info = entry.get("info", {})
        task = info.get("task", "?")
        metrics = entry.get("metrics", {})

        lines.append("")
        lines.append(f"  {ds_name}  ({task}, n_feat={info.get('n_features','?')}, "
                     f"n_train={info.get('n_train','?')}, n_test={info.get('n_test','?')})")
        lines.append(sep)

        # Determine metric columns based on task
        if task == "classification":
            header = f"  {'Model':<22s} {'Accuracy':>10s} {'Time(s)':>10s}"
            metric_key = "accuracy"
            formatter = lambda v: f"{v:.4f}"
        else:
            header = f"  {'Model':<22s} {'MSE':>10s} {'R2':>10s} {'Time(s)':>10s}"
            metric_key = "mse"

        lines.append(header)
        lines.append("  " + "-" * (width - 2))

        # Find v3 models vs sklearn models
        v3_models = {k: v for k, v in metrics.items() if k.startswith("V3")}
        sk_models = {k: v for k, v in metrics.items() if not k.startswith("V3")}

        # Print sklearn baselines first
        for model_name, model_metrics in sk_models.items():
            if "error" in model_metrics:
                lines.append(f"  {_pad_col(model_name, 22)} {'FAIL':>10s}")
                continue
            if task == "classification":
                acc = model_metrics.get("accuracy", 0)
                t = model_metrics.get("time", 0)
                lines.append(f"  {_pad_col(model_name, 22)} {acc:>10.4f} {t:>10.2f}")
            else:
                mse = model_metrics.get("mse", float("inf"))
                r2 = model_metrics.get("r2", -float("inf"))
                t = model_metrics.get("time", 0)
                lines.append(f"  {_pad_col(model_name, 22)} {mse:>10.4f} {r2:>10.4f} {t:>10.2f}")

        lines.append("  " + "-" * (width - 2))

        # Print v3 models
        for model_name, model_metrics in v3_models.items():
            if "error" in model_metrics:
                lines.append(f"  {_pad_col(model_name, 22)} {'FAIL':>10s}")
                continue
            if task == "classification":
                acc = model_metrics.get("accuracy", 0)
                t = model_metrics.get("time", 0)
                n_p = model_metrics.get("n_params", "?")
                lines.append(f"  {_pad_col(model_name, 22)} {acc:>10.4f} {t:>10.2f} "
                             f"(params={n_p})")
            else:
                mse = model_metrics.get("mse", float("inf"))
                r2 = model_metrics.get("r2", -float("inf"))
                t = model_metrics.get("time", 0)
                n_p = model_metrics.get("n_params", "?")
                lines.append(f"  {_pad_col(model_name, 22)} {mse:>10.4f} {r2:>10.4f} {t:>10.2f} "
                             f"(params={n_p})")

    lines.append("")
    lines.append("=" * width)
    return "\n".join(lines)


def generate_summary_ranking(results: Dict[str, Dict]) -> str:
    """
    Generate a summary ranking of models across all datasets.

    Returns a string with per-dataset best model and average rank.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  Best Model per Dataset")
    lines.append("=" * 70)

    for ds_name, entry in results.items():
        metrics = entry.get("metrics", {})
        task = entry.get("info", {}).get("task", "?")
        if task == "classification":
            metric_key = "accuracy"
            reverse = True
        else:
            metric_key = "mse"
            reverse = False

        sorted_models = sorted(
            [(k, v.get(metric_key, float("inf" if not reverse else -float("inf"))))
             for k, v in metrics.items() if metric_key in v],
            key=lambda x: x[1], reverse=reverse,
        )
        if sorted_models:
            best_name, best_val = sorted_models[0]
            lines.append(f"  {ds_name:<20s}: {best_name:<25s} ({best_val:.4f})")

    lines.append("=" * 70)
    return "\n".join(lines)
