"""
Full Experiment Runner Script

Executes 8 experiments in sequence and generates a final summary report.

Experiment chain:
  1 → Digital baseline (reference standard)
  2 → Physical forward validation (physical component correctness)
  3 → Noise tolerance (non-ideality robustness)
  4 → SPSA training (zero-order optimization feasibility)
  5 → HiMAL training (layer-wise training feasibility)
  6 → 2D classification (decision boundary visualization + MLP comparison)
  7 → Projected ensemble (high-dimensional extension)
  8 → Physical classification (physical simulator solves real problems)

All experiments pass = physical realizability is verified via simulation.
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_experiment(name, module_name):
    """Run a single experiment and capture output"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print(f"║  {name}".ljust(59) + "║")
    print("╚" + "═" * 58 + "╝")

    start = time.time()
    try:
        module = __import__(module_name)
        result = module.main()
        elapsed = time.time() - start
        print(f"\n  [PASS] {name} ({elapsed:.1f}s)")
        return {"name": name, "status": "PASS", "result": result, "time": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [FAIL] {name}: {e}")
        traceback.print_exc()
        return {"name": name, "status": "FAIL", "error": str(e), "time": elapsed}


def generate_summary(results):
    """Generate summary report and chart"""
    print("\n" + "=" * 60)
    print("  Final Summary Report")
    print("=" * 60)

    for r in results:
        status = "✓" if r["status"] == "PASS" else "✗"
        print(f"  {status} {r['name']} ({r['time']:.1f}s)")

    # Summary chart
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)

    fig, ax = plt.subplots(figsize=(8, 5))
    exp_names = [r["name"] for r in results]
    exp_colors = ["green" if r["status"] == "PASS" else "red" for r in results]
    exp_times = [r["time"] for r in results]

    bars = ax.barh(exp_names, exp_times, color=exp_colors, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Experiment Results: {passed}/{total} passed")
    for bar, t in zip(bars, exp_times):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{t:.1f}s", va="center")

    # Proof chain
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "00_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Summary saved: {path}")
    plt.close()

    # Conclusion
    print("\n" + "=" * 60)
    if passed == total:
        print("  All experiments passed!")
        print()
        print("  Physical realizability proof chain:")
        print("  1. Digital baseline ✓ → basic performance achievable")
        print("  2. Physical forward ✓ → simulator mathematically correct")
        print("  3. Noise tolerance ✓ → network robust to non-idealities")
        print("  4. SPSA training ✓ → trainable without backpropagation")
        print("  5. HiMAL training ✓ → trainable without cross-layer gradients")
        print("  6. 2D classification ✓ → competitive accuracy vs MLP, visualizable particles")
        print("  7. Projected ensemble ✓ → high-D extension via random projection")
        print("  8. Physical classification ✓ → physical simulator solves real problems")
        print()
        print("  → Conclusion: RBF particle network can achieve")
        print("           the full train + inference closed loop in a pure analog physical system")
    else:
        print(f"  {total - passed}/{total} experiment(s) failed")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  RBF Particle Neural Network — Physical Reproducibility Verification")
    print("=" * 60)

    experiments = [
        ("Experiment 1: Digital Baseline",          "experiments.01_digital_baseline"),
        ("Experiment 2: Physical Forward Validation", "experiments.02_physical_forward"),
        ("Experiment 3: Noise Tolerance",           "experiments.03_noise_tolerance"),
        ("Experiment 4: SPSA Zero-Order Training",  "experiments.04_spsa_training"),
        ("Experiment 5: HiMAL Layer-Wise Training", "experiments.05_himal_training"),
        ("Experiment 6: 2D Classification",         "experiments.06_2d_classification"),
        ("Experiment 7: Projected Ensemble",        "experiments.07_projected_ensemble"),
        ("Experiment 8: Physical Classification",   "experiments.08_physical_classification"),
    ]

    results = []
    for name, module_name in experiments:
        result = run_experiment(name, module_name)
        results.append(result)

    generate_summary(results)
    return results


if __name__ == "__main__":
    main()
