"""
V3 E10: Noise Robustness — Physical Classification Under Circuit Non-Idealities

Trains once on clean data (noise=none), then evaluates at four noise levels
to measure accuracy degradation under realistic circuit conditions.

Noise levels (from physical_simulator.components.NoiseConfig):
  none:   ideal components, zero non-idealities
  low:    mismatch=0.01, noise=1e-7, drift=0.0005
  medium: mismatch=0.02, noise=1e-6, drift=0.001, stuck=0.01
  high:   mismatch=0.05, noise=1e-5, drift=0.005, stuck=0.05

Physical meaning:
  mismatch → transistor V_th variation between identical devices
  noise    → thermal / 1/f noise in amplifiers
  drift    → memristor conductance drift over read cycles
  stuck    → fraction of dead memristor cells
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from physical_simulator import NoiseConfig
from experiments.benchmarks.datasets import load_dataset
from network.combined import CombinedHiMALRBF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["moons", "iris"]
NOISE_LEVELS = ["none", "low", "medium", "high"]
N_EVAL = 3  # repeat evaluations per noise level (noise is stochastic)
EPOCHS = 30


def evaluate_noise(model, X_te, y_te, noise_level, n_repeats=N_EVAL):
    """Evaluate accuracy at a given noise level, averaged over n_repeats."""
    NoiseConfig.set_level(noise_level)
    accs = []
    for _ in range(n_repeats):
        pred = model.predict(X_te)
        acc = float((pred.ravel() == y_te.ravel()).mean())
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def main():
    print("=" * 65)
    print("  V3 E10: Noise Robustness")
    print("=" * 65)

    all_results = {}

    for ds in DATASETS:
        print(f"\n--- {ds} ---", flush=True)
        X_tr, X_te, y_tr, y_te, info = load_dataset(ds)
        nf, n_cls = info["n_features"], info["n_classes"]
        print(f"  {nf}D, {info['n_train']}tr/{info['n_test']}te, {n_cls}cls")

        # Train once, noise=none
        NoiseConfig.set_level("none")
        model = CombinedHiMALRBF(
            input_dim=nf,
            layer_sizes=(8, 5, 4),
            output_dims=(max(nf, 4), max(nf // 2, 3), n_cls),
            task="classification", n_classes=n_cls,
            use_spsa=True, spsa_avg=3,
        )
        t0 = time.time()
        model.train(X_tr, y_tr, epochs_per_layer=EPOCHS, lr=0.015, verbose=False)
        train_t = time.time() - t0
        print(f"  Trained: {train_t:.0f}s, {model.n_total_params} params")

        # Evaluate at each noise level
        results = {}
        for level in NOISE_LEVELS:
            mean_acc, std_acc = evaluate_noise(model, X_te, y_te, level)
            results[level] = {"mean": mean_acc, "std": std_acc}
            print(f"    {level:<8s}: acc={mean_acc:.4f} +/- {std_acc:.4f}", flush=True)

        all_results[ds] = results

    # ── Summary table ──
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    header = f"  {'Dataset':<12s}"
    for level in NOISE_LEVELS:
        header += f" {level:>16s}"
    header += "  degradation"
    print(header)
    print("  " + "-" * 58)

    for ds in DATASETS:
        r = all_results[ds]
        none_acc = r["none"]["mean"]
        line = f"  {ds:<12s}"
        for level in NOISE_LEVELS:
            line += f" {r[level]['mean']:>15.4f}"
        degradation = (r["high"]["mean"] - none_acc) / max(none_acc, 1e-10) * 100
        line += f"  {degradation:+.1f}%"
        print(line)

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(NOISE_LEVELS))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    for i, ds in enumerate(DATASETS):
        means = [all_results[ds][lvl]["mean"] for lvl in NOISE_LEVELS]
        stds = [all_results[ds][lvl]["std"] for lvl in NOISE_LEVELS]
        bars = ax.bar(x + i * width, means, width, label=ds, color=colors[i],
                      yerr=stds, capsize=4)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance (2-class)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(NOISE_LEVELS)
    ax.set_ylabel("Accuracy")
    ax.set_title("V3 Noise Robustness: Physical Classification")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    path = os.path.join(OUT_DIR, "reg_03_noise_robustness_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")

    NoiseConfig.set_level("none")
    print(f"\n{'=' * 65}")
    print("  DONE")


if __name__ == "__main__":
    main()
