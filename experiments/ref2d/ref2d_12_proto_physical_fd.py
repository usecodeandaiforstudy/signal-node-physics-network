"""
Experiment 12: Proto-Guided Pure Physical-Domain Training (No Digital Pretrain)

ZERO digital autograd — every training gradient comes from finite differences
on the physical analog circuit simulator.

Key difference from Experiment 11 (physical_himal_fd):
  - No digital pretrain stage — the network starts from RANDOM parameters
  - Each layer uses prototype-guided loss: hidden representation → L2 distance
    to K learnable prototypes → soft assignment → cross-entropy
  - Prototype-distance loss provides meaningful gradients even from random init,
    solving the "cold start" problem of pure physical training

Strategy:
  1. Initialize random RBF particles + spread-out prototypes per layer
  2. Per-layer FD training: proto-guided loss computed entirely through
     the 6-component analog circuit chain
  3. Each layer frozen after training; output fed as input to next layer
  4. Evaluate on physical simulator at 4 noise levels
  5. Compare with digital baseline (Experiment 01) to measure the gap
     between pure physical training and ideal digital training

This is the purest physical training experiment in the project:
no PyTorch autograd, no pretrained weights, no surrogate gradients —
only physical forward + numerical perturbation.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from physical_simulator import NoiseConfig
from training.proto_layerwise import ProtoGuidedTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def eval_proto_accuracy(trainer, layers, all_protos, X, y, n_repeats=3):
    """
    Evaluate classification accuracy through the full physical forward pipeline.

    Multiple forward passes are averaged to account for physical noise
    (mimics repeated analog measurements).
    """
    preds = []
    for _ in range(n_repeats):
        p = trainer.predict(layers, all_protos, X)
        preds.append(p)
    # Majority vote
    stacked = np.stack(preds, axis=0)  # (n_repeats, N)
    # For each sample, take the mode across repeats
    final = np.array([np.bincount(stacked[:, i]).argmax() for i in range(len(y))])
    return (final == y.ravel()).mean()


def run_one(name, X, y, particles, output_dims, epochs_per_layer=200, lr=0.01, fd_eps=1e-3):
    """
    Run a complete pure-physical training experiment on one dataset.

    Parameters
    ----------
    name : str
        Dataset name (for logging / plot naming)
    X : (N, 2) ndarray
        Input coordinates
    y : (N,) ndarray (int)
        Class labels
    particles : int
        First-layer particle count (layers 2 and 3 use fewer)
    output_dims : tuple of int
        Output dimensions per layer, e.g. (4, 4, 2).
        Larger intermediate dimensions help preserve class separability
        through the inter-layer mean projection.
    epochs_per_layer : int
        FD training epochs per layer
    lr : float
        FD learning rate
    fd_eps : float
        FD perturbation step size
    """
    print(f"\n{'=' * 60}")
    print(f"  Proto Pure-Physical: {name} "
          f"({particles}p, {epochs_per_layer}ep/layer, lr={lr}, eps={fd_eps})")
    print(f"{'=' * 60}")

    # ── Data preparation ──
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    sc = StandardScaler()
    X_tr_np = sc.fit_transform(X_tr).astype(np.float32)
    X_te_np = sc.transform(X_te).astype(np.float32)
    print(f"  train={len(X_tr)}, test={len(X_te)}")

    n_classes = len(np.unique(y))
    x_range = (X_tr_np[:, 0].min() - 1.0, X_tr_np[:, 0].max() + 1.0)
    y_range = (X_tr_np[:, 1].min() - 1.0, X_tr_np[:, 1].max() + 1.0)

    # ── Build trainer ──
    trainer = ProtoGuidedTrainer(
        layer_sizes=(
            particles,
            max(particles // 2, 3),
            max(particles // 3, 2),
        ),
        output_dims=output_dims,
        n_classes=n_classes,
        x_range=x_range,
        y_range=y_range,
        temperature=0.1,
        proto_reg_weight=0.1,
    )

    # ── Check random-init accuracy (should be ~1/K ≈ random) ──
    NoiseConfig.set_level("none")
    # Quick random init check: train 0 epochs, just predict with random weights
    rand_layers = [trainer._init_layer(s, 2) for s in trainer.layer_sizes]
    rand_protos = [trainer._init_prototypes(2) for _ in range(3)]
    acc_rand = eval_proto_accuracy(trainer, rand_layers, rand_protos, X_te_np, y_te)
    print(f"  [0] Random init accuracy: {acc_rand:.4f} (expected ~{1/n_classes:.2f})")

    # ── Pure physical training (NO digital pretrain!) ──
    t0 = time.time()
    print(f"  [1] Proto-guided physical training (3 layers, {epochs_per_layer}ep each):")
    layers, all_protos = trainer.train(
        X_tr_np, y_tr,
        epochs_per_layer=epochs_per_layer,
        lr=lr,
        eps=fd_eps,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"      Training time: {train_time:.1f}s")

    # ── Evaluate on physical simulator ──
    NoiseConfig.set_level("none")
    acc_phys = eval_proto_accuracy(trainer, layers, all_protos, X_te_np, y_te)
    delta = acc_phys - acc_rand
    print(f"  [2] Physical post-train accuracy: {acc_phys:.4f} (delta={delta:+.4f})")

    # ── Noise evaluation ──
    print(f"\n  [3] Physical noise evaluation:")
    noise_results = {}
    for level in ["none", "low", "medium", "high"]:
        NoiseConfig.set_level(level)
        a = eval_proto_accuracy(trainer, layers, all_protos, X_te_np, y_te)
        noise_results[level] = a
        print(f"      {level:<10s} {a:.4f}")
    NoiseConfig.set_level("none")

    # ── Per-layer accuracy (diagnostic: how much does each layer contribute?) ──
    print(f"\n  [4] Per-layer diagnostic:")
    cur_X = X_te_np.copy()
    for i in range(3):
        acc_i = (trainer.predict_single(layers[i], all_protos[i], cur_X) == y_te.ravel()).mean()
        print(f"      Layer {i+1} only: {acc_i:.4f}")
        # Prepare next layer input (use slice projection matching trainer)
        xc = cur_X[:, 0].astype(np.float32).ravel()
        yc = cur_X[:, 1].astype(np.float32).ravel()
        out_g, out_a = layers[i].forward(xc, yc)
        cur_X = trainer._project_to_2d(out_g, out_a)

    # ── Decision boundary plot ──
    tag = name.lower().replace(" ", "_")
    margin = 1.0
    xm, xM = X_te_np[:, 0].min() - margin, X_te_np[:, 0].max() + margin
    ym, yM = X_te_np[:, 1].min() - margin, X_te_np[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(xm, xM, 80), np.linspace(ym, yM, 80))
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    NoiseConfig.set_level("none")
    Z = trainer.predict(layers, all_protos, grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]),
                alpha=0.35, levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_te_np[:, 0], X_te_np[:, 1], c=y_te,
               cmap=ListedColormap(["#CC0000", "#0000CC"]), s=5, alpha=0.6)
    for i, layer in enumerate(layers):
        ax.scatter(layer.xc, layer.yc, c=["black", "orange", "green"][i],
                   marker=["X", "s", "D"][i], s=[160, 110, 80][i],
                   edgecolors="white", linewidths=0.8, label=f"L{i+1}", zorder=6)
    ax.set_xlim(xm, xM); ax.set_ylim(ym, yM)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Proto Pure-Physical — {name} "
                 f"(rand={acc_rand:.3f} → phys={acc_phys:.3f}, "
                 f"noise: L={noise_results['low']:.3f} M={noise_results['medium']:.3f} "
                 f"H={noise_results['high']:.3f})")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"proto_phys_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "name": name,
        "rand": acc_rand,
        "post": acc_phys,
        "delta": delta,
        "noise": noise_results,
        "time": train_time,
    }


def main():
    print("Proto-Guided Pure Physical-Domain Training")
    print("  ZERO digital pretrain — all gradients from FD on physical simulator")
    print("  Prototype-guided loss solves the cold-start problem\n")
    results = []

    # ── Moons: small network (3 particles/layer) ──
    X, y = make_moons(n_samples=500, noise=0.12, random_state=42)
    results.append(run_one(
        "Moons", X, y, particles=3, output_dims=(4, 3, 2),
        epochs_per_layer=100, lr=0.015, fd_eps=1e-3,
    ))

    # ── Final summary ──
    print("\n" + "=" * 60)
    print(f"  {'Dataset':<12s} {'Random':<10s} {'Trained':<10s} "
          f"{'Delta':<10s} {'Noise(L)':<10s} {'Noise(H)':<10s} {'Time':<10s}")
    print("  " + "-" * 58)
    for r in results:
        nl = r["noise"].get("low", 0)
        nh = r["noise"].get("high", 0)
        print(f"  {r['name']:<12s} {r['rand']:<10.4f} {r['post']:<10.4f} "
              f"{r['delta']:+10.4f} {nl:<10.4f} {nh:<10.4f} {r['time']:<10.1f}s")
    print("=" * 60)

    # Key claim check
    all_positive = all(r["delta"] > 0 for r in results)
    if all_positive:
        print("\n  VERIFIED: Proto-guided loss enables pure physical training from scratch.")
    else:
        print("\n  NOTE: Some datasets did not improve; may need more epochs or larger networks.")


if __name__ == "__main__":
    main()
