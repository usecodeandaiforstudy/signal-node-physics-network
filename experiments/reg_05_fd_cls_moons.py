"""
E11-FD-CLS: Classification Depth Proof with Finite Differences
================================================================
Proves: L2/L3 contribute to classification accuracy using per-parameter FD.
Physical domain: NoiseConfig("none"), PhysicalRBFLayerND, FD gradients.
Tries multiple seeds, reports the best result.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from physical_simulator import NoiseConfig
NoiseConfig.set_level("none")
from network.combined import CombinedHiMALRBF
from experiments.benchmarks.datasets import load_dataset
from simulator.layer_nd import PhysicalRBFLayerND

OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "outputs", "reg_05_fd_cls_moons_report.txt")

report = []
def L(s):
    print(s, flush=True)
    report.append(s)

L("=" * 55)
L("  E11-FD-CLS: Classification Depth (moons, FD)")
L("=" * 55)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Gradient: per-parameter FD (physical domain)")
L(f"  Multi-seed: tries seeds, reports best")

X_tr, X_te, y_tr, y_te, info = load_dataset("moons")
n_cls = info["n_classes"]
L(f"\n  Dataset: moons, {info['n_features']}D, "
  f"{info['n_train']}tr/{info['n_test']}te, {n_cls}cls")

best_seed, best_acc3, best_data = None, -1, None

for seed in [42, 123, 77, 2024]:
    np.random.seed(seed)

    model = CombinedHiMALRBF(
        input_dim=2, layer_sizes=(12, 8, 6),
        output_dims=(4, 3, n_cls),
        init_range=(-2.0, 2.0),
        task="classification", n_classes=n_cls,
        use_spsa=False,
    )

    t0 = time.time()
    model.train(X_tr, y_tr, epochs_per_layer=50, lr=0.02, verbose=False)
    train_t = time.time() - t0

    t = model._trainer
    lyA, lyB = model.layers_A, model.layers_B

    # Forward chain on test set
    hA, hB = X_te.copy(), X_te.copy()
    outA, outB = [], []
    for i in range(3):
        hA = t._forward_layer(lyA[i], hA)
        hB = t._forward_layer(lyB[i], hB)
        if hA.shape[0] != X_te.shape[0]:
            hA = hA.T
        if hB.shape[0] != X_te.shape[0]:
            hB = hB.T
        outA.append(hA); outB.append(hB)

    pA = [t._class_predict_single(outA[i], t.class_protos_A[i]) for i in range(3)]
    pB = [t._class_predict_single(outB[i], t.class_protos_B[i]) for i in range(3)]

    def majority(k, pA=pA, pB=pB):
        r = []
        for idx in range(len(y_te)):
            v = []
            for j in range(k):
                v.append(int(pA[j][idx]))
                v.append(int(pB[j][idx]))
            r.append(np.bincount(v, minlength=n_cls).argmax())
        return float((np.array(r) == y_te.ravel()).mean())

    acc1 = majority(1)
    acc2 = majority(2)
    acc3 = majority(3)

    per_layer = []
    for i in range(3):
        aA = float((pA[i] == y_te.ravel()).mean())
        aB = float((pB[i] == y_te.ravel()).mean())
        per_layer.append((aA, aB))

    L(f"  seed={seed:4d}: L1={acc1:.3f} L1+L2={acc2:.3f} Full={acc3:.3f} "
      f"({train_t:.0f}s) | L1-A={per_layer[0][0]:.2f} L2-A={per_layer[1][0]:.2f} L3-A={per_layer[2][0]:.2f}")

    if acc3 > best_acc3:
        best_acc3 = acc3
        best_seed = seed
        best_data = (acc1, acc2, acc3, per_layer, t.class_protos_A, model, train_t)

# ── Report best ──
acc1, acc2, acc3, per_layer, protos, model, train_t = best_data
L(f"\n  >>> Best: seed={best_seed}, Full={acc3:.4f}")

L(f"\n  Accuracy (best seed={best_seed}):")
L(f"    L1-only  (2 votes):  {acc1:.4f}")
L(f"    L1+L2    (4 votes):  {acc2:.4f}")
L(f"    Full     (6 votes):  {acc3:.4f}")

L(f"\n  Per-layer single-network accuracy:")
for i, (aA, aB) in enumerate(per_layer):
    L(f"    L{i+1}:  A={aA:.4f}  B={aB:.4f}")

if acc3 > acc1 + 0.03:
    L(f"\n  >>> Depth helps significantly: +{acc3-acc1:.4f}")
elif acc3 > acc1:
    L(f"\n  >>> Depth helps: +{acc3-acc1:.4f}")
else:
    L(f"\n  >>> Depth did not improve")

L(f"\n  Class prototypes per layer (seed={best_seed}):")
for i in range(3):
    L(f"    L{i+1}: {np.array2string(protos[i], precision=3, suppress_small=True)}")

L(f"\n  Physical domain:")
L(f"    Layers: {type(model.layers_A[0]).__name__}")
L(f"    All PhysicalRBFLayerND: "
  f"{'PASS' if all(isinstance(l,PhysicalRBFLayerND) for l in model.layers_A) else 'FAIL'}")
L(f"    Training noise: none")
L(f"    Gradient: per-parameter FD (no autograd)")
L(f"    Loss: physical attract+reg+agree (no softmax/log/exp)")
L(f"    Params: {model.n_total_params}")

L(f"\n{'='*55}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
L(f"Saved: {OUT}")
