"""
fun_02_baseline_remeasure.py -- the repository's CNN, on the same splits, chosen on
VALIDATION instead of on the test set.

WHY THIS FILE EXISTS
--------------------
`experiments/cifar_06_three_way_compare.py` reports `peak_test`, the maximum test accuracy
over all 600 epochs, and restores the weights from that epoch (lines 178, 184, 186).  The
published 54.7% is therefore a maximum over 600 evaluations on the test set, with the model
selected at that maximum.  The run also used 500 training images and a 150-image test set,
while the `fun_*` pipeline uses 400 training images and 1000 test images.

None of that is visible in a side-by-side table, so this file measures the CNN the same way
the `fun_*` scripts are measured:

    same split function (fun_common.draw_split), same seeds
    same 400 training images
    epoch chosen on a 100-image VALIDATION split
    scored on the same 1000-image test set
    `peak_test` is still recorded, so the size of the selection effect is measured
    rather than argued about

The CNNModel below is copied verbatim from cifar_06; that module cannot be imported because
it runs the whole experiment at import time and has no __main__ guard.

    python experiments/fun_02_baseline_remeasure.py --data-seeds 2 --init-seeds 1
    python experiments/fun_02_baseline_remeasure.py --data-seeds 5 --init-seeds 2

RECORDED RESULT (5 data seeds x 2 initialisations = 10 runs, 200 test images per class)
    honest      51.14%  (sd 2.24)   test accuracy at the best-validation epoch
    peak_test   53.26%  (sd 1.40)   the original protocol
    final       50.92%  (sd 1.40)   last epoch, no selection at all
    inflation   +2.12 points, mean over runs
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "extensions", "multigraph"), ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fun_common import (N_CLASSES, SHIPPED_BASELINE, draw_split, load_gray,   # noqa
                        split_fit_val)

import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402

# constants copied from experiments/cifar_06_three_way_compare.py lines 47-52
EPOCHS = 600
LR = 0.005
TEMPERATURE = 0.06
PROTO_REG = 0.02
EVAL_EVERY = 25
EARLY_STOP_PATIENCE = 14

PUBLISHED = 54.7        # cifar_06, peak-test protocol, 150-image test set
FUNCTIONAL = 50.94      # fun_01_detrend, same split family (5 seeds)


class CNNModel(nn.Module):
    """Copied verbatim from cifar_06 lines 122-145.

    Conv(1->6) -> Conv(6->12) -> Conv(12->16) -> global average pool -> FC(16->6), with a
    prototype-distance classifier and a prototype-separation regulariser.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 6)
        self.protos = nn.Parameter(torch.zeros(N_CLASSES, 6))
        nn.init.orthogonal_(self.protos)
        self.T = TEMPERATURE

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.conv3(x))
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

    def loss(self, imgs, labels):
        f = self.forward(imgs)
        d = torch.cdist(f, self.protos)
        ce = nn.functional.cross_entropy(-d / self.T,
                                         torch.tensor(labels, dtype=torch.long))
        pd = torch.cdist(self.protos, self.protos)
        return ce + PROTO_REG * ((pd - (1.0 - torch.eye(N_CLASSES))) ** 2).mean()


def accuracy(model, x, y):
    with torch.no_grad():
        pred = torch.cdist(model.forward(x), model.protos).argmin(1)
    return float((pred == torch.tensor(y, dtype=torch.long)).float().mean())


def train_one(x_tr, y_tr, x_va, y_va, x_te, y_te, seed, verbose=False):
    """One run.  Selection is on validation; `peak_test` is recorded for comparison."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    model = CNNModel()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[400, 550], gamma=0.3)

    best_val, best_state, stale, best_ep = -1.0, None, 0, -1
    peak_test, final_test, done = 0.0, 0.0, 0
    for ep in range(EPOCHS):
        loss = model.loss(x_tr, y_tr)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        done = ep + 1
        if done % EVAL_EVERY == 0:
            acc_va = accuracy(model, x_va, y_va)
            acc_te = accuracy(model, x_te, y_te)
            final_test = acc_te
            peak_test = max(peak_test, acc_te)
            if acc_va > best_val:
                best_val, best_ep = acc_va, done
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= EARLY_STOP_PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    honest = accuracy(model, x_te, y_te)
    if verbose:
        print("      init %4d  stop %4d  best-val ep %4d (val %.3f)  honest %.4f  "
              "peak %.4f  final %.4f"
              % (seed, done, best_ep, best_val, honest, peak_test, final_test))
    return dict(honest=honest, peak=peak_test, final=final_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seeds", type=int, default=3,
                    help="must match the fun_* seeds for the splits to line up")
    ap.add_argument("--init-seeds", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=200,
                    help="test images per class; 200 gives 1000 test images")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gray, labels = load_gray()
    print("=" * 78)
    print("  fun_02_baseline_remeasure -- the same CNN, chosen on validation")
    print("=" * 78)
    print("  architecture copied from cifar_06 (3 conv + GAP, ~2596 parameters)")
    print("  %d epochs full batch, AdamW lr=%.3f, MultiStep[400,550]" % (EPOCHS, LR))
    print("  training on the same 400 images the fun_* pipeline fits on (80 per class,")
    print("  the remaining 20 per class held out for validation);")
    print("  epoch chosen on a 100-image validation split; scored on %d test images per"
          % args.n_test)
    print("  class.  The published %.1f%% used 500 training images and selected on TEST."
          % PUBLISHED)

    rows = []
    for ds in range(max(1, int(args.data_seeds))):
        tr, te = draw_split(gray, labels, ds, n_test=args.n_test)
        fit_idx, va_idx = split_fit_val(tr, labels, 0.8, seed=ds)

        def t3(A):
            return torch.tensor(A[:, None, :, :], dtype=torch.float32)

        x_tr, x_va, x_te = t3(gray[fit_idx]), t3(gray[va_idx]), t3(gray[te])
        y_tr, y_va, y_te = labels[fit_idx], labels[va_idx], labels[te]
        for k in range(max(1, int(args.init_seeds))):
            out = train_one(x_tr, y_tr, x_va, y_va, x_te, y_te, 1000 * ds + k,
                            args.verbose)
            rows.append(out)
        print("    data seed %d done (%d runs)" % (ds, len(rows)))

    def agg(key):
        v = 100.0 * np.array([r[key] for r in rows])
        return float(v.mean()), float(v.std())

    h, h_sd = agg("honest")
    p, p_sd = agg("peak")
    f, f_sd = agg("final")
    infl = 100.0 * np.array([r["peak"] - r["honest"] for r in rows])

    print("\n  %-10s %-9s %-9s %s" % ("metric", "mean", "sd", "meaning"))
    print("  %-10s %-9.2f %-9.2f %s" % ("honest", h, h_sd,
                                        "test at the best-VALIDATION epoch"))
    print("  %-10s %-9.2f %-9.2f %s" % ("peak_test", p, p_sd,
                                        "max test over training (cifar_06's protocol)"))
    print("  %-10s %-9.2f %-9.2f %s" % ("final", f, f_sd,
                                        "last epoch, no selection"))
    print("\n  selection effect: peak_test - honest = %+.2f points (mean over %d runs)"
          % (float(infl.mean()), len(rows)))

    print("\n  SAME-PROTOCOL COMPARISON (same splits, same training images, same test")
    print("  set, no selection on test):")
    print("    CNN          %.2f%% (+-%.2f, %d runs)" % (h, h_sd, len(rows)))
    print("    fun_01       %.2f%% (5 seeds)" % FUNCTIONAL)
    print("    difference   %+.2f points" % (h - FUNCTIONAL))
    print("\n  NOT like-for-like, quoted for context only:")
    print("    published CNN %.1f%%  (500 training images, peak-test selection, 150"
          " test images)" % PUBLISHED)
    print("    cifar_02      %.1f%%  (different architecture, single seeded draw)"
          % SHIPPED_BASELINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
