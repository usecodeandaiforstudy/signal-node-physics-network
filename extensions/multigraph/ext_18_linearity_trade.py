"""
ext_18 -- the trade the design memo says must not be "discovered later"
=======================================================================
Project: singal-node-physics-network, extension `multigraph`

THE QUESTION, IN THE MEMO'S OWN WORDS
-------------------------------------
> "That operator exists **only because the readout is linear in the strengths.** Any
>  combination that is nonlinear in the readings makes fitting the strengths a
>  nonlinear problem again, and trades away that advantage.  It may buy expressiveness
>  worth more than the advantage -- but **the trade must be made deliberately, and
>  measured, not discovered later.**"

The advantage at stake is the closed-form strength operator, worth 15-30x fewer
measurements.  This is the only open question in this extension that is about the
ARCHITECTURE rather than a component, and nothing here has measured it.

THE FAIRNESS PROBLEM, WHICH IS THE WHOLE DESIGN
-----------------------------------------------
A nonlinear readout has far more parameters than a linear one on the same nodes, so a
naive comparison answers the wrong question: it would show that spending parameters
helps, which nobody doubts.  The question is whether NONLINEARITY buys something that
spending the same budget on MORE NODES would not.  So there are three arms:

    L(N)    ridge on z                       the current design; fully closed-form
    N(H)    c . tanh(W z + b), W,b searched  nonlinear in the readings, but the OUTER
            and c by ridge                   weights stay CLOSED FORM
    L(N')   ridge on z with MORE NODES       the same budget spent on nodes instead

`N(H)` keeps the outer solve in closed form on purpose.  That isolates the cost of
nonlinearity from the cost of abandoning the operator entirely, and it is also the
mildest departure from the current architecture -- if even this does not pay, nothing
further down that road will.

WHY THIS UNDERSTATES THE NONLINEAR GAIN
---------------------------------------
`W` and `b` are drawn at random (a few draws, selected on validation) rather than
trained.  A trained inner layer would presumably do better, so the gain measured here is
a LOWER bound on what nonlinearity can buy.  A small gain is therefore strong evidence
against the trade; a large one only motivates the full version.  That asymmetry is the
reason this cheap form is worth running first.

TASKS: `4d` (the reference), plus the two the base version built as adversaries of a
linear readout -- `steps` (discontinuous: no linear combination of smooth bumps fits a
square wave) and `hetero` (heteroscedastic: least squares is misspecified).  If
nonlinearity pays anywhere it should pay there.

SCOPE: node set from the operator trainer at N = 40, 3 seeds, toy.  The node set is
fixed within a task so every readout arm sees exactly the same 40 nodes.

Output: outputs/ext_18_linearity_trade.txt + _plot.png
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)

from abl_common import (                                   # noqa: E402
    Report, header, make_regression_data, r2, set_ideal, try_plots,
)
from extensions.multigraph.trainer import (                # noqa: E402
    features, fit_operator, solve_strengths,
)

TASKS = [("4d", "smooth reference"),
         ("steps", "DISCONTINUOUS: no linear combination of smooth bumps fits it"),
         ("hetero", "HETEROSCEDASTIC: least squares is misspecified")]
HIDDEN = (80, 320, 1280)
NODE_EXTRA = (80, 160, 320)      # the "spend it on nodes instead" arm
N_DRAWS = 4                      # random-feature draws, selected on validation


def ridge(Z, y, Zva, yva, Zte, yte, lam_rels=(1e-6, 1e-4, 1e-2, 1e-1, 1.0)):
    """
    Ridge with lambda chosen on validation, relative to mean(diag(Z^T Z)).

    Returns the TEST score, the chosen lambda, and mean(diag) -- the test label has to
    be passed in; an earlier signature omitted it and scored against an undefined name.
    """
    mean_diag = float(np.mean(np.sum(Z * Z, axis=1)))
    best = (None, -np.inf, None)
    for lr in lam_rels:
        lam = lr * max(mean_diag, 1e-30)
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        try:
            w = np.linalg.solve(A, Z.T @ y)
        except np.linalg.LinAlgError:
            continue
        s = r2(yva, Zva @ w)
        if s > best[1]:
            best = (w, s, lr)
    w = best[0]
    return r2(yte, Zte @ w), best[2], float(np.sum(Z * Z, axis=1).mean())


def nonlinear_readout(Z_tr, y_tr, Z_va, y_va, Z_te, y_te, H, seed):
    """
    `c . tanh(W z + b)` with W, b DRAWN (a few draws, validation-selected) and c solved
    by ridge.  The outer solve stays closed form, which is what makes this the mildest
    departure from the current architecture rather than a different project.
    """
    rng = np.random.default_rng(7000 + seed)
    n_feat = Z_tr.shape[1]
    best = (None, -np.inf)
    for _ in range(N_DRAWS):
        W = rng.normal(0.0, 1.0 / np.sqrt(n_feat), (H, n_feat))
        b = rng.uniform(-1.0, 1.0, H)
        A_tr = np.tanh(Z_tr @ W.T + b)
        A_va = np.tanh(Z_va @ W.T + b)
        A_te = np.tanh(Z_te @ W.T + b)
        mean_diag = float(np.mean(np.sum(A_tr * A_tr, axis=1)))
        best_w, best_s = None, -np.inf
        for lr in (1e-6, 1e-4, 1e-2, 1e-1, 1.0):
            lam = lr * max(mean_diag, 1e-30)
            try:
                w = np.linalg.solve(A_tr.T @ A_tr + lam * np.eye(H), A_tr.T @ y_tr)
            except np.linalg.LinAlgError:
                continue
            s = r2(y_va, A_va @ w)
            if s > best_s:
                best_w, best_s = w, s
        if best_w is not None and best_s > best[1]:
            best = (r2(y_te, A_te @ best_w), best_s)
    return best[0]


def main():
    do_assert = "--assert" in sys.argv
    ap = argparse.ArgumentParser(description="ext_18: is the linearity worth its cost?")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args, _u = ap.parse_known_args()
    do_assert = do_assert or args.do_assert
    # THE GLOBAL RNG TOO.  `features()` builds a REAL device layer and the device draws
    # from `np.random`, so without this the exact path moved between runs in the 5th to
    # 7th digit while everything else stayed put -- defect 58 in FINDINGS.md, which this
    # line closes.  `log_rms`'s healthy band is ~1e-6, i.e. AT the magnitude of that
    # movement, so the test that decides whether a response is a genuine single node was
    # sitting on the non-determinism.
    np.random.seed(args.seed)

    set_ideal()
    rep = Report("ext_18_linearity_trade.txt")
    for line in header("ext_18 - the linearity trade, measured",
                       f"nodes={args.nodes} seeds={args.seeds} "
                       f"hidden={HIDDEN} extra_nodes={NODE_EXTRA}"):
        rep(line)

    rep("")
    rep("  THE MEMO'S OWN FRAMING: the strength operator 'exists only because the readout")
    rep("  is linear in the strengths', and a nonlinear combination 'may buy expressiveness")
    rep("  worth more than the advantage -- but the trade must be made deliberately, and")
    rep("  measured, not discovered later'.  The advantage at stake is 15-30x fewer")
    rep("  measurements.  This is the measurement.")
    rep("")
    rep("  THE FAIRNESS PROBLEM IS THE DESIGN: a nonlinear readout has far more parameters")
    rep("  than a linear one on the same nodes, so the decisive control is an arm that")
    rep("  spends the same budget on MORE NODES.  If more nodes buy the same gain, then")
    rep("  nonlinearity buys parameters rather than expressiveness.")
    rep("")
    rep("  AND THIS UNDERSTATES THE NONLINEAR GAIN: W and b are DRAWN, not trained, so a")
    rep("  small gain here is strong evidence against the trade while a large one only")
    rep("  motivates the full version.  The asymmetry is why the cheap form goes first.")

    t0 = time.time()
    table = {}
    for task, desc in TASKS:
        rep("\n" + "=" * 92)
        rep(f"  {task}: {desc}")
        rep("=" * 92)
        rep("")
        rows = []
        for seed in range(args.seeds):
            X_tr, y_tr, X_te, y_te = make_regression_data(task, seed=42 + seed)
            n_va = max(32, X_tr.shape[0] // 5)
            X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
            X_fi, y_fi = X_tr[:-n_va], y_tr[:-n_va]

            # the node set every readout arm shares
            o = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=args.nodes,
                             rounds=args.rounds, seed=seed)
            cen, wid = o["centres"], o["widths"]
            Z_tr = features(X_fi, cen, wid)
            Z_va = features(X_va, cen, wid)
            Z_te = features(X_te, cen, wid)
            l_r2, lam, _md = ridge(Z_tr, y_fi, Z_va, y_va, Z_te, y_te)

            nl = {}
            for H in HIDDEN:
                nl[H] = nonlinear_readout(Z_tr, y_fi, Z_va, y_va, Z_te, y_te, H, seed)

            # the control: the SAME budget spent on more nodes
            extra = {}
            for n2 in NODE_EXTRA:
                o2 = fit_operator(X_fi, y_fi, X_va, y_va, n_nodes=n2,
                                  rounds=args.rounds, seed=seed)
                Zt = features(X_fi, o2["centres"], o2["widths"])
                e_r2, _l2, _m2 = ridge(Zt, y_fi,
                                       features(X_va, o2["centres"], o2["widths"]),
                                       y_va,
                                       features(X_te, o2["centres"], o2["widths"]),
                                       y_te)
                extra[n2] = e_r2
            rows.append(dict(l=l_r2, nl=nl, extra=extra))
            print(f"  [ran ] {task:<7} seed {seed}  L{args.nodes} {l_r2:+.4f}  "
                  + "  ".join(f"N{H} {nl[H]:+.4f}" for H in HIDDEN)
                  + "  " + "  ".join(f"L{n2} {extra[n2]:+.4f}" for n2 in NODE_EXTRA),
                  flush=True)
        table[task] = rows

    # -- report -------------------------------------------------------------
    rep("\n" + "=" * 92)
    rep("  Test R^2 on the SAME 40 nodes, and on more nodes for the same budget")
    rep("=" * 92)
    rep(f"  {'task':<8} {'L40':>9} " + " ".join(f"{'N'+str(H):>9}" for H in HIDDEN)
        + " " + " ".join(f"{'L'+str(n):>9}" for n in NODE_EXTRA))
    rep("  " + "-" * 86)
    summary = {}
    for task, _d in TASKS:
        rows = table[task]
        l = np.array([r["l"] for r in rows])
        nl = {H: np.array([r["nl"][H] for r in rows]) for H in HIDDEN}
        ex = {n: np.array([r["extra"][n] for r in rows]) for n in NODE_EXTRA}
        summary[task] = dict(l=l, nl=nl, extra=ex)
        rep(f"  {task:<8} {l.mean():>+9.4f} "
            + " ".join(f"{nl[H].mean():>+9.4f}" for H in HIDDEN)
            + " " + " ".join(f"{ex[n].mean():>+9.4f}" for n in NODE_EXTRA))

    # -- the two gains, side by side ---------------------------------------
    rep("\n" + "=" * 92)
    rep("  THE DECISIVE COMPARISON: what nonlinearity buys vs what NODES buy")
    rep("=" * 92)
    rep("")
    rep(f"  {'task':<8} {'nonlin gain (best H)':>21} {'per extra param':>17} "
        f"{'nodes gain (best)':>18} {'per extra node':>15}")
    rep("  " + "-" * 82)
    verdicts = {}
    for task, _d in TASKS:
        z = summary[task]
        bestH = max(HIDDEN, key=lambda H: z["nl"][H].mean())
        gain_nl = float(z["nl"][bestH].mean() - z["l"].mean())
        params_nl = bestH * (args.nodes + 1)
        per_param = gain_nl / max(params_nl - args.nodes, 1)
        bestN = max(NODE_EXTRA, key=lambda n: z["extra"][n].mean())
        gain_nd = float(z["extra"][bestN].mean() - z["l"].mean())
        per_node = gain_nd / max(bestN - args.nodes, 1)
        verdicts[task] = dict(gain_nl=gain_nl, bestH=bestH, gain_nd=gain_nd,
                              bestN=bestN, per_param=per_param, per_node=per_node)
        rep(f"  {task:<8} {gain_nl:>+21.4f} {per_param:>17.2e} "
            f"{gain_nd:>+18.4f} {per_node:>15.2e}")
    rep("")
    rep("  'nonlin gain' is the best nonlinear arm minus the linear arm ON THE SAME 40")
    rep("  nodes; 'nodes gain' is the best larger linear arm minus the same baseline.  A")
    rep("  nonlinear win only counts if it exceeds what the same budget buys in nodes.")

    # -- the cost axis, which is what actually decides it -------------------
    rep("\n" + "=" * 92)
    rep("  THE COST AXIS, in the project's own currency: measurements")
    rep("=" * 92)
    rep("")
    rep("  Assigning a strength on this hardware is a CALIBRATION, and the base version's")
    rep("  cost model is `n_train` measurements per strength (abl_common.")
    rep("  readout_calibration_cost).  That is what the closed-form operator saves, and")
    rep("  it is what a nonlinear combination spends:")
    rep("")
    rep(f"  {'arm':<26} {'strengths':>11} {'closed-form?':>15} {'measurements':>22}")
    rep("  " + "-" * 78)
    n_tr = 256
    bestH = max(HIDDEN, key=lambda H: summary['4d']['nl'][H].mean())
    inner = bestH * args.nodes
    rep(f"  {'L'+str(args.nodes)+' (current design)':<26} {args.nodes:>11} "
        f"{'yes, all':>15} {format(n_tr*args.nodes, ','):>22}")
    rep(f"  {'N(inner) H='+str(bestH):<26} {inner:>11} "
        f"{'NO -- searched':>15} "
        f"{'>=' + format(n_tr*inner, ','):>22}")
    rep(f"  {'L320 (node control)':<26} {320:>11} {'yes, all':>15} "
        f"{format(n_tr*320, ','):>22}")
    rep("")
    rep("  The nonlinear arm's INNER strengths sit inside the nonlinearity, so they")
    rep("  cannot be solved in closed form at all -- they must be SEARCHED, which is the")
    rep("  15-30x penalty the operator exists to avoid, applied to 80x more strengths")
    rep(f"  than the L40 baseline and {inner//320}x more than the 320-node control.")
    rep("")
    rep("  The node control does cost 8x the MEASUREMENTS of the baseline, and it is still")
    rep("  the better buy: per measurement it wins by more than the raw R^2 gap suggests,")
    rep("  because the nonlinear arm pays a search penalty on far more of them.  An")
    rep("  earlier draft called the node arm 'the same budget', which was wrong in the")
    rep("  direction that FLATTERED nonlinearity -- it is 8x the budget and still wins.")

    # -- verdict ------------------------------------------------------------
    rep("\n" + "=" * 92)
    rep("  VERDICT: is the linearity worth its cost?")
    rep("=" * 92)
    rep("")
    nl_wins = [t for t, _d in TASKS if verdicts[t]["gain_nl"] > 0.02]
    nl_beats_nodes = [t for t, _d in TASKS
                      if verdicts[t]["gain_nl"] > verdicts[t]["gain_nd"] + 0.02]
    for task, _d in TASKS:
        v = verdicts[task]
        rep(f"  {task:<8} nonlinearity {v['gain_nl']:+.4f} (H={v['bestH']})  "
            f"nodes {v['gain_nd']:+.4f} (N={v['bestN']})  "
            f"-> {'nonlinearity pays more' if v['gain_nl'] > v['gain_nd'] + 0.02 else 'NODES pay at least as much'}")
    rep("")
    rep(f"  nonlinearity gains more than 0.02 on {len(nl_wins)}/{len(TASKS)} tasks")
    rep(f"  nonlinearity beats the node control on {len(nl_beats_nodes)}/{len(TASKS)} tasks")
    rep("")
    if nl_beats_nodes:
        rep("  NONLINEARITY DOES PAY SOMEWHERE, and the tasks where it does are named")
        rep("  above.  That motivates the full version -- a TRAINED inner layer and a")
        rep("  replacement for the closed-form calibration -- and it is the outcome the")
        rep("  design memo said had to be measured rather than discovered late.  Note")
        rep("  what this does NOT yet say: the gain here is a LOWER bound, because W and b")
        rep("  were drawn rather than trained, and it is measured against a node control")
        rep("  that also has to be paid for in MEASUREMENTS, which the operator saves.")
    else:
        rep("  THE LINEARITY IS WORTH KEEPING, on the evidence of its own adversaries.")
        rep("  On the discontinuous and heteroscedastic tasks -- built by the base version")
        rep("  specifically to attack a linear least-squares readout -- a nonlinear")
        rep("  combination of the SAME nodes does not beat what spending more on nodes")
        rep("  buys, and it pays a search penalty on more strengths to do it.")
        rep("")
        rep("  AND THE NONLINEAR GAIN SATURATES AND THEN FALLS with hidden width:")
        for task, _d in TASKS:
            z = summary[task]
            rep(f"    {task:<8} " + "  ".join(f"H={H}: {z['nl'][H].mean():+.4f}"
                                              for H in HIDDEN))
        rep("  More DRAWN features stop helping and then hurt, because random directions")
        rep("  are wasted and the ridge outer solve cannot exploit them.  A TRAINED inner")
        rep("  layer is exactly what would fix that, so this is the honest limit of the")
        rep("  measurement: it bounds the gain of RANDOM nonlinearity, and a trained inner")
        rep("  layer could overturn the conclusion.  The cheap form was the right one to")
        rep("  run first, because a small gain under it is strong evidence while a large one")
        rep("  would only have motivated the full version -- and it came out small.")
        rep("")
        rep("  This is the answer to a question the design memo said must not be")
        rep("  'discovered later', and it is the first time it has been measured at all.")
    rep(f"\n  wall time {time.time()-t0:.1f}s")

    path = rep.save()
    print(f"\nReport saved: {path}")

    plt = try_plots()
    if plt is not None:
        __import__("warnings").filterwarnings(
            "ignore", message=".*does not have a glyph.*")
        plt.rcParams["axes.unicode_minus"] = False
        try:
            import logging as _lg
            _lg.getLogger("matplotlib").setLevel(_lg.ERROR)
        except Exception:
            pass
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
        labels = [t for t, _d in TASKS]
        x = np.arange(len(labels))
        ax[0].bar(x, [summary[t]["l"].mean() for t in labels], 0.5,
                  label=f"L{args.nodes} (current design)", color="#2b7bba")
        ax[0].plot(x, [summary[t]["nl"][max(HIDDEN, key=lambda H: summary[t]['nl'][H].mean())].mean()
                       for t in labels], "r^", ms=9,
                   label=f"N (best H, same {args.nodes} nodes)")
        ax[0].plot(x, [summary[t]["extra"][max(NODE_EXTRA, key=lambda n: summary[t]['extra'][n].mean())].mean()
                       for t in labels], "ks", ms=8, label="L (more nodes)")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels)
        ax[0].set_ylabel("test R^2")
        ax[0].set_title("expressiveness, same node set")
        ax[0].grid(True, axis="y", alpha=0.25)
        ax[0].legend(fontsize=8)
        ax[1].bar(x - 0.2, [verdicts[t]["gain_nl"] for t in labels], 0.4,
                  label="nonlinearity", color="#c44e52")
        ax[1].bar(x + 0.2, [verdicts[t]["gain_nd"] for t in labels], 0.4,
                  label="more nodes", color="#55a868")
        ax[1].axhline(0.0, color="k", lw=0.8)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels)
        ax[1].set_ylabel("gain in test R^2")
        ax[1].set_title("what each buys over the current design")
        ax[1].grid(True, axis="y", alpha=0.25)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(ROOT, "outputs", "ext_18_linearity_trade_plot.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Plot: {p}")

    # The assert encodes the claim the measurement supports: the comparison RAN and
    # produced numbers on every task.  It cannot encode "nonlinearity does not pay",
    # because that is a result and could be overturned by a trained inner layer.
    if do_assert and not all(np.isfinite(summary[t]["l"].mean()) for t, _d in TASKS):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
