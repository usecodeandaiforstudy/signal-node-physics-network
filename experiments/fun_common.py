"""
fun_common.py -- shared pieces for the functional-pipeline experiments (`fun_*`).

WHAT IS FIXED, AND WHAT IS LEFT TO THE DESIGNER
-----------------------------------------------
FIXED: the node.  A node is a parameterised function of a small patch,

    phi_j(p) = exp( -||p - c_j||^2 / h_j^2 ),    p = the k*k values of a k x k window

and it is admissible because `log phi_j` is DEGREE 2 in the patch entries, hence linear in
the natural parameters (c_j / h_j^2, 1 / h_j^2).  That linearity is what lets the geometry
of a node be recovered from its responses in closed form, and it is the same condition for
any window size.  A physical device whose response has this form can serve as a node; it
does not need to be simulated.

LEFT TO THE DESIGNER -- two hooks, at the bottom of this file:

    transform_input(imgs) -> field      what the node bank is applied to
    use_field(phi)        -> anything   what is done with the (nodes x positions) array

The two implementations in the `fun_*` scripts are two particular choices.  They are not
the only ones, and they are not a template to copy.

WHAT IS IN HERE
---------------
  constants, the four-state verdict, paired-comparison CI
  data: load_gray, draw_split (class-stratified fit/val), stratified_split
  primitives: patchify, responses, blur, centre_field
  NodeBank: centres + bandwidths -> the phi_j(x) array
  choose_centres_bandwidths: the sampling rule (kept numerically identical to the
      version the recorded numbers were produced with)
  fit_trend: layer 1's closed-form weights (reconstruct a blurred target)
  reduce_mean: one reducer for phi
  fit_readout: ridge to +-1 one-vs-rest, lambda chosen on validation
  illum, gamma_map, probe_dg: the transformations and the group probe
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "extensions", "multigraph"), ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trainer import solve_strengths                       # noqa: E402

# --------------------------------------------------------------------- constants
N_CLASSES = 5               # the CIFAR subset used throughout the cifar_* scripts
SIDE = 32
N_TRAIN = 100               # images per class used for fitting
N_TEST = 30                 # images per class used for scoring (the repo's default)
SHIPPED_BASELINE = 36.0     # cifar_02, seeded, single draw -- a reference, not a control
BW = 0.7                    # bandwidth = BW * median distance to the centre
T975 = (float("nan"), 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
        2.262, 2.228, 2.201, 2.179, 2.160, 2.145)
QUANT = 1.0 / 255.0         # one grey level; residuals below this are quantisation noise
SPLITS = ("tr", "va", "te")


# ------------------------------------------------------------------ the verdict
def state(m, half):
    """
    Four states.  A positive mean whose interval still contains zero is UNRESOLVED, and a
    negative mean whose interval contains zero is NULL; neither is a FAIL.  This rule lives
    in one place because it was written out inline in several scripts and regressed to a
    three-state version each time, which printed FAIL on results that were merely
    unresolved.
    """
    if not np.isfinite(half):
        return "UNRESOLVED"
    if m > 0.0:
        return "PASS" if (m - half) > 0.0 else "UNRESOLVED"
    return "FAIL" if (m + half) < 0.0 else "NULL"


def paired(a, b, scale=100.0):
    """Paired difference and its 95% half-width.  `a` and `b` must be seed-aligned."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan"), float("inf")
    d = scale * (a[:n] - b[:n])
    if n < 2:
        return float(d.mean()), float("inf")
    return float(d.mean()), float(T975[min(n - 1, len(T975) - 1)]
                                  * d.std(ddof=1) / np.sqrt(n))


# ------------------------------------------------------------------------ data
def load_gray(cifar_dir=None):
    """
    The 5-class grayscale subset: data_batch_1..5, 0.299/0.587/0.114, scaled to [0,1].

    NOTE ON THE SPLIT: `draw_split` takes the test images from the SAME training batches,
    which is what the cifar_* scripts do.  It is not a clean held-out CIFAR test set and
    numbers from it must not be compared with published CIFAR results.
    """
    d = cifar_dir or os.path.join(ROOT, "cifar")
    xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(d, "data_batch_%d" % i), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        xs.append(batch[b"data"])
        ys.extend(batch[b"labels"])
    x = np.vstack(xs).reshape(-1, 3, SIDE, SIDE).astype(np.float32) / 255.0
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    return gray.astype(np.float64), np.asarray(ys)


def draw_split(gray, labels, seed, n_train=N_TRAIN, n_test=N_TEST):
    """
    Per-class draw of `n_train + n_test` images.

    The indices come out CLASS BY CLASS, so a contiguous slice is a single class -- that
    is why `split_fit_val` below exists and why every script calls it.
    """
    rng = np.random.default_rng(100 * seed)
    tr, te = [], []
    for c in range(N_CLASSES):
        idx = np.where(labels == c)[0]
        idx = rng.permutation(idx)
        n = min(int(n_train), len(idx) - int(n_test))
        tr.extend(idx[:n])
        te.extend(idx[n:n + int(n_test)])
    return np.asarray(tr), np.asarray(te)


def split_fit_val(tr, labels, frac=0.8, seed=0):
    """Per-class fit/validation split.  A contiguous slice of `tr` is one class."""
    rng = np.random.default_rng(50000 + seed)
    fit, va = [], []
    for c in range(N_CLASSES):
        idx = tr[labels[tr] == c]
        idx = rng.permutation(idx)
        k = int(round(frac * len(idx)))
        k = max(1, min(k, len(idx) - 1))
        fit.extend(idx[:k])
        va.extend(idx[k:])
    return np.asarray(fit), np.asarray(va)


# ------------------------------------------------------------------ primitives
def patchify(field, k):
    """(B, S, S, c) -> (B, S-k+1, S-k+1, c*k*k).  This is the adjacency."""
    S0 = field.shape[1]
    outs = [field[:, di:S0 - k + 1 + di, dj:S0 - k + 1 + dj, :]
            for di in range(k) for dj in range(k)]
    return np.concatenate(outs, axis=-1)


def responses(Q, C, h):
    """
    exp(-||q - c||^2 / h^2) for every row of Q and every centre.

    Written as |q|^2 - 2 q.c + |c|^2 rather than a broadcasting subtraction: the broadcast
    form materialises (rows, centres, dim), which is hundreds of MB at these sizes.
    """
    qq = np.einsum("ij,ij->i", Q, Q)[:, None]
    cc = np.einsum("ij,ij->i", C, C)[None, :]
    d2 = qq - 2.0 * (Q @ C.T) + cc
    return np.exp(-np.maximum(d2, 0.0) / (h[None, :] ** 2))


def blur(imgs, sigma):
    """Separable Gaussian with reflect padding; sigma in pixels.  No scipy needed."""
    if sigma <= 0:
        return imgs
    rad = max(1, int(round(3.0 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = imgs
    for axis in (1, 2):
        pad = [(0, 0), (0, 0), (0, 0)]
        pad[axis] = (rad, rad)
        p = np.pad(out, pad, mode="reflect")
        acc = np.zeros_like(out)
        n = out.shape[axis]
        for i, w in enumerate(k):
            sl = [slice(None)] * 3
            sl[axis] = slice(i, i + n)
            acc += w * p[tuple(sl)]
        out = acc
    return out


def centre_field(imgs):
    """The centre pixel of every k x k patch: the grid a high-pass lives on."""
    return imgs[:, 1:-1, 1:-1]


# -------------------------------------------------------------------- the nodes
def choose_centres_bandwidths(P, n, rng, max_rows=3000):
    """
    Centres drawn from the training patch distribution; bandwidth from the distances.

    Kept numerically identical to the rule the recorded numbers were produced with: BOTH
    the centres and the distance subsample are drawn from the same `rng`, in this order.
    Changing the order, the subsample size, or the seeding shifts every recorded number.
    """
    n = int(max(1, min(n, len(P))))
    C = P[rng.choice(len(P), size=n, replace=False)].copy()
    rows = P[rng.choice(len(P), size=min(max_rows, len(P)), replace=False)]
    qq = np.einsum("ij,ij->i", rows, rows)[:, None]
    cc = np.einsum("ij,ij->i", C, C)[None, :]
    d = np.sqrt(np.maximum(qq - 2.0 * (rows @ C.T) + cc, 0.0))
    return C, np.maximum(np.median(d, axis=0) * BW, 1e-9)


class NodeBank:
    """A fixed bank of nodes.  `field` returns the (nodes x positions) response array."""

    def __init__(self, centres, bandwidths):
        self.centres = np.asarray(centres, dtype=np.float64)
        self.bandwidths = np.asarray(bandwidths, dtype=np.float64)

    @property
    def n(self):
        return int(self.centres.shape[0])

    def field(self, imgs, k=3, chunk=40):
        out = []
        for i in range(0, len(imgs), chunk):
            P = patchify(imgs[i:i + chunk][..., None], k)
            b, H, W, M = P.shape
            Z = responses(P.reshape(-1, M), self.centres, self.bandwidths)
            out.append(Z.reshape(b, H, W, -1))
        return np.concatenate(out, axis=0)

    def take(self, idx):
        return NodeBank(self.centres[idx], self.bandwidths[idx])


def reduce_mean(field):
    """Positional mean: the response field -> one number per node."""
    return field.mean(axis=(1, 2))


# ------------------------------------------------------- layer 1: closed-form w
def fit_trend(bank, imgs_fit, imgs_va, sigma,
              eps_grid=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1)):
    """
    Fit `w` so that sum_j w_j phi_j(u)(x) reconstructs a BLURRED image.

    Supervised?  No -- the target is G_sigma * u, so this needs no labels.  The ridge
    strength is chosen on validation R^2, computed from the accumulated normal equations
    (`RSS = y2 - 2 w.b + w.G.w`), so the pixels are never re-scanned.

    Fitting to `u` itself instead of a blurred target is kernel interpolation: with 100
    nodes centred on patches from the same images it reached R^2 = 0.999 and left a
    residual with 0.6% of the variance, which is not a trend.
    """
    def accum(imgs):
        n1 = bank.n
        G = np.zeros((n1, n1))
        b = np.zeros(n1)
        y2 = 0.0
        for i in range(0, len(imgs), 40):
            blk = imgs[i:i + 40]
            Z = bank.field(blk, k=3).reshape(-1, n1)
            t = centre_field(blur(blk, sigma)).reshape(-1)
            G += Z.T @ Z
            b += Z.T @ t
            y2 += float(t @ t)
        return G, b, y2

    G, b, y2 = accum(imgs_fit)
    Gv, bv, y2v = accum(imgs_va)
    scale = float(np.mean(np.diag(G))) + 1e-12
    best = (-np.inf, None, None)
    for e in eps_grid:
        try:
            w = np.linalg.solve(G + e * scale * np.eye(G.shape[0]), b)
        except np.linalg.LinAlgError:
            continue
        r2 = 1.0 - float(y2v - 2.0 * (w @ bv) + w @ (Gv @ w)) / max(y2v, 1e-12)
        if r2 > best[0]:
            best = (r2, w, e)
    return best[1], float(best[0]), float(best[2])


def trend_field(bank, w, imgs):
    """m(x) = sum_j w_j phi_j(u)(x), on the centre grid."""
    return (bank.field(imgs, k=3) @ w)


# ------------------------------------------------------------ layer 3: readout
def fit_readout(Zfit, yfit, Zva, yva):
    """Ridge to +-1 one-vs-rest targets, one lambda per class chosen on validation."""
    Yfit = np.where(yfit[:, None] == np.arange(N_CLASSES)[None, :], 1.0, -1.0)
    Yva = np.where(yva[:, None] == np.arange(N_CLASSES)[None, :], 1.0, -1.0)
    W = np.zeros((Zfit.shape[1], N_CLASSES))
    for c in range(N_CLASSES):
        w, _lr = solve_strengths(Zfit, Yfit[:, c], Zva, Yva[:, c])
        W[:, c] = w
    return W


def clean_features(F, sd_frac=1e-3, corr_max=0.999):
    """
    Drop near-constant and near-duplicate columns, then scale.

    Dropping before scaling is not cosmetic: a node that responds the same way to every
    image gives a nearly constant column, and scaling that to unit variance amplifies
    noise.  Duplicates are found with CENTRED correlation; an uncentred cosine keeps a
    single column when all activations are positive and similarly shaped.
    """
    X = F["tr"]
    sd = X.std(axis=0)
    top = float(sd.max()) if sd.size else 0.0
    live = sd > max(1e-12, sd_frac * top)
    idx = np.where(live)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(sd)) if sd.size else 0])
    Z = X[:, idx] / sd[idx]
    Zc = Z - Z.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(Zc, axis=0)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    Zn = Zc / nrm
    keep = [0]
    for j in range(1, Zn.shape[1]):
        c = np.abs(Zn[:, keep].T @ Zn[:, j])
        if float(c.max()) < corr_max:
            keep.append(j)
    idx = idx[np.asarray(keep)]
    return ({s: F[s][:, idx] / sd[idx] for s in SPLITS}, int(idx.size), int(X.shape[1]),
            idx)


# ---------------------------------------------------- transformations and probe
def illum(imgs, a, b):
    """I -> a*I + b, clipped to [0,1].  Returns (images, saturation_fraction)."""
    out = a * imgs + b
    sat = float(np.mean((out < 0.0) | (out > 1.0)))
    return np.clip(out, 0.0, 1.0), sat


def gamma_map(imgs, g):
    """I -> I**g on the clipped range.  Strictly increasing for g > 0."""
    return np.clip(imgs, 0.0, 1.0) ** g


def probe_dg(feature_fn, imgs, transform):
    """
    d_g = mean |f(u) - f(g.u)| / std(f), measured on a FITTED model.

    This is the cheapest way to decide whether a symmetry layer has anything to buy: a
    SMALL d_g means the model already ignores the transformation, so imposing it would
    only spend parameters.  A LARGE d_g is necessary but not sufficient -- the label must
    also be invariant to the transformation for the gap to be worth closing.
    """
    base = feature_fn(imgs)
    moved = feature_fn(transform(imgs))
    scale = float(np.std(base) + 1e-12)
    return float(np.mean(np.abs(moved - base)) / scale)


# --------------------------------------------------------------- the two hooks
def transform_input(imgs):
    """
    HOOK 1 -- what the node bank is applied to.

    Replace this to change what the nodes look at: the raw field, a high-pass residual, a
    normalised field, a transformed field, or something computed by an earlier bank.  The
    identity below is the do-nothing default.
    """
    return imgs


def use_field(field):
    """
    HOOK 2 -- what is done with the (nodes x positions) response array.

    Replace this to change how a result is produced from the responses.  The positional
    mean below is ONE choice; keeping the field (to drive a further bank or a physical
    process), pooling over regions, taking a second moment, or anything else belongs here.
    It is not a template to copy.
    """
    return reduce_mean(field)
