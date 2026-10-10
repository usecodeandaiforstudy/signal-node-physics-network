"""
E05-NONLINEAR-REG v2: L2/L3 Contribution on Non-Linear Task
=============================================================
Tests whether deeper RBF layers contribute on a genuinely non-linear
regression task with feature interactions:
  y = sin(3*x1) + cos(2*x2) + 0.5*x1*x2 + noise

Upgrades from v1:
  - K-Means init (smart particle placement)
  - Anisotropic h + multi-scale k + Cauchy kernel
  - More particles (20/15/10) + more epochs (80)
  - Momentum FD + L2 regularization

Checks:
  1. Per-layer contribution (L1 only / L1+L2 / L3 full)
  2. Proto-weighted vs direct-h inference
  3. Learned feature scales and anisotropic h
"""
import sys, os, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from physical_simulator import NoiseConfig
NoiseConfig.set_level("none")
from physical_simulator.components import (
    differential_pair, gilbert_square, kirchhoff_sum,
    bjt_gaussian_kernel, bjt_log_amplifier, gilbert_divide,
    current_mode_normalize, memristor_mvm,
)
from training.himal_nd import HiMALND, _zero_pad_sum


# ═══════════════════════════════════════════════════════
#  v6-style RBF Layer (same as reg_06_california_v6)
# ═══════════════════════════════════════════════════════

class PhysicalRBFLayerNDv6:
    """3-channel: narrow(k_n) + wide(k_w) Gaussian + Cauchy, anisotropic h."""

    def __init__(self, scale, centers, h_aniso, k_n, k_w,
                 W_n, W_w, W_cauchy, use_pu=False):
        self.scale = np.asarray(scale, dtype=np.float32).flatten()
        self.centers = np.asarray(centers, dtype=np.float32)
        if self.centers.ndim == 1: self.centers = self.centers.reshape(1, -1)
        self.h_aniso = np.asarray(h_aniso, dtype=np.float32)
        if self.h_aniso.ndim == 1: self.h_aniso = self.h_aniso.reshape(1, -1)
        self.k_n = np.asarray(k_n, dtype=np.float32).flatten()
        self.k_w = np.asarray(k_w, dtype=np.float32).flatten()
        self.W_n = np.asarray(W_n, dtype=np.float32)
        self.W_w = np.asarray(W_w, dtype=np.float32)
        self.W_c = np.asarray(W_cauchy, dtype=np.float32)
        self.use_pu = use_pu
        self.D = self.centers.shape[0]; self.N = self.centers.shape[1]
        self.d_out = self.W_n.shape[0]

    def forward(self, X):
        # Feature scaling: programmable gain amplifier per input pin
        Xs = X.astype(np.float32) * self.scale.reshape(1, -1)
        d2 = None
        for i in range(self.D):
            dx = differential_pair(Xs[:, i], self.centers[i, :], self.h_aniso[i, :],
                                   mismatch_std=0.0, noise_std=0.0)
            d2_i = gilbert_square(dx, np.zeros_like(dx), mismatch_std=0.0, noise_std=0.0)
            d2 = d2_i if d2 is None else kirchhoff_sum(d2, d2_i)
        if d2 is None: d2 = np.zeros((Xs.shape[0], self.N), dtype=np.float32)

        z_base = bjt_gaussian_kernel(d2, k=1.0, temperature_voltage=1.0)
        log_z = bjt_log_amplifier(z_base, temperature_voltage=1.0)
        def ak(kv):
            lz = log_z * kv.reshape(1, -1).astype(np.float32)
            return np.exp(np.clip(lz, -50, 20))
        z_n = ak(self.k_n); z_w = ak(self.k_w)
        z_c = gilbert_divide(np.ones_like(d2, dtype=np.float32),
                             1.0+d2.astype(np.float32), temperature_voltage=1.0)
        if self.use_pu:
            z_n=current_mode_normalize(z_n,0.0); z_w=current_mode_normalize(z_w,0.0)
            z_c=current_mode_normalize(z_c,0.0)
        on=memristor_mvm(z_n,self.W_n,0.0); ow=memristor_mvm(z_w,self.W_w,0.0)
        oc=memristor_mvm(z_c,self.W_c,0.0)
        return on, ow, oc

    def fused_output(self, X):
        on, ow, oc = self.forward(X); return on+ow+oc

    def get_params_vector(self):
        return np.concatenate([
            self.scale, self.centers.flatten(), self.h_aniso.flatten(),
            self.k_n, self.k_w,
            self.W_n.flatten(), self.W_w.flatten(), self.W_c.flatten(),
        ]).astype(np.float32)

    def set_params_from_vector(self, vec):
        D,N,do=self.D,self.N,self.d_out; p=0
        self.scale=vec[p:p+D].copy(); p+=D
        self.centers=vec[p:p+D*N].reshape(D,N).copy(); p+=D*N
        self.h_aniso=vec[p:p+D*N].reshape(D,N).copy(); p+=D*N
        self.k_n=vec[p:p+N].copy(); p+=N
        self.k_w=vec[p:p+N].copy(); p+=N
        sz=do*N
        self.W_n=vec[p:p+sz].reshape(do,N).copy(); p+=sz
        self.W_w=vec[p:p+sz].reshape(do,N).copy(); p+=sz
        self.W_c=vec[p:p+sz].reshape(do,N).copy()


# ═══════════════════════════════════════════════════════
#  K-Means init
# ═══════════════════════════════════════════════════════

def kmeans_init(X, n_particles, n_vars, random_state=42):
    X_np = np.asarray(X, dtype=np.float32)
    N, D = X_np.shape
    n_clusters = min(n_particles, N)
    if n_clusters >= 2:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        km.fit(X_np)
        centers = km.cluster_centers_.T.astype(np.float32)
        labels = km.labels_
        h_aniso = np.zeros((D, n_clusters), dtype=np.float32)
        for k in range(n_clusters):
            mask = labels == k
            h_val = max(np.linalg.norm(X_np[mask]-centers[:,k], axis=1).mean()*2.0, 0.02) if mask.sum()>1 else 0.05
            h_aniso[:, k] = h_val
        if n_particles > n_clusters:
            extra = n_particles - n_clusters
            idx = np.random.choice(n_clusters, extra, replace=True)
            c_extra = centers[:, idx] + np.random.normal(0, 0.05, (D, extra)).astype(np.float32)
            h_extra = h_aniso[:, idx] * 0.8
            centers = np.concatenate([centers, c_extra], axis=1)
            h_aniso = np.concatenate([h_aniso, h_extra], axis=1)
    else:
        lo, hi = float(X_np.min()), float(X_np.max())
        dsize = hi-lo if hi>lo else 2.0
        centers = np.random.uniform(lo-0.5, hi+0.5, (D, n_particles)).astype(np.float32)
        h_aniso = np.full((D, n_particles), dsize/max(np.sqrt(n_particles),1.0), dtype=np.float32)
    h_aniso = np.maximum(h_aniso, 0.02)
    k_n = np.full(n_particles, 0.5, dtype=np.float32)
    k_w = np.full(n_particles, 0.3, dtype=np.float32)
    Wn = np.random.uniform(-0.3, 0.3, (n_vars, n_particles)).astype(np.float32)
    Ww = np.random.uniform(-0.3, 0.3, (n_vars, n_particles)).astype(np.float32)
    Wc = np.random.uniform(-0.3, 0.3, (n_vars, n_particles)).astype(np.float32)
    return PhysicalRBFLayerNDv6(
        scale=np.ones(D, np.float32), centers=centers, h_aniso=h_aniso,
        k_n=k_n, k_w=k_w, W_n=Wn, W_w=Ww, W_cauchy=Wc, use_pu=False)


# ═══════════════════════════════════════════════════════
#  Trainer (monkey-patched: K-Means + Momentum + L2)
# ═══════════════════════════════════════════════════════

def make_train_one_layer(beta=0.85, lam=0.0):
    def train_one_layer(trainer_self, data, n_particles, input_dim, n_vars,
                        epochs=200, lr=0.01, eps=1e-4, verbose=True):
        layer_A = kmeans_init(data["XA"], n_particles, n_vars, 42)
        layer_B = kmeans_init(data["XB"], n_particles, n_vars, 43)
        gamma = trainer_self.alignment_weight
        trainer_self._last_losses = {}

        def pair_loss(lA, lB, d, gv):
            Xa, ya = d["XA"], d["yA"]; Xb, yb = d["XB"], d["yB"]; Xt = d["XT"]
            hA = lA.fused_output(Xa); hB = lB.fused_output(Xb)
            if hA.shape[0]!=ya.shape[0]: hA=hA.T
            if hB.shape[0]!=yb.shape[0]: hB=hB.T
            m=min(hA.shape[1],ya.shape[1])
            fa=float(np.mean((hA[:,:m]-ya[:,:m])**2))
            fb=float(np.mean((hB[:,:m]-yb[:,:m])**2))
            al=0.0
            if len(Xt)>0:
                hTA=lA.fused_output(Xt); hTB=lB.fused_output(Xt)
                al=float(np.mean((hTA.ravel()-hTB.ravel())**2))
            trainer_self._last_losses={"fit_A":fa,"fit_B":fb,"align":al}
            l2=0.0
            if lam>0:
                l2=lam*(float(np.sum(lA.get_params_vector()**2))+
                        float(np.sum(lB.get_params_vector()**2)))
            return float(fa+fb+gv*al)+l2

        vel_A=np.zeros(layer_A.get_params_vector().size,dtype=np.float64)
        vel_B=np.zeros(layer_B.get_params_vector().size,dtype=np.float64)
        for ep in range(epochs):
            for _,layer,vel in [("A",layer_A,vel_A),("B",layer_B,vel_B)]:
                vec=layer.get_params_vector()
                grad=np.zeros_like(vec,dtype=np.float64)
                for j in range(len(vec)):
                    orig=vec[j]
                    vec[j]=orig+eps; layer.set_params_from_vector(vec.astype(np.float32))
                    Lp=pair_loss(layer_A,layer_B,data,gamma)
                    vec[j]=orig-eps; layer.set_params_from_vector(vec.astype(np.float32))
                    Lm=pair_loss(layer_A,layer_B,data,gamma)
                    vec[j]=orig; grad[j]=(Lp-Lm)/(2*eps)
                vel[:]=beta*vel-lr*grad
                vec=vec.astype(np.float64)+vel
                layer.set_params_from_vector(vec.astype(np.float32))
            if verbose and (ep+1)%40==0:
                loss=pair_loss(layer_A,layer_B,data,gamma)
                d=trainer_self._last_losses
                print(f"  Layer epoch {ep+1:3d}/{epochs} [FD+M] | loss={loss:.6f} | "
                      f"fit_A={d.get('fit_A',0):.4f} fit_B={d.get('fit_B',0):.4f} "
                      f"align={d.get('align',0):.4f}")
        return layer_A, layer_B
    return train_one_layer


# ═══════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════

print("=" * 60)
print("  E05 v2: Non-Linear Regression — Layer Contribution Test")
print("=" * 60)
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  y = sin(3x1) + cos(2x2) + 0.5*x1*x2 + noise")

np.random.seed(42)
N = 300
X = np.random.uniform(-2, 2, (N, 2)).astype(np.float32)
y = (np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 1]
     + np.random.normal(0, 0.15, N)).astype(np.float32)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr).astype(np.float32)
X_te = sc.transform(X_te).astype(np.float32)

mean_baseline = float(np.mean((y_te - y_tr.mean()) ** 2))
print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, Baseline MSE: {mean_baseline:.3f}")

# ── Config ──
LAYER_SIZES = (20, 15, 10)
OUTPUT_DIMS = (4, 3, 2)
EPOCHS, LR, EPS = 80, 0.02, 0.01
BETA, LAM = 0.85, 0.003
GAMMA = 0.04

print(f"  Particles: {LAYER_SIZES}, output_dims: {OUTPUT_DIMS}")
print(f"  Epochs: {EPOCHS}, lr={LR}, eps={EPS}, beta={BETA}, lam={LAM}, gamma={GAMMA}")
print()

# ═══ Train ═══
trainer = HiMALND(
    layer_sizes=LAYER_SIZES, output_dims=OUTPUT_DIMS,
    input_dim=2, init_range=(-3.0, 3.0),
    alignment_weight=GAMMA, n_protos=10, temperature=0.15,
    task="regression", use_spsa=False,
)
trainer.train_one_layer = make_train_one_layer(beta=BETA, lam=LAM).__get__(trainer, HiMALND)
trainer._forward_layer = types.MethodType(lambda self, l, X: l.fused_output(X), trainer)

t0 = time.time()
layers_A, layers_B, _ = trainer.train(
    X_tr, y_tr, split_method="median_x",
    epochs_per_layer=EPOCHS, lr=LR, eps=EPS, verbose=True,
)
print(f"\n  Training: {time.time()-t0:.0f}s, "
      f"{sum(l.get_params_vector().size for l in layers_A)*2} params (dual)")

# ═══ Evaluation ═══
def predict_layers(trainer, lA, lB, X, n_layers):
    hA, hB = X.copy(), X.copy()
    acc_A, acc_B = None, None
    for i in range(n_layers):
        hA_out = trainer._forward_layer(lA[i], hA)
        hB_out = trainer._forward_layer(lB[i], hB)
        if hA_out.shape[0] != hA.shape[0]: hA_out = hA_out.T
        if hB_out.shape[0] != hB.shape[0]: hB_out = hB_out.T
        pa = trainer._proto_predict(hA_out, trainer.proto_y_A[i], trainer.proto_h_A[i])
        pb = trainer._proto_predict(hB_out, trainer.proto_y_B[i], trainer.proto_h_B[i])
        if acc_A is None: acc_A, acc_B = pa.copy(), pb.copy()
        else: acc_A=_zero_pad_sum(acc_A,pa); acc_B=_zero_pad_sum(acc_B,pb)
        hA, hB = hA_out, hB_out
    return ((acc_A+acc_B)/2.0).ravel()

def mse(pred, y): return float(np.mean((pred-y)**2))

mse_l1 = mse(predict_layers(trainer, layers_A, layers_B, X_te, 1), y_te)
mse_l2 = mse(predict_layers(trainer, layers_A, layers_B, X_te, 2), y_te)
mse_l3 = mse(predict_layers(trainer, layers_A, layers_B, X_te, 3), y_te)

r2_l1 = 1-mse_l1/mean_baseline
r2_l2 = 1-mse_l2/mean_baseline
r2_l3 = 1-mse_l3/mean_baseline

print(f"\n  --- Layer Contribution ---")
print(f"  L1 only:         MSE={mse_l1:.4f}, R²={r2_l1:.4f}")
print(f"  L1+L2:           MSE={mse_l2:.4f}, R²={r2_l2:.4f}  (Δ L2={mse_l1-mse_l2:+.4f})")
print(f"  L1+L2+L3 (full): MSE={mse_l3:.4f}, R²={r2_l3:.4f}  (Δ L3={mse_l2-mse_l3:+.4f})")

# Proto-weighted vs direct-h (L1 only — wire-direct accumulation broken in old path)
def old_predict_l1(lA, lB, X):
    hoA = trainer._forward_layer(lA[0], X)
    hoB = trainer._forward_layer(lB[0], X)
    if hoA.shape[0]!=X.shape[0]: hoA=hoA.T
    if hoB.shape[0]!=X.shape[0]: hoB=hoB.T
    return ((hoA[:,0]+hoB[:,0])/2.0)

mse_old = mse(old_predict_l1(layers_A, layers_B, X_te), y_te)
print(f"\n  [Proto-weighted vs Direct-h L1]:")
print(f"  Direct-h L1: MSE={mse_old:.4f}, R²={1-mse_old/mean_baseline:.4f}")
print(f"  Proto-wtd L1: MSE={mse_l1:.4f}, R²={r2_l1:.4f}")
print(f"  Improvement: {(mse_old-mse_l1)/max(mse_old,1e-8)*100:.1f}%")

# Learned parameters
sc_vals = layers_A[0].scale
h_vals = layers_A[0].h_aniso.mean(axis=1)
print(f"\n  Learned: scale={[f'{s:.3f}' for s in sc_vals]}, "
      f"h per dim={[f'{h:.3f}' for h in h_vals]}")

# ═══ Summary ═══
print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"  R² L1-only:     {r2_l1:.4f}")
print(f"  R² L1+L2:       {r2_l2:.4f}")
print(f"  R² Full (L3):   {r2_l3:.4f}")
l2_gain = mse_l1 - mse_l2
l3_gain = mse_l2 - mse_l3
if l2_gain > 0.005 or l3_gain > 0.005:
    print(f"  >>> Depth helps on non-linear task: L2 ΔMSE={l2_gain:+.4f}, L3 ΔMSE={l3_gain:+.4f}")
else:
    print(f"  >>> Depth does NOT help — L1 captures all structure")
    print(f"  >>> Implication: even non-linear 2D functions are smooth enough")
    print(f"  >>> for single-layer RBF with enough particles")
print(f"{'='*60}")
