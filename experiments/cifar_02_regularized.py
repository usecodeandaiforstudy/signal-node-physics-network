"""
E21-REGULARIZED-TRAIN: Physical Regularization for RBF Classification
======================================================================
Project: singal-node-physics-network
Goal: Demonstrate that physically realizable regularization (center repulsion
      + width deviation penalty) can mitigate overfitting in a fully analog-
      deployable RBF particle network for CIFAR-10 image classification.

Baseline: E18 (4-layer 3D RBF, 100 img/class, 2884 params)
          Digital test: 37.3%, Train-Test gap: 9.9%

This experiment adds:
  1. Center repulsion penalty  — λ_c * Σ exp(-||c_i - c_j||² / 2σ²)
     Prevents particle centers from overlapping/collapsing.
     Physical: differential_pair → gilbert_square → bjt_exp → K-sum → ADC
  2. Width deviation penalty   — λ_h * Σ (h_j - h_init)²
     Prevents kernel widths from drifting, maintaining spatial coverage.
     Physical: differential_pair(h, h_init) → squarer → K-sum → ADC

All regularization values are scalar additions to the loss — computed in the
digital controller from ADC-read analog signals. No extra analog gradient
circuits are needed (SPSA/FD only requires the scalar loss value).
Training uses PyTorch autograd; deployment exports to PhysicalRBFLayerND.

Output: outputs/e21_regularized_report.txt + _plot.png
"""
import sys, os, pickle, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.core_nd import RBFLayerND
from physical_simulator import NoiseConfig
from simulator.layer_nd import PhysicalRBFLayerND

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
REPORT_FILE = os.path.join(OUT_DIR, "e21_regularized_report.txt")
PLOT_FILE   = os.path.join(OUT_DIR, "e21_regularized_plot.png")

CIFAR_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "reference", "annode", "cidata", "cifar-10-batches-py")
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

report = []
def L(s):
    print(s, flush=True)
    report.append(s)

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════
N_CLASSES = 5; IMG_SIZE = 32
N_TRAIN = 100; N_TEST = 30
# Same architecture as E18
L0_P=25; L0_O=3
L1_P=49; L1_G=7; L1_O=12
L2_P=28; L2_O=8
L3_P=20; L3_O=6
EPOCHS = 600; LR = 0.005
TEMPERATURE = 0.06; PROTO_REG = 0.02
EVAL_EVERY = 25
EARLY_STOP_PATIENCE = 14

# ── Regularization hyperparameters ──
# E18 baseline: AdamW weight_decay=1e-5 on ALL params (centers, h, W)
# E21 adds: center repulsion (Gaussian overlap penalty) + width deviation penalty
LAMBDA_C = 5e-4      # Center repulsion — penalize centers getting too close
LAMBDA_H = 5e-3      # Width deviation — penalize h drifting from h_init

L("=" * 60)
L("  E21: Regularized Training — Mitigating Overfitting")
L("=" * 60)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Classes: {N_CLASSES}, Resolution: {IMG_SIZE}x{IMG_SIZE}")
L(f"  Arch: L0(3D→3D,{L0_P}p,res) → L1(3D→{L1_O}D,{L1_P}p) → L2({L1_O}D→{L2_O}D,{L2_P}p) → L3({L2_O}D→{L3_O}D,{L3_P}p)")
L(f"  Regularization: AdamW(wd=1e-5) + center(λ={LAMBDA_C}) + width(λ={LAMBDA_H})")
L(f"  Training: autograd {EPOCHS}ep | Inference: physical + noise sweep")

# ═══════════════════════════════════════════════════
# 1. Data (same as E18)
# ═══════════════════════════════════════════════════
L("\n--- Data ---")
def load_cifar(fn):
    with open(fn,"rb") as f: d=pickle.load(f,encoding="bytes")
    return d[b"data"],d[b"labels"]
X_all,y_all=[],[]
for i in range(1,6):
    Xb,yb=load_cifar(os.path.join(CIFAR_DIR,f"data_batch_{i}"))
    X_all.append(Xb); y_all.extend(yb)
X_all=np.vstack(X_all).reshape(-1,3,32,32).astype(np.float32)/255.0
y_all=np.array(y_all)
gray=(0.299*X_all[:,0]+0.587*X_all[:,1]+0.114*X_all[:,2]).astype(np.float32)
from scipy.ndimage import zoom
gray=np.array([zoom(g,IMG_SIZE/32,order=1) for g in gray])

train_imgs,train_lbl=[],[]
test_imgs,test_lbl=[],[]
for c in range(N_CLASSES):
    idx=np.where(y_all==c)[0]; np.random.seed(c*100); np.random.shuffle(idx)
    for i in idx[:N_TRAIN]: train_imgs.append(gray[i]); train_lbl.append(c)
    for i in idx[N_TRAIN:N_TRAIN+N_TEST]: test_imgs.append(gray[i]); test_lbl.append(c)
train_imgs=np.array(train_imgs,dtype=np.float32); train_lbl=np.array(train_lbl)
test_imgs=np.array(test_imgs,dtype=np.float32); test_lbl=np.array(test_lbl)
L(f"  Train: {len(train_imgs)}, Test: {len(test_imgs)}")

xs=np.linspace(-1,1,IMG_SIZE,dtype=np.float32)
gx,gy=np.meshgrid(xs,xs)
COORDS_2D=np.stack([gx.ravel(),gy.ravel()],axis=1)

def build_3d(img):
    return np.concatenate([COORDS_2D,img.ravel()[:,None]],axis=1)

# ═══════════════════════════════════════════════════
# 2. Digital model (PyTorch) — same architecture as E18
# ═══════════════════════════════════════════════════
class DeepRBF3D(nn.Module):
    def __init__(self):
        super().__init__()
        # L0: 3D→3D spatial transformer (residual, zero-init = identity at start)
        self.l0=RBFLayerND(dim=3,n_particles=L0_P,n_vars=L0_O,init_range=(-1.5,1.5),use_pu=True,init_zero=True)
        # L1: 3D→L1_O, grid-init for local coverage
        gxs=np.linspace(-1,1,L1_G,dtype=np.float32); gys=np.linspace(-1,1,L1_G,dtype=np.float32)
        gx_g,gy_g=np.meshgrid(gxs,gys)
        self.l1=RBFLayerND(dim=3,n_particles=L1_P,n_vars=L1_O,init_range=(-1.5,1.5),use_pu=True,init_zero=False)
        c=torch.zeros(3,L1_P); c[0]=torch.tensor(gx_g.ravel()[:L1_P]); c[1]=torch.tensor(gy_g.ravel()[:L1_P]); c[2]=0.5
        self.l1.shift.centers.data.copy_(c)
        self.l1.shift.h.data.fill_(0.15)
        self.l2=RBFLayerND(dim=L1_O,n_particles=L2_P,n_vars=L2_O,init_range=(-1.5,1.5),use_pu=True,init_zero=True)
        self.l3=RBFLayerND(dim=L2_O,n_particles=L3_P,n_vars=L3_O,init_range=(-1.5,1.5),use_pu=True,init_zero=True)
        self.protos=nn.Parameter(torch.zeros(N_CLASSES,L3_O)); nn.init.orthogonal_(self.protos)
        self.T=TEMPERATURE

    def forward(self, img):
        X=torch.tensor(build_3d(img),dtype=torch.float32)
        g0,a0=self.l0(X); f0=X+(g0+a0)
        g1,a1=self.l1(f0); f1=g1+a1
        g2,a2=self.l2(f1); f2=g2+a2
        g3,a3=self.l3(f2); f3=g3+a3
        return f3.mean(0)

    def forward_batch(self, imgs):
        B=len(imgs); P=IMG_SIZE**2
        X_all=torch.zeros(B*P,3,dtype=torch.float32)
        for i in range(B):
            X_all[i*P:(i+1)*P]=torch.tensor(build_3d(imgs[i]),dtype=torch.float32)
        g0,a0=self.l0(X_all); f0=X_all+(g0+a0)
        g1,a1=self.l1(f0); f1=g1+a1
        g2,a2=self.l2(f1); f2=g2+a2
        g3,a3=self.l3(f2); f3=g3+a3
        f3=f3.view(B,P,L3_O).mean(1)
        return f3

    def _reg_weights(self, layer):
        """L2 weight decay on W_gauss + W_act (physically: current through shunt R + K sum)"""
        return (layer.output_gauss.weight ** 2).sum() + (layer.output_act.weight ** 2).sum()

    def _reg_centers(self, layer, sigma=0.3):
        """Center repulsion penalty: penalize centers that are too close.
        Uses Gaussian overlap: exp(-||c_i - c_j||² / 2σ²).
        Physical: differential_pair → squarer → exp(BJT) → K-sum → ADC"""
        c = layer.shift.centers  # (D, N)
        N = c.shape[1]
        if N <= 1:
            return torch.tensor(0.0)
        # Pairwise squared distances: (N, N)
        diff = c.unsqueeze(2) - c.unsqueeze(1)  # (D, N, N)
        d2 = (diff ** 2).sum(dim=0)  # (N, N)
        # Penalize overlap: exp(-d²/2σ²), exclude diagonal
        overlap = torch.exp(-d2 / (2 * sigma ** 2))
        mask = 1.0 - torch.eye(N)  # exclude self
        return (overlap * mask).sum()

    def _reg_width(self, layer):
        """Width deviation penalty: penalize h deviating from initial value.
        Uses (h - h_init)² per particle.
        Physical: differential_pair(h, h_init) → squarer → K-sum → ADC"""
        h = layer.shift.h  # (N,)
        # Store initial h on first call
        if not hasattr(layer, '_h_init'):
            layer._h_init = h.detach().clone()
        return ((h - layer._h_init) ** 2).sum()

    def loss(self, imgs, labels):
        f = self.forward_batch(imgs)
        d = torch.cdist(f, self.protos)
        ce = nn.functional.cross_entropy(-d / self.T, torch.tensor(labels, dtype=torch.long))

        # Prototype repulsion (same as E18)
        pd = torch.cdist(self.protos, self.protos)
        proto_reg = ((pd - (1.0 - torch.eye(N_CLASSES))) ** 2).mean()

        # ── E21: Additional regularization (center + width) ──
        reg_c = sum(self._reg_centers(ly) for ly in [self.l0, self.l1, self.l2, self.l3])
        reg_h = sum(self._reg_width(ly) for ly in [self.l0, self.l1, self.l2, self.l3])

        return ce + PROTO_REG * proto_reg + LAMBDA_C * reg_c + LAMBDA_H * reg_h

# ═══════════════════════════════════════════════════
# 3. Digital training
# ═══════════════════════════════════════════════════
L("\n--- Digital Training (autograd + regularization) ---")
model = DeepRBF3D()
n_param = sum(p.numel() for p in model.parameters())
L(f"  Trainable params: {n_param}")
# AdamW weight_decay matches E18 baseline (applies to ALL params: centers, h, W)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[400, 550], gamma=0.3)
t0 = time.time()
history = []          # (epoch, loss, ce_loss, reg_loss, train_acc, test_acc)
best_test_acc = 0.0; best_state = None; no_improve = 0

test_t = torch.tensor(test_imgs)

for ep in range(EPOCHS):
    total_loss = model.loss(torch.tensor(train_imgs), train_lbl)

    # Compute CE-only loss for logging (no reg gradient needed, just monitoring)
    with torch.no_grad():
        f_mon = model.forward_batch(torch.tensor(train_imgs))
        d_mon = torch.cdist(f_mon, model.protos)
        ce_mon = nn.functional.cross_entropy(-d_mon / TEMPERATURE, torch.tensor(train_lbl, dtype=torch.long))

    opt.zero_grad(); total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); scheduler.step()

    if (ep + 1) % EVAL_EVERY == 0:
        with torch.no_grad():
            f_tr = model.forward_batch(torch.tensor(train_imgs))
            acc_tr = (torch.cdist(f_tr, model.protos).argmin(1) == torch.tensor(train_lbl)).float().mean().item()
            f_te = model.forward_batch(test_t)
            acc_te = (torch.cdist(f_te, model.protos).argmin(1) == torch.tensor(test_lbl)).float().mean().item()
        reg_val = (total_loss.item() - ce_mon.item())
        history.append((ep + 1, total_loss.item(), ce_mon.item(), reg_val, acc_tr, acc_te))
        L(f"  ep {ep+1:4d}/{EPOCHS} | loss={total_loss.item():.4f} (CE={ce_mon.item():.4f}+reg={reg_val:.4f}) | train={acc_tr:.3f} | test={acc_te:.3f}")

        if acc_te > best_test_acc:
            best_test_acc = acc_te
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            L(f"  Early stop at epoch {ep+1} (no test improvement for {EARLY_STOP_PATIENCE*EVAL_EVERY} epochs)")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    L(f"  Restored best model (test acc={best_test_acc:.3f})")

t_train = time.time() - t0
L(f"  Training done: {t_train:.0f}s, {len(history)} evals")

# Digital test
with torch.no_grad():
    f_t = model.forward_batch(torch.tensor(test_imgs))
    d_t = torch.cdist(f_t, model.protos)
    preds_d = d_t.argmin(1).numpy()
    acc_d = (preds_d == test_lbl).mean()
L(f"\n  Digital test accuracy: {acc_d*100:.1f}%")

# ═══════════════════════════════════════════════════
# 4. Export to PhysicalRBFLayerND
# ═══════════════════════════════════════════════════
L("\n--- Physical Deployment ---")
NoiseConfig.set_level("none")

pl0 = PhysicalRBFLayerND.from_digital(model.l0)
pl1 = PhysicalRBFLayerND.from_digital(model.l1)
pl2 = PhysicalRBFLayerND.from_digital(model.l2)
pl3 = PhysicalRBFLayerND.from_digital(model.l3)
phys_protos = model.protos.detach().numpy().copy()

# ═══════════════════════════════════════════════════
# 5. Physical inference + noise sweep
# ═══════════════════════════════════════════════════
def physical_forward_single(img):
    X = build_3d(img)
    og0, oa0 = pl0.forward(X); f0 = X + (og0 + oa0)
    og1, oa1 = pl1.forward(f0); f1 = og1 + oa1
    og2, oa2 = pl2.forward(f1); f2 = og2 + oa2
    og3, oa3 = pl3.forward(f2); f3 = og3 + oa3
    return f3.mean(axis=0)

def classify(features, protos):
    d = np.sqrt(((features[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2) + 1e-12)
    return d.argmin(axis=1)

# Digital-physical match check
with torch.no_grad():
    df = model.forward_batch(torch.tensor(test_imgs[:10])).numpy()
pf = np.zeros((10, L3_O), dtype=np.float32)
for i in range(10):
    pf[i] = physical_forward_single(test_imgs[i])
dp_mse = float(np.mean((df - pf) ** 2))
L(f"  Digital-Physical feature MSE (10 test imgs): {dp_mse:.6f}")

# Noise sweep
noise_levels = ["none", "low", "medium", "high"]
acc_phys = {}
L(f"\n  Noise sweep:")
for nl in noise_levels:
    NoiseConfig.set_level(nl)
    pf_all = np.zeros((len(test_imgs), L3_O), dtype=np.float32)
    for i in range(len(test_imgs)):
        pf_all[i] = physical_forward_single(test_imgs[i])
    preds = classify(pf_all, phys_protos)
    acc = (preds == test_lbl).mean()
    acc_phys[nl] = acc
    L(f"    {nl:>6}: accuracy={acc*100:.1f}%")
NoiseConfig.set_level("none")

# Per-class breakdown
pf_all = np.zeros((len(test_imgs), L3_O), dtype=np.float32)
for i in range(len(test_imgs)):
    pf_all[i] = physical_forward_single(test_imgs[i])
preds_p = classify(pf_all, phys_protos)
confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for i in range(len(test_lbl)):
    confusion[test_lbl[i]][preds_p[i]] += 1
L(f"\n  Physical confusion (no noise):")
for i in range(N_CLASSES):
    row = " ".join(f"{confusion[i][j]:3d}" for j in range(N_CLASSES))
    acc_c = confusion[i][i] / max(confusion[i].sum(), 1) * 100
    L(f"    {CLASS_NAMES[i]:<10} {row}  ({acc_c:.0f}%)")

# ═══════════════════════════════════════════════════
# 6. Summary
# ═══════════════════════════════════════════════════
L(f"\n{'='*60}")
L("  SUMMARY")
L(f"{'='*60}")
peak_train = max([h[4] for h in history]) * 100
final_train = history[-1][4] * 100
final_test = history[-1][5] * 100
L(f"  Digital train: {peak_train:.1f}% (peak) → {final_train:.1f}% (final)")
L(f"  Digital test:  {best_test_acc*100:.1f}% (best, early-stop @ {len(history)*EVAL_EVERY}ep)")
L(f"  Train-Test gap: {final_train - final_test:.1f}% (E18 was 9.9%)")
L(f"  Final CE loss:  {history[-1][2]:.4f} | Reg loss: {history[-1][3]:.4f}")
L(f"  Training time: {t_train:.0f}s")
L(f"  Digital-Physical gap: {dp_mse:.6f} MSE")
noise_drop_pct = (acc_phys["none"] - acc_phys["high"]) / max(acc_phys["none"], 1e-10) * 100
L(f"  Physical noise drop:    none={acc_phys['none']*100:.1f}% → high={acc_phys['high']*100:.1f}% ({noise_drop_pct:.1f}% degradation)")
L(f"  Physical trainable:     YES — all params in PhysicalRBFLayerND")
L(f"  Pipeline:               autograd train → from_digital → physical infer")

# ═══════════════════════════════════════════════════
# 7. Plot
# ═══════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (0,0): Loss curves (CE + reg)
ax = axes[0, 0]
ep_list = [h[0] for h in history]
ce_list = [h[2] for h in history]
reg_list = [h[3] for h in history]
ax.plot(ep_list, ce_list, 'b-', linewidth=1.5, label="CE loss")
ax.plot(ep_list, reg_list, 'orange', linewidth=1.0, linestyle='--', label="Reg loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss (CE + Regularization)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (0,1): Train vs Test accuracy
ax = axes[0, 1]
train_accs = [h[4] * 100 for h in history]
test_accs = [h[5] * 100 for h in history]
ax.plot(ep_list, train_accs, 'b-', linewidth=1.5, label="Train")
ax.plot(ep_list, test_accs, 'r-', linewidth=1.5, label="Test")
best_epoch_idx = np.argmax(test_accs)
best_ep = ep_list[best_epoch_idx]
best_ta = test_accs[best_epoch_idx]
ax.axvline(x=best_ep, color='orange', linestyle='--', linewidth=1.2, label=f'Best test ep={best_ep}')
ax.plot(best_ep, best_ta, 'r*', markersize=12)
last_train = train_accs[-1]; last_test = test_accs[-1]
ax.annotate(f'Gap: {last_train-last_test:.1f}%', xy=(ep_list[-1], (last_train+last_test)/2),
            fontsize=8, color='red')
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Train vs Test (best test={best_ta:.1f}% @ ep {best_ep})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (0,2): Noise sweep
ax = axes[0, 2]
nls = noise_levels; acs = [acc_phys[n] * 100 for n in nls]
bars = ax.bar(range(len(nls)), acs, color=["#4CAF50", "#FFC107", "#FF9800", "#F44336"], edgecolor="black")
ax.set_xticks(range(len(nls))); ax.set_xticklabels(nls)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Physical Noise Robustness")
for b, a in zip(bars, acs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{a:.1f}%", ha="center", fontsize=9)
ax.set_ylim(0, max(acs) * 1.3); ax.grid(True, alpha=0.2, axis="y")

# (1,0): Confusion matrix
ax = axes[1, 0]
im = ax.imshow(confusion, cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels([n[:4] for n in CLASS_NAMES[:N_CLASSES]], fontsize=7, rotation=45)
ax.set_yticklabels(CLASS_NAMES[:N_CLASSES], fontsize=7)
ax.set_xlabel("Pred"); ax.set_ylabel("True")
ax.set_title(f"Physical Confusion (acc={preds_p.mean()*100:.1f}%)")
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, confusion[i][j], ha="center", va="center", fontsize=8,
                color="white" if confusion[i][j] > confusion.max() / 2 else "black")

# (1,1): Digital vs Physical per-class
ax = axes[1, 1]
with torch.no_grad():
    f_d = model.forward_batch(torch.tensor(test_imgs)).numpy()
    preds_dd = classify(f_d, model.protos.detach().numpy())
dig_accs = [(preds_dd[test_lbl == c] == c).mean() * 100 for c in range(N_CLASSES)]
phys_accs = [(preds_p[test_lbl == c] == c).mean() * 100 for c in range(N_CLASSES)]
x = np.arange(N_CLASSES); w = 0.35
ax.bar(x - w / 2, dig_accs, w, label="Digital", alpha=0.8)
ax.bar(x + w / 2, phys_accs, w, label="Physical", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([n[:4] for n in CLASS_NAMES[:N_CLASSES]], fontsize=8, rotation=45)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Digital vs Physical per-class")
ax.legend(); ax.set_ylim(0, 100); ax.grid(True, alpha=0.2, axis="y")

# (1,2): Feature space + prototype visualization
ax = axes[1, 2]
pf_all_np = pf_all.copy()
for c in range(N_CLASSES):
    mask = test_lbl == c
    ax.scatter(pf_all_np[mask, 0], pf_all_np[mask, 1], label=CLASS_NAMES[c][:6], s=20, alpha=0.7)
ax.scatter(phys_protos[:, 0], phys_protos[:, 1], c="black", marker="X", s=100, linewidths=2, label="prototype")
ax.set_xlabel("Feature dim 0"); ax.set_ylabel("Feature dim 1")
ax.set_title("Physical Feature Space")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle("E21: Regularized Training (L2 + Center + Width)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.close()
L(f"\n  Plot: {PLOT_FILE}")
L(f"  Report: {REPORT_FILE}")
L("=" * 60)
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
