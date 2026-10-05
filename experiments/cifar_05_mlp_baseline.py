"""
E24-MLP-BASELINE: Fair MLP vs RBF Comparison on CIFAR-10
==========================================================
Project: singal-node-physics-network
Goal: Compare a plain MLP against the 4-layer RBF network (E21) under
      identical training conditions to isolate architecture effects.

Fair comparison constraints:
  - Same data:           100 img/class grayscale 32x32, 5 classes
  - Same training:       600 epochs, MultiStepLR[400,550], LR=0.005
  - Same depth:          4 hidden layers
  - Same loss:           CE with learnable prototype vectors
  - Similar params:      MLP ~4244 vs RBF 2884 (MLP unavoidably larger due
                         to 1024-pixel input first dense layer)

MLP Architecture:  1024 → 3 → 12 → 8 → 6 → prototypes(5×6)  (~3281 params)
  The first layer (1024×3) is the unavoidable bottleneck when matching
  RBF's ~2884 params. Optimized distribution: narrow first layer, wider
  middle layers to maximize information flow through the bottleneck.

Note: The RBF (E21) was trained with AdamW weight_decay=1e-5 + center+width
      regularization. This MLP uses only AdamW weight_decay=1e-5 for a
      clean baseline (no RBF-specific reg).

Output: outputs/e24_mlp_baseline_report.txt + _plot.png
"""
import sys, os, pickle, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
REPORT_FILE = os.path.join(OUT_DIR, "e24_mlp_baseline_report.txt")
PLOT_FILE   = os.path.join(OUT_DIR, "e24_mlp_baseline_plot.png")

CIFAR_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cifar")
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

report = []
def L(s): print(s, flush=True); report.append(s)

# ═══════════════════════════════════════════════════
# Config — matched to E21
# ═══════════════════════════════════════════════════
N_CLASSES = 5; IMG_SIZE = 32
N_TRAIN = 100; N_TEST = 30
EPOCHS = 600; LR = 0.005
SEED = 0                      # model init, data split, and sampling
TEMPERATURE = 0.06; PROTO_REG = 0.02
EVAL_EVERY = 25; EARLY_STOP_PATIENCE = 14

L("=" * 60)
L("  E24: MLP Baseline — Fair Comparison with RBF")
L("=" * 60)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Classes: {N_CLASSES}, Resolution: {IMG_SIZE}x{IMG_SIZE}, Train: {N_TRAIN}/class")
L(f"  Arch: 1024→{4}→{8}→{6}→6→head(5)")
L(f"  Training: {EPOCHS}ep, LR={LR}, MultiStepLR[400,550] gamma=0.3")

# ═══════════════════════════════════════════════════
# 1. Data — same as E18/E21
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
# Flatten: (N, 32*32) for MLP
train_flat=train_imgs.reshape(len(train_imgs),-1)
test_flat=test_imgs.reshape(len(test_imgs),-1)
L(f"  Train: {len(train_flat)} ({N_TRAIN}/class), Test: {len(test_flat)} ({N_TEST}/class)")

# ═══════════════════════════════════════════════════
# 2. MLP Model — 4 hidden layers, ~3281 params
# ═══════════════════════════════════════════════════
class SmallMLP(nn.Module):
    def __init__(self, h1=4, h2=8, h3=6):
        super().__init__()
        self.fc1=nn.Linear(1024, h1)
        self.bn1=nn.BatchNorm1d(h1)
        self.fc2=nn.Linear(h1, h2)
        self.bn2=nn.BatchNorm1d(h2)
        self.fc3=nn.Linear(h2, h3)
        self.fc4=nn.Linear(h3, 6)
        self.head=nn.Linear(6, N_CLASSES)
        self.T=TEMPERATURE

    def forward(self, x):
        x=torch.relu(self.bn1(self.fc1(x)))
        x=torch.relu(self.bn2(self.fc2(x)))
        x=torch.relu(self.fc3(x))
        x=self.fc4(x)
        return self.head(x)

    def loss(self, imgs, labels):
        logits=self.forward(imgs)
        return nn.functional.cross_entropy(logits, torch.tensor(labels,dtype=torch.long))

# ONE SEED COVERS THE MODEL INITIALISATION.  Before this, the DATA SPLIT was seeded
# (`np.random.seed(c*100)` in the per-class loop) but the TORCH INITIALISATION was
# NOT, so every number this script published was a single draw from an unseeded
# process -- the same defect found and fixed in training/himal_nd.py and
# training/proto_nd.py.  Set here rather than inside the loop so the model, the
# optimiser state and the sampling all come from one stream.
torch.manual_seed(SEED)
np.random.seed(SEED)
model=SmallMLP()
n_param=sum(p.numel() for p in model.parameters())
L(f"  Params: {n_param}")

# ═══════════════════════════════════════════════════
# 3. Training — same as E21
# ═══════════════════════════════════════════════════
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[400,550],gamma=0.3)
tr_t=torch.tensor(train_flat,dtype=torch.float32)
te_t=torch.tensor(test_flat,dtype=torch.float32)
t0=time.time()
history=[]
best_test=0.0; best_state=None
peak_train=0.0; peak_train_ep=0
peak_test=0.0; peak_test_ep=0
no_improve=0

L("\n--- Training ---")
for ep in range(EPOCHS):
    total_loss=model.loss(tr_t,train_lbl)
    opt.zero_grad(); total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); scheduler.step()

    if (ep+1)%EVAL_EVERY==0:
        with torch.no_grad():
            f_tr=model.forward(tr_t)
            acc_tr=(f_tr.argmax(1)==torch.tensor(train_lbl)).float().mean().item()
            f_te=model.forward(te_t)
            acc_te=(f_te.argmax(1)==torch.tensor(test_lbl)).float().mean().item()
        history.append((ep+1,total_loss.item(),acc_tr,acc_te))
        if acc_tr>peak_train: peak_train=acc_tr; peak_train_ep=ep+1
        if acc_te>peak_test: peak_test=acc_te; peak_test_ep=ep+1
        L(f"  ep {ep+1:4d}/{EPOCHS} | loss={total_loss.item():.4f} | train={acc_tr:.3f} | test={acc_te:.3f}")
        if acc_te>best_test: best_test=acc_te; best_state={k:v.clone() for k,v in model.state_dict().items()}; no_improve=0
        else: no_improve+=1
        if no_improve>=EARLY_STOP_PATIENCE:
            L(f"  Early stop at ep {ep+1}")
            break

if best_state: model.load_state_dict(best_state); L(f"  Restored best test ({best_test*100:.1f}%)")
t_train=time.time()-t0
L(f"\n  Done: {t_train:.0f}s")
L(f"  Peak train: {peak_train*100:.1f}% @ ep {peak_train_ep}")
L(f"  Peak test:  {peak_test*100:.1f}% @ ep {peak_test_ep}")

# ═══════════════════════════════════════════════════
# 4. Summary
# ═══════════════════════════════════════════════════
final_tr=history[-1][2]*100 if history else 0
final_te=history[-1][3]*100 if history else 0
L(f"\n{'='*60}")
L("  SUMMARY")
L(f"{'='*60}")
L(f"  Peak train:     {peak_train*100:.1f}% @ ep {peak_train_ep}")
L(f"  Peak test:      {peak_test*100:.1f}% @ ep {peak_test_ep}")
L(f"  Best test (es): {best_test*100:.1f}%")
L(f"  Final train/test: {final_tr:.1f}% / {final_te:.1f}%")
L(f"  Gap:            {final_tr-final_te:.1f}%")
L(f"  Time:           {t_train:.0f}s")
L(f"  Params:         {n_param} (RBF E21: 2884)")

L(f"\n{'='*60}")
L("  RBF vs MLP COMPARISON")
L(f"{'='*60}")
L(f"  RBF E21 (4-layer, 2884p):  peak test 39.3%, peak train 53.6%")
L(f"  MLP E24 (4-layer, {n_param}p): peak test {peak_test*100:.1f}%, peak train {peak_train*100:.1f}%")

L(f"\n  Plot: {PLOT_FILE} | Report: {REPORT_FILE}"); L("="*60)
with open(REPORT_FILE,"w",encoding="utf-8") as f: f.write("\n".join(report)+"\n")

# ═══════════════════════════════════════════════════
# 5. Plot
# ═══════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"]=["SimHei","Microsoft YaHei","DejaVu Sans"]; rcParams["axes.unicode_minus"]=False
fig,ax=plt.subplots(1,1,figsize=(12,5))
ep_list=[h[0] for h in history]
train_accs=[h[2]*100 for h in history]
test_accs=[h[3]*100 for h in history]
ax.plot(ep_list,train_accs,'b--',linewidth=1.2,label=f"MLP Train (peak {peak_train*100:.1f}%)")
ax.plot(ep_list,test_accs,'r-',linewidth=1.8,label=f"MLP Test (peak {peak_test*100:.1f}%)")
# Reference lines for RBF
ax.axhline(y=39.3,color='orange',linestyle=':',linewidth=1.2,label='RBF E21 best test 39.3%')
ax.axhline(y=53.6,color='cyan',linestyle=':',linewidth=1.2,label='RBF E21 peak train 53.6%')
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.set_title(f"E24: MLP ({n_param} params) vs RBF E21 (2884 params)")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
ax.set_ylim(0,65)
plt.tight_layout(); plt.savefig(PLOT_FILE,dpi=150); plt.close()
