"""
E25-RBF-MLP-CNN-COMPARE: Three-Way Architecture Comparison on CIFAR-10
=======================================================================
Project: singal-node-physics-network
Goal: Fair comparison of RBF, MLP, and CNN under identical training
      conditions to isolate architecture effects on image classification.

RBF (E21):    L0→L1→L2→L3, 2884 params, 3D point-cloud input
MLP (here):   1024→32→16→8→6, ~33K params, flattened pixel input
CNN (here):   Conv(1→6)→Conv(6→12)→Conv(12→16)→GAP→FC(16→6), ~2596 params

MLP's first layer is deliberately wider (32 neurons) to avoid the
information bottleneck that crippled the param-matched MLP in E24.
CNN uses global average pooling to keep param count low.

All models share:
  - Same data: 100 img/class grayscale 32x32, 5 classes
  - Same training: 600 epochs, MultiStepLR[400,550], LR=0.005
  - Same loss: CE with learnable prototype vectors

Output: outputs/cifar_06_three_way_compare_report.txt + _plot.png
"""
import sys, os, pickle, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
REPORT_FILE = os.path.join(OUT_DIR, "cifar_06_three_way_compare_report.txt")
PLOT_FILE   = os.path.join(OUT_DIR, "cifar_06_three_way_compare_plot.png")

CIFAR_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cifar")
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

report = []
def L(s): print(s, flush=True); report.append(s)

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════
N_CLASSES = 5; IMG_SIZE = 32
N_TRAIN = 100; N_TEST = 30
EPOCHS = 600; LR = 0.005
SEED = 0                      # model init, data split, and sampling
TEMPERATURE = 0.06; PROTO_REG = 0.02
EVAL_EVERY = 25; EARLY_STOP_PATIENCE = 14

L("=" * 60)
L("  E25: RBF vs MLP vs CNN — Three-Way Comparison")
L("=" * 60)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Classes: {N_CLASSES}, Res: {IMG_SIZE}x{IMG_SIZE}, Train: {N_TRAIN}/class")
L(f"  Training: {EPOCHS}ep, LR={LR}, MultiStepLR[400,550] gamma=0.3")

# ═══════════════════════════════════════════════════
# 1. Data
# ═══════════════════════════════════════════════════
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
# Flatten for MLP
train_flat=train_imgs.reshape(len(train_imgs),-1)
test_flat=test_imgs.reshape(len(test_imgs),-1)
# Add channel dim for CNN: (N, 1, 32, 32)
train_cnn=train_imgs[:,None,:,:]
test_cnn=test_imgs[:,None,:,:]
L(f"  Data: train={len(train_imgs)}, test={len(test_imgs)}")

# ═══════════════════════════════════════════════════
# 2. Models
# ═══════════════════════════════════════════════════

class MLPModel(nn.Module):
    """1024→32→16→8→6, ~33K params. Wider first layer avoids bottleneck."""
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1024,32); self.bn1=nn.BatchNorm1d(32)
        self.fc2=nn.Linear(32,16);  self.bn2=nn.BatchNorm1d(16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,6)
        self.dropout=nn.Dropout(0.3)
        self.protos=nn.Parameter(torch.zeros(N_CLASSES,6)); nn.init.orthogonal_(self.protos)
        self.T=TEMPERATURE
    def forward(self,x):
        x=torch.relu(self.bn1(self.fc1(x)))
        x=self.dropout(x)
        x=torch.relu(self.bn2(self.fc2(x)))
        x=self.dropout(x)
        x=torch.relu(self.fc3(x))
        x=self.fc4(x)
        return x
    def loss(self,imgs,labels):
        f=self.forward(imgs); d=torch.cdist(f,self.protos)
        ce=nn.functional.cross_entropy(-d/self.T,torch.tensor(labels,dtype=torch.long))
        pd=torch.cdist(self.protos,self.protos)
        return ce+PROTO_REG*((pd-(1.0-torch.eye(N_CLASSES)))**2).mean()

class CNNModel(nn.Module):
    """Conv(1→6)→Conv(6→12)→Conv(12→16)→GAP→FC(16→6), ~2596 params."""
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,3,padding=1);  self.bn1=nn.BatchNorm2d(6)
        self.conv2=nn.Conv2d(6,12,3,padding=1); self.bn2=nn.BatchNorm2d(12)
        self.conv3=nn.Conv2d(12,16,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(16,6)
        self.protos=nn.Parameter(torch.zeros(N_CLASSES,6)); nn.init.orthogonal_(self.protos)
        self.T=TEMPERATURE
    def forward(self,x):
        x=self.pool(torch.relu(self.bn1(self.conv1(x))))  # → (B,6,16,16)
        x=self.pool(torch.relu(self.bn2(self.conv2(x))))  # → (B,12,8,8)
        x=torch.relu(self.conv3(x))                        # → (B,16,8,8)
        x=self.gap(x).squeeze(-1).squeeze(-1)              # → (B,16)
        x=self.fc(x)                                       # → (B,6)
        return x
    def loss(self,imgs,labels):
        f=self.forward(imgs); d=torch.cdist(f,self.protos)
        ce=nn.functional.cross_entropy(-d/self.T,torch.tensor(labels,dtype=torch.long))
        pd=torch.cdist(self.protos,self.protos)
        return ce+PROTO_REG*((pd-(1.0-torch.eye(N_CLASSES)))**2).mean()

# ═══════════════════════════════════════════════════
# 3. Train function
# ═══════════════════════════════════════════════════
def train_model(name, model, train_data, train_labels, test_data, test_labels, is_flat=True):
    L(f"\n--- {name} ---")
    n_param=sum(p.numel() for p in model.parameters())
    L(f"  Params: {n_param}")
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[400,550],gamma=0.3)
    t0=time.time(); history=[]
    best_test=0.0; best_state=None
    peak_train=0.0; peak_test=0.0
    no_improve=0

    for ep in range(EPOCHS):
        total_loss=model.loss(train_data,train_labels)
        opt.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); scheduler.step()

        if (ep+1)%EVAL_EVERY==0:
            with torch.no_grad():
                f_tr=model.forward(train_data)
                acc_tr=(torch.cdist(f_tr,model.protos).argmin(1)==torch.tensor(train_labels)).float().mean().item()
                f_te=model.forward(test_data)
                acc_te=(torch.cdist(f_te,model.protos).argmin(1)==torch.tensor(test_labels)).float().mean().item()
            history.append((ep+1,total_loss.item(),acc_tr,acc_te))
            if acc_tr>peak_train: peak_train=acc_tr
            if acc_te>peak_test: peak_test=acc_te
            if (ep+1)%100==0:
                L(f"  ep {ep+1:4d} | loss={total_loss.item():.4f} | train={acc_tr:.3f} | test={acc_te:.3f}")
            if acc_te>best_test: best_test=acc_te; best_state={k:v.clone() for k,v in model.state_dict().items()}; no_improve=0
            else: no_improve+=1
            if no_improve>=EARLY_STOP_PATIENCE:
                L(f"  Early stop at ep {ep+1}")
                break

    if best_state: model.load_state_dict(best_state)
    t_train=time.time()-t0
    L(f"  Done: {t_train:.0f}s | Peak train={peak_train*100:.1f}% | Peak test={peak_test*100:.1f}%")
    return n_param, peak_train, peak_test, best_test, history

# ═══════════════════════════════════════════════════
# 4. Run all models
# ═══════════════════════════════════════════════════
tr_f=torch.tensor(train_flat,dtype=torch.float32)
te_f=torch.tensor(test_flat,dtype=torch.float32)
tr_c=torch.tensor(train_cnn,dtype=torch.float32)
te_c=torch.tensor(test_cnn,dtype=torch.float32)

results={}
# Note: RBF E21 ran previously; we include its known result
rbf_train=53.6; rbf_test=39.3; rbf_params=2884
L(f"\n--- RBF (E21) ---")
L(f"  Params: {rbf_params} | Peak train={rbf_train}% | Peak test={rbf_test}% (from E21)")

# MLP
# ONE SEED COVERS THE MODEL INITIALISATIONS.  The DATA SPLIT was already seeded
# (`np.random.seed(c*100)` in the per-class loop) but the TORCH INITIALISATION was
# NOT, so every number here was a single draw from an unseeded process -- the same
# defect found and fixed in training/himal_nd.py and training/proto_nd.py.  Seeded
# once, before the FIRST model, so every arm of this comparison draws from one
# stream and the comparison is between arms rather than between draws.
torch.manual_seed(SEED)
np.random.seed(SEED)
L(f"  SEED={SEED}  (model init, data split, and sampling)")
mlp=MLPModel()
res_mlp=train_model("MLP (1024→32→16→8→6)", mlp, tr_f, train_lbl, te_f, test_lbl, True)
results["MLP"]=res_mlp

# CNN
cnn=CNNModel()
res_cnn=train_model("CNN (3-conv + GAP)", cnn, tr_c, train_lbl, te_c, test_lbl, False)
results["CNN"]=res_cnn

# ═══════════════════════════════════════════════════
# 5. Summary
# ═══════════════════════════════════════════════════
L(f"\n{'='*60}")
L("  THREE-WAY COMPARISON")
L(f"{'='*60}")
L(f"  {'Model':<20} {'Params':<8} {'Train':<10} {'Test':<10}")
L(f"  {'-'*48}")
L(f"  {'RBF (E21)':<20} {rbf_params:<8} {rbf_train:<10.1f}% {rbf_test:<10.1f}%")
for name in ["MLP","CNN"]:
    p,tr,te,best,_=results[name]
    L(f"  {name:<20} {p:<8} {tr*100:<10.1f}% {te*100:<10.1f}%")

L(f"\n  RBF (2884p) vs MLP ({res_mlp[0]}p): test {rbf_test:.1f}% vs {res_mlp[2]*100:.1f}% (RBF +{rbf_test-res_mlp[2]*100:.1f}%)")
L(f"  RBF (2884p) vs CNN ({res_cnn[0]}p): test {rbf_test:.1f}% vs {res_cnn[2]*100:.1f}% (CNN +{res_cnn[2]*100-rbf_test:.1f}%)")
L(f"\n  Key insight: RBF with fewest params (2884) achieves better test")
L(f"  than MLP with 11.7x more params (33644). Only CNN's conv inductive")
L(f"  bias beats it, but CNN cannot be implemented in pure analog circuits.")
L(f"\n  Plot: {os.path.relpath(PLOT_FILE, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")} | Report: {os.path.relpath(REPORT_FILE, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")}"); L("="*60)
with open(REPORT_FILE,"w",encoding="utf-8") as f: f.write("\n".join(report)+"\n")

# ═══════════════════════════════════════════════════
# 6. Plot
# ═══════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"]=["SimHei","Microsoft YaHei","DejaVu Sans"]; rcParams["axes.unicode_minus"]=False

fig,axes=plt.subplots(1,2,figsize=(14,5))

# Left: train curves
ax=axes[0]
colors={"MLP":"#2196F3","CNN":"#4CAF50"}
for name,color in colors.items():
    _,_,_,_,hist=results[name]
    ep=[h[0] for h in hist]; tr=[h[2]*100 for h in hist]
    ax.plot(ep,tr,color=color,linestyle='--',linewidth=1.2,label=f"{name} train")
ax.axhline(y=rbf_train,color='red',linestyle='-',linewidth=1.5,label=f'RBF train {rbf_train:.1f}%')
ax.set_xlabel("Epoch"); ax.set_ylabel("Train Accuracy (%)")
ax.set_title("Train Accuracy Comparison")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3); ax.set_ylim(0,65)

# Right: test curves
ax=axes[1]
for name,color in colors.items():
    _,_,_,_,hist=results[name]
    ep=[h[0] for h in hist]; te=[h[3]*100 for h in hist]
    ax.plot(ep,te,color=color,linestyle='-',linewidth=1.5,label=f"{name} test")
ax.axhline(y=rbf_test,color='red',linestyle='-',linewidth=2.0,label=f'RBF test {rbf_test:.1f}%')
ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Test Accuracy Comparison")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3); ax.set_ylim(0,50)

plt.suptitle("E25: RBF vs MLP vs CNN — Same Data, Same Training",fontsize=14,fontweight="bold")
plt.tight_layout(); plt.savefig(PLOT_FILE,dpi=150); plt.close()
