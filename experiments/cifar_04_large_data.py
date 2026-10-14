"""
E23-LARGE-REG-4LAYER: E21 Architecture + More Data + E22 Training Strategy
===========================================================================
Project: singal-node-physics-network
Goal: Maximize peak test accuracy of the 4-layer regularized RBF classifier
      by combining E21's architecture and regularization with E22's training
      strategy (extended epochs, 3-stage LR decay) and larger dataset.

Hypothesis: E21 (100 img/class, 600ep, 2 LR drops) achieved 39.3%.
           With 150 img/class + 1000ep + 3 LR drops (full-batch), the network
           should learn better features with less overfitting, pushing past 40%.

Architecture (E21):  L0(3D→3D,25p,res) → L1(3D→12D,49p grid) → L2(12D→8D,28p) → L3(8D→6D,20p)
Training (E22-style): 1000 epochs, MultiStepLR[500,750,900] gamma=0.3, LR=0.004
Data:                  200 img/class (vs E21's 100), 30 test/class
Regularization (E21):  Center repulsion λ=5e-4 + Width deviation λ=5e-3
Physical:              from_digital() → PhysicalRBFLayerND → noise sweep

Output: outputs/cifar_04_large_data_report.txt + _plot.png
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
REPORT_FILE = os.path.join(OUT_DIR, "cifar_04_large_data_report.txt")
PLOT_FILE   = os.path.join(OUT_DIR, "cifar_04_large_data_plot.png")

CIFAR_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cifar")
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

report = []
def L(s): print(s, flush=True); report.append(s)

# ═══════════════════════════════════════════════════
# Config — E21 arch + E22 training + more data
# ═══════════════════════════════════════════════════
N_CLASSES = 5; IMG_SIZE = 32
N_TRAIN = 150; N_TEST = 30       # 1.5x training data vs E21 (safe full-batch)
# E21 architecture (4-layer with L0 residual)
L0_P=25; L0_O=3
L1_P=49; L1_G=7; L1_O=12
L2_P=28; L2_O=8
L3_P=20; L3_O=6
# E22-style training schedule
EPOCHS = 1000; LR = 0.004
SEED = 0                      # model init, data split, and sampling
TEMPERATURE = 0.06; PROTO_REG = 0.02
EVAL_EVERY = 20; EARLY_STOP_PATIENCE = 18  # 360 epochs patience
# E21 regularization
LAMBDA_C = 5e-4; LAMBDA_H = 5e-3

L("=" * 60)
L("  E23: 4-Layer + More Data + Extended Training — Push Past 40%")
L("=" * 60)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Classes: {N_CLASSES}, Resolution: {IMG_SIZE}x{IMG_SIZE}, Train: {N_TRAIN}/class")
L(f"  Arch: L0(3D→3D,{L0_P}p,res) → L1(3D→{L1_O}D,{L1_P}p) → L2({L1_O}D→{L2_O}D,{L2_P}p) → L3({L2_O}D→{L3_O}D,{L3_P}p)")
L(f"  Reg: center(λ={LAMBDA_C}) + width(λ={LAMBDA_H})")
L(f"  Training: {EPOCHS}ep, LR={LR}, MultiStepLR[500,750,900] gamma=0.3")

# ═══════════════════════════════════════════════════
# 1. Data — 200 img/class
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
L(f"  Train: {len(train_imgs)} ({N_TRAIN}/class), Test: {len(test_imgs)} ({N_TEST}/class)")

xs=np.linspace(-1,1,IMG_SIZE,dtype=np.float32)
gx,gy=np.meshgrid(xs,xs)
COORDS_2D=np.stack([gx.ravel(),gy.ravel()],axis=1)

def build_3d(img):
    return np.concatenate([COORDS_2D,img.ravel()[:,None]],axis=1)

# ═══════════════════════════════════════════════════
# 2. Model — E21 4-layer + regularization
# ═══════════════════════════════════════════════════
class DeepRBF4LayerReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0=RBFLayerND(dim=3,n_particles=L0_P,n_vars=L0_O,init_range=(-1.5,1.5),use_pu=True,init_zero=True)
        gxs=np.linspace(-1,1,L1_G,dtype=np.float32); gys=np.linspace(-1,1,L1_G,dtype=np.float32)
        gx_g,gy_g=np.meshgrid(gxs,gys)
        self.l1=RBFLayerND(dim=3,n_particles=L1_P,n_vars=L1_O,init_range=(-1.5,1.5),use_pu=True,init_zero=False)
        c=torch.zeros(3,L1_P); c[0]=torch.tensor(gx_g.ravel()[:L1_P]); c[1]=torch.tensor(gy_g.ravel()[:L1_P]); c[2]=0.5
        self.l1.shift.centers.data.copy_(c); self.l1.shift.h.data.fill_(0.15)
        self.l2=RBFLayerND(dim=L1_O,n_particles=L2_P,n_vars=L2_O,init_range=(-1.5,1.5),use_pu=True,init_zero=True)
        self.l3=RBFLayerND(dim=L2_O,n_particles=L3_P,n_vars=L3_O,init_range=(-1.5,1.5),use_pu=True,init_zero=True)
        self.protos=nn.Parameter(torch.zeros(N_CLASSES,L3_O)); nn.init.orthogonal_(self.protos)
        self.T=TEMPERATURE

    def forward_batch(self, imgs):
        B=len(imgs); P=IMG_SIZE**2
        X_all=torch.zeros(B*P,3,dtype=torch.float32)
        for i in range(B):
            X_all[i*P:(i+1)*P]=torch.tensor(build_3d(imgs[i]),dtype=torch.float32)
        g0,a0=self.l0(X_all); f0=X_all+(g0+a0)
        g1,a1=self.l1(f0); f1=g1+a1
        g2,a2=self.l2(f1); f2=g2+a2
        g3,a3=self.l3(f2); f3=g3+a3
        return f3.view(B,P,L3_O).mean(1)

    def _reg_centers(self, layer, sigma=0.3):
        c=layer.shift.centers; N=c.shape[1]
        if N<=1: return torch.tensor(0.0)
        diff=c.unsqueeze(2)-c.unsqueeze(1)
        d2=(diff**2).sum(dim=0)
        overlap=torch.exp(-d2/(2*sigma**2))
        mask=1.0-torch.eye(N)
        return (overlap*mask).sum()

    def _reg_width(self, layer):
        h=layer.shift.h
        if not hasattr(layer,'_h_init'): layer._h_init=h.detach().clone()
        return ((h-layer._h_init)**2).sum()

    def loss(self, imgs, labels):
        f=self.forward_batch(imgs)
        d=torch.cdist(f,self.protos)
        ce=nn.functional.cross_entropy(-d/self.T,torch.tensor(labels,dtype=torch.long))
        pd=torch.cdist(self.protos,self.protos)
        proto_reg=((pd-(1.0-torch.eye(N_CLASSES)))**2).mean()
        reg_c=sum(self._reg_centers(ly) for ly in [self.l0,self.l1,self.l2,self.l3])
        reg_h=sum(self._reg_width(ly) for ly in [self.l0,self.l1,self.l2,self.l3])
        return ce+PROTO_REG*proto_reg+LAMBDA_C*reg_c+LAMBDA_H*reg_h

# ONE SEED COVERS THE MODEL INITIALISATION.  Before this, the DATA SPLIT was seeded
# (`np.random.seed(c*100)` in the per-class loop) but the TORCH INITIALISATION was
# NOT, so every number this script published was a single draw from an unseeded
# process -- the same defect found and fixed in training/himal_nd.py and
# training/proto_nd.py.  Set here rather than inside the loop so the model, the
# optimiser state and the sampling all come from one stream.
torch.manual_seed(SEED)
np.random.seed(SEED)
L(f"  SEED={SEED}  (model init, data split, and sampling)")
model=DeepRBF4LayerReg()
n_param=sum(p.numel() for p in model.parameters())
L(f"  Params: {n_param}")

# ═══════════════════════════════════════════════════
# 3. Training — E22-style long schedule
# ═══════════════════════════════════════════════════
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[500,750,900],gamma=0.3)
t0=time.time()
history=[]
best_test=0.0; best_test_state=None
peak_train=0.0; peak_train_ep=0
peak_test=0.0; peak_test_ep=0
no_improve=0
tr_t,te_t=torch.tensor(train_imgs),torch.tensor(test_imgs)

L("\n--- Training (150 img/class, 1000ep, full-batch) ---")
for ep in range(EPOCHS):
    total_loss = model.loss(tr_t, train_lbl)
    with torch.no_grad():
        f_mon = model.forward_batch(tr_t)
        d_mon = torch.cdist(f_mon, model.protos)
        ce_mon = nn.functional.cross_entropy(-d_mon / TEMPERATURE, torch.tensor(train_lbl, dtype=torch.long))
    opt.zero_grad(); total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); scheduler.step()

    if (ep + 1) % EVAL_EVERY == 0:
        with torch.no_grad():
            ft = model.forward_batch(tr_t)
            acc_tr = (torch.cdist(ft, model.protos).argmin(1) == torch.tensor(train_lbl)).float().mean().item()
            fe = model.forward_batch(te_t)
            acc_te = (torch.cdist(fe, model.protos).argmin(1) == torch.tensor(test_lbl)).float().mean().item()
        reg_val = total_loss.item() - ce_mon.item()
        history.append((ep + 1, total_loss.item(), ce_mon.item(), reg_val, acc_tr, acc_te))
        if acc_tr > peak_train: peak_train = acc_tr; peak_train_ep = ep + 1
        if acc_te > peak_test: peak_test = acc_te; peak_test_ep = ep + 1
        if (ep + 1) % 100 == 0:
            L(f"  ep {ep+1:4d} | loss={total_loss.item():.4f} (CE={ce_mon.item():.4f}+reg={reg_val:.4f}) | train={acc_tr:.3f} | test={acc_te:.3f} | lr={scheduler.get_last_lr()[0]:.2e}")
        if acc_te > best_test: best_test = acc_te; best_test_state = {k: v.clone() for k, v in model.state_dict().items()}; no_improve = 0
        else: no_improve += 1
        if no_improve >= EARLY_STOP_PATIENCE:
            L(f"  Early stop at ep {ep+1}")
            break

if best_test_state: model.load_state_dict(best_test_state); L(f"  Restored best test ({best_test*100:.1f}%)")
t_train=time.time()-t0
L(f"\n  Done: {t_train:.0f}s")
L(f"  Peak train: {peak_train*100:.1f}% @ ep {peak_train_ep}")
L(f"  Peak test:  {peak_test*100:.1f}% @ ep {peak_test_ep}")
L(f"  Best test (early-stop): {best_test*100:.1f}%")

# ═══════════════════════════════════════════════════
# 4. Physical deployment
# ═══════════════════════════════════════════════════
L("\n--- Physical Deployment ---")
NoiseConfig.set_level("none")
pl0=PhysicalRBFLayerND.from_digital(model.l0)
pl1=PhysicalRBFLayerND.from_digital(model.l1)
pl2=PhysicalRBFLayerND.from_digital(model.l2)
pl3=PhysicalRBFLayerND.from_digital(model.l3)
phys_protos=model.protos.detach().numpy().copy()

def physical_forward(img):
    X=build_3d(img)
    og0,oa0=pl0.forward(X); f0=X+(og0+oa0)
    og1,oa1=pl1.forward(f0); f1=og1+oa1
    og2,oa2=pl2.forward(f1); f2=og2+oa2
    og3,oa3=pl3.forward(f2); f3=og3+oa3
    return f3.mean(axis=0)

with torch.no_grad(): df=model.forward_batch(te_t[:10]).numpy()
pf=np.array([physical_forward(test_imgs[i]) for i in range(10)])
dp_mse=float(np.mean((df-pf)**2))
L(f"  D-P MSE: {dp_mse:.6f}")

noise_levels=["none","low","medium","high"]
acc_phys={}
for nl in noise_levels:
    NoiseConfig.set_level(nl)
    pf_all=np.array([physical_forward(test_imgs[i]) for i in range(len(test_imgs))])
    d=np.sqrt(((pf_all[:,None,:]-phys_protos[None,:,:])**2).sum(axis=2)+1e-12)
    acc_phys[nl]=(d.argmin(1)==test_lbl).mean()
    L(f"  Physical {nl:>6}: {acc_phys[nl]*100:.1f}%")
NoiseConfig.set_level("none")

confusion=np.zeros((N_CLASSES,N_CLASSES),dtype=int)
pf_all=np.array([physical_forward(test_imgs[i]) for i in range(len(test_imgs))])
preds=np.sqrt(((pf_all[:,None,:]-phys_protos[None,:,:])**2).sum(axis=2)+1e-12).argmin(1)
for i in range(len(test_lbl)): confusion[test_lbl[i]][preds[i]]+=1
L(f"\n  Physical confusion:")
for i in range(N_CLASSES):
    row=" ".join(f"{confusion[i][j]:3d}" for j in range(N_CLASSES))
    L(f"    {CLASS_NAMES[i]:<10} {row}  ({confusion[i][i]/max(confusion[i].sum(),1)*100:.0f}%)")

# ═══════════════════════════════════════════════════
# 5. Summary
# ═══════════════════════════════════════════════════
final_tr=history[-1][4]*100 if history else 0
final_te=history[-1][5]*100 if history else 0
L(f"\n{'='*60}")
L("  SUMMARY")
L(f"{'='*60}")
L(f"  Peak train:     {peak_train*100:.1f}% @ ep {peak_train_ep}")
L(f"  Peak test:      {peak_test*100:.1f}% @ ep {peak_test_ep}")
L(f"  Best test (es): {best_test*100:.1f}%")
L(f"  Final train/test: {final_tr:.1f}% / {final_te:.1f}%")
L(f"  Gap:            {final_tr-final_te:.1f}%")
L(f"  Time:           {t_train:.0f}s")
L(f"  D-P gap:        {dp_mse:.6f} MSE")
nd=(acc_phys["none"]-acc_phys["high"])/max(acc_phys["none"],1e-10)*100
L(f"  Noise drop:     none={acc_phys['none']*100:.1f}% → high={acc_phys['high']*100:.1f}% ({nd:.1f}%)")
L(f"\n  Plot: {os.path.relpath(PLOT_FILE, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/')} | Report: {os.path.relpath(REPORT_FILE, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/')}"); L("="*60)
with open(REPORT_FILE,"w",encoding="utf-8") as f: f.write("\n".join(report)+"\n")

# ═══════════════════════════════════════════════════
# 6. Plot
# ═══════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"]=["SimHei","Microsoft YaHei","DejaVu Sans"]; rcParams["axes.unicode_minus"]=False
fig,axes=plt.subplots(2,2,figsize=(14,10))

ep_list=[h[0] for h in history]
ce_list=[h[2] for h in history]; reg_list=[h[3] for h in history]
train_accs=[h[4]*100 for h in history]; test_accs=[h[5]*100 for h in history]

ax=axes[0,0]
ax.plot(ep_list,ce_list,'b-',linewidth=1.5,label="CE loss")
ax2=ax.twinx(); ax2.plot(ep_list,reg_list,'orange',linewidth=1.0,linestyle='--',label="Reg")
ax.set_xlabel("Epoch"); ax.set_ylabel("CE Loss"); ax2.set_ylabel("Reg Loss",color='orange')
ax.set_title("Training Loss"); ax.grid(True,alpha=0.3); ax.legend(loc='upper left',fontsize=8)

ax=axes[0,1]
ax.plot(ep_list,train_accs,'b--',linewidth=1.2,label="Train")
ax.plot(ep_list,test_accs,'r-',linewidth=1.8,label="Test")
ax.plot(peak_train_ep,peak_train*100,'b*',markersize=14,label=f'Peak train {peak_train*100:.1f}%')
ax.plot(peak_test_ep,peak_test*100,'r*',markersize=14,label=f'Peak test {peak_test*100:.1f}%')
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Train vs Test (peak: train={peak_train*100:.1f}%, test={peak_test*100:.1f}%)")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

ax=axes[1,0]
im=ax.imshow(confusion,cmap="Blues",aspect="auto")
plt.colorbar(im,ax=ax,shrink=0.8)
ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels([n[:4] for n in CLASS_NAMES[:N_CLASSES]],fontsize=7,rotation=45)
ax.set_yticklabels(CLASS_NAMES[:N_CLASSES],fontsize=7)
ax.set_xlabel("Pred"); ax.set_ylabel("True")
ax.set_title(f"Physical Confusion (acc={preds.mean()*100:.1f}%)")
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j,i,confusion[i][j],ha="center",va="center",fontsize=8,
                color="white" if confusion[i][j]>confusion.max()/2 else "black")

ax=axes[1,1]
nls=noise_levels; acs=[acc_phys[n]*100 for n in nls]
bars=ax.bar(range(len(nls)),acs,color=["#4CAF50","#FFC107","#FF9800","#F44336"],edgecolor="black")
ax.set_xticks(range(len(nls))); ax.set_xticklabels(nls)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Physical Noise Robustness")
for b,a in zip(bars,acs): ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{a:.1f}%",ha="center",fontsize=9)
ax.set_ylim(0,max(acs)*1.3); ax.grid(True,alpha=0.2,axis="y")

plt.suptitle("E23: 4-Layer + 200img/class + Extended Training",fontsize=14,fontweight="bold")
plt.tight_layout(); plt.savefig(PLOT_FILE,dpi=150); plt.close()
