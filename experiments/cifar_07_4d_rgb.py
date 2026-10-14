"""
cifar_07_4d_rgb.py — 4D 编码 RGB 实验：粒子生活在 (x, y, 通道, 强度) 空间

动机
----
cifar_01_baseline 在网络之外用固定亮度权重 (0.299R + 0.587G + 0.114B) 把 RGB
融合成灰度图，通道组合是人工先验，网络本身不具备"自适应通道组合"能力。

本实验把通道提升为 RBF 空间的第 4 个坐标（c ∈ {-1, 0, +1} 对应 R, G, B）：
  1. 每个像素变成 3 个 4D 点 (x, y, c, I)，每图点数 1024 → 3072
  2. 粒子中心在通道轴上可训练 → 训练后粒子在通道轴的分布 = 学到的通道注意力
  3. 测试期通道打乱（RGB 循环置换）验证网络是否真的使用了通道身份信息
  4. 数字-物理部署与噪声扫描与基线完全一致

与 cifar_01_baseline 严格可比：相同数据划分 / 层宽 / 训练超参 / 早停策略，
唯一差异是输入编码（3D 灰度 → 4D RGB）。

新增分析（相对基线）：
  - L1 粒子通道坐标直方图 + 每通道平均响应 vs 固定亮度权重
  - 通道打乱推理测试（特征 MSE + 精度下降）
  - 3D 基线结果自动对比（若 outputs/cifar_01_baseline_report.txt 存在）

用法：
    python experiments/cifar_07_4d_rgb.py           # 完整实验
    python experiments/cifar_07_4d_rgb.py --smoke   # 快速冒烟（小样本少轮次）
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.core_nd import RBFLayerND
from physical_simulator import NoiseConfig
from simulator.layer_nd import PhysicalRBFLayerND

# Every other experiment writes to the repository's own outputs/.  This one used to
# resolve a nested `experiments/outputs/` relative to itself, which meant a run from
# inside experiments/ produced a second outputs directory beside the scripts -- two
# files in a place nothing else uses and no document refers to.  That directory has
# been removed and this now matches the convention.
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
MATCHED = "--matched" in sys.argv   # 匹配容量消融：L1 每通道获得完整 7×7 空间覆盖
TAG = "cifar_07_4d_rgb_matched" if MATCHED else "cifar_07_4d_rgb"
REPORT_FILE = os.path.join(OUT_DIR, f"{TAG}_report.txt")
PLOT_FILE   = os.path.join(OUT_DIR, f"{TAG}_plot.png")
BASELINE_REPORT = os.path.join(OUT_DIR, "cifar_01_baseline_report.txt")

CIFAR_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cifar")
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

report = []
def L(s):
    print(s, flush=True)
    report.append(s)

# ═══════════════════════════════════════════════════
# Config —— 与 cifar_01_baseline 完全一致（除数据编码外）
# ═══════════════════════════════════════════════════
SMOKE = "--smoke" in sys.argv
N_CLASSES = 5; IMG_SIZE = 32
N_TRAIN = 20 if SMOKE else 100
N_TEST  = 10 if SMOKE else 30
# 4 层: L0(4D空间变换,残差) → L1(4D→12D) → L2 → L3
# 注意：L0 带残差（f0 = X + g0 + a0），输出维度必须等于输入维度 4
L0_P=25; L0_O=4
# 默认: 49 粒子在 (x,y) 上 7×7，通道轴循环分配（R/G/B 各约 1/3）
# --matched: 147 粒子 = 7×7 空间 × 3 通道，每通道完整空间覆盖
#            （与 3D 基线每通道的空间分辨率一致，检验"通道信息已存在但容量不足"假说）
L1_G=7
L1_P = L1_G * L1_G * 3 if MATCHED else 49
L1_O=12
L2_P=28; L2_O=8
L3_P=20; L3_O=6
EPOCHS = 40 if SMOKE else 600; LR = 0.005
SEED = 0                      # model init, data split, and sampling
TEMPERATURE = 0.06; PROTO_REG = 0.02
EVAL_EVERY = 5 if SMOKE else 25
EARLY_STOP_PATIENCE = 14
CHUNK = 64   # 4D 点数 ×3，分块前向以控制显存

L("=" * 60)
L("  C07: 4D Encoding RGB — RBF Particles in (x, y, channel, I)")
L("=" * 60)
L(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}" + ("  [SMOKE MODE]" if SMOKE else ""))
L(f"  Classes: {N_CLASSES}, Resolution: {IMG_SIZE}x{IMG_SIZE}x3 (RGB)")
L(f"  Encoding: pixel → 3 points (x, y, c, I), c ∈ {{-1,0,+1}} = R,G,B")
L(f"  L1 init: {'MATCHED — 7×7 spatial × 3 channels = 147 (每通道完整空间覆盖)' if MATCHED else 'standard — 7×7 spatial, channel cycling (49)'}")
L(f"  Arch: L0(4D→4D,{L0_P}p,res) → L1(4D→{L1_O}D,{L1_P}p) → L2({L1_O}D→{L2_O}D,{L2_P}p) → L3({L2_O}D→{L3_O}D,{L3_P}p)")
L(f"  Training: autograd {EPOCHS}ep (chunked B={CHUNK}) | deploy: physical + noise sweep")

# ═══════════════════════════════════════════════════
# 1. Data —— RGB 直接进网络，不做灰度化
# ═══════════════════════════════════════════════════
L("\n--- Data ---")
if not os.path.exists(os.path.join(CIFAR_DIR, "data_batch_1")):
    L("  [ERROR] CIFAR-10 data not found at cifar/")
    L("  Fix: Copy-Item 'reference\\annode\\cidata\\cifar-10-batches-py' 'cifar' -Recurse")
    L("  or:  python experiments/download_cifar.py")
    sys.exit(1)

import pickle
def load_cifar(fn):
    with open(fn, "rb") as f: d = pickle.load(f, encoding="bytes")
    return d[b"data"], d[b"labels"]

X_all, y_all = [], []
for i in range(1, 6):
    Xb, yb = load_cifar(os.path.join(CIFAR_DIR, f"data_batch_{i}"))
    X_all.append(Xb); y_all.extend(yb)
X_all = np.vstack(X_all).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
y_all = np.array(y_all)

train_imgs, train_lbl = [], []
test_imgs, test_lbl = [], []
for c in range(N_CLASSES):
    idx = np.where(y_all == c)[0]; np.random.seed(c * 100); np.random.shuffle(idx)
    for i in idx[:N_TRAIN]: train_imgs.append(X_all[i]); train_lbl.append(c)
    for i in idx[N_TRAIN:N_TRAIN + N_TEST]: test_imgs.append(X_all[i]); test_lbl.append(c)
train_imgs = np.array(train_imgs, dtype=np.float32); train_lbl = np.array(train_lbl)
test_imgs = np.array(test_imgs, dtype=np.float32); test_lbl = np.array(test_lbl)
L(f"  Train: {len(train_imgs)}, Test: {len(test_imgs)} (RGB, no grayscale)")

xs = np.linspace(-1, 1, IMG_SIZE, dtype=np.float32)
gx, gy = np.meshgrid(xs, xs)
COORDS_2D = np.stack([gx.ravel(), gy.ravel()], axis=1)          # (1024, 2)
CH_VALS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)           # R, G, B

def build_4d(img_rgb):
    """(3,32,32) → (3072, 4): 每像素生成 3 个点 (x, y, c, I)"""
    pts = []
    for ci in range(3):
        pts.append(np.concatenate([
            COORDS_2D,
            np.full((IMG_SIZE * IMG_SIZE, 1), CH_VALS[ci], dtype=np.float32),
            img_rgb[ci].ravel()[:, None].astype(np.float32),
        ], axis=1))
    return np.concatenate(pts, axis=0)

# ═══════════════════════════════════════════════════
# 2. Digital model (PyTorch)
# ═══════════════════════════════════════════════════
class DeepRBF4D(nn.Module):
    def __init__(self):
        super().__init__()
        # L0: 4D→4D 空间变换（残差，zero-init = 起始恒等）
        self.l0 = RBFLayerND(dim=4, n_particles=L0_P, n_vars=L0_O,
                             init_range=(-1.5, 1.5), use_pu=True, init_zero=True)
        # L1: 4D→L1_O，(x,y) 网格覆盖 + 通道轴初始化
        gxs = np.linspace(-1, 1, L1_G, dtype=np.float32)
        gx_g, gy_g = np.meshgrid(gxs, gxs)
        base_xy = np.stack([gx_g.ravel(), gy_g.ravel()], axis=1)     # (L1_G², 2)
        self.l1 = RBFLayerND(dim=4, n_particles=L1_P, n_vars=L1_O,
                             init_range=(-1.5, 1.5), use_pu=True, init_zero=False)
        if MATCHED:
            # 147 = 49 空间 × 3 通道：每通道获得完整 7×7 空间覆盖
            xy = np.tile(base_xy, (3, 1))
            ch = np.repeat(CH_VALS, len(base_xy))
        else:
            # 49 粒子：7×7 空间位置循环分配通道（R/G/B 各约 1/3）
            xy = base_xy[:L1_P]
            ch = np.array([CH_VALS[j % 3] for j in range(L1_P)], dtype=np.float32)
        c = torch.zeros(4, L1_P)
        c[0] = torch.tensor(xy[:, 0])
        c[1] = torch.tensor(xy[:, 1])
        c[2] = torch.tensor(ch)
        c[3] = 0.5
        self.l1.shift.centers.data.copy_(c)
        self.l1.shift.h.data.fill_(0.15)
        self.l2 = RBFLayerND(dim=L1_O, n_particles=L2_P, n_vars=L2_O,
                             init_range=(-1.5, 1.5), use_pu=True, init_zero=True)
        self.l3 = RBFLayerND(dim=L2_O, n_particles=L3_P, n_vars=L3_O,
                             init_range=(-1.5, 1.5), use_pu=True, init_zero=True)
        self.protos = nn.Parameter(torch.zeros(N_CLASSES, L3_O))
        nn.init.orthogonal_(self.protos)
        self.T = TEMPERATURE

    def forward_chunk(self, imgs):
        """imgs: (b,3,32,32) tensor → (b, L3_O) 特征（分块内全点云）"""
        b = len(imgs); P = 3 * IMG_SIZE * IMG_SIZE
        X = torch.zeros(b * P, 4, dtype=torch.float32)
        for i in range(b):
            X[i * P:(i + 1) * P] = torch.tensor(build_4d(imgs[i].numpy()))
        g0, a0 = self.l0(X); f0 = X + (g0 + a0)
        g1, a1 = self.l1(f0); f1 = g1 + a1
        g2, a2 = self.l2(f1); f2 = g2 + a2
        g3, a3 = self.l3(f2); f3 = g3 + a3
        return f3.view(b, P, L3_O).mean(1)

    def forward_batch(self, imgs):
        """分块前向（no_grad 或训练均可），返回 (B, L3_O)"""
        feats = []
        for s in range(0, len(imgs), CHUNK):
            feats.append(self.forward_chunk(imgs[s:s + CHUNK]))
        return torch.cat(feats, dim=0)

    def train_epoch(self, imgs, lbls, opt):
        """分块梯度累积：梯度数学上等价于全批，但显存只占一个 chunk"""
        opt.zero_grad()
        B = len(imgs); n_chunks = (B + CHUNK - 1) // CHUNK
        total_loss = 0.0
        for ci, s in enumerate(range(0, B, CHUNK)):
            e = min(s + CHUNK, B)
            f = self.forward_chunk(imgs[s:e])
            ce = nn.functional.cross_entropy(
                -torch.cdist(f, self.protos) / self.T, lbls[s:e],
                reduction="sum") / B
            if ci == n_chunks - 1:  # 原型分离正则加在最后一个 chunk
                pd = torch.cdist(self.protos, self.protos)
                reg = ((pd - (1.0 - torch.eye(N_CLASSES))) ** 2).mean()
                ce = ce + PROTO_REG * reg
            total_loss += ce.item()
            ce.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        return total_loss

# ═══════════════════════════════════════════════════
# 3. Digital training
# ═══════════════════════════════════════════════════
L("\n--- Digital Training (autograd) ---")
# ONE SEED COVERS THE MODEL INITIALISATION.  Before this, the DATA SPLIT was seeded
# (`np.random.seed(c*100)` in the per-class loop) but the TORCH INITIALISATION was
# NOT, so every number this script published was a single draw from an unseeded
# process -- the same defect found and fixed in training/himal_nd.py and
# training/proto_nd.py.  Set here rather than inside the loop so the model, the
# optimiser state and the sampling all come from one stream.
torch.manual_seed(SEED)
np.random.seed(SEED)
L(f"  SEED={SEED}  (model init, data split, and sampling)")
model = DeepRBF4D()
n_param = sum(p.numel() for p in model.parameters())
L(f"  Trainable params: {n_param}")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[400, 550], gamma=0.3)
t0 = time.time()
history = []          # (epoch, train_loss, train_acc, test_acc)
best_test_acc = 0.0; best_state = None; no_improve = 0

train_t = torch.tensor(train_imgs); train_y = torch.tensor(train_lbl, dtype=torch.long)
test_t = torch.tensor(test_imgs);   test_y = torch.tensor(test_lbl, dtype=torch.long)

for ep in range(EPOCHS):
    loss_v = model.train_epoch(train_t, train_y, opt)
    scheduler.step()

    if (ep + 1) % EVAL_EVERY == 0:
        with torch.no_grad():
            f_tr = model.forward_batch(train_t)
            acc_tr = (torch.cdist(f_tr, model.protos).argmin(1) == train_y).float().mean().item()
            f_te = model.forward_batch(test_t)
            acc_te = (torch.cdist(f_te, model.protos).argmin(1) == test_y).float().mean().item()
        history.append((ep + 1, loss_v, acc_tr, acc_te))
        L(f"  ep {ep+1:4d}/{EPOCHS} | loss={loss_v:.4f} | train={acc_tr:.3f} | test={acc_te:.3f}")

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

with torch.no_grad():
    f_t = model.forward_batch(torch.tensor(test_imgs))
    preds_d = torch.cdist(f_t, model.protos).argmin(1).numpy()
acc_d = (preds_d == test_lbl).mean()
L(f"\n  Digital test accuracy: {acc_d*100:.1f}%")

# ---- 通道打乱推理测试（数字侧）：验证网络是否真的使用通道身份 ----
def channel_shuffle_eval(imgs_np, perm):
    """perm[c] = 源通道：新通道 c 取原通道 perm[c] 的强度"""
    with torch.no_grad():
        f = model.forward_batch(torch.tensor(imgs_np[:, perm, :, :].copy()))
        preds = torch.cdist(f, model.protos).argmin(1).numpy()
    return (preds == test_lbl).mean(), preds, f.numpy()

L("\n--- Channel-Shuffle Test (digital, no retrain) ---")
f_orig = f_t.numpy()
acc_shuf, preds_shuf, f_shuf = channel_shuffle_eval(test_imgs, perm=[1, 2, 0])   # R←G, G←B, B←R
shuf_mse = float(np.mean((f_orig - f_shuf) ** 2))
L(f"  original         : acc={acc_d*100:.1f}%")
L(f"  RGB→GBR shuffled : acc={acc_shuf*100:.1f}%  (drop {100*(acc_d-acc_shuf):.1f} pts)")
L(f"  feature MSE (orig vs shuffled): {shuf_mse:.6f}")
L("  → 大幅下降 = 网络使用了通道身份信息（通道自适应）；几乎不变 = 通道可换（未利用色彩）")

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
def physical_forward_single(img_rgb):
    X = build_4d(img_rgb)
    og0, oa0 = pl0.forward(X); f0 = X + (og0 + oa0)
    og1, oa1 = pl1.forward(f0); f1 = og1 + oa1
    og2, oa2 = pl2.forward(f1); f2 = og2 + oa2
    og3, oa3 = pl3.forward(f2); f3 = og3 + oa3
    return f3.mean(axis=0)

def classify(features, protos):
    d = np.sqrt(((features[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2) + 1e-12)
    return d.argmin(axis=1)

with torch.no_grad():
    df = model.forward_batch(torch.tensor(test_imgs[:10])).numpy()
pf = np.zeros((10, L3_O), dtype=np.float32)
for i in range(10): pf[i] = physical_forward_single(test_imgs[i])
dp_mse = float(np.mean((df - pf) ** 2))
L(f"  Digital-Physical feature MSE (10 test imgs): {dp_mse:.6f}")

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
# 6. 通道自适应分析：训练后 L1 粒子在通道轴上的分布
# ═══════════════════════════════════════════════════
L("\n--- Channel Adaptation Analysis (L1 particles) ---")
centers_l1 = model.l1.shift.centers.detach().numpy()   # (4, P)
h_l1 = model.l1.shift.h.detach().numpy()               # (P,)
c_pos = centers_l1[2]                                  # 通道坐标

# 每个粒子对 R/G/B 的响应（高斯核）：resp = exp(-((c - c0)/h)^2)
c_grid = CH_VALS
resp = np.exp(-((c_grid[None, :] - c_pos[:, None]) / h_l1[:, None]) ** 2)  # (P, 3)
resp_norm = resp / (resp.sum(axis=1, keepdims=True) + 1e-12)
mean_resp = resp_norm.mean(axis=0)                     # 学到的平均通道注意力
dominant = resp_norm.argmax(axis=1)
hist = [int((dominant == k).sum()) for k in range(3)]

L(f"  L1 粒子通道坐标: min={c_pos.min():+.2f}, max={c_pos.max():+.2f}, "
  f"mean={c_pos.mean():+.2f}, std={c_pos.std():.3f}")
L(f"  主导通道粒子数: R={hist[0]}, G={hist[1]}, B={hist[2]} (init 每通道≈{L1_P//3})")
L(f"  粒子宽度 h: mean={h_l1.mean():.3f}, min={h_l1.min():.3f}, max={h_l1.max():.3f}")
L(f"  学到的平均通道注意力: R={mean_resp[0]:.3f}, G={mean_resp[1]:.3f}, B={mean_resp[2]:.3f}")
L(f"  固定亮度权重(基线先验): R=0.299, G=0.587, B=0.114")
L("  → 若学到的注意力明显偏离亮度权重且通道间有分化，即粒子自发实现了通道自适应")

# ═══════════════════════════════════════════════════
# 7. Summary
# ═══════════════════════════════════════════════════
L(f"\n{'='*60}")
L("  SUMMARY")
L(f"{'='*60}")
peak_train = max([h[2] for h in history]) * 100
L(f"  Digital train: {peak_train:.1f}% (peak) → {history[-1][2]*100:.1f}% (final)")
L(f"  Digital test:  {best_test_acc*100:.1f}% (best, early-stop @ {len(history)*EVAL_EVERY}ep)")
L(f"  Train-Test gap: {(history[-1][2]-history[-1][3])*100:.1f}% (overfitting)")
L(f"  Training time: {t_train:.0f}s")
L(f"  Params: {n_param}")
L(f"  Digital-Physical gap: {dp_mse:.6f} MSE")
noise_drop_pct = (acc_phys["none"] - acc_phys["high"]) / max(acc_phys["none"], 1e-10) * 100
L(f"  Physical noise drop: none={acc_phys['none']*100:.1f}% → high={acc_phys['high']*100:.1f}% ({noise_drop_pct:.1f}% degradation)")
L(f"  Channel-shuffle drop: {100*(acc_d-acc_shuf):.1f} pts (digital)")
L(f"  Learned channel attention: R={mean_resp[0]:.3f}, G={mean_resp[1]:.3f}, B={mean_resp[2]:.3f}")

# 与 3D 基线自动对比
def _peek_report(path):
    """从已有报告里提取 (test acc, params)，读不到返回 None"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            t = f.read()
        import re
        m = re.search(r"Digital test accuracy:\s*([\d.]+)%", t)
        mp = re.search(r"Trainable params:\s*(\d+)", t)
        return (m.group(1), mp.group(1) if mp else "?") if m else None
    except Exception:
        return None

if not SMOKE:
    refs = [("3D 灰度基线 (cifar_01, 49p)", BASELINE_REPORT)]
    if MATCHED:
        refs.append(("4D 标准 (49p, 通道循环)",
                     os.path.join(OUT_DIR, "cifar_07_4d_rgb_report.txt")))
    L("")
    for ref_name, ref_path in refs:
        r = _peek_report(ref_path)
        if r:
            L(f"  [对比 {ref_name}] test acc={r[0]}%, params={r[1]}  →  本次: {best_test_acc*100:.1f}%, {n_param}")
    if MATCHED:
        L("  → 若 147p-4D > 49p-4D 明显，支持\"通道信息已存在、此前受容量限制\"的假说")

# ═══════════════════════════════════════════════════
# 8. Plot
# ═══════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 训练损失
ax = axes[0, 0]
ep_list = [h[0] for h in history]; loss_list = [h[1] for h in history]
ax.plot(ep_list, loss_list, 'b-', linewidth=1.5)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
ax.grid(True, alpha=0.3)

# 训练/测试精度
ax = axes[0, 1]
train_accs = [h[2] * 100 for h in history]; test_accs = [h[3] * 100 for h in history]
ax.plot(ep_list, train_accs, 'b-', linewidth=1.5, label="Train")
ax.plot(ep_list, test_accs, 'r-', linewidth=1.5, label="Test")
best_ep = ep_list[int(np.argmax(test_accs))]; best_ta = max(test_accs)
ax.axvline(x=best_ep, color='orange', linestyle='--', linewidth=1.2, label=f'Best test ep={best_ep}')
ax.plot(best_ep, best_ta, 'r*', markersize=12)
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Train vs Test (best test={best_ta:.1f}% @ ep {best_ep})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 噪声扫描
ax = axes[0, 2]
acs = [acc_phys[n] * 100 for n in noise_levels]
bars = ax.bar(range(len(noise_levels)), acs,
              color=["#4CAF50", "#FFC107", "#FF9800", "#F44336"], edgecolor="black")
ax.set_xticks(range(len(noise_levels))); ax.set_xticklabels(noise_levels)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Physical Noise Robustness")
for b, a in zip(bars, acs):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{a:.1f}%", ha="center", fontsize=9)
ax.set_ylim(0, max(acs) * 1.3); ax.grid(True, alpha=0.2, axis="y")

# 混淆矩阵
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
                color="white" if confusion[i][j] > confusion.max()/2 else "black")

# ★ 通道自适应：L1 粒子通道坐标直方图 + 学到的注意力 vs 亮度权重
ax = axes[1, 1]
ax.hist(c_pos, bins=15, range=(-1.2, 1.2), color="#5C9ECC", edgecolor="black", alpha=0.85)
for xv, name in zip(CH_VALS, ["R", "G", "B"]):
    ax.axvline(x=xv, color=["#D62728", "#2CA02C", "#1F77B4"][["R","G","B"].index(name)],
               linestyle="--", linewidth=1.2)
    ax.text(xv, ax.get_ylim()[1]*0.95, name, ha="center", fontsize=9,
            color=["#D62728", "#2CA02C", "#1F77B4"][["R","G","B"].index(name)])
ax.set_xlabel("Particle channel coordinate (L1)"); ax.set_ylabel("# particles")
ax.set_title(f"Learned Channel Distribution (R/G/B: {hist[0]}/{hist[1]}/{hist[2]})")
ax.text(0.02, 0.02,
        f"attention R={mean_resp[0]:.2f} G={mean_resp[1]:.2f} B={mean_resp[2]:.2f}\n"
        f"luminance  R=0.30 G=0.59 B=0.11",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))
ax.grid(True, alpha=0.3)

# 特征空间
ax = axes[1, 2]
for c in range(N_CLASSES):
    mask = test_lbl == c
    ax.scatter(pf_all[mask, 0], pf_all[mask, 1], label=CLASS_NAMES[c][:6], s=20, alpha=0.7)
ax.scatter(phys_protos[:, 0], phys_protos[:, 1], c="black", marker="X", s=100,
           linewidths=2, label="prototype")
ax.set_xlabel("Feature dim 0"); ax.set_ylabel("Feature dim 1")
ax.set_title("Physical Feature Space")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle("C07: 4D Encoding RGB — RBF Particles in (x, y, channel, I)"
             + ("  [MATCHED 147p]" if MATCHED else ""),
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.close()
L(f"\n  Plot: {os.path.relpath(PLOT_FILE, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/')}")
L(f"  Report: {REPORT_FILE}")
L("=" * 60)
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
