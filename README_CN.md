# 物理域 RBF 粒子神经网络

[![CI](https://github.com/usecodeandaiforstudy/signal-node-physics-network/actions/workflows/ci.yml/badge.svg)](https://github.com/usecodeandaiforstudy/signal-node-physics-network/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

**节点就是物理源点**的神经网络 —— 训练不靠梯度，而靠**读器件的算子**：强度由闭式解标定，几何由场读数恢复。整套东西的目标是**在物理基底上运行**，本仓库实现并测量了三个这样的基底。

---

## 1. 这是什么

普通神经网络把知识存在**矩阵**里，训练就是靠反向传播调整那些矩阵。本项目把矩阵换成**一个场**：

* **一个节点**是物理介质中一个**可移动的源点** —— 一个有位置、有影响半径的粒子；
* 输入扰动这个介质，每个节点给出一个**非线性激活**，其**形状**由该节点的几何（位置与宽度）决定；
* 网络的输出是这些被激活的源点的**加权叠加**；

即 `out(x) = Σ_j W_gj · z_j(x)`，其中 `z_j(x)` 是节点 `j` 的物理响应，`W_gj` 是该源点被驱动的强度。**"先激活、后组合"** —— 非线性来自相互作用的**几何**，而不是来自对加权和施加的某个固定激活函数。

**为什么这不只是一个类比。** 训练算法要设定的两个量，在物理上都是**真实的旋钮**：把源点驱动得更强，是一个真实的电流或幅度；移动一个源点或改变它的半径，是一个真实的位置或势阱刚度。所以训练**不必被仿真** —— 它可以是**在器件上**执行的一个流程，花的是**测量次数**而不是梯度。

---

## 2. 核心思想

**① 用节点替代矩阵。** 一层不是"权重矩阵 + 激活函数"，而是**一组源点以及它们共同产生的场**。参数住在几何里（每个源点在哪、有多宽）与每个源点的驱动强度里。

**② 权重就是节点的激活强度。** `W_g` 不是一个抽象系数：**它是每个源点被驱动得多强**。正因如此，读出才有物理意义，也正因如此，**标定**（测量响应、解出驱动量）才是一种合法的训练方式。

**③ 用读器件的算子训练，而不是用梯度。** 训练规则是：**源强度用闭式解标定，把每一次物理测量都投给几何。** 因为高斯型节点的响应是

```
log z_j(x) = -alpha_j ||x||^2 + gamma_j . x - delta_j
```

即**对节点自身的自然参数是线性的**，所以一次线性求解就能从**纯场读数**中**闭式**恢复中心与宽度 —— 而强度由第二次闭式求解给出。**两个族都不需要梯度搜索。**

**④ 这在任何满足一个可检验条件的物理系统上都成立。** 不是口号式的"任意系统"：节点的响应必须满足

> **`log z(x)` 在输入的某个【固定的、有限的】基 `phi(x)` 里【线性】。**

高斯满足它（其指数是一个二次多项式，系数编码了中心）。**任何多项式指数按其次数都满足** —— 四次势阱需要四次基，然后转移到机器精度。而响应**在任何有限基里都不是对数线性**的源点 —— Laplace、Yukawa —— **被同一个测试排除**，并且由算子自己的残差显示出来。查表进一步放宽到**"任何是单标量固定函数的核"**，而一族以 `h` 索引的表够到了 Yukawa。

---

## 3. 架构

```
输入 x
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  层  （一个物理单元，N 个节点）                                     │
│                                                                  │
│   节点 j：  中心 c_j（D 维）      宽度 h_j                          │
│             激活  z_j(x) = T( ||(x - c_j)/h_j||^2 )               │
│                                                                  │
│   节点上的叠加：  field_i(x) = Σ_j W_ij · z_j(x)                   │
│   （介质是线性的，所以这个和是【精确】的 —— 而且经过验证）             │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
   下一层（wire-direct：上一层的输出就是下一层的输入）
   │
   ▼
   标量残差读出：  pred = Σ_层 out_层[:, 0]
```

**所以前向**就是：由输入与节点中心的距离算出每个节点的激活，按其宽度缩放，再叠加。

**两个训练算子：**

| 算子 | 读什么 | 产出什么 | 代价 |
|---|---|---|---|
| **几何算子** | 探针点上的**场** | 每个节点的**中心与宽度**，闭式，**外加一个一致性残差** | **0 次前向** |
| **强度算子** | 激活值与目标 | 驱动强度，闭式（岭回归）求解 | `n_train × N` 次传感读数 |

**而那个残差是诊断，不是副产品。** 几何算子返回 `log_rms` —— 一个真正的单节点必须满足的那个恒等式留下的残差。它回答的是 ***"这个响应真的是一个节点吗？"*** —— **这是基于搜索的训练器问不出来的问题**，因为搜索从不检查它的模型是否匹配它正在建模的对象。

**已实现三个基底**，每个都是一个**真实的层**（而不是 trainer 里的一条公式）：

| 基底 | 介质是什么 | 验证到什么程度 |
|---|---|---|
| **模拟电路** — `physical_simulator/` | 差分对 → Gilbert 平方器 → BJT 指数器 → 忆阻器 crossbar | 本仓库最初的载体，选它是因为容易仿真 |
| **扩散单元** — `substrate_diffusion/` | 线性介质中的扩散物种；核就是该介质的格林函数 | **其传播子与独立的有限差分 PDE 解吻合到 `1.62e-05` —— 即求解器自身的误差** |
| **非简谐阱** — `extensions/multigraph/ext_38_third_substrate.py` | `V(r) = κr⁴` 中的粒子，平衡密度为 `exp(−(κ/kT)r⁴)` | **一个不同的指数；用正确的基去读时，算子以机器精度（中心 `1.5e-15`、宽度 `7.3e-15`）转移过去** |

---

## 4. 现在能做到什么程度

**以下都是在本仓库任务上【测出来】的，全部可用 [`FINDINGS.md`](FINDINGS.md) 与 [`extensions/multigraph/FINDINGS.md`](extensions/multigraph/FINDINGS.md) 里的命令复现。**

| 能力 | 实测结果 |
|---|---|
| **从场读数闭式恢复几何** | 精确恢复**通过了算子自己的 `1e-3` 相对误差合格线**；**1% 读数噪声**下中心误差 **`1.2e-03`**，3% 下 `4.1e-03`。在四次基底上：中心 **`1.5e-15`**、宽度 **`7.3e-15`** |
| **算子 对 梯度训练** | 2 维回归 **`+0.98` 对 `+0.23`** · 4 维 **`+0.58` 对 `+0.05`** · moons 分类 **`1.00` 对 `0.73`** |
| **算子的代价** | **0 次前向** —— 而对手是 600 轮 autograd |
| **训练规则的测量效率** | 同等精度下**测量次数少 15–30 倍** |
| **错误的基底会被【检出】，而不是被静默地错拟合** | Laplace/Yukawa 响应的 `log_rms` 为 **`4.16e-01`**，而高斯节点是 `3e-06` |
| **查表无需改码即被检出** | 3 个被查表的源点全部被 `log_rms` 标记 |
| **多图成本** | **传感前叠加把传感读数降为 `k/m`** —— `k=6, m=2` 时从 `12` 降到 `4`，输出吻合到 `7.4e-08` |
| **可工作规模** | 每层 20–40 个粒子、3 层、2–10 维输入 |

**而做不到什么，直说：**

* **没有任何东西在真硬件上被做出来过。** 三个基底都是**模拟的单元**。算子从未对着真实器件跑过，而扩散单元的 PDE 验证核对的是**模型的核**与物理 —— **不是造出来的单元与模型**。
* **尺度是玩具尺度。** 每个任务都小，每个超参都是出厂默认，**高维那一轴是【被测出来够不着】，而不是被假定**。
* **本项目起步时那个深层架构不成立。** 它在仓库能构造的**每一个**任务上都被一个**零前向基线**打败。**交付的是算子那一半；深层叠加不是。**
* **目前只覆盖一个基底族** —— 多项式指数，加上基于查表的扩展。非对数线性的核需要查表，而查表需要一个固定的单标量形式。

完整记录 —— 测了什么、怎么测的、以及什么限定了这些数字 —— 在 [`FINDINGS.md`](FINDINGS.md)：其中 [§9](FINDINGS.md) 是限定这些数字的 caveat、[§11](FINDINGS.md) 是还开着什么。**引用任何数字之前请先读它们。**

---

## 5. 快速开始

```bash
git clone https://github.com/usecodeandaiforstudy/signal-node-physics-network.git
cd signal-node-physics-network
pip install -r requirements.txt      # 或：pip install -e .
```

**验证基底物理（约 4 秒）：**

```bash
python experiments/sub_01_diffusion_verify.py --assert
```

**跑算子套件** —— 这是项目真正为之站台的部分，其脚本多数在数秒内结束：

```bash
python extensions/multigraph/ext_01_baseline.py --assert    # 正确性锚点
python extensions/multigraph/ext_07_operator_trainer.py     # 算子 对 梯度训练
python extensions/multigraph/ext_06_node_operator.py        # 从场读数恢复几何
```

**在数字模型里训练一个小网络：**

```python
from network.combined import CombinedHiMALRBF
from experiments.benchmarks.datasets import load_dataset

X_tr, X_te, y_tr, y_te, _ = load_dataset("moons")
model = CombinedHiMALRBF(input_dim=2, layer_sizes=(8, 5, 3), output_dims=(3, 2, 2),
                         task="classification", n_classes=2, use_spsa=True, spsa_avg=3)
model.train(X_tr, y_tr, epochs_per_layer=30, lr=0.02)
print((model.predict(X_te).ravel() == y_te.ravel()).mean())
```

---

## 6. 每个文件的作用

### 文档

| 文件 | 作用 |
|---|---|
| `README.md` / `README_CN.md` | 本文件（英文 / 中文） |
| `THEORY.md` / `THEORY_CN.md` | 完整推导：物理映射、训练方案 A–D、以及算子的数学 |
| `FINDINGS.md` | **本项目确立了什么，以此为准。** 第 0–12 节：复现命令、基底验证、容量分析、头号训练规则、局域性、成本模型、更正、开放项、产物清单 |
| `extensions/multigraph/FINDINGS.md` | 扩展自己的记录：§12.1–12.36 —— 每个算子结果、每个负面结果，以及一份带编号的缺陷清单 |
| `extensions/multigraph/DESIGN.md` | 多图扩展的设计：记号、两个半边、组合算子的数学、参数经济学、以及任何实现都必须满足的约束 |
| `extensions/multigraph/BASE_CORRECTIONS.md` | 扩展对本体头号主张的审计 —— 一条主张一项，附定案的那个测量 |
| `CONTRIBUTING.md` | 本仓库实际运行的约定：一个数字怎么写下来、一条主张怎么撤回 |
| `CITATION.cff` | 如何引用 |

### 库

| 路径 | 作用 |
|---|---|
| `simulator/layer_nd.py` | **`PhysicalRBFLayerND`，D 维物理层**：`centers (D,N)`、`h (N,)`、`W_gauss`、`W_act`；`forward` 返回两个通道 |
| `simulator/network_nd.py` | 把层叠成网络，配标量残差读出 |
| `network/core_nd.py` | `RBFLayerND`，**数字（PyTorch）层** —— 物理层由它导出 |
| `network/combined.py` | `CombinedHiMALRBF` 与 `SourceStack`，物理模型所对照的数字训练模型 |
| `network/utils_nd.py` | 初始化、缩放与参数量工具 |
| `physical_simulator/components.py` | **模拟电路**的元件：差分对、Gilbert 乘法器与平方器、BJT 指数器、忆阻器 crossbar、积分器、ADC，以及噪声/配置模型 |
| `physical_simulator/layer.py` | `PhysicalRBFLayer`，由电路支撑的层 |
| `physical_simulator/network.py` | 电路支撑的网络组装与数字↔物理导出 |
| `substrate_diffusion/components.py` | **扩散单元**的物理：格林函数、传感链（噪声、漂移、削顶、量化）、以及 `readout` 级 |
| `substrate_diffusion/layer.py` | `DiffusionRBFLayerND`，由扩散支撑的层，可标定、可播种 |
| `training/himal_nd.py` | `HiMALND`，D 维层次残差训练器（SPSA 与有限差分）—— **梯度训练的对照臂** |
| `training/proto_nd.py`、`training/proto_layerwise.py` | 原型引导训练器，v2 与 v1 |
| `training/himal.py`、`training/spsa.py`、`training/twin.py` | 对 `forward_training/` 里 v1 模块的**再导出**：一份实现、两条导入路径 |
| `forward_training/` | **v1 树**，作为溯源保留；`experiments/ref2d/` 从这里导入 |
| `rbf_network/` | 最初的 2 维 RBF 网络，ND 版的前身 |

### 实验 —— 本体

| 路径 | 作用 |
|---|---|
| `experiments/abl_common.py` | **所有东西共用的骨架**：数字↔物理桥、层构造器、闭式读出（`solve_readout_closed_form`）、成本记账（`readout_calibration_cost`）、测量路径、以及 `selftest()` |
| `experiments/sub_01_diffusion_verify.py` | **基底验证套件** —— 核 对 PDE 解、精确叠加、成本模型。`--assert` 强制 `FINDINGS.md` §2/§3/§8 赖以成立的阈值 |
| `experiments/abl_01…abl_07` | 七个消融：容量切分、局域性、失配敏感度、标定鲁棒性、孪生误差、任务扫描、收敛网格 |
| `experiments/abl_run_all.py` | 跑全部七个（`--quick` 供 CI 冒烟） |
| `experiments/nd_01…nd_09` | ND 数字基线、物理前向验证、物理训练、HiMAL 管线、对比、消融、验证、基准 |
| `experiments/cifar_01…cifar_07` | CIFAR-10 那条线 —— 本项目**最早**的工作：3D 点云编码、autograd 训练、物理导出、三方对比。已播种可复现；数字小是设计使然 |
| `experiments/reg_01…reg_06` | 回归任务，含 California Housing |
| `experiments/train_01…train_03` | 分组有限差分、分组 SPSA、以及零 ADC 的模拟 SPSA |
| `experiments/ref2d/` | 2 维参考系列，含 v1 训练器 |
| `experiments/benchmarks/` | 数据集加载、基线模型、以及基准运行器的报告写出 |
| `experiments/legacy_suite.py` | **最初的九部分验证套件** —— 数字-物理一致性、有限差分深度证明、噪声鲁棒性、v1 基线、深度消融。保留是因为早期章节仍引用它的数字；当前那套批跑是 `abl_run_all.py` |
| `experiments/download_cifar.py` | 把 CIFAR-10 下到 `cifar/`（已 ignore） |

### 实验 —— 扩展

| 路径 | 作用 |
|---|---|
| `extensions/multigraph/trainer.py` | **算子训练器**：`features`（忠实读器件的激活）、`fit_operator`、`solve_strengths`、以及候选提出/细化循环 |
| `extensions/multigraph/operators.py` | **算子本体**：`design_matrix`、`node_from_response`（闭式几何求解与一致性残差）、`node_response`、`propose_node`、`star_probes`、`leverage_select`、`identify_placement`、`best_weight` |
| `extensions/multigraph/graphs.py` | 多图层：`Graph`、`Linear`/`Nonlinear` 组合、`MultiGraphStack`、以及必须精确退化到本体的那个构造 |
| `extensions/multigraph/ext_01…ext_51` | **实验**，按序。每个都有复现命令，并在扩展的 `FINDINGS.md` 里有已跟踪的产物；编号与它的章节对应 |

### 仓库机制

| 路径 | 作用 |
|---|---|
| `outputs/` | 每个实验的产物。只有被文档引用的才被跟踪（见 `.gitignore`），所以克隆下来拿到的是证据而不是缓存 |
| `cifar/` | CIFAR-10 数据，需下载，已 ignore |
| `reference/` | 本地草稿，已 ignore |
| `.github/workflows/ci.yml` | CI：三个 Python 版本上的基底断言与七个消融；加上扩展的正确性锚点、**其 51 个脚本里的 30 个带 `--assert`**、以及"每个脚本都有断言、每个被引用的产物都存在"的检查 |
| `setup.py`、`requirements.txt` | 打包 |

---

## 7. 想深入读什么

| 你想要 | 读 |
|---|---|
| 数学 | [`THEORY_CN.md`](THEORY_CN.md) |
| 究竟确立了什么（带数字） | [`FINDINGS.md`](FINDINGS.md) |
| 算子工作的全貌 | [`extensions/multigraph/FINDINGS.md`](extensions/multigraph/FINDINGS.md) |
| 扩展为什么这样设计 | [`extensions/multigraph/DESIGN.md`](extensions/multigraph/DESIGN.md) |
| 什么限定了这些数字、为什么 | [`FINDINGS.md`](FINDINGS.md) §9 |
| 还开着什么 | [`FINDINGS.md`](FINDINGS.md) §11 |

**关于记录方式。** 本项目先提出主张、再测量它；**而当测量与主张冲突时，主张被【换成测出来的那个】**，而不是在旁边加注 —— **所以这些文档里留下的是活下来的版本，带着它的数字**。`FINDINGS.md` §9 收集的是**读者要正确使用这些数字就必须知道的 caveat**：CIFAR 那条线公布的数字是未播种过程的单次抽样、电路路径上"理想"档并非无噪、几何臂在等成本下落败。**如果本页的某个数字与那些文档冲突，以文档为准。**

---

## 8. 状态、许可、引用

**状态。** 研究代码：玩具尺度、模拟基底、无硬件。成立与不成立的在上面 §4；完整记录在 [`FINDINGS.md`](FINDINGS.md)。

**许可。** MIT —— 见 [`LICENSE`](LICENSE)。

**引用。** 见 [`CITATION.cff`](CITATION.cff)。请引用**方法** —— 几何算子、基底准入条件、以测量计的成本记账 —— 而不是早期那些头号数字（FINDINGS 的更正对它们有限定）。

**贡献。** 见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。**扩大基底覆盖、把仿真换成测量、或推翻一条主张** —— 这正是本项目准备接受的贡献。
