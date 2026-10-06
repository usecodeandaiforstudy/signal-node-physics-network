# 物理域 RBF 粒子神经网络

[![CI](https://github.com/usecodeandaiforstudy/signal-node-physics-network/actions/workflows/ci.yml/badge.svg)](https://github.com/usecodeandaiforstudy/signal-node-physics-network/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)

---

> ## ⚠️ 状态说明：本页有六条结论已被实测推翻
>
> 本 README 写于 [`FINDINGS.md`](FINDINGS.md) 的实验之前。项目现在确立了什么是以
> [`FINDINGS.md`](FINDINGS.md) 为准的——**引用本页任何内容前请先读
> [FINDINGS.md 第 9 节](FINDINGS.md)**。简要对照：
>
> | 本页的声称 | 实测状态 |
> |---|---|
> | "CIFAR-10 39.3%（SOTA 级结果）" | 不是 SOTA：只用了 10 类中的 5 类、灰度图、每类 100 张，而且模型是**用测试集**选出来的（"peak test accuracy"），MLP 与 CNN 基线同样如此 |
> | "数字-物理误差 7.80×10⁻⁷" | 该数字是 float32 舍入加上归一化除法里的 `+1e-8` 保护项，**不是物理保真度**。核现在改为对独立 PDE 解验证 |
> | "证明参数分组独立" | 被引用的实验（`experiments/train_01_grouped_fd.py`）用了**错误的索引约定**（已修，结果需重跑）。正确测量结果：无归一化时严格独立，**PU 归一化下不独立**——而分类路径正是用 PU |
> | "K 个参数组 = K 个硬件通道，互不依赖" | 同上；PU 下耦合实测 0.063–0.261 |
> | "全模拟梯度链（O(1) 延迟）" | 这些函数**只被定义、全仓库无任何调用点** |
> | "四级硬件复杂度（0 ADC / 1 ADC / …）" | 已被取代：四档的差别只在**几何**如何优化。源强度应当**闭式标定，而不是搜索** |
>
> 取代它们的结果更强、且有实测支撑：第二个经 PDE 验证的物理基底；一张基底成本表；
> 以及一条训练规则——**标定源强度，把所有物理测量投给几何**——同等精度下
> **测量次数少 15–30 倍**。详见 [`FINDINGS.md`](FINDINGS.md)。

---

## 🎯 本项目确立了什么

**这里的主张不是"一种新的网络架构"** ✓ —— 而是：**一个由物理源点组成的网络，可以用【读器件的算子】来训练**，花的是测量而不是梯度 —— **并且这在任何满足一个可检验条件的基底上都成立**。

**1. 一个几何算子，带一致性残差。** 节点的中心与宽度由场读数**闭式**恢复，因为 `log z(x)` 在输入的某个固定有限基里线性。同一个算子返回 **`log_rms`**，它判断这个响应**是不是一个真正的单节点** —— **这是任何基于搜索的训练器报不出来的诊断。**

**而它在本仓库里也是赢的那个臂**：2 维回归 **`+0.98` 对梯度训练的 `+0.23`** ✓ · 4 维 **`+0.58` 对 `+0.05`** ✓ · moons 分类 **`1.00` 对 `0.73`** ✓ —— **代价是零次前向** ✓，而对手是 600 轮 autograd。

**2. 一条可检验的基底准入条件。** "任意物理系统"变成了一个**测试**而不是口号：源的响应必须使 `log z(x)` 在某个**固定有限基**里线性。高斯通过 ✓；任何多项式指数按其次数通过 ✓；**Laplace 或 Yukawa 不通过** ✗ —— **而算子自己的残差显示了这件事**：`4.2e-01` 对高斯节点的 `3e-06` ✓。

**3. 以测量次数为成本单位。** 比较按**读出标定次数**进行，而不是轮数或秒数。正因如此，"零次前向"才是测量结果而不是口号。

**4. 而适用面正在被扩大。** 查表把准入条件放宽到**"任何是单标量固定函数的核"** ✓，而以 `h` 索引的一族表够到了 Yukawa ✓ —— 并给出一条一般教训：**表的参数化变量必须匹配核的光滑性**（用 `u` 而不是 `s = √(u/k)` 建表时，误差停在 `2.2` 且从 4 张表到 64 张表都不动）。

> **而什么是【不】成立的，也是实测出来的**：本项目起步时的**深层**架构，在仓库能构造的**每一个**任务上都被一个**零前向基线**打败 ✗；本应帮助它的闭式标定也没有帮助 ✗。CIFAR 那条线在 5 类上是 36–39%，而随机是 20%，且它的架构比较**被噪声主导**——光是种子就让数字移动最多 6.6 个点 ✗。**没有任何东西在真硬件上做出来过**；三个基底都是模拟器件。
>
> 证据：本体 [`FINDINGS.md`](FINDINGS.md) 第 6、9、11 节，以及扩展的第 12.30–12.34 节。

---

## 📖 简介

> 一种基于物理场叠加原理的神经网络架构——每个"神经元"是 D 维空间中可移动的粒子，非线性来自粒子与输入的空间距离。本架构可映射到多种物理平台（模拟电路、光学系统、波动场等）；本项目选择模拟电路作为实现载体，便于在计算机上进行仿真验证。
>
> *（这句原先写作"**新型**神经网络架构"。**它其实是把已有思想做成了物理实现，先把这一点说清楚才是重点**——见下方[与已有工作的关系](#-与已有工作的关系)。本项目里更窄、可能确实新的是**基底准入条件**与**几何算子的残差**，而不是这一层本身。）*

本项目探索神经网络在物理系统中从训练到推理完整流程的可行性。核心设计是"先激活后组合"：粒子先对输入产生高斯响应，再线性叠加——非线性来自粒子与输入的空间距离，而非固定的激活函数。网络支持分类、回归和高维图像处理，提供 SPSA、有限差分、数字孪生等多种训练方式，全部前向操作有对应模拟电路实现，数字-物理导出精度极高。

主要成果：
- 以 2884 参数在 CIFAR-10 5 类上达到 39.3% 分类准确率，超过同量级 MLP（34.0%），泛化比 73% 为三者最高 *（已撤回：见上方状态说明与 [FINDINGS.md](FINDINGS.md) 第 9.4 节——不是 SOTA，且模型用测试集选出）*
- 数字-物理模拟器匹配误差仅 7.80×10⁻⁷，权重可从 PyTorch 直接导出到物理模拟器 *（已撤回：那是 float32 舍入加 PU 的 epsilon，不是物理保真度；权重导出本身成立，且现在已在两个互不相关的基底上验证）*
- 证明参数分组独立——RBF 粒子的输出贡献互不依赖，天然支持多路硬件并行训练 *（已撤回并由测量取代：无归一化时严格独立，PU 下不独立——[FINDINGS.md](FINDINGS.md) 第 7 节）*
- 四级硬件复杂度下均实现有效梯度估计（0 ADC / 1 ADC / 全模拟梯度链 / 数字孪生）*（全模拟梯度链是死代码，ADC 接口从未被调用；四档的差别只在几何如何优化——[FINDINGS.md](FINDINGS.md) 第 9 节）*

## ✨ 核心特性

- 🧠 **可物理部署**：每个运算（距离、高斯核、MVM、梯度估计）均有对应模拟电路，全模拟推理零数字依赖 *（其中**梯度估计**这一半是死代码，已不再作为特性提供——见 [THEORY.md](THEORY.md) Method C 与 [FINDINGS.md](FINDINGS.md) §11 第 8 项；推理本身不需要它）*
- 🔬 **数字-物理等价**：物理模拟器与 PyTorch 模型数学精确等价，支持直接导出权重
- 🖼️ **打破高维限制**：3D 结构化编码使 RBF 能处理 1024 维图像，2884 参数超越同任务更大 MLP
- ⚡ **无反向传播训练**：SPSA（0 ADC）/ 逐参数 FD（1 ADC）/ 模拟梯度链（O(1) 延迟）*（梯度链只被定义、从未被调用）*
- 🔀 **天然高并行**：参数分组独立，K 组参数 = K 路硬件可并行训练，互不依赖 *（无归一化时成立；PU 下不成立——[FINDINGS.md](FINDINGS.md) 第 7 节）*
- 🛡️ **物理噪声鲁棒**：四档电路噪声下精度仅降 0.01，适应真实硬件非理想性 *（该测量在训练集上完成；取代它的器件鲁棒性结果见 [FINDINGS.md](FINDINGS.md) 第 6.2 节）*

## 🔗 与已有工作的关系

**本项目所依赖的那一层并不新，而先把这一点说清楚才是重点。** "隐层参数不训练、输出层用闭式解求线性读出"是成熟做法：**极限学习机**（Huang 等 2006）、**随机特征**（Rahimi & Recht 2008）、**储池计算**（Jaeger 2001）做的都是这件事；**RBF 网络**本身可追到 Broomhead & Lowe 1988。**模拟存内计算**是一个很大的领域，**在物理基底【之内】训练**也早已确立（例如 Wright 等，*Deep physical neural networks*，Nature 2022）。本项目自己的 `FINDINGS.md` §11 第 5 项**也直接测过**储池变体。

**所以诚实的主张是下面四条，每一条都比"一种新型架构"窄得多：**

1. **一条可检验的基底准入条件。** [`FINDINGS.md`](FINDINGS.md) §12.16 把"任意物理系统"变成了一个**测试**：节点的响应必须使 `log z(x)` 在输入的某个**固定有限基**里**线性**——因为几何算子正是靠这一点解出中心与宽度的。高斯通过（其指数是二次多项式），任何多项式指数按其次数通过，而 **Laplace 或 Yukawa 源不通过**——而且是用残差显示出来的，不是靠论证。
2. **一个带一致性残差的几何算子。** 中心与宽度由场读数**闭式**恢复，而 `log_rms` 报告这个响应**是不是一个真正的单节点**。**基于搜索的训练器报不出这件事**，所以它是一种硬件诊断能力，而不只是训练技巧。
3. **以测量次数为成本单位。** 比较是按*读出标定次数*进行的，而不是按轮数或秒数。正因如此，"闭式臂花 **零** 次前向"才是一个测量结果而不是一句口号。
4. **一份负面结果记录。** 本页六条主张已在上方状态框中撤回；深层架构在本仓库能构造的**每一个**任务上都被**零前向基线**打败；另有 58 条缺陷连同各自代价被列出。

**本项目【不】主张**：一个新范式、state-of-the-art 精度，或闭式读出层是它自己的想法。

## 🏗️ 架构概览

网络由 D 维空间中可移动的粒子构成，每层 N 个粒子，各有中心坐标和影响半径 h。输入进入后依次经过三层 RBF 层，每层独立计算高斯核响应并通过忆阻器交叉阵列做线性投影，层间 wire-direct 直连。

```
输入 X (batch, D)
    → Layer 1: 差分对(dx) → Gilbert平方器(d²) → BJT指数器(exp) → 忆阻器MVM → 输出
    → Layer 2: 差分对(dx) → Gilbert平方器(d²) → BJT指数器(exp) → 忆阻器MVM → 残差叠加
    → Layer 3: 差分对(dx) → Gilbert平方器(d²) → BJT指数器(exp) → 忆阻器MVM → 残差叠加
    → 最终输出
```

物理信号链包含 6 种电路元件：差分放大器 → Gilbert 平方器 → 基尔霍夫电流求和 → BJT 亚阈值指数器 → 电流镜归一化 → 忆阻器交叉阵列。

完整推导（物理系统映射、训练算法、理论分析）见 [THEORY_CN.md](THEORY_CN.md)。

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- pip

### 安装

```bash
git clone https://github.com/usecodeandaiforstudy/signal-node-physics-network.git
cd signal-node-physics-network
pip install -r requirements.txt
```

可选：以可编辑模式安装，使项目模块可被任意路径直接 import：

```bash
pip install -e .
```

### 最小示例

**方式一：SPSA 训练（物理域，0 ADC，最简硬件）**

```python
from network.combined import CombinedHiMALRBF
from experiments.benchmarks.datasets import load_dataset

X_tr, X_te, y_tr, y_te, _ = load_dataset("moons")

model = CombinedHiMALRBF(
    input_dim=2, layer_sizes=(8, 5, 3), output_dims=(3, 2, 2),
    task="classification", n_classes=2, use_spsa=True, spsa_avg=3,
)
model.train(X_tr, y_tr, epochs_per_layer=30, lr=0.02)
print(f"Acc: {(model.predict(X_te).ravel() == y_te.ravel()).mean():.3f}")
```

**方式二：逐参数 FD 训练（物理域，1 ADC，精确梯度）**

```python
model = CombinedHiMALRBF(
    input_dim=2, layer_sizes=(8, 5, 3), output_dims=(3, 2, 2),
    task="classification", n_classes=2, use_spsa=False,
)
model.train(X_tr, y_tr, epochs_per_layer=40, lr=0.015, eps=1e-3)
print(f"Acc: {(model.predict(X_te).ravel() == y_te.ravel()).mean():.3f}")
```

**方式三：数字孪生训练（PyTorch autograd，快速原型）**

```python
# 参考 experiments/cifar_01_baseline.py —— 构建网络、autograd 训练、
# 训练完成后通过 from_digital() 将权重导出到物理模拟器
```

## 📂 项目结构

```
singal-node-physics-network/
├── README.md / README_CN.md        # 项目说明
├── THEORY.md / THEORY_CN.md        # 完整理论推导
├── LICENSE                         # MIT
├── requirements.txt                # 依赖清单
├── setup.py                        # pip install -e . 支持
│
├── physical_simulator/             # 模拟电路元件
│   └── components.py               #   差分对、Gilbert平方器、BJT指数器、忆阻器MVM等
├── simulator/                      # ND 物理模拟器
│   ├── layer_nd.py                 #   PhysicalRBFLayerND
│   └── network_nd.py               #   PhysicalResidualRBFNetworkND
├── network/                        # 数字网络定义（PyTorch，参考 / 数字孪生）
│   ├── combined.py                 #   CombinedHiMALRBF 统一入口
│   └── core_nd.py                  #   RBFLayerND + ResidualRBFNetworkND
├── training/                       # 训练算法
│   └── himal_nd.py                 #   HiMAL-ND 逐层双网络训练器
├── rbf_network/                    # 2D RBF 网络（早期版本，保留供参考）
│
├── experiments/
│   ├── run_all.py                  #   快速冒烟测试（10 项基本验证，约 5 分钟）
│   ├── download_cifar.py           #   下载 CIFAR-10 数据集到 cifar/ 目录
│   ├── ref2d/                      #   简单 2D 参考实现（13 files，仅供入门参考）
│   ├── reg_01_proto_baseline.py    #   三通道架构 + Proto 残差基线
│   ├── reg_02_direct_residual.py   #   直接残差使深度生效
│   ├── reg_03_noise_robustness.py  #   噪声鲁棒性验证
│   ├── reg_04_fd_reg_1d.py         #   1D sin(x) 最简验证
│   ├── reg_05_fd_cls_moons.py      #   FD 分类深度证明
│   ├── reg_06_california_v6.py     #   California Housing 真实数据基准
│   ├── cifar_01_baseline.py        #   CIFAR-10 5 类：首次成功
│   ├── cifar_02_regularized.py     #   CIFAR-10 5 类：正则化（最优 39.3%）
│   ├── cifar_03_3layer_reg.py      #   CIFAR-10 5 类：三层消融
│   ├── cifar_04_large_data.py      #   CIFAR-10 5 类：容量瓶颈分析
│   ├── cifar_05_mlp_baseline.py    #   CIFAR-10 5 类：MLP 对比
│   ├── cifar_06_three_way_compare.py  # CIFAR-10 5 类：RBF vs MLP vs CNN
│   ├── train_01_grouped_fd.py      #   分组 FD：证明参数组间独立
│   ├── train_02_grouped_spsa.py    #   分组 SPSA
│   ├── train_03_analog_spsa.py     #   零 ADC SPSA：最简硬件方案
│   ├── nd_01_baseline.py           #   ND 数字基线
│   ├── nd_02_physical.py           #   物理前向验证
│   └── ...（到 nd_09_spsa_benchmark.py，共 9 files）
│
├── outputs/                        # 实验输出
├── cifar/                          # CIFAR-10 数据（由 download_cifar.py 下载，gitignore）
└── reference/                      # 参考文献与实验总结（gitignore）
```

## 📊 实验结果

### 本项目真正为之站台的结果

**这些来自算子的工作，也是项目存在的理由。** 完整表格、成本与限度见
[`extensions/multigraph/FINDINGS.md`](extensions/multigraph/FINDINGS.md)；
每一行都对应一个脚本，带复现命令与已跟踪的产物。

| 结果 | 指标 | 数值 |
|---|---|---|
| **几何算子 对 梯度训练**，2 维回归 | 测试 R² | **+0.98** 对 +0.23 |
| 同上，4 维回归 | 测试 R² | **+0.58** 对 +0.05 |
| 同上，moons / 4 维 moons 分类 | 精度 | **1.00 / 0.99** 对 0.73 / 0.74 |
| **算子的代价** | 前向次数 | **0**（对手是 600 轮 autograd） |
| **标定强度 + 测量几何**，对"搜索任一半" | 达到 R² = 0.30 所需测量次数 | **少 15–30 倍** |
| 同上，在 3 000 次测量处 | 测试 R² | **+0.39 / +0.53 / +0.54**，而每个搜索臂 ≤ +0.06 |
| **基底准入条件**，Laplace 对高斯 | 算子 `log_rms` | **4.2e-01** 对 3e-06 —— 这个测试**检出了它** |
| **一致性残差检出查表**，无需改码 | `log_rms` 在 M=512 / 64 / 16 | 3 个全部检出 |
| 容量切分，等测量成本 | 赢家 | 读出，8/8 个 架构×协议 组合 |
| 第二基底 | 核对独立 PDE 解 | 1.62e-05（即求解器自身的误差） |
| 粒子独立性 | 激活耦合 | 无归一化时严格为 0；PU 下 0.063–0.261 |
| **深层架构**，仓库能构造的每个任务 | 对零前向基线 | **全部被打败** *（12.30–12.31）* |

### 冒烟测试

```bash
python experiments/run_all.py   # 快速验证 10 项基本指标，约 5 分钟，全部 PASS
```

`run_all.py` 是一个快速冒烟测试，用小规模数据和精简训练轮次验证数字-物理一致性、深度有效性、噪声鲁棒性、参数效率等基本命题。**它不会运行完整的 CIFAR-10 训练或其他耗时实验。**

> **关于实验结果的说明**：受限于计算资源，部分实验的训练轮次和参数规模未达到最优配置。例如 CIFAR-10 分类实验的 600 epochs 和 2884 参数，在更大的训练轮次或更多粒子数下仍有提升空间。当前报告的数字应视为方法有效性的下界认证，而非该架构的性能上限。使用者可在此基础上增加训练资源以探索更高精度。

### 复现完整实验结果

运行 CIFAR-10 相关实验前，需先下载数据集（约 180MB）：

```bash
python experiments/download_cifar.py
```

如果你已手动下载过 CIFAR-10，也可直接将解压后的数据放到项目根目录的 `cifar/` 目录即可。

| 要验证的结论 | 运行命令 | 预计耗时 |
|---|---|---|
| CIFAR-10 5 类 RBF 最佳精度 | `python experiments/cifar_02_regularized.py` | ~30 min |
| CIFAR-10 5 类三模型对比 | `python experiments/cifar_06_three_way_compare.py` | ~60 min |
| MLP 同量级对比 | `python experiments/cifar_05_mlp_baseline.py` | ~1 min |
| 回归深度有效性（直接残差） | `python experiments/reg_02_direct_residual.py` | ~5 min |
| 物理噪声鲁棒性 | `python experiments/reg_03_noise_robustness.py` | ~2 min |
| California Housing 真实数据 | `python experiments/reg_06_california_v6.py` | ~10 min |
| 分组并行可行性证明 | `python experiments/train_01_grouped_fd.py` | ~5 min |
| 零 ADC SPSA 验证 | `python experiments/train_03_analog_spsa.py` | ~2 min |

### 主要结果（早期，保留以备溯源）

> **请把这一块读作历史，而不是主张。** 除最后两行外每一行都**已被取代**；见上方状态说明与
> [`FINDINGS.md`](FINDINGS.md)。这是 CIFAR 时期的工作，**早于上方那些算子结果**，
> 而其中若干数字正是状态说明所撤回的那些。

| 实验 | 任务 | 指标 | RBF | MLP | CNN |
|------|------|------|-----|-----|-----|
| CIFAR-10 5 类 | 32×32 灰度图 | 准确率 | **39.3%** (2884p) | 34.0% (4295p) / 38.7% (33644p) | 61.3% (2632p) |
| 数字-物理一致性 | 多种维度、粒子数 | 最大误差 | **7.80×10⁻⁷** | —— | —— |
| 回归深度 | 2D 非线性函数 | R² 增益 | +0.064（直接残差）vs +0.013（Proto）**——已撤回：受控重测显示 direct 的深度增益为 `-0.0284`，而 proto 为 `+0.0069`，符号相反。原来那对数字来自两个不同的程序：`reg_01`（替换了 `HiMALND` 的方法并换了核族）与 `reg_02`（根本没有导入它）。见扩展的 [FINDINGS.md](FINDINGS.md) §12.17。** | —— | —— |
| 回归真实数据 | California Housing | R² | 0.530 | 0.551 | —— |
| 噪声鲁棒性 | 4 档电路噪声 | 精度下降 | 0.010 | —— | —— |
| 参数效率 | 4D 合成分类 | 准确率/参数 | 0.81 / 595p | 0.88 / 17410p (29×) | —— |

**CIFAR-10 5 类核心对比**：

| 模型 | 参数量 | Test Acc | Train Acc | 泛化比 | 全模拟可部署 |
|------|--------|----------|-----------|--------|:---:|
| RBF | 2,884 | 39.3% | 53.6% | 73% | ✓ |
| MLP (同量级) | 4,295 | 34.0% | 99.6% | 34% | ✗ |
| MLP (充足) | 33,644 | 38.7% | 97.8% | 40% | ✗ |
| CNN | 2,632 | 61.3% | 100.0% | 61% | ✗ |

RBF 是唯一可全模拟电路部署的方案；同量级 MLP 严重过拟合；MLP 在放大到 11.7× 参数后才能接近 RBF 的测试精度。

## 📝 许可证

MIT License

## 📖 引用

```bibtex
@article{singal-node-physics-network,
  title={Physical-Domain RBF Particle Neural Network},
  year={2026}
}
```
