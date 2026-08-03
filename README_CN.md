# 物理域 RBF 粒子神经网络

## 📖 简介

> 一种基于物理场叠加原理的新型神经网络架构——每个"神经元"是 D 维空间中可移动的粒子，非线性来自粒子与输入的空间距离。本架构可映射到多种物理平台（模拟电路、光学系统、波动场等）；本项目选择模拟电路作为实现载体，便于在计算机上进行仿真验证。

本项目探索神经网络在物理系统中从训练到推理完整流程的可行性。核心设计是"先激活后组合"：粒子先对输入产生高斯响应，再线性叠加——非线性来自粒子与输入的空间距离，而非固定的激活函数。网络支持分类、回归和高维图像处理，提供 SPSA、有限差分、数字孪生等多种训练方式，全部前向操作有对应模拟电路实现，数字-物理导出精度极高。

主要成果：
- 以 2884 参数在 CIFAR-10 5 类上达到 39.3% 分类准确率，超过同量级 MLP（34.0%），泛化比 73% 为三者最高
- 数字-物理模拟器匹配误差仅 7.80×10⁻⁷，权重可从 PyTorch 直接导出到物理模拟器
- 证明参数分组独立——RBF 粒子的输出贡献互不依赖，天然支持多路硬件并行训练
- 四级硬件复杂度下均实现有效梯度估计（0 ADC / 1 ADC / 全模拟梯度链 / 数字孪生）

## ✨ 核心特性

- 🧠 **可物理部署**：每个运算（距离、高斯核、MVM、梯度估计）均有对应模拟电路，全模拟推理零数字依赖
- 🔬 **数字-物理等价**：物理模拟器与 PyTorch 模型数学精确等价，支持直接导出权重
- 🖼️ **打破高维限制**：3D 结构化编码使 RBF 能处理 1024 维图像，2884 参数超越同任务更大 MLP
- ⚡ **无反向传播训练**：SPSA（0 ADC）/ 逐参数 FD（1 ADC）/ 模拟梯度链（O(1) 延迟）
- 🔀 **天然高并行**：参数分组独立，K 组参数 = K 路硬件可并行训练，互不依赖
- 🛡️ **物理噪声鲁棒**：四档电路噪声下精度仅降 0.01，适应真实硬件非理想性

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
└── reference/                      # 参考文献与实验总结（gitignore）
```

## 📊 实验结果

### 冒烟测试

```bash
python experiments/run_all.py   # 快速验证 10 项基本指标，约 5 分钟，全部 PASS
```

`run_all.py` 是一个快速冒烟测试，用小规模数据和精简训练轮次验证数字-物理一致性、深度有效性、噪声鲁棒性、参数效率等基本命题。**它不会运行完整的 CIFAR-10 训练或其他耗时实验。**

> **关于实验结果的说明**：受限于计算资源，部分实验的训练轮次和参数规模未达到最优配置。例如 CIFAR-10 分类实验的 600 epochs 和 2884 参数，在更大的训练轮次或更多粒子数下仍有提升空间。当前报告的数字应视为方法有效性的下界认证，而非该架构的性能上限。使用者可在此基础上增加训练资源以探索更高精度。

### 复现完整实验结果

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

### 主要结果

| 实验 | 任务 | 指标 | RBF | MLP | CNN |
|------|------|------|-----|-----|-----|
| CIFAR-10 5 类 | 32×32 灰度图 | 准确率 | **39.3%** (2884p) | 34.0% (4295p) / 38.7% (33644p) | 61.3% (2632p) |
| 数字-物理一致性 | 多种维度、粒子数 | 最大误差 | **7.80×10⁻⁷** | —— | —— |
| 回归深度 | 2D 非线性函数 | R² 增益 | +0.064（直接残差）vs +0.013（Proto） | —— | —— |
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
