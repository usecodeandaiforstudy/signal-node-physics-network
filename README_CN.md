# 物理域 RBF 粒子神经网络

将每个神经元设计为 D 维空间可移动粒子的神经网络架构。网络的前向传播、损失计算和参数训练全部映射到模拟电路元件，旨在探索神经网络从训练到推理的完整流程在物理系统中模拟实现的可行性。

## 核心思路

标准神经网络在模拟硬件上高效运行面临两个困难：非线性激活函数（ReLU、softmax）在电路中实现代价高；反向传播需要完整计算图和精确梯度，模拟系统物理上不可行。

本项目的路径：**围绕模拟电路天然擅长的运算设计网络架构，用有限差分或零阶优化替代反向传播。**

### "先激活，后组合" vs 标准网络

标准网络的计算顺序是 `activate(combine(x))`——先加权求和再过非线性。本网络逆转这一范式：`combine(activate(x))`——每个粒子先独立对输入产生高斯响应（激活），再将所有粒子的响应线性叠加（组合）。非线性不再来自固定的激活函数，而是来自粒子与输入的空间关系：输入点离粒子越远，该粒子的贡献越小。

### 粒子信源叠加模型

将网络理解为 D 维空间中的一组**粒子信源**：输入 X 是探测点位置，粒子中心是信源位置，高斯激活是信源在探测点处激发的场强，权重是接收器灵敏度。网络输出 = 所有信源的场强加权叠加。**训练等价于信源在空间中迁移、自适应调整影响范围的物理演化过程。**

这种设计使网络不再是一个"黑箱函数拟合器"，而是一个可解释的物理场叠加系统。信号链不仅适用于模拟电路，还可映射到光学系统（高斯光束叠加）、波动场（点源格林函数叠加）、扩散场（热核叠加）等多种物理平台。详见 [THEORY_CN.md](THEORY_CN.md) 第 0 节。

## 当前状态

- ND 架构支持任意维度输入，2D 为特例
- 分类和回归共用一套物理组件
- 训练→前向→推理全物理域，零 PyTorch autograd
- **分类 moons（FD）**：L1=0.92 → Full=0.94（+0.02 深度贡献）
- **回归（FD 残差）**：R²=0.121→0.125，proto_y 幅度 0.92→0.77→0.76（残差有效）
- **数字-物理一致性**：max|误差| = 7.80×10⁻⁷（远低于 10⁻⁴ 阈值）
- **噪声鲁棒性**：none→high 精度下降仅 0.010（四档噪声级别）
- **参数效率**：RBF 595 参数/0.81 vs MLP 17410 参数/0.88（29× 更少参数，精度差 <10%）
- **数字基线**：V1 RBF acc=0.97（140p），V3 ND acc=0.82（133p）
- **深度消融**：等 9 粒子下 1L 和 3L 均成功训练（深度贡献见 Part 2: +0.020）
- `experiments/run_all.py` 一键验证全部（9 项实验全部 PASS）

## 项目结构

```
singal-node-physics-network/
├── physical_simulator/              # 底层模拟电路元件
│   ├── components.py                #   6 种电路元件（差分对、Gilbert、BJT、电流镜、忆阻器、基尔霍夫求和）
│   ├── layer.py                     #   2D 物理 RBF 层（V1 参考）
│   └── network.py                   #   2D 物理残差网络（V1 参考）
│
├── simulator/                       # ND 物理模拟器（当前主力）
│   ├── layer_nd.py                  #   PhysicalRBFLayerND — D 维粒子 RBF 层
│   └── network_nd.py                #   PhysicalResidualRBFNetworkND — 三层残差网络
│
├── network/                         # 网络定义
│   ├── combined.py                  #   CombinedHiMALRBF — 统一分类+回归入口
│   ├── core_nd.py                   #   ND 数字 RBF 层（PyTorch，参考）
│   └── utils_nd.py
│
├── training/                        # 训练算法
│   ├── himal_nd.py                  #   HiMAL-ND — 当前主力训练器（分类+回归）
│   ├── proto_nd.py                  #   ND 原型引导训练
│   └── proto_layerwise.py           #   2D 原型引导训练（V1 参考）
│
├── rbf_network/                     # 2D 数字 RBF 网络（PyTorch，V1 参考）
│
├── experiments/                     # 实验
│   ├── run_all.py                   #   一键验证全部 9 项（一致性/深度/噪声/效率/消融）
│   ├── e11_fd_cls.py                #   FD 分类深度证明（核心）
│   ├── e11_fd_reg.py                #   FD 回归深度证明（核心）
│   ├── e01-e10_*.py                 #   V3 ND 实验套件（独立运行）
│   ├── 01-13_*.py                   #   V1 2D 参考实验（独立运行）
│   └── benchmarks/                  #   sklearn 基线对比
│
├── outputs/                         # 实验输出
├── reference/                       # 理论参考文献
├── README_CN.md                     # 本文件
└── THEORY_CN.md                     # 完整理论推导
```

## 网络架构

### 粒子模型

每个粒子在 D 维空间中有一个中心坐标 `centers (D, N)` 和影响半径 `h`。给定输入 `X (batch, D)`，粒子激活值为高斯核响应。所有参数（中心、宽度、权重）均可通过有限差分训练。

### 三层结构

```
输入 X (batch, D)
    → Layer1: PhysicalRBFLayerND → 输出 (batch, d1)
    → Layer2: PhysicalRBFLayerND → 输出 (batch, d2)  [残差累积]
    → Layer3: PhysicalRBFLayerND → 输出 (batch, d3)  [残差累积]
    → 最终输出
```

层间使用 **wire-direct** 直连：上一层的多维输出直接作为下一层的多维输入，不做均值投影。

### 物理信号链（6 组件）

```
输入电压 → 差分放大器(dx) → Gilbert平方器(d²) → 基尔霍夫求和
        → BJT指数器(exp) → 电流镜归一化 → 忆阻器交叉阵列(W·ẑ) → 输出
```

### HiMAL-ND 训练

***HiMAL（Hierarchical Multi-Alignment Learning，分层多对齐学习）*** 在本项目中适配为 RBF 网络的逐层双网络训练方法。核心哲学是用结构化分解替代全局优化：

- 数据按中位数分区，A/B 两个子网络各自学习不同区域
- 每层独立训练后冻结，过渡区对齐损失（替代跨层反向传播）保证两网络在交界面输出一致
- 分类使用 prototype-guided 损失（无 softmax/log/exp）
- 回归使用残差训练：L2/L3 学习前层未拟合的剩余部分
- 梯度通过逐参数有限差分（FD）或 SPSA 估计，零 autograd

## 实验目录

### V3 核心实验（ND 架构，纯物理域）

| 实验 | 作用 |
|------|------|
| **e11_fd_cls** ★ | FD 分类深度证明：moons 多 seed 取最优，验证 L1→L1+L2→Full 精度递增 |
| **e11_fd_reg** ★ | FD 回归深度证明：残差训练 + proto_y 幅度递减，验证 L2/L3 学到残差 |
| e01_nd_baseline | ND 数字基线：验证 ND 网络在 2D+4D 上的正确性 |
| e02_nd_physical | 物理前向验证：PhysicalRBFLayerND 与数字模型输出一致 |
| e03_physical_training | 纯物理 FD 原型训练：从随机初始化在物理域收敛 |
| e04_himal_ffa_nd | HiMAL-FFA-ND 管线调试：split/训练/预测全流程 |
| e04b_himal_comparison | HiMAL 分层贡献对比：量化每层精度贡献 |
| e05_nonlinear_regression | 非线性回归层贡献：复杂函数上测试深度帮助 |
| e06_ablation | 消融研究：单/双网络、层数对精度的影响 |
| e07_physical_rbf_net | 物理网络验证：前向一致性和 proto 预测正确性 |
| e08_comprehensive | 综合实验：三种配置 + 噪声鲁棒性 |
| e09_spsa_benchmark | SPSA vs FD 速度/精度对比 |
| e10_noise_robustness | 噪声鲁棒性：四种噪声级别下的精度退化 |
| benchmarks/run_benchmark.py | sklearn 基线对比：产出可引用性能表格 |

### V1 参考实验（2D 架构，PyTorch + 物理模拟器）

| 实验 | 作用 |
|------|------|
| 01_digital_baseline | PyTorch 数字基线：moons 分类 + 回归 |
| 02_physical_forward | 2D 物理前向验证 |
| 03_noise_tolerance | 噪声容限测试 |
| 06_2d_classification | 2D 分类 + 决策边界可视化 |
| 07_projected_ensemble | 4D 随机投影集成 |
| 08_physical_classification | 物理网络 2D 分类 |
| 09_spsa_rbf_classification | HiMAL 双网络 RBF 分类（autograd, ~90%+） |
| 10_himal_rbf_classification | HiMAL 训练后物理评估 |
| 11_physical_himal_fd | 物理 FD 逐层微调 |
| 12_proto_physical_fd | 原型引导纯物理 FD 训练（零 autograd） |
| 13_layer_depth_ablation | 层深度消融：等粒子数下多层优于单层 |

## 训练方法

| 方法 | 反向传播 | 物理可行性 |
|------|---------|-----------|
| SPSA 零阶优化 | 不需要 | 高 — 仅需测量损失值 |
| HiMAL-ND 逐层双网络 | 不需要 | 高 — 逐层独立，过渡区对齐 |
| 逐参数有限差分 (FD) | 不需要 | 最高 — 精确梯度，物理可测量 |
| 原型引导纯物理 | 不需要 | 最高 — 零数字预训练 |

## 快速开始

```bash
pip install torch numpy matplotlib scipy pandas scikit-learn
python experiments/run_all.py
```

## 理论文档

[THEORY_CN.md](THEORY_CN.md) 提供完整推导：ND 粒子模型、物理信号链、HiMAL-ND 训练算法、残差学习。

## 声明

本项目是在有限的实验条件下进行的。由于采用数值模拟来仿真物理系统（模拟电路、忆阻器阵列等物理组件），每次前向传播和参数更新都需要通过物理信号链计算，计算开销远高于标准神经网络的矩阵乘法。受限于算力，当前实验的参数量和训练轮次都较小，仅在 moons、4D 合成分类、正弦回归等简单数据集上完成验证。

本项目的核心目标是**提出一种新的网络架构范式**——"先激活后组合"的粒子信源叠加模型——使神经网络具备更好的可解释性，并能在模拟电路、光学系统、波动场等物理平台上实现完整的训练和推理。九个验证实验（`run_all.py`）覆盖了数字-物理一致性、深度有效性、噪声鲁棒性、参数效率等关键命题，可作为后续研究的基准。

受作者代码水平和算力所限，项目可能存在不足。欢迎条件更好的使用者在本代码基础上调整网络结构、增加参数规模和训练轮次、在更复杂的数据集上开展进一步实验，或在真实物理硬件上验证本架构。希望本项目的思路和代码对你有帮助。

## License

MIT
