# 术语表

按模块分组。每条给定义、本仓库里的一个数字或代码位置，以及最常被混淆的近义词。数据中心侧的术语（warp、ZeRO、KV cache 等）见 [mlsys-infra 术语表](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/GLOSSARY.md)。

## M01 端侧 roofline

- **roofline**：可达吞吐 = min(峰值算力, 算术强度 × 带宽)。端侧要画多条屋顶：L1、L2、DRAM 各一条带宽，fp32 FMA、int8 VNNI 各一条算力。本仓库 `host-roofline` 在本机实测这些平台。
- **算术强度（arithmetic intensity）**：每字节访存做的运算数，FLOP/byte 或 OP/byte。int8 GEMV 约 2 OP/byte，depthwise 3×3 fp32 完美复用时 4.5 FLOP/byte。
- **ridge point（拐点）**：峰值算力 / 带宽。强度低于它就是访存受限。拐点随存储层级变化：同一颗 CPU，L1 的拐点远低于 DRAM 的拐点。
- **SIMD 宽度与 VNNI**：AVX-512 一条指令处理 16 个 fp32 或 64 个 int8。VNNI（`vpdpbusd`）一条指令完成 u8×s8 的 4 路点积并累加进 int32，是 x86 上 int8 GEMM 的主力；Arm 对应 `SDOT`/`UDOT`、`SMMLA`。

## M02 整数契约

- **requantization（重量化）**：int32 累加结果乘以 M = s_w·s_x/s_y 再加 zero-point、饱和到 int8/uint8。M 用定点 m0·2^−shift 表示。
- **zero-point**：非对称量化的整数偏移，real = s·(q − z)。权重通常对称（z=0），激活常用非对称 uint8。
- **per-channel / per-tensor**：每个输出通道一个权重 scale vs 整张量一个。BN 折叠后通道幅值拉开，逐张量会把小通道量化成 0；本仓库 W4 逐张量 76.42%、逐通道 83.20%（lab2-quantization）。
- **SRDHM**：SaturatingRoundingDoublingHighMul，gemmlowp/TFLite 的定点乘法：(a·b·2 + 2^30) >> 31，平局向 +∞。与随后的 RoundingDivideByPOT 构成两次舍入。
- **round-half-even（银行家舍入）**：平局取偶数。ONNX Runtime 的 QLinearConv 用 fp32 乘法 + 这一舍入，本仓库在 12 个配置（含刻意构造的平局）上逐位复现。
- **double rounding（两次舍入）**：先舍入到 Q31 再舍入到整数，与一次舍入结果差 1 的概率约 2^−(shift+1)。`TFLITE_SINGLE_ROUNDING` 改为一次舍入。
- **饱和（saturation）**：`vpmaddubsw` 把两个 u8×s8 乘积之和饱和到 int16；权重超过 ±64 时就会溢出，wmax=127 时 100% 输出出错。
- **BN folding（BN 折叠）**：W' = W·γ/√(var+ε)，b' = (b − mean)·γ/√(var+ε) + β。整数图里没有独立的 BN，必须折进卷积。

## M03 校准与误差定位

- **calibration（校准）**：用少量样本确定激活的量化范围。本仓库实现 minmax、percentile、MSE、KL 与 KL（排除零值）五种。
- **KL 校准**：TensorRT 的 entropy calibration，在 |x| 直方图上搜索阈值使 KL(P‖Q) 最小。ReLU6 输出的零值尖峰会把阈值拉到 MSE 阈值的约 1/4，W8A8 精度只有 63.7%（其余三种校准都在 88% 左右）；排除精确零值后与浮点持平。
- **percentile 校准**：取 99.99% 分位为上界，便宜、对长尾稳健，是本仓库 A8 下最好的之一。
- **敏感度分析（sensitivity）**：一次只量化一层，看精度掉多少；混合精度按它排序，比按层大小排序有效得多。
- **误差定位（localization）**：逐层比较量化与浮点激活的 SQNR 或相对误差，找第一个越过阈值的层，而不是只看最终精度。
- **QAT（量化感知训练）**：训练时前向插入伪量化，反向用 STE 近似梯度。
- **STE（straight-through estimator）**：round 的导数几乎处处为零，反向时把它当作恒等（在裁剪区间内为 1，区间外为 0）。
- **QDQ 格式**：ONNX 里用 QuantizeLinear/DequantizeLinear 对标出量化点，由运行时融合成整数算子；与 QOperator（直接 QLinearConv）格式相对。

## M04 静态内存规划

- **arena**：TFLM 等运行时在启动时一次性分配的整块内存，所有激活按编译期算好的偏移放进去。
- **liveness（存活期）**：张量从产生到最后一次被读的区间。存活字节的最大值是 arena 的下界。
- **碎片（fragmentation）**：arena 高水位减去存活峰值。first-fit 在 MBv2 上多 13.3%，按大小贪心为 0。
- **in-place depthwise（原地 depthwise）**：depthwise 输出逐通道覆盖输入，省一整块输出缓冲，只需一个通道的 scratch。
- **patch-based inference（按 patch 推理）**：MCUNetV2 的做法，前几层按空间 tile 执行，用 halo 重算换峰值。
- **halo**：卷积 tile 需要的、超出 tile 本身的边界像素，宽度由 receptive field 回推。

## M05 卷积实现

- **im2col**：把每个输出位置的 KH·KW·IC 输入 patch 展开成一行，卷积变 GEMM。工作区 OH·OW·KH·KW·IC 个元素。
- **implicit GEMM（隐式 GEMM）**：GEMM 的 A 矩阵按需从输入计算地址，不显式展开，省工作区。
- **Winograd F(m, r)**：用 Toom-Cook 插值把 m×m 输出、r×r 卷积核的乘法从 m²r² 降到 (m+r−1)²。F(2,3) 省 2.25 倍、F(4,3) 省 4 倍；代价是变换矩阵放大数值误差，fp16 下 F(4,3) 相对误差 1.5e-2。
- **depthwise 卷积**：每个通道独立卷积，几乎没有跨通道复用，算术强度低，通常访存受限，需要专用核。
- **NHWC / NCHW**：通道在最内维 vs 空间在最内维。端侧 CPU/NPU 多用 NHWC（或 NC4HW4 这类分块布局），便于 SIMD 沿通道向量化。

## M06 调度与搜索

- **tiling（分块）**：把循环切成块，使块的工作集落在某一级缓存或 scratchpad 里。
- **schedule（调度）**：同一计算的循环顺序、分块大小、向量化、展开、并行等选择的组合；与「算什么」（compute）分离，这是 Halide/TVM 的核心思想。
- **search space（搜索空间）**：所有合法调度的集合。本仓库 GEMM 空间有循环顺序、三维 tile、是否打包 B、展开因子。
- **cost model（代价模型）**：从调度特征预测性能，用来在真机测量之前筛候选；本仓库用岭回归，TVM Ansor 用 XGBoost。
- **auto-tuning（自动调优）**：在搜索空间里用随机、进化或代价模型引导的搜索，在固定测量预算下找最快的调度。
- **寄存器分块（register blocking）**：最内层的小块（如 2 行 × 64 列）让累加器常驻寄存器，每个加载的向量复用多次；累加器数 ti·tj/8 要在 8 到 16 之间（E23）。本仓库最好的调度 `ojki_ikji_t2x64x64` 就是它。
- **Spearman 秩相关**：代价模型预测名次与实测名次的相关（E24）。比 R² 更贴近用途，因为搜索只需要排序对；本仓库留出 0.47 / 0.53。
- **复测（remeasure）**：对搜索报告的前几名重新计时。单次最好值有向上偏差（赢家的诅咒），本仓库前 8 名复测平均低 3%–6%。

## M07 DMA 与双缓冲

- **DMA**：不占用计算核的数据搬运引擎，在 L2/L3 与 L1 之间搬 tile。
- **scratchpad（L1/TCDM）**：软件管理的片上 SRAM，不是硬件管理的 cache；PULP 集群上是 64–128 KB 的 TCDM。
- **double buffering（双缓冲）**：两块 tile 缓冲交替使用，DMA 搬下一块的同时计算当前块。总时间 n·max(T_dma, T_comp) + min(T_dma, T_comp)。
- **DMA 启动开销（setup）**：每次 DMA 的固定周期数；tile 越小，开销占比越大，这是分块搜索不选最小 tile 的原因。
- **Dory**：PULP 平台的分块与代码生成工具，在 L1 容量约束下搜 tile 形状，目标是最小化总周期。

## M08 块量化与端侧解码

- **Q4_0 / Q8_0**：llama.cpp 的块量化格式，每 32 个权重共享一个 fp16 scale。Q4_0 18 字节/块（4.5 bit/权重），Q8_0 34 字节/块（8.5 bit/权重）。
- **Q4_1**：在 Q4_0 基础上多一个 fp16 最小值（非对称），20 字节/块。
- **GGUF**：llama.cpp 的模型文件格式，张量按块量化格式存放，带元数据。本仓库的 NumPy 打包器与 `gguf.quants` 逐字节一致。
- **decode（解码）**：自回归生成每一步只处理一个 token，矩阵乘退化成 GEMV，时间由权重字节数 / 带宽决定。
- **perplexity（困惑度）**：exp(平均 token 交叉熵)。本仓库与 mlsys-infra 用同一协议：wikitext-2 test、1024 token 窗口、固定种子。
- **融合反量化（fused dequant）**：在 GEMV 内核里边解包 4 bit 边乘加，不把反量化后的 fp32 权重写回内存。本仓库 7B 形状单线程 1743 µs，先反量化 4541 µs。
- **整数点积（Q4 × Q8）**：激活每次调用量化成 Q8_0，块内用 `vpdpbusd`（Arm 上 `SDOT`）做 u8 × s8，(q − 8) 的偏移用 8·Σx 一次扣掉（E26），每块只做一次浮点乘。本仓库 1034 µs，比 fp32 快 4.6 倍。
- **SQNR**：信号功率与量化误差功率之比，dB。每少 1 bit 约少 6 dB（E27）；GPT-2 各层 Q8_0 45.08、Q4_0 21.03 dB。
