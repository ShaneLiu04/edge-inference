# Module 02 — 整数契约：零点、int32 累加、requant 的两次舍入与 SIMD 饱和

- 课程出处：MIT 6.5940 Fall 2024《Quantization I》 · gemmlowp / TFLite `quantization_util` · ONNX `QLinearConv` 规范 · CMSIS-NN `arm_nn_requantize`
- 必读论文：Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, 2018；Krishnamoorthi, Quantizing deep convolutional networks for efficient inference, 2018
- 对应实验：[int8-kernels](../solutions/edge-runtime/int8-kernels/README.md)
- 实测结论：同一组 int32 累加，TFLite 式两步定点 requant 与 ONNX Runtime 在 99.93–99.97% 的输出上一致，其余差 1 个量化级，不一致率按 2^-(shift+1) 随 shift 变化；单次舍入与 ORT 100% 一致；AVX2 `vpmaddubsw` 在 |w| ≤ 64 时零错误，|w| = 127 时 4096 列全部饱和；VNNI 打包 GEMM 比逐点积快 2.0–30.8 倍，requant 融合进寄存器比单独标量 epilogue 快 1.2–3.5 倍；ONNX Runtime int8 在 1 线程比 fp32 快 1.39–3.08 倍。Xeon 8470Q 上 AMX tile GEMM 512³ 1704 GOP/s（VNNI 打包的 3.0 倍），K = 24 时只有 0.6 倍
- 状态：已验证（`artifacts/raw/summary.json`、`requant_agreement.csv`、`double_rounding.csv`、`maddubs_saturation.csv`，`claims.json` 校验）

## 摘要

int8 推理不是「把 float 换成 int8」：它是一份契约，规定零点在哪一侧、偏置用什么单位、累加器多宽、最后怎么把 int32 舍入回 8 bit。
契约的每一条都决定了 kernel 怎么写：零点修正项可以预先折进偏置，所以权重对称、激活带零点；累加器上界决定了归约长度的上限；requant 用定点乘法加移位，在没有 FPU 的 MCU 上也是精确可复现的。
但「精确可复现」指同一实现与自己一致。TFLite 的两步定点舍入、ORT 的 float 乘法加四舍六入五成双、精确实数舍入，三者在约 0.05% 的输出上差 1 个级。这不是 bug，而是契约里没写死的那一行。
最后，SIMD 指令自己也有契约：AVX2 的 `vpmaddubsw` 把两个 u8×s8 乘积相加后饱和到 int16，满幅 int8 权重会静默出错。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 权重对称（零点 0）、激活非对称（uint8 加零点）：Σ(x − z_x)·w = Σx·w − z_x·Σw，第二项与输入无关，预先算进偏置。
2. 偏置是 int32，单位是 s_x·s_w[c]；requant 乘数 M = s_x·s_w[c] / s_y，逐通道时是一个数组。
3. 定点 requant：M = M0·2^-31·2^-shift，M0 ∈ [2^30, 2^31)；先 SRDHM（舍入取高 32 位），再按 2 的幂舍入右移。两次舍入。
4. 两次舍入与精确结果不一致的比例约 2^-(shift+1)；shift 越小（M 越大）越常见。ORT 用 float 乘法，只在精确平局（x.5）上与精确结果不同。
5. `vpmaddubsw` 的中间和是饱和 int16：2·255·|w| ≤ 32767 要求 |w| ≤ 64。VNNI 的 `vpdpbusd` 直接累加到 int32，没有这个问题。

**必须能默写的公式**

```text
E4  整数卷积：  acc[c] = Σ_k x_q[k]·w_q[k,c] − z_x·Σ_k w_q[k,c] + b_q[c]，b_q[c] = round(b[c] / (s_x·s_w[c]))
    累加上界：  |acc| ≤ 255 · 127 · K（u8 激活减零点后 ≤ 255，权重 ≤ 127，K 为归约长度）
E5  requant：  y_q = clamp(z_y + round(M[c]·acc[c]), 0, 255)，M[c] = s_x·s_w[c] / s_y = M0·2^-(31+shift)
E6  两步舍入不一致率 ≈ 2^-(shift+1)；maddubs 无饱和条件 2·255·max|w| ≤ 32767 ⇒ max|w| ≤ 64
```

**一句话区分**

- fake quant vs 整数图：前者在 float 里模拟舍入（M03 的 QuantSim），后者真的只有整数运算；两者的差就是 requant 舍入方式。
- SRDHM vs 普通乘法取高位：SRDHM 先加 2^30 再截断（舍入），且 INT32_MIN×INT32_MIN 饱和；直接右移是截断，有系统负偏差。
- QDQ vs QOperator：ONNX 的两种表示，前者在图里显式放 Quantize/Dequantize 节点，由 runtime 融合成整数 kernel；后者直接是 QLinearConv 节点。

## 学习目标

- 从 E4 推出零点修正项为什么能预计算，以及为什么权重一侧不要零点；
- 实现 `quantize_multiplier`、SRDHM、`RoundingDivideByPOT`，与 TFLite 位级一致；
- 解释两步定点舍入、单次舍入、float 舍入、精确舍入四者何时不同，并预测不一致率；
- 写出 `vpmaddubsw` 的饱和条件，并说明 VNNI 打包布局为什么避免了水平归约；
- 在 ONNX Runtime 上复现一层 QLinearConv，逐元素对比。

## 先修知识

- M01：int8 点积峰值与 fp32 FMA 峰值之比；
- M03 的前半：scale 与零点从哪来（本讲假设它们已知）；
- mlsys-infra M06：训练侧 INT8/FP8 的 scale 管理，与本讲推理侧整数契约对照。

## 核心概念

### 1. 契约的五个字段

`int8.cpp` 开头写明了本仓库采用的约定，与 ONNX QLinearConv 和 TFLite 一致：

- x：uint8，零点 z_x，逐张量 scale s_x；
- w：int8，对称（零点 0），逐输出通道 scale s_w[c]；
- bias：int32，单位 s_x·s_w[c]；
- 累加：int32；
- y：uint8，零点 z_y，scale s_y；ReLU/ReLU6 就是 clamp 的上下界。

padding 的实数值是 0，所以 padding 位置的 uint8 值是 z_x，不是 0。这是手写 kernel 最常见的错。

### 2. 四种 requant

`requant.py` 与 `int8.cpp` 位级一致地实现了：

- fixed：TFLite / gemmlowp，Q31 乘数 + SRDHM + RoundingDivideByPOT，两次舍入，平局远离零（SRDHM 平局向 +∞）；
- fixed_single：TFLite 编译时开 `TFLITE_SINGLE_ROUNDING`，一次 64 位乘法后一次舍入；
- float：ORT / MLAS，fp32 乘法，`nearbyintf` 四舍六入五成双；
- double：精确实数乘法，平局远离零，作为参考。

### 3. 饱和与打包

AVX2 没有 u8×s8 直接累加到 int32 的指令，`vpmaddubsw` 把相邻两对乘积加成 int16（饱和），再用 `vpmaddwd` 乘 1 加到 int32。
AVX-512 VNNI 的 `vpdpbusd` 一条指令完成 4 对 u8×s8 乘积并累加到 int32。
朴素的「每个输出一个点积」写法要在最后做 16 个 lane 的水平求和；oneDNN 式打包把权重排成 [N/16][K/4][16 列][4 个 k]，一次 32 位广播 x[k..k+3] 乘一次 64 字节权重加载就得到 16 个输出列，没有水平归约。

## 关键推导

### 零点修正为什么能预计算

```text
Σ_k (x_q[k] − z_x) · w_q[k,c] = Σ_k x_q[k]·w_q[k,c] − z_x · Σ_k w_q[k,c]
                                                 └──── 只依赖权重，编译期算好 ────┘
```

若权重也有零点 z_w，还会多出 −z_w·Σx_q（依赖输入，每个输出位置都要算一次）和 K·z_x·z_w 两项。这就是权重对称的理由：省掉一次与输入相关的归约。

### 累加器够不够

u8 激活减零点后绝对值 ≤ 255，int8 对称权重 ≤ 127：

```text
|acc| ≤ 255 · 127 · K
3×3 卷积、IC = 32：K = 288，上界 9,326,880（实测最大 192,519）
上界触到 2^31 − 1 需要 K ≥ 66,311；3×3 卷积 IC ≈ 7,368
```

`requant_agreement.csv` 的 `acc_bound` 列就是这个式子，实测 |acc| 比上界小 5–49 倍（depthwise 的 K = 9 最贴近上界，3×3 与 grouped 最远）：真实数据不满幅，但上界是写 kernel 时唯一能依赖的保证。

### quantize_multiplier

M < 1 时 frexp 给出 M = q·2^e，q ∈ [0.5, 1)。M0 = round(q·2^31) ∈ [2^30, 2^31]；若舍入到 2^31，减半并 e+1。shift = −e ≥ 0。
M0 有 31 位有效精度，乘数本身的相对误差 ≤ 2^-31，远小于输出的 1 个量化级，不是不一致的来源。

## 深入推导

### 两步舍入的不一致率

fixed 先算 t = round(acc·M0 / 2^31)，再算 round(t / 2^shift)。第一步把精确值 v·2^shift（v = acc·M）舍入到整数，误差 ≤ 0.5，即在输出单位上误差 ≤ 2^-(shift+1)。
第二步舍入时，只有当精确值 v 的小数部分落在 0.5 附近、宽度约 2^-(shift+1) 的窗口里，第一步的误差才可能把它推过 0.5 的边界：

```text
P(fixed ≠ exact) ≈ 窗口宽度 / 1 = 2^-(shift+1)        （v 的小数部分近似均匀分布时）
shift = 16：预测 7.6e-6，实测 9e-6         shift = 5：预测 0.0156，实测 0.0155–0.0158
shift = 1 ：预测 0.25，   实测 0.2498–0.2506
单次舍入（fixed_single）：0 到 2.2e-4，只来自精确平局
```

`double_rounding.csv` 对 40 个随机乘数各跑 10^6 个累加值，两步舍入的不一致率与 2^-(shift+1) 的比值在 0.46–1.18 之间（shift 大时不一致事件只有几个到几十个，波动大）。
本实验 6 个层各通道的 shift 中位数是 9–11（范围 7–14），预测不一致率 2^-10 到 2^-12，即 0.02%–0.1%；实测 fixed 与 ORT 一致 99.93–99.97%，落在这个区间里。

### 平局：五成双与远离零

构造「精确平局」的测试数据（`_ties` 层：调整 scale 让大量 M·acc 恰好是 x.5）：

- ORT（float、五成双）与 float 参考 100% 一致；
- 与 double 参考（远离零）一致 99.47–99.78%，差的正好是偶数侧舍入的那一半平局；
- fixed 两步舍入降到 98.97–99.55%，fixed_single 99.49–99.87%。

结论：跨 runtime 比对 int8 输出时，「完全一致」只在同一种舍入规则内成立；跨规则要允许 1 个级的差异，并报告差异比例。

### maddubs 的饱和

`vpmaddubsw` 计算 x[2i]·w[2i] + x[2i+1]·w[2i+1]，结果饱和到 int16：

```text
最坏情况 2 · 255 · |w|max ≤ 32767  ⇒  |w|max ≤ 64.25
|w|max = 63 或 64：4096 列、K = 256，0 个错误
|w|max = 100：98.95% 的列出错，最大绝对误差 43,326
|w|max = 127：100% 出错，最大绝对误差 111,688
```

工程解法有三个：权重限制在 7 bit（[-64, 64]），x 用 7 bit，或者换 VNNI。oneDNN 在没有 VNNI 的 AVX2 机器上把权重缩放 0.5 并在 requant 里补偿。

### 换一台服务器：AMX 的有效比例与 epilogue 的份额

int8-kernels 在 Xeon 8470Q（Sapphire Rapids）上重跑，并新增一条 AMX tile 路径（`arch_prctl` 申请 XTILEDATA，2×2 输出块）。AMX 一个 tile 在 K 方向固定吃 64 字节：

```text
有效比例 η(K) = K / (64 · ⌈K/64⌉)
K = 24（MBv2 1×1 expand）：η = 0.375，AMX 162 GOP/s 对 VNNI 打包 270，0.6 倍
512³：                     η = 1，    AMX 1704 对 563，3.0 倍
prefill 256×4096×4096：    AMX 1659 对 280，5.9 倍（VNNI 内核把 16 MiB 的 B 扫 64 遍，AMX 8 遍）
```

契约本身不变：AMX 仍是 u8 × s8 → int32，requant 语义与 ORT 的逐位对齐结论照旧成立。变的是 epilogue 的份额：GEMM 快了 3 倍、requant 的时间不变，
单独一遍 SIMD requant 的损失从 VNNI 时的 12% 变成 AMX 时的 43%（1704 → 969 GOP/s）。这是 Amdahl：GEMM 越快，epilogue 越必须融合进寄存器阶段。

编译器也会改变结论。同一段标量 int8 循环，本地 GCC 13.2 自动向量化成 `vpdpbusd`（100 GOP/s），服务器 GCC 11.4 只能先把字节扩成 16 位（20 GOP/s）；
汇编计数 78 对 60 条 `vpdpbusd`、14 对 80 条字节扩展。手写 intrinsics 的版本两边都稳定，所以报告加速比时要写编译器版本。

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md) 的本机数字：1 线程 fp32 FMA 峰值 155 GFLOP/s，int8 VNNI 峰值 396 GOP/s，只有 fp32 的 2.56 倍，低于「每 lane 4 个乘加」推出的 4 倍：`vpdpbusd` 的吞吐低于 FMA，int8 峰值要实测，不能按位宽换算。

```text
VNNI 打包 GEMM（不含 requant）：        square_512 212 GOP/s，pw_project_14 221 GOP/s（峰值的 53–56%）
逐点积 VNNI（每个输出水平求和）：       square_512 104 GOP/s，pw_expand_56（K = 24）4.2 GOP/s
  K = 24 时一次点积只有 24 个乘加，水平求和与尾部处理占满时间 → 打包快 30.8 倍
requant 单独标量 epilogue：              pw_expand_56 打包 129 → 20 GOP/s（int32 写回再读、逐元素标量）
requant SIMD epilogue / 融合进寄存器：   48.1 / 66.0 GOP/s，融合比标量快 3.3 倍
```

ONNX Runtime 1 线程：int8 比 fp32 快 1.39–3.08 倍（conv3x3 3.08 倍、pw_project_56 2.92 倍、depthwise 1.39 倍）；4 线程时 0.76–2.14 倍：depthwise 上 int8 反而更慢（0.76 倍），pw_expand_14 只剩 1.24 倍——每层只有 100–250 µs，4 线程的派发与同步开销占了主导。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| requant 四种方式 | `code/int8ref/requant.py`、`cpp/int8.cpp` | Python 与 C++ 位级一致（`cpp_*_exact` 全为 True） |
| 整数卷积与零点 | `int8.cpp` `conv_int32`、`code/int8ref/layer.py` | 6 种层（3×3、1×1、depthwise、stride 2、stem、grouped）累加位级一致 |
| ORT 对照 | `code/int8ref/ort.py` | QLinearConv 输出与 float 方式 100% 一致 |
| 两步舍入 | `scripts/run_experiments.py --only rounding` | 不一致率与 2^-(shift+1) 同量级 |
| maddubs 饱和 | `int8 --mode=saturate` | |w| ≤ 64 零错误 |
| GEMM 与 epilogue | `int8 --mode=bench` | 所有 int8 方法与标量参考位级一致（`gemm_all_exact`） |

## 易错点与反例

- padding 填 0：uint8 的 0 是实数 −z_x·s_x，不是 0；要填 z_x。
- 在 requant 里用 `>>` 截断代替舍入：系统性负偏差，每层累积。
- 以为同一模型在 TFLite 和 ORT 上输出应逐位一致：舍入规则不同，0.05% 量级的 1 级差异是预期行为。
- 在 AVX2 上用满幅 int8 权重配 `vpmaddubsw`：静默饱和，测试数据不满幅时还发现不了。
- 把 int8 TOPS 与 fp32 FLOPS 之比当成加速比：本机峰值比 2.56，实测 ORT 1 线程加速 1.4–3.1 倍，4 线程可能小于 1。
- bias 用 int8 或 int16：单位是 s_x·s_w，数值常常是几千到几万，必须 int32。

## 实验或练习

- 假设：两步定点舍入与精确舍入的不一致率约 2^-(shift+1)；ORT 与 float 方式完全一致。
- 配置：6 种层 × 普通数据/构造平局两种输入；40 个随机乘数 × 10^6 个累加值。
- 结果：普通数据 fixed 与 ORT 一致 99.93–99.97%，float/fixed_single/double 100%；平局数据 ORT 与 float 100%、与 double 99.47–99.78%；不一致率曲线贴住 2^-(shift+1)。
- 练习：把 `requant_fixed` 改成 TFLITE_SINGLE_ROUNDING 的定义，重跑平局数据，解释为什么它与 ORT 的差异只剩平局上的方向（向 +∞ vs 五成双）。

## 高频问题

- 问：为什么 requant 要用定点而不是 float？
  答：Cortex-M0/M3 没有 FPU；有 FPU 的芯片上 float 乘法也比 32×32→64 的整数乘法加移位慢或不可确定（不同编译选项下 FMA 收缩）。定点在任何平台上位级可复现。
- 问：逐通道量化让 kernel 变复杂多少？
  答：累加完全不变，只有 epilogue 的 M0、shift、零点修正从标量变成按列取的数组；SIMD 版本按 16 列一组加载（`RequantSoA`）。
- 问：int32 累加会溢出吗？
  答：上界 255·127·K，K 到 6.6 万以上才可能。3×3 卷积 IC 7,000 以上、或者超长的全连接层要分块累加。实测数据比上界小一个数量级以上。
- 问：ORT 和 TFLite 的 int8 输出差一点，哪个对？
  答：都对，契约不同。ORT 用 float 乘法加五成双，TFLite 默认两步定点。部署验收应该用目标 runtime 的舍入规则生成参考输出。

## 面试追问链

1. 问：给你一个 int8 conv 的 scale 与零点，写出整数计算过程。
   答：E4：累加 Σx·w，减去预计算的 z_x·Σw，加 int32 bias；E5：乘 M0、SRDHM、舍入右移 shift、加 z_y、clamp 到 [0, 255]（ReLU6 时 clamp 上界是 6/s_y + z_y）。
2. 问：M0 和 shift 怎么算？
   答：frexp(M) 得到 q·2^e，M0 = round(q·2^31)，shift = −e；M0 舍入到 2^31 时减半并调整指数。
3. 问：两次舍入会带来多大误差？
   答：最多 1 个量化级，发生比例约 2^-(shift+1)（E6）。本实验 6 层上 0.03–0.07% 的输出与 ORT 差 1 级，平局数据上约 1%。
4. 问：能消掉吗？
   答：用 64 位乘法一次舍入（TFLITE_SINGLE_ROUNDING）。6 个层的普通数据上与 ORT 100% 一致，随机累加值上与精确结果只在平局处不同（0–2.2e-4）；代价是需要 64 位乘法，Cortex-M 上有 `SMULL`，开销很小。
5. 问：在 x86 上写 int8 GEMM，AVX2 和 AVX-512 VNNI 有什么区别？
   答：AVX2 用 `vpmaddubsw` + `vpmaddwd`，中间有饱和 int16，要求 |w| ≤ 64（E6）；VNNI 的 `vpdpbusd` 直接到 int32。性能上更大的差别在布局：打包成 [N/16][K/4][16][4] 后没有水平求和，K 小时快 30 倍。
6. 问：requant 放在哪里做？
   答：融合进 GEMM 的寄存器 epilogue，int32 tile 不写回内存。本机 pw_expand_56 上，单独的标量 epilogue 把 129 GOP/s 拖到 20，融合后 66；K 小的层 epilogue 占比最大。
7. 问：有了 AMX，是不是所有 int8 层都走 AMX？
   答：不是，按 K 选路径。tile 的 K 固定 64 字节，K = 24 时有效比例 0.375，Xeon 8470Q 上 AMX 162 对 VNNI 270 GOP/s；512³ 上 3.0 倍。GEMM 变快后单独 requant 的损失从 12% 涨到 43%，融合更要紧。

## 自测题

1. s_x = 0.02、s_w = 0.005、s_y = 0.05，M 是多少？M0 和 shift 呢？
   要点：M = 0.002 = 0.512·2^-8，M0 = round(0.512·2^31) = 1,099,511,628，shift = 8。
2. 上题 shift = 8，两步舍入的不一致率大约多少？
   要点：2^-9 ≈ 0.2%。
3. u8 激活、int8 权重、1×1 卷积 IC = 4096，int32 够吗？
   要点：255·127·4096 = 132,648,960 < 2^31 − 1，够。
4. AVX2 上权重最大 80，用 `vpmaddubsw` 安全吗？
   要点：2·255·80 = 40,800 > 32,767，不安全；要把权重限制到 64 以内或换指令。

## 关联笔记

- [M03 校准与误差定位](module-03-calibration-localization.md)：scale 与零点从哪来，fake quant 与整数图的差别
- [M01 端侧 roofline](module-01-edge-roofline.md)：int8 峰值与 fp32 峰值之比、带宽拐点
- [M05 卷积 kernel](module-05-convolution-kernels.md)：同一层的 fp32 im2col/implicit GEMM/Winograd 实现
- [mlsys-infra M14 显存优化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-14-memory-optimization.md)：训练侧混合精度的 scale 管理与本讲推理侧整数契约的对照
- [mlsys-infra M17 压缩与量化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-17-compression-quantization.md)：服务端 INT8 GEMM、权重量化与 kernel 融合

## 本讲小结

整数契约规定了零点在激活一侧、偏置是 int32、requant 是定点乘法加舍入移位；每一条都对应 kernel 里的一行代码。
「位级一致」只在同一种舍入规则内成立：两步定点舍入与精确舍入的差异率约 2^-(shift+1)，平局上五成双与远离零各占一半。
SIMD 指令有自己的契约：`vpmaddubsw` 要求 |w| ≤ 64；VNNI 的真正收益来自打包布局和融合 epilogue，而不只是指令本身。

## 延伸阅读

- Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, CVPR 2018, https://arxiv.org/abs/1712.05877
- gemmlowp, fixedpoint.h（SaturatingRoundingDoublingHighMul、RoundingDivideByPOT）, https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h
- TFLite `quantization_util.cc` 与 `TFLITE_SINGLE_ROUNDING`, https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/common.h
- ONNX QLinearConv, https://onnx.ai/onnx/operators/onnx__QLinearConv.html
- oneDNN int8 computations, https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html
