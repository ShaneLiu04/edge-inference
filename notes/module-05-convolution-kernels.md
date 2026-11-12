# Module 05 — 卷积 kernel：im2col、implicit GEMM、depthwise 的布局、Winograd 的数值代价

- 课程出处：MIT 6.5940 Fall 2024《TinyEngine and Parallel Processing》 · CMU 15-418《Parallel Computing》 · Lavin & Gray《Fast Algorithms for Convolutional Neural Networks》 · NNPACK / XNNPACK 源码
- 必读论文：Lavin & Gray, Fast Algorithms for Convolutional Neural Networks, 2016；Chellapilla et al., High Performance Convolutional Neural Networks for Document Processing, 2006（im2col）；Zhang et al., High Performance Zero-Memory Overhead Direct Convolutions, 2018
- 对应实验：[conv-kernels](../solutions/edge-runtime/conv-kernels/README.md)
- 实测结论：本机单线程 fp32，9 个 MobileNetV2/ResNet 层上 im2col 与 implicit GEMM 在 15.8–36.6 GFLOP/s，implicit GEMM 不要工作区（im2col 在 ResNet 56×56 上要 7.2 MB）；depthwise 改 NHWC + SIMD 比按通道的 CHW 实现快 8.2–9.0 倍；Winograd F(2,3) 在 14×14×256 上等效 75.0 GFLOP/s（implicit GEMM 的 2.15 倍，理论 2.25 倍），在 56×56×64 上只有 1.16 倍；F(4,3) 的 fp16 误差是直接卷积的 7.2 倍，换成 0..4 的插值点直接溢出。Xeon 8470Q 复测：56² 层 Winograd 反而比 implicit GEMM 慢 32%，14² 层快 3.0 倍；depthwise NHWC 仍快 8.2 倍
- 状态：已验证（`artifacts/raw/summary.json`、`conv_bench.csv`、`winograd_numerics.csv`，`claims.json` 校验）

## 摘要

卷积有三种主流实现：im2col 把每个输出位置的感受野展开成矩阵的一行再调 GEMM；implicit GEMM 不展开，在 GEMM 的内层循环里直接按卷积下标取数；Winograd 在变换域里用更少的乘法算一个输出 tile。
在 MCU 上，im2col 的工作区本身就可能放不下（M04 的峰值之外再加一块）；在 CPU 上，implicit GEMM 与 im2col 速度相当、却不要工作区。
depthwise 卷积的算术强度太低，GEMM 化没有意义，决定速度的是布局：NHWC 让通道维连续，一条 SIMD 指令同时算 16 个通道。
Winograd 省乘法，但变换矩阵的范数随 tile 增大而爆炸：F(2,3) 在 fp32/fp16 下与直接卷积几乎同精度，F(4,3) 的 fp16 误差已经是 7 倍，插值点选错则直接溢出。int8 Winograd 要额外的位宽，这是 TFLite/CMSIS-NN 不用它的原因。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. im2col 工作区 = OH·OW·KH·KW·IC·元素字节（E19）；ResNet 56×56×64 的 3×3 是 7.2 MB fp32，比输入输出加起来还大 4.5 倍。
2. implicit GEMM 速度与 im2col 相当（本机 22.4–34.9 vs 15.8–36.6 GFLOP/s），工作区为 0；在 stem（IC = 3）上反而快 1.8 倍，因为 im2col 的展开拷贝占了主导。
3. depthwise 强度 = 2·KH·KW/元素字节（E21），fp32 3×3 是 4.5 FLOP/byte，远在拐点左边；NHWC + SIMD 10.4–11.4 GFLOP/s，CHW 按通道只有 1.2–1.3。
4. Winograd F(m,3) 的乘法减少 = 9m²/(m+2)²（E20）：F(2,3) 2.25 倍、F(4,3) 4 倍、F(6,3) 5.06 倍；实测加速只在通道多、空间小的层上接近理论值。
5. 变换的输入放大 = ‖Bᵀ‖∞²：F(2,3) 4、F(4,3) 100、F(6,3) 225；int8 输入经 F(2,3) 变换需要 10 bit，F(4,3) 需要 15 bit。

**必须能默写的公式**

```text
E19  im2col 工作区：  W = OH·OW·KH·KW·IC·b       （1×1 stride 1 时退化为一次拷贝 H·W·IC·b）
E20  Winograd F(m,r)：Y = Aᵀ[(G g Gᵀ) ⊙ (Bᵀ d B)]A，每 tile 乘法 (m+r−1)²，直接卷积 m²r²
                       乘法减少 = m²r² / (m+r−1)²；输入放大 = (max_i Σ_j |Bᵀ_ij|)²
E21  depthwise 强度：   I = 2·KH·KW / b（每个输入元素读一次，权重可忽略），fp32 3×3 = 4.5 FLOP/byte
```

**一句话区分**

- im2col vs implicit GEMM：显式展开到工作区再 GEMM vs 在 GEMM 的地址计算里隐式展开；算术相同，差在内存。
- direct conv vs implicit GEMM：前者按卷积的六重循环写（通常对 OC 向量化），后者按 GEMM 的 M/N/K 分块写；后者能复用 GEMM 的寄存器分块。
- Winograd vs FFT：都在变换域做逐点乘；Winograd 用小 tile 的多项式插值，FFT 用整图的单位根；3×3 小卷积核上 Winograd 的变换开销小得多。

## 学习目标

- 写出 im2col 的下标映射与工作区大小，说明它为什么在 MCU 上被 implicit GEMM 或 partial im2col 取代；
- 解释 depthwise 为什么不能靠 GEMM 化提速，以及 NHWC 布局怎么让它向量化；
- 从 Toom-Cook 插值构造 F(m,3) 的 A、B、G 矩阵，并推出乘法减少与输入放大；
- 用实测说明 Winograd 在哪类层上接近理论加速、在哪类层上不划算；
- 判断一个部署目标（fp32、fp16、int8）能否用 Winograd。

## 先修知识

- M01：本机 fp32 FMA 峰值与 DRAM 带宽、拐点；
- M04：im2col 工作区与激活峰值同时存在，要计入 arena；
- mlsys-infra M08：GPU 上的 GEMM 分块与 tensor core，作为本讲 CPU kernel 的对照。

## 核心概念

### 1. im2col

输出位置 (oy, ox) 的感受野是 KH×KW×IC 个输入值，展开成矩阵的一行；所有 OH·OW 个位置构成 (OH·OW) × (KH·KW·IC) 的矩阵，乘以 (KH·KW·IC) × OC 的权重矩阵。
展开后的矩阵里每个输入元素出现 KH·KW 次（stride 1），工作区是输入的 KH·KW 倍。CMSIS-NN 的做法是 partial im2col：一次只展开 2 个输出位置，工作区降到 2·KH·KW·IC。

### 2. implicit GEMM

把 GEMM 的 K 维（KH·KW·IC）直接映射回 (ky, kx, ic)，在加载 A 矩阵元素时算 `x[(oy·s + ky − pad)·W + ox·s + kx][ic]`，越界即 padding。
GEMM 的寄存器分块（M 方向几行、N 方向 16 列）不变，所以速度与 im2col + 同一个 GEMM 相当，省掉的是展开拷贝与工作区。

### 3. depthwise 的布局

每个输出通道只依赖一个输入通道，没有跨通道归约，GEMM 的 K 维只有 KH·KW = 9。
CHW 布局下按通道循环、在空间维向量化，3×3 窗口的边界处理让 SIMD 利用率很低；NHWC 布局下通道维连续，内层对 16 个通道同时做 9 次 FMA，边界处理与通道无关。

### 4. Winograd

1-D 的 F(m, r)：用 m + r − 1 个点上的多项式求值与插值，把 m 个输出、r 抽头的相关运算从 m·r 次乘法降到 m + r − 1 次。2-D 嵌套后从 m²r² 降到 (m+r−1)²。
权重变换 G g Gᵀ 对常量权重只做一次（编译期）；输入变换 Bᵀ d B 与输出变换 Aᵀ (·) A 每个 tile 都要做，它们只有加减与小常数乘法。多通道时在变换域对 IC 求和，逐点乘变成 (m+r−1)² 个独立的 GEMM。

## 关键推导

### im2col 工作区有多大

```text
ResNet 3×3，56×56×64 → 56×56×64，fp32：
  W = 56·56·9·64·4 = 7,225,344 B（实测 workspace_bytes 一致）
  输入 + 输出 = 2·56·56·64·4 = 1,605,632 B → 工作区是激活的 4.5 倍
MBv2 stem，224×224×3 → 112×112×32，stride 2：
  W = 112·112·27·4 = 1,354,752 B
```

在 STM32F746（320 KB SRAM）上任何一个都放不下；即使 int8（除以 4），ResNet 层也要 1.8 MB。

### 1×1 卷积为什么不需要 im2col

1×1 stride 1 时感受野就是一个像素的 IC 个通道，NHWC 下输入本身就是 (H·W) × IC 的矩阵，im2col 只是一次拷贝。`pointwise_gemm` 直接把输入当 A 矩阵：56×56 的两层比 im2col 版本快 7–19%（拷贝 301 KB–1.8 MB 不再发生），14×14 的层持平（差 1.6%）（拷贝只有 75 KB，留在缓存里）。

### Winograd 的乘法减少与实测加速

```text
F(2,3)：每 tile 4×4 = 16 次乘法，直接卷积 2²·3² = 36 次 → 2.25 倍
F(4,3)：6×6 = 36 vs 16·9 = 144 → 4 倍
F(6,3)：8×8 = 64 vs 36·9 = 324 → 5.06 倍
实测 F(2,3)（等效 GFLOP/s 按直接卷积的 FLOP 计）：
  ResNet 14×14×256：75.0 vs implicit GEMM 34.9 → 2.15 倍（接近 2.25）
  ResNet 56×56×64： 39.7 vs implicit GEMM 34.3 → 1.16 倍
```

两层的 MACs 相同（1.16 亿），差别在通道数：变换开销 ∝ tile 数 ×（IC + OC），逐点乘 ∝ tile 数 × IC × OC，两者之比 ∝ (IC + OC)/(IC·OC) = 2/C。C = 64 时变换的相对开销是 C = 256 时的 4 倍；而且 56×56 的变换域缓冲 6.7 MB 超出每核 L2，逐点乘的 GEMM 变成带宽受限。

## 深入推导

### 变换矩阵的范数为什么爆炸

Toom-Cook 构造里 G 的行是 a_j^k / f_j，f_j = Π_{l≠j}(a_j − a_l)；B 的列是 Π_{l≠j}(x − a_l) 的系数。点越多、越分散，B 的系数越大（多项式的系数随根的绝对值指数增长），G 的分母越大。
输入放大 ‖Bᵀ‖∞² 直接决定变换后的动态范围：

```text
F(2,3) 点 {0, ±1, ∞}：          ‖Bᵀ‖∞ = 2，放大 4
F(4,3) 点 {0, ±1, ±2, ∞}：      放大 100
F(6,3) 点 {0, ±1, ±2, ±1/2, ∞}：放大 225
F(4,3) 点 {0, 1, 2, 3, 4, ∞}：   放大 14,400（不对称、大点）
F(6,3) 点 {0, ±1, ±2, ±3, ∞}：   放大 10,000
```

放大 100 倍约吃掉 log2(100) ≈ 6.6 bit 的余量：fp32 的 24 位尾数仍够用，fp16 的 11 位尾数就只剩一半；fp16 最大值 65,504，放大 14,400 倍时输入只要到 4.5 就溢出。

### 实测误差（64 通道、26×26，相对 fp64 直接卷积）

```text
                     fp32 rms 相对误差    fp16 rms 相对误差
direct               2.6e-7              2.0e-3
F(2,3)               2.8e-7              2.2e-3          ← 与直接卷积同一量级
F(4,3)               1.9e-6              1.5e-2          ← fp16 下 7.2 倍
F(6,3)               4.0e-6              3.0e-2
F(4,3) 点 0..4       1.7e-2              inf（溢出）
F(6,3) 点 0..±3      2.0e-4              2.06（不可用）
```

工程结论：fp32 上 F(4,3)、F(6,3) 都可用（cuDNN 提供 F(4,3)，NNPACK 的 wt8x8 就是 F(6,3)）；fp16 上只有 F(2,3) 与直接卷积同一量级；点的选择不是自由的，{0, ±1, ±2, ±1/2} 是经验上范数最小的一族，换成 {0, ±1, ±2, ±3} 误差大 50 倍。

### int8 Winograd 要多少位

int8 输入 |d| ≤ 127，F(2,3) 变换后 ≤ 4·127 = 508，需要 10 bit（有符号）；F(4,3) ≤ 12,700，需要 15 bit。变换后的输入不能再用 int8 逐点乘，要么用 int16 乘法（吞吐减半，抵消一半乘法节省），要么在变换域重新量化（引入额外误差）。这是 TFLite Micro、CMSIS-NN 不提供 int8 Winograd 的原因。

### 换一台服务器：Winograd 的收益是位置性的

同一份代码在 Xeon 8470Q 上单线程重跑（GCC 11.4，`artifacts/raw/xeon8470q/`），单位 GFLOP/s：

```text
层                implicit GEMM 本地 → 服务器   Winograd F(2,3) 本地 → 服务器   Winograd / implicit
ResNet 3×3 56²    34.3 → 80.4                  39.7 → 54.3                     1.16 → 0.68
ResNet 3×3 14²    34.9 → 48.2                  75.0 → 144.5                    2.15 → 3.00
depthwise 56²     NHWC SIMD 10.8 → 17.0，CHW 通用 1.20 → 2.07                   9.0 → 8.2 倍
```

把 Winograd 的时间拆成乘法和变换两部分：t_w = t_g / 2.25 + t_x，收益 = t_g / t_w。服务器上 implicit GEMM 快了 2.3 倍，逐 tile 加减与散布写的变换只快 1.4 倍，
t_x 的相对份额变大，56² 层从快 16% 变成慢 32%；14² 层 K = C × 9 较小、GEMM 本身效率低，少做 2.25 倍乘法的优势仍在。
depthwise 的布局结论是结构性的，只由访存方式决定，换机器只改常数。框架在运行时按设备测量选算法（cuDNN 的 benchmark 模式、ncnn 的按层选择），原因就在这里。

## 白板估算

本机单线程 fp32 FMA 峰值 155 GFLOP/s、DRAM 读 13.4 GB/s（M01）：

```text
depthwise 3×3，56×56×144，fp32：
  FLOP = 2·9·56·56·144 = 8.1 M；激活字节 = 2·56·56·144·4 = 3.6 MB
  强度 = 8.1 M / 3.6 MB ≈ 2.25 FLOP/byte（输入输出各一次）
  带宽上限 ≈ 2.25 × 13.4 ≈ 30 GFLOP/s（从 DRAM）；工作集 3.6 MB 落在 L3（本机 27–40 GB/s）时更高
  实测 NHWC SIMD 10.8 GFLOP/s：离 DRAM 屋顶还差 2.8 倍，瓶颈是窗口边界与寄存器复用，不是带宽
im2col 与 implicit GEMM：15.8–36.6 GFLOP/s，峰值的 10–24%；教学用的 GEMM 没有做 B 矩阵打包与 6×16 寄存器分块
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 四种 dense 实现 | `cpp/conv.cpp`：`naive_nchw`、`im2col_gemm`、`implicit_gemm`、`pointwise_gemm` | 与 fp64 参考的相对误差 < 2e-6（`cpp_check.csv`） |
| depthwise 两种布局 | `dw_generic_chw`、`dw_nhwc_simd` | stride 1 与 stride 2 都与参考一致 |
| Winograd 构造 | `code/convlab/winograd.py` `transform` | F(2,3) 与 Lavin 的矩阵一致；fp64 下 1-D、2-D 精确 |
| Winograd C++ | `winograd_f2x3` | 奇数尺寸（部分 tile）也与参考一致 |
| 数值实验 | `scripts/run_experiments.py --only winograd` | 误差随 tile 增大，fp16 F(4,3) 点 0..4 溢出 |

## 易错点与反例

- 把 im2col 工作区忘在内存预算外：它和输入、输出同时存活，arena 要再加 E19。
- 对 depthwise 做 GEMM 化：K = 9 太短，寄存器分块无从谈起；实测 CHW 通用实现只有 1.3 GFLOP/s。
- 只看乘法次数评估 Winograd：56×56×64 上理论 2.25 倍，实测 1.16 倍；变换开销与变换域缓冲决定实际收益。
- 在 fp16 上用 F(6,3)：rms 误差 3%，换了点集则完全不可用。
- 用 stride 2 套 Winograd：F(m,r) 的推导要求 stride 1；stride 2 要拆成 4 个 stride 1 的子卷积（多相分解），收益很小。
- 在 IC = 3 的 stem 上用 im2col：展开拷贝的开销压过计算，implicit GEMM 快 1.8 倍。

## 实验或练习

- 假设：implicit GEMM 速度与 im2col 相当且无工作区；depthwise 的速度由布局决定；Winograd 加速随 IC 增大而接近理论值。
- 配置：MobileNetV2 的 stem、1×1 expand/project、depthwise（stride 1/2）与 ResNet 的两个 3×3 层，单线程 fp32，15 次取中位数。
- 结果：implicit 22.4–34.9、im2col 15.8–36.6 GFLOP/s；depthwise NHWC 10.4–11.4 vs CHW 1.2–1.3；Winograd F(2,3) 14×14×256 上 2.15 倍、56×56×64 上 1.16 倍。
- 练习：实现 partial im2col（一次 2 个输出位置），测工作区与速度；把 Winograd 扩展到 F(4,3)，在两个 ResNet 层上比较实测加速与 E20。

## 高频问题

- 问：TFLite Micro / CMSIS-NN 的卷积用什么实现？
  答：CMSIS-NN 对 1×1 直接当 GEMM，对一般卷积用 partial im2col（2 个输出位置）加 SMLAD 的 2×2 寄存器块；depthwise 有专门的 3×3 kernel。不用 Winograd，因为 int8 变换要更多位宽。
- 问：implicit GEMM 的代价是什么？
  答：A 矩阵的地址不再连续，每次取数要算卷积下标、判断 padding；在 GPU 上这部分由硬件地址生成或 TMA 隐藏，CPU 上靠预计算偏移表。
- 问：depthwise 在推理里占多少时间？
  答：远高于它的 MACs 占比。本机同一个 56×56×144 的倒残差块（expand、depthwise、project 各用最快实现）：depthwise 占 MACs 的 15.8%，占时间的 33.4%；它的 GFLOP/s 只有 1×1 的约三分之一。
- 问：什么时候值得用 Winograd？
  答：fp32/fp16、3×3 stride 1、通道多（≥128）、空间小，且变换域缓冲能放进缓存时。int8 部署与 depthwise 都不适用。

## 面试追问链

1. 问：给一个 3×3 卷积，你会怎么实现？
   答：先看数据类型与目标。MCU int8：partial im2col + GEMM 微核；CPU fp32：implicit GEMM，通道多时 Winograd F(2,3) 或 F(4,3)；depthwise 单独写 NHWC 的 SIMD kernel。
2. 问：im2col 的内存代价能估吗？
   答：E19。ResNet 56×56×64 的 3×3 是 7.2 MB fp32，激活的 4.5 倍；MCU 上不可接受，用 implicit GEMM 或 partial im2col。
3. 问：Winograd 为什么能省乘法？
   答：E20。1-D 时把相关运算写成多项式乘法，在 m + r − 1 个点上求值、逐点乘、插值回来；2-D 嵌套后 F(2,3) 从 36 次降到 16 次。
4. 问：代价在哪？
   答：变换的加法开销、变换域缓冲、数值误差。输入放大 ‖Bᵀ‖∞²：F(2,3) 4 倍、F(4,3) 100 倍；fp16 F(4,3) 误差是直接卷积的 7.2 倍，点选错直接溢出。
5. 问：int8 能用吗？
   答：F(2,3) 的变换后输入要 10 bit，F(4,3) 要 15 bit，逐点乘必须 int16，吞吐减半，节省被抵消一半；还要处理变换后的重新量化。所以端侧 int8 runtime 一般不用。
6. 问：depthwise 为什么慢，怎么救？
   答：E21：fp32 3×3 强度 4.5 FLOP/byte（每个输入读一次的理想情况），实际滑窗还会重复读；GEMM 化没有 K 维可分块。救法是 NHWC 布局让通道维向量化（本机 8.2–9.0 倍），以及与相邻 1×1 融合，让 depthwise 的输入不落回内存。
7. 问：Winograd 在你的开发机上快，部署机上慢，哪个对？
   答：都对，它是位置性结论。收益 = t_g / (t_g / 2.25 + t_x)，GEMM 变快、变换不变时收益缩小：Xeon 8470Q 上 implicit GEMM 快了 2.3 倍，56² 层 Winograd 从 1.16 倍变成 0.68 倍，14² 层仍是 3.0 倍。按目标设备逐层测量再选。

## 自测题

1. 5×5 卷积、28×28×32 → 28×28×64，stride 1，int8 im2col 工作区多大？
   要点：28·28·25·32 = 627,200 B。
2. F(3,2)（3 个输出、2 抽头）1-D 需要几次乘法？直接计算几次？
   要点：3 + 2 − 1 = 4 次；直接 6 次。
3. F(2,3) 2-D 的输入放大是 4，int8 输入变换后最大是多少？要几位？
   要点：4·127 = 508，有符号要 10 bit。
4. depthwise 3×3、int8、每个输入读一次，强度是多少？本机 int8 拐点约 29 op/byte，它在哪一侧？
   要点：2·9/1 = 18 op/byte，在拐点左侧，带宽受限。

## 关联笔记

- [M01 端侧 roofline](module-01-edge-roofline.md)：本讲的 GFLOP/s 要放在本机屋顶下看
- [M04 静态内存规划](module-04-static-memory-planning.md)：im2col 工作区与激活峰值同时存活
- [M06 调度搜索](module-06-schedule-search.md)：GEMM 的分块与循环顺序可以搜索而不是手写
- [M02 整数契约](module-02-integer-contract.md)：int8 卷积的累加与 requant
- [mlsys-infra M07 矩阵乘与分块](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-07-matmul-hardware-aware.md)：GPU 上 GEMM 分块、implicit GEMM 与 tensor core 的对照

## 本讲小结

im2col、implicit GEMM 的算术相同，差在工作区；端侧优先 implicit GEMM 或 partial im2col。
depthwise 的速度由布局决定，NHWC + SIMD 比 CHW 快一个数量级，GEMM 化无济于事。
Winograd 用变换换乘法，收益取决于通道数与缓存，代价是随 tile 指数增长的输入放大：fp16 只敢用 F(2,3)，int8 基本不用。

## 延伸阅读

- Lavin & Gray, Fast Algorithms for Convolutional Neural Networks, CVPR 2016, https://arxiv.org/abs/1509.09308
- Zhang et al., High Performance Zero-Memory Overhead Direct Convolutions, ICML 2018, https://arxiv.org/abs/1809.10170
- Barabasz et al., Error Analysis and Improving the Accuracy of Winograd Convolution for Deep Neural Networks, 2020, https://arxiv.org/abs/1803.10986
- wincnn（Winograd 矩阵生成器）, https://github.com/andravin/wincnn
- CMSIS-NN convolution kernels, https://github.com/ARM-software/CMSIS-NN
