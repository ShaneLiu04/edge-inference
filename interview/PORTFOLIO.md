# 项目作品集：面试用 STAR 故事

每条按「20 秒结论 → STAR → 白板 → 不能推出的结论 → 追问」组织。数字来自各实验 `artifacts/raw/`，关键数字由根目录 [`claims.json`](../claims.json) 校验。
平台：故事 1–6 在 Ice Lake 服务器虚拟机（4 物理核 / 8 逻辑核，AVX-512 + VNNI）上，故事 7 加上 Xeon 8470Q 与 RTX 6000D，实测屋顶见 [HARDWARE-NUMBERS](HARDWARE-NUMBERS.md)。
MCU、NPU、Pi 5、Orin 的数字全部是设备表参数或模型输出，每个故事里都标明。

讲法：只准备两个故事。部署 / 量化岗用 1 和 3，CPU 推理引擎岗用 2 和 5，MCU / NPU 岗用 4 和 3，端侧 LLM 岗用 6 和 2。
先说结果数字，90 秒停下来让对方打断。虚拟机里没有性能计数器，被问到端口压力、缓存命中时要说清是推算。
数据中心一侧的故事（FlashAttention、分页 KV、ZeRO、GPTQ）在 mlsys-infra [PORTFOLIO](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/interview/PORTFOLIO.md)；两边可以配对讲，见文末。

## 索引

| # | 20 秒结论 | 考察点 | 实验 · 笔记 |
| ---: | --- | --- | --- |
| 1 | TensorRT 式 KL 校准在 ReLU6 网络上把 W8A8 从 88% 拉到 63.68%：精确零主导直方图；去掉零 bin 回到 88.06% | 校准、误差定位 | [lab2](../solutions/mit6.5940/lab2-quantization/README.md) · [M03](../notes/module-03-calibration-localization.md) |
| 2 | 同一条 VNNI 指令，打包布局比逐点积快 2.0–30.8 倍；标量 requant 把 129 GOP/s 拖到 20，融合后 66 | int8 kernel、布局 | [int8-kernels](../solutions/edge-runtime/int8-kernels/README.md) · [M02](../notes/module-02-integer-contract.md) |
| 3 | C++、Python、ONNX Runtime 三方整数逐位对齐；两步舍入差 1 个级的比例贴住 2^-(shift+1) | 整数契约、验收 | [int8-kernels](../solutions/edge-runtime/int8-kernels/README.md) · [M02](../notes/module-02-integer-contract.md) |
| 4 | 按大小贪心在 11 个配置上都等于激活下界；patch 推理把 MBv2-0.35@160 从 375 KiB 压到 100 KiB，塞进 256 KB MCU | 内存规划 | [tinyengine-memory](../solutions/mit6.5940/tinyengine-memory/README.md) · [M04](../notes/module-04-static-memory-planning.md) |
| 5 | 自写 AutoTVM 闭环，搜索比教科书分块快 2.4 倍；代价模型 Spearman 只有 0.47，不如随机 | 调度搜索、测量 | [tensor-program-tuning](../solutions/mlc/tensor-program-tuning/README.md) · [M06](../notes/module-06-schedule-search.md) |
| 6 | Q4 × Q8 GEMV 比 fp32 快 4.6 倍，字节却少 7.1 倍：被每块指令限住，多累加器的假设被实验否定 | 端侧 LLM 解码 | [q4-gemv](../solutions/edge-runtime/q4-gemv/README.md) · [M08](../notes/module-08-block-quantized-decode.md) |
| 7 | 同一份代码换到 Sapphire Rapids + RTX 6000D：Winograd、512 位向量、双缓冲三条结论翻转；AMX 只在 K 填满 tile 时赢 | 跨平台、性能可迁移性 | [CROSS-HOST](CROSS-HOST.md) · [报告](../reports/edge-runtime-server/edge_runtime_server_report.pdf) |

---

## 1. KL 校准掉 24 个点

**20 秒**：在 CIFAR-10 上的 MobileNetV2 式网络（20.9 万参数）上比较 5 种激活校准。minmax、percentile、MSE 在 A8 上差距不到 0.2 个点，只有 TensorRT 式 KL 掉到 63.68%（A6 时 16.30%）。逐点定位发现 ReLU6 输出里有大量精确 0，它们在合并组里被摊薄，给 KL 多加一项只随截断点增大的惩罚，阈值中位数只有 MSE 阈值的 0.27 倍。去掉精确零之后 88.06%。

- **S**：MIT 6.5940 Lab 2 停在 VGG 的模拟量化；端侧模型是 depthwise + 线性瓶颈 + 残差，校准更难。
- **T**：找出哪种校准在什么结构上失效，并能定位到具体的量化点。
- **A**：BN 折叠（logits 最大差 4.9e-6）→ 粒度对比 → 5 种校准器 → 一次只量化一个点的定位 → 逐层 SQNR → 按敏感度的混合精度 → QAT → ONNX Runtime 部署。
- **R**：
  - 定位：单点最多掉 7.0 个百分点，合起来 W8A8 只剩 63.68%。
  - 修复：去零 bin 后 A8 88.06%、A6 87.97%。
  - 混合精度：W4 基线 83.04%（98.6 KiB）；按敏感度升 8 层到 W8 87.69%（108.1 KiB），按参数量升 8 层 83.62%（176.6 KiB）。
  - QAT 3 个 epoch：83.04 → 87.63%。
  - ORT：int8 QDQ 88.09%，单线程 372.4 → 253.9 µs；Entropy 默认参数和 MinMax 一样，改成 2048 bin 后掉到 87.17%。
- **白板**：E8 的两项（舍入噪声 ∝ hi²，裁剪项 ∝ 尾部质量），以及 E9 的直方图：第 0 个 bin 的精确零被合并组摊薄，KL 多出约 p₀·ln(i/2^b)，只随截断点 i 增大，于是搜索偏向小阈值；这一项与真实误差无关，因为零点精确表示 0。
- **不能推出**：不能说「KL 不好」。它在没有大量精确零的分布上是合理的；也不能把 88.06 和 88.09 的 0.03 个点差当结论，单次评测的标准误约 0.3 个点。
- **追问**：混合精度为什么按敏感度而不是按大小？最敏感的 W4 层只占很小的参数量；按大小升层花了 78 KiB 只回来 0.6 个点。

## 2. int8 GEMM：布局比指令重要

**20 秒**：在 6 个 GEMM 形状上比较 7 种 int8 实现和 fp32。同样用 `vpdpbusd`，把权重打包成 [N/16][K/4][16][4] 比逐点积快 2.0–30.8 倍，K = 24 时差最大。打包之后的瓶颈变成 requant：标量 epilogue 把 129 GOP/s 拖到 20，融合进寄存器后 66，比标量 epilogue 快 1.2–3.5 倍。

- **S**：端侧 runtime（XNNPACK、ncnn、CMSIS-NN）的 int8 GEMM 都有打包和融合 epilogue，课程作业通常只写到点积。
- **T**：量出每一步各值多少，以及 x86 上 `vpmaddubsw` 的饱和陷阱。
- **A**：标量（交给编译器自动向量化）、maddubs 点积、VNNI 点积、VNNI 打包、打包 + 标量 requant、打包 + SIMD requant、打包 + 融合 requant；15 次取中位数；每个实现和 Python 参考逐位比对。
- **R**：
  - 打包 VNNI 对逐点积 2.0–30.8 倍。
  - maddubs：|w| ≤ 64 零错误，|w| = 100 时 99% 的列出错，|w| = 127 全错。
  - ORT `QLinearConv` int8 对 fp32：1 线程 1.39–3.08 倍，4 线程 0.76–2.14 倍，depthwise 上 int8 更慢。
  - 本机 int8 峰值只是 fp32 的 2.56 倍（按位宽是 4 倍）。
- **白板**：打包后一条 dpbusd 更新 16 个输出列 × 4 个 K，激活广播一次复用 16 次；逐点积每个输出一次水平归约。K 小时归约开销占比最高，所以 K = 24 差 30 倍。
- **不能推出**：不能说「int8 比 fp32 快 4 倍」；也不能把本机的倍数直接套到 Arm `SDOT` 上，指令宽度和端口数不同。
- **追问**：depthwise 上 ORT int8 为什么更慢？没有跨通道归约，int8 的指令优势用不上，还要多付量化、反量化和 requant；E21 的强度说明它是访存问题。

## 3. 三方整数逐位对齐

**20 秒**：同一个 int8 卷积，C++ kernel、Python 参考、ONNX Runtime `QLinearConv` 三方逐元素比较。int32 累加在 6 种层上 100% 相等。TFLite 式两步舍入（SRDHM + rounding shift）与 ORT 在 99.93–99.97% 的输出上一致，差 1 的比例随 shift 按 2^-(shift+1) 变化；换成单步舍入后 100% 一致。

- **S**：「两个 runtime 结果对不上」是部署里最常见的工单，通常的处理是放宽容差。
- **T**：区分「契约差异」和「bug」，给出可复现的判据。
- **A**：写 Python 参考（int64 累加、显式舍入模式）；C++ 三种 requant（浮点乘数、两步定点、单步定点）；构造 .5 平局的测试集；扫 shift 画不一致率曲线。
- **R**：
  - 普通数据：两步定点 99.93–99.97%，浮点、单步定点、double 100%。
  - 平局数据：ORT 与 round-half-even 的 float 100% 一致，与「远离零」的 double 99.47–99.78%，两步定点 98.97–99.55%。
  - 实际 |acc| 比满幅上界小 5–49 倍。
- **白板**：E5 的 M = M0·2^-(31+shift)；两步舍入在第一步舍掉的低位跨过 .5 时与单步结果差 1，概率 ≈ 2^-(shift+1)（E6）。
- **不能推出**：不能说 ORT「更准」，它只是用了另一种舍入；也不能说所有 NPU 都和 ORT 一致，要用同样的平局测试集去测。
- **追问**：验收标准怎么写？整数之间要求差值只有 ±1、比例与舍入模型吻合、差 > 1 的输出为 0；不用 allclose（mlsys-infra 的 D4 同一套思路）。

## 4. 激活峰值：从下界出发

**20 秒**：在 PyTorch 模型上 trace 出张量存活期，比较 4 种规划器和理论下界。按大小降序贪心（TFLM 的 `GreedyMemoryPlanner`）在 11 个配置上都正好等于下界，first-fit 多 13.3%。原地 depthwise 再降 6.7%；patch 推理把 MBv2-0.35@160 从 384,000 B 压到 102,400 B，可以放进 256 KB 的 nRF52840，代价是整网 MACs 多 27%。

- **S**：MIT 6.5940 的 TinyEngine / MCUNetV2：激活放不进几百 KB 的 SRAM。
- **T**：先给下界，再量规划器离下界多远，最后用执行方式（原地、patch）突破下界。
- **A**：图 trace 与存活期；naive、first-fit、按大小贪心、贪心 + 原地 depthwise；halo 回推（E12）和重算比例（E13）；扫描 cut 位置和 patch 数。
- **R**：
  - MBv2-1.0@224：下界 1,505,280 B，每张量独占 4.7 倍，原地后 1,404,928 B。
  - 1.0@224 cut 4、2×2：566,784 B，整网 MACs +7.7%。
  - 0.35@160 cut 8、4×4：MACs 4.2 倍，峰值只降到 177,360 B。
- **白板**：E10 下界；原地 depthwise 只多一个通道的 scratch，峰值移到 expand 处（200,704 + 1,204,224 B）；E14 的 stage 输出整块驻留项。
- **不能推出**：设备上的峰值还要加算子 scratch、im2col 缓冲和运行时开销，这里只算激活；没有在真 MCU 上跑。
- **追问**：贪心不是 NP-hard 问题的最优解吗？是近似，但这些网络是链状加少量残差，下界可达，贪心正好达到；有复杂分支的图上要报碎片率（E11）。

## 5. 4 核 CPU 上的 AutoTVM

**20 秒**：不依赖 TVM，自写「调度空间 → 生成 C++ → 并行编译 → 顺序实测 → 搜索 → 代价模型」的闭环，问题是 256³ 和 MobileNetV2 pw_expand_14 两个 fp32 GEMM。进化搜索 3 个种子的中位最优 84.5 / 82.9 GFLOP/s，是教科书分块的 2.4 / 1.6 倍，接近单线程 numpy 的 90.1 / 95.4。岭回归代价模型留出 Spearman 只有 0.47 / 0.53，64 次预算下不如随机。

- **S**：MLC 课程讲张量程序抽象，AutoTVM / Ansor 讲搜索；课程作业很少碰真实测量噪声。
- **T**：比较 grid、random、代价模型引导、进化四种策略，并弄清好调度为什么好。
- **A**：30,240 / 3,888 个点的空间（E22）；编译缓存；每点 15 次取中位数；复测前 8 名；`-mprefer-vector-width=512` 对照；读汇编。
- **R**：
  - 好调度是寄存器块（E23）：`ojki_ikji_t2x64x64` 16 个累加器、32 条 ymm FMA；最内层 i 时 ti ≤ 4 中位 52.9、ti > 4 中位 10.9。
  - 噪声：前 8 名复测平均低 3%–6%，单个 −9%～+5%；搜索的第一名 85.7 复测是 80.1。
  - 512 位：16 个候选里 13 个变慢，中位数低 12%–22%；同一调度 zmm FMA 条数减半，速度 80.1 → 55.7。
- **白板**：Spearman 公式（E24）；累加器数要在 8（FMA 延迟 4 × 2 端口）和 16（ymm 寄存器数）之间。
- **不能推出**：不能说「代价模型没用」，这里只有 25 个手写特征、几百个样本；也不能说 AVX-512 在所有 CPU 上都慢，虚拟机里没有计数器，降频还是端口原因没有区分。
- **追问**：和 mlsys-infra 的 H100 搜索有什么不同？那边是解析模型，没有噪声，能比 200 次重复；这边噪声和点之间的差距同量级，只能比中位数和达到 90% 的次数。

## 6. Q4 解码：字节比 7.1，加速 4.6

**20 秒**：实现 llama.cpp 的 Q8_0 / Q4_0 / Q4_1（与 gguf 参考逐字节一致），写 9 种 GEMV，在 4 个 LLM 层形状上测。7B 形状 4096² 单线程 Q4 × Q8 VNNI 1034 µs，比 fp32 的 4763 快 4.6 倍，但字节少 7.1 倍，只用到 9.1 GB/s。我先猜是累加器依赖链，加 4 个独立累加器后 Q4 只快 4%，假设被否定；瓶颈是每块固定的十几个 µop。GPT-2 124M 在 wikitext-2 全部 280 个窗口上：fp32 29.951、Q8_0 29.953、Q4_0 31.88、Q4_1 32.15。

- **S**：端侧 LLM 解码是 batch 1 的 GEMV，llama.cpp 用块量化减字节。
- **T**：弄清块量化省下的字节能变成多少速度，以及精度代价。
- **A**：NumPy 打包与 gguf 逐字节比对；C++ fp32、fp16、Q8 / Q4 浮点融合、先反量化、VNNI 整数点积、4 累加器版；1 / 4 线程；不依赖 transformers 的 GPT-2 前向，协议与 mlsys-infra 相同。
- **R**：
  - 先反量化再点积 4541 µs，和 fp32 差不多；浮点融合 1743 µs；整数点积 1034 µs。
  - 两次空闲运行：Q4 × Q8 1005 / 1034 µs（差 3%），Q8 × Q8 1235 / 1009 µs（差 18%）。指令受限的内核稳定，缓存敏感的内核漂。
  - 单线程 Q8 与 Q4 持平（1009 对 1034 µs），4 线程反超（275 对 311 µs），但 Q8 测到 64.7 GB/s，超过 DRAM 流式 48.4，有 L3 命中。
  - 层平均 SQNR：Q8_0 45.08、Q4_1 21.96、Q4_0 21.03 dB；Q4_1 SQNR 更高、困惑度更差。
  - mlsys-infra 同协议 W4 g128：RTN 33.27，GPTQ 31.20。
- **白板**：E25 的 tok/s ≤ BW / 权重字节（Llama-7B Q4_0 3.79 GB：Pi 5 上限 4.5 tok/s、Orin Nano 17.9）；E26 的偏移项 8·Σx 每个激活块只算一次。
- **不能推出**：没有测完整 7B 模型，7B 的 token/s 是由单层 GEMV 推算的；Pi 5 和 Orin 的数是设备表上界；Q4_1 和 Q4_0 困惑度差 0.27 没有置信区间。
- **追问**：4 线程 Q8 更快，那该选 Q8？矩阵能部分留在缓存时才成立；7B 的 GB 级权重每个 token 都从 DRAM 流过，按 48.4 GB/s 封顶算，Q8 约 148 ms/token，Q4 约 125 ms，结论很可能反转。
  服务器上 llama.cpp 跑完整 Qwen2.5-7B 证实了反转：16 线程 Q4_0 37.4、Q8_0 22.0 tok/s。

## 7. 换一台机器，三条结论翻转

**20 秒**：把全部 edge-runtime 实验原样搬到 Xeon 8470Q（Sapphire Rapids）和 RTX 6000D 上重跑。大部分指标快 1.3–2.7 倍，与单核睿频和更大缓存一致；但 Winograd 在 56² 层从快 16% 变成慢 32%，512 位向量从中位数慢 12% 变成前 8 名全部更快，两线程双缓冲在所有工作量下都慢于串行（0.63–0.95 倍）。新增的 AMX 路径在 512³ 上是 VNNI 的 3.0 倍，K = 24 时只有 0.6 倍；GPU 上 Q4 GEMV 读到实测带宽的 99.5%，但只有 CUDA graph 才把它兑现成 3.1 倍的解码速度。

- **S**：第一版所有结论都来自一台 4 核虚拟机，面试官会问「换台机器还成立吗」。
- **T**：区分结构性结论（只由访存 / 计算比决定）和位置性结论（取决于两个量谁大）。
- **A**：同一份代码、绑 NUMA1 物理核、1–16 线程扫描；新增 AMX tile GEMM（`arch_prctl` 申请 XTILEDATA，2×2 输出块）、Triton 融合反量化 GEMV（AoS → SoA、L2 冷计时、eager 与 graph）；`compare_hosts.py` 生成 34 项对照，汇编计数拆分编译器因素。
- **R**：
  - 单核 DRAM 17.5 GB/s，只高 30%；16 线程 193.5。llama.cpp 单线程 Q4_0 7B 3.97 tok/s × 每 token 4.1 GB = 16.3 GB/s，是单核带宽的 93%。
  - AMX：512³ 1704 GOP/s；prefill 5.9 倍（VNNI 内核把 16 MiB 的 B 扫 64 遍，AMX 8 遍）；单独 requant 的损失从 12% 变成 43%。
  - 编译器：同一段标量 int8，GCC 13 自动向量化成 `vpdpbusd`（100 GOP/s），GCC 11 只有 20。
  - GPU：只算 Linear 的 Qwen2.5-7B 解码，fp16 92、Q8 163、Q4 284 tok/s（graph）；eager 下 Q4 只有 176，22 µs 的下发地板吃掉一半收益。
- **白板**：Winograd 收益 t_g / (t_g / 2.25 + t_x) 随 GEMM 变快而缩小；AMX 有效比例 η(K) = K / (64⌈K/64⌉)；Amdahl 1 / ((1 − f) / s + f)。
- **不能推出**：两台机器的编译器不同，跨主机比值混合了硬件与编译器；服务器 cgroup 只给 22 个 CPU 配额且有其他租户；都没有性能计数器。
- **追问**：AMX 峰值 4.3 TOP/s，你只到 40%，差在哪？2×2 输出块里每个 tile 只复用 2 次，tile load 没被藏住；更大的块受限于 8 个 tile 寄存器。没有计数器，这是推算。

---

## 和 mlsys-infra 配对讲

| 本仓库 | mlsys-infra | 一起讲的线索 |
| --- | --- | --- |
| 1 KL 校准 | 6 GPTQ / SmoothQuant | 校准假设失效：KL 的零尖峰对 SmoothQuant 的 α |
| 3 整数逐位对齐 | 4 分片 AdamW 逐位相等 | 用逐位相等而不是 allclose 做验收 |
| 5 调度搜索 | 5 `torch.compile` 的字节比 | 编译器的收益由 roofline 决定；有噪声和无噪声的搜索 |
| 6 Q4 解码 | 2 推测解码、3 分页 KV | decode 受带宽限：端侧减字节，数据中心加 batch 或摊薄验证 |
