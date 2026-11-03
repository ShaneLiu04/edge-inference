# 系统设计题

端侧的设计题同样考**先算账再画图**：放得下吗（SRAM / DRAM 容量）、算得过来吗（MACs / 算力）、搬得过来吗（字节 / 带宽）、对得上吗（整数语义）。每题按五步走：澄清 → 估算 → 架构 → 瓶颈与取舍 → 追问。公式编号见 [公式卡](FORMULAS.md)，参数见 [硬件数量级](HARDWARE-NUMBERS.md)，可以用 `python tools/edge_calc.py` 复核。数据中心侧的设计题（70B 在线推理、预训练、多 LoRA 服务）在 [mlsys-infra SYSTEM-DESIGN](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/interview/SYSTEM-DESIGN.md)。

| # | 题目 | 主要模块 | 一句话答案 |
| ---: | --- | --- | --- |
| D1 | 256 KB SRAM 的 MCU 上做视觉唤醒词 | [M02](../notes/module-02-integer-contract.md) [M04](../notes/module-04-static-memory-planning.md) | 激活峰值先撞上限；MBv2-0.35@96 下界 135 KiB 放得下，@160 要 patch 推理才放得下 |
| D2 | 手机上跑 1B 级 LLM，目标 20 token/s | [M08](../notes/module-08-block-quantized-decode.md) [M01](../notes/module-01-edge-roofline.md) | decode 由权重字节 / 带宽决定；Q4_0 约 0.70 GB，LPDDR5 51.2 GB/s 下限 13.6 ms/token，20 token/s 要求打到 70% 带宽 |
| D3 | 模型上 NPU，有两个算子不支持 | [M06](../notes/module-06-schedule-search.md) [M07](../notes/module-07-dma-double-buffering.md) | 回退的代价是布局转换和来回搬运，不是那个算子本身；先算回退的字节，再决定改模型还是写自定义核 |
| D4 | 量化模型的验收流水线（PyTorch → ONNX → 板子） | [M02](../notes/module-02-integer-contract.md) [M03](../notes/module-03-calibration-localization.md) | 三层对齐：浮点参考、伪量化模拟、整数运行时；整数之间要求 ±1 且比例符合舍入模型，不用 allclose |

---

## D1. 256 KB SRAM 的 MCU 上做视觉唤醒词

**澄清**：nRF52840 一档（Cortex-M4F 64 MHz，256 KB SRAM，1 MB Flash）。任务是二分类「画面里有没有人」，精度目标与 MLPerf Tiny VWW 相当，延迟 < 1 s。

**估算**

```text
激活（int8，存活峰值，tinyengine-memory 实测）
  MBv2-0.35@96   下界 138,240 B = 135 KiB     → 放得下，剩 ~120 KB 给栈、scratch、外设缓冲
  MBv2-0.35@160  下界 384,000 B = 375 KiB     → 放不下
                 cut=4、3×3 patch：102,400 B = 100 KiB，整网 MACs × 1.27
权重（Flash）  MBv2-0.35 去掉 1000 类分类头约 0.4 M 参数 × 1 B = 0.4 MB < 1 MB
算力           MBv2-0.35@96 约 11 M MACs；M4 的 SMLAD 每周期 2 个 16 位乘加，
               实际 CMSIS-NN 约 0.5–1 MAC/cycle → 11e6 / (64e6 × 0.7) ≈ 0.25 s
```

**架构**：离线 PTQ（per-channel 对称权重、per-tensor 非对称激活，int32 bias）→ TFLM 或 TinyEngine 风格的代码生成 → 编译期 arena 规划（按大小贪心）→ 原地 depthwise。

**瓶颈与取舍**：分辨率每提一档，激活峰值按 H·W 平方增长，算力也同比增长；patch 推理只解决前者。精度不够时优先提分辨率（配合 patch），再考虑加宽；加宽同时增大 Flash 和峰值。

**追问**：arena 里除了激活还有什么？（算子 scratch、im2col 缓冲、持久的量化参数。im2col 3×3 的工作区是输出空间 × 9 × IC，在 MCU 上常常比激活先撞上限，CMSIS-NN 因此对 1×1 和 depthwise 走专用核。）

---

## D2. 手机上跑 1B 级 LLM，目标 20 token/s

**澄清**：Llama-3.2-1B 一档（1.24 B 参数，16 层，GQA 8 个 KV 头，head dim 64）。LPDDR5-6400、64 bit。上下文 4K，batch 1。

**估算**

```text
带宽     6400e6 × 8 B = 51.2 GB/s（总线算术，有效带宽通常 60%–80%）
权重     Q4_0：1.24e9 × 0.5625 B = 0.70 GB   Q8_0：1.32 GB   fp16：2.48 GB
decode   Q4_0 下限 0.70 GB / 51.2 GB/s = 13.6 ms → 73 token/s 上限
         目标 20 token/s = 50 ms/token → 只需打到 27% 带宽；若 Q8_0 需 51%；fp16 需 97%
KV       2 × 16 × 8 × 64 × 2 B = 32 KiB/token（fp16）；4K 上下文 128 MiB，每步也要读一遍
         4K 时 KV 每步 128 MiB，已是 Q4_0 权重的 19%
prefill  2 × 1.24e9 × 512 token ≈ 1.3 TFLOP，算力受限，要走 int8 GEMM 或 NPU
```

**架构**：权重 Q4_0（或 Q4_K 一类带组内最小值的格式），激活每层在线量化成 Q8_0，GEMV 用整数点积（本仓库 `gemv_q4_q8_vnni` 的 x86 版本；Arm 上是 `SDOT`）。prefill 走 GEMM 路径，decode 走 GEMV 路径；KV fp16，长上下文时 KV 也量化到 8 bit。

**瓶颈与取舍**：decode 的字节账里，4-bit 权重之后 KV 和每块 fp16 scale 的占比上升；Q4_0 的 0.5625 B/参数里有 0.0625 是 scale。困惑度用 wikitext-2 同一协议验收（本仓库 GPT-2：fp32 与 Q8_0、Q4_1、Q4_0 的对比见 [q4-gemv](../solutions/edge-runtime/q4-gemv/README.md)）。

**追问**：为什么不把 decode 放 NPU？（batch 1 的 GEMV 是带宽受限，NPU 和 CPU 共享同一条 LPDDR，NPU 的 TOPS 用不上；NPU 的收益在 prefill。）

---

## D3. 模型上 NPU，有两个算子不支持

**澄清**：一个检测模型 int8 上 NPU，HardSwish 和某个自定义后处理算子不支持，编译器把它们回退到 CPU。

**估算**

```text
回退一个 56×56×24 的激活：
  NPU → CPU：int8 NHWC 75 KB，可能还要从 NC/16HW16 这类分块布局转回 NHWC
  CPU 上反量化成 fp32：300 KB；算完再量化、转回 NPU 布局，再搬回
  搬运 2 × 75 KB 在 10 GB/s 有效带宽上是 15 µs，但每次回退还有一次同步和缓存维护（几十到几百 µs）
  这一层在 NPU 上本来只要几 µs
```

**架构**：先列出图里每个回退点和它两侧的张量大小；能用等价改写的先改（HardSwish → ReLU6 乘加组合，或直接换成 ReLU6 再微调），后处理整体放 CPU（在图末尾只回退一次），中间段不允许回退。

**瓶颈与取舍**：一次回退在图中间会把图切成三段，每段之间都要同步；同样两个算子，放在末尾只付一次代价。改模型要重新验收精度（D4），写自定义核要维护工具链版本。

**追问**：怎么知道 NPU 上每段多快？（逐段计时，和每段的字节数、MACs 放到同一张 roofline 上；NPU 的分块和 DMA 是否重叠用 M07 的 n·max + min 估。）

---

## D4. 量化模型的验收流水线

**澄清**：PyTorch 训练的模型，经 ONNX 导出、静态 int8 量化，部署到 ONNX Runtime 和一块 NPU。要一套能发现回归的验收流程。

**估算与标准**

```text
层级 1  fp32 PyTorch vs fp32 ONNX                      逐元素 rel err < 1e-5（BN 折叠前后同样）
层级 2  伪量化模拟（QuantSim）vs fp32                    精度差 < 0.5 个点；逐层 SQNR 曲线留档
层级 3  ORT 整数 vs 本仓库 NumPy/C++ 整数参考            同一 scale 下逐位相等（舍入语义已知时），
                                                       或差 1 的比例 ≤ 2^−(shift+1) 量级
层级 4  NPU 整数 vs ORT 整数                            差 1 比例与舍入模型吻合；差 >1 的输出数为 0
```

**架构**：每次改动（模型、校准集、工具链版本）都跑这四层，产物写成 JSONL，关键数字进 `claims.json`。层级 3、4 用构造的平局输入单独测一遍舍入语义（本仓库 `test_ties_separate_half_even_from_half_away`）。

**瓶颈与取舍**：层级 2 和 3 之间最容易出现「模拟不掉点、部署掉点」：量化点位置不同（ReLU 是否融进 requant）、BN 是否在导出前折叠、激活是否按非对称量化。把量化点列表作为验收产物之一，两边逐项对。

**追问**：校准集要多大？（本仓库 CIFAR 规模的模型上，minmax 和 MSE 的 A6 精度在 8 到 2048 张之间变化不到 0.15 个点；样本数不是主要矛盾，校准方法和量化点位置才是。）
