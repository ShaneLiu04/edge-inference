# Module 08 — 块量化解码：Q4_0 / Q8_0、带宽上界、整数点积与困惑度

- 课程出处：llama.cpp / ggml 的块量化格式与 `vec_dot_q4_0_q8_0` · MIT 6.5940 Fall 2024《LLM Deployment Techniques》《TinyChat》 · MLC LLM 的权重量化
- 必读论文：Frantar et al., GPTQ, 2023；Lin et al., AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, 2024；Dettmers & Zettlemoyer, The case for 4-bit precision: k-bit Inference Scaling Laws, 2023
- 对应实验：[q4-gemv](../solutions/edge-runtime/q4-gemv/README.md)
- 实测结论：7B 形状 4096×4096 的 GEMV 单线程 fp32 4763 µs，Q4_0 × Q8 VNNI 1034 µs（4.6 倍，字节少 7.1 倍），实际带宽只有 9.1 GB/s，被解包指令限住；先反量化再点积 4541 µs，与 fp32 差不多；单线程 Q8 × Q8 1009 µs 已与 Q4 持平，4 线程时 275 µs 快过 Q4 的 311 µs。GPT-2 124M 在 wikitext-2 全部 280 个窗口上 fp32 29.95，Q8_0 29.95，Q4_0 31.88，Q4_1 32.15；Q4_1 的层平均 SQNR 更高（21.96 vs 21.03 dB）但困惑度更差
- 状态：已验证（数字来自 `artifacts/raw/summary.json`、`gemv_bench.csv`、`gpt2_ppl.csv`，`claims.json` 校验）

## 摘要

端侧 LLM 解码每生成一个 token 要把全部权重读一遍，batch 1 的 GEMV 强度约 2 OP/B（fp16 时 1 OP/B），远低于任何 CPU 的 ridge，所以 token/s 的上界就是「带宽 ÷ 权重字节」。
块量化把每 32 个权重和一个 fp16 scale 打成一块：Q8_0 8.5 bit/权重，Q4_0 4.5，Q4_1 5.0。字节减少多少，上界就提高多少。
但上界不等于实测：Q4 的每一块都要拆半字节、转 scale、做一次浮点乘加，单线程时这些指令比读字节还慢，实测只拿到字节比的 65%。
精度一侧，GPT-2 上 Q8_0 几乎无损，Q4_0（每 32 个一组）比 mlsys-infra 的 W4 g128 RTN（33.27）好，接近 GPTQ g128（31.20）。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 解码是 GEMV：每个权重只用一次，强度 2 / (bit/8) OP/B；Q4_0 时 3.6 OP/B，仍低于本机 int8 ridge 29.5。
2. token/s ≤ 带宽 / 权重字节。Llama-7B Q4_0 3.79 GB：Pi 5（17 GB/s）4.5 tok/s，Orin Nano（68 GB/s）17.9 tok/s。
3. 「融合」才有收益：先把整行反量化成 fp32 再点积，多写多读一遍 fp32，4541 µs 和 fp32 的 4763 µs 差不多。
4. 整数点积：激活每次调用量化成 Q8 一次，块内用 VNNI 做 u8 × s8，(q − 8) 的偏移用 Σx 一次扣掉；比浮点融合版快 1.7 倍。
5. 每少 4 bit，SQNR 少约 24 dB：GPT-2 各层 Q8_0 45.08 dB，Q4_0 21.03 dB，差 24.04 dB；但 SQNR 高不保证困惑度低。

**必须能默写的公式**

```text
E25  块格式字节：bpw = 块字节 × 8 / 32；Q8_0 = 34 B → 8.5，Q4_0 = 18 B → 4.5，Q4_1 = 20 B → 5.0
     解码上界：  tok/s ≤ BW / (N_params × bpw / 8)
E26  Q4 × Q8 块点积：Σ_j d_w (q_j − 8) · d_x x_j = d_w d_x (Σ_j q_j x_j − 8 Σ_j x_j)
     q_j ∈ [0, 15] 当 u8，x_j ∈ [−127, 127] 当 s8；dpbusd 每 4 个乘积累加进一个 int32，256 位一条正好一块 32 个
E27  量化噪声：  均匀量化误差功率 d²/12，d ∝ 块内最大值 / 2^(b−1)
     SQNR(b) ≈ 6.02·b + C（C 由块内分布的峰均比决定），Q8 − Q4 ≈ 24 dB
```

**一句话区分**

- Q4_0 vs Q4_1：对称、一个 scale，最大幅值元素精确落在 −8 vs 非对称、scale + min，16 级铺满 [min, max]。
- 融合 vs 先反量化：解包在寄存器里完成 vs 写出 fp32 行再读回；后者把 Q4 的字节优势全部还回去。
- 浮点融合 vs 整数点积：每个权重转 fp32 再 FMA vs 激活先量化，块内整数乘加，每块只做一次浮点乘。
- 带宽上界 vs 实测：字节决定的天花板 vs 指令吞吐与缓存决定的实际值；Q4 单线程被指令限住。

## 学习目标

- 能写出 Q8_0 / Q4_0 / Q4_1 的块布局与反量化公式，算出 bit/权重。
- 能从模型参数量和设备带宽估出解码 token/s 上界，并解释实测为什么达不到。
- 能推导 Q4 × Q8 整数点积里的偏移项，说明为什么 VNNI 需要 u8 × s8。
- 能解释为什么 4 线程时 Q8 可能比 Q4 快，以及这个结论在真实大模型上为什么可能反转。
- 能把 GPT-2 的困惑度结果放到 RTN、GPTQ、分组大小的坐标里比较。

## 先修知识

- [M01 端侧 roofline](module-01-edge-roofline.md)：强度、ridge、带宽上界。
- [M02 整数契约](module-02-integer-contract.md)：VNNI dpbusd、u8 × s8、零点展开。
- [M03 校准与误差定位](module-03-calibration-localization.md)：粒度、SQNR、逐层定位。
- 困惑度：exp(平均 token 交叉熵)。

## 核心概念

### 1. 为什么解码是带宽问题

batch 1 时每层是 y = W·x，每个权重只参与一次乘加。读 W 的字节数远大于读 x，强度 ≈ 2 / (每权重字节)。
fp16 时 1 OP/B，Q4_0 时 3.6 OP/B，本机 int8 ridge 29.5、fp32 ridge 11.5，全部在 roofline 左侧。
prefill（一次处理 prompt 的许多 token）是 GEMM，强度随 token 数增长，是计算问题；端侧体验里「首 token 慢」与「后续 token 慢」是两个不同的瓶颈。

### 2. 块量化格式

每 32 个连续权重一块，块内共享一个 fp16 scale d（Q4_1 再加一个 fp16 最小值 m）。
Q4_0 的 d = max / −8，其中 max 保留最大幅值元素的符号，使它精确映射到 q = 0（值 −8d），额外的一个量化级落在另一侧。
半字节排布：字节 j 的低 4 位是元素 j，高 4 位是元素 j + 16，这样一次 AND 和一次移位就得到两个 16 元素的半块。
本仓库的 NumPy 打包与 llama.cpp 的 gguf 参考实现逐字节一致（`test_byte_exact_with_gguf_reference`）。

### 3. 三种 Q4 内核

| 内核 | 做法 | 7B 形状单线程 |
| --- | --- | ---: |
| 先反量化 | 整行解成 fp32 写到缓冲，再 fp32 点积 | 4541 µs |
| 浮点融合 | 寄存器里解包、转 fp32、乘 d、FMA | 1743 µs |
| 整数点积 | x 先量化成 Q8，块内 dpbusd，每块一次浮点乘 | 1034 µs |

### 4. 多线程与缓存

按行切给 4 个线程。fp32 4.12 倍、Q8 × Q8 3.66 倍、Q4 × Q8 3.32 倍。
4096² 的矩阵 Q8 是 17.8 MB、fp16 33.5 MB；本机 4 线程流式读 DRAM 是 48.4 GB/s，而 Q8 × Q8 测到 64.7 GB/s、fp16 65.6 GB/s，说明重复调用时矩阵有一部分留在 L3。
真实 7B 模型的权重是 GB 级，每个 token 都从 DRAM 流过，这部分缓存收益不存在。

## 关键推导

### E25 解码上界

每个 token 读全部权重一次，忽略 KV cache 与激活：

```text
t_token ≥ N_params × bpw / 8 / BW
Llama-7B（6.74B 参数）Q4_0：6.74e9 × 4.5 / 8 = 3.79 GB
  Pi 5   17.07 GB/s → 4.5 tok/s
  Orin Nano 68 GB/s → 17.9 tok/s
  本机 4 线程（Q4 × Q8 实测 30.3 GB/s）→ 8.0 tok/s
```

### E26 Q4 × Q8 整数点积

```text
权重块：w_j = d_w (q_j − 8)，q_j ∈ [0, 15]
激活块：x_j ≈ d_x · s_j，s_j ∈ [−127, 127]
Σ_j w_j x_j ≈ d_w d_x Σ_j (q_j − 8) s_j = d_w d_x (Σ_j q_j s_j − 8 Σ_j s_j)
```

dpbusd 要求一边无符号：q_j 本来就在 [0, 15]，正好当 u8；Σ s_j 每个激活块只算一次，所有输出行共用。
Q8 × Q8 时两边都有符号，llama.cpp 的做法是 |w| 当 u8、sign(w)·x 当 s8（`mul_sum_i8_pairs`）。

## 深入推导

### 为什么 Q4 单线程拿不到字节比

字节比 fp32 / Q4_0 = 32 / 4.5 = 7.1，实测 4763 / 1034 = 4.6。
`gemv_q4_q8_vnni` 每 32 个权重要：读 18 B、AND 与移位拆半字节并拼成 256 位（`vinserti128`）、读 32 B 激活、一条 256 位 dpbusd、广播 Σx 并做整数减法、
两次标量 fp16 → fp32（F16C `vcvtph2ps`）与一次乘法、广播 scale、一次 int → float 转换和一次 FMA。真正的乘加只有一条指令，其余约 15 个 µop 都是每块的固定开销。

第一个假设是累加链：只有一个累加器，每块的 FMA 要等上一块（延迟 4 拍）。`gemv_q4_q8_vnni_4acc` 用四个独立累加器验证它（与原版在同一次 bench 里计时）：

```text
                        1 个累加器    4 个累加器
7B 形状 Q4 × Q8 1 线程   1034 µs       990 µs     （−4%）
7B 形状 Q8 × Q8 1 线程   1009 µs       903 µs     （−11%）
4 线程                   Q4 311 → 303 µs（−3%），Q8 275 → 297 µs（+8%）
每块耗时（Q4）           1.97 / 1.89 ns，按 2.42 GHz 约 4.8 / 4.6 拍
```

Q4 只快 4%，假设被否定：去掉依赖链之后每块仍要约 4.6 拍，瓶颈是每块的指令吞吐，不是依赖。
按 µop 数估算，拼接、广播、fp16 转换集中在 shuffle 端口（Ice Lake 的 port 5），每块 4–5 个，与 4.6 拍同量级；但虚拟机里没有性能计数器，这只是估算。
Q8 × Q8 不用拆半字节，每块 µop 更少，原来更接近延迟链，所以单线程多累加器有 11% 收益；4 线程时它已经受带宽限，多累加器反而慢 8%。
GOP/s 换算：2 × 4096² / 1034 µs = 32 GOP/s，只有本机 VNNI 峰值 396 的 8%；带宽 9.1 GB/s，只有单线程 DRAM 13.4 的 68%。两个都没吃满，说明瓶颈是每块固定的指令序列（延迟链 + 端口），不是 ALU 也不是带宽。
Q8 × Q8 每块读 34 B 但不用拆半字节，单线程 1009 µs、17.7 GB/s，超过单核只读的 13.4，说明 17.8 MB 的矩阵有一部分留在 L3。
上一次空闲运行里 Q8 × Q8 是 1235 µs，两次差 18%；Q4 × Q8 两次是 1005 / 1034 µs，只差 3%。受带宽和缓存状态影响的内核在两次运行之间波动，受指令限的内核不动，这本身就是区分两种瓶颈的一个观测。

### 4 线程时 Q8 反超 Q4，在真实模型上会不会反转

4 线程把带宽扩到 48 GB/s 以上，Q8 的瓶颈仍是带宽，Q4 的瓶颈仍是每核指令，扩展更差（3.32 vs 3.66 倍），于是 275 µs 对 311 µs，Q8 更快。
但 Q8 的 64.7 GB/s 超过了 DRAM 流式带宽，含有 L3 命中。7B 模型全部权重流过 DRAM 时，Q8 最多 48.4 GB/s → 7.16 GB / 48.4 = 148 ms/token（6.8 tok/s），而 Q4 在 30.3 GB/s 时仍低于 DRAM 上限 → 3.79 / 30.3 = 125 ms/token（8.0 tok/s）。
所以「4 线程 Q8 更快」只在矩阵能部分驻留缓存时成立；对真实大模型，结论很可能回到 Q4 更快。本仓库没有测完整 7B 模型，这一段是推算。

### SQNR 与困惑度为什么不一致

Q4_1 的层平均 SQNR 21.96 dB 高于 Q4_0 的 21.03 dB，困惑度却是 32.15 vs 31.88。
可能的原因：Q4_0 让每块最大幅值元素精确无误差，Q4_1 的最小值和最大值都在网格端点，但中间的零点不精确；LLM 权重里少数大幅值元素对输出贡献不成比例，均值 SQNR 低估了它们。
这是假设：本仓库没有做「只保护最大元素」的消融，也没有算两种格式困惑度差的置信区间（同一组 280 个窗口，成对差 0.27）。

## 白板估算

```text
Llama-1B 一层 MLP：8192 × 2048 权重 = 16.8 M
  fp16 33.5 MB，Q4_0 9.4 MB
  单线程 DRAM 13.4 GB/s：fp16 2.5 ms，Q4 0.70 ms（上界）
  实测：fp16 2161 µs，Q4 × Q8 1075 µs —— Q4 只快 2.0 倍，字节比 3.6 倍
GPT-2 124M：权重约 0.25 GB（fp16），Q4_0 0.070 GB
  Pi 5：fp16 69 tok/s，Q4_0 245 tok/s（上界，不含 KV、lm_head 与调度开销）
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 块格式打包 | `code/blockq/pack.py` `quantize_q8_0`、`quantize_q4_0`、`quantize_q4_1` | 与 gguf 参考逐字节一致；Q4_0 最大幅值元素精确 |
| C++ 打包与内核 | `cpp/gemv.cpp` `gemv_q4_q8_vnni`、`gemv_q8_q8_vnni`、`*_4acc`、`gemv_q4_dequant_first` | 打包与 NumPy 逐字节一致；每个内核对 double 参考的相对误差；4 累加器版与原版误差一致 |
| 带宽与上界 | `scripts/run_experiments.py` `decode_bound` | `decode_bound.csv` 每格式 × 设备一行 |
| GPT-2 困惑度 | `code/blockq/gpt2.py`、`run_experiments.py --only gpt2` | 280 个窗口，fp32 与 mlsys-infra 的 fp16 29.96 一致 |
| 图 | `artifacts/figures/gemv_bench.png` | 每个形状 × 方法的 GB/s，1 与 4 线程 |

## 易错点与反例

- 「Q4 比 fp16 少 3.6 倍字节，就快 3.6 倍」：单线程 7B 形状 Q4 × Q8 对 fp16 只快 2.2 倍（2312 / 1034），被解包指令限住。
- 「先反量化，再用现成的 fp32 GEMV」：多一次 fp32 写和读，4541 µs，只比 fp32 的 4763 快 5%。
- 「GB/s 超过 DRAM 带宽就是测错了」：不一定，重复调用时矩阵部分驻留 L3；要报告矩阵大小，并和真实模型的权重量级区分。
- 「SQNR 高的格式困惑度一定低」：Q4_1 是反例。
- 只量化 Linear、忘了 lm_head 与 embedding：GPT-2 的 lm_head 与 wte 共享，这里保持 fp32；端到端字节要把它算进去（124M 里约 38.6M）。

## 实验或练习

- 假设：解码时间与权重字节成正比；Q4 精度损失可接受。
- 配置：四个形状（GPT-2 MLP、Llama-1B attn 与 MLP、Llama-7B attn）× 7 种方法 × 1/4 线程，15 次重复取中位数；GPT-2 124M 在 wikitext-2 test 的 280 个 1024 token 窗口上算困惑度。
- 结果：Q4 × Q8 单线程比 fp32 快 4.6 倍、比 fp16 快 2.2 倍，Q8 × Q8 与它持平；4 线程时 Q8 × Q8 最快；困惑度 Q8_0 +0.007%，Q4_0 +6.4%，Q4_1 +7.3%。
- 练习：减少每块的 shuffle：一次处理两行（共享 x 的加载、Σx 广播与 d_x），或用 512 位一次处理两块、把两个 scale 拼成一个向量，看单线程能否接近 13 GB/s。

## 高频问题

- 问：为什么 llama.cpp 要把激活也量化成 Q8？
  答：这样块内可以用整数点积（VNNI / ARM sdot），每 32 个乘加只做一次浮点乘；激活量化每次调用只做一次，所有输出行共用。
- 问：为什么块大小是 32？
  答：scale 的开销是 16 bit / 32 = 0.5 bit/权重；块越小精度越好、开销越大。g128 时开销 0.125 bit，但 GPT-2 上 RTN 困惑度从 31.88 升到 33.27。
- 问：端侧为什么还要多线程？
  答：单核带宽拿不满 DRAM（本机单线程 13.4 vs 4 线程 48.4 GB/s），而且 Q4 单核被指令限住，多核同时解决两件事。

## 面试追问链

1. 问：一个 7B 模型在手机上能跑多快？
   答：E25。Q4_0 3.79 GB，手机 LPDDR5 有效 30–50 GB/s，上界 8–13 tok/s；实际还要乘一个内核效率，本机单线程只有字节比的 65%。
2. 问：为什么不直接用 int4 × int4？
   答：激活的分布有离群值，4 bit 激活精度不够；而且解码是带宽问题，激活只有一个向量，量化到 8 bit 已经不占带宽。
3. 问：Q4 内核的瓶颈怎么定位？
   答：算两个利用率：GOP/s 对 VNNI 峰值（8%），GB/s 对 DRAM（70%），两个都不满就是指令序列本身；再看 Q8 × Q8（不拆半字节）能否接近带宽，它能（17.7 GB/s，含部分 L3 命中），所以拆半字节和每块开销是嫌疑。再排除依赖链：四个累加器只快 4%，说明不是延迟而是每块的指令吞吐；下一步是减少每块的 shuffle µop（多行共享激活、512 位一次两块）。
4. 问：4 线程时 Q8 更快，那应该部署 Q8 吗？
   答：先看矩阵是否驻留缓存。本机 4096² 的 Q8 测到 64.7 GB/s，超过 DRAM 流式 48.4，说明有 L3 命中；整模型流式读时 Q8 被 DRAM 限在约 6.8 tok/s，Q4 约 8.0 tok/s，结论会反转。
5. 问：Q4_0 与 GPTQ 的关系？
   答：Q4_0 是格式加 RTN；GPTQ 是在同样格式下用二阶信息逐列补偿误差。mlsys-infra 的 GPT-2 上 W4 g128：RTN 33.27 → GPTQ 31.20；本仓库 g32 RTN 31.88，分组更细抵掉了大部分差距，代价是每权重多 0.375 bit。
6. 问：这和数据中心推理的什么问题同形？
   答：mlsys-infra M15/M16 的解码同样被带宽限住，batch 才是把 GEMV 变成 GEMM 的手段；端侧通常 batch 1，只能靠减字节。

## 自测题

1. Llama-1B（1.24B 参数）Q8_0 多少 GB？在 17 GB/s 的 Pi 5 上上界多少 tok/s？
   要点：1.24e9 × 8.5 / 8 = 1.32 GB；12.9 tok/s。
2. 一个 Q4_0 块 16 个字节里，第 5 个字节的高 4 位是哪个元素？
   要点：元素 5 + 16 = 21。
3. 为什么 Q4 × Q8 的偏移项 8·Σs_j 每个激活块只算一次？
   要点：它只依赖激活，不依赖权重行；所有输出行共用。
4. Q8 与 Q4 的理论 SQNR 差多少？本仓库实测？
   要点：约 6.02 × 4 = 24 dB；45.08 − 21.03 = 24.04 dB。

## 关联笔记

- [M01 端侧 roofline](module-01-edge-roofline.md)：GEMV 强度与带宽上界
- [M02 整数契约](module-02-integer-contract.md)：dpbusd 的 u8 × s8 与零点展开
- [M03 校准与误差定位](module-03-calibration-localization.md)：粒度与 SQNR
- [mlsys-infra M17 压缩与量化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-17-compression-quantization.md)：GPTQ、AWQ、SmoothQuant 与同一 GPT-2 协议的困惑度
- [mlsys-infra M15 推理：KV cache 与推测解码](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-15-inference-kv-speculative.md)：解码的带宽上界与 KV 字节

## 本讲小结

解码是 GEMV，token/s 的天花板就是带宽除以权重字节，块量化把这个天花板抬高 3.6–7.1 倍。
能不能碰到天花板取决于内核：先反量化毫无收益，浮点融合拿到一半，整数点积单线程拿到字节比的 65%，剩下的被每块的解包指令吃掉。
精度上 Q8_0 无损、Q4_0 g32 掉 6.4%；格式之间要看困惑度而不是 SQNR。

## 延伸阅读

- ggml / llama.cpp，`ggml-common.h` 块布局与 `ggml-cpu/arch/x86/quants.c` 的 `ggml_vec_dot_q4_0_q8_0`，https://github.com/ggml-org/llama.cpp
- Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, ICLR 2023, https://arxiv.org/abs/2210.17323
- Lin et al., AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, MLSys 2024, https://arxiv.org/abs/2306.00978
- Dettmers & Zettlemoyer, The case for 4-bit precision: k-bit Inference Scaling Laws, ICML 2023, https://arxiv.org/abs/2212.09720
